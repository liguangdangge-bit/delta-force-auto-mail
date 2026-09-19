from __future__ import annotations

from collections.abc import Callable
from threading import Event
from datetime import datetime
from pathlib import Path
from bulletbot.reporting.run_storage import rotate_screenshots
from typing import Protocol

import cv2
from PIL import Image

from bulletbot.domain.models import (
    MarketDepthLevel,
    PageState,
    SellListingSnapshot,
    SellSlotsSnapshot,
)
from bulletbot.mail_storage.capture import ScreenCapture
from bulletbot.mail_storage.input_control import InputController
from bulletbot.mail_storage.models import WindowInfo
from bulletbot.navigation.models import GamePageId, NavigationResult, UnifiedPageObservation
from bulletbot.ocr.sell_ocr import SellOcr
from bulletbot.vision.page_classifier import PageClassifier
from bulletbot.vision.page_recognizer import PageRecognizer
from bulletbot.vision.page_templates import PageTemplateMatcher


class _NavigationPort(Protocol):
    def observe_page(self) -> UnifiedPageObservation: ...

    def navigate(
        self,
        target_page: GamePageId,
        *,
        reason: str,
    ) -> NavigationResult: ...


class WarehouseSellRuntimePort:
    """Execute market-sell listing actions under the mail worker's input lease."""

    _QUANTITY_MINUS = (0.589, 0.519)
    _QUANTITY_PLUS = (0.777, 0.519)
    _QUANTITY_SLIDER_LEFT = 0.611
    _QUANTITY_SLIDER_RIGHT = 0.758
    _PRICE_INPUT = (0.683, 0.596)
    _PRICE_COMMIT_BLANK = (0.500, 0.270)
    _CONFIRM_LISTING = (0.683, 0.694)
    _CLAIM_PROCEEDS = (0.478, 0.142)
    _CLAIM_CLICK_SETTLE_SECONDS = 2.0
    _CLAIM_FIRST_SPACE_SECONDS = 2.0
    _CLAIM_SECOND_SPACE_SECONDS = 2.5
    _CLAIM_RETRY_SECONDS = 3.0
    _CLAIM_DOUBLE_SPACE_INTERVAL = 0.15
    _CLAIM_PAGE_SETTLE_SECONDS = 0.3
    # Keep the cursor on the inventory scrollbar so item hover cards cannot
    # obscure template matching or make an unchanged bottom view look different.
    _INVENTORY_SCROLL_POINT = (0.944, 0.55)
    _WHEEL_DELTA_PER_NOTCH = -120

    def __init__(
        self,
        *,
        window_provider: Callable[[], WindowInfo | None],
        generation_provider: Callable[[], int],
        capture: ScreenCapture,
        input_controller: InputController,
        navigation: _NavigationPort,
        stop_event: Event,
        page_recognizer: PageRecognizer | None = None,
        page_templates: PageTemplateMatcher | None = None,
        sell_ocr: SellOcr | None = None,
        diagnostic_directory: Path | None = None,
        log: Callable[[str], None] = lambda _: None,
    ) -> None:
        self._diagnostic_directory = diagnostic_directory
        self._log = log
        self._window_provider = window_provider
        self._generation_provider = generation_provider
        self._capture = capture
        self._input = input_controller
        self._navigation = navigation
        self._stop = stop_event
        self._page_recognizer = page_recognizer or PageRecognizer()
        self._page_templates = page_templates or PageTemplateMatcher()
        self._sell_ocr = sell_ocr or SellOcr()

    def navigate_to(self, page: GamePageId, *, reason: str) -> bool:
        self._window()
        return self._navigation.navigate(page, reason=reason).succeeded

    def capture_image(self) -> Image.Image:
        frame = self._capture.capture(self._window())
        if not frame.healthy:
            reason = (f"游戏窗口截图无效：尺寸={frame.image.shape}，"
                      f"灰度均值={frame.mean:.3f}（要求≥2），标准差={frame.std:.3f}（要求≥3）")
            self._log(reason)
            if self._diagnostic_directory is not None:
                try:
                    directory = Path(self._diagnostic_directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    if frame.image.size:
                        path = directory / f"sell_invalid_frame_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                        Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)).save(path)
                        rotate_screenshots(directory)
                        self._log(f"上架无效原始画面：{path}")
                    else:
                        self._log("上架截图为空，没有可保存的像素")
                except Exception as exc:
                    self._log(f"上架无效截图保存失败：{exc}")
            raise RuntimeError(reason)
        return Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))

    def capture_listing(
        self,
        *,
        include_depth: bool,
        expected_name: str,
    ) -> SellListingSnapshot:
        return self._sell_ocr.read_listing(
            self.capture_image(),
            include_depth=include_depth,
            expected_name=expected_name,
        )

    def capture_listing_identity_and_lowest(
        self,
        *,
        expected_name: str,
    ) -> tuple[str | None, float | None, int | None]:
        image = self.capture_image()
        product_name, product_confidence = self._sell_ocr.read_listing_identity(
            image,
            expected_name=expected_name,
        )
        return (
            product_name,
            product_confidence,
            self._sell_ocr.read_listing_lowest_price(image),
        )

    def capture_listing_lowest_price_full_editor(
        self,
    ) -> tuple[int | None, Image.Image]:
        image = self.capture_image()
        return self._sell_ocr.read_listing_lowest_price_full_editor(image), image

    def capture_listing_price(self) -> int | None:
        return self._sell_ocr.read_listing_price(self.capture_image())

    def capture_listing_quantity_and_slots(
        self,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        return self._sell_ocr.read_listing_quantity_and_slots(self.capture_image())

    def capture_listing_depth_prices(self) -> tuple[MarketDepthLevel, ...]:
        return self._sell_ocr.read_listing_depth_prices(self.capture_image())

    def capture_sell_slots(self) -> SellSlotsSnapshot:
        return self._sell_ocr.read_slots(self.capture_image())

    def listing_editor_ready(self) -> bool:
        return self.inspect_listing_editor()[0]

    def inspect_listing_editor(self) -> tuple[bool, tuple[str, ...]]:
        image = self.capture_image()
        title_matches = self._page_templates.match(
            image,
            state=PageState.LISTING_EDITOR,
        )
        title_score = max(
            (match.score for match in title_matches),
            default=None,
        )
        title_text = None
        title_confidence = None
        title_confirmed = title_score is not None
        if not title_confirmed:
            title_reading = self._sell_ocr.read_listing_editor_header(image)
            if title_reading is not None:
                title_text = title_reading.text
                title_confidence = title_reading.confidence
                title_confirmed = self._sell_ocr.is_listing_editor_header(title_text)
        action_score = PageClassifier.listing_action_score(image)
        action_confirmed = PageClassifier.has_listing_action(image)

        title_template_evidence = (
            f"标题模板={title_score:.1%}"
            if title_score is not None
            else "标题模板=未命中"
        )
        if title_score is not None:
            title_ocr_evidence = "标题OCR=未执行（模板已确认）"
        else:
            title_ocr_evidence = (
                f"标题OCR={title_text or '未识别'}"
                + (
                    f" {title_confidence:.1%}"
                    if title_confidence is not None
                    else ""
                )
            )
        evidence = (
            title_template_evidence,
            title_ocr_evidence,
            f"上架按钮绿色特征={action_score:.1%}",
        )
        return title_confirmed and action_confirmed, evidence

    def observe_page(self) -> GamePageId:
        state = self._page_recognizer.recognize(self.capture_image()).state
        return GamePageId(state.value)

    def observe_recovery_page(self) -> UnifiedPageObservation:
        return self._navigation.observe_page()

    def window_generation(self) -> int:
        return int(self._generation_provider())

    def click_normalized(self, point: tuple[float, float]) -> None:
        self._input.click_normalized(
            self._window(),
            point,
            hold_seconds=0.04,
        )

    def move_normalized(self, point: tuple[float, float]) -> None:
        self._input.move_normalized(self._window(), point)

    def right_click_normalized(self, point: tuple[float, float]) -> None:
        self._input.right_click_normalized(self._window(), point)

    def scroll_inventory(self, notches: int) -> None:
        window = self._window()
        for _step in range(max(0, int(notches))):
            self._input.scroll_normalized(
                window,
                self._INVENTORY_SCROLL_POINT,
                self._WHEEL_DELTA_PER_NOTCH,
            )

    def press_escape(self) -> None:
        self._input.press_escape(self._window())

    def estimate_quantity(self, target_quantity: int, inventory_total: int) -> None:
        if inventory_total <= 1:
            raise ValueError("出售库存总数必须大于 1，才能使用数量滑条")
        if not 1 <= target_quantity <= inventory_total:
            raise ValueError("目标出售数量超出库存范围")
        ratio = (target_quantity - 1) / (inventory_total - 1)
        x = self._QUANTITY_SLIDER_LEFT + (
            self._QUANTITY_SLIDER_RIGHT - self._QUANTITY_SLIDER_LEFT
        ) * ratio
        self._input.click_normalized(
            self._window(),
            (x, self._QUANTITY_MINUS[1]),
            hold_seconds=0.08,
        )

    def adjust_quantity(self, delta: int) -> None:
        if delta == 0:
            raise ValueError("出售数量不需要调整")
        point = self._QUANTITY_PLUS if delta > 0 else self._QUANTITY_MINUS
        self._input.click_normalized(
            self._window(),
            point,
            clicks=abs(delta),
            interval_seconds=0.018,
        )

    def replace_price(self, value: int) -> None:
        if value <= 0:
            raise ValueError("挂牌价格必须大于 0")
        window = self._window()
        self._input.click_normalized(window, self._PRICE_INPUT, hold_seconds=0.04)
        self._input.replace_text_with_digits(window, value)
        self._input.click_normalized(window, self._PRICE_COMMIT_BLANK)

    def confirm_listing(self) -> None:
        self._input.click_normalized(
            self._window(),
            self._CONFIRM_LISTING,
            hold_seconds=0.07,
        )

    def claim_sale_proceeds(self) -> bool:
        self._log(f"领取货款：等待 {self._CLAIM_CLICK_SETTLE_SECONDS:g} 秒后点击金币按钮")
        self.wait(self._CLAIM_CLICK_SETTLE_SECONDS)
        if self.stopped():
            return False
        self._input.click_normalized(
            self._window(), self._CLAIM_PROCEEDS, hold_seconds=0.07,
        )
        # 点击金币后开始计时，不再因首次识别仍是出售页而提前退出。
        for attempt, delay in enumerate((self._CLAIM_FIRST_SPACE_SECONDS,
                                         self._CLAIM_SECOND_SPACE_SECONDS), 1):
            self._log(f"领取货款：等待 {delay:g} 秒后发送第 {attempt} 次空格")
            self.wait(delay)
            if self.stopped():
                return False
            self._input.press_space(self._window())
            self._log(f"领取货款：已发送第 {attempt} 次空格")
        self.wait(self._CLAIM_PAGE_SETTLE_SECONDS)
        if self.stopped():
            return False
        observed = self.observe_page()
        self._log(f"领取货款：两次空格后识别页面={observed.value}")
        if observed is GamePageId.MARKET_SELL:
            return True

        self._log(f"领取货款：尚未返回出售页，等待 {self._CLAIM_RETRY_SECONDS:g} 秒后连续补按两次空格")
        self.wait(self._CLAIM_RETRY_SECONDS)
        for attempt in range(2):
            if self.stopped():
                return False
            self._input.press_space(self._window())
            self._log(f"领取货款：已补按第 {attempt + 1}/2 次空格")
            if attempt == 0:
                self.wait(self._CLAIM_DOUBLE_SPACE_INTERVAL)
        self.wait(self._CLAIM_PAGE_SETTLE_SECONDS)
        if self.stopped():
            return False
        observed = self.observe_page()
        self._log(f"领取货款：补按后识别页面={observed.value}")
        if observed is GamePageId.MARKET_SELL:
            return True

        self._log("领取货款：补按后仍未返回出售页，交给统一导航恢复")
        recovered = self.navigate_to(
            GamePageId.MARKET_SELL, reason="领取金币后未返回出售页，统一恢复",
        )
        self._log("领取货款：统一导航已恢复出售页" if recovered else
                  "领取货款：统一导航未能恢复出售页")
        return recovered

    def wait(self, seconds: float) -> None:
        self._stop.wait(max(0.0, seconds))

    def stopped(self) -> bool:
        return self._stop.is_set()

    def _window(self) -> WindowInfo:
        while not self._stop.is_set():
            window = self._window_provider()
            if window is not None:
                return window
            self._stop.wait(0.5)
        raise RuntimeError("等待游戏窗口重新绑定时任务已停止")
