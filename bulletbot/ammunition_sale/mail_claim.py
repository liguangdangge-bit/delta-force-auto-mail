from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Protocol

import cv2
import numpy as np

from bulletbot.loadout_purchase.ammunition_catalog import (
    market_ammunition_name,
    market_price_matches,
)
from bulletbot.ocr.numbers import parse_ocr_integer, parse_ocr_integer_pair
from bulletbot.vision.inventory_templates import InventoryTemplateMatcher

from .mail_vision import (
    attachment_card_bounds,
    mail_claim_available,
    mail_list_signature,
    numbered_attachment_points,
    selected_mail_row,
)
from .session import AmmunitionMailClaimResult
from .settings import AmmunitionSaleRule


MAIL_LIST_SCROLL_POINT = (0.15, 0.50)
MAIL_LIST_FIRST_ROW_POINT = (0.15, 0.17)
MAIL_LIST_SCROLL_DELTA = 120
MAIL_LIST_TOP_MAX_STEPS = 40
MAIL_LIST_STABLE_FRAMES = 2
MAIL_LIST_SETTLE_SECONDS = 0.25
MAIL_LIST_RECENTER_TRIGGER_Y = 540
MAIL_LIST_RECENTER_TARGET_Y = 360
MAIL_LIST_RECENTER_MAX_STEPS = 2
MAIL_LIST_MIN_SCROLL_MOVEMENT = 20
MAIL_DETAIL_RECT = (620, 350, 1400, 550)
MAIL_COUNT_RECT = (215, 875, 365, 940)


class MailClaimPort(Protocol):
    def navigate_to_mail(self) -> bool: ...

    def capture(self) -> np.ndarray: ...

    def click(self, point: tuple[float, float], label: str) -> None: ...

    def scroll(self, point: tuple[float, float], delta: int, label: str) -> None: ...

    def close_attachment_detail(self) -> None: ...

    def claim_all(self) -> bool: ...

    def wait(self, seconds: float) -> None: ...

    def stopped(self) -> bool: ...

    def safe_point(self) -> None: ...


class RegionOcr(Protocol):
    def detect(self, image: np.ndarray): ...


class LineOcr(Protocol):
    def recognize_line(self, image: np.ndarray) -> tuple[str, float] | None: ...


@dataclass(frozen=True)
class MailAttachmentIdentity:
    name: str
    price: int | None
    confidence: float


class AmmunitionMailClaimWorkflow:
    def __init__(
        self,
        port: MailClaimPort,
        *,
        text_ocr: RegionOcr,
        numeric_ocr: LineOcr,
        log: Callable[[str], None] | None = None,
        finish_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._port = port
        self._text_ocr = text_ocr
        self._numeric_ocr = numeric_ocr
        self._log = log or (lambda _message: None)
        self._finish_requested = finish_requested or (lambda: False)

    def claim(self, rule: AmmunitionSaleRule) -> AmmunitionMailClaimResult:
        rule.validate()
        if self._finish_requested():
            return AmmunitionMailClaimResult(message="定时结束，不再领取邮件")
        if not self._port.navigate_to_mail():
            raise RuntimeError("无法进入邮件附件界面")
        self._scroll_list_to_top()
        if self._finish_requested():
            return AmmunitionMailClaimResult(message="定时结束，不再扫描邮件")
        self._port.click(MAIL_LIST_FIRST_ROW_POINT, "选择第一封邮件")

        frame = self._port.capture()
        total_mails = self._read_mail_count(frame)
        if total_mails == 0:
            return AmmunitionMailClaimResult(
                reached_end=True,
                message="当前没有邮件",
            )
        maximum = total_mails if total_mails is not None else 200
        claimed = 0
        scanned = 0
        reached_end = False

        for mail_index in range(1, maximum + 1):
            self._port.safe_point()
            if self._port.stopped():
                raise RuntimeError("领邮件自动卖已停止")
            if self._finish_requested():
                break
            frame = self._port.capture()
            selection = selected_mail_row(frame)
            if selection is None:
                raise RuntimeError(f"未定位到第 {mail_index} 封邮件的高亮框")
            if not mail_claim_available(frame):
                reached_end = True
                self._log(
                    f"第 {mail_index} 封邮件已领取或不可领取，邮件扫描结束"
                )
                break

            scanned += 1
            cards = attachment_card_bounds(frame)
            candidates = numbered_attachment_points(frame, cards)
            if candidates:
                self._port.click(candidates[0], f"查看第 {mail_index} 封邮件的弹药附件")
                identity = self._read_attachment_identity()
                if identity is not None and self._identity_matches(
                    identity,
                    rule.ammunition_name,
                ):
                    self._log(
                        f"第 {mail_index} 封邮件匹配 {rule.ammunition_name}"
                    )
                    if self._finish_requested():
                        self._port.close_attachment_detail()
                        break
                    if not self._port.claim_all():
                        raise RuntimeError(
                            f"第 {mail_index} 封邮件点击领取后未确认领取完成"
                        )
                    claimed += 1
                    if self._finish_requested():
                        break
                    if claimed >= rule.mail_claim_limit:
                        return AmmunitionMailClaimResult(
                            claimed_mails=claimed,
                            scanned_mails=scanned,
                            reached_end=False,
                            message=(
                                f"已达到本轮领取上限 {rule.mail_claim_limit} 封"
                            ),
                        )
                    frame = self._port.capture()
                    selection = selected_mail_row(frame)
                    if selection is None:
                        raise RuntimeError("领取后未能重新定位当前邮件")
                else:
                    actual = identity.name if identity is not None else "未识别"
                    self._log(
                        f"第 {mail_index} 封邮件弹药为 {actual}，"
                        f"不是目标 {rule.ammunition_name}"
                    )
                    self._port.close_attachment_detail()

            if self._finish_requested():
                break
            if mail_index >= maximum:
                reached_end = True
                break
            if not self._advance_to_next_mail(mail_index):
                reached_end = True
                break

        message = (
            f"已扫描 {scanned} 封邮件，领取 {claimed} 封目标子弹"
            + ("；已到邮件末尾" if reached_end else "")
        )
        return AmmunitionMailClaimResult(
            claimed_mails=claimed,
            scanned_mails=scanned,
            reached_end=reached_end,
            message=message,
        )

    def _scroll_list_to_top(self) -> None:
        previous: np.ndarray | None = None
        stable = 0
        for _step in range(MAIL_LIST_TOP_MAX_STEPS):
            if self._finish_requested():
                return
            frame = self._port.capture()
            signature = mail_list_signature(frame)
            if previous is not None:
                difference = float(np.mean(cv2.absdiff(signature, previous)))
                stable = stable + 1 if difference < 2.0 else 0
                if stable >= MAIL_LIST_STABLE_FRAMES:
                    return
            previous = signature
            self._port.scroll(
                MAIL_LIST_SCROLL_POINT,
                MAIL_LIST_SCROLL_DELTA,
                "邮件列表滚动到顶部",
            )
            self._port.wait(MAIL_LIST_SETTLE_SECONDS)
        raise RuntimeError("邮件列表在有限滚动次数内未能确认顶部")

    def _advance_to_next_mail(self, current_index: int) -> bool:
        selection = selected_mail_row(self._port.capture())
        if selection is None:
            return False
        selection = self._recenter_mail_selection(selection, current_index)
        if selection is None:
            return False

        if selection.next_row_normalized[1] > 0.83:
            return False
        previous_center = selection.center_y
        self._port.click(
            selection.next_row_normalized,
            f"选择第 {current_index + 1} 封邮件",
        )
        selected_frame = self._port.capture()
        selected = selected_mail_row(selected_frame)
        if selected is None or selected.center_y <= previous_center + 20:
            return False
        return True

    def _recenter_mail_selection(self, selection, current_index: int):
        if selection.center_y < MAIL_LIST_RECENTER_TRIGGER_Y:
            self._log(
                f"第 {current_index} 封邮件高亮已在中上方，直接选择下一封"
            )
            return selection
        for step in range(1, MAIL_LIST_RECENTER_MAX_STEPS + 1):
            previous_center = selection.center_y
            self._port.scroll(
                MAIL_LIST_SCROLL_POINT,
                -MAIL_LIST_SCROLL_DELTA,
                f"邮件高亮位于中下方，回到中上方（{step}/2）",
            )
            self._port.wait(MAIL_LIST_SETTLE_SECONDS)
            selection = selected_mail_row(self._port.capture())
            if selection is None:
                return None
            if (
                previous_center - selection.center_y
                < MAIL_LIST_MIN_SCROLL_MOVEMENT
            ):
                break
            if selection.center_y <= MAIL_LIST_RECENTER_TARGET_Y:
                break
        return selection

    def _read_attachment_identity(self) -> MailAttachmentIdentity | None:
        best: MailAttachmentIdentity | None = None
        for _attempt in range(2):
            frame = self._port.capture()
            left, top, right, bottom = MAIL_DETAIL_RECT
            crop = frame[top:bottom, left:right]
            regions = self._text_ocr.detect(crop)
            prices: list[tuple[int, float]] = []
            for region in regions:
                text = str(region.text).strip()
                value = parse_ocr_integer(text)
                if value is not None:
                    prices.append((value, float(region.confidence)))
            title = self._joined_popup_title(regions)
            if title is not None:
                name, confidence = title
                price = max(prices, key=lambda value: value[1])[0] if prices else None
                candidate = MailAttachmentIdentity(name, price, confidence)
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
                if confidence >= 0.85:
                    break
            self._port.wait(0.25)
        return best

    @staticmethod
    def _joined_popup_title(regions) -> tuple[str, float] | None:
        title_fragments = []
        for region in regions:
            text = str(region.text).strip()
            if not text or not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text):
                continue
            if re.fullmatch(r"[\d\s,.$]+", text):
                continue
            left, top, right, bottom = region.bounds
            if top > 125:
                continue
            title_fragments.append(
                (left, top, right, bottom, text, float(region.confidence))
            )
        if not title_fragments:
            return None

        groups: list[list[tuple[int, int, int, int, str, float]]] = []
        for fragment in sorted(title_fragments, key=lambda item: (item[1], item[0])):
            center_y = (fragment[1] + fragment[3]) / 2
            group = next(
                (
                    candidate
                    for candidate in groups
                    if abs(
                        center_y
                        - sum((item[1] + item[3]) / 2 for item in candidate)
                        / len(candidate)
                    )
                    <= 12
                ),
                None,
            )
            if group is None:
                groups.append([fragment])
            else:
                group.append(fragment)

        candidates: list[tuple[str, float, int]] = []
        for group in groups:
            ordered = sorted(group, key=lambda item: item[0])
            joined = " ".join(item[4] for item in ordered)
            confidence = sum(item[5] for item in ordered) / len(ordered)
            candidates.append((joined, confidence, min(item[1] for item in ordered)))
        name, confidence, _top = min(
            candidates,
            key=lambda item: (item[2], -len(item[0]), -item[1]),
        )
        return name, confidence

    def _read_mail_count(self, frame: np.ndarray) -> int | None:
        left, top, right, bottom = MAIL_COUNT_RECT
        crop = frame[top:bottom, left:right]
        prepared = cv2.resize(
            crop,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )
        recognized = self._numeric_ocr.recognize_line(prepared)
        if recognized is None:
            self._log("未读出邮件总数，将以列表末尾为停止条件")
            return None
        pair = parse_ocr_integer_pair(recognized[0])
        if pair is None or not 0 <= pair[0] <= pair[1] <= 200:
            self._log("邮件总数格式无效，将以列表末尾为停止条件")
            return None
        self._log(f"当前邮件数量：{pair[0]}/{pair[1]}")
        return pair[0]

    @staticmethod
    def _identity_matches(
        identity: MailAttachmentIdentity,
        expected_name: str,
    ) -> bool:
        expected = InventoryTemplateMatcher.normalize_product_name(
            market_ammunition_name(expected_name)
        )
        actual = InventoryTemplateMatcher.normalize_product_name(identity.name)
        if actual != expected:
            return False
        return market_price_matches(expected_name, identity.price)
