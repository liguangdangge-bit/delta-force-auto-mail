from __future__ import annotations

from bulletbot.navigation.action_recovery import PageRecoveryFailed

from collections.abc import Callable
import time
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import SellSlotsSnapshot
from bulletbot.navigation.models import GamePageId
from bulletbot.ocr.sell_ocr import SellOcr
from bulletbot.vision.reference_layout import ReferenceLayout
from bulletbot.vision.warehouse_unlisting import (
    UnlistControlMatch,
    UnlistDialogSnapshot,
    WarehouseUnlistingDetector,
)

from .warehouse_sell_session import (
    PendingWarehouseListing,
    SellSlotRecoveryOutcome,
    SellSlotRecoveryResult,
)


class WarehouseSellRecoveryPort(Protocol):
    def navigate_to(self, page: GamePageId, *, reason: str) -> bool: ...

    def capture_image(self) -> Image.Image: ...

    def observe_page(self) -> GamePageId: ...

    def click_normalized(self, point: tuple[float, float]) -> None: ...

    def press_escape(self) -> None: ...

    def wait(self, seconds: float) -> None: ...

    def stopped(self) -> bool: ...


class WarehouseSellRecoveryWorkflow:
    """Wait for a sell slot or safely remove every current market order."""

    _SLOT_STABILITY_ATTEMPTS = 5
    _DIALOG_OPEN_ATTEMPTS = 2
    _DIALOG_FRAME_ATTEMPTS = 6
    _DIALOG_FRAME_WAIT_SECONDS = 0.25
    _DIALOG_RETRY_DELAY_SECONDS = 0.5
    _CONFIRM_RESULT_ATTEMPTS = 6
    _CONFIRM_RESULT_WAIT_SECONDS = 0.25
    _EMPTY_CONFIRMATIONS = 2
    _EMPTY_MAX_MEAN_DELTA = 5.0
    _MAX_UNLIST_ORDERS = 32
    _EMPTY_ORDER_REGION = (0.04, 0.15, 0.49, 0.31)

    def __init__(
        self,
        port: WarehouseSellRecoveryPort,
        *,
        sell_ocr: SellOcr | None = None,
        detector: WarehouseUnlistingDetector | None = None,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] | None = None,
        on_stage: Callable[[str, str], None] | None = None,
        on_checkpoint: Callable[[str, dict[str, object]], None] | None = None,
        pause_at_safe_point: Callable[[], None] | None = None,
        finish_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._port = port
        self._finish_requested = finish_requested or (lambda: False)
        self._timed_finish = finish_requested is not None
        self._sell_ocr = sell_ocr or SellOcr()
        self._detector = detector or WarehouseUnlistingDetector()
        self._clock = clock
        self._log = log or (lambda _message: None)
        self._on_stage = on_stage or (lambda _stage, _message: None)
        self._on_checkpoint = on_checkpoint or (lambda _stage, _data: None)
        self._pause_at_safe_point = pause_at_safe_point or (lambda: None)
        self._pause_deferral_depth = 0

    def resolve(
        self,
        pending: PendingWarehouseListing,
        *,
        wait_when_slots_full: bool,
        poll_seconds: int,
        uncertain_confirm_pending: bool = False,
        force_unlist: bool = False,
    ) -> SellSlotRecoveryResult:
        try:
            self._check_stopped()
            if uncertain_confirm_pending:
                image = self._port.capture_image()
                if self._detector.read_unlist_dialog(image).header is not None:
                    # Cancel an unconfirmed modal first. If the earlier click
                    # already completed, no modal exists and the new first row
                    # can be handled normally.
                    self._port.press_escape()
                    self._port.wait(0.35)
            if not self._port.navigate_to(
                GamePageId.MARKET_SELL,
                reason="检查上一轮交易行直售挂单的售位",
            ):
                return self._result(
                    SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                    "无法安全导航到交易行出售页",
                )

            while True:
                self._check_stopped()
                if self._finish_requested():
                    return self._result(SellSlotRecoveryOutcome.FINISHED, "当前售位等待已结束，未开始新的下架")
                slots = self._read_stable_slots()
                if slots is None:
                    return self._result(
                        SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                        "连续读取后仍无法确认有效售位",
                    )
                assert slots.used is not None and slots.total is not None
                if self._finish_requested():
                    return self._result(SellSlotRecoveryOutcome.FINISHED, "当前售位等待已结束")
                if force_unlist:
                    self._on_stage("unlisting", "恢复未完成的全部下架流程")
                    return self._unlist_all(
                        slots,
                        allow_empty_without_decrease=True,
                    )
                if slots.used < slots.total:
                    self._on_stage(
                        "available",
                        f"售位已释放：{slots.used}/{slots.total}",
                    )
                    return self._result(
                        SellSlotRecoveryOutcome.SLOT_AVAILABLE,
                        f"售位已释放：{slots.used}/{slots.total}",
                        slots,
                    )

                now = self._clock()
                if not wait_when_slots_full or pending.expired(now):
                    reason = (
                        "用户关闭满售位等待"
                        if not wait_when_slots_full
                        else "挂单等待时间已到"
                    )
                    self._on_stage("unlisting", f"{reason}，开始连续下架全部挂单")
                    return self._unlist_all(
                        slots,
                        allow_empty_without_decrease=False,
                    )

                remaining = max(0.0, pending.deadline - now)
                wait_seconds = min(float(poll_seconds), remaining)
                self._on_stage(
                    "waiting",
                    f"售位已满 {slots.used}/{slots.total}，"
                    f"{wait_seconds:.0f} 秒后复查",
                )
                if self._timed_finish:
                    while wait_seconds > 0 and not self._finish_requested():
                        self._check_stopped()
                        step = min(0.1, wait_seconds)
                        self._port.wait(step)
                        wait_seconds -= step
                else:
                    self._port.wait(max(0.05, wait_seconds))
        except _RecoveryStopped:
            return self._result(
                SellSlotRecoveryOutcome.STOPPED,
                "售位等待与下架流程已停止",
            )
        except PageRecoveryFailed:
            raise
        except Exception as exc:
            self._log(f"售位等待与下架异常：{exc}")
            return self._result(
                SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                f"售位等待与下架异常：{exc}",
            )

    def _read_stable_slots(self) -> SellSlotsSnapshot | None:
        previous: tuple[int, int] | None = None
        for _attempt in range(self._SLOT_STABILITY_ATTEMPTS):
            self._check_stopped()
            slots = self._sell_ocr.read_slots(self._port.capture_image())
            if self._valid_slots(slots):
                current = (int(slots.used), int(slots.total))
                if current == previous:
                    return slots
                previous = current
            else:
                previous = None
            self._port.wait(0.25)
        return None

    def _unlist_all(
        self,
        initial_slots: SellSlotsSnapshot,
        *,
        allow_empty_without_decrease: bool = True,
    ) -> SellSlotRecoveryResult:
        if not self._port.navigate_to(
            GamePageId.MARKET_SELL,
            reason="超时后连续下架全部挂单",
        ):
            return self._result(
                SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                "下架前无法确认交易行出售页",
                initial_slots,
            )

        unlisted = 0
        empty_image: Image.Image | None = None
        empty_confirmations = 0
        for _order_index in range(self._MAX_UNLIST_ORDERS + 1):
            self._check_stopped()
            if self._finish_requested():
                return self._result(SellSlotRecoveryOutcome.FINISHED,
                    "当前下架操作已确认，不再下架其他挂单", initial_slots, unlisted)
            image = self._port.capture_image()
            action = self._detector.find_first_unlist_action(image)
            if action is None:
                if self._port.observe_page() is not GamePageId.MARKET_SELL:
                    return self._result(
                        SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                        "下架按钮消失时未确认处于交易行出售页",
                        initial_slots,
                        unlisted,
                    )
                if empty_image is not None and self._empty_region_stable(
                    empty_image,
                    image,
                ):
                    empty_confirmations += 1
                else:
                    empty_confirmations = 1
                empty_image = image
                if empty_confirmations >= self._EMPTY_CONFIRMATIONS:
                    final_slots = self._sell_ocr.read_slots(image)
                    if (
                        not allow_empty_without_decrease
                        and unlisted == 0
                        and (
                            not self._valid_slots(final_slots)
                            or final_slots.used >= initial_slots.used
                        )
                    ):
                        return self._result(
                            SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                            "下架按钮已消失，但上架数量没有减少",
                            final_slots,
                            unlisted,
                        )
                    self._log(
                        f"连续确认下架按钮已经消失；共下架 {unlisted} 个挂单"
                    )
                    return self._result(
                        SellSlotRecoveryOutcome.UNLISTED,
                        "全部挂单已下架，物品已回到仓库",
                        final_slots if self._valid_slots(final_slots) else initial_slots,
                        unlisted,
                    )
                self._port.wait(0.35)
                continue

            empty_image = None
            empty_confirmations = 0
            if unlisted >= self._MAX_UNLIST_ORDERS:
                return self._result(
                    SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                    f"连续下架达到安全上限 {self._MAX_UNLIST_ORDERS}",
                    initial_slots,
                    unlisted,
                )
            before_slots = self._sell_ocr.read_slots(image)
            if not self._valid_slots(before_slots):
                before_slots = self._read_stable_slots()
            if before_slots is None or not self._valid_slots(before_slots):
                return self._result(
                    SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                    "下架前无法确认当前上架数量",
                    initial_slots,
                    unlisted,
                )
            if self._finish_requested():
                return self._result(SellSlotRecoveryOutcome.FINISHED,
                    "下架操作已结束，未打开新的下架确认框", before_slots, unlisted)
            # Keep the modal open-and-confirm sequence atomic for pausing.
            # The confirmation overlay is not a safe page to revalidate.
            self._pause_deferral_depth += 1
            dialog, dialog_image = self._open_unlist_dialog(action)
            if dialog is None or dialog_image is None or dialog.confirm_action is None:
                self._pause_deferral_depth = max(
                    0,
                    self._pause_deferral_depth - 1,
                )
                self._pause_at_safe_point()
                return self._result(
                    SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                    "未同时确认下架物品标题和最终下架按钮",
                    initial_slots,
                    unlisted,
                )
            self._on_checkpoint(
                "unlist_confirm_pending",
                {"unlisted_orders": unlisted},
            )
            confirmed_slots = self._confirm_unlist(
                dialog,
                dialog_image,
                before_slots,
            )
            if confirmed_slots is None:
                self._pause_deferral_depth = max(
                    0,
                    self._pause_deferral_depth - 1,
                )
                return self._result(
                    SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
                    "点击最终下架后结果不确定，已停止重复点击",
                    initial_slots,
                    unlisted,
                )
            unlisted += 1
            self._on_checkpoint(
                "unlisting",
                {"unlisted_orders": unlisted},
            )
            self._log(f"第 {unlisted} 个挂单已确认下架")
            self._pause_deferral_depth = max(
                0,
                self._pause_deferral_depth - 1,
            )
            self._pause_at_safe_point()

        return self._result(
            SellSlotRecoveryOutcome.RECOVERY_REQUIRED,
            f"连续下架达到安全上限 {self._MAX_UNLIST_ORDERS}",
            initial_slots,
            unlisted,
        )

    def _open_unlist_dialog(
        self,
        initial_action: UnlistControlMatch,
    ) -> tuple[UnlistDialogSnapshot | None, Image.Image | None]:
        action = initial_action
        action_image = self._port.capture_image()
        for open_attempt in range(self._DIALOG_OPEN_ATTEMPTS):
            self._port.click_normalized(
                self._normalized_center(action, action_image)
            )
            dialog: UnlistDialogSnapshot | None = None
            dialog_image: Image.Image | None = None
            for _frame_attempt in range(self._DIALOG_FRAME_ATTEMPTS):
                self._check_stopped()
                self._port.wait(self._DIALOG_FRAME_WAIT_SECONDS)
                candidate_image = self._port.capture_image()
                candidate = self._detector.read_unlist_dialog(candidate_image)
                if candidate.confirmed:
                    # Keep waiting until the full 1.5 second settle window has
                    # elapsed; use the latest confirmed frame for the click.
                    dialog = candidate
                    dialog_image = candidate_image
            if dialog is not None and dialog_image is not None:
                return dialog, dialog_image
            # A missed click can leave the cursor hovering over the action
            # without opening the modal. Give the game another short settle
            # period before retrying the same first-row action.
            self._port.wait(self._DIALOG_RETRY_DELAY_SECONDS)
            retry_image = self._port.capture_image()
            retry_dialog = self._detector.read_unlist_dialog(retry_image)
            if retry_dialog.confirmed:
                return retry_dialog, retry_image
            if open_attempt + 1 < self._DIALOG_OPEN_ATTEMPTS:
                retry_action = self._detector.find_first_unlist_action(retry_image)
                if retry_action is not None:
                    action = retry_action
                    action_image = retry_image
                    continue
            self._port.press_escape()
            self._port.wait(0.25)
            if not self._port.navigate_to(
                GamePageId.MARKET_SELL,
                reason="下架确认弹窗识别失败后安全重试",
            ):
                break
            if open_attempt + 1 < self._DIALOG_OPEN_ATTEMPTS:
                action_image = self._port.capture_image()
                next_action = self._detector.find_first_unlist_action(action_image)
                if next_action is None:
                    break
                action = next_action
        return None, None

    def _confirm_unlist(
        self,
        dialog: UnlistDialogSnapshot,
        dialog_image: Image.Image,
        before_slots: SellSlotsSnapshot,
    ) -> SellSlotsSnapshot | None:
        assert dialog.confirm_action is not None
        self._port.click_normalized(
            self._normalized_center(dialog.confirm_action, dialog_image)
        )
        confirmed_slots = self._wait_confirm_result(before_slots)
        if confirmed_slots is not None:
            return confirmed_slots

        # If the modal is still visible after the normal ~1.5 second result
        # window, the first click may have been swallowed while the dialog was
        # settling. Retry the modal action once after the requested 0.5 second
        # grace period. The quantity check remains the only success signal.
        self._port.wait(self._DIALOG_RETRY_DELAY_SECONDS)
        retry_image = self._port.capture_image()
        retry_dialog = self._detector.read_unlist_dialog(retry_image)
        if not retry_dialog.confirmed or retry_dialog.confirm_action is None:
            return None
        self._port.click_normalized(
            self._normalized_center(retry_dialog.confirm_action, retry_image)
        )
        return self._wait_confirm_result(before_slots)

    def _wait_confirm_result(
        self,
        before_slots: SellSlotsSnapshot,
    ) -> SellSlotsSnapshot | None:
        for _attempt in range(self._CONFIRM_RESULT_ATTEMPTS):
            self._check_stopped()
            self._port.wait(self._CONFIRM_RESULT_WAIT_SECONDS)
            image = self._port.capture_image()
            if self._detector.read_unlist_dialog(image).header is not None:
                continue
            if self._port.observe_page() is not GamePageId.MARKET_SELL:
                continue
            slots = self._sell_ocr.read_slots(image)
            if self._valid_slots(slots) and slots.used < before_slots.used:
                return slots
        return None

    def _check_stopped(self) -> None:
        if self._port.stopped():
            raise _RecoveryStopped()
        if self._pause_deferral_depth == 0:
            self._pause_at_safe_point()

    @staticmethod
    def _valid_slots(slots: SellSlotsSnapshot) -> bool:
        return bool(
            slots.used is not None
            and slots.total is not None
            and 0 <= slots.used <= slots.total
            and 1 <= slots.total <= 99
            and slots.confidence is not None
            and slots.confidence >= 0.70
        )

    @staticmethod
    def _normalized_center(
        match: UnlistControlMatch,
        image: Image.Image,
    ) -> tuple[float, float]:
        return (
            (match.bounds.left + match.bounds.width / 2) / image.width,
            (match.bounds.top + match.bounds.height / 2) / image.height,
        )

    @classmethod
    def _empty_region_stable(
        cls,
        previous: Image.Image,
        current: Image.Image,
    ) -> bool:
        if previous.size != current.size:
            return False
        bounds = ReferenceLayout(*current.size).top_box(cls._EMPTY_ORDER_REGION)
        before = cv2.cvtColor(
            np.asarray(previous.crop(bounds).convert("RGB")),
            cv2.COLOR_RGB2GRAY,
        )
        after = cv2.cvtColor(
            np.asarray(current.crop(bounds).convert("RGB")),
            cv2.COLOR_RGB2GRAY,
        )
        if before.size == 0 or before.shape != after.shape:
            return False
        return float(np.mean(cv2.absdiff(before, after))) <= cls._EMPTY_MAX_MEAN_DELTA

    @staticmethod
    def _result(
        outcome: SellSlotRecoveryOutcome,
        message: str,
        slots: SellSlotsSnapshot | None = None,
        unlisted_orders: int = 0,
    ) -> SellSlotRecoveryResult:
        return SellSlotRecoveryResult(
            outcome=outcome,
            message=message,
            used_slots=slots.used if slots is not None else None,
            total_slots=slots.total if slots is not None else None,
            unlisted_orders=unlisted_orders,
        )


class _RecoveryStopped(Exception):
    pass
