from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from time import perf_counter
from typing import Callable, Protocol

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import (
    InventoryMatch,
    MarketDepthLevel,
    Rect,
    SellListingSnapshot,
    SellSlotsSnapshot,
)
from bulletbot.navigation.models import GamePageId, PageKind
from bulletbot.navigation.catalog import page_descriptor
from bulletbot.navigation.action_recovery import (
    ActionState, PageActionStalled, PageRecoveryFailed, PageRecoveryStopped,
    recover_page_action,
)
from bulletbot.ocr.sell_ocr import SellOcr
from bulletbot.ocr.warehouse_ocr import WarehouseOcr
from bulletbot.vision.warehouse_inventory import WarehouseInventoryDetector

from .ammunition_catalog import (
    market_ammunition_name,
    market_price_matches,
    market_price_range_text,
)
from .warehouse_sell_session import (
    ListingPriceMode,
    WarehouseListedBatch,
    WarehouseSellOutcome,
    WarehouseSellRequest,
    WarehouseSellResult,
    SELL_UNITS_PER_SLOT,
)


class _RescanListingInventory(Exception):
    pass


class WarehouseSellPort(Protocol):
    def navigate_to(self, page: GamePageId, *, reason: str) -> bool: ...

    def capture_image(self) -> Image.Image: ...

    def capture_listing(
        self,
        *,
        include_depth: bool,
        expected_name: str,
    ) -> SellListingSnapshot: ...

    def capture_listing_identity_and_lowest(
        self,
        *,
        expected_name: str,
    ) -> tuple[str | None, float | None, int | None]: ...

    def capture_listing_lowest_price_full_editor(
        self,
    ) -> tuple[int | None, Image.Image]: ...

    def capture_listing_price(self) -> int | None: ...

    def capture_listing_quantity_and_slots(
        self,
    ) -> tuple[int | None, int | None, int | None, int | None]: ...

    def capture_listing_depth_prices(self) -> tuple[MarketDepthLevel, ...]: ...

    def capture_sell_slots(self) -> SellSlotsSnapshot: ...

    def listing_editor_ready(self) -> bool: ...

    def inspect_listing_editor(self) -> tuple[bool, tuple[str, ...]]: ...

    def observe_page(self) -> GamePageId: ...

    def window_generation(self) -> int: ...

    def click_normalized(self, point: tuple[float, float]) -> None: ...

    def move_normalized(self, point: tuple[float, float]) -> None: ...

    def scroll_inventory(self, notches: int) -> None: ...

    def press_escape(self) -> None: ...

    def estimate_quantity(self, target_quantity: int, inventory_total: int) -> None: ...

    def adjust_quantity(self, delta: int) -> None: ...

    def replace_price(self, value: int) -> None: ...

    def confirm_listing(self) -> None: ...

    def claim_sale_proceeds(self) -> bool: ...

    def wait(self, seconds: float) -> None: ...

    def stopped(self) -> bool: ...


@dataclass(frozen=True)
class ListingPricePlan:
    price: int | None
    tick: int | None
    reason: str


class WarehouseSellWorkflow:
    """List one ammunition type from the market sell inventory.

    The workflow owns business ordering and validation only.  Window capture,
    navigation, and input stay behind a port so the mail-storage worker can
    execute it under its existing input lease.
    """

    _GRID_STABLE_ATTEMPTS = 3
    _GRID_STABLE_MAX_MEAN_DELTA = 4.0
    _HOVER_CONFIRMATIONS = 2
    _SLOT_STABILITY_ATTEMPTS = 5
    _MAX_CELL_CANDIDATES_PER_BOX = 12
    _MAIN_BOX_INDEX = 0
    _MAIN_BOX_SCROLL_NOTCHES = 9
    _MAIN_BOX_MAX_VIEWPORTS = 24
    _MAIN_BOX_SCROLL_END_MAX_MEAN_DELTA = 2.5
    _QUANTITY_SLIDER_THRESHOLD = 200
    _MAX_QUANTITY_CORRECTIONS = 3
    _PRICE_INPUT_ATTEMPTS = 2
    _PAGE_WAIT_ATTEMPTS = 8
    _LISTING_EDITOR_INITIAL_WAIT_SECONDS = 1.0
    _LISTING_EDITOR_RETRY_WAIT_SECONDS = 0.5
    _LISTING_LOWEST_RETRY_WAIT_SECONDS = 0.5
    _LISTING_SAFE_HOVER = (0.90, 0.90)

    def __init__(
        self,
        port: WarehouseSellPort,
        *,
        inventory_detector: WarehouseInventoryDetector | None = None,
        warehouse_ocr: WarehouseOcr | None = None,
        sell_ocr: SellOcr | None = None,
        log: Callable[[str], None] | None = None,
        on_checkpoint: Callable[[str, dict[str, object]], None] | None = None,
        on_diagnostic_frame: Callable[[str, Image.Image], None] | None = None,
        on_first_batch_listed: Callable[[], bool] | None = None,
        pause_at_safe_point: Callable[[], None] | None = None,
        finish_requested: Callable[[], bool] | None = None,
        restart_game: Callable[[str], None] | None = None,
    ) -> None:
        self._port = port
        self._inventory = inventory_detector or WarehouseInventoryDetector()
        self._warehouse_ocr = warehouse_ocr or WarehouseOcr()
        self._sell_ocr = sell_ocr or SellOcr()
        self._log = log or (lambda _message: None)
        self._on_checkpoint = on_checkpoint or (lambda _stage, _data: None)
        self._on_diagnostic_frame = on_diagnostic_frame or (
            lambda _label, _image: None
        )
        self._on_first_batch_listed = on_first_batch_listed or (lambda: True)
        self._pause_at_safe_point = pause_at_safe_point or (lambda: None)
        self._pause_deferral_depth = 0
        self._finish_requested = finish_requested or (lambda: False)
        self._restart_game = restart_game
        self._editor_restart_used = False
        self._editor_page_relocations = 0

    def run(
        self,
        request: WarehouseSellRequest,
        *,
        prior_batches: tuple[WarehouseListedBatch, ...] = (),
        sale_gate_passed: bool | None = None,
    ) -> WarehouseSellResult:
        batches = list(prior_batches)
        gate_passed = bool(prior_batches) if sale_gate_passed is None else bool(
            sale_gate_passed
        )
        first_lowest_price = (
            prior_batches[0].lowest_price if prior_batches else None
        )
        try:
            self._check_stopped()
            if not self._port.navigate_to(
                GamePageId.MARKET_SELL,
                reason="配装买入完成后进入交易行出售页直售",
            ):
                return self._result(
                    WarehouseSellOutcome.RECOVERY_REQUIRED,
                    batches,
                    "无法导航到交易行出售页",
                    first_lowest_price,
                )

            while len(batches) < request.max_batches:
                self._check_stopped()
                market_slots = self._read_stable_market_slots()
                if market_slots is None:
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "交易行出售页未能稳定读取上架数量",
                        first_lowest_price,
                    )
                assert market_slots.used is not None
                assert market_slots.total is not None
                self._log(
                    f"交易行当前上架数量："
                    f"{market_slots.used}/{market_slots.total}"
                )
                if market_slots.available == 0:
                    return self._result(
                        WarehouseSellOutcome.NO_SELL_SLOT,
                        batches,
                        f"当前售位已满 {market_slots.used}/{market_slots.total}，"
                        "等待售位释放后继续上架剩余子弹",
                        first_lowest_price,
                    )
                try:
                    listing = self._open_listing_editor(request.ammunition_name)
                except _RescanListingInventory:
                    continue
                if listing is None:
                    outcome = (
                        WarehouseSellOutcome.LISTED
                        if batches or gate_passed
                        else WarehouseSellOutcome.INVENTORY_NOT_FOUND
                    )
                    message = (
                        f"{request.ammunition_name} 的出售页库存已经上架完毕"
                        if batches or gate_passed
                        else "交易行出售页前 7 个实际库存箱均未确认到 "
                        f"{request.ammunition_name}"
                    )
                    return self._result(
                        outcome,
                        batches,
                        message,
                        first_lowest_price,
                    )

                if not self._listing_identity_confirmed(
                    listing,
                    request.ammunition_name,
                ):
                    self._save_current_editor_frame("listing_editor_identity_failed")
                    self._recover_to_market_sell("上架编辑器商品名称未确认")
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "上架编辑器中的完整商品名称与目标不一致",
                        first_lowest_price,
                    )
                if listing.lowest_price is None or listing.lowest_price <= 0:
                    self._recover_to_market_sell("最低价 OCR 失败")
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "未识别到有效的当前最低价",
                        first_lowest_price,
                    )
                if not market_price_matches(
                    request.ammunition_name,
                    listing.lowest_price,
                ):
                    self._recover_to_market_sell(".300 BLK 价格档位与目标不一致")
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        f"{request.ammunition_name} 的市场最低价 "
                        f"{listing.lowest_price:,} 不在确认档位 "
                        f"{market_price_range_text(request.ammunition_name)} 内",
                        first_lowest_price,
                    )
                if not gate_passed:
                    first_lowest_price = listing.lowest_price
                if not gate_passed or request.check_trigger_each_batch:
                    if listing.lowest_price < request.sell_trigger_price:
                        self._recover_to_market_sell("当前行情低于卖出阈值，取消本批上架")
                        return self._result(
                            WarehouseSellOutcome.BELOW_TRIGGER,
                            batches,
                            f"当前最低价 {listing.lowest_price:,} 低于阈值 "
                            f"{request.sell_trigger_price:,}",
                            first_lowest_price,
                        )
                    gate_passed = True

                if request.price_mode in {
                    ListingPriceMode.ONE_TICK_LOWER,
                    ListingPriceMode.TWO_TICKS_LOWER,
                }:
                    listing = replace(
                        listing,
                        depth_levels=self._capture_depth_prices(),
                    )
                price_plan = self.calculate_listing_price(listing, request)
                if price_plan.price is None and (
                    request.price_mode
                    in {
                        ListingPriceMode.ONE_TICK_LOWER,
                        ListingPriceMode.TWO_TICKS_LOWER,
                    }
                ):
                    # A small price-column OCR miss is common during chart
                    # animation.  Re-read once before leaving the editor.
                    self._port.wait(0.35)
                    listing = replace(
                        listing,
                        depth_levels=self._capture_depth_prices(),
                    )
                    price_plan = self.calculate_listing_price(listing, request)
                if price_plan.price is None:
                    if price_plan.reason == "价格柱间隔不一致，拒绝猜测":
                        self._log("价格柱间隔不一致：按 Esc 退出编辑器，重新打开读取行情")
                        self._port.press_escape()
                        self._port.wait(0.30)
                        self._check_stopped()
                        if not self._port.navigate_to(GamePageId.MARKET_SELL, reason="价格柱不一致，重开上架编辑器"):
                            return self._result(WarehouseSellOutcome.RECOVERY_REQUIRED, batches,
                                                "价格柱重试无法返回出售页", first_lowest_price)
                        continue
                    self._recover_to_market_sell("无法计算挂牌价")
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        price_plan.reason,
                        first_lowest_price,
                    )
                priced = self._apply_price(
                    request,
                    listing,
                    price_plan.price,
                )
                if priced is None:
                    self._recover_to_market_sell("挂牌价未通过 OCR 回读")
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "挂牌价格输入后连续回读不一致",
                        first_lowest_price,
                    )
                priced = self._refresh_quantity_and_slots(priced)
                quantity_plan = self._quantity_plan(priced)
                if quantity_plan is None:
                    self._recover_to_market_sell("售位或库存数量不可用")
                    return self._result(
                        WarehouseSellOutcome.NO_SELL_SLOT,
                        batches,
                        "当前没有可安全使用的售位，或库存数量未识别",
                        first_lowest_price,
                    )
                target_quantity, expected_slots, slot_capacity = quantity_plan
                self._log(
                    f"上架数量目标：OCR 售位上限 "
                    f"{priced.available_slots} × 固定 {SELL_UNITS_PER_SLOT} 发"
                    f" = {slot_capacity} 发；出售页库存 "
                    f"{priced.inventory_total} 发；本批精确目标 "
                    f"{target_quantity} 发"
                )
                adjusted = self._adjust_quantity(
                    priced,
                    target_quantity,
                    expected_slots,
                )
                if adjusted is None:
                    self._recover_to_market_sell("上架数量未通过 OCR 回读")
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "粗调及加减精调后，OCR 仍未确认精确目标数量",
                        first_lowest_price,
                    )
                if not self._final_listing_valid(
                    adjusted,
                    request.ammunition_name,
                    target_quantity,
                    expected_slots,
                    price_plan.price,
                ):
                    self._recover_to_market_sell("最终上架校验失败")
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "最终名称、数量、价格或售位校验失败",
                        first_lowest_price,
                    )
                self._check_stopped()
                self._log(
                    f"数量和售位 OCR 已精确确认，立即上架 "
                    f"{request.ammunition_name}：{adjusted.quantity} 发，"
                    f"{price_plan.price:,}/发"
                )
                self._pause_deferral_depth += 1
                self._on_checkpoint(
                    "listing_confirm_pending",
                    {
                        "ammunition_name": request.ammunition_name,
                        "listed_batch_count": len(batches),
                        "listed_quantity_total": sum(
                            batch.quantity for batch in batches
                        ),
                        "pending_batch_quantity": int(adjusted.quantity or 0),
                        "pending_batch_price": price_plan.price,
                        "first_lowest_price": first_lowest_price,
                    },
                )
                self._port.confirm_listing()
                returned = self._wait_for_page(GamePageId.MARKET_SELL, seconds=0.45)
                if not returned:
                    # The final action may already have succeeded.  Never click
                    # it a second time when the outcome is not observable.
                    self._pause_deferral_depth = max(
                        0,
                        self._pause_deferral_depth - 1,
                    )
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "点击最终上架后未确认返回交易行出售页，已停止重复点击",
                        first_lowest_price,
                    )
                actual_quantity = int(adjusted.quantity or 0)
                inventory_before = int(adjusted.inventory_total or 0)
                batches.append(
                    WarehouseListedBatch(
                        quantity=actual_quantity,
                        unit_price=price_plan.price,
                        lowest_price=listing.lowest_price,
                        inventory_before=inventory_before,
                    )
                )
                self._on_checkpoint(
                    "listing_confirmed",
                    {
                        "ammunition_name": request.ammunition_name,
                        "listed_batch_count": len(batches),
                        "listed_quantity_total": sum(
                            batch.quantity for batch in batches
                        ),
                        "pending_batch_quantity": 0,
                        "pending_batch_price": None,
                        "first_lowest_price": first_lowest_price,
                    },
                )
                self._log(
                    f"第 {len(batches)} 批已上架并返回交易行出售页；"
                    f"累计 {sum(batch.quantity for batch in batches)} 发"
                )
                if len(batches) == 1 and not self._on_first_batch_listed():
                    self._pause_deferral_depth = max(
                        0,
                        self._pause_deferral_depth - 1,
                    )
                    return self._result(
                        WarehouseSellOutcome.RECOVERY_REQUIRED,
                        batches,
                        "本轮首批上架后的交易行货款领取未能返回出售页",
                        first_lowest_price,
                    )
                self._pause_deferral_depth = max(
                    0,
                    self._pause_deferral_depth - 1,
                )
                if (request.finish_after_below_price is not None
                        and listing.lowest_price < request.finish_after_below_price):
                    return self._result(
                        WarehouseSellOutcome.LISTED, batches,
                        "市场原始最低档跌破高价下界，本单已上架，停止下一单",
                        first_lowest_price,
                    )
                if inventory_before > 0 and actual_quantity >= inventory_before:
                    self._log(
                        f"本批已覆盖上架前全部库存 {inventory_before} 发，"
                        "出售页库存已清空，结束本轮扫描"
                    )
                    return self._result(
                        WarehouseSellOutcome.LISTED,
                        batches,
                        f"{request.ammunition_name} 的出售页库存已经上架完毕",
                        first_lowest_price,
                    )
                self._pause_at_safe_point()

            return self._result(
                WarehouseSellOutcome.RECOVERY_REQUIRED,
                batches,
                f"达到最大分批次数 {request.max_batches}，仍需重新核对出售页库存",
                first_lowest_price,
            )
        except _WorkflowFinished:
            # No final listing click is outstanding here. Leave the editor so
            # the next scheduled feature can navigate from a known page.
            try:
                if self._port.observe_page() is GamePageId.LISTING_EDITOR:
                    self._port.press_escape()
                    self._port.wait(0.30)
                if not self._port.navigate_to(GamePageId.MARKET_SELL, reason="定时交接：结束当前上架操作"):
                    raise RuntimeError("未能退出上架编辑器并返回出售页")
            except Exception as exc:
                return self._result(WarehouseSellOutcome.RECOVERY_REQUIRED, batches,
                                    str(exc), first_lowest_price)
            return self._result(WarehouseSellOutcome.FINISHED, batches,
                                "上架操作已结束，不再处理剩余库存", first_lowest_price)
        except _WorkflowStopped:
            return self._result(
                WarehouseSellOutcome.STOPPED,
                batches,
                "交易行直售已停止",
                first_lowest_price,
            )
        except PageRecoveryFailed:
            raise
        except Exception as exc:
            self._log(f"交易行直售异常：{exc}")
            if self._port.stopped():
                return self._result(WarehouseSellOutcome.STOPPED, batches,
                                    "交易行直售恢复已停止", first_lowest_price)
            self._recover_to_market_sell("交易行直售异常恢复")
            return self._result(
                WarehouseSellOutcome.RECOVERY_REQUIRED,
                batches,
                f"交易行直售异常：{exc}",
                first_lowest_price,
            )

    @staticmethod
    def calculate_listing_price(
        listing: SellListingSnapshot,
        request: WarehouseSellRequest,
    ) -> ListingPricePlan:
        lowest = listing.lowest_price
        if lowest is None or lowest <= 0:
            return ListingPricePlan(None, None, "当前最低价无效")
        if request.price_mode is ListingPriceMode.CURRENT_LOWEST:
            return ListingPricePlan(lowest, None, "按当前最低价原柱上架")
        if request.price_mode is ListingPriceMode.FIXED:
            return ListingPricePlan(
                request.fixed_price,
                None,
                "按用户配置的固定价格上架",
            )

        undercut_columns = (
            2 if request.price_mode is ListingPriceMode.TWO_TICKS_LOWER else 1
        )
        undercut_label = "两" if undercut_columns == 2 else "一"
        prices = sorted({level.price for level in listing.depth_levels if level.price > 0})
        if len(prices) < 3:
            return ListingPricePlan(
                None,
                None,
                f"有效价格柱不足 3 根，不能计算降{undercut_label}柱价格",
            )
        differences = [
            current - previous
            for previous, current in zip(prices, prices[1:])
            if current > previous
        ]
        tick = round(median(differences)) if differences else 0
        if tick <= 0 or any(
            abs(value - tick) > max(1, tick * 0.15) for value in differences
        ):
            return ListingPricePlan(None, tick or None, "价格柱间隔不一致，拒绝猜测")
        target = lowest - tick * undercut_columns
        if target <= 0:
            return ListingPricePlan(
                None,
                tick,
                f"降{undercut_label}柱后的挂牌价不是正数",
            )
        return ListingPricePlan(
            target,
            tick,
            f"按 {tick:,} 的柱间隔降{undercut_label}柱上架",
        )

    def _open_listing_editor(
        self,
        ammunition_name: str,
    ) -> SellListingSnapshot | None:
        image = self._port.capture_image()
        boxes = self._inventory.detect_box_buttons(image)
        self._log(
            f"交易行出售页识别到 {len(boxes)} 个实际库存箱；"
            "按从上到下顺序扫描前 7 个"
        )
        if not boxes:
            raise RuntimeError("交易行出售页未识别到任何实际库存箱")
        scan_targets = tuple(
            (f"出售库存箱 {box.index + 1}", box.index, box.bounds)
            for box in boxes[: self._inventory.MAX_BOXES]
        )
        incomplete_pages: list[str] = []
        for page_label, box_index, button_bounds in scan_targets:
            self._check_stopped()
            self._port.click_normalized(
                self._center_normalized(button_bounds, image)
            )
            stable = self._wait_for_stable_grid()
            if stable is None:
                self._log(f"{page_label}的库存网格未稳定，跳过")
                incomplete_pages.append(page_label)
                continue
            viewport_index = 1
            while True:
                listing = self._scan_inventory_view(
                    page_label,
                    box_index,
                    viewport_index,
                    stable,
                    ammunition_name,
                )
                if listing is not None:
                    return listing
                if box_index != self._MAIN_BOX_INDEX:
                    break
                if viewport_index >= self._MAIN_BOX_MAX_VIEWPORTS:
                    self._log(
                        f"{page_label}已扫描 {viewport_index} 屏仍未确认到底，"
                        "停止继续滚动"
                    )
                    incomplete_pages.append(f"{page_label}（未确认到底）")
                    break

                self._check_stopped()
                self._log(
                    f"{page_label}第 {viewport_index} 屏未找到 {ammunition_name}；"
                    f"向下滚动 {self._MAIN_BOX_SCROLL_NOTCHES} 个滚轮刻度后重新识别"
                )
                self._port.scroll_inventory(self._MAIN_BOX_SCROLL_NOTCHES)
                next_stable = self._wait_for_stable_grid()
                if next_stable is None:
                    self._log(f"{page_label}滚动后的库存网格未稳定，停止滚动扫描")
                    incomplete_pages.append(f"{page_label}（滚动后未稳定）")
                    break
                grid_bounds = self._sell_ocr.inventory_grid_bounds(next_stable)
                difference = self._grid_mean_delta(
                    stable,
                    next_stable,
                    grid_bounds,
                )
                if difference <= self._MAIN_BOX_SCROLL_END_MAX_MEAN_DELTA:
                    self._log(
                        f"{page_label}已滚动到底；"
                        f"库存网格变化 {difference:.2f}，切换下一个箱子"
                    )
                    break
                viewport_index += 1
                stable = next_stable
                self._log(
                    f"{page_label}进入第 {viewport_index} 屏；"
                    f"库存网格变化 {difference:.2f}"
                )
        if incomplete_pages:
            labels = "、".join(incomplete_pages)
            raise RuntimeError(f"{labels}的网格未完成稳定扫描")
        return None

    def _scan_inventory_view(
        self,
        page_label: str,
        box_index: int,
        viewport_index: int,
        stable: Image.Image,
        ammunition_name: str,
    ) -> SellListingSnapshot | None:
        viewport_label = (
            f"第 {viewport_index} 屏"
            if box_index == self._MAIN_BOX_INDEX
            else "当前屏"
        )
        excluded: set[tuple[int, int, int]] = set()
        for _attempt in range(self._MAX_CELL_CANDIDATES_PER_BOX):
            candidate = self._sell_ocr.find_inventory_match(
                stable,
                ammunition_name,
                box_index,
                excluded_cells=excluded,
            )
            if candidate is None:
                self._log(
                    f"{page_label}{viewport_label}未提名到 "
                    f"{ammunition_name} 候选格"
                )
                return None
            self._log(
                f"{page_label}{viewport_label}提名 {ammunition_name} 候选："
                f"第 {candidate.row + 1} 行第 {candidate.column + 1} 列，"
                f"{candidate.source} {candidate.confidence:.1%}"
            )
            if not self._rect_inside(
                candidate.bounds,
                self._sell_ocr.inventory_grid_bounds(stable),
            ):
                raise RuntimeError(f"{page_label}的候选格超出库存网格")
            confirmed_generation = self._hover_name_confirmed(
                candidate,
                ammunition_name,
                stable,
            )
            if confirmed_generation is None:
                excluded.add((box_index, candidate.row, candidate.column))
                continue
            listing = self._enter_listing_editor(
                candidate,
                ammunition_name,
                stable,
                confirmed_generation,
            )
            if listing is not None:
                return listing
            raise RuntimeError(
                f"已精确确认{page_label}中的目标子弹，"
                "但左键后未进入上架编辑器"
            )
        return None

    def _wait_for_stable_grid(self) -> Image.Image | None:
        self._port.wait(0.35)
        previous = self._port.capture_image()
        for _attempt in range(self._GRID_STABLE_ATTEMPTS):
            self._check_stopped()
            self._port.wait(0.20)
            current = self._port.capture_image()
            bounds = self._sell_ocr.inventory_grid_bounds(current)
            if self._grid_mean_delta(previous, current, bounds) <= self._GRID_STABLE_MAX_MEAN_DELTA:
                return current
            previous = current
        return None

    def _hover_name_confirmed(
        self,
        candidate: InventoryMatch,
        ammunition_name: str,
        image: Image.Image,
    ) -> int | None:
        generation = self._port.window_generation()
        point = self._center_normalized(candidate.bounds, image)
        self._port.move_normalized(point)
        confirmations = 0
        last_frame = image
        for _attempt in range(self._HOVER_CONFIRMATIONS * 2):
            self._port.wait(0.22)
            frame = self._port.capture_image()
            last_frame = frame
            if self._port.window_generation() != generation:
                raise RuntimeError("悬停确认期间游戏窗口代次发生变化")
            match = self._warehouse_ocr.read_hover_name(
                frame,
                ammunition_name,
                candidate.bounds,
            )
            if match is None:
                confirmations = 0
                continue
            confirmations += 1
            if confirmations >= self._HOVER_CONFIRMATIONS:
                self._log(
                    f"悬浮名称连续确认成功：{match.text} "
                    + (
                        f"/ {match.price:,} "
                        if match.price is not None
                        else ""
                    )
                    + f"({match.confidence:.1%})"
                )
                return generation
        self._log(
            f"候选格第 {candidate.row + 1} 行第 {candidate.column + 1} 列"
            f"未能连续确认悬浮名称 {ammunition_name}"
        )
        self._save_diagnostic_frame(
            f"market_sell_hover_failed_tab_{candidate.tab_index}_"
            f"row_{candidate.row}_column_{candidate.column}",
            last_frame,
        )
        return None

    def _save_diagnostic_frame(self, label: str, image: Image.Image) -> None:
        try:
            self._on_diagnostic_frame(label, image)
        except Exception as exc:
            self._log(f"保存交易行直售诊断帧失败：{exc}")

    def _enter_listing_editor(
        self,
        candidate: InventoryMatch,
        ammunition_name: str,
        image: Image.Image,
        confirmed_generation: int,
    ) -> SellListingSnapshot | None:
        point = self._center_normalized(candidate.bounds, image)
        if self._port.window_generation() != confirmed_generation:
            return None
        self._log("悬浮名称已确认，左键目标子弹打开上架编辑器")
        self._port.click_normalized(point)
        if not self._wait_for_listing_editor():
            if self._restart_game is None:
                self._save_current_editor_frame("listing_editor_entry_failed")
                self._recover_to_market_sell("左键后等待上架编辑器超时")
                return None
            self._recover_listing_entry(candidate, ammunition_name)
        self._editor_restart_used = False
        self._editor_page_relocations = 0
        self._port.move_normalized(self._LISTING_SAFE_HOVER)
        self._log("已确认上架编辑器并将鼠标移出控件区，避免遮挡价格和数量")
        expected_market_name = market_ammunition_name(ammunition_name)
        product_name = None
        product_confidence = None
        lowest_price = None
        for attempt in range(1, 3):
            started = perf_counter()
            read_name, read_confidence, read_lowest = (
                self._port.capture_listing_identity_and_lowest(
                    expected_name=expected_market_name,
                )
            )
            if read_name:
                product_name = read_name
                product_confidence = read_confidence
            lowest_price = read_lowest
            self._log(
                f"上架编辑器扩大区域 OCR[名称+最低价，尝试 {attempt}/2]："
                f"{(perf_counter() - started) * 1000:.0f} ms；"
                f"名称={read_name or '未识别'}，"
                f"最低价={read_lowest if read_lowest is not None else '未识别'}"
            )
            if lowest_price is not None and lowest_price > 0:
                break
            if attempt == 1:
                self._log("首次最低价 OCR 未识别；等待 0.5 秒后重试扩大区域")
                self._port.wait(self._LISTING_LOWEST_RETRY_WAIT_SECONDS)

        if lowest_price is None or lowest_price <= 0:
            started = perf_counter()
            lowest_price, fallback_image = (
                self._port.capture_listing_lowest_price_full_editor()
            )
            self._log(
                "上架编辑器整框 OCR[最低价兜底 3/3]："
                f"{(perf_counter() - started) * 1000:.0f} ms；"
                f"最低价={lowest_price if lowest_price is not None else '未识别'}"
            )
            if lowest_price is None or lowest_price <= 0:
                self._save_diagnostic_frame(
                    "listing_editor_lowest_price_failed",
                    fallback_image,
                )
                self._log("最低价三次识别均失败；已保存上架编辑器完整截图")
        return SellListingSnapshot(
            product_name=product_name,
            product_confidence=product_confidence,
            quantity=None,
            inventory_total=None,
            used_slots=None,
            available_slots=None,
            lowest_price=lowest_price,
            depth_levels=(),
            listing_price=None,
            expected_income=None,
        )

    def _save_current_editor_frame(self, label: str) -> None:
        try:
            self._save_diagnostic_frame(label, self._port.capture_image())
            self._log(f"已保存上架编辑器诊断截图：{label}")
        except Exception as exc:
            self._log(f"捕获上架编辑器诊断截图失败：{exc}")

    def _capture_depth_prices(self) -> tuple[MarketDepthLevel, ...]:
        started = perf_counter()
        levels = self._port.capture_listing_depth_prices()
        self._log(
            "上架编辑器局部 OCR[价格柱]："
            f"{(perf_counter() - started) * 1000:.0f} ms；"
            f"价格={[level.price for level in levels]}"
        )
        return levels

    @staticmethod
    def _quantity_plan(
        listing: SellListingSnapshot,
    ) -> tuple[int, int, int] | None:
        inventory_total = listing.inventory_total
        available_slots = listing.available_slots
        if (
            inventory_total is None
            or inventory_total <= 0
            or available_slots is None
            or available_slots <= 0
        ):
            return None
        slot_capacity = available_slots * SELL_UNITS_PER_SLOT
        target = min(inventory_total, slot_capacity)
        if target <= 0:
            return None
        expected_slots = (
            target + SELL_UNITS_PER_SLOT - 1
        ) // SELL_UNITS_PER_SLOT
        return target, expected_slots, slot_capacity

    def _adjust_quantity(
        self,
        listing: SellListingSnapshot,
        target: int,
        expected_slots: int,
    ) -> SellListingSnapshot | None:
        current = listing
        if self._quantity_acceptable(
            current,
            target,
            expected_slots,
        ):
            return current
        if (
            current.quantity is not None
            and current.inventory_total is not None
            and current.inventory_total > 1
            and abs(target - current.quantity) >= self._QUANTITY_SLIDER_THRESHOLD
        ):
            self._log(
                f"数量粗调：当前 {current.quantity} 发，滑杆定位到 "
                f"{target} 发"
            )
            self._port.estimate_quantity(target, current.inventory_total)
            self._port.wait(0.40)
            current = self._refresh_quantity_and_slots(current)

        for _correction in range(self._MAX_QUANTITY_CORRECTIONS + 1):
            if self._quantity_acceptable(
                current,
                target,
                expected_slots,
            ):
                return current
            if (
                current.quantity is None
                or current.inventory_total is None
                or not 1 <= current.quantity <= current.inventory_total
            ):
                return None
            remaining = target - current.quantity
            self._log(
                f"数量精调：OCR 回读 {current.quantity} 发，"
                f"目标差值 {remaining:+d}，一次通过加减按钮修正 "
                f"{remaining:+d} 次"
            )
            self._port.adjust_quantity(remaining)
            self._port.wait(0.40)
            current = self._refresh_quantity_and_slots(current)
        return None

    def _refresh_quantity_and_slots(
        self,
        listing: SellListingSnapshot,
    ) -> SellListingSnapshot:
        started = perf_counter()
        quantity, inventory_total, used_slots, available_slots = (
            self._port.capture_listing_quantity_and_slots()
        )
        self._log(
            "上架编辑器局部 OCR[数量+售位]："
            f"{(perf_counter() - started) * 1000:.0f} ms；"
            f"数量={quantity}/{inventory_total}，"
            f"售位={used_slots}/{available_slots}"
        )
        return replace(
            listing,
            quantity=quantity,
            inventory_total=inventory_total,
            used_slots=used_slots,
            available_slots=available_slots,
        )

    def _apply_price(
        self,
        request: WarehouseSellRequest,
        listing: SellListingSnapshot,
        target_price: int,
    ) -> SellListingSnapshot | None:
        current = listing
        if request.price_mode is ListingPriceMode.CURRENT_LOWEST:
            started = perf_counter()
            listing_price = self._port.capture_listing_price()
            self._log(
                "上架编辑器局部 OCR[价格框]："
                f"{(perf_counter() - started) * 1000:.0f} ms；"
                f"价格={listing_price if listing_price is not None else '未识别'}"
            )
            current = replace(current, listing_price=listing_price)
            return current if listing_price == target_price else None
        for _attempt in range(self._PRICE_INPUT_ATTEMPTS):
            if current.listing_price == target_price:
                return current
            self._port.replace_price(target_price)
            self._port.wait(0.40)
            started = perf_counter()
            listing_price = self._port.capture_listing_price()
            self._log(
                "上架编辑器局部 OCR[价格框]："
                f"{(perf_counter() - started) * 1000:.0f} ms；"
                f"价格={listing_price if listing_price is not None else '未识别'}"
            )
            current = replace(
                current,
                listing_price=listing_price,
            )
        return current if current.listing_price == target_price else None

    @classmethod
    def _final_listing_valid(
        cls,
        listing: SellListingSnapshot,
        ammunition_name: str,
        target_quantity: int,
        expected_slots: int,
        target_price: int,
    ) -> bool:
        return bool(
            cls._listing_identity_confirmed(listing, ammunition_name)
            and cls._quantity_acceptable(
                listing,
                target_quantity,
                expected_slots,
            )
            and listing.listing_price == target_price
            and listing.used_slots is not None
            and listing.available_slots is not None
            and 0 < listing.used_slots <= listing.available_slots
        )

    def _recover_listing_entry(self, candidate: InventoryMatch, ammunition_name: str) -> None:
        previous_page = None

        def observe() -> ActionState:
            nonlocal previous_page
            self._check_stopped()
            reader = getattr(self._port, "observe_recovery_page", self._port.observe_page)
            observed = reader()
            page = getattr(observed, "page_id", observed)
            if page != previous_page:
                self._log(f"打开编辑器后全图观察：{page_descriptor(page).chinese_name}")
                previous_page = page
            if not getattr(observed, "stable", True):
                return ActionState.UNKNOWN
            if page is GamePageId.LISTING_EDITOR:
                return ActionState.TARGET
            if page is GamePageId.MARKET_SELL:
                return ActionState.SOURCE
            if page_descriptor(page).kind in {PageKind.META, PageKind.TRANSITION}:
                return ActionState.UNKNOWN
            return ActionState.OTHER

        def retry() -> None:
            image = self._port.capture_image()
            fresh = self._sell_ocr.find_inventory_match(image, ammunition_name, candidate.tab_index)
            if fresh is None or not self._rect_inside(
                fresh.bounds, self._sell_ocr.inventory_grid_bounds(image),
            ):
                self._log("补点前未定位到目标子弹，跳过点击并重新观察页面")
                return
            generation = self._hover_name_confirmed(fresh, ammunition_name, image)
            if generation is None:
                self._log("补点前未确认目标子弹名称，跳过点击并重新观察页面")
                return
            if observe() is ActionState.SOURCE and self._port.window_generation() == generation:
                self._port.click_normalized(self._center_normalized(fresh.bounds, image))

        try:
            state = recover_page_action(
                observe=observe, retry=retry, wait=self._port.wait,
                stopped=self._port.stopped, log=self._log, label="打开目标子弹上架编辑器",
            )
        except PageRecoveryStopped as exc:
            raise _WorkflowStopped() from exc
        except PageActionStalled as exc:
            self._save_current_editor_frame("listing_editor_stalled")
            if self._editor_restart_used:
                raise PageRecoveryFailed("重启后仍无法打开上架编辑器，结束本次恢复") from exc
            self._editor_restart_used = True
            self._restart_game(str(exc))
            state = ActionState.OTHER
        if state is ActionState.TARGET:
            return
        self._editor_page_relocations += 1
        if self._editor_page_relocations > 3:
            raise PageRecoveryFailed("上架编辑器恢复反复偏离目标页面，停止继续输入")
        if not self._port.navigate_to(GamePageId.MARKET_SELL, reason="保留出售任务，重新扫描库存"):
            raise PageRecoveryFailed("上架编辑器恢复后无法返回出售页")
        raise _RescanListingInventory()

    def _wait_for_listing_editor(self) -> bool:
        """Allow the editor to settle, then use at most one retrying capture.

        The click opens a modal editor asynchronously.  Repeatedly sampling
        eight times made a transient or visually different editor look like a
        failed click and delayed recovery.  Keep the cursor on the clicked
        item until this bounded confirmation finishes.
        """
        self._check_stopped()
        self._port.wait(self._LISTING_EDITOR_INITIAL_WAIT_SECONDS)
        if self._listing_editor_ready("进入确认 1"):
            return True
        self._check_stopped()
        self._port.wait(self._LISTING_EDITOR_RETRY_WAIT_SECONDS)
        return self._listing_editor_ready("进入确认 2")

    def _listing_editor_ready(self, label: str) -> bool:
        started = perf_counter()
        inspector = getattr(self._port, "inspect_listing_editor", None)
        if callable(inspector):
            ready, evidence = inspector()
        else:
            ready = self._port.listing_editor_ready()
            evidence = ()
        details = f"；证据={'，'.join(evidence)}" if evidence else ""
        self._log(
            f"上架编辑器页面诊断[{label}]："
            f"{(perf_counter() - started) * 1000:.0f} ms；"
            f"结果={'确认' if ready else '未确认'}{details}"
        )
        return ready

    @staticmethod
    def _listing_identity_confirmed(
        listing: SellListingSnapshot,
        ammunition_name: str,
    ) -> bool:
        expected_names = {
            ammunition_name,
            market_ammunition_name(ammunition_name),
        }
        return any(
            SellOcr.listing_name_match_state(
                expected_name,
                listing.product_name or "",
            )
            == "match"
            for expected_name in expected_names
        )

    @staticmethod
    def _quantity_acceptable(
        listing: SellListingSnapshot,
        target_quantity: int,
        expected_slots: int,
    ) -> bool:
        return bool(
            listing.quantity == target_quantity
            and listing.used_slots == expected_slots
            and listing.available_slots is not None
            and expected_slots <= listing.available_slots
        )

    def _read_stable_market_slots(self) -> SellSlotsSnapshot | None:
        previous: tuple[int, int] | None = None
        for _attempt in range(self._SLOT_STABILITY_ATTEMPTS):
            self._check_stopped()
            slots = self._port.capture_sell_slots()
            valid = bool(
                slots.used is not None
                and slots.total is not None
                and 0 <= slots.used <= slots.total
                and 1 <= slots.total <= 99
                and slots.confidence is not None
                and slots.confidence >= 0.70
            )
            if valid:
                current = (int(slots.used), int(slots.total))
                if current == previous:
                    return slots
                previous = current
            else:
                previous = None
            self._port.wait(0.25)
        return None

    def _wait_for_page(self, page: GamePageId, *, seconds: float) -> bool:
        for _attempt in range(self._PAGE_WAIT_ATTEMPTS):
            self._check_stopped()
            self._port.wait(seconds)
            current = self._port.observe_page()
            if current is page:
                return True
            if current not in {
                GamePageId.UNKNOWN,
                GamePageId.TRANSITION,
                GamePageId.LISTING_EDITOR,
            }:
                return False
        return False

    def _recover_to_market_sell(self, reason: str) -> None:
        try:
            current = self._port.observe_page()
            if current is GamePageId.LISTING_EDITOR:
                self._port.press_escape()
                self._port.wait(0.30)
            self._port.navigate_to(GamePageId.MARKET_SELL, reason=reason)
        except PageRecoveryFailed:
            raise
        except Exception as exc:
            self._log(f"返回交易行出售页恢复失败：{exc}")

    def _check_stopped(self) -> None:
        if self._port.stopped():
            raise _WorkflowStopped()
        if self._pause_deferral_depth == 0:
            self._pause_at_safe_point()
            if self._finish_requested():
                raise _WorkflowFinished()

    @staticmethod
    def _center_normalized(bounds: Rect, image: Image.Image) -> tuple[float, float]:
        return (
            (bounds.left + bounds.width / 2) / image.width,
            (bounds.top + bounds.height / 2) / image.height,
        )

    @staticmethod
    def _rect_inside(inner: Rect, outer: Rect) -> bool:
        return (
            inner.width > 0
            and inner.height > 0
            and outer.left <= inner.left
            and outer.top <= inner.top
            and inner.right <= outer.right
            and inner.bottom <= outer.bottom
        )

    @staticmethod
    def _grid_mean_delta(
        previous: Image.Image,
        current: Image.Image,
        bounds: Rect,
    ) -> float:
        if previous.size != current.size:
            return float("inf")
        box = (bounds.left, bounds.top, bounds.right, bounds.bottom)
        before = cv2.cvtColor(np.asarray(previous.crop(box)), cv2.COLOR_RGB2GRAY)
        after = cv2.cvtColor(np.asarray(current.crop(box)), cv2.COLOR_RGB2GRAY)
        if before.shape != after.shape or before.size == 0:
            return float("inf")
        return float(np.mean(cv2.absdiff(before, after)))

    @staticmethod
    def _result(
        outcome: WarehouseSellOutcome,
        batches: list[WarehouseListedBatch],
        message: str,
        first_lowest_price: int | None,
    ) -> WarehouseSellResult:
        return WarehouseSellResult(
            outcome=outcome,
            batches=tuple(batches),
            message=message,
            first_lowest_price=first_lowest_price,
        )


class _WorkflowFinished(Exception):
    pass


class _WorkflowStopped(Exception):
    pass
