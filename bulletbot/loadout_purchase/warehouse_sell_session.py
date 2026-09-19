from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .ammunition_catalog import canonical_ammunition_name


SELL_UNITS_PER_SLOT = 200


class ListingPriceMode(StrEnum):
    CURRENT_LOWEST = "current_lowest"
    ONE_TICK_LOWER = "one_tick_lower"
    TWO_TICKS_LOWER = "two_ticks_lower"
    FIXED = "fixed"


class WarehouseSellOutcome(StrEnum):
    FINISHED = "finished"
    LISTED = "listed"
    BELOW_TRIGGER = "below_trigger"
    INVENTORY_NOT_FOUND = "inventory_not_found"
    NO_SELL_SLOT = "no_sell_slot"
    RECOVERY_REQUIRED = "recovery_required"
    STOPPED = "stopped"


class SellSlotRecoveryOutcome(StrEnum):
    SLOT_AVAILABLE = "slot_available"
    UNLISTED = "unlisted"
    RECOVERY_REQUIRED = "recovery_required"
    STOPPED = "stopped"
    FINISHED = "finished"


@dataclass(frozen=True)
class WarehouseSellRequest:
    ammunition_name: str
    sell_trigger_price: int
    price_mode: ListingPriceMode | str = ListingPriceMode.CURRENT_LOWEST
    fixed_price: int | None = None
    max_batches: int = 32
    finish_after_below_price: int | None = None
    check_trigger_each_batch: bool = False

    def __post_init__(self) -> None:
        name = canonical_ammunition_name(self.ammunition_name)
        if not name:
            raise ValueError("交易行直售子弹名称不能为空")
        if self.sell_trigger_price <= 0:
            raise ValueError("卖出阈值必须大于 0")
        if self.max_batches < 1:
            raise ValueError("最大分批次数必须至少为 1")
        if self.finish_after_below_price is not None and self.finish_after_below_price <= 0:
            raise ValueError("高价出售复查门槛必须为正数")
        try:
            price_mode = ListingPriceMode(self.price_mode)
        except ValueError as exc:
            raise ValueError(f"未知的交易行直售价格模式：{self.price_mode}") from exc
        object.__setattr__(self, "ammunition_name", name)
        object.__setattr__(self, "price_mode", price_mode)
        if price_mode is ListingPriceMode.FIXED:
            if self.fixed_price is None or self.fixed_price <= 0:
                raise ValueError("自定义价格模式必须提供大于 0 的固定价格")
        elif self.fixed_price is not None and self.fixed_price <= 0:
            raise ValueError("固定价格必须大于 0")


@dataclass(frozen=True)
class WarehouseListedBatch:
    quantity: int
    unit_price: int
    lowest_price: int
    inventory_before: int


@dataclass(frozen=True)
class WarehouseSellResult:
    outcome: WarehouseSellOutcome
    batches: tuple[WarehouseListedBatch, ...] = ()
    message: str = ""
    first_lowest_price: int | None = None

    @property
    def listed_quantity(self) -> int:
        return sum(batch.quantity for batch in self.batches)

    @property
    def completed(self) -> bool:
        return self.outcome is WarehouseSellOutcome.LISTED


@dataclass(frozen=True)
class PendingWarehouseListing:
    scheme_index: int
    ammunition_name: str
    listed_quantity: int
    started_at: float
    deadline: float
    additional_ammunition_names: tuple[str, ...] = ()

    def expired(self, now: float) -> bool:
        return now >= self.deadline

    @property
    def ammunition_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for raw_name in (self.ammunition_name, *self.additional_ammunition_names):
            name = raw_name.strip()
            if name and name not in names:
                names.append(name)
        return tuple(names)


@dataclass(frozen=True)
class SellSlotRecoveryResult:
    outcome: SellSlotRecoveryOutcome
    message: str
    used_slots: int | None = None
    total_slots: int | None = None
    unlisted_orders: int = 0
