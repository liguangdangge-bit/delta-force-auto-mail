from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class WindowBinding:
    handle: int
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def bounds(self) -> Rect:
        return Rect(self.left, self.top, self.width, self.height)


class PageState(str, Enum):
    FAVORITES = "favorites"
    DETAIL = "detail"
    WAREHOUSE = "warehouse"
    WAREHOUSE_SELL_DIALOG = "warehouse_sell_dialog"
    MARKET_SELL = "market_sell"
    MARKET_CLAIM_COMPLETE = "market_claim_complete"
    LISTING_EDITOR = "listing_editor"
    MARKET_OTHER = "market_other"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PageObservation:
    state: PageState
    confidence: float
    evidence: tuple[str, ...] = ()
    screenshot_path: str | None = None


@dataclass(frozen=True)
class CardRegion:
    index: int
    column: int
    row: int
    bounds: Rect
    name_bounds: Rect
    price_bounds: Rect


@dataclass(frozen=True)
class OcrText:
    text: str
    confidence: float


@dataclass(frozen=True)
class MarketObservation:
    card: CardRegion
    name: OcrText | None
    price: int | None
    price_confidence: float | None


@dataclass(frozen=True)
class SellSlotsSnapshot:
    used: int | None
    total: int | None
    confidence: float | None = None

    @property
    def available(self) -> int | None:
        if self.used is None or self.total is None:
            return None
        return max(0, self.total - self.used)


@dataclass(frozen=True)
class SellOrderSnapshot:
    product_name: str | None
    quantity: int | None
    unit_price: int | None
    name_confidence: float | None = None
    quantity_confidence: float | None = None
    price_confidence: float | None = None


@dataclass(frozen=True)
class InventoryMatch:
    name: str
    confidence: float
    bounds: Rect
    tab_index: int
    row: int
    column: int
    source: str = "ocr"


@dataclass(frozen=True)
class MarketDepthLevel:
    price: int
    volume: int | None


@dataclass(frozen=True)
class SellListingSnapshot:
    product_name: str | None
    product_confidence: float | None
    quantity: int | None
    inventory_total: int | None
    used_slots: int | None
    available_slots: int | None
    lowest_price: int | None
    depth_levels: tuple[MarketDepthLevel, ...]
    listing_price: int | None
    expected_income: int | None
    screenshot_path: str | None = None


@dataclass(frozen=True)
class ScanSnapshot:
    capture_path: str | None
    annotated_path: str | None
    card_regions: tuple[CardRegion, ...]
    observations: tuple[MarketObservation, ...] = ()
    is_detail_page: bool = False


@dataclass(frozen=True)
class DetailSnapshot:
    is_detail_page: bool
    price: int | None
    price_confidence: float | None
    screenshot_path: str
    product_name: str | None = None
    product_name_confidence: float | None = None
    quantity: int | None = None
    max_quantity: int | None = None
    average_unit_price: int | None = None
    average_price_confidence: float | None = None
    price_button_bounds: Rect | None = None
    slider_bounds: Rect | None = None
    minus_button_bounds: Rect | None = None
    plus_button_bounds: Rect | None = None
    average_price_signature: str | None = None


@dataclass(frozen=True)
class BalanceSnapshot:
    balance: int | None
    confidence: float | None
    screenshot_path: str
    selected_ocr_text: str | None = None
    ocr_candidates: tuple[OcrText, ...] = ()
    diagnostic_screenshot_path: str | None = None


@dataclass(frozen=True)
class WatchRule:
    rule_id: str
    name: str
    target_price: int
    buy_quantity: int
    enabled: bool
    priority: int
    expected_sell_price: int = 0
    minimum_profit_per_unit: int = 50
    auto_sell_enabled: bool = False
    sell_trigger_price: int = 0
    sell_units_per_slot: int = 200
    allow_sell_undercut: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WatchRule":
        return cls(
            rule_id=str(value.get("rule_id", "")),
            name=str(value.get("name", "")),
            target_price=max(0, int(value.get("target_price", 0))),
            buy_quantity=max(1, int(value.get("buy_quantity", 31))),
            enabled=bool(value.get("enabled", True)),
            priority=int(value.get("priority", 0)),
            expected_sell_price=max(0, int(value.get("expected_sell_price", 0))),
            minimum_profit_per_unit=max(
                0, int(value.get("minimum_profit_per_unit", 50))
            ),
            auto_sell_enabled=bool(value.get("auto_sell_enabled", False)),
            sell_trigger_price=max(0, int(value.get("sell_trigger_price", 0))),
            sell_units_per_slot=max(1, int(value.get("sell_units_per_slot", 200))),
            allow_sell_undercut=bool(value.get("allow_sell_undercut", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "target_price": self.target_price,
            "buy_quantity": self.buy_quantity,
            "enabled": self.enabled,
            "priority": self.priority,
            "expected_sell_price": self.expected_sell_price,
            "minimum_profit_per_unit": self.minimum_profit_per_unit,
            "auto_sell_enabled": self.auto_sell_enabled,
            "sell_trigger_price": self.sell_trigger_price,
            "sell_units_per_slot": self.sell_units_per_slot,
            "allow_sell_undercut": self.allow_sell_undercut,
        }
