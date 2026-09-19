from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

from bulletbot.vision.inventory_templates import InventoryTemplateMatcher


@dataclass(frozen=True)
class AmmunitionDefinition:
    canonical_name: str
    source_name: str
    market_name: str
    display_name: str
    market_price_min: int | None = None
    market_price_max: int | None = None

    def accepts_market_price(self, price: int | None) -> bool:
        if self.market_price_min is None and self.market_price_max is None:
            return True
        if price is None or price <= 0:
            return False
        if self.market_price_min is not None and price < self.market_price_min:
            return False
        if self.market_price_max is not None and price > self.market_price_max:
            return False
        return True


def _catalog_path() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS"))
    else:
        root = Path(__file__).resolve().parents[2]
    return root / "assets" / "ammunition_names.csv"


def _optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


@lru_cache(maxsize=1)
def ammunition_definitions() -> tuple[AmmunitionDefinition, ...]:
    definitions: list[AmmunitionDefinition] = []
    try:
        with _catalog_path().open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                canonical_name = str(row.get("canonical_name", "") or "").strip()
                if not canonical_name:
                    continue
                source_name = str(row.get("source_name", "") or "").strip()
                market_name = str(row.get("market_name", "") or "").strip()
                display_name = str(row.get("display_name", "") or "").strip()
                definitions.append(
                    AmmunitionDefinition(
                        canonical_name=canonical_name,
                        source_name=source_name or canonical_name,
                        market_name=market_name or canonical_name,
                        display_name=display_name or canonical_name,
                        market_price_min=_optional_int(row.get("market_price_min")),
                        market_price_max=_optional_int(row.get("market_price_max")),
                    )
                )
    except OSError:
        return ()
    return tuple(definitions)


def ammunition_definition(product_name: str) -> AmmunitionDefinition | None:
    expected = InventoryTemplateMatcher.normalize_product_name(product_name)
    if not expected:
        return None
    for definition in ammunition_definitions():
        aliases = (
            definition.canonical_name,
            definition.source_name,
            definition.display_name,
        )
        if any(
            InventoryTemplateMatcher.normalize_product_name(alias) == expected
            for alias in aliases
        ):
            return definition
    return None


def canonical_ammunition_name(product_name: str) -> str:
    definition = ammunition_definition(product_name)
    return definition.canonical_name if definition is not None else product_name.strip()


def market_ammunition_name(product_name: str) -> str:
    definition = ammunition_definition(product_name)
    return definition.market_name if definition is not None else product_name.strip()


def display_ammunition_name(product_name: str) -> str:
    definition = ammunition_definition(product_name)
    return definition.display_name if definition is not None else product_name.strip()


def market_price_matches(product_name: str, price: int | None) -> bool:
    definition = ammunition_definition(product_name)
    return definition is None or definition.accepts_market_price(price)


def market_price_range_text(product_name: str) -> str:
    definition = ammunition_definition(product_name)
    if definition is None:
        return "无限制"
    minimum = definition.market_price_min
    maximum = definition.market_price_max
    if minimum is not None and maximum is not None:
        return f"{minimum:,}-{maximum:,}"
    if minimum is not None:
        return f">={minimum:,}"
    if maximum is not None:
        return f"<={maximum:,}"
    return "无限制"


@lru_cache(maxsize=1)
def supported_ammunition_names() -> tuple[str, ...]:
    return tuple(definition.canonical_name for definition in ammunition_definitions())
