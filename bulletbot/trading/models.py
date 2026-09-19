from __future__ import annotations

from dataclasses import dataclass

from bulletbot.domain.models import MarketObservation


@dataclass(frozen=True)
class TradingRule:
    name: str
    normalized_name: str
    target_price: int
    requested_quantity: int
    expected_sell_price: int
    minimum_profit_per_unit: int
    auto_sell_enabled: bool
    sell_trigger_price: int
    sell_units_per_slot: int
    allow_sell_undercut: bool


@dataclass(frozen=True)
class TradingCandidate:
    rule: TradingRule
    observation: MarketObservation


@dataclass(frozen=True)
class TradeEvaluation:
    candidate: TradingCandidate
    lowest_price: int
    average_unit_price: int
    quantity: int
    expected_sell_price: int | None
    net_profit_per_unit: int | None

    @property
    def buy_discount(self) -> int:
        return self.candidate.rule.target_price - self.average_unit_price

    @property
    def is_eligible(self) -> bool:
        return self.average_unit_price <= self.candidate.rule.target_price

