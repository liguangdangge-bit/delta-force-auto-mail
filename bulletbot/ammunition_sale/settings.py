from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from bulletbot.loadout_purchase.ammunition_catalog import (
    canonical_ammunition_name,
    supported_ammunition_names,
)
from bulletbot.loadout_purchase.warehouse_sell_session import (
    ListingPriceMode,
    WarehouseSellRequest,
)


CURRENT_SETTINGS_VERSION = 2


class AmmunitionSaleSource(StrEnum):
    WAREHOUSE_ONLY = "warehouse_only"
    WAREHOUSE_AND_MAIL = "warehouse_and_mail"


class AmmunitionSaleCondition(StrEnum):
    DIRECT = "direct"
    MINIMUM_PRICE = "minimum_price"


@dataclass(frozen=True)
class AmmunitionSaleRule:
    enabled: bool = False
    ammunition_name: str = ""
    source: AmmunitionSaleSource = AmmunitionSaleSource.WAREHOUSE_ONLY
    condition: AmmunitionSaleCondition = AmmunitionSaleCondition.DIRECT
    trigger_price: int = 0
    price_mode: ListingPriceMode = ListingPriceMode.CURRENT_LOWEST
    fixed_price: int = 0
    mail_claim_limit: int = 3

    def validate(self) -> None:
        try:
            source = AmmunitionSaleSource(self.source)
            condition = AmmunitionSaleCondition(self.condition)
            price_mode = ListingPriceMode(self.price_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("领邮件自动卖规则包含未知选项") from exc
        if not self.enabled:
            return
        name = canonical_ammunition_name(self.ammunition_name)
        if not name:
            raise ValueError("已启用的领邮件自动卖规则必须选择子弹名称")
        if name not in supported_ammunition_names():
            raise ValueError(f"不支持的子弹名称：{self.ammunition_name}")
        if condition is AmmunitionSaleCondition.MINIMUM_PRICE:
            if self.trigger_price <= 0:
                raise ValueError(f"{name} 的最低触发价必须大于 0")
        elif self.trigger_price < 0:
            raise ValueError(f"{name} 的最低触发价不能小于 0")
        if price_mode is ListingPriceMode.FIXED:
            if self.fixed_price <= 0:
                raise ValueError(f"{name} 选择固定价时必须填写有效价格")
            if (
                condition is AmmunitionSaleCondition.MINIMUM_PRICE
                and self.fixed_price < self.trigger_price
            ):
                raise ValueError(f"{name} 的固定挂牌价不能低于最低触发价")
        elif self.fixed_price < 0:
            raise ValueError(f"{name} 的固定挂牌价不能小于 0")
        if not 1 <= self.mail_claim_limit <= 200:
            raise ValueError(f"{name} 的每轮邮件领取数必须位于 1-200 封")
        _ = source

    def warehouse_sell_request(self) -> WarehouseSellRequest:
        self.validate()
        if not self.enabled:
            raise ValueError("当前领邮件自动卖规则未启用")
        condition = AmmunitionSaleCondition(self.condition)
        price_mode = ListingPriceMode(self.price_mode)
        return WarehouseSellRequest(
            ammunition_name=canonical_ammunition_name(self.ammunition_name),
            sell_trigger_price=(
                self.trigger_price
                if condition is AmmunitionSaleCondition.MINIMUM_PRICE
                else 1
            ),
            price_mode=price_mode,
            fixed_price=self.fixed_price if price_mode is ListingPriceMode.FIXED else None,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AmmunitionSaleRule":
        try:
            source = AmmunitionSaleSource(
                value.get("source", AmmunitionSaleSource.WAREHOUSE_ONLY)
            )
        except (TypeError, ValueError):
            source = AmmunitionSaleSource.WAREHOUSE_ONLY
        try:
            condition = AmmunitionSaleCondition(
                value.get("condition", AmmunitionSaleCondition.DIRECT)
            )
        except (TypeError, ValueError):
            condition = AmmunitionSaleCondition.DIRECT
        try:
            price_mode = ListingPriceMode(
                value.get("price_mode", ListingPriceMode.CURRENT_LOWEST)
            )
        except (TypeError, ValueError):
            price_mode = ListingPriceMode.CURRENT_LOWEST
        return cls(
            enabled=bool(value.get("enabled", False)),
            ammunition_name=canonical_ammunition_name(
                str(value.get("ammunition_name", "")).strip()
            ),
            source=source,
            condition=condition,
            trigger_price=max(0, int(value.get("trigger_price", 0))),
            price_mode=price_mode,
            fixed_price=max(0, int(value.get("fixed_price", 0))),
            mail_claim_limit=min(
                200,
                max(1, int(value.get("mail_claim_limit", 3))),
            ),
        )


def _default_rules() -> tuple[AmmunitionSaleRule, ...]:
    return (AmmunitionSaleRule(),)


@dataclass(frozen=True)
class AmmunitionSaleSettings:
    settings_version: int = CURRENT_SETTINGS_VERSION
    rules: tuple[AmmunitionSaleRule, ...] = field(default_factory=_default_rules)
    low_price_attempts: int = 3
    short_retry_seconds: float = 3.0
    cooldown_seconds: float = 180.0
    sell_slot_poll_seconds: float = 5.0

    @property
    def enabled_rules(self) -> tuple[AmmunitionSaleRule, ...]:
        return tuple(rule for rule in self.rules if rule.enabled)

    def validate(self, *, require_enabled: bool = False) -> None:
        if not self.rules:
            raise ValueError("领邮件自动卖至少需要保留一行配置")
        if len(self.rules) > len(supported_ammunition_names()):
            raise ValueError("领邮件自动卖规则数量不能超过子弹目录数量")
        for rule in self.rules:
            rule.validate()
        enabled_names = [
            canonical_ammunition_name(rule.ammunition_name)
            for rule in self.enabled_rules
        ]
        if len(enabled_names) != len(set(enabled_names)):
            raise ValueError("已启用的领邮件自动卖规则不能包含重复子弹")
        if require_enabled and not enabled_names:
            raise ValueError("请至少启用一条领邮件自动卖规则")
        if self.low_price_attempts < 1:
            raise ValueError("低价复核次数必须至少为 1")
        if not 0.5 <= self.short_retry_seconds <= 60:
            raise ValueError("低价短间隔复核必须位于 0.5-60 秒")
        if not 1 <= self.cooldown_seconds <= 86_400:
            raise ValueError("低价冷却时间必须位于 1-86400 秒")
        if not 1 <= self.sell_slot_poll_seconds <= 120:
            raise ValueError("售位检测间隔必须位于 1-120 秒")

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings_version": CURRENT_SETTINGS_VERSION,
            "rules": [asdict(rule) for rule in self.rules],
            "low_price_attempts": self.low_price_attempts,
            "short_retry_seconds": self.short_retry_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "sell_slot_poll_seconds": self.sell_slot_poll_seconds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AmmunitionSaleSettings":
        raw_rules = value.get("rules", [])
        rules = tuple(
            AmmunitionSaleRule.from_dict(raw)
            for raw in raw_rules
            if isinstance(raw, dict)
        )
        settings = cls(
            rules=rules or _default_rules(),
            low_price_attempts=max(1, int(value.get("low_price_attempts", 3))),
            short_retry_seconds=float(value.get("short_retry_seconds", 3.0)),
            cooldown_seconds=float(value.get("cooldown_seconds", 180.0)),
            sell_slot_poll_seconds=float(value.get("sell_slot_poll_seconds", 5.0)),
        )
        settings.validate()
        return settings


class AmmunitionSaleSettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> AmmunitionSaleSettings:
        if not self.path.exists():
            return AmmunitionSaleSettings()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return AmmunitionSaleSettings()
        if not isinstance(value, dict):
            return AmmunitionSaleSettings()
        try:
            return AmmunitionSaleSettings.from_dict(value)
        except (TypeError, ValueError):
            return AmmunitionSaleSettings()

    def save(self, settings: AmmunitionSaleSettings) -> None:
        settings.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(settings.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
