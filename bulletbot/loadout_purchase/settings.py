from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .warehouse_sell_session import ListingPriceMode, WarehouseSellRequest


CURRENT_SETTINGS_VERSION = 10
MEMORY_MANAGER_OPT_IN_VERSION = 10
PURCHASE_SCHEME_INDICES = (1, 2, 3)
FAST_LOADOUT_SCHEME_PAIRS = ((1, 2), (2, 3), (1, 3))
PARTIAL_TRANSFER_SCHEME_INDEX = 4
BLANK_SCHEME_INDEX = 5


@dataclass(frozen=True)
class LoadoutSchemeRule:
    scheme_index: int
    enabled: bool = False
    slot_count: int = 74
    units_per_slot: int = 60
    max_unit_price: int = 0
    warehouse_sell_enabled: bool = False
    ammunition_name: str = ""
    sell_trigger_price: int = 0
    sell_price_mode: ListingPriceMode = ListingPriceMode.CURRENT_LOWEST
    fixed_sell_price: int = 0

    @property
    def total_quantity(self) -> int:
        return self.slot_count * self.units_per_slot

    @property
    def max_total_price(self) -> int:
        return self.total_quantity * self.max_unit_price

    def validate(self) -> None:
        if self.scheme_index not in PURCHASE_SCHEME_INDICES:
            raise ValueError("配装买入方案序号必须位于 1-3")
        if self.slot_count <= 0:
            raise ValueError(f"配装方案 {self.scheme_index} 的子弹格数必须大于零")
        if self.units_per_slot <= 0:
            raise ValueError(f"配装方案 {self.scheme_index} 的每格数量必须大于零")
        if self.enabled and self.max_unit_price <= 0:
            raise ValueError(f"配装方案 {self.scheme_index} 的最高单价必须大于零")
        try:
            ListingPriceMode(self.sell_price_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"配装方案 {self.scheme_index} 的挂牌方式无效"
            ) from exc
        if self.enabled and self.warehouse_sell_enabled:
            try:
                self.warehouse_sell_request()
            except ValueError as exc:
                raise ValueError(
                    f"配装方案 {self.scheme_index} 的交易行直售配置无效：{exc}"
                ) from exc

    def warehouse_sell_request(self) -> WarehouseSellRequest:
        if not self.warehouse_sell_enabled:
            raise ValueError("当前方案没有启用交易行直售")
        return WarehouseSellRequest(
            ammunition_name=self.ammunition_name,
            sell_trigger_price=self.sell_trigger_price,
            price_mode=ListingPriceMode(self.sell_price_mode),
            fixed_price=(
                self.fixed_sell_price
                if ListingPriceMode(self.sell_price_mode) is ListingPriceMode.FIXED
                else None
            ),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, scheme_index: int) -> "LoadoutSchemeRule":
        try:
            sell_price_mode = ListingPriceMode(
                value.get("sell_price_mode", ListingPriceMode.CURRENT_LOWEST)
            )
        except (TypeError, ValueError):
            sell_price_mode = ListingPriceMode.CURRENT_LOWEST
        return cls(
            scheme_index=scheme_index,
            enabled=bool(value.get("enabled", False)),
            slot_count=max(1, int(value.get("slot_count", 74))),
            units_per_slot=max(1, int(value.get("units_per_slot", 60))),
            max_unit_price=max(0, int(value.get("max_unit_price", 0))),
            warehouse_sell_enabled=bool(
                value.get("warehouse_sell_enabled", False)
            ),
            ammunition_name=str(value.get("ammunition_name", "")).strip(),
            sell_trigger_price=max(0, int(value.get("sell_trigger_price", 0))),
            sell_price_mode=sell_price_mode,
            fixed_sell_price=max(0, int(value.get("fixed_sell_price", 0))),
        )


def _default_rules() -> tuple[LoadoutSchemeRule, ...]:
    return tuple(LoadoutSchemeRule(index) for index in PURCHASE_SCHEME_INDICES)


@dataclass(frozen=True)
class FastLoadoutRule:
    scheme_pair: tuple[int, int] = FAST_LOADOUT_SCHEME_PAIRS[0]
    slot_count: int = 74
    units_per_slot: int = 60
    max_unit_price: int = 0

    @property
    def total_quantity(self) -> int:
        return self.slot_count * self.units_per_slot

    @property
    def max_total_price(self) -> int:
        return self.total_quantity * self.max_unit_price

    def validate(self, *, require_price: bool = False) -> None:
        if self.scheme_pair not in FAST_LOADOUT_SCHEME_PAIRS:
            raise ValueError("极速配装方案组合必须是 1+2、2+3 或 1+3")
        if self.slot_count <= 0:
            raise ValueError("极速配装的子弹格数必须大于零")
        if self.units_per_slot <= 0:
            raise ValueError("极速配装的每格数量必须大于零")
        if self.max_unit_price < 0 or (require_price and self.max_unit_price <= 0):
            raise ValueError("极速配装的最高单价必须大于零")

    def purchase_rules(
        self,
        base_rules: tuple[LoadoutSchemeRule, ...] | None = None,
    ) -> tuple[LoadoutSchemeRule, ...]:
        self.validate(require_price=True)
        source_rules = base_rules or _default_rules()
        if (
            tuple(rule.scheme_index for rule in source_rules)
            != PURCHASE_SCHEME_INDICES
        ):
            raise ValueError("极速配装的购买后处理必须按顺序包含方案 1-3")
        return tuple(
            replace(
                rule,
                enabled=rule.scheme_index in self.scheme_pair,
                slot_count=self.slot_count,
                units_per_slot=self.units_per_slot,
                max_unit_price=self.max_unit_price,
            )
            for rule in source_rules
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FastLoadoutRule":
        raw_pair = value.get("scheme_pair", FAST_LOADOUT_SCHEME_PAIRS[0])
        try:
            pair = tuple(int(index) for index in raw_pair)
        except (TypeError, ValueError):
            pair = FAST_LOADOUT_SCHEME_PAIRS[0]
        if pair not in FAST_LOADOUT_SCHEME_PAIRS:
            pair = FAST_LOADOUT_SCHEME_PAIRS[0]
        return cls(
            scheme_pair=pair,
            slot_count=max(1, int(value.get("slot_count", 74))),
            units_per_slot=max(1, int(value.get("units_per_slot", 60))),
            max_unit_price=max(0, int(value.get("max_unit_price", 0))),
        )


@dataclass(frozen=True)
class LoadoutPurchaseSettings:
    settings_version: int = CURRENT_SETTINGS_VERSION
    schemes: tuple[LoadoutSchemeRule, ...] = field(default_factory=_default_rules)
    scan_interval_seconds: float = 1.0
    game_exit_timeout_seconds: int = 240
    launcher_window_timeout_seconds: int = 240
    game_window_timeout_seconds: int = 240
    first_missing_resource_timeout_seconds: int = 180
    warfare_switch_timeout_seconds: int = 120
    map_fast_path_timeout_seconds: int = 8
    map_page_timeout_seconds: int = 60
    purchase_result_timeout_seconds: int = 12
    blank_scheme_index: int = BLANK_SCHEME_INDEX
    wait_when_slots_full: bool = True
    sell_wait_timeout_minutes: int = 10
    sell_slot_poll_seconds: int = 5
    fast_loadout_click_settle_ms: int = 80
    fast_loadout_switch_delay_ms: int = 100
    scheduled_game_restart_enabled: bool = True
    scheduled_game_restart_interval_minutes: int = 20
    memory_cleanup_enabled: bool = False
    memory_cleanup_interval_minutes: int = 10
    fast_rule: FastLoadoutRule = field(default_factory=FastLoadoutRule)

    @property
    def enabled_schemes(self) -> tuple[LoadoutSchemeRule, ...]:
        return tuple(rule for rule in self.schemes if rule.enabled)

    def validate(self, *, require_enabled: bool = False) -> None:
        indices = tuple(rule.scheme_index for rule in self.schemes)
        if indices != PURCHASE_SCHEME_INDICES:
            raise ValueError("配装买入配置必须按顺序包含方案 1-3")
        for rule in self.schemes:
            rule.validate()
        self.fast_rule.validate()
        if require_enabled and not self.enabled_schemes:
            raise ValueError("请至少启用一个配装买入方案")
        if self.blank_scheme_index != BLANK_SCHEME_INDEX:
            raise ValueError("配装方案 5 必须保留为空白方案")
        if not 0.4 <= self.scan_interval_seconds <= 30.0:
            raise ValueError("方案切换确认超时必须位于 0.4-30 秒")
        if not 60 <= self.game_exit_timeout_seconds <= 600:
            raise ValueError("游戏完全退出等待必须位于 60-600 秒")
        if not 120 <= self.launcher_window_timeout_seconds <= 600:
            raise ValueError("启动器窗口等待必须位于 120-600 秒")
        if not 180 <= self.game_window_timeout_seconds <= 600:
            raise ValueError("新游戏窗口等待必须位于 180-600 秒")
        if not 180 <= self.first_missing_resource_timeout_seconds <= 300:
            raise ValueError("第一次资源缺失提醒等待必须位于 180-300 秒")
        if not 120 <= self.warfare_switch_timeout_seconds <= 300:
            raise ValueError("切换全面战场等待必须位于 120-300 秒")
        if not 1 <= self.map_fast_path_timeout_seconds <= 60:
            raise ValueError("地图页面快速路径等待必须位于 1-60 秒")
        if not 60 <= self.map_page_timeout_seconds <= 120:
            raise ValueError("地图页面恢复等待必须位于 60-120 秒")
        if not 12 <= self.purchase_result_timeout_seconds <= 120:
            raise ValueError("购买结果确认等待必须位于 12-120 秒")
        if not 1 <= self.sell_wait_timeout_minutes <= 1440:
            raise ValueError("挂单最长等待时间必须位于 1-1440 分钟")
        if not 5 <= self.sell_slot_poll_seconds <= 120:
            raise ValueError("售位检测间隔必须位于 5-120 秒")
        if not 80 <= self.fast_loadout_click_settle_ms <= 1000:
            raise ValueError("极速方案点击等待必须位于 80-1000 毫秒")
        if not 0 <= self.fast_loadout_switch_delay_ms <= 1000:
            raise ValueError("极速高价切换等待必须位于 0-1000 毫秒")
        if not 1 <= self.scheduled_game_restart_interval_minutes <= 1440:
            raise ValueError("定时重启间隔必须位于 1-1440 分钟")
        if not 1 <= self.memory_cleanup_interval_minutes <= 1440:
            raise ValueError("内存清理间隔必须位于 1-1440 分钟")

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings_version": CURRENT_SETTINGS_VERSION,
            "schemes": [asdict(rule) for rule in self.schemes],
            "scan_interval_seconds": self.scan_interval_seconds,
            "game_exit_timeout_seconds": self.game_exit_timeout_seconds,
            "launcher_window_timeout_seconds": self.launcher_window_timeout_seconds,
            "game_window_timeout_seconds": self.game_window_timeout_seconds,
            "first_missing_resource_timeout_seconds": self.first_missing_resource_timeout_seconds,
            "warfare_switch_timeout_seconds": self.warfare_switch_timeout_seconds,
            "map_fast_path_timeout_seconds": self.map_fast_path_timeout_seconds,
            "map_page_timeout_seconds": self.map_page_timeout_seconds,
            "purchase_result_timeout_seconds": self.purchase_result_timeout_seconds,
            "blank_scheme_index": self.blank_scheme_index,
            "wait_when_slots_full": self.wait_when_slots_full,
            "sell_wait_timeout_minutes": self.sell_wait_timeout_minutes,
            "sell_slot_poll_seconds": self.sell_slot_poll_seconds,
            "fast_loadout_click_settle_ms": self.fast_loadout_click_settle_ms,
            "fast_loadout_switch_delay_ms": self.fast_loadout_switch_delay_ms,
            "scheduled_game_restart_enabled": self.scheduled_game_restart_enabled,
            "scheduled_game_restart_interval_minutes": (
                self.scheduled_game_restart_interval_minutes
            ),
            "memory_cleanup_enabled": self.memory_cleanup_enabled,
            "memory_cleanup_interval_minutes": self.memory_cleanup_interval_minutes,
            "fast_rule": asdict(self.fast_rule),
        }

    def for_fast_mode(self) -> "LoadoutPurchaseSettings":
        self.fast_rule.validate(require_price=True)
        effective = replace(
            self,
            schemes=self.fast_rule.purchase_rules(self.schemes),
        )
        effective.validate(require_enabled=True)
        return effective

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LoadoutPurchaseSettings":
        try:
            stored_version = int(value.get("settings_version", 0))
        except (TypeError, ValueError):
            stored_version = 0
        raw_schemes = value.get("schemes", [])
        by_index: dict[int, dict[str, Any]] = {}
        if isinstance(raw_schemes, list):
            for raw in raw_schemes:
                if not isinstance(raw, dict):
                    continue
                try:
                    index = int(raw.get("scheme_index", 0))
                except (TypeError, ValueError):
                    continue
                if index in PURCHASE_SCHEME_INDICES:
                    by_index[index] = raw
        settings = cls(
            schemes=tuple(
                LoadoutSchemeRule.from_dict(by_index.get(index, {}), scheme_index=index)
                for index in PURCHASE_SCHEME_INDICES
            ),
            scan_interval_seconds=float(value.get("scan_interval_seconds", 1.0)),
            game_exit_timeout_seconds=int(
                value.get("game_exit_timeout_seconds", 240)
            ),
            launcher_window_timeout_seconds=int(
                value.get("launcher_window_timeout_seconds", 240)
            ),
            game_window_timeout_seconds=int(
                value.get("game_window_timeout_seconds", 240)
            ),
            first_missing_resource_timeout_seconds=int(
                value.get("first_missing_resource_timeout_seconds", 180)
            ),
            warfare_switch_timeout_seconds=int(
                value.get("warfare_switch_timeout_seconds", 120)
            ),
            map_fast_path_timeout_seconds=int(
                value.get("map_fast_path_timeout_seconds", 8)
            ),
            map_page_timeout_seconds=int(value.get("map_page_timeout_seconds", 60)),
            purchase_result_timeout_seconds=int(
                value.get("purchase_result_timeout_seconds", 12)
            ),
            blank_scheme_index=int(value.get("blank_scheme_index", BLANK_SCHEME_INDEX)),
            wait_when_slots_full=bool(value.get("wait_when_slots_full", True)),
            sell_wait_timeout_minutes=int(
                value.get("sell_wait_timeout_minutes", 10)
            ),
            sell_slot_poll_seconds=int(value.get("sell_slot_poll_seconds", 5)),
            fast_loadout_click_settle_ms=max(
                80,
                min(1000, int(value.get("fast_loadout_click_settle_ms", 80))),
            ),
            fast_loadout_switch_delay_ms=int(
                value.get("fast_loadout_switch_delay_ms", 100)
            ),
            scheduled_game_restart_enabled=bool(
                value.get("scheduled_game_restart_enabled", True)
            ),
            scheduled_game_restart_interval_minutes=int(
                value.get("scheduled_game_restart_interval_minutes", 20)
            ),
            memory_cleanup_enabled=(
                bool(value.get("memory_cleanup_enabled", False))
                if stored_version >= MEMORY_MANAGER_OPT_IN_VERSION
                else False
            ),
            memory_cleanup_interval_minutes=int(
                value.get("memory_cleanup_interval_minutes", 10)
            ),
            fast_rule=FastLoadoutRule.from_dict(
                value.get("fast_rule", {})
                if isinstance(value.get("fast_rule", {}), dict)
                else {}
            ),
        )
        settings.validate()
        return settings


class LoadoutPurchaseSettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> LoadoutPurchaseSettings:
        if not self.path.exists():
            return LoadoutPurchaseSettings()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return LoadoutPurchaseSettings()
        if not isinstance(value, dict):
            return LoadoutPurchaseSettings()
        try:
            return LoadoutPurchaseSettings.from_dict(value)
        except (TypeError, ValueError):
            return LoadoutPurchaseSettings()

    def save(self, settings: LoadoutPurchaseSettings) -> None:
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
