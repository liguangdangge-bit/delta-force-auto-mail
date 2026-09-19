from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bulletbot.platform.game_process_control import (
    DELTA_FORCE_GAME_PROCESS,
    DELTA_FORCE_LAUNCHER_PROCESS,
)

from .models import WindowFingerprint
from .paths import user_data_dir


DEFAULT_PAK = Path(
    r"C:\Program Files (x86)\Delta Force\Delta Force\DeltaForce\PackContent\Paks\pak-0-2-pakchunk90-WindowsClient.pak"
)
DEFAULT_OPTIONAL_PAK = DEFAULT_PAK.with_name(
    "pak-0-2-pakchunk90optional-WindowsClient.pak"
)
DEFAULT_STAGE = Path(DEFAULT_PAK.anchor) / "StandaloneMailBot_PakStage" / DEFAULT_PAK.name
MAX_PAK_FILES = 2
CURRENT_SETTINGS_VERSION = 11


def default_game_window() -> WindowFingerprint:
    return WindowFingerprint(
        process_name=DELTA_FORCE_GAME_PROCESS,
        title="三角洲行动",
        class_name="UnrealWindow",
        minimum_client_size=(960, 540),
    )


def default_launcher_window() -> WindowFingerprint:
    return WindowFingerprint(
        process_name=DELTA_FORCE_LAUNCHER_PROCESS,
        title="三角洲行动",
        class_name="TWINCONTROL",
        minimum_client_size=(960, 540),
    )


def default_launcher_settings_window() -> WindowFingerprint:
    return WindowFingerprint(
        process_name=DELTA_FORCE_LAUNCHER_PROCESS,
        title="设置",
        class_name="TWINCONTROL",
        minimum_client_size=(600, 400),
    )


def _fingerprint_or_default(
    value: dict[str, Any] | None,
    default_factory: Callable[[], WindowFingerprint],
) -> WindowFingerprint:
    fingerprint = WindowFingerprint.from_dict(value)
    return fingerprint if fingerprint.configured() else default_factory()


DEFAULT_POINTS: dict[str, tuple[float, float]] = {
    "map_card": (0.856, 0.825),
    "map_expand": (0.150, 0.785),
    "longbow_node": (0.252, 0.340),
    "zero_dam_node": (0.429, 0.248),
    "start_action": (0.860, 0.595),
    "open_loadout": (0.808, 0.895),
    "confirm_loadout": (0.876, 0.881),
    "depart": (0.905, 0.895),
    "entry_warning_continue": (0.500, 0.650),
    "launcher_account_button": (0.962, 0.156),
    "launcher_account_resource_entry": (0.902, 0.222),
    "launcher_delete_longbow": (0.688, 0.733),
    "launcher_confirm_delete": (0.493, 0.717),
    "launcher_close_settings": (0.706, 0.276),
    "launcher_start_game": (0.848, 0.873),
    "dialog_confirm": (0.505, 0.625),
    "initial_firestorm_mode": (0.180, 0.895),
    "generic_confirm": (0.505, 0.625),
    "mail_scheme_card": (0.103, 0.397),
    "loadout_scheme_1": (0.103, 0.397),
    "loadout_scheme_2": (0.103, 0.463),
    "loadout_scheme_3": (0.103, 0.528),
    "loadout_scheme_4": (0.103, 0.593),
    "loadout_scheme_5": (0.103, 0.657),
    "current_loadout": (0.103, 0.283),
    "balance_hover": (0.855, 0.050),
    "use_scheme": (0.876, 0.807),
    "save_scheme": (0.876, 0.876),
    "save_scheme_slot_4": (0.647, 0.473),
    "save_scheme_submit": (0.501, 0.698),
    "save_scheme_overwrite_confirm": (0.500, 0.650),
    "confirm_equip": (0.876, 0.881),
    "warfare_mode": (0.140, 0.377),
    "firestorm_mode": (0.140, 0.255),
    "cancel_reconnect": (0.417, 0.625),
    "abandon_match": (0.417, 0.625),
    "mail_icon": (0.912, 0.050),
    "mail_partial_claim": (0.909, 0.831),
    "mail_attachment_first": (0.331, 0.864),
    "mail_attachment_second": (0.378, 0.864),
    # When the first visible attachments are ammunition, the gear attachments
    # are exposed at the far right after scrolling the attachment strip down.
    "mail_attachment_last": (0.722, 0.864),
    "mail_attachment_last_second": (0.768, 0.864),
    "mail_attachment_scroll": (0.600, 0.864),
    "mail_claim": (0.876, 0.877),
    "game_home_tab": (0.100, 0.050),
    "warehouse_tab": (0.168, 0.050),
    "market_tab": (0.371, 0.050),
    "market_buy_tab": (0.105, 0.102),
    "market_sell_tab": (0.168, 0.102),
}

LEGACY_POINT_MIGRATIONS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "longbow_node": ((0.475, 0.510), DEFAULT_POINTS["longbow_node"]),
    "map_expand": ((0.505, 0.710), DEFAULT_POINTS["map_expand"]),
    "zero_dam_node": ((0.466, 0.500), DEFAULT_POINTS["zero_dam_node"]),
    "initial_firestorm_mode": (
        (0.270, 0.400),
        DEFAULT_POINTS["initial_firestorm_mode"],
    ),
    "warfare_mode": ((0.140, 0.435), DEFAULT_POINTS["warfare_mode"]),
}


def _points_equal(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return all(abs(actual - expected) < 1e-6 for actual, expected in zip(left, right))


@dataclass
class AppSettings:
    settings_version: int = CURRENT_SETTINGS_VERSION
    game_window: WindowFingerprint = field(default_factory=default_game_window)
    launcher_window: WindowFingerprint = field(default_factory=default_launcher_window)
    launcher_settings_window: WindowFingerprint = field(
        default_factory=default_launcher_settings_window
    )
    pak_source: str = str(DEFAULT_PAK)
    pak_sources: list[str] = field(default_factory=list)
    pak_stage: str = str(DEFAULT_STAGE)
    observation_interval_seconds: float = 0.5
    action_settle_seconds: float = 1.0
    allow_input: bool = True
    allow_file_operations: bool = True
    mail_scheme_index: int = 1
    action_points: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_POINTS)
    )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        sources = list(self.effective_pak_sources())
        value["pak_source"] = sources[0] if sources else ""
        value["pak_sources"] = sources
        value["game_window"] = self.game_window.to_dict()
        value["launcher_window"] = self.launcher_window.to_dict()
        value["launcher_settings_window"] = self.launcher_settings_window.to_dict()
        value["action_points"] = {
            name: list(point) for name, point in self.action_points.items()
        }
        return value

    def effective_pak_sources(self) -> tuple[str, ...]:
        raw_sources = self.pak_sources or [self.pak_source]
        return tuple(
            str(source).strip()
            for source in raw_sources
            if str(source).strip()
        )

    def effective_pak_paths(self) -> tuple[Path, ...]:
        return tuple(Path(source).expanduser() for source in self.effective_pak_sources())

    def effective_staged_pak_paths(self) -> tuple[Path, ...]:
        if not self.pak_stage.strip():
            return ()
        sources = self.effective_pak_paths()
        configured_stage = Path(self.pak_stage).expanduser()
        if len(sources) == 1:
            return (configured_stage,)
        stage_directory = configured_stage.parent
        return tuple(stage_directory / source.name for source in sources)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AppSettings":
        source_version = int(value.get("settings_version", 1))
        legacy_pak_source = str(value.get("pak_source", DEFAULT_PAK)).strip()
        raw_pak_sources = value.get("pak_sources", [])
        if isinstance(raw_pak_sources, str):
            raw_pak_sources = [raw_pak_sources]
        pak_sources = (
            [str(source).strip() for source in raw_pak_sources if str(source).strip()]
            if isinstance(raw_pak_sources, (list, tuple))
            else []
        )
        settings = cls(
            settings_version=CURRENT_SETTINGS_VERSION,
            game_window=_fingerprint_or_default(
                value.get("game_window"),
                default_game_window,
            ),
            launcher_window=_fingerprint_or_default(
                value.get("launcher_window"),
                default_launcher_window,
            ),
            launcher_settings_window=_fingerprint_or_default(
                value.get("launcher_settings_window"),
                default_launcher_settings_window,
            ),
            pak_source=pak_sources[0] if pak_sources else legacy_pak_source,
            pak_sources=pak_sources,
            pak_stage=str(value.get("pak_stage", DEFAULT_STAGE)),
            observation_interval_seconds=float(value.get("observation_interval_seconds", 0.5)),
            action_settle_seconds=float(value.get("action_settle_seconds", 1.0)),
            allow_input=(
                True
                if source_version < CURRENT_SETTINGS_VERSION
                else bool(value.get("allow_input", True))
            ),
            allow_file_operations=(
                True
                if source_version < CURRENT_SETTINGS_VERSION
                else bool(value.get("allow_file_operations", True))
            ),
            mail_scheme_index=min(3, max(1, int(value.get("mail_scheme_index", 1)))),
        )
        points = dict(DEFAULT_POINTS)
        for name, point in value.get("action_points", {}).items():
            if isinstance(point, (list, tuple)) and len(point) == 2:
                point_name = str(name)
                saved_point = (float(point[0]), float(point[1]))
                migration = LEGACY_POINT_MIGRATIONS.get(point_name)
                if source_version < CURRENT_SETTINGS_VERSION and migration is not None:
                    old_default, new_default = migration
                    if _points_equal(saved_point, old_default):
                        saved_point = new_default
                points[point_name] = saved_point
        settings.action_points = points
        return settings


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or user_data_dir() / "settings.json"

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return AppSettings.from_dict(json.load(handle))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(settings.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

