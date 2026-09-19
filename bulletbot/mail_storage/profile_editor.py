from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from bulletbot.platform.game_window_session import SharedGameWindowSession

from .models import WindowFingerprint, WindowInfo
from .pak_transaction import PakTransaction
from .settings import MAX_PAK_FILES, AppSettings, SettingsStore
from .windows import (
    WindowBindingManager,
    enumerate_windows,
    fingerprint_title_matches,
    is_delta_force_game_window,
    is_delta_force_launcher_main_window,
    is_delta_force_launcher_settings_window,
    is_delta_force_launcher_window,
    is_standard_delta_force_game_window,
)


WindowEnumerator = Callable[[], list[WindowInfo]]


@dataclass(frozen=True)
class ProfileBindingState:
    target: str
    bound: WindowInfo | None
    configured: bool
    ambiguous_count: int


class MailProfileEditor:
    """Edit one Mail profile while preserving its window matching semantics."""

    TARGETS = ("game", "launcher", "launcher_settings")

    def __init__(
        self,
        path: Path,
        settings: AppSettings,
        *,
        imported_from: Path | None = None,
        window_enumerator: WindowEnumerator = enumerate_windows,
        game_window_session: SharedGameWindowSession | None = None,
    ) -> None:
        self.path = path
        self.settings = settings
        self.imported_from = imported_from
        self._window_enumerator = window_enumerator
        self._game_window_session = game_window_session
        self._windows: tuple[WindowInfo, ...] = ()
        if game_window_session is None:
            self.game = WindowBindingManager(
                settings.game_window,
                candidate_filter=self._is_game_candidate,
            )
        else:
            game_window_session.seed_fingerprint(settings.game_window)
            self.game = game_window_session
        self.launcher = WindowBindingManager(
            settings.launcher_window,
            candidate_filter=self._is_launcher_main_candidate,
        )
        self.launcher_settings = WindowBindingManager(
            settings.launcher_settings_window,
            candidate_filter=self._is_launcher_settings_candidate,
        )

    @property
    def windows(self) -> tuple[WindowInfo, ...]:
        return self._windows

    def candidate_windows(self, target: str) -> tuple[WindowInfo, ...]:
        self._manager(target)
        # Automatic matching stays target-specific, but manual selection must
        # expose every visible top-level window returned by Windows.
        return self._windows

    def refresh(
        self,
        windows: list[WindowInfo] | tuple[WindowInfo, ...] | None = None,
    ) -> dict[str, ProfileBindingState]:
        candidates = list(windows) if windows is not None else self._window_enumerator()
        self._windows = tuple(candidates)
        self.game.refresh(candidates)
        self.launcher.refresh(candidates)
        self.launcher_settings.refresh(candidates)
        return self.binding_states()

    def binding_states(self) -> dict[str, ProfileBindingState]:
        return {
            target: ProfileBindingState(
                target=target,
                bound=manager.bound,
                configured=manager.fingerprint.configured(),
                ambiguous_count=len(manager.ambiguous),
            )
            for target, manager in self._managers().items()
        }

    def bind(self, target: str, hwnd: int) -> ProfileBindingState:
        manager = self._manager(target)
        window = next((item for item in self._windows if item.hwnd == hwnd), None)
        if window is None:
            raise ValueError("所选窗口已经关闭，请刷新窗口列表")
        for other_target, other_manager in self._managers().items():
            if other_target == target or other_manager.bound is None:
                continue
            if other_manager.bound.hwnd == window.hwnd:
                labels = {
                    "game": "游戏窗口",
                    "launcher": "启动器主窗口",
                    "launcher_settings": "启动器设置窗口",
                }
                raise ValueError(
                    f"所选窗口已经绑定为{labels[other_target]}，"
                    "三个目标窗口不能使用同一个窗口"
                )

        fingerprint = manager.bind(window)
        if target == "launcher_settings":
            fingerprint.minimum_client_size = (
                min(window.client_rect.width, 600),
                min(window.client_rect.height, 400),
            )
        setattr(self.settings, self._settings_attribute(target), fingerprint)
        self.save()
        return self.binding_states()[target]

    def update_environment(
        self,
        *,
        pak_source: str,
        pak_stage: str,
        allow_input: bool,
        allow_file_operations: bool,
        mail_scheme_index: int | None = None,
        pak_sources: Sequence[str] | None = None,
    ) -> None:
        self.settings.game_window = self.game.fingerprint
        self.settings.launcher_window = self.launcher.fingerprint
        self.settings.launcher_settings_window = self.launcher_settings.fingerprint
        if pak_sources is None:
            self.settings.pak_source = pak_source.strip()
            self.settings.pak_sources = []
        else:
            cleaned_sources = [source.strip() for source in pak_sources if source.strip()]
            if len(cleaned_sources) > MAX_PAK_FILES:
                raise ValueError(f"最多只能配置 {MAX_PAK_FILES} 个 PAK 文件")
            if len({source.casefold() for source in cleaned_sources}) != len(
                cleaned_sources
            ):
                raise ValueError("PAK 文件路径不能重复")
            self.settings.pak_sources = cleaned_sources
            self.settings.pak_source = cleaned_sources[0] if cleaned_sources else ""
        self.settings.pak_stage = pak_stage.strip()
        self.settings.allow_input = bool(allow_input)
        self.settings.allow_file_operations = bool(allow_file_operations)
        if mail_scheme_index is not None:
            if mail_scheme_index not in {1, 2, 3}:
                raise ValueError("卡邮件配装方案必须位于 1-3")
            self.settings.mail_scheme_index = mail_scheme_index

    def save(self) -> None:
        self.update_environment(
            pak_source=self.settings.pak_source,
            pak_stage=self.settings.pak_stage,
            allow_input=self.settings.allow_input,
            allow_file_operations=self.settings.allow_file_operations,
            pak_sources=self.settings.effective_pak_sources(),
        )
        SettingsStore(self.path).save(self.settings)
        self.imported_from = None

    def pak_status(self) -> str:
        sources = self.settings.effective_pak_paths()
        staged_paths = self.settings.effective_staged_pak_paths()
        if (
            not sources
            or len(sources) > MAX_PAK_FILES
            or len(sources) != len(staged_paths)
            or any(source == staged for source, staged in zip(sources, staged_paths))
        ):
            return "invalid"
        return PakTransaction(sources, staged_paths, enabled=False).inspect()

    def _managers(self) -> dict[str, WindowBindingManager]:
        return {
            "game": self.game,
            "launcher": self.launcher,
            "launcher_settings": self.launcher_settings,
        }

    def _manager(self, target: str) -> WindowBindingManager:
        try:
            return self._managers()[target]
        except KeyError as exc:
            raise ValueError(f"未知窗口绑定类型：{target}") from exc

    @staticmethod
    def _settings_attribute(target: str) -> str:
        return {
            "game": "game_window",
            "launcher": "launcher_window",
            "launcher_settings": "launcher_settings_window",
        }[target]

    def _is_launcher_settings_candidate(self, window: WindowInfo) -> bool:
        launcher = self.launcher.bound
        if launcher is not None and window.hwnd == launcher.hwnd:
            return False
        expected_title = self.launcher_settings.fingerprint.title.strip()
        if self.launcher_settings.fingerprint.configured():
            return bool(
                (
                    not expected_title
                    or fingerprint_title_matches(
                        self.launcher_settings.fingerprint,
                        window,
                    )
                )
                and self.launcher_settings.fingerprint.score(window) >= 50
            )
        return is_delta_force_launcher_settings_window(window)

    def _is_game_candidate(self, window: WindowInfo) -> bool:
        return is_standard_delta_force_game_window(
            window
        ) or self._matches_saved_title(
            self.game.fingerprint,
            window,
        )

    def _is_launcher_main_candidate(self, window: WindowInfo) -> bool:
        if (
            is_delta_force_launcher_settings_window(window)
            and "设置" not in self.launcher.fingerprint.title
        ):
            return False
        return is_delta_force_launcher_window(window) and (
            is_delta_force_launcher_main_window(window)
            or self._matches_saved_title(self.launcher.fingerprint, window)
        )

    @staticmethod
    def _matches_saved_window(
        fingerprint: WindowFingerprint,
        window: WindowInfo,
    ) -> bool:
        return fingerprint.configured() and fingerprint.score(window) >= 50

    @classmethod
    def _matches_saved_title(
        cls,
        fingerprint: WindowFingerprint,
        window: WindowInfo,
    ) -> bool:
        return cls._matches_saved_window(
            fingerprint,
            window,
        ) and fingerprint_title_matches(fingerprint, window)
