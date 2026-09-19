from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import threading

from bulletbot.domain.models import WindowBinding
from bulletbot.mail_storage.models import (
    Rect as MailRect,
    WindowFingerprint,
    WindowInfo,
)
from bulletbot.mail_storage.windows import (
    WindowBindingManager,
    enumerate_windows,
    is_delta_force_game_window,
)
from bulletbot.window.service import WindowService


WindowEnumerator = Callable[[], list[WindowInfo]]
BindingRefresher = Callable[[WindowBinding], WindowBinding]


class GameWindowBindingError(RuntimeError):
    """The shared game window cannot be resolved safely."""


class GameWindowAmbiguousError(GameWindowBindingError):
    def __init__(self, candidates: Sequence[WindowInfo]) -> None:
        self.candidates = tuple(candidates)
        super().__init__(
            f"找到 {len(self.candidates)} 个相似的三角洲行动窗口，"
            "为避免误绑，请在窗口列表中手动选择。"
        )


@dataclass(frozen=True)
class GameWindowSessionSnapshot:
    binding: WindowBinding | None
    window: WindowInfo | None
    fingerprint: WindowFingerprint
    candidates: tuple[WindowInfo, ...]
    ambiguous: tuple[WindowInfo, ...]
    generation: int

    @property
    def manual_binding_required(self) -> bool:
        return self.binding is None and bool(self.ambiguous)


class SharedGameWindowSession:
    """One game-window identity shared by trading, Mail, and live preview."""

    def __init__(
        self,
        fingerprint: WindowFingerprint | None = None,
        *,
        window_enumerator: WindowEnumerator = enumerate_windows,
        binding_refresher: BindingRefresher = WindowService.refresh,
    ) -> None:
        self._manager = WindowBindingManager(
            fingerprint,
            candidate_filter=self._is_game_candidate,
        )
        self._window_enumerator = window_enumerator
        self._binding_refresher = binding_refresher
        self._binding: WindowBinding | None = None
        self._windows: tuple[WindowInfo, ...] = ()
        self._candidates: tuple[WindowInfo, ...] = ()
        self._ambiguous: tuple[WindowInfo, ...] = ()
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def fingerprint(self) -> WindowFingerprint:
        return self._manager.fingerprint

    @property
    def bound(self) -> WindowInfo | None:
        return self._manager.bound

    @property
    def binding(self) -> WindowBinding | None:
        with self._lock:
            return self._binding

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def windows(self) -> tuple[WindowInfo, ...]:
        with self._lock:
            return self._windows

    @property
    def candidates(self) -> tuple[WindowInfo, ...]:
        with self._lock:
            return self._candidates

    @property
    def ambiguous(self) -> list[WindowInfo]:
        with self._lock:
            return list(self._ambiguous)

    def seed_fingerprint(self, fingerprint: WindowFingerprint) -> None:
        """Use a saved identity only while the shared session has none."""

        if not fingerprint.configured():
            return
        with self._lock:
            if self._binding is not None or self._manager.fingerprint.configured():
                return
            self._manager.set_fingerprint(fingerprint)

    def set_fingerprint(self, fingerprint: WindowFingerprint) -> None:
        with self._lock:
            self._manager.set_fingerprint(fingerprint)
            self._binding = None
            self._ambiguous = ()

    def clear(self) -> None:
        with self._lock:
            self._manager.clear()
            self._set_binding(None)
            self._ambiguous = ()

    def snapshot(self) -> GameWindowSessionSnapshot:
        with self._lock:
            return GameWindowSessionSnapshot(
                binding=self._binding,
                window=self._manager.bound,
                fingerprint=self._manager.fingerprint,
                candidates=self._candidates,
                ambiguous=self._ambiguous,
                generation=self._generation,
            )

    def list_bindings(self, *, refresh: bool = True) -> list[WindowBinding]:
        if refresh:
            self.refresh()
        with self._lock:
            return [self._to_binding(window) for window in self._candidates]

    def bind(self, window: WindowInfo) -> WindowFingerprint:
        with self._lock:
            fingerprint = self._manager.bind(window)
            self._set_binding(self._to_binding(window))
            self._ambiguous = ()
            return fingerprint

    def bind_handle(
        self,
        hwnd: int,
        windows: Sequence[WindowInfo] | None = None,
    ) -> WindowBinding:
        available = list(windows) if windows is not None else list(self._windows)
        window = next((item for item in available if item.hwnd == hwnd), None)
        if window is None:
            available = self._window_enumerator()
            window = next((item for item in available if item.hwnd == hwnd), None)
        if window is None:
            raise GameWindowBindingError("所选游戏窗口已经关闭，请刷新窗口列表。")
        self.bind(window)
        binding = self.binding
        assert binding is not None
        return binding

    def refresh_binding(self) -> WindowBinding:
        """Refresh the current HWND cheaply, enumerating only after it fails."""

        with self._lock:
            current = self._binding
        if current is None:
            self.refresh()
            return self.require_binding()
        try:
            refreshed = self._binding_refresher(current)
        except (OSError, RuntimeError):
            self.refresh()
            return self.require_binding()

        with self._lock:
            bound = self._manager.bound
            if bound is not None and bound.hwnd == refreshed.handle:
                rect = MailRect(
                    refreshed.left,
                    refreshed.top,
                    refreshed.width,
                    refreshed.height,
                )
                if bound.client_rect != rect:
                    self._manager.bind(replace(bound, client_rect=rect))
            self._set_binding(refreshed)
            return refreshed

    def refresh(
        self,
        windows: list[WindowInfo] | tuple[WindowInfo, ...] | None = None,
    ) -> WindowInfo | None:
        available = list(windows) if windows is not None else self._window_enumerator()
        automatic_candidates = tuple(
            window for window in available if self._is_game_candidate(window)
        )
        with self._lock:
            self._windows = tuple(available)
            self._candidates = tuple(available)
            resolved = self._manager.refresh(available)
            ambiguous = tuple(self._manager.ambiguous)
            if resolved is None and not ambiguous and len(automatic_candidates) == 1:
                resolved = automatic_candidates[0]
                self._manager.bind(resolved)
            elif resolved is None and not ambiguous and len(automatic_candidates) > 1:
                ambiguous = automatic_candidates

            self._ambiguous = ambiguous
            self._set_binding(
                self._to_binding(resolved) if resolved is not None else None
            )
            return resolved

    def _is_game_candidate(self, window: WindowInfo) -> bool:
        fingerprint = self._manager.fingerprint
        return is_delta_force_game_window(window) or (
            fingerprint.configured() and fingerprint.score(window) >= 50
        )

    def require_binding(self) -> WindowBinding:
        with self._lock:
            if self._binding is not None:
                return self._binding
            ambiguous = self._ambiguous
        if ambiguous:
            raise GameWindowAmbiguousError(ambiguous)
        raise GameWindowBindingError(
            "未找到可安全绑定的三角洲行动窗口，请打开游戏并刷新窗口列表。"
        )

    def _set_binding(self, binding: WindowBinding | None) -> None:
        previous = self._binding
        self._binding = binding
        if binding is None:
            return
        previous_identity = (
            None
            if previous is None
            else (
                previous.handle,
                previous.left,
                previous.top,
                previous.width,
                previous.height,
            )
        )
        current_identity = (
            binding.handle,
            binding.left,
            binding.top,
            binding.width,
            binding.height,
        )
        if previous_identity != current_identity:
            self._generation += 1

    @staticmethod
    def _to_binding(window: WindowInfo) -> WindowBinding:
        rect = window.client_rect
        return WindowBinding(
            handle=window.hwnd,
            title=window.title,
            left=rect.left,
            top=rect.top,
            width=rect.width,
            height=rect.height,
        )
