from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

import psutil

from bulletbot.platform.game_process_control import (
    DELTA_FORCE_GAME_PROCESS,
    DELTA_FORCE_LAUNCHER_PROCESS,
)

from .models import Rect, WindowFingerprint, WindowInfo


DELTA_FORCE_GAME_TITLE_KEYWORDS = ("三角洲行动",)
DELTA_FORCE_LAUNCHER_TITLE_KEYWORDS = ("三角洲行动",)
DELTA_FORCE_LAUNCHER_SETTINGS_TITLE_KEYWORDS = ("设置",)


def window_title_contains(window: WindowInfo, keywords: tuple[str, ...]) -> bool:
    title = window.title.strip().casefold()
    return bool(title) and any(keyword.casefold() in title for keyword in keywords)


def fingerprint_title_matches(
    fingerprint: WindowFingerprint,
    window: WindowInfo,
) -> bool:
    expected = fingerprint.title.strip().casefold()
    actual = window.title.strip().casefold()
    return bool(expected and actual) and (expected in actual or actual in expected)


def is_delta_force_game_process(process_name: str, process_path: str = "") -> bool:
    normalized_name = process_name.strip().casefold()
    normalized_path = process_path.replace("/", "\\").rstrip("\\")
    path_name = normalized_path.rsplit("\\", 1)[-1].casefold()
    return DELTA_FORCE_GAME_PROCESS in {normalized_name, path_name}


def is_delta_force_game_window(window: WindowInfo) -> bool:
    return is_delta_force_game_process(window.process_name, window.process_path)


def is_delta_force_splash_window(window: WindowInfo) -> bool:
    return (is_delta_force_game_window(window)
            and window.class_name.strip().casefold() == "splashscreenclass")


def is_standard_delta_force_game_window(window: WindowInfo) -> bool:
    return is_delta_force_game_window(window) and window_title_contains(
        window,
        DELTA_FORCE_GAME_TITLE_KEYWORDS,
    )


def is_delta_force_launcher_window(window: WindowInfo) -> bool:
    process_name = window.process_name.strip().casefold()
    process_path = window.process_path.replace("/", "\\").rstrip("\\")
    path_name = process_path.rsplit("\\", 1)[-1].casefold()
    return DELTA_FORCE_LAUNCHER_PROCESS in {process_name, path_name}


def is_delta_force_launcher_main_window(window: WindowInfo) -> bool:
    return is_delta_force_launcher_window(window) and window_title_contains(
        window,
        DELTA_FORCE_LAUNCHER_TITLE_KEYWORDS,
    )


def is_delta_force_launcher_settings_window(window: WindowInfo) -> bool:
    return is_delta_force_launcher_window(window) and window_title_contains(
        window,
        DELTA_FORCE_LAUNCHER_SETTINGS_TITLE_KEYWORDS,
    )


user32 = ctypes.WinDLL("user32", use_last_error=True)

user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.c_void_p]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.c_void_p]
user32.ClientToScreen.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


def enable_dpi_awareness() -> None:
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except AttributeError:
            pass


def _process_identity(pid: int) -> tuple[str, str]:
    try:
        process = psutil.Process(pid)
        path = process.exe()
        return Path(path).name, path
    except (psutil.Error, OSError):
        return "", ""


def _client_rect(hwnd: int) -> Rect | None:
    raw = _Rect()
    if not user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(raw)):
        return None
    origin = _Point(raw.left, raw.top)
    if not user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(origin)):
        return None
    return Rect(origin.x, origin.y, raw.right - raw.left, raw.bottom - raw.top)


def enumerate_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        title = title_buffer.value.strip()

        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, len(class_buffer))
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = _client_rect(int(hwnd))
        if rect is None or not rect.valid():
            return True
        process_name, process_path = _process_identity(int(pid.value))
        if not process_name:
            return True
        if class_buffer.value in {"Progman", "WorkerW", "Shell_TrayWnd"}:
            return True
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                class_name=class_buffer.value,
                pid=int(pid.value),
                process_name=process_name,
                process_path=process_path,
                client_rect=rect,
                minimized=bool(user32.IsIconic(hwnd)),
            )
        )
        return True

    if not user32.EnumWindows(callback, 0):
        error = ctypes.get_last_error()
        if error:
            raise ctypes.WinError(error)
    return sorted(windows, key=lambda item: (item.process_name.casefold(), item.title.casefold()))


def matching_process_running(fingerprint: WindowFingerprint) -> bool:
    expected_name = fingerprint.process_name.casefold()
    expected_path = fingerprint.process_path.casefold()
    if not expected_name and not expected_path:
        return False
    for process in psutil.process_iter(["name", "exe"]):
        try:
            actual_name = (process.info.get("name") or "").casefold()
            actual_path = (process.info.get("exe") or "").casefold()
            if expected_path and actual_path == expected_path:
                return True
            if expected_name and actual_name == expected_name:
                return True
        except (psutil.Error, OSError):
            continue
    return False


class WindowBindingManager:
    def __init__(
        self,
        fingerprint: WindowFingerprint | None = None,
        candidate_filter: Callable[[WindowInfo], bool] | None = None,
        require_exact_title: bool = False,
    ) -> None:
        self._fingerprint = fingerprint or WindowFingerprint()
        self._candidate_filter = candidate_filter
        self._require_exact_title = require_exact_title
        self._bound: WindowInfo | None = None
        self._ambiguous: list[WindowInfo] = []
        self._lock = threading.RLock()

    @property
    def fingerprint(self) -> WindowFingerprint:
        with self._lock:
            return self._fingerprint

    @property
    def bound(self) -> WindowInfo | None:
        with self._lock:
            return self._bound

    @property
    def ambiguous(self) -> list[WindowInfo]:
        with self._lock:
            return list(self._ambiguous)

    def bind(self, window: WindowInfo) -> WindowFingerprint:
        with self._lock:
            self._fingerprint = WindowFingerprint.from_window(window)
            self._bound = window
            self._ambiguous = []
            return self._fingerprint

    def set_fingerprint(self, fingerprint: WindowFingerprint) -> None:
        with self._lock:
            self._fingerprint = fingerprint
            self._bound = None
            self._ambiguous = []

    def clear(self) -> None:
        with self._lock:
            self._bound = None
            self._ambiguous = []

    def _matches_saved_identity(self, window: WindowInfo) -> bool:
        if not self._require_exact_title:
            return True
        expected_title = self._fingerprint.title.strip()
        return not expected_title or window.title.strip().casefold() == expected_title.casefold()

    def refresh(self, windows: list[WindowInfo] | None = None) -> WindowInfo | None:
        candidates = windows if windows is not None else enumerate_windows()
        # A saved splash fingerprint/handle must not pin startup to a temporary
        # window. Keep untitled non-splash windows eligible for identity scoring.
        candidates = [window for window in candidates if not is_delta_force_splash_window(window)]
        if self._candidate_filter is not None:
            candidates = [window for window in candidates if self._candidate_filter(window)]
        by_handle = {item.hwnd: item for item in candidates}
        with self._lock:
            if self._bound is not None:
                current = by_handle.get(self._bound.hwnd)
                if (
                    current is not None
                    and self._matches_saved_identity(current)
                    and self._fingerprint.score(current) >= 50
                ):
                    self._bound = current
                    self._ambiguous = []
                    return current
                self._bound = None

            if not self._fingerprint.configured():
                self._ambiguous = []
                return None

            scored = sorted(
                (
                    (self._fingerprint.score(window), window)
                    for window in candidates
                    if window.client_rect.valid(self._fingerprint.minimum_client_size)
                    and self._matches_saved_identity(window)
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            scored = [item for item in scored if item[0] >= 50]
            if not scored:
                self._ambiguous = []
                return None
            if len(scored) > 1 and scored[0][0] - scored[1][0] < 10:
                self._ambiguous = [item[1] for item in scored[:5]]
                return None
            self._bound = scored[0][1]
            self._ambiguous = []
            return self._bound
