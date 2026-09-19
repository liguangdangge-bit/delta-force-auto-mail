from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes

from bulletbot.platform.input_ownership import InputLease, InputOwner

from .models import WindowInfo
from .window_errors import WindowActivationError


user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_KEYUP = 0x0002
VK_MENU = 0x12
VK_CONTROL = 0x11
VK_F4 = 0x73
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_TAB = 0x09
VK_L = 0x4C
VK_A = 0x41
VK_DELETE = 0x2E
SW_RESTORE = 9
HWND_TOP = 0
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
SWP_NOACTIVATE = 0x0010
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_SHOWWINDOW = 0x0040
ACTIVATION_TIMEOUT_SECONDS = 3.0
ACTIVATION_RETRY_SECONDS = 0.2


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.GetForegroundWindow.restype = wintypes.HWND
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


def _send(inputs: list[_Input]) -> None:
    array_type = _Input * len(inputs)
    values = array_type(*inputs)
    sent = user32.SendInput(len(inputs), values, ctypes.sizeof(_Input))
    if sent != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())


def window_snapshot(hwnd: int) -> dict:
    """Best-effort snapshot, including invisible/destroyed handles omitted by enumeration."""
    from .windows import _client_rect, _process_identity

    result = {"hwnd": hwnd, "hwnd_hex": f"0x{hwnd:X}"}
    try:
        handle = wintypes.HWND(hwnd)
        result.update(valid=bool(user32.IsWindow(handle)),
                      visible=bool(user32.IsWindowVisible(handle)),
                      minimized=bool(user32.IsIconic(handle)))
        title = ctypes.create_unicode_buffer(1024)
        user32.GetWindowTextW(handle, title, len(title))
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        rect = _client_rect(hwnd)
        result.update(title=title.value, pid=pid.value,
                      process=_process_identity(pid.value)[0] if pid.value else "",
                      client_size=[rect.width, rect.height] if rect else None)
    except Exception as exc:
        result["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    return result


class InputController:
    def __init__(
        self,
        enabled: bool = False,
        input_lease: InputLease | None = None,
    ) -> None:
        self.enabled = enabled
        self.input_lease = input_lease
        self._raised_launcher: int | None = None

    def activate(self, window: WindowInfo, *, timeout_seconds=ACTIVATION_TIMEOUT_SECONDS,
                 check_stop=None) -> None:
        from .windows import is_delta_force_launcher_window

        if not self.enabled:
            return
        if self.input_lease is None:
            raise RuntimeError("卡邮件流程没有取得游戏输入控制权")
        self.input_lease.assert_active(InputOwner.MAIL_STORAGE)
        if self._raised_launcher != window.hwnd:
            self.restore_launcher_topmost()
            if self._raised_launcher is not None:
                raise WindowActivationError("无法恢复上一个启动器窗口的置顶状态", {
                    "target": window_snapshot(self._raised_launcher),
                })
        try:
            if is_delta_force_launcher_window(window):
                handle = wintypes.HWND(window.hwnd)
                if user32.IsIconic(handle):
                    user32.ShowWindow(handle, SW_RESTORE)
                if not user32.GetWindowLongW(handle, GWL_EXSTYLE) & WS_EX_TOPMOST:
                    if not user32.SetWindowPos(
                        handle, wintypes.HWND(HWND_TOPMOST), 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
                    ):
                        raise WindowActivationError("无法临时置顶启动器", {
                            "target": window_snapshot(window.hwnd),
                        })
                    self._raised_launcher = window.hwnd
            self._activate_window(window, timeout_seconds=timeout_seconds, check_stop=check_stop)
        except BaseException:
            self.restore_launcher_topmost()
            raise

    def restore_launcher_topmost(self) -> None:
        hwnd = self._raised_launcher
        if hwnd is None:
            return
        handle = wintypes.HWND(hwnd)
        if user32.IsWindow(handle) and not user32.SetWindowPos(
            handle, wintypes.HWND(HWND_NOTOPMOST), 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        ):
            logging.getLogger(__name__).warning("无法取消启动器临时置顶：HWND 0x%X", hwnd)
            return
        self._raised_launcher = None

    def _activate_window(self, window: WindowInfo, *, timeout_seconds=ACTIVATION_TIMEOUT_SECONDS,
                         check_stop=None) -> None:
        if not self.enabled:
            return
        if self.input_lease is None:
            raise RuntimeError("卡邮件流程没有取得游戏输入控制权")
        if not window.minimized and self._foreground_window() == window.hwnd:
            return
        if window.minimized:
            user32.ShowWindow(wintypes.HWND(window.hwnd), SW_RESTORE)

        started = time.monotonic()
        timeout_seconds = max(0.0, min(ACTIVATION_TIMEOUT_SECONDS, timeout_seconds))
        deadline = started + timeout_seconds
        attempts = 0
        results = None
        while True:
            if check_stop is not None:
                check_stop()
            if time.monotonic() >= deadline:
                raise WindowActivationError(
                    f"无法将目标窗口切换到前台（已重试 {timeout_seconds:.1f} 秒）",
                    {"target": window_snapshot(window.hwnd),
                     "foreground": window_snapshot(self._foreground_window()),
                     "attempts": attempts, "elapsed_seconds": time.monotonic() - started,
                     "api_results": results},
                )
            results = self._request_foreground(window.hwnd)
            attempts += 1
            time.sleep(min(ACTIVATION_RETRY_SECONDS, max(0.0, deadline - time.monotonic())))
            if check_stop is not None:
                check_stop()
            if self._foreground_window() == window.hwnd:
                return

    @staticmethod
    def _foreground_window() -> int:
        return int(user32.GetForegroundWindow() or 0)

    @staticmethod
    def _request_foreground(hwnd: int) -> dict:
        target = wintypes.HWND(hwnd)
        foreground = InputController._foreground_window()
        current_thread = int(kernel32.GetCurrentThreadId())
        foreground_thread = 0
        attached = False

        if foreground and foreground != hwnd:
            foreground_thread = int(
                user32.GetWindowThreadProcessId(wintypes.HWND(foreground), None)
            )
            if foreground_thread and foreground_thread != current_thread:
                attached = bool(
                    user32.AttachThreadInput(current_thread, foreground_thread, True)
                )

        try:
            # SetWindowPos raises the main window above ordinary overlays, while
            # AttachThreadInput lets the worker thread request foreground focus.
            brought = bool(user32.BringWindowToTop(target))
            positioned = bool(user32.SetWindowPos(
                target,
                wintypes.HWND(HWND_TOP),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
            ))
            activated = bool(user32.SetForegroundWindow(target))
            return {"AttachThreadInput": attached, "BringWindowToTop": brought,
                    "SetWindowPos": positioned, "SetForegroundWindow": activated}
        finally:
            if attached:
                user32.AttachThreadInput(current_thread, foreground_thread, False)

    def click_normalized(
        self,
        window: WindowInfo,
        point: tuple[float, float],
        *,
        clicks: int = 1,
        interval_seconds: float = 0.0,
        hold_seconds: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        if clicks < 1:
            raise ValueError("点击次数必须至少为 1")
        if interval_seconds < 0 or hold_seconds < 0:
            raise ValueError("点击间隔和按住时间不能为负数")
        self.activate(window)
        absolute_x, absolute_y = self._normalized_absolute_point(window, point)
        common = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        _send(
            [
                _Input(
                    INPUT_MOUSE,
                    _InputUnion(
                        mi=_MouseInput(
                            absolute_x,
                            absolute_y,
                            0,
                            common | MOUSEEVENTF_MOVE,
                            0,
                            None,
                        )
                    ),
                )
            ]
        )
        for index in range(clicks):
            _send(
                [
                    _Input(
                        INPUT_MOUSE,
                        _InputUnion(
                            mi=_MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None)
                        ),
                    )
                ]
            )
            if hold_seconds:
                time.sleep(hold_seconds)
            _send(
                [
                    _Input(
                        INPUT_MOUSE,
                        _InputUnion(
                            mi=_MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None)
                        ),
                    )
                ]
            )
            if interval_seconds and index + 1 < clicks:
                time.sleep(interval_seconds)

    def right_click_normalized(
        self,
        window: WindowInfo,
        point: tuple[float, float],
        *,
        hold_seconds: float = 0.04,
    ) -> None:
        if not self.enabled:
            return
        if hold_seconds < 0:
            raise ValueError("右键按住时间不能为负数")
        self.activate(window)
        absolute_x, absolute_y = self._normalized_absolute_point(window, point)
        common = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        _send(
            [
                _Input(
                    INPUT_MOUSE,
                    _InputUnion(
                        mi=_MouseInput(
                            absolute_x,
                            absolute_y,
                            0,
                            common | MOUSEEVENTF_MOVE,
                            0,
                            None,
                        )
                    ),
                ),
                _Input(
                    INPUT_MOUSE,
                    _InputUnion(
                        mi=_MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTDOWN, 0, None)
                    ),
                ),
            ]
        )
        if hold_seconds:
            time.sleep(hold_seconds)
        _send(
            [
                _Input(
                    INPUT_MOUSE,
                    _InputUnion(
                        mi=_MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, None)
                    ),
                )
            ]
        )

    def move_normalized(self, window: WindowInfo, point: tuple[float, float]) -> None:
        if not self.enabled:
            return
        self.activate(window)
        absolute_x, absolute_y = self._normalized_absolute_point(window, point)
        common = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        _send(
            [
                _Input(
                    INPUT_MOUSE,
                    _InputUnion(
                        mi=_MouseInput(
                            absolute_x,
                            absolute_y,
                            0,
                            common | MOUSEEVENTF_MOVE,
                            0,
                            None,
                        )
                    ),
                )
            ]
        )

    @staticmethod
    def _normalized_absolute_point(
        window: WindowInfo,
        point: tuple[float, float],
    ) -> tuple[int, int]:
        x_ratio = min(max(float(point[0]), 0.0), 1.0)
        y_ratio = min(max(float(point[1]), 0.0), 1.0)
        x = window.client_rect.left + round(window.client_rect.width * x_ratio)
        y = window.client_rect.top + round(window.client_rect.height * y_ratio)

        virtual_left = user32.GetSystemMetrics(76)
        virtual_top = user32.GetSystemMetrics(77)
        virtual_width = user32.GetSystemMetrics(78)
        virtual_height = user32.GetSystemMetrics(79)
        absolute_x = round((x - virtual_left) * 65535 / max(virtual_width - 1, 1))
        absolute_y = round((y - virtual_top) * 65535 / max(virtual_height - 1, 1))
        return absolute_x, absolute_y

    def hotkey_alt_f4(self, window: WindowInfo) -> None:
        if not self.enabled:
            return
        self.activate(window)
        _send(
            [
                _Input(INPUT_KEYBOARD, _InputUnion(ki=_KeyboardInput(VK_MENU, 0, 0, 0, None))),
                _Input(INPUT_KEYBOARD, _InputUnion(ki=_KeyboardInput(VK_F4, 0, 0, 0, None))),
                _Input(INPUT_KEYBOARD, _InputUnion(ki=_KeyboardInput(VK_F4, 0, KEYEVENTF_KEYUP, 0, None))),
                _Input(INPUT_KEYBOARD, _InputUnion(ki=_KeyboardInput(VK_MENU, 0, KEYEVENTF_KEYUP, 0, None))),
            ]
        )

    def scroll_normalized(
        self,
        window: WindowInfo,
        point: tuple[float, float],
        delta: int,
    ) -> None:
        if not self.enabled:
            return
        self.activate(window)
        x = window.client_rect.left + round(window.client_rect.width * point[0])
        y = window.client_rect.top + round(window.client_rect.height * point[1])
        virtual_left = user32.GetSystemMetrics(76)
        virtual_top = user32.GetSystemMetrics(77)
        virtual_width = user32.GetSystemMetrics(78)
        virtual_height = user32.GetSystemMetrics(79)
        absolute_x = round((x - virtual_left) * 65535 / max(virtual_width - 1, 1))
        absolute_y = round((y - virtual_top) * 65535 / max(virtual_height - 1, 1))
        common = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        wheel_data = ctypes.c_ulong(delta & 0xFFFFFFFF).value
        _send(
            [
                _Input(INPUT_MOUSE, _InputUnion(mi=_MouseInput(absolute_x, absolute_y, 0, common | MOUSEEVENTF_MOVE, 0, None))),
                _Input(INPUT_MOUSE, _InputUnion(mi=_MouseInput(0, 0, wheel_data, MOUSEEVENTF_WHEEL, 0, None))),
            ]
        )

    def press_escape(self, window: WindowInfo) -> None:
        self.press_key(window, VK_ESCAPE)

    def press_tab(self, window: WindowInfo) -> None:
        self.press_key(window, VK_TAB)

    def press_space(self, window: WindowInfo) -> None:
        self.press_key(window, VK_SPACE)

    def press_loadout_scheme(self, window: WindowInfo) -> None:
        self.press_key(window, VK_L)

    def replace_text_with_digits(self, window: WindowInfo, value: int | str) -> None:
        digits = str(value)
        if not digits or not digits.isascii() or not digits.isdigit():
            raise ValueError("价格输入只允许 ASCII 数字")
        if not self.enabled:
            return
        self.activate(window)
        inputs = [
            _Input(
                INPUT_KEYBOARD,
                _InputUnion(ki=_KeyboardInput(VK_CONTROL, 0, 0, 0, None)),
            ),
            _Input(
                INPUT_KEYBOARD,
                _InputUnion(ki=_KeyboardInput(VK_A, 0, 0, 0, None)),
            ),
            _Input(
                INPUT_KEYBOARD,
                _InputUnion(
                    ki=_KeyboardInput(VK_A, 0, KEYEVENTF_KEYUP, 0, None)
                ),
            ),
            _Input(
                INPUT_KEYBOARD,
                _InputUnion(
                    ki=_KeyboardInput(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0, None)
                ),
            ),
            _Input(
                INPUT_KEYBOARD,
                _InputUnion(ki=_KeyboardInput(VK_DELETE, 0, 0, 0, None)),
            ),
            _Input(
                INPUT_KEYBOARD,
                _InputUnion(
                    ki=_KeyboardInput(VK_DELETE, 0, KEYEVENTF_KEYUP, 0, None)
                ),
            ),
        ]
        for digit in digits:
            virtual_key = ord(digit)
            inputs.extend(
                (
                    _Input(
                        INPUT_KEYBOARD,
                        _InputUnion(
                            ki=_KeyboardInput(virtual_key, 0, 0, 0, None)
                        ),
                    ),
                    _Input(
                        INPUT_KEYBOARD,
                        _InputUnion(
                            ki=_KeyboardInput(
                                virtual_key,
                                0,
                                KEYEVENTF_KEYUP,
                                0,
                                None,
                            )
                        ),
                    ),
                )
            )
        _send(inputs)

    def press_key(self, window: WindowInfo, virtual_key: int) -> None:
        if not self.enabled:
            return
        self.activate(window)
        _send(
            [
                _Input(INPUT_KEYBOARD, _InputUnion(ki=_KeyboardInput(virtual_key, 0, 0, 0, None))),
                _Input(INPUT_KEYBOARD, _InputUnion(ki=_KeyboardInput(virtual_key, 0, KEYEVENTF_KEYUP, 0, None))),
            ]
        )
