from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class GlobalHotkeys(QObject):
    """Global controls that keep the game window in the foreground."""

    stop_requested = pyqtSignal()
    pause_toggled = pyqtSignal()
    toggle_window_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._keyboard = None
        self._registered_hotkeys: list[object] = []

    def start(self) -> str | None:
        try:
            import keyboard
        except ImportError:
            return "未安装 keyboard，无法启用全局快捷键。"
        try:
            self._keyboard = keyboard
            self._registered_hotkeys = [
                keyboard.add_hotkey(
                    "ctrl+esc",
                    self.stop_requested.emit,
                    suppress=True,
                ),
                keyboard.add_hotkey(
                    "ctrl+space",
                    self.pause_toggled.emit,
                    suppress=True,
                ),
                keyboard.add_hotkey("ctrl+alt+2", self.toggle_window_requested.emit),
            ]
        except Exception as exc:
            self.stop()
            return f"无法注册全局快捷键：{exc}"
        return None

    def stop(self) -> None:
        if self._keyboard is None:
            return
        for hotkey in self._registered_hotkeys:
            try:
                self._keyboard.remove_hotkey(hotkey)
            except Exception:
                pass
        self._registered_hotkeys = []
