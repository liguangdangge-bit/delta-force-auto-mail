"""One bounded launcher restart after a game-exit request; never starts the game."""
from __future__ import annotations

import os
from pathlib import Path

import psutil

from .game_process_control import DELTA_FORCE_LAUNCHER_PROCESS


class LauncherRecovery:
    def __init__(self, executable: str = "") -> None:
        self.executable = executable
        self.attempted = False
        self.closing = None
        self.closing_created = None
        self.close_deadline = 0.0
        self.missing_since = None
        self.reopened_at = None

    def tick(self, now: float, *, visible: bool, running: bool = False) -> str | None:
        if self.closing is not None:
            try:
                same = abs(self.closing.create_time() - self.closing_created) < .001 and self.closing.is_running()
            except psutil.NoSuchProcess:
                same = False
            if same:
                if now >= self.close_deadline:
                    raise RuntimeError("启动器结束超时，未重复打开新实例")
                return None
            self.closing = None
            return self._open(now)

        if visible:
            self.missing_since = None
        elif self.missing_since is None:
            self.missing_since = now
        missing = self.missing_since is not None and now - self.missing_since >= 60
        # Allow the newly opened launcher's first refresh to settle.
        running = running and (self.reopened_at is None or now - self.reopened_at >= 5)
        if not missing and not running:
            return None
        if self.attempted:
            raise RuntimeError("已重开启动器一次，但仍无窗口或仍显示游戏运行中；停止重复重启")
        self.attempted = True
        self._begin_close(now)
        if self.closing is None:
            return self._open(now)
        return "检测到启动器仍显示游戏运行中" if running else "一分钟未出现启动器窗口"

    def _begin_close(self, now: float) -> None:
        candidates = []
        for p in psutil.process_iter(['name']):
            if (p.info['name'] or '').casefold() == DELTA_FORCE_LAUNCHER_PROCESS:
                candidates.append(p)
        if len(candidates) > 1:
            raise RuntimeError("存在多个三角洲启动器，无法确定重启对象")
        process = candidates[0] if candidates else None
        # Resolve the real path before closing; saved configuration is fallback.
        if process is not None:
            try:
                actual = process.exe()
                if actual:
                    self.executable = actual
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        path = Path(self.executable)
        if not path.is_absolute() or path.name.casefold() != DELTA_FORCE_LAUNCHER_PROCESS or not path.is_file():
            raise RuntimeError("启动器路径无效，请重新绑定启动器；尚未结束启动器")
        if process is not None:
            try:
                created = process.create_time()
                # psutil.kill also checks PID reuse before terminating.
                process.kill()
            except psutil.NoSuchProcess:
                return
            self.closing = process
            self.closing_created = created
            self.close_deadline = now + 10

    def _open(self, now: float) -> str:
        path = Path(self.executable)
        os.startfile(str(path), cwd=str(path.parent))
        self.reopened_at = now
        self.missing_since = now
        return "已请求重新打开启动器，继续检查窗口及游戏退出状态"
