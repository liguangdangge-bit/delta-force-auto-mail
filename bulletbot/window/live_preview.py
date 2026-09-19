from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from threading import Condition, Lock
from time import monotonic

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

from bulletbot.domain.models import WindowBinding

logger = logging.getLogger(__name__)

try:
    from windows_capture import WindowsCapture
except ImportError:  # pragma: no cover - exercised by packaged fallback only
    WindowsCapture = None


def client_crop_bounds(
    frame_width: int,
    frame_height: int,
    client_width: int,
    client_height: int,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    """Return a safe client-area crop inside a captured outer-window frame."""

    width = min(max(1, client_width), max(1, frame_width))
    height = min(max(1, client_height), max(1, frame_height))
    left = min(max(0, offset_x), max(0, frame_width - width))
    top = min(max(0, offset_y), max(0, frame_height - height))
    return left, top, left + width, top + height


def _capture_border_toggle_unsupported(exc: BaseException) -> bool:
    """Identify the Windows Graphics Capture error for borderless capture."""

    message = str(exc).casefold()
    return (
        "capture border" in message
        and ("not supported" in message or "unsupported" in message)
    )


def _client_geometry(hwnd: int) -> tuple[int, int, int, int]:
    user32 = ctypes.windll.user32
    outer = wintypes.RECT()
    client = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetWindowRect(hwnd, ctypes.byref(outer)):
        raise RuntimeError("无法读取游戏窗口边框。")
    if not user32.GetClientRect(hwnd, ctypes.byref(client)):
        raise RuntimeError("无法读取游戏客户区。")
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise RuntimeError("无法换算游戏客户区坐标。")
    return (
        int(origin.x - outer.left),
        int(origin.y - outer.top),
        int(client.right - client.left),
        int(client.bottom - client.top),
    )


class LiveWindowPreview(QObject):
    """Share one HWND capture stream between preview and on-demand OCR."""

    frame_ready = pyqtSignal(QImage)
    failed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = Lock()
        self._frame_condition = Condition()
        self._capture = None
        self._control = None
        self._generation = 0
        self._running = False
        self._hwnd: int | None = None
        self._fps = 5
        self._preview_width = 480
        self._preview_enabled = True
        self._binding_size: tuple[int, int] | None = None
        self._frame_request_id = 0
        self._completed_request_id = 0
        self._latest_full_frame: np.ndarray | None = None
        self._last_error = ""
        self._stream_failed = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(
        self,
        binding: WindowBinding,
        *,
        fps: int = 5,
        preview_width: int = 480,
        preview_enabled: bool = True,
        force_restart: bool = False,
    ) -> None:
        fps = max(1, min(30, int(fps)))
        preview_width = max(240, min(960, int(preview_width)))
        if (
            not force_restart
            and self._running
            and not self._stream_failed
            and self._hwnd == binding.handle
            and self._binding_size == (binding.width, binding.height)
            and self._fps == fps
            and self._preview_width == preview_width
        ):
            self._preview_enabled = preview_enabled
            return
        self.stop()
        if WindowsCapture is None:
            self.failed.emit("实时预览依赖 windows-capture 未安装。")
            return

        self._generation += 1
        generation = self._generation
        self._hwnd = binding.handle
        self._fps = fps
        self._preview_width = preview_width
        self._preview_enabled = preview_enabled
        self._binding_size = (binding.width, binding.height)
        with self._frame_condition:
            self._stream_failed = False
            self._last_error = ""
        interval = 1.0 / fps
        next_preview_at = 0.0
        source_size: tuple[int, int] | None = None
        geometry = _client_geometry(binding.handle)
        failure_reported = False

        def on_frame_arrived(frame, control) -> None:
            nonlocal next_preview_at, source_size, geometry, failure_reported
            if generation != self._generation:
                control.stop()
                return
            now = monotonic()
            with self._frame_condition:
                request_id = self._frame_request_id
                full_frame_requested = (
                    self._completed_request_id < request_id
                )
            preview_due = self._preview_enabled and now >= next_preview_at
            if not preview_due and not full_frame_requested:
                return
            if preview_due:
                next_preview_at = now + interval
            try:
                current_size = (frame.width, frame.height)
                if current_size != source_size:
                    source_size = current_size
                    geometry = _client_geometry(binding.handle)
                offset_x, offset_y, client_width, client_height = geometry
                left, top, right, bottom = client_crop_bounds(
                    frame.width,
                    frame.height,
                    client_width,
                    client_height,
                    offset_x,
                    offset_y,
                )
                client_frame = frame.frame_buffer[top:bottom, left:right]
                if full_frame_requested:
                    full_rgb = cv2.cvtColor(client_frame, cv2.COLOR_BGRA2RGB).copy()
                    with self._frame_condition:
                        if generation != self._generation:
                            return
                        self._latest_full_frame = full_rgb
                        self._completed_request_id = request_id
                        self._last_error = ""
                        self._frame_condition.notify_all()
                if preview_due:
                    target_height = max(
                        1,
                        round(client_frame.shape[0] * preview_width / client_frame.shape[1]),
                    )
                    preview = cv2.resize(
                        client_frame,
                        (preview_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGRA2RGB)
                    image = QImage(
                        rgb.data,
                        rgb.shape[1],
                        rgb.shape[0],
                        rgb.strides[0],
                        QImage.Format_RGB888,
                    ).copy()
                    self.frame_ready.emit(image)
            except Exception as exc:
                if not failure_reported:
                    failure_reported = True
                    message = f"游戏窗口取帧失败：{exc}"
                    with self._frame_condition:
                        self._stream_failed = True
                        self._last_error = message
                        self._frame_condition.notify_all()
                    self.failed.emit(message)
                control.stop()

        def on_closed() -> None:
            if generation != self._generation:
                return
            with self._lock:
                self._running = False
            with self._frame_condition:
                self._stream_failed = True
                if not self._last_error:
                    self._last_error = "游戏窗口采集流已关闭。"
                self._frame_condition.notify_all()

        def create_capture(draw_border: bool | None):
            capture = WindowsCapture(
                cursor_capture=False,
                draw_border=draw_border,
                dirty_region=False,
                window_hwnd=binding.handle,
            )
            capture.event(on_frame_arrived)
            capture.event(on_closed)
            return capture

        def report_start_failure(exc: Exception) -> None:
            message = f"无法启动游戏窗口采集：{exc}"
            with self._frame_condition:
                self._stream_failed = True
                self._last_error = message
                self._frame_condition.notify_all()
            self.failed.emit(message)

        capture = create_capture(draw_border=False)
        try:
            control = capture.start_free_threaded()
        except Exception as exc:
            if not _capture_border_toggle_unsupported(exc):
                report_start_failure(exc)
                return
            capture = create_capture(draw_border=None)
            try:
                control = capture.start_free_threaded()
            except Exception as retry_exc:
                report_start_failure(retry_exc)
                return
        with self._lock:
            self._capture = capture
            self._control = control
            self._running = not self._stream_failed

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = bool(enabled)

    def rebind(self, binding: WindowBinding) -> None:
        """Move an active preview stream to a new shared-session binding."""

        if not self._running:
            return
        self.start(
            binding,
            fps=self._fps,
            preview_width=self._preview_width,
            preview_enabled=self._preview_enabled,
        )

    def capture_fresh(
        self,
        binding: WindowBinding,
        *,
        timeout: float = 2.0,
    ) -> Image.Image:
        """Wait for a fresh frame, then reopen WGC up to twice on timeout.

        The initial wait uses ``timeout``; retries wait 2 and 5 seconds.
        Wait deadlines start after stream startup.
        Arm the request before starting WGC so a static window's initial frame
        can satisfy it, even when delivered inside start_free_threaded().
        """

        for attempt, wait_timeout in enumerate((max(0.1, timeout), 2.0, 5.0)):
            with self._frame_condition:
                if attempt and (
                    generation != self._generation
                    or not self._running
                    or self._stream_failed
                ):
                    raise RuntimeError(self._last_error or "窗口采集已中断。")
                self._frame_request_id += 1
                request_id = self._frame_request_id
            if attempt:
                logger.warning(
                    "WGC 等待新帧超时（%s，HWND 0x%X），"
                    "第 %d/2 次主动重建采集流取图，等待最多 %g 秒",
                    binding.title, binding.handle, attempt, wait_timeout,
                )
                self.start(
                    binding,
                    fps=self._fps,
                    preview_width=self._preview_width,
                    preview_enabled=self._preview_enabled,
                    force_restart=True,
                )
            elif (
                not self._running
                or self._stream_failed
                or self._hwnd != binding.handle
                or self._binding_size != (binding.width, binding.height)
            ):
                self.start(binding, preview_enabled=False)
            if not self._running:
                raise RuntimeError(self._last_error or "窗口采集尚未启动。")

            deadline = monotonic() + wait_timeout
            generation = self._generation
            with self._frame_condition:
                while True:
                    if (
                        generation != self._generation
                        or not self._running
                        or self._stream_failed
                    ):
                        raise RuntimeError(self._last_error or "窗口采集已中断。")
                    if self._completed_request_id >= request_id:
                        if self._latest_full_frame is None:
                            raise RuntimeError("窗口没有返回有效画面。")
                        frame = self._latest_full_frame.copy()
                        return Image.fromarray(frame, mode="RGB")
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._frame_condition.wait(remaining)
        raise RuntimeError(
            f"等待窗口最新画面超时（{binding.title}，HWND 0x{binding.handle:X}）；"
            "主动重建 WGC 采集流重试 2 次（分别等待 2 秒、5 秒）后仍未收到新帧。"
        )

    def stop(self) -> None:
        self._generation += 1
        with self._lock:
            control = self._control
            self._control = None
            self._capture = None
            self._running = False
            self._hwnd = None
            self._binding_size = None
        with self._frame_condition:
            self._stream_failed = True
            self._last_error = "游戏窗口采集已停止。"
            self._frame_condition.notify_all()
        if control is not None:
            try:
                control.stop()
            except Exception:
                pass
        with self._frame_condition:
            self._latest_full_frame = None
            self._completed_request_id = 0
