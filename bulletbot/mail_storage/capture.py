from __future__ import annotations

from dataclasses import dataclass
import ctypes
import logging
from ctypes import wintypes
from time import monotonic, time
from threading import Lock

import cv2
import mss
import numpy as np

from .models import Rect, WindowInfo
from .window_errors import WindowActivationError, WindowUnavailableError

logger = logging.getLogger(__name__)


class CaptureBackendError(RuntimeError):
    """A capture backend failed; another backend may still work."""


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    timestamp: float
    mean: float
    std: float

    @property
    def healthy(self) -> bool:
        return self.image.size > 0 and self.std >= 3.0 and self.mean >= 2.0


class ScreenCapture:
    """Use WGC for trading and MSS for the explicitly selected mail phase."""

    def __init__(self, provider_factory=None, *, prepare_mss_window=None,
                 clock=monotonic, retry_seconds=300.0):
        self._provider_factory = provider_factory
        self._provider = None
        self._lock = Lock()
        self._backend = "wgc"
        self._preferred_backend = "wgc"
        self._clock = clock
        self._retry_seconds = retry_seconds
        self._retry_at = 0.0
        self._prepare_mss_window = prepare_mss_window

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def preferred_backend(self) -> str:
        return self._preferred_backend

    def _release_wgc(self):
        provider, self._provider = self._provider, None
        if provider is not None:
            try:
                provider.stop()
            except Exception:
                logger.warning("WGC 采集流关闭失败", exc_info=True)

    def set_backend(self, backend: str) -> None:
        if backend not in {"wgc", "mss"}:
            raise ValueError(f"不支持的截图方式：{backend}")
        with self._lock:
            if backend == self._preferred_backend:
                return
            self._release_wgc()
            self._preferred_backend = backend
            self._backend = backend
            self._retry_at = 0.0

    def capture(self, window: WindowInfo) -> CapturedFrame:
        return self._capture_window(window)

    def capture_region(self, window: WindowInfo, region: Rect) -> CapturedFrame:
        return self._capture_window(window, region)

    def _capture_window(self, window, region=None):
        with self._lock:
            candidate = self._backend
            if candidate != self._preferred_backend and self._clock() >= self._retry_at:
                candidate = self._preferred_backend
            try:
                frame = self._capture_backend(candidate, window, region)
            except CaptureBackendError as first:
                if candidate == "wgc":
                    self._release_wgc()
                alternate = "mss" if candidate == "wgc" else "wgc"
                logger.warning("%s 采集失败，尝试 %s（HWND 0x%X）：%s",
                               candidate.upper(), alternate.upper(), window.hwnd, first)
                try:
                    frame = self._capture_backend(alternate, window, region)
                except CaptureBackendError as second:
                    if alternate == "wgc":
                        self._release_wgc()
                    raise CaptureBackendError(f"两种截图方式均失败：{first}；{second}") from second
                candidate = alternate
                if candidate != self._preferred_backend:
                    self._retry_at = self._clock() + self._retry_seconds
            if candidate != self._backend:
                logger.warning("截图方式已切换为 %s，阶段优先 %s（HWND 0x%X）",
                               candidate.upper(), self._preferred_backend.upper(), window.hwnd)
            self._backend = candidate
            if candidate == "mss":
                self._release_wgc()
            return frame

    def _capture_backend(self, backend, window, region):
        if backend == "mss":
            return self._capture_mss_window(window, region)
        return self._capture_wgc_window(window, region)

    def _capture_wgc_window(self, window, region=None):
        from bulletbot.domain.models import WindowBinding
        from bulletbot.window.live_preview import LiveWindowPreview

        client = window.client_rect
        if client.width <= 0 or client.height <= 0:
            raise ValueError("截图区域尺寸无效")
        binding = WindowBinding(window.hwnd, window.title, client.left, client.top,
                                client.width, client.height)
        try:
            if self._provider is None:
                self._provider = (self._provider_factory or LiveWindowPreview)()
            full = self._provider.capture_fresh(binding)
            if full.size != (client.width, client.height):
                raise RuntimeError(f"客户区尺寸不匹配：{full.size}，期望 {client.width}×{client.height}")
            if region is not None:
                left = max(0, min(region.left, client.width - 1))
                top = max(0, min(region.top, client.height - 1))
                right = max(left + 1, min(region.right, client.width))
                bottom = max(top + 1, min(region.bottom, client.height))
                image = full.crop((left, top, right, bottom))
            else:
                image = full
            bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise CaptureBackendError(f"WGC 窗口截图失败（{window.title}，HWND 0x{window.hwnd:X}）：{exc}") from exc
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return CapturedFrame(bgr, time(), float(gray.mean()), float(gray.std()))

    def _capture_mss_window(self, window, region=None):
        # Preserve cancellation, ownership errors and typed activation failures.
        if self._prepare_mss_window is not None:
            self._prepare_mss_window(window)
        stage = "client_geometry"
        try:
            client = _current_client_rect(window.hwnd)
            if region is None:
                region = Rect(0, 0, client.width, client.height)
            left = max(0, min(region.left, client.width - 1))
            top = max(0, min(region.top, client.height - 1))
            right = max(left + 1, min(region.right, client.width))
            bottom = max(top + 1, min(region.bottom, client.height))
            monitor = dict(left=client.left + left, top=client.top + top,
                           width=right - left, height=bottom - top)
            # MSS owns thread-local GDI resources; create and close them on
            # the worker making this capture, including diagnostic captures.
            stage = "mss_grab"
            with mss.mss() as instance:
                raw = np.asarray(instance.grab(monitor))
            stage = "image_conversion"
            bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            if bgr.shape[:2] != (monitor["height"], monitor["width"]):
                raise RuntimeError("截图尺寸与客户区不一致")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return CapturedFrame(bgr, time(), float(gray.mean()), float(gray.std()))
        except (WindowActivationError, WindowUnavailableError):
            raise
        except Exception as exc:
            raise CaptureBackendError(
                f"MSS 窗口截图失败（{window.title}，HWND 0x{window.hwnd:X}，阶段 {stage}）：{exc}"
            ) from exc

    def close(self):
        with self._lock:
            self._release_wgc()
            self._backend = self._preferred_backend
            self._retry_at = 0.0


def _current_client_rect(hwnd: int) -> Rect:
    user32 = ctypes.windll.user32
    handle = wintypes.HWND(hwnd)
    if not user32.IsWindow(handle) or not user32.IsWindowVisible(handle):
        raise WindowUnavailableError("目标窗口已关闭或不可见")
    if user32.IsIconic(handle):
        raise WindowUnavailableError("目标窗口已最小化")
    client = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(handle, ctypes.byref(client)):
        raise WindowUnavailableError("无法读取目标窗口客户区")
    if not user32.ClientToScreen(handle, ctypes.byref(origin)):
        raise WindowUnavailableError("无法读取目标窗口客户区坐标")
    rect = Rect(origin.x, origin.y, client.right - client.left, client.bottom - client.top)
    if rect.width <= 0 or rect.height <= 0:
        raise WindowUnavailableError("目标窗口客户区尺寸无效")
    return rect
