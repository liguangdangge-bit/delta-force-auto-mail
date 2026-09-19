from __future__ import annotations

import ctypes
from ctypes import wintypes

from PIL import Image, ImageGrab

from bulletbot.domain.models import Rect, WindowBinding


class WindowService:
    """Find and capture a top-level Windows window by its native handle."""

    _GW_OWNER = 4
    _MIN_WIDTH = 640
    _MIN_HEIGHT = 480

    def __init__(self) -> None:
        self._frame_provider = None
        self._resilient_capture = None

    def set_frame_provider(self, provider) -> None:
        if self._resilient_capture is not None:
            self._resilient_capture.close()
            self._resilient_capture = None
        self._frame_provider = provider

    def notify_binding_changed(self, binding: WindowBinding) -> None:
        rebind = getattr(self._frame_provider, "rebind", None)
        if callable(rebind):
            rebind(binding)

    @staticmethod
    def enable_dpi_awareness() -> None:
        """Keep Win32 client coordinates and screenshot pixels in one scale."""

        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    @classmethod
    def list_windows(cls) -> list[WindowBinding]:
        bindings: list[WindowBinding] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def visit(handle: int, _lparam: int) -> bool:
            binding = cls._binding_from_handle(handle)
            if binding is not None:
                bindings.append(binding)
            return True

        ctypes.windll.user32.EnumWindows(visit, 0)
        return sorted(bindings, key=lambda binding: binding.title.casefold())

    @classmethod
    def foreground_window(cls) -> WindowBinding:
        handle = ctypes.windll.user32.GetForegroundWindow()
        binding = cls._binding_from_handle(handle)
        if binding is None:
            raise RuntimeError("无法获取当前前台游戏窗口。")
        return binding

    @classmethod
    def refresh(cls, binding: WindowBinding) -> WindowBinding:
        refreshed = cls._binding_from_handle(binding.handle)
        if refreshed is None:
            raise RuntimeError("游戏窗口已关闭、最小化，或不再可见。")
        return refreshed

    @staticmethod
    def assert_foreground(binding: WindowBinding) -> None:
        foreground_handle = ctypes.windll.user32.GetForegroundWindow()
        if int(foreground_handle or 0) != binding.handle:
            raise RuntimeError("游戏窗口当前不在前台。请切回游戏收藏页后再开始观察。")

    @staticmethod
    def activate(binding: WindowBinding) -> None:
        user32 = ctypes.windll.user32
        user32.ShowWindow(binding.handle, 9)
        user32.SetForegroundWindow(binding.handle)

    @classmethod
    def _binding_from_handle(cls, handle: int) -> WindowBinding | None:
        if not handle:
            return None
        user32 = ctypes.windll.user32
        if not user32.IsWindowVisible(handle) or user32.IsIconic(handle):
            return None
        if user32.GetWindow(handle, cls._GW_OWNER):
            return None

        title_length = user32.GetWindowTextLengthW(handle)
        if title_length <= 0:
            return None
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(handle, title_buffer, len(title_buffer))
        title = title_buffer.value.strip()
        if not title:
            return None

        client_rect = wintypes.RECT()
        if not user32.GetClientRect(handle, ctypes.byref(client_rect)):
            return None
        width = client_rect.right - client_rect.left
        height = client_rect.bottom - client_rect.top
        if width < cls._MIN_WIDTH or height < cls._MIN_HEIGHT:
            return None

        client_origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(handle, ctypes.byref(client_origin)):
            return None
        return WindowBinding(
            handle=int(handle),
            title=title,
            left=int(client_origin.x),
            top=int(client_origin.y),
            width=int(width),
            height=int(height),
        )

    def capture_wgc(self, binding: WindowBinding) -> Image.Image:
        """Prefer WGC, using MSS temporarily if WGC capture fails."""
        from bulletbot.mail_storage.capture import ScreenCapture
        from bulletbot.mail_storage.models import Rect as CaptureRect, WindowInfo
        from bulletbot.mail_storage.window_errors import WindowActivationError

        def provider_factory():
            if self._frame_provider is None:
                raise RuntimeError("WGC 截图服务未配置")
            return self._frame_provider

        def prepare(window):
            target = WindowBinding(window.hwnd, window.title, window.client_rect.left,
                                   window.client_rect.top, window.client_rect.width,
                                   window.client_rect.height)
            self.activate(target)
            try:
                self.assert_foreground(target)
            except RuntimeError as exc:
                raise WindowActivationError(str(exc), {}) from exc

        if self._resilient_capture is None:
            self._resilient_capture = ScreenCapture(provider_factory, prepare_mss_window=prepare)
        window = WindowInfo(binding.handle, binding.title, "", 0, "", "",
                            CaptureRect(binding.left, binding.top, binding.width, binding.height))
        frame = self._resilient_capture.capture(window)
        if frame.image.shape[:2] != (binding.height, binding.width):
            raise RuntimeError("截图尺寸与当前游戏客户区不一致")
        return Image.fromarray(frame.image[:, :, ::-1].copy(), mode="RGB")

    def capture(self, binding: WindowBinding) -> Image.Image:
        """Capture fresh pixels from the visible client area of the game.

        DirectX games frequently return black or stale images through the
        PrintWindow-style HWND capture path. A desktop grab is more reliable
        for the live frame, but it is allowed only while the bound game window
        is the foreground window. The UI hides itself before requesting a frame.
        """
        if self._frame_provider is not None:
            try:
                return self._frame_provider.capture_fresh(binding).convert("RGB")
            except Exception:
                # Windows Graphics Capture can stop while a game is minimized
                # or using an unsupported exclusive-fullscreen presentation.
                # Keep the foreground desktop crop as a bounded fallback.
                pass
        try:
            image = ImageGrab.grab(
                bbox=(
                    binding.left,
                    binding.top,
                    binding.left + binding.width,
                    binding.top + binding.height,
                ),
                all_screens=True,
            )
        except Exception as exc:
            raise RuntimeError(f"无法截取游戏画面：{exc}") from exc
        if image.width < 2 or image.height < 2:
            raise RuntimeError("游戏窗口截图为空。请确认游戏没有最小化。")
        return image.convert("RGB")

    def capture_region(self, binding: WindowBinding, region: Rect) -> Image.Image:
        """Capture a small client-area region, falling back to a full frame."""

        left = max(0, min(region.left, binding.width - 1))
        top = max(0, min(region.top, binding.height - 1))
        right = max(left + 1, min(region.right, binding.width))
        bottom = max(top + 1, min(region.bottom, binding.height))
        bbox = (
            binding.left + left,
            binding.top + top,
            binding.left + right,
            binding.top + bottom,
        )
        try:
            image = ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")
            if image.width < 2 or image.height < 2:
                raise RuntimeError("局部截图为空。")
            return image
        except Exception:
            # Some exclusive-fullscreen or minimized-window paths only work
            # through the configured full-frame provider.
            full = self.capture(binding)
            return full.crop((left, top, right, bottom)).convert("RGB")
