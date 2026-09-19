"""Bounded recovery and diagnostics for the first resource prompt after restart."""
from dataclasses import asdict

import numpy as np

from .capture import CapturedFrame
from .input_control import window_snapshot, InputController
from .models import PageObservation, PageType
from .window_errors import WindowActivationError, WindowUnavailableError
from .windows import enumerate_windows, is_delta_force_splash_window


class StartupWait:
    def __init__(self, controller, binding, timeout):
        self.c = controller
        self.binding = binding
        self.started = controller._monotonic()
        self.deadline = self.started + timeout
        self.failures = 0
        self.last_error = None
        self.last_window = None
        self.last_frame = None
        self.last_page = None
        self.frame_hwnd = None
        self.pending = False
        self.last_report = float('-inf')
        self.last_signature = None
        self.reselect = False
        self.c.diagnostics.event('startup_wait_begin', timeout_seconds=timeout)
        self.c._emit(f"资源启动等待开始：总时限 {timeout:.1f} 秒；窗口短暂异常将刷新后重试")

    def event(self, event, **details):
        now = self.c._monotonic()
        payload = dict(details)
        payload.update(elapsed_seconds=now - self.started,
                       remaining_seconds=max(0, self.deadline - now),
                       recovery_count=self.failures)
        self.c.diagnostics.event(event, **payload)

    def refresh(self):
        windows = enumerate_windows()
        window = self.binding.refresh(windows)
        signature = (window.hwnd, window.pid) if window else None
        old = (self.last_window.hwnd, self.last_window.pid) if self.last_window else None
        if self.reselect or signature != old or (window is None and self.c._monotonic() - self.last_report >= 10):
            self.event('startup_window_selection', old_hwnd=old[0] if old else None,
                       selected=asdict(window) if window else None,
                       candidates=[dict(asdict(w), identity_score=self.binding.fingerprint.score(w),
                                        excluded_reason='splash_screen' if is_delta_force_splash_window(w) else None)
                                   for w in windows
                                   if w.process_name.casefold() == self.binding.fingerprint.process_name.casefold()],
                       reason=('recheck_after_capture_failure' if self.reselect else 'identity_selector')
                       if window else 'missing_or_ambiguous',
                       ambiguous=[w.hwnd for w in self.binding.ambiguous])
            self.c._emit(f"资源等待窗口：{hex(old[0]) if old else '无'} → "
                         f"{window.label if window else '无匹配窗口或候选存在歧义'}")
            self.last_report = self.c._monotonic()
        self.reselect = False
        self.last_window = window
        return window

    def observe(self, window, *args, **kwargs):
        try:
            result = self.c._observe(window, *args, **kwargs)
        except (WindowActivationError, WindowUnavailableError) as exc:
            self.failures += 1
            self.pending = True
            self.last_error = f'{type(exc).__name__}: {exc}'
            details = getattr(exc, 'details', None) or {
                'target': window_snapshot(window.hwnd),
                'foreground': window_snapshot(InputController._foreground_window())}
            details = dict(details)
            if 'elapsed_seconds' in details:
                details['activation_elapsed_seconds'] = details.pop('elapsed_seconds')
            self.event('startup_window_failure', error=self.last_error, **details)
            signature = (type(exc).__name__, window.hwnd)
            if signature != self.last_signature or self.c._monotonic() - self.last_report >= 10:
                self.c._emit(f"资源等待窗口暂不可用：{self.last_error}；"
                             f"剩余 {max(0, self.deadline - self.c._monotonic()):.1f} 秒，将重新检查窗口", 'warning')
                self.last_signature = signature
                self.last_report = self.c._monotonic()
                # Never attempt another activation just to collect a diagnostic image.
                if self.last_frame is not None:
                    try:
                        path = self.c.diagnostics.save_frame('startup_last_success_before_failure', self.last_frame.image)
                        self.event('startup_diagnostic_frame', path=str(path), source='last_successful_frame')
                    except Exception as save_exc:
                        self.event('startup_diagnostic_frame_failed', error=str(save_exc))
                else:
                    self.event('startup_diagnostic_frame_failed', error='尚无成功截图；未再次激活窗口取证')
            self.binding.clear()
            self.reselect = True
            return CapturedFrame(np.zeros((1, 1, 3), dtype=np.uint8), 0, 0, 0), PageObservation(PageType.INVALID_FRAME, 0)
        self.last_frame = result[0]
        self.last_page = result[1].page_type.value
        if self.frame_hwnd != window.hwnd:
            try:
                path = self.c.diagnostics.save_frame('startup_window_first_frame', result[0].image)
                self.event('startup_diagnostic_frame', path=str(path), hwnd=window.hwnd,
                           source='first_frame_after_binding')
            except Exception as exc:
                self.event('startup_diagnostic_frame_failed', error=str(exc))
            self.frame_hwnd = window.hwnd
        if self.pending:
            self.event('startup_capture_recovered', hwnd=window.hwnd, page=result[1].page_type.value)
            self.c._emit(f"资源等待截图已恢复：HWND 0x{window.hwnd:X}；继续等待目标页面")
            self.pending = False
        return result

    def finish(self, outcome):
        self.event('startup_wait_finished', outcome=outcome, last_error=self.last_error,
                   last_page=self.last_page,
                   target=window_snapshot(self.last_window.hwnd) if self.last_window else None,
                   foreground=window_snapshot(InputController._foreground_window()))
