from __future__ import annotations

import threading
from datetime import datetime
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bulletbot.ammunition_sale import AmmunitionSaleResult, AmmunitionSaleSettings
from bulletbot.platform.input_ownership import (
    GLOBAL_INPUT_OWNERSHIP,
    InputLease,
    InputOwner,
    InputOwnershipManager,
)
from bulletbot.loadout_purchase import (
    LoadoutPurchaseCheckpointStore,
    LoadoutPurchaseSettings,
)

from .models import MailWorkflowResult, PageObservation, WorkflowEvent
from .settings import AppSettings
from .windows import (
    WindowBindingManager,
    enumerate_windows,
    fingerprint_title_matches,
    is_delta_force_game_process,
    is_delta_force_game_window,
    is_delta_force_launcher_main_window,
    is_delta_force_launcher_settings_window,
    is_delta_force_launcher_window,
    is_standard_delta_force_game_window,
)
from .workflow import WorkflowController

if TYPE_CHECKING:
    from bulletbot.platform.game_window_session import SharedGameWindowSession


MailProfile = AppSettings
EventCallback = Callable[[WorkflowEvent], None]
FrameCallback = Callable[[np.ndarray, PageObservation, str], None]
ControllerFactory = Callable[..., WorkflowController]


class MailStorageWorkflow:
    """UI-independent lifecycle boundary for the mail storage workflow."""

    def __init__(
        self,
        *,
        on_event: EventCallback | None = None,
        on_frame: FrameCallback | None = None,
        controller_factory: ControllerFactory = WorkflowController,
        input_ownership: InputOwnershipManager = GLOBAL_INPUT_OWNERSHIP,
        game_window_session: SharedGameWindowSession | None = None,
        diagnostics_root: Path | None = None,
        loadout_checkpoint_store: LoadoutPurchaseCheckpointStore | None = None,
        loadout_purchase_settings_provider: Callable[[], LoadoutPurchaseSettings]
        | None = None,
    ) -> None:
        self._on_event = on_event
        self._on_frame = on_frame
        self._controller_factory = controller_factory
        self._input_ownership = input_ownership
        self._game_window_session = game_window_session
        self._diagnostics_root = diagnostics_root
        self._loadout_checkpoint_store = loadout_checkpoint_store
        self._loadout_purchase_settings_provider = loadout_purchase_settings_provider
        self._controller: WorkflowController | None = None
        self._completion_thread: threading.Thread | None = None
        self._input_lease: InputLease | None = None
        self._lock = threading.Lock()

    def set_observers(
        self,
        *,
        on_event: EventCallback | None = None,
        on_frame: FrameCallback | None = None,
    ) -> None:
        with self._lock:
            controller_running = (
                self._controller is not None and self._controller.running
            )
            cleanup_running = (
                self._completion_thread is not None
                and self._completion_thread.is_alive()
            )
            if controller_running or cleanup_running:
                raise RuntimeError("卡邮件流程运行中，不能更换界面观察回调")
            self._on_event = on_event
            self._on_frame = on_frame

    def _dispatch_event(self, event: WorkflowEvent) -> None:
        with self._lock:
            callback = self._on_event
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            pass

    def _dispatch_frame(
        self,
        image: np.ndarray,
        observation: PageObservation,
        target: str,
    ) -> None:
        with self._lock:
            callback = self._on_frame
        if callback is None:
            return
        try:
            callback(image, observation, target)
        except Exception:
            pass

    def start(
        self,
        profile: MailProfile,
        *,
        preferred_game_hwnd: int | None = None,
        loadout_purchase_settings: LoadoutPurchaseSettings | None = None,
        loadout_purchase_fast_mode: bool = False,
        ammunition_sale_settings: AmmunitionSaleSettings | None = None,
        loadout_end_at: datetime | None = None,
        close_game_only: bool = False,
        prepare_game_only: bool = False,
        restart_game_only: bool = False,
        scheduled_end_at: datetime | None = None,
        scheduled_finish_mode: str = "safe",
        save_images: bool = True,
    ) -> None:
        with self._lock:
            if self._controller is not None and self._controller.running:
                raise RuntimeError("卡邮件流程已经在运行")

            lease = self._input_ownership.acquire(InputOwner.MAIL_STORAGE)
            try:
                saved_game = profile.game_window
                if (
                    preferred_game_hwnd is None
                    and (saved_game.process_name or saved_game.process_path)
                    and not is_delta_force_game_process(
                        saved_game.process_name,
                        saved_game.process_path,
                    )
                ):
                    raise ValueError(
                        "Mail 配置中的游戏窗口不是三角洲行动客户端，请重新绑定游戏窗口"
                    )
                if self._game_window_session is None:
                    game_binding = WindowBindingManager(
                        profile.game_window,
                        candidate_filter=lambda window: (
                            is_standard_delta_force_game_window(window)
                            or (
                                profile.game_window.score(window) >= 50
                                and fingerprint_title_matches(
                                    profile.game_window, window
                                )
                            )
                        ),
                    )
                else:
                    self._game_window_session.seed_fingerprint(profile.game_window)
                    game_binding = self._game_window_session
                if preferred_game_hwnd is not None:
                    if self._game_window_session is not None:
                        game_binding.bind_handle(preferred_game_hwnd)
                    else:
                        preferred_game = next(
                            (
                                window
                                for window in enumerate_windows()
                                if window.hwnd == preferred_game_hwnd
                            ),
                            None,
                        )
                        if preferred_game is None:
                            raise RuntimeError("交易模块绑定的游戏窗口已经关闭")
                        if not is_delta_force_game_window(preferred_game):
                            raise RuntimeError(
                                "交易模块交接的窗口不是三角洲行动客户端，拒绝启动卡邮件"
                            )
                        game_binding.bind(preferred_game)
                elif self._game_window_session is not None:
                    game_binding.refresh()
                    if loadout_purchase_settings is None and not close_game_only and not prepare_game_only:
                        game_binding.require_binding()
                controller_kwargs = {
                    "on_event": self._dispatch_event,
                    "on_frame": self._dispatch_frame,
                    "input_lease": lease,
                }
                if not save_images:
                    controller_kwargs["save_diagnostic_images"] = False
                    controller_kwargs["on_frame"] = lambda *_args: None
                if self._diagnostics_root is not None:
                    controller_kwargs["diagnostics_root"] = self._diagnostics_root
                if self._loadout_checkpoint_store is not None:
                    controller_kwargs["loadout_checkpoint_store"] = (
                        self._loadout_checkpoint_store
                    )
                if self._loadout_purchase_settings_provider is not None:
                    controller_kwargs["loadout_purchase_settings_provider"] = (
                        self._loadout_purchase_settings_provider
                    )
                controller = self._controller_factory(
                    profile,
                    game_binding,
                    WindowBindingManager(
                        profile.launcher_window,
                        candidate_filter=lambda window: (
                            is_delta_force_launcher_window(window)
                            and not (
                                is_delta_force_launcher_settings_window(window)
                                and "设置" not in profile.launcher_window.title
                            )
                            and (
                                is_delta_force_launcher_main_window(window)
                                or fingerprint_title_matches(
                                    profile.launcher_window,
                                    window,
                                )
                            )
                        ),
                    ),
                    WindowBindingManager(
                        profile.launcher_settings_window,
                        candidate_filter=lambda window: (
                            is_delta_force_launcher_window(window)
                            and (
                                is_delta_force_launcher_settings_window(window)
                                or fingerprint_title_matches(
                                    profile.launcher_settings_window,
                                    window,
                                )
                            )
                        ),
                    ),
                    **controller_kwargs,
                )
                self._controller = controller
                self._input_lease = lease
                controller._scheduled_finish_mode = scheduled_finish_mode
                if scheduled_end_at is not None:
                    controller._scheduled_end_at = scheduled_end_at
                if restart_game_only:
                    controller.start_game_restart(loadout_purchase_settings or LoadoutPurchaseSettings())
                elif prepare_game_only:
                    controller.start_game_prepare(loadout_purchase_settings or LoadoutPurchaseSettings())
                elif close_game_only:
                    controller.start_game_close(loadout_purchase_settings or LoadoutPurchaseSettings())
                elif ammunition_sale_settings is not None:
                    if loadout_purchase_settings is not None:
                        raise ValueError("不能同时启动配装买入和领邮件自动卖")
                    ammunition_sale_settings.validate(require_enabled=True)
                    starter = getattr(controller, "start_ammunition_sale", None)
                    if not callable(starter):
                        raise RuntimeError("当前工作流不支持领邮件自动卖")
                    starter(ammunition_sale_settings)
                elif loadout_purchase_settings is None:
                    controller.start_workflow()
                else:
                    loadout_purchase_settings.validate(require_enabled=True)
                    timing_kwargs = {"end_at": loadout_end_at} if loadout_end_at is not None else {}
                    if loadout_purchase_fast_mode:
                        controller.start_loadout_purchase(
                            loadout_purchase_settings,
                            fast_mode=True,
                            **timing_kwargs,
                        )
                    else:
                        controller.start_loadout_purchase(
                            loadout_purchase_settings, **timing_kwargs
                        )
                completion_thread = threading.Thread(
                    target=self._wait_for_completion,
                    args=(controller, lease),
                    name="DeltaForceBulletBot-mail-storage-cleanup",
                    daemon=False,
                )
                self._completion_thread = completion_thread
                completion_thread.start()
            except Exception:
                if self._controller is not None and self._controller.running:
                    self._controller.request_stop()
                    self._controller.join()
                self._controller = None
                self._completion_thread = None
                self._input_lease = None
                lease.release()
                raise

    def start_loadout_purchase(
        self,
        profile: MailProfile,
        settings: LoadoutPurchaseSettings,
        *,
        preferred_game_hwnd: int | None = None,
        fast_mode: bool = False,
        end_at: datetime | None = None,
    ) -> None:
        self.start(
            profile,
            preferred_game_hwnd=preferred_game_hwnd,
            loadout_purchase_settings=settings,
            loadout_purchase_fast_mode=fast_mode,
            loadout_end_at=end_at,
        )

    def start_game_close(self, profile: MailProfile, settings: LoadoutPurchaseSettings) -> None:
        self.start(profile, loadout_purchase_settings=settings, close_game_only=True)

    def start_game_prepare(self, profile: MailProfile, settings: LoadoutPurchaseSettings) -> None:
        self.start(profile, loadout_purchase_settings=settings, prepare_game_only=True)

    def start_game_restart(self, profile: MailProfile, settings: LoadoutPurchaseSettings) -> None:
        self.start(profile, loadout_purchase_settings=settings, restart_game_only=True)

    def request_finish_cycle(self) -> None:
        with self._lock:
            controller = self._controller
        if controller is not None:
            controller.request_finish_cycle()

    def request_finish_loadout_round(self) -> None:
        with self._lock:
            controller = self._controller
        if controller is not None:
            controller.request_finish_loadout_round()

    def start_ammunition_sale(
        self,
        profile: MailProfile,
        settings: AmmunitionSaleSettings,
        *,
        preferred_game_hwnd: int | None = None,
        end_at: datetime | None = None,
        finish_mode: str = "safe",
    ) -> None:
        self.start(
            profile,
            preferred_game_hwnd=preferred_game_hwnd,
            ammunition_sale_settings=settings,
            **({"scheduled_end_at": end_at} if end_at is not None else {}),
            scheduled_finish_mode=finish_mode,
        )

    def _wait_for_completion(
        self,
        controller: WorkflowController,
        lease: InputLease,
    ) -> None:
        try:
            controller.join()
        finally:
            try:
                lease.release()
            finally:
                with self._lock:
                    if self._input_lease is lease:
                        self._input_lease = None

    def request_stop(self, *, discard_loadout_checkpoint: bool = False) -> None:
        with self._lock:
            controller = self._controller
        if controller is not None:
            if discard_loadout_checkpoint:
                controller.request_stop(discard_loadout_checkpoint=True)
            else:
                controller.request_stop()

    def request_pause(self) -> bool:
        with self._lock:
            controller = self._controller
        request = getattr(controller, "request_pause", None)
        return bool(callable(request) and request())

    def request_resume(self) -> bool:
        with self._lock:
            controller = self._controller
        request = getattr(controller, "request_resume", None)
        return bool(callable(request) and request())

    def join(self, timeout: float | None = None) -> None:
        with self._lock:
            completion_thread = self._completion_thread
        if completion_thread is not None:
            completion_thread.join(timeout)

    @property
    def running(self) -> bool:
        with self._lock:
            controller = self._controller
            completion_thread = self._completion_thread
        if completion_thread is not None:
            return completion_thread.is_alive()
        return controller is not None and controller.running

    @property
    def pause_requested(self) -> bool:
        with self._lock:
            controller = self._controller
        return bool(getattr(controller, "pause_requested", False))

    @property
    def paused(self) -> bool:
        with self._lock:
            controller = self._controller
        return bool(getattr(controller, "paused", False))

    @property
    def result(self) -> MailWorkflowResult | AmmunitionSaleResult | None:
        with self._lock:
            controller = self._controller
        return controller.result if controller is not None else None
