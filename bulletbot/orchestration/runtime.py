from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from bulletbot.ammunition_sale import (
    AmmunitionSaleOutcome,
    AmmunitionSaleResult,
    AmmunitionSaleSettings,
    AmmunitionSaleSettingsStore,
)
from bulletbot.loadout_purchase import (
    LoadoutPurchaseCheckpointStore,
    LoadoutPurchaseSettings,
    LoadoutPurchaseSettingsStore,
)
from bulletbot.mail_storage.models import MailWorkflowOutcome, MailWorkflowResult
from bulletbot.mail_storage.service import MailStorageWorkflow
from bulletbot.mail_storage.profile_editor import MailProfileEditor
from bulletbot.mail_storage.settings import MAX_PAK_FILES, AppSettings, SettingsStore
from bulletbot.trading.favorites_scan import normalize_product_name

from .checkpoint import SessionCheckpointStore
from .configuration import (
    MailIntegrationSettings,
    MailIntegrationSettingsStore,
    load_mail_profile,
)
from .coordinator import MailWorkflow, OrchestrationError, TradingMailOrchestrator
from .models import OrchestrationState

if TYPE_CHECKING:
    from bulletbot.platform.game_window_session import SharedGameWindowSession


_STATE_LABELS = {
    OrchestrationState.IDLE: "待启动",
    OrchestrationState.TRADING_STARTING: "交易启动中",
    OrchestrationState.TRADING_ACTIVE: "自动交易中",
    OrchestrationState.MAIL_PENDING: "等待卡邮件",
    OrchestrationState.TRADING_DRAINING: "交易收尾中",
    OrchestrationState.MAIL_STARTING: "卡邮件启动中",
    OrchestrationState.MAIL_RUNNING: "卡邮件运行中",
    OrchestrationState.MAIL_VERIFYING: "验证卡邮件结果",
    OrchestrationState.TRADING_REINITIALIZING: "交易环境重建中",
    OrchestrationState.SESSION_STOPPED: "会话已停止",
    OrchestrationState.SESSION_ERROR: "需要人工恢复",
}

_ACTIVE_STATES = frozenset(
    {
        OrchestrationState.TRADING_STARTING,
        OrchestrationState.TRADING_ACTIVE,
        OrchestrationState.MAIL_PENDING,
        OrchestrationState.TRADING_DRAINING,
        OrchestrationState.MAIL_STARTING,
        OrchestrationState.MAIL_RUNNING,
        OrchestrationState.MAIL_VERIFYING,
        OrchestrationState.TRADING_REINITIALIZING,
    }
)


@dataclass(frozen=True)
class MailIntegrationSnapshot:
    enabled: bool
    ready: bool
    state: OrchestrationState | None
    state_label: str
    product_name: str
    confirmed_quantity: int
    quantity_threshold: int
    completed_rounds: int | None = None
    pak_restored: bool | None = None
    trigger_reason: str | None = None
    error: str | None = None
    profile_path: str = ""
    workflow_running: bool = False
    standalone_active: bool = False
    standalone_stop_requested: bool = False
    standalone_result: MailWorkflowResult | None = None
    standalone_error: str | None = None
    loadout_purchase_active: bool = False
    loadout_purchase_stop_requested: bool = False
    loadout_purchase_result: MailWorkflowResult | None = None
    loadout_purchase_error: str | None = None
    ammunition_sale_active: bool = False
    ammunition_sale_stop_requested: bool = False
    ammunition_sale_result: AmmunitionSaleResult | None = None
    ammunition_sale_error: str | None = None

    @property
    def active(self) -> bool:
        return (
            self.standalone_active
            or self.loadout_purchase_active
            or self.ammunition_sale_active
            or self.workflow_running
            or self.state in _ACTIVE_STATES
        )

    @property
    def standalone_requires_recovery(self) -> bool:
        result = self.standalone_result
        return bool(
            self.standalone_error
            or (
                result is not None
                and (
                    result.outcome == MailWorkflowOutcome.FAILED
                    or not result.pak_restored
                )
            )
        )

    @property
    def loadout_purchase_requires_recovery(self) -> bool:
        result = self.loadout_purchase_result
        return bool(
            self.loadout_purchase_error
            or (
                result is not None
                and (
                    result.outcome == MailWorkflowOutcome.FAILED
                    or not result.pak_restored
                )
            )
        )


class MailIntegrationRuntime:
    """Own configuration, validation, and the replaceable session coordinator."""

    def __init__(
        self,
        project_root: Path,
        *,
        settings_store: MailIntegrationSettingsStore | None = None,
        checkpoint_store: SessionCheckpointStore | None = None,
        workflow_factory: Callable[[], MailWorkflow] | None = None,
        loadout_purchase_store: LoadoutPurchaseSettingsStore | None = None,
        loadout_checkpoint_store: LoadoutPurchaseCheckpointStore | None = None,
        ammunition_sale_store: AmmunitionSaleSettingsStore | None = None,
        legacy_profile_path: Path | None = None,
        game_window_session: SharedGameWindowSession | None = None,
        run_directory: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.settings_store = settings_store or MailIntegrationSettingsStore(
            self.project_root / "config" / "mail_integration.json"
        )
        self.checkpoint_store = checkpoint_store or SessionCheckpointStore(
            self.project_root / "data" / "mail_integration_checkpoint.json"
        )
        self._workflow_factory = workflow_factory
        self.loadout_purchase_store = loadout_purchase_store or (
            LoadoutPurchaseSettingsStore(
                self.project_root / "config" / "loadout_purchase.json"
            )
        )
        self.loadout_checkpoint_store = loadout_checkpoint_store or (
            LoadoutPurchaseCheckpointStore(
                self.project_root / "data" / "loadout_purchase_checkpoint.json"
            )
        )
        self.ammunition_sale_store = ammunition_sale_store or (
            AmmunitionSaleSettingsStore(
                self.project_root / "config" / "ammunition_sale.json"
            )
        )
        self.discarded_loadout_checkpoint_on_startup = False
        self._loadout_checkpoint_startup_error: str | None = None
        try:
            if self.loadout_checkpoint_store.path.exists():
                self.loadout_checkpoint_store.delete()
                self.discarded_loadout_checkpoint_on_startup = True
        except OSError as exc:
            self._loadout_checkpoint_startup_error = (
                f"无法废弃上一进程的配装买入检查点：{exc}"
            )
        self._game_window_session = game_window_session
        self.run_directory = (
            run_directory.expanduser().resolve() if run_directory is not None else None
        )
        self._workflow_on_event: Callable[..., None] | None = None
        self._workflow_on_frame: Callable[..., None] | None = None
        local_app_data = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        self.legacy_profile_path = legacy_profile_path or (
            self.project_root / "config" / "legacy_mail_settings.json"
        )
        self._settings = MailIntegrationSettings()
        self._orchestrator: TradingMailOrchestrator | None = None
        self._standalone_workflow: MailWorkflow | None = None
        self._standalone_stop_requested = False
        self._standalone_result: MailWorkflowResult | None = None
        self._standalone_error: str | None = None
        self._loadout_purchase_workflow: MailWorkflow | None = None
        self._loadout_purchase_stop_requested = False
        self._loadout_purchase_result: MailWorkflowResult | None = None
        self._loadout_purchase_error: str | None = None
        self._page_restart_workflow: MailWorkflow | None = None
        self._ammunition_sale_workflow: MailWorkflow | None = None
        self._ammunition_sale_stop_requested = False
        self._ammunition_sale_result: AmmunitionSaleResult | None = None
        self._ammunition_sale_error: str | None = None
        self.startup_error: str | None = None
        self._load()
        if self._loadout_checkpoint_startup_error is not None:
            self.startup_error = (
                f"{self.startup_error}；{self._loadout_checkpoint_startup_error}"
                if self.startup_error
                else self._loadout_checkpoint_startup_error
            )

    @property
    def settings(self) -> MailIntegrationSettings:
        return self._settings

    @property
    def orchestrator(self) -> TradingMailOrchestrator | None:
        return self._orchestrator

    def set_workflow_observers(
        self,
        *,
        on_event: Callable[..., None] | None = None,
        on_frame: Callable[..., None] | None = None,
    ) -> None:
        if self._standalone_workflow is not None:
            raise OrchestrationError("独立卡邮件运行中，不能更换界面观察回调")
        if self._loadout_purchase_workflow is not None:
            raise OrchestrationError("配装买入法运行中，不能更换界面观察回调")
        if self._ammunition_sale_workflow is not None:
            raise OrchestrationError("领邮件自动卖运行中，不能更换界面观察回调")
        if self._orchestrator is not None:
            self._orchestrator.set_mail_observers(
                on_event=on_event,
                on_frame=on_frame,
            )
        self._workflow_on_event = on_event
        self._workflow_on_frame = on_frame

    def start_standalone_mail(self, *, save_images: bool = True, scheme_index: int | None = None, end_at: datetime | None = None) -> None:
        """Start Mail without entering or modifying a trading session."""

        self._ensure_reconfiguration_allowed()
        if self.snapshot().standalone_requires_recovery:
            raise OrchestrationError(
                "上一次独立卡邮件未安全完成；请先确认 PAK 和游戏环境已恢复"
            )
        if self.snapshot().loadout_purchase_requires_recovery:
            raise OrchestrationError(
                "上一次配装买入法未安全完成；请先确认 PAK 和游戏环境已恢复"
            )
        if (
            self._orchestrator is not None
            and self._orchestrator.state == OrchestrationState.SESSION_ERROR
        ):
            raise OrchestrationError("一体化会话仍需人工恢复，不能启动独立卡邮件")

        raw_profile_path = (
            self._settings.mail_profile_path.strip()
            or "config/mail_storage_profile.json"
        )
        profile = self.create_profile_editor(raw_profile_path).settings
        if scheme_index is not None:
            if type(scheme_index) is not int or scheme_index not in (1, 2, 3):
                raise ValueError("卡邮件方案只能选择 1、2、3")
            profile = replace(profile, mail_scheme_index=scheme_index)
        self._validate_profile(profile)
        self._standalone_result = None
        self._standalone_error = None
        self._standalone_stop_requested = False
        try:
            workflow = self._create_workflow()
            self._standalone_workflow = workflow
            workflow.start(profile, **({"save_images": False} if not save_images else {}),
                           **({"scheduled_end_at": end_at} if end_at is not None else {}))
        except Exception as exc:
            self._standalone_workflow = None
            self._standalone_error = f"独立卡邮件启动失败：{exc}"
            raise OrchestrationError(self._standalone_error) from exc

    @property
    def loadout_purchase_settings(self) -> LoadoutPurchaseSettings:
        return self.loadout_purchase_store.load()

    def save_loadout_purchase_settings(
        self,
        settings: LoadoutPurchaseSettings,
    ) -> None:
        self._ensure_reconfiguration_allowed()
        settings.validate()
        self.loadout_purchase_store.save(settings)

    @property
    def ammunition_sale_settings(self) -> AmmunitionSaleSettings:
        return self.ammunition_sale_store.load()

    def save_ammunition_sale_settings(
        self,
        settings: AmmunitionSaleSettings,
    ) -> None:
        self._ensure_reconfiguration_allowed()
        settings.validate()
        self.ammunition_sale_store.save(settings)

    def start_ammunition_sale(
        self,
        settings: AmmunitionSaleSettings | None = None,
        *,
        preferred_game_hwnd: int | None = None,
        end_at: datetime | None = None,
        finish_mode: str = "safe",
    ) -> None:
        sale_settings = settings or self.ammunition_sale_store.load()
        sale_settings.validate(require_enabled=True)
        self._ensure_reconfiguration_allowed()
        raw_profile_path = (
            self._settings.mail_profile_path.strip()
            or "config/mail_storage_profile.json"
        )
        profile = self.create_profile_editor(raw_profile_path).settings
        if not profile.allow_input:
            raise OrchestrationError("领邮件自动卖需要启用游戏输入操作")
        if preferred_game_hwnd is None and not profile.game_window.configured():
            raise OrchestrationError("领邮件自动卖缺少游戏窗口")
        self.ammunition_sale_store.save(sale_settings)
        self._ammunition_sale_result = None
        self._ammunition_sale_error = None
        self._ammunition_sale_stop_requested = False
        try:
            workflow = self._create_workflow()
            starter = getattr(workflow, "start_ammunition_sale", None)
            if not callable(starter):
                raise RuntimeError("当前工作流不支持领邮件自动卖")
            self._ammunition_sale_workflow = workflow
            starter(
                profile,
                sale_settings,
                preferred_game_hwnd=preferred_game_hwnd,
                **({"end_at": end_at} if end_at is not None else {}),
                **({"finish_mode": finish_mode} if finish_mode != "safe" else {}),
            )
        except Exception as exc:
            self._ammunition_sale_workflow = None
            self._ammunition_sale_error = f"领邮件自动卖启动失败：{exc}"
            raise OrchestrationError(self._ammunition_sale_error) from exc

    def request_finish_scheduled_cycle(self, mode: str) -> bool:
        workflow = self._standalone_workflow if mode == "mail" else self._ammunition_sale_workflow
        if workflow is None:
            return False
        workflow.request_finish_cycle()
        workflow.request_resume()
        return True

    def request_stop_ammunition_sale(self) -> bool:
        workflow = self._ammunition_sale_workflow
        if workflow is None:
            return False
        if workflow.running:
            workflow.request_stop()
        self._ammunition_sale_stop_requested = True
        return True

    def poll_ammunition_sale(self) -> AmmunitionSaleResult | None:
        workflow = self._ammunition_sale_workflow
        if workflow is None:
            return self._ammunition_sale_result
        if workflow.running:
            return None
        workflow.join(0)
        result = workflow.result
        self._ammunition_sale_workflow = None
        self._ammunition_sale_stop_requested = False
        if not isinstance(result, AmmunitionSaleResult):
            self._ammunition_sale_error = (
                "领邮件自动卖工作线程已经结束，但没有返回结构化结果。"
            )
            raise OrchestrationError(self._ammunition_sale_error)
        self._ammunition_sale_result = result
        self._ammunition_sale_error = (
            f"领邮件自动卖未安全完成：{result.error or result.message}"
            if result.outcome is AmmunitionSaleOutcome.FAILED
            else None
        )
        return result

    def start_loadout_purchase(
        self,
        settings: LoadoutPurchaseSettings | None = None,
        *, end_at: datetime | None = None,
    ) -> None:
        purchase_settings = settings or self.loadout_purchase_store.load()
        purchase_settings.validate(require_enabled=True)
        self._start_loadout_purchase(
            saved_settings=purchase_settings,
            runtime_settings=purchase_settings,
            fast_mode=False,
            end_at=end_at,
        )

    def start_fast_loadout_purchase(
        self,
        settings: LoadoutPurchaseSettings | None = None,
        *, end_at: datetime | None = None,
    ) -> None:
        saved_settings = settings or self.loadout_purchase_store.load()
        saved_settings.fast_rule.validate(require_price=True)
        self._start_loadout_purchase(
            saved_settings=saved_settings,
            runtime_settings=saved_settings.for_fast_mode(),
            fast_mode=True,
            end_at=end_at,
        )

    def start_scheduled_game_close(self, settings: LoadoutPurchaseSettings) -> None:
        from bulletbot.loadout_purchase import LoadoutCheckpointStage

        checkpoint = self.loadout_checkpoint_store.load()
        if checkpoint is not None and checkpoint.stage not in {
            LoadoutCheckpointStage.PURCHASE_SCAN, LoadoutCheckpointStage.WAITING_SALE,
        }:
            raise OrchestrationError("上次配装仍有未完成的购买后处理，请先恢复该轮任务再关闭游戏")
        self._start_loadout_purchase(saved_settings=settings, runtime_settings=settings,
                                    fast_mode=False, close_game_only=True)

    def request_finish_loadout_round(self) -> bool:
        workflow = self._loadout_purchase_workflow
        if workflow is None:
            return False
        workflow.request_finish_loadout_round()
        self._loadout_purchase_stop_requested = True
        return True

    def _start_loadout_purchase(
        self,
        *,
        saved_settings: LoadoutPurchaseSettings,
        runtime_settings: LoadoutPurchaseSettings,
        fast_mode: bool,
        end_at: datetime | None = None,
        close_game_only: bool = False,
    ) -> None:
        self._ensure_reconfiguration_allowed()
        if self._loadout_checkpoint_startup_error is not None:
            raise OrchestrationError(self._loadout_checkpoint_startup_error)
        snapshot = self.snapshot()
        if snapshot.loadout_purchase_requires_recovery:
            raise OrchestrationError(
                "上一次配装买入法未安全完成；请先确认环境已恢复"
            )
        if snapshot.standalone_requires_recovery:
            raise OrchestrationError(
                "独立卡邮件仍需人工确认环境恢复，不能启动配装买入法"
            )
        if (
            self._orchestrator is not None
            and self._orchestrator.state == OrchestrationState.SESSION_ERROR
        ):
            raise OrchestrationError("一体化会话仍需人工恢复，不能启动配装买入法")

        runtime_settings.validate(require_enabled=not close_game_only)
        raw_profile_path = (
            self._settings.mail_profile_path.strip()
            or "config/mail_storage_profile.json"
        )
        profile = self.create_profile_editor(raw_profile_path).settings
        self._validate_profile(profile)
        self.loadout_purchase_store.save(saved_settings)
        self._loadout_purchase_result = None
        self._loadout_purchase_error = None
        self._loadout_purchase_stop_requested = False
        try:
            workflow = self._create_workflow()
            starter = getattr(workflow, "start_loadout_purchase", None)
            if not callable(starter):
                raise RuntimeError("当前卡邮件工作流不支持配装买入模式")
            self._loadout_purchase_workflow = workflow
            timing_kwargs = {"end_at": end_at} if end_at is not None else {}
            if close_game_only:
                workflow.start_game_close(profile, runtime_settings)
            elif fast_mode:
                starter(profile, runtime_settings, fast_mode=True, **timing_kwargs)
            else:
                starter(profile, runtime_settings, **timing_kwargs)
        except Exception as exc:
            self._loadout_purchase_workflow = None
            mode_label = "极速配装模式" if fast_mode else "配装买入法"
            self._loadout_purchase_error = f"{mode_label}启动失败：{exc}"
            raise OrchestrationError(self._loadout_purchase_error) from exc

    def request_stop_loadout_purchase(self) -> bool:
        workflow = self._loadout_purchase_workflow
        if workflow is None:
            return False
        if workflow.running:
            try:
                workflow.request_stop(discard_loadout_checkpoint=True)
            except TypeError:
                # Compatibility for injected test or third-party workflows.
                workflow.request_stop()
                self.loadout_checkpoint_store.delete()
        else:
            self.loadout_checkpoint_store.delete()
        self._loadout_purchase_stop_requested = True
        return True

    def poll_loadout_purchase(self) -> MailWorkflowResult | None:
        workflow = self._loadout_purchase_workflow
        if workflow is None:
            return self._loadout_purchase_result
        if workflow.running:
            return None
        workflow.join(0)
        result = workflow.result
        self._loadout_purchase_workflow = None
        self._loadout_purchase_stop_requested = False
        if result is None:
            self._loadout_purchase_error = (
                "配装买入工作线程已经结束，但没有返回结构化结果。"
            )
            raise OrchestrationError(self._loadout_purchase_error)
        self._loadout_purchase_result = result
        if result.outcome == MailWorkflowOutcome.FAILED or not result.pak_restored:
            detail = result.error or result.message
            if not result.pak_restored:
                detail = f"{detail}；PAK 未确认恢复"
            self._loadout_purchase_error = f"配装买入法未安全完成：{detail}"
        else:
            self._loadout_purchase_error = None
        return result

    def acknowledge_loadout_purchase_recovery(self) -> None:
        snapshot = self.snapshot()
        if snapshot.loadout_purchase_active:
            raise OrchestrationError("配装买入法仍在清理；请等待 PAK 恢复检查结束")
        if not snapshot.loadout_purchase_requires_recovery:
            raise OrchestrationError("当前没有需要确认的配装买入异常")
        self._loadout_purchase_result = None
        self._loadout_purchase_error = None
        self._loadout_purchase_stop_requested = False
        self.loadout_checkpoint_store.delete()

    def active_workflow_pause_state(self) -> tuple[bool, bool]:
        workflow = self._active_pause_workflow()
        if workflow is None:
            return False, False
        return (
            bool(getattr(workflow, "pause_requested", False)),
            bool(getattr(workflow, "paused", False)),
        )

    def request_pause_active_workflow(self) -> bool:
        workflow = self._active_pause_workflow()
        request = getattr(workflow, "request_pause", None)
        return bool(callable(request) and request())

    def request_resume_active_workflow(self) -> bool:
        workflow = self._active_pause_workflow()
        request = getattr(workflow, "request_resume", None)
        return bool(callable(request) and request())

    def _active_pause_workflow(self):
        if (
            self._ammunition_sale_workflow is not None
            and self._ammunition_sale_workflow.running
        ):
            return self._ammunition_sale_workflow
        if (
            self._loadout_purchase_workflow is not None
            and self._loadout_purchase_workflow.running
        ):
            return self._loadout_purchase_workflow
        if self._standalone_workflow is not None and self._standalone_workflow.running:
            return self._standalone_workflow
        if self._orchestrator is not None and self._orchestrator.mail_workflow_running:
            return self._orchestrator
        return None

    def request_stop_standalone_mail(self) -> bool:
        workflow = self._standalone_workflow
        if workflow is None:
            return False
        if workflow.running:
            workflow.request_stop()
        self._standalone_stop_requested = True
        return True

    def poll_standalone_mail(self) -> MailWorkflowResult | None:
        workflow = self._standalone_workflow
        if workflow is None:
            return self._standalone_result
        if workflow.running:
            return None
        workflow.join(0)
        result = workflow.result
        self._standalone_workflow = None
        self._standalone_stop_requested = False
        if result is None:
            self._standalone_error = (
                "独立卡邮件工作线程已经结束，但没有返回结构化结果。"
            )
            raise OrchestrationError(self._standalone_error)
        self._standalone_result = result
        if result.outcome == MailWorkflowOutcome.FAILED or not result.pak_restored:
            detail = result.error or result.message
            if not result.pak_restored:
                detail = f"{detail}；PAK 未确认恢复"
            self._standalone_error = f"独立卡邮件未安全完成：{detail}"
        else:
            self._standalone_error = None
        return result

    def acknowledge_standalone_recovery(self) -> None:
        snapshot = self.snapshot()
        if snapshot.standalone_active:
            raise OrchestrationError("独立卡邮件仍在清理；请等待 PAK 恢复检查结束")
        if not snapshot.standalone_requires_recovery:
            raise OrchestrationError("当前没有需要确认的独立卡邮件异常")
        self._standalone_result = None
        self._standalone_error = None
        self._standalone_stop_requested = False

    def create_profile_editor(self, raw_path: str) -> MailProfileEditor:
        default_profile_path = self._resolve_profile_path(
            "config/mail_storage_profile.json"
        )
        profile_path = (
            self._resolve_profile_path(raw_path.strip())
            if raw_path.strip()
            else default_profile_path
        )
        imported_from = None
        if profile_path.is_file():
            profile = load_mail_profile(profile_path)
        elif (
            profile_path == default_profile_path
            and self.legacy_profile_path.is_file()
        ):
            profile = load_mail_profile(self.legacy_profile_path)
            imported_from = self.legacy_profile_path
        else:
            profile = AppSettings()
        return MailProfileEditor(
            profile_path,
            profile,
            imported_from=imported_from,
            game_window_session=self._game_window_session,
        )

    def validate_settings(
        self,
        settings: MailIntegrationSettings,
        mail_profile: AppSettings | None = None,
    ) -> None:
        self._ensure_reconfiguration_allowed()
        settings.validate()
        if not settings.enabled:
            return
        profile = mail_profile or load_mail_profile(
            self._resolve_profile_path(settings.mail_profile_path)
        )
        self._validate_profile(profile)
        if (
            settings.product_name
            and not normalize_product_name(settings.product_name)
        ):
            raise ValueError("关联商品名称无法用于精确匹配")

    def configure(
        self,
        settings: MailIntegrationSettings,
        mail_profile: AppSettings | None = None,
    ) -> TradingMailOrchestrator | None:
        self.validate_settings(settings, mail_profile)
        if mail_profile is not None:
            if not settings.mail_profile_path.strip():
                raise ValueError("缺少 Mail 配置文件路径")
            SettingsStore(
                self._resolve_profile_path(settings.mail_profile_path)
            ).save(mail_profile)

        previous_checkpoint = self.checkpoint_store.load()
        normalized_name = normalize_product_name(settings.product_name)
        starts_new_cycle = bool(
            settings.enabled
            and previous_checkpoint is not None
            and (
                previous_checkpoint.normalized_product_name != normalized_name
                or previous_checkpoint.quantity_threshold
                != settings.quantity_threshold
            )
        )
        if starts_new_cycle and previous_checkpoint is not None:
            if previous_checkpoint.state not in {
                OrchestrationState.IDLE,
                OrchestrationState.SESSION_STOPPED,
            }:
                raise OrchestrationError(
                    "当前联动会话尚未安全结束，不能应用新的商品或数量阈值"
                )
            self.checkpoint_store.delete()

        try:
            candidate = (
                self._build_orchestrator(settings, mail_profile)
                if settings.enabled and normalized_name
                else None
            )
            self.settings_store.save(settings)
        except Exception:
            if starts_new_cycle and previous_checkpoint is not None:
                self.checkpoint_store.delete()
                self.checkpoint_store.save(previous_checkpoint)
            raise
        self._settings = settings
        self._orchestrator = candidate
        self.startup_error = None
        return candidate

    def prepare_trading_product(
        self,
        product_name: str,
    ) -> TradingMailOrchestrator | None:
        """Bind linked Mail counting to the single product selected for this run."""

        if not self._settings.enabled:
            return None
        normalized_name = normalize_product_name(product_name)
        if not normalized_name:
            raise ValueError("当前交易商品名称无法用于卡邮件计数")
        if (
            self._orchestrator is not None
            and normalize_product_name(self._orchestrator.configured_product)
            == normalized_name
        ):
            return self._orchestrator
        return self.configure(
            replace(self._settings, product_name=product_name.strip())
        )

    def acknowledge_error(self) -> None:
        if self._orchestrator is None:
            raise OrchestrationError("当前没有可确认的卡邮件会话")
        if self._orchestrator.mail_workflow_running:
            raise OrchestrationError("Mail 工作线程仍在清理；请等待 PAK 恢复检查结束")
        self._orchestrator.acknowledge_error(environment_verified=True)

    def reset_checkpoint(self) -> TradingMailOrchestrator | None:
        self._ensure_reconfiguration_allowed()
        self.checkpoint_store.delete()
        self._orchestrator = None
        try:
            self._orchestrator = (
                self._build_orchestrator(self._settings)
                if self._settings.enabled and self._settings.product_name.strip()
                else None
            )
            self.startup_error = None
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self.startup_error = f"卡邮件联动未就绪：{exc}"
            raise
        return self._orchestrator

    def restart_game_for_page_recovery(self, *, stop_event, pause_event) -> MailWorkflowResult:
        """Restart under the existing input lease lifecycle; the caller retains its task."""
        self._ensure_reconfiguration_allowed()
        if stop_event.is_set():
            raise OrchestrationError("页面恢复重启已取消")
        profile = self.create_profile_editor(self._settings.mail_profile_path).settings
        if not profile.allow_input:
            raise OrchestrationError("游戏重启需要启用输入操作")
        workflow = self._create_workflow()
        self._page_restart_workflow = workflow
        try:
            starter = getattr(workflow, "start_game_restart", None)
            if not callable(starter):
                raise OrchestrationError("当前工作流不支持页面恢复重启")
            starter(profile, self.loadout_purchase_store.load())
            paused = False
            while workflow.running:
                if stop_event.is_set():
                    workflow.request_stop()
                elif pause_event.is_set() != paused:
                    paused = pause_event.is_set()
                    workflow.request_pause() if paused else workflow.request_resume()
                workflow.join(0.1)
            result = workflow.result
            if result is None:
                raise OrchestrationError("游戏重启未返回结果")
            return result
        finally:
            if workflow.running:
                workflow.request_stop()
                workflow.join()
            self._page_restart_workflow = None

    def snapshot(self) -> MailIntegrationSnapshot:
        orchestrator = self._orchestrator
        standalone_fields = {
            "standalone_active": self._standalone_workflow is not None,
            "standalone_stop_requested": self._standalone_stop_requested,
            "standalone_result": self._standalone_result,
            "standalone_error": self._standalone_error,
            "loadout_purchase_active": self._loadout_purchase_workflow is not None,
            "loadout_purchase_stop_requested": self._loadout_purchase_stop_requested,
            "loadout_purchase_result": self._loadout_purchase_result,
            "loadout_purchase_error": self._loadout_purchase_error,
            "ammunition_sale_active": self._ammunition_sale_workflow is not None,
            "ammunition_sale_stop_requested": self._ammunition_sale_stop_requested,
            "ammunition_sale_result": self._ammunition_sale_result,
            "ammunition_sale_error": self._ammunition_sale_error,
        }
        if orchestrator is None:
            return MailIntegrationSnapshot(
                enabled=self._settings.enabled,
                ready=False,
                state=None,
                state_label=(
                    "配置错误"
                    if self.startup_error
                    else (
                        "等待交易商品"
                        if self._settings.enabled
                        else "未启用"
                    )
                ),
                product_name=self._settings.product_name,
                confirmed_quantity=0,
                quantity_threshold=self._settings.quantity_threshold,
                error=self.startup_error,
                profile_path=self._settings.mail_profile_path,
                workflow_running=self._page_restart_workflow is not None,
                **standalone_fields,
            )
        checkpoint = orchestrator.checkpoint
        result = checkpoint.last_mail_result or {}
        raw_completed_rounds = result.get("completed_rounds")
        try:
            completed_rounds = (
                int(raw_completed_rounds)
                if raw_completed_rounds is not None
                else None
            )
        except (TypeError, ValueError):
            completed_rounds = None
        pak_restored = result.get("pak_restored")
        return MailIntegrationSnapshot(
            enabled=True,
            ready=True,
            state=checkpoint.state,
            state_label=_STATE_LABELS[checkpoint.state],
            product_name=checkpoint.product_name,
            confirmed_quantity=checkpoint.confirmed_quantity,
            quantity_threshold=checkpoint.quantity_threshold,
            completed_rounds=completed_rounds,
            pak_restored=(pak_restored if isinstance(pak_restored, bool) else None),
            trigger_reason=checkpoint.trigger_reason,
            error=checkpoint.error,
            profile_path=self._settings.mail_profile_path,
            workflow_running=(orchestrator.mail_workflow_running or self._page_restart_workflow is not None),
            **standalone_fields,
        )

    def _load(self) -> None:
        try:
            self._settings = self.settings_store.load()
            if self._settings.enabled and self._settings.product_name.strip():
                self._orchestrator = self._build_orchestrator(self._settings)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._orchestrator = None
            self.startup_error = f"卡邮件联动未就绪：{exc}"

    def _build_orchestrator(
        self,
        settings: MailIntegrationSettings,
        mail_profile: AppSettings | None = None,
    ) -> TradingMailOrchestrator:
        profile = mail_profile or load_mail_profile(
            self._resolve_profile_path(settings.mail_profile_path)
        )
        self._validate_profile(profile)
        normalized_name = normalize_product_name(settings.product_name)
        if not normalized_name:
            raise ValueError("关联商品名称无法用于精确匹配")
        return TradingMailOrchestrator(
            product_name=settings.product_name,
            normalized_product_name=normalized_name,
            quantity_threshold=settings.quantity_threshold,
            mail_profile=profile,
            mail_workflow=self._create_workflow(),
            checkpoint_store=self.checkpoint_store,
        )

    def _create_workflow(self) -> MailWorkflow:
        workflow = (
            self._workflow_factory()
            if self._workflow_factory is not None
            else MailStorageWorkflow(
                game_window_session=self._game_window_session,
                diagnostics_root=self.run_directory,
                loadout_checkpoint_store=self.loadout_checkpoint_store,
                loadout_purchase_settings_provider=self.loadout_purchase_store.load,
            )
        )
        setter = getattr(workflow, "set_observers", None)
        if callable(setter):
            setter(
                on_event=self._workflow_on_event,
                on_frame=self._workflow_on_frame,
            )
        return workflow

    @staticmethod
    def _validate_profile(profile: AppSettings) -> None:
        if not profile.allow_input or not profile.allow_file_operations:
            raise ValueError("Mail 配置必须同时允许输入操作和 PAK 文件操作")
        if not profile.game_window.configured():
            raise ValueError("Mail 配置缺少游戏窗口指纹")
        if not profile.launcher_window.configured():
            raise ValueError("Mail 配置缺少启动器窗口指纹")
        if not profile.launcher_settings_window.configured():
            raise ValueError("Mail 配置缺少启动器设置窗口指纹")
        sources = profile.effective_pak_paths()
        staged_paths = profile.effective_staged_pak_paths()
        if (
            not sources
            or len(sources) > MAX_PAK_FILES
            or len(sources) != len(staged_paths)
        ):
            raise ValueError("Mail 配置中的 PAK 原路径和暂存路径无效")
        if len(set(sources)) != len(sources):
            raise ValueError("Mail 配置中的 PAK 原路径不能重复")
        if len({source.name.casefold() for source in sources}) != len(sources):
            raise ValueError("两个 PAK 文件不能使用相同文件名")
        if any(source == stage for source, stage in zip(sources, staged_paths)):
            raise ValueError("Mail 配置中的 PAK 原路径和暂存路径无效")
        for source, stage in zip(sources, staged_paths):
            if (
                source.drive
                and stage.drive
                and source.drive.casefold() != stage.drive.casefold()
            ):
                raise ValueError("PAK 原文件和暂存文件夹必须位于同一磁盘")

    def _resolve_profile_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.project_root):
            raise ValueError("请将配置文件复制到独立卡邮件项目目录内，再选择该副本")
        return resolved

    def _ensure_reconfiguration_allowed(self) -> None:
        if self._page_restart_workflow is not None:
            raise OrchestrationError("页面恢复正在重启游戏，请等待恢复结束")
        if self._ammunition_sale_workflow is not None:
            raise OrchestrationError("领邮件自动卖运行中；请先停止并等待任务结束")
        if self._loadout_purchase_workflow is not None:
            raise OrchestrationError("配装买入法运行中；请先停止并等待资源清理完成")
        if self._standalone_workflow is not None:
            raise OrchestrationError("独立卡邮件运行中；请先停止并等待资源清理完成")
        if self._orchestrator is not None and self._orchestrator.state in _ACTIVE_STATES:
            raise OrchestrationError("一体化会话运行中；请先停止会话再修改配置或重置检查点")
        if self._orchestrator is not None and self._orchestrator.mail_workflow_running:
            raise OrchestrationError("Mail 工作线程仍在清理；请等待 PAK 恢复检查结束")
