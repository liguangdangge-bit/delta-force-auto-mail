from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from bulletbot.mail_storage.models import (
    MailWorkflowOutcome,
    MailWorkflowResult,
    PageType,
)
from bulletbot.mail_storage.service import MailProfile

from .checkpoint import SessionCheckpointStore
from .models import (
    MailTriggerKind,
    OrchestrationState,
    PurchaseCountUpdate,
    SessionCheckpoint,
    utc_now_text,
)


class MailWorkflow(Protocol):
    def start(
        self,
        profile: MailProfile,
        *,
        preferred_game_hwnd: int | None = None,
    ) -> None: ...

    def request_stop(self, *, discard_loadout_checkpoint: bool = False) -> None: ...

    def request_pause(self) -> bool: ...

    def request_resume(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    @property
    def running(self) -> bool: ...

    @property
    def pause_requested(self) -> bool: ...

    @property
    def paused(self) -> bool: ...

    @property
    def result(self) -> MailWorkflowResult | None: ...


class OrchestrationError(RuntimeError):
    pass


_INTERRUPTED_STATES = frozenset(
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

_SAFE_TRADING_HANDOFF_PAGES = frozenset(
    {
        PageType.GAME_HOME_PREPARE.value,
        PageType.GAME_HOME_READY.value,
    }
)


class TradingMailOrchestrator:
    """Own the durable state transitions between trading and mail storage."""

    def __init__(
        self,
        *,
        product_name: str,
        normalized_product_name: str,
        quantity_threshold: int,
        mail_profile: MailProfile,
        mail_workflow: MailWorkflow,
        checkpoint_store: SessionCheckpointStore,
    ) -> None:
        product_name = product_name.strip()
        normalized_product_name = normalized_product_name.strip()
        if not product_name or not normalized_product_name:
            raise ValueError("卡邮件联动必须配置商品名称")
        if quantity_threshold <= 0:
            raise ValueError("卡邮件触发数量必须大于零")
        self._mail_profile = mail_profile
        self._mail_workflow = mail_workflow
        self._store = checkpoint_store
        loaded = checkpoint_store.load()
        if loaded is None:
            self._checkpoint = SessionCheckpoint(
                product_name=product_name,
                normalized_product_name=normalized_product_name,
                quantity_threshold=quantity_threshold,
            )
            self._store.save(self._checkpoint)
        else:
            if (
                loaded.normalized_product_name != normalized_product_name
                or loaded.quantity_threshold != quantity_threshold
            ):
                raise OrchestrationError(
                    "现有会话检查点与当前商品或数量阈值不一致；"
                    "请先人工处理旧检查点。"
                )
            self._checkpoint = loaded
            if loaded.state in _INTERRUPTED_STATES:
                self._transition(
                    OrchestrationState.SESSION_ERROR,
                    error=(
                        f"检测到上次会话中断于 {loaded.state.value}；"
                        "已保留确认成交数量，禁止自动继续。"
                    ),
                    mail_stop_requested=False,
                )

    @property
    def state(self) -> OrchestrationState:
        return self._checkpoint.state

    @property
    def checkpoint(self) -> SessionCheckpoint:
        return self._checkpoint

    @property
    def confirmed_quantity(self) -> int:
        return self._checkpoint.confirmed_quantity

    @property
    def quantity_threshold(self) -> int:
        return self._checkpoint.quantity_threshold

    @property
    def configured_product(self) -> str:
        return self._checkpoint.product_name

    @property
    def handoff_active(self) -> bool:
        return self._mail_workflow.running or self.state in {
            OrchestrationState.MAIL_PENDING,
            OrchestrationState.TRADING_DRAINING,
            OrchestrationState.MAIL_STARTING,
            OrchestrationState.MAIL_RUNNING,
            OrchestrationState.MAIL_VERIFYING,
            OrchestrationState.TRADING_REINITIALIZING,
        }

    @property
    def mail_workflow_running(self) -> bool:
        return self._mail_workflow.running

    @property
    def pause_requested(self) -> bool:
        return bool(getattr(self._mail_workflow, "pause_requested", False))

    @property
    def paused(self) -> bool:
        return bool(getattr(self._mail_workflow, "paused", False))

    def request_pause(self) -> bool:
        if not self._mail_workflow.running:
            return False
        request = getattr(self._mail_workflow, "request_pause", None)
        return bool(callable(request) and request())

    def request_resume(self) -> bool:
        if not self._mail_workflow.running:
            return False
        request = getattr(self._mail_workflow, "request_resume", None)
        return bool(callable(request) and request())

    def set_mail_observers(
        self,
        *,
        on_event: Callable[..., None] | None = None,
        on_frame: Callable[..., None] | None = None,
    ) -> None:
        if self._mail_workflow.running:
            raise OrchestrationError("卡邮件工作线程运行中，不能更换界面观察回调")
        setter = getattr(self._mail_workflow, "set_observers", None)
        if callable(setter):
            setter(on_event=on_event, on_frame=on_frame)

    def begin_trading(self, product_name: str, normalized_product_name: str) -> None:
        self._require_product(normalized_product_name)
        if self.state == OrchestrationState.TRADING_ACTIVE:
            return
        if self.state == OrchestrationState.SESSION_ERROR:
            raise OrchestrationError(
                self._checkpoint.error or "会话处于错误恢复状态，不能自动继续"
            )
        if self.state not in {
            OrchestrationState.IDLE,
            OrchestrationState.SESSION_STOPPED,
        }:
            raise OrchestrationError(f"当前状态不能启动交易：{self.state.value}")
        if self.confirmed_quantity >= self.quantity_threshold:
            raise OrchestrationError(
                "保留的确认成交数量已经达到卡邮件阈值，请先完成卡邮件恢复。"
            )
        self._transition(
            OrchestrationState.TRADING_STARTING,
            product_name=product_name.strip() or self.configured_product,
            error=None,
            mail_stop_requested=False,
        )
        self._transition(OrchestrationState.TRADING_ACTIVE)

    def acknowledge_error(self, *, environment_verified: bool = False) -> None:
        if self.state != OrchestrationState.SESSION_ERROR:
            raise OrchestrationError("只有错误恢复状态可以人工确认")
        if not environment_verified:
            raise OrchestrationError(
                "必须先人工确认 PAK、游戏窗口和交易页面均已恢复。"
            )
        self._transition(
            OrchestrationState.SESSION_STOPPED,
            error=None,
            mail_stop_requested=False,
        )

    def record_confirmed_purchase(
        self,
        *,
        normalized_product_name: str,
        quantity: int,
    ) -> PurchaseCountUpdate:
        self._require_state(OrchestrationState.TRADING_ACTIVE)
        self._require_product(normalized_product_name)
        if quantity <= 0:
            raise ValueError("确认成交数量必须大于零")
        total = self.confirmed_quantity + quantity
        requested = total >= self.quantity_threshold
        if requested:
            self._transition(
                OrchestrationState.MAIL_PENDING,
                confirmed_quantity=total,
                trigger_kind=MailTriggerKind.QUANTITY_THRESHOLD,
                trigger_reason=(
                    f"确认成交累计 {total} 发，达到阈值 "
                    f"{self.quantity_threshold} 发"
                ),
            )
        else:
            self._update(confirmed_quantity=total)
        return PurchaseCountUpdate(
            quantity_added=quantity,
            confirmed_quantity=total,
            quantity_threshold=self.quantity_threshold,
            mail_requested=requested,
        )

    def request_storage_fallback(
        self,
        *,
        normalized_product_name: str,
        reason: str,
    ) -> bool:
        self._require_product(normalized_product_name)
        if self.state == OrchestrationState.MAIL_PENDING:
            return False
        self._require_state(OrchestrationState.TRADING_ACTIVE)
        self._transition(
            OrchestrationState.MAIL_PENDING,
            trigger_kind=MailTriggerKind.STORAGE_FALLBACK,
            trigger_reason=reason.strip() or "疑似仓库空间不足",
        )
        return True

    def start_mail_after_trading_drained(
        self,
        *,
        preferred_game_hwnd: int | None = None,
    ) -> None:
        self._require_state(OrchestrationState.MAIL_PENDING)
        self._transition(OrchestrationState.TRADING_DRAINING)
        self._transition(OrchestrationState.MAIL_STARTING)
        try:
            if preferred_game_hwnd is None:
                self._mail_workflow.start(self._mail_profile)
            else:
                self._mail_workflow.start(
                    self._mail_profile,
                    preferred_game_hwnd=preferred_game_hwnd,
                )
        except Exception as exc:
            self.fail(f"卡邮件流程启动失败：{exc}")
            raise OrchestrationError(self._checkpoint.error or str(exc)) from exc
        try:
            self._transition(OrchestrationState.MAIL_RUNNING)
        except OSError as exc:
            self._mail_workflow.request_stop()
            raise OrchestrationError(
                f"卡邮件已启动，但运行检查点写入失败，已请求安全停止：{exc}"
            ) from exc

    def poll_mail(self) -> MailWorkflowResult | None:
        self._require_state(OrchestrationState.MAIL_RUNNING)
        if self._mail_workflow.running:
            return None
        self._mail_workflow.join(0)
        self._transition(OrchestrationState.MAIL_VERIFYING)
        result = self._mail_workflow.result
        if result is None:
            self.fail("卡邮件工作线程已经结束，但没有返回结构化结果。")
            return None
        result_value = result.to_dict()
        if (
            result.outcome == MailWorkflowOutcome.COMPLETED
            and result.pak_restored
            and result.final_page in _SAFE_TRADING_HANDOFF_PAGES
        ):
            self._transition(
                OrchestrationState.TRADING_REINITIALIZING,
                last_mail_result=result_value,
                error=None,
                mail_stop_requested=False,
            )
        elif (
            result.outcome == MailWorkflowOutcome.STOPPED
            and result.pak_restored
        ):
            self._transition(
                OrchestrationState.SESSION_STOPPED,
                last_mail_result=result_value,
                error=None,
                mail_stop_requested=False,
            )
        else:
            detail = result.error or result.message
            if not result.pak_restored:
                detail = f"{detail}；PAK 未确认恢复"
            elif (
                result.outcome == MailWorkflowOutcome.COMPLETED
                and result.final_page not in _SAFE_TRADING_HANDOFF_PAGES
            ):
                detail = (
                    f"{detail}；结束页面 {result.final_page or 'unknown'} "
                    "不是可交接的游戏主界面"
                )
            self.fail(
                f"卡邮件流程未通过完成验证：{detail}",
                last_mail_result=result_value,
            )
        return result

    def complete_trading_reinitialization(self) -> None:
        self._require_state(OrchestrationState.TRADING_REINITIALIZING)
        self._transition(
            OrchestrationState.TRADING_ACTIVE,
            confirmed_quantity=0,
            trigger_kind=None,
            trigger_reason=None,
            error=None,
            mail_stop_requested=False,
        )

    def request_stop(self) -> None:
        if self._mail_workflow.running or self.state in {
            OrchestrationState.MAIL_STARTING,
            OrchestrationState.MAIL_RUNNING,
            OrchestrationState.MAIL_VERIFYING,
        }:
            self._mail_workflow.request_stop()
            self._update(mail_stop_requested=True)
            return
        if self.state in {
            OrchestrationState.IDLE,
            OrchestrationState.SESSION_STOPPED,
            OrchestrationState.SESSION_ERROR,
        }:
            return
        self._transition(
            OrchestrationState.SESSION_STOPPED,
            mail_stop_requested=False,
        )

    def fail(
        self,
        message: str,
        *,
        last_mail_result: dict[str, object] | None = None,
    ) -> None:
        updates: dict[str, object] = {
            "error": message,
            "mail_stop_requested": False,
        }
        if last_mail_result is not None:
            updates["last_mail_result"] = last_mail_result
        self._transition(OrchestrationState.SESSION_ERROR, **updates)

    def _require_product(self, normalized_product_name: str) -> None:
        if normalized_product_name.strip() != self._checkpoint.normalized_product_name:
            raise OrchestrationError(
                f"当前交易商品与卡邮件方案不匹配：{normalized_product_name}"
            )

    def _require_state(self, expected: OrchestrationState) -> None:
        if self.state != expected:
            raise OrchestrationError(
                f"状态转换无效：期望 {expected.value}，实际 {self.state.value}"
            )

    def _transition(self, state: OrchestrationState, **changes: object) -> None:
        self._update(state=state, **changes)

    def _update(self, **changes: object) -> None:
        changes["updated_at"] = utc_now_text()
        checkpoint = replace(self._checkpoint, **changes)
        self._store.save(checkpoint)
        self._checkpoint = checkpoint
