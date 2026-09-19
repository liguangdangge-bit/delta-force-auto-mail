from __future__ import annotations

from collections.abc import Callable
from time import monotonic, sleep

from .actions import NavigationActionError, NavigationActionExecutor
from .action_recovery import ActionState, PageRecoveryStopped, recover_page_action
from .catalog import format_page, page_descriptor
from .diagnostics import NavigationJournal
from .graph import DEFAULT_NAVIGATION_GRAPH, NavigationGraph
from .models import (
    GamePageId,
    NavigationBudgets,
    NavigationEdge,
    NavigationIntent,
    NavigationResult,
    NavigationStatus,
    PageKind,
    RebindStatus,
    UnifiedPageObservation,
    UnknownPageRecoveryPolicy,
    WindowRebindResult,
)


PageObserver = Callable[[], UnifiedPageObservation]
WindowRebinder = Callable[[], WindowRebindResult]
StopRequested = Callable[[], bool]
Sleeper = Callable[[float], None]


class UnifiedPageNavigator:
    """Navigate one verified edge at a time with bounded recovery."""

    def __init__(
        self,
        *,
        observer: PageObserver,
        actions: NavigationActionExecutor,
        graph: NavigationGraph = DEFAULT_NAVIGATION_GRAPH,
        budgets: NavigationBudgets = NavigationBudgets(),
        window_rebinder: WindowRebinder | None = None,
        unknown_recovery: UnknownPageRecoveryPolicy | None = None,
        stop_requested: StopRequested = lambda: False,
        sleeper: Sleeper = sleep,
        journal: NavigationJournal | None = None,
        recover_actions: Callable[[], bool] = lambda: False,
        recovery_log: Callable[[str], None] = lambda _message: None,
    ) -> None:
        self._observer = observer
        self._actions = actions
        self._graph = graph
        self._budgets = budgets
        self._window_rebinder = window_rebinder
        self._unknown_recovery = unknown_recovery
        self._stop_requested = stop_requested
        self._sleep = sleeper
        self._journal = journal or NavigationJournal()
        self._recover_actions = recover_actions
        self._recovery_log = recovery_log

    def navigate(self, intent: NavigationIntent) -> NavigationResult:
        history_start = self._journal.sequence
        recognition_retries = 0
        action_retries = 0
        window_rebinds = 0
        executed_actions = 0
        action_attempts: dict[tuple[GamePageId, GamePageId, str], int] = {}
        observation: UnifiedPageObservation | None = None
        pending_edge: NavigationEdge | None = None
        pending_deadline = 0.0
        last_generation: int | None = None
        stability_rechecks = 0
        unknown_wait_completed = False
        unknown_action_index = 0
        unknown_recovery_round = 1

        invalid_targets = {
            page_id
            for page_id in intent.target_pages
            if page_descriptor(page_id).kind in {PageKind.META, PageKind.TRANSITION}
        }
        if invalid_targets:
            invalid = sorted(invalid_targets, key=lambda page_id: page_id.value)[0]
            return self._finish(
                history_start,
                NavigationStatus.NO_SAFE_PATH,
                intent,
                None,
                f"{format_page(invalid)} 不能作为稳定导航目标",
                recognition_retries,
                action_retries,
                window_rebinds,
            )

        self._journal.record(
            "request",
            f"业务 {intent.workflow.value} 请求到达 {format_page(intent.target_page)}："
            f"{intent.reason}",
        )

        while True:
            if self._stop_requested():
                return self._finish(
                    history_start,
                    NavigationStatus.STOPPED,
                    intent,
                    observation,
                    "收到停止请求，未继续执行导航动作",
                    recognition_retries,
                    action_retries,
                    window_rebinds,
                )

            if observation is None:
                try:
                    observation = self._observer()
                    if not isinstance(observation, UnifiedPageObservation):
                        raise TypeError("页面观察器未返回 UnifiedPageObservation")
                except Exception as exc:
                    self._journal.record(
                        "observation_error",
                        f"读取当前页面失败：{exc}",
                    )
                    observation = None

            if observation is not None:
                self._journal.record(
                    "observation",
                    f"当前：{format_page(observation.page_id)}，"
                    f"置信度 {observation.confidence:.0%}，"
                    f"稳定={'是' if observation.stable else '否'}",
                    page_id=observation.page_id,
                    window_generation=observation.window_generation,
                )
                if (
                    last_generation is not None
                    and observation.window_generation != last_generation
                ):
                    self._journal.record(
                        "window_generation",
                        f"游戏窗口代次从 {last_generation} 变为 "
                        f"{observation.window_generation}，旧页面上下文已丢弃",
                        page_id=observation.page_id,
                        window_generation=observation.window_generation,
                    )
                    pending_edge = None
                    pending_deadline = 0.0
                last_generation = observation.window_generation

            unusable = self._unusable_reason(observation)
            if unusable is not None:
                unknown_page = bool(
                    observation is not None
                    and observation.page_id in {GamePageId.UNKNOWN, GamePageId.INVALID_FRAME}
                )
                # A safe action can briefly produce an unrecognised animation
                # frame before its target page appears. Keep that frame inside
                # the action's bounded settle window instead of immediately
                # spending the global recognition and window-rebind budgets.
                waiting_for_action = pending_edge is not None
                if waiting_for_action and monotonic() < pending_deadline:
                    assert pending_edge is not None
                    self._journal.record(
                        "action_wait",
                        f"动作 {pending_edge.action.value} 后{unusable}；"
                        "等待目标页面稳定",
                        page_id=(observation.page_id if observation else None),
                        action=pending_edge.action,
                        window_generation=(
                            observation.window_generation if observation else 0
                        ),
                    )
                    observation = None
                    self._sleep(0.1)
                    continue
                if waiting_for_action:
                    assert pending_edge is not None
                    self._journal.record(
                        "action_timeout",
                        f"动作 {pending_edge.action.value} 后等待页面稳定超时；"
                        "转入识别与窗口恢复",
                        page_id=(observation.page_id if observation else None),
                        action=pending_edge.action,
                        window_generation=(
                            observation.window_generation if observation else 0
                        ),
                    )
                    pending_edge = None
                    pending_deadline = 0.0
                known_but_unstable = bool(
                    observation is not None
                    and observation.page_id
                    not in {GamePageId.UNKNOWN, GamePageId.INVALID_FRAME}
                    and not observation.stable
                )
                if known_but_unstable and stability_rechecks < 1:
                    stability_rechecks += 1
                    self._journal.record(
                        "stability_wait",
                        f"{unusable}；获取下一帧确认页面稳定",
                        page_id=observation.page_id,
                        attempt=stability_rechecks,
                        window_generation=observation.window_generation,
                    )
                    observation = None
                    self._sleep(0.1)
                    continue
                if (
                    unknown_page
                    and self._unknown_recovery is not None
                    and not unknown_wait_completed
                ):
                    unknown_wait_completed = True
                    recognition_retries += 1
                    self._journal.record(
                        "unknown_wait",
                        "当前页面未识别；等待 "
                        f"{self._unknown_recovery.wait_seconds:g} 秒后重新获取新帧",
                        page_id=observation.page_id,
                        attempt=1,
                        window_generation=observation.window_generation,
                    )
                    observation = None
                    self._sleep(self._unknown_recovery.wait_seconds)
                    continue
                if (
                    not (unknown_page and self._unknown_recovery is not None)
                    and recognition_retries < self._budgets.recognition_retries
                ):
                    recognition_retries += 1
                    self._journal.record(
                        "recognition_retry",
                        f"{unusable}；重新获取新帧 "
                        f"{recognition_retries}/{self._budgets.recognition_retries}",
                        page_id=(observation.page_id if observation else None),
                        attempt=recognition_retries,
                        window_generation=(
                            observation.window_generation if observation else 0
                        ),
                    )
                    observation = None
                    self._sleep(0.1)
                    continue

                unknown_or_invalid = observation is None or observation.page_id in {
                    GamePageId.UNKNOWN,
                    GamePageId.INVALID_FRAME,
                }
                if (
                    unknown_or_invalid
                    and self._window_rebinder is not None
                    and window_rebinds < self._budgets.window_rebinds
                ):
                    window_rebinds += 1
                    rebind_result = self._try_rebind(window_rebinds)
                    if rebind_result.status == RebindStatus.MANUAL_REQUIRED:
                        return self._finish(
                            history_start,
                            NavigationStatus.MANUAL_BINDING_REQUIRED,
                            intent,
                            observation,
                            rebind_result.message or "存在多个相似游戏窗口",
                            recognition_retries,
                            action_retries,
                            window_rebinds,
                        )
                    if rebind_result.status != RebindStatus.REBOUND:
                        return self._finish(
                            history_start,
                            NavigationStatus.FAILED,
                            intent,
                            observation,
                            rebind_result.message or "刷新窗口后仍未找到游戏窗口",
                            recognition_retries,
                            action_retries,
                            window_rebinds,
                        )
                    observation = None
                    pending_edge = None
                    pending_deadline = 0.0
                    stability_rechecks = 0
                    continue

                if (
                    unknown_page
                    and self._unknown_recovery is not None
                    and unknown_action_index < len(self._unknown_recovery.actions)
                ):
                    if observation.page_id == GamePageId.INVALID_FRAME:
                        # Black frames receive the same timed observation budget,
                        # but never authorize blind recovery keys.
                        unknown_action_index += 1
                        self._journal.record(
                            "unknown_wait", "黑屏恢复：继续按时间观察，不发送盲按键",
                            page_id=observation.page_id,
                        )
                        self._sleep(self._unknown_recovery.action_wait_seconds)
                        observation = None
                        continue
                    action = self._unknown_recovery.actions[unknown_action_index]
                    unknown_action_index += 1
                    self._journal.record(
                        "unknown_action",
                        f"页面等待和窗口刷新后仍未识别；执行 {action.value} "
                        f"恢复 {unknown_action_index}/"
                        f"{len(self._unknown_recovery.actions)}",
                        page_id=observation.page_id,
                        action=action,
                        attempt=unknown_action_index,
                        window_generation=observation.window_generation,
                    )
                    if self._stop_requested():
                        return self._finish(
                            history_start,
                            NavigationStatus.STOPPED,
                            intent,
                            observation,
                            "未知页面恢复输入前收到停止请求",
                            recognition_retries,
                            action_retries,
                            window_rebinds,
                        )
                    try:
                        self._actions.execute(action, observation)
                    except Exception as exc:
                        return self._finish(
                            history_start,
                            NavigationStatus.FAILED,
                            intent,
                            observation,
                            f"未知页面恢复动作 {action.value} 执行失败：{exc}",
                            recognition_retries,
                            action_retries,
                            window_rebinds,
                        )
                    executed_actions += 1
                    observation = None
                    self._sleep(self._unknown_recovery.action_wait_seconds)
                    continue

                if (
                    unknown_page
                    and self._unknown_recovery is not None
                    and unknown_recovery_round < self._unknown_recovery.max_rounds
                ):
                    unknown_recovery_round += 1
                    unknown_wait_completed = False
                    unknown_action_index = 0
                    self._journal.record(
                        "unknown_recovery_round",
                        f"统一恢复第 {unknown_recovery_round}/"
                        f"{self._unknown_recovery.max_rounds} 轮：重新识别后再试键",
                        page_id=observation.page_id,
                        attempt=unknown_recovery_round,
                        window_generation=observation.window_generation,
                    )
                    observation = None
                    continue

                status = (
                    NavigationStatus.RETRYABLE
                    if unknown_or_invalid and self._window_rebinder is None
                    else NavigationStatus.FAILED
                )
                return self._finish(
                    history_start,
                    status,
                    intent,
                    observation,
                    f"页面观察不可用于安全导航：{unusable}",
                    recognition_retries,
                    action_retries,
                    window_rebinds,
                )

            assert observation is not None
            stability_rechecks = 0
            if pending_edge is not None:
                if observation.page_id in pending_edge.expected_pages:
                    self._journal.record(
                        "action_verified",
                        f"动作 {pending_edge.action.value} 已到达允许页面 "
                        f"{format_page(observation.page_id)}",
                        page_id=observation.page_id,
                        action=pending_edge.action,
                        window_generation=observation.window_generation,
                    )
                else:
                    self._journal.record(
                        "unexpected_page",
                        f"动作 {pending_edge.action.value} 后到达 "
                        f"{format_page(observation.page_id)}；将重新查询安全路径",
                        page_id=observation.page_id,
                        action=pending_edge.action,
                        window_generation=observation.window_generation,
                    )
                pending_edge = None
                pending_deadline = 0.0

            if observation.page_id in intent.target_pages:
                status = (
                    NavigationStatus.ARRIVED
                    if executed_actions == 0
                    and recognition_retries == 0
                    and window_rebinds == 0
                    else NavigationStatus.RECOVERED
                )
                return self._finish(
                    history_start,
                    status,
                    intent,
                    observation,
                    f"已到达 {format_page(observation.page_id)}",
                    recognition_retries,
                    action_retries,
                    window_rebinds,
                )

            edge = self._graph.next_edge_to_any(
                observation.page_id,
                intent.target_pages,
                max_edges=self._budgets.max_steps,
            )
            if edge is None:
                return self._finish(
                    history_start,
                    NavigationStatus.NO_SAFE_PATH,
                    intent,
                    observation,
                    f"没有从 {format_page(observation.page_id)} 到 "
                    f"{format_page(intent.target_page)} 的登记安全路径",
                    recognition_retries,
                    action_retries,
                    window_rebinds,
                )
            if executed_actions >= self._budgets.max_steps:
                return self._finish(
                    history_start,
                    NavigationStatus.FAILED,
                    intent,
                    observation,
                    f"导航动作达到最大步数 {self._budgets.max_steps}",
                    recognition_retries,
                    action_retries,
                    window_rebinds,
                )

            attempt_key = (edge.source, edge.target, edge.action.value)
            attempt = action_attempts.get(attempt_key, 0) + 1
            if attempt > 1:
                if not edge.repeatable or attempt > 1 + self._budgets.action_retries:
                    return self._finish(
                        history_start,
                        NavigationStatus.FAILED,
                        intent,
                        observation,
                        f"动作 {edge.action.value} 的安全重试预算已耗尽",
                        recognition_retries,
                        action_retries,
                        window_rebinds,
                    )
                if (
                    attempt == 2
                    and edge.action in self._budgets.delayed_retry_actions
                    and self._budgets.action_retry_delay_seconds > 0
                ):
                    delay = self._budgets.action_retry_delay_seconds
                    self._journal.record(
                        "action_retry_delay",
                        f"动作 {edge.action.value} 首次未完成；等待 "
                        f"{delay:g} 秒后再进行第 2 次尝试",
                        page_id=observation.page_id,
                        action=edge.action,
                        attempt=attempt,
                        window_generation=observation.window_generation,
                    )
                    # The injected sleeper is workflow-owned for the Mail
                    # flow, so stop requests remain interruptible while we
                    # wait for a potentially stalled game UI.
                    self._sleep(delay)
                action_retries += 1
            action_attempts[attempt_key] = attempt
            self._journal.record(
                "action",
                f"执行 {edge.action.value}：{format_page(edge.source)} -> "
                f"{format_page(edge.target)}；尝试 {attempt}/"
                f"{1 + self._budgets.action_retries}",
                page_id=observation.page_id,
                action=edge.action,
                attempt=attempt,
                window_generation=observation.window_generation,
            )
            if self._stop_requested():
                return self._finish(
                    history_start,
                    NavigationStatus.STOPPED,
                    intent,
                    observation,
                    "发送输入前收到停止请求",
                    recognition_retries,
                    action_retries,
                    window_rebinds,
                )
            try:
                self._actions.execute(edge.action, observation)
            except Exception as exc:
                if not isinstance(exc, NavigationActionError):
                    exc = NavigationActionError(
                        f"导航动作 {edge.action.value} 执行失败：{exc}"
                    )
                self._journal.record(
                    "action_error",
                    str(exc),
                    page_id=observation.page_id,
                    action=edge.action,
                    attempt=attempt,
                    window_generation=observation.window_generation,
                )
                if not edge.repeatable or attempt >= 1 + self._budgets.action_retries:
                    return self._finish(
                        history_start,
                        NavigationStatus.FAILED,
                        intent,
                        observation,
                        str(exc),
                        recognition_retries,
                        action_retries,
                        window_rebinds,
                    )

            executed_actions += 1
            pending_edge = edge
            pending_deadline = monotonic() + edge.timeout_seconds
            observation = None
            self._sleep(edge.settle_seconds)
            if self._recover_actions() and edge.repeatable:
                generation = last_generation
                recovery_page = None

                def observe_action() -> ActionState:
                    nonlocal observation, recovery_page
                    observation = self._observer()
                    page_key = (observation.page_id, observation.stable)
                    if page_key != recovery_page:
                        self._recovery_log(
                            f"动作后全图观察：{format_page(observation.page_id)}，"
                            f"稳定={'是' if observation.stable else '否'}"
                        )
                        recovery_page = page_key
                    if observation.window_generation != generation:
                        return ActionState.OTHER
                    if self._unusable_reason(observation) is not None:
                        return ActionState.UNKNOWN
                    if observation.page_id in edge.expected_pages | intent.target_pages:
                        return ActionState.TARGET
                    if observation.page_id == edge.source:
                        return ActionState.SOURCE
                    return ActionState.OTHER

                def retry_action() -> None:
                    nonlocal action_retries
                    self._actions.execute(edge.action, observation)
                    action_retries += 1

                def log_recovery(message: str) -> None:
                    self._journal.record("action_recovery", message, action=edge.action)
                    self._recovery_log(message)

                try:
                    recover_page_action(
                        observe=observe_action,
                        retry=retry_action,
                        wait=self._sleep,
                        stopped=self._stop_requested,
                        label=edge.action.value,
                        log=log_recovery,
                    )
                except PageRecoveryStopped:
                    return self._finish(
                        history_start, NavigationStatus.STOPPED, intent, observation,
                        "页面恢复已停止", recognition_retries, action_retries, window_rebinds,
                    )
                pending_edge = None
                pending_deadline = 0.0

    @staticmethod
    def _unusable_reason(
        observation: UnifiedPageObservation | None,
    ) -> str | None:
        if observation is None:
            return "未取得页面观察"
        if observation.page_id == GamePageId.UNKNOWN:
            return "当前页面未识别"
        if observation.page_id == GamePageId.INVALID_FRAME:
            return "当前画面无效或黑屏"
        if page_descriptor(observation.page_id).kind == PageKind.TRANSITION:
            return f"当前仍在{observation.chinese_name}"
        if not observation.stable:
            return f"{observation.chinese_name}尚未连续确认"
        return None

    def _try_rebind(self, attempt: int) -> WindowRebindResult:
        self._journal.record(
            "window_rebind",
            f"刷新窗口列表并尝试自动重新绑定 {attempt}/"
            f"{self._budgets.window_rebinds}",
            attempt=attempt,
        )
        try:
            result = self._window_rebinder()
            if not isinstance(result, WindowRebindResult):
                raise TypeError("窗口重绑器未返回 WindowRebindResult")
        except Exception as exc:
            result = WindowRebindResult(
                RebindStatus.FAILED,
                message=f"窗口重新绑定失败：{exc}",
            )
        self._journal.record(
            "window_rebind_result",
            result.message or result.status.value,
            attempt=attempt,
            window_generation=result.generation,
        )
        return result

    def _finish(
        self,
        history_start: int,
        status: NavigationStatus,
        intent: NavigationIntent,
        observation: UnifiedPageObservation | None,
        message: str,
        recognition_retries: int,
        action_retries: int,
        window_rebinds: int,
    ) -> NavigationResult:
        self._journal.record(
            "result",
            f"{status.value}: {message}",
            page_id=(observation.page_id if observation else None),
            window_generation=(
                observation.window_generation if observation else 0
            ),
        )
        failure_screenshot = (
            observation.screenshot_path if observation is not None else None
        )
        if status in {NavigationStatus.FAILED, NavigationStatus.NO_SAFE_PATH}:
            failure_screenshot = failure_screenshot or self._journal.capture_failure(
                f"navigation_{intent.workflow.value}_{intent.target_page.value}"
            )
        return NavigationResult(
            status=status,
            target_page=intent.target_page,
            current_page=(observation.page_id if observation else None),
            message=message,
            history=self._journal.entries_since(history_start),
            recognition_retries_used=recognition_retries,
            action_retries_used=action_retries,
            window_rebinds_used=window_rebinds,
            failure_screenshot_path=failure_screenshot,
        )
