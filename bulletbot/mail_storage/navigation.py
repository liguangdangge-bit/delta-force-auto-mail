from __future__ import annotations

from bulletbot.navigation.action_recovery import PageActionStalled, PageRecoveryFailed, navigation_recovery_exhausted

from bulletbot.navigation import (
    GamePageId,
    MappedNavigationActionExecutor,
    NavigationAction,
    NavigationBudgets,
    NavigationIntent,
    NavigationJournal,
    NavigationResult,
    NavigationStatus,
    RebindStatus,
    UnifiedPageNavigator,
    UnknownPageRecoveryPolicy,
    WindowRebindResult,
    WorkflowKind,
)


# A stuck game UI can recover on its own after a long frame/render stall.  Keep
# this as a named policy value so the delay is visible and easy to tune without
# changing the navigator's generic retry budget.
GAME_HOME_ACTION_RETRY_DELAY_SECONDS = 120.0


class MailUnifiedNavigation:
    """Bind the shared navigator to reversible Mail workflow inputs."""

    def __init__(self, workflow) -> None:
        self._workflow = workflow
        self._navigator = UnifiedPageNavigator(
            observer=workflow._observe_unified_game,
            actions=MappedNavigationActionExecutor(
                {
                    NavigationAction.ESCAPE: lambda _page: self._press("escape"),
                    NavigationAction.SPACE: lambda _page: self._press("space"),
                    NavigationAction.TAB: lambda _page: self._press("tab"),
                    NavigationAction.OPEN_GAME_HOME: lambda _page: self._click(
                        "game_home_tab",
                        "打开开始游戏主界面",
                    ),
                    NavigationAction.OPEN_WAREHOUSE: lambda _page: self._click(
                        "warehouse_tab",
                        "打开仓库",
                    ),
                    NavigationAction.OPEN_MARKET: lambda _page: self._click(
                        "market_tab",
                        "打开交易行",
                    ),
                    NavigationAction.OPEN_MARKET_BUY: lambda _page: self._click(
                        "market_buy_tab",
                        "打开交易行购买页",
                    ),
                    NavigationAction.OPEN_MARKET_SELL: lambda _page: self._click(
                        "market_sell_tab",
                        "打开交易行出售页",
                    ),
                    NavigationAction.OPEN_GAME_MAP: lambda _page: self._click(
                        "map_card",
                        "打开端游对局地图",
                    ),
                    NavigationAction.EXPAND_MAP: lambda _page: self._click(
                        "map_expand",
                        "返回地图全览",
                    ),
                    NavigationAction.SELECT_LONGBOW: lambda _page: self._click(
                        "longbow_node",
                        "选择长弓溪谷",
                    ),
                }
            ),
            budgets=NavigationBudgets(
                recognition_retries=1,
                action_retries=1,
                window_rebinds=1,
                max_steps=12,
                action_retry_delay_seconds=GAME_HOME_ACTION_RETRY_DELAY_SECONDS,
                delayed_retry_actions=frozenset(
                    {NavigationAction.OPEN_GAME_HOME}
                ),
            ),
            window_rebinder=self._rebind_window,
            unknown_recovery=UnknownPageRecoveryPolicy(),
            stop_requested=workflow._stop.is_set,
            sleeper=workflow._sleep,
            journal=NavigationJournal(
                failure_capture=workflow._capture_unified_navigation_failure,
            ),
            recover_actions=lambda: bool(
                getattr(workflow, "_page_recovery_enabled", False)
                or getattr(workflow, "_mail_game_phase_active", False)
            ),
            recovery_log=workflow._emit,
        )

    def navigate(
        self,
        target_page: GamePageId,
        *,
        reason: str,
        acceptable_pages: frozenset[GamePageId] = frozenset(),
    ) -> NavigationResult:
        self._workflow._unified_page_stability.reset()
        intent = NavigationIntent(
            workflow=WorkflowKind.MAIL_STORAGE,
            target_page=target_page,
            acceptable_pages=acceptable_pages,
            reason=reason,
        )
        try:
            result = self._navigator.navigate(intent)
            if navigation_recovery_exhausted(result) and (
                getattr(self._workflow, "_mail_game_phase_active", False)
                or getattr(self._workflow, "_page_recovery_enabled", False)
            ):
                raise PageActionStalled(result.message)
        except PageActionStalled as exc:
            if getattr(self._workflow, "_mail_game_phase_active", False):
                # The mail phase owns the safe resume point and PAK lifecycle.
                raise
            self._workflow._check_stop()
            self._workflow._restart_for_page_recovery(str(exc))
            self._workflow._unified_page_stability.reset()
            try:
                result = self._navigator.navigate(intent)
            except PageActionStalled as retry_exc:
                raise PageRecoveryFailed(f"重启后仍无法到达 {target_page.value}：{retry_exc}") from retry_exc
            if not result.succeeded and result.status is not NavigationStatus.STOPPED:
                raise PageRecoveryFailed(f"重启后原导航任务未完成：{result.message}")
        for entry in result.history:
            if entry.event in {
                "action",
                "action_verified",
                "recognition_retry",
                "unknown_wait",
                "unknown_action",
                "stability_wait",
                "action_timeout",
                "action_retry_delay",
                "window_rebind",
                "window_generation",
                "unexpected_page",
                "result",
            }:
                self._workflow._emit(f"统一导航：{entry.message}")
        return result

    def observe_page(self):
        return self._workflow._observe_unified_game()

    def _press(self, key: str) -> None:
        game = self._workflow._require_bound(self._workflow.game, "游戏")
        self._workflow._press_key(game, key, f"{key}：统一页面导航")

    def _click(self, point_name: str, label: str) -> None:
        game = self._workflow._require_bound(self._workflow.game, "游戏")
        self._workflow._click(game, point_name, label)

    def _rebind_window(self) -> WindowRebindResult:
        try:
            window = self._workflow.game.refresh()
        except Exception as exc:
            return WindowRebindResult(
                RebindStatus.FAILED,
                message=f"刷新游戏窗口失败：{exc}",
            )
        if window is not None:
            generation = getattr(self._workflow.game, "generation", 0)
            return WindowRebindResult(
                RebindStatus.REBOUND,
                generation=generation,
                message=f"已重新绑定游戏窗口：{window.title}",
            )
        if self._workflow.game.ambiguous:
            return WindowRebindResult(
                RebindStatus.MANUAL_REQUIRED,
                message="存在多个相似游戏窗口，请在主界面手动选择",
            )
        return WindowRebindResult(
            RebindStatus.NOT_FOUND,
            message="刷新窗口列表后仍未找到游戏窗口",
        )
