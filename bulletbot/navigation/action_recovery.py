from __future__ import annotations

from collections.abc import Callable
from enum import Enum


RETRY_DELAYS_SECONDS = (3.0, 10.0, 60.0, 60.0)


class ActionState(Enum):
    TARGET = "target"
    SOURCE = "source"
    OTHER = "other"
    UNKNOWN = "unknown"


class PageActionStalled(RuntimeError):
    """A reversible action exhausted its observation and retry budget."""


class PageRecoveryFailed(RuntimeError):
    """Terminal recovery failure; callers must not restart their retry loop."""


class PageRecoveryStopped(RuntimeError):
    pass


def navigation_recovery_exhausted(result) -> bool:
    """Only terminal game-navigation failures qualify, never stop/ambiguity."""
    from .models import GamePageId, NavigationStatus

    return result.status in {NavigationStatus.FAILED, NavigationStatus.RETRYABLE} and (
        result.current_page in {None, GamePageId.UNKNOWN, GamePageId.INVALID_FRAME}
        or any(entry.event == "action_timeout" for entry in result.history)
    )


def recover_page_action(
    *,
    observe: Callable[[], ActionState],
    retry: Callable[[], None],
    wait: Callable[[float], None],
    stopped: Callable[[], bool],
    log: Callable[[str], None],
    label: str,
    delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
    settle_seconds: float = 3.0,
) -> ActionState:
    """Observe throughout each delay and re-observe immediately before input.

    Unknown/transition frames consume bounded waiting time, never authorize a
    click. The caller owns full-frame classification and fresh target location.
    """
    def sample() -> ActionState:
        if stopped():
            raise PageRecoveryStopped(label)
        return observe()

    def observe_for(seconds: float) -> ActionState:
        remaining = seconds
        while True:
            state = sample()
            if state in {ActionState.TARGET, ActionState.OTHER} or remaining <= 0:
                return state
            step = min(0.5, remaining)
            wait(step)
            remaining -= step

    state = sample()
    for index, delay in enumerate(delays, 1):
        if state in {ActionState.TARGET, ActionState.OTHER}:
            return state
        log(f"{label}未确认生效；观察 {delay:g} 秒后判断是否补点（{index}/{len(delays)}）")
        state = observe_for(delay)
        if state in {ActionState.TARGET, ActionState.OTHER}:
            return state
        if state is ActionState.SOURCE:
            if stopped():
                raise PageRecoveryStopped(label)
            log(f"{label}仍在原页面，重新定位后补点（{index}/{len(delays)}）")
            retry()
        else:
            log(f"{label}当前为未知或过渡页面，本次不补点")
        state = sample()
    state = observe_for(settle_seconds)
    if state in {ActionState.TARGET, ActionState.OTHER}:
        return state
    raise PageActionStalled(f"{label}经过 3/10/60/60 秒观察与有限补点后仍未推进")
