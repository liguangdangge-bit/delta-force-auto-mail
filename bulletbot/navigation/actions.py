from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from .models import NavigationAction, UnifiedPageObservation


class NavigationActionError(RuntimeError):
    pass


class NavigationActionExecutor(Protocol):
    def execute(
        self,
        action: NavigationAction,
        observation: UnifiedPageObservation,
    ) -> None: ...


NavigationActionHandler = Callable[[UnifiedPageObservation], None]


class MappedNavigationActionExecutor:
    """Dispatch only explicitly registered, page-neutral navigation actions."""

    def __init__(
        self,
        handlers: Mapping[NavigationAction, NavigationActionHandler],
    ) -> None:
        self._handlers = dict(handlers)

    @property
    def supported_actions(self) -> frozenset[NavigationAction]:
        return frozenset(self._handlers)

    def execute(
        self,
        action: NavigationAction,
        observation: UnifiedPageObservation,
    ) -> None:
        handler = self._handlers.get(action)
        if handler is None:
            raise NavigationActionError(
                f"导航动作 {action.value} 没有登记执行器，已拒绝发送输入"
            )
        try:
            handler(observation)
        except NavigationActionError:
            raise
        except Exception as exc:
            raise NavigationActionError(
                f"导航动作 {action.value} 执行失败：{exc}"
            ) from exc
