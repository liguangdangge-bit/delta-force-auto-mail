from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from .catalog import PAGE_CATALOG
from .models import (
    GamePageId,
    NavigationAction,
    NavigationEdge,
    NavigationRisk,
)


class NavigationGraphError(ValueError):
    pass


class NavigationGraph:
    """Directed graph containing only registered, reversible page actions."""

    def __init__(self, edges: Iterable[NavigationEdge] = ()) -> None:
        self._edges: list[NavigationEdge] = []
        self._outgoing: dict[GamePageId, list[NavigationEdge]] = {}
        for edge in edges:
            self.add(edge)

    @property
    def edges(self) -> tuple[NavigationEdge, ...]:
        return tuple(self._edges)

    def outgoing(self, page_id: GamePageId) -> tuple[NavigationEdge, ...]:
        return tuple(self._outgoing.get(page_id, ()))

    def add(self, edge: NavigationEdge) -> None:
        if edge.source not in PAGE_CATALOG or edge.target not in PAGE_CATALOG:
            raise NavigationGraphError("导航边引用了未登记的页面")
        if edge.target not in edge.expected_pages:
            raise NavigationGraphError("导航边的预期页面必须包含目标页面")
        if edge.timeout_seconds <= 0 or edge.settle_seconds < 0:
            raise NavigationGraphError("导航等待时间无效")
        duplicate = any(
            existing.source == edge.source
            and existing.target == edge.target
            and existing.action == edge.action
            for existing in self._edges
        )
        if duplicate:
            raise NavigationGraphError("重复的导航边")
        self._edges.append(edge)
        self._outgoing.setdefault(edge.source, []).append(edge)

    def find_path(
        self,
        source: GamePageId,
        target: GamePageId,
        *,
        allowed_risks: frozenset[NavigationRisk] = frozenset(
            {NavigationRisk.SAFE}
        ),
        max_edges: int = 12,
    ) -> tuple[NavigationEdge, ...] | None:
        if source == target:
            return ()
        if max_edges < 1:
            return None
        queue = deque([(source, ())])
        visited = {source}
        while queue:
            page_id, path = queue.popleft()
            if len(path) >= max_edges:
                continue
            for edge in self._outgoing.get(page_id, ()):
                if edge.risk not in allowed_risks:
                    continue
                next_path = (*path, edge)
                if edge.target == target:
                    return next_path
                if edge.target in visited:
                    continue
                visited.add(edge.target)
                queue.append((edge.target, next_path))
        return None

    def next_edge(
        self,
        source: GamePageId,
        target: GamePageId,
        *,
        allowed_risks: frozenset[NavigationRisk] = frozenset(
            {NavigationRisk.SAFE}
        ),
        max_edges: int = 12,
    ) -> NavigationEdge | None:
        path = self.find_path(
            source,
            target,
            allowed_risks=allowed_risks,
            max_edges=max_edges,
        )
        return path[0] if path else None

    def find_path_to_any(
        self,
        source: GamePageId,
        targets: frozenset[GamePageId],
        *,
        allowed_risks: frozenset[NavigationRisk] = frozenset(
            {NavigationRisk.SAFE}
        ),
        max_edges: int = 12,
    ) -> tuple[NavigationEdge, ...] | None:
        if source in targets:
            return ()
        paths = (
            path
            for target in targets
            if (
                path := self.find_path(
                    source,
                    target,
                    allowed_risks=allowed_risks,
                    max_edges=max_edges,
                )
            )
            is not None
        )
        return min(
            paths,
            key=lambda path: (
                len(path),
                tuple(edge.target.value for edge in path),
            ),
            default=None,
        )

    def next_edge_to_any(
        self,
        source: GamePageId,
        targets: frozenset[GamePageId],
        *,
        allowed_risks: frozenset[NavigationRisk] = frozenset(
            {NavigationRisk.SAFE}
        ),
        max_edges: int = 12,
    ) -> NavigationEdge | None:
        path = self.find_path_to_any(
            source,
            targets,
            allowed_risks=allowed_risks,
            max_edges=max_edges,
        )
        return path[0] if path else None


def _edge(
    source: GamePageId,
    target: GamePageId,
    action: NavigationAction,
    *,
    expected: Iterable[GamePageId] = (),
    repeatable: bool = True,
    timeout: float = 5.0,
    settle: float = 0.35,
    risk: NavigationRisk = NavigationRisk.SAFE,
) -> NavigationEdge:
    return NavigationEdge(
        source=source,
        target=target,
        action=action,
        expected_pages=frozenset({target, *expected}),
        repeatable=repeatable,
        timeout_seconds=timeout,
        settle_seconds=settle,
        risk=risk,
    )


_HOME_PAGES = (GamePageId.GAME_HOME_PREPARE, GamePageId.GAME_HOME_READY)
_MAP_DETAIL_PAGES = (
    GamePageId.MAP_OVERVIEW,
    GamePageId.MAP_OTHER,
    GamePageId.MAP_ZERO_DAM_START,
    GamePageId.MAP_LONGBOW_START,
    GamePageId.MAP_LONGBOW_DOWNLOAD,
)
_MAIL_ESCAPE_PAGES = (
    GamePageId.MAIL_SCHEME_SELECTED,
    GamePageId.LOADOUT_SCHEMES,
    GamePageId.LOADOUT,
    GamePageId.MAP_ZERO_DAM_START,
    GamePageId.MAP_SELECTION,
    GamePageId.MAP_OVERVIEW,
    GamePageId.MAP_OTHER,
)


DEFAULT_NAVIGATION_EDGES = (
    _edge(
        GamePageId.LOADOUT_PURCHASE_CONFIRM,
        GamePageId.LOADOUT_SCHEMES,
        NavigationAction.ESCAPE,
        expected=(GamePageId.MAIL_SCHEME_SELECTED,),
    ),
    _edge(
        GamePageId.LOADOUT_PRICE_CHANGE,
        GamePageId.LOADOUT_SCHEMES,
        NavigationAction.ESCAPE,
        expected=(GamePageId.MAIL_SCHEME_SELECTED,),
    ),
    _edge(
        GamePageId.MARKET_DETAIL,
        GamePageId.FAVORITES,
        NavigationAction.ESCAPE,
    ),
    _edge(
        GamePageId.WAREHOUSE_SELL_DIALOG,
        GamePageId.WAREHOUSE,
        NavigationAction.ESCAPE,
    ),
    _edge(
        GamePageId.LISTING_EDITOR,
        GamePageId.MARKET_SELL,
        NavigationAction.ESCAPE,
        expected=(GamePageId.WAREHOUSE_SELL_DIALOG,),
    ),
    _edge(
        GamePageId.MARKET_CLAIM_COMPLETE,
        GamePageId.MARKET_SELL,
        NavigationAction.SPACE,
    ),
    _edge(
        GamePageId.MARKET_SELL,
        GamePageId.FAVORITES,
        NavigationAction.OPEN_MARKET_BUY,
    ),
    _edge(
        GamePageId.FAVORITES,
        GamePageId.MARKET_SELL,
        NavigationAction.OPEN_MARKET_SELL,
    ),
    _edge(
        GamePageId.MARKET_OTHER,
        GamePageId.MARKET_SELL,
        NavigationAction.OPEN_MARKET_SELL,
    ),
    _edge(
        GamePageId.MARKET_SELL,
        GamePageId.GAME_HOME_PREPARE,
        NavigationAction.OPEN_GAME_HOME,
        expected=(GamePageId.GAME_HOME_READY,),
    ),
    _edge(
        GamePageId.MARKET_OTHER,
        GamePageId.FAVORITES,
        NavigationAction.OPEN_MARKET_BUY,
    ),
    _edge(
        GamePageId.MARKET_OTHER,
        GamePageId.GAME_HOME_PREPARE,
        NavigationAction.OPEN_GAME_HOME,
        expected=(GamePageId.GAME_HOME_READY,),
    ),
    _edge(
        GamePageId.FAVORITES,
        GamePageId.WAREHOUSE,
        NavigationAction.OPEN_WAREHOUSE,
    ),
    _edge(
        GamePageId.WAREHOUSE,
        GamePageId.FAVORITES,
        NavigationAction.OPEN_MARKET,
        expected=(GamePageId.MARKET_OTHER,),
    ),
    _edge(
        GamePageId.FAVORITES,
        GamePageId.GAME_HOME_PREPARE,
        NavigationAction.OPEN_GAME_HOME,
        expected=(GamePageId.GAME_HOME_READY,),
    ),
    _edge(
        GamePageId.WAREHOUSE,
        GamePageId.GAME_HOME_PREPARE,
        NavigationAction.OPEN_GAME_HOME,
        expected=(GamePageId.GAME_HOME_READY,),
    ),
    _edge(
        GamePageId.GAME_HOME_PREPARE,
        GamePageId.FAVORITES,
        NavigationAction.OPEN_MARKET,
        expected=(GamePageId.MARKET_OTHER,),
    ),
    _edge(
        GamePageId.GAME_HOME_READY,
        GamePageId.FAVORITES,
        NavigationAction.OPEN_MARKET,
        expected=(GamePageId.MARKET_OTHER,),
    ),
    _edge(
        GamePageId.GAME_HOME_PREPARE,
        GamePageId.WAREHOUSE,
        NavigationAction.OPEN_WAREHOUSE,
    ),
    _edge(
        GamePageId.GAME_HOME_READY,
        GamePageId.WAREHOUSE,
        NavigationAction.OPEN_WAREHOUSE,
    ),
    _edge(
        GamePageId.MODE_MENU,
        GamePageId.GAME_HOME_PREPARE,
        NavigationAction.ESCAPE,
        expected=(GamePageId.GAME_HOME_READY,),
    ),
    _edge(
        GamePageId.MAIL_INBOX,
        GamePageId.GAME_HOME_PREPARE,
        NavigationAction.ESCAPE,
    ),
    _edge(
        GamePageId.MAIL_CLAIM_COMPLETE,
        GamePageId.MAIL_INBOX,
        NavigationAction.SPACE,
    ),
    _edge(
        GamePageId.ACTIVITY_REMINDER,
        GamePageId.GAME_HOME_PREPARE,
        NavigationAction.SPACE,
        expected=(GamePageId.GAME_TRANSITION,),
    ),
    _edge(
        GamePageId.GAME_HOME_PREPARE,
        GamePageId.MAP_OVERVIEW,
        NavigationAction.OPEN_GAME_MAP,
        expected=(
            GamePageId.MAP_SELECTION,
            GamePageId.MAP_OTHER,
            GamePageId.MAP_ZERO_DAM_START,
            GamePageId.MAP_LONGBOW_START,
            GamePageId.MAP_LONGBOW_DOWNLOAD,
        ),
    ),
    _edge(
        GamePageId.GAME_HOME_READY,
        GamePageId.MAP_OVERVIEW,
        NavigationAction.OPEN_GAME_MAP,
        expected=(
            GamePageId.MAP_SELECTION,
            GamePageId.MAP_OTHER,
            GamePageId.MAP_ZERO_DAM_START,
            GamePageId.MAP_LONGBOW_START,
            GamePageId.MAP_LONGBOW_DOWNLOAD,
        ),
    ),
    *(
        _edge(
            page_id,
            GamePageId.MAP_SELECTION,
            NavigationAction.EXPAND_MAP,
            expected=(GamePageId.MAP_LONGBOW_START, GamePageId.MAP_LONGBOW_DOWNLOAD),
        )
        for page_id in _MAP_DETAIL_PAGES
    ),
    _edge(
        GamePageId.MAP_SELECTION,
        GamePageId.MAP_LONGBOW_START,
        NavigationAction.SELECT_LONGBOW,
        expected=(GamePageId.MAP_LONGBOW_DOWNLOAD,),
        timeout=60.0,
    ),
    *(
        _edge(
            page_id,
            GamePageId.GAME_HOME_PREPARE,
            NavigationAction.ESCAPE,
            expected=(*_HOME_PAGES, *_MAIL_ESCAPE_PAGES),
        )
        for page_id in _MAIL_ESCAPE_PAGES
    ),
)


DEFAULT_NAVIGATION_GRAPH = NavigationGraph(DEFAULT_NAVIGATION_EDGES)
