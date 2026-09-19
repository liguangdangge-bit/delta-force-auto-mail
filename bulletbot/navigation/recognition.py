from __future__ import annotations

from dataclasses import replace

from .catalog import page_descriptor, page_label
from .models import (
    GamePageId,
    PageKind,
    PageSurface,
    UnifiedPageObservation,
)


_META_PAGES = {GamePageId.UNKNOWN, GamePageId.INVALID_FRAME}
_WEAK_PAGES = {*_META_PAGES, GamePageId.MARKET_OTHER}
_DECISIVE_CONFIDENCE = 0.90
_DECISIVE_MARGIN = 0.12
_CROSS_WORKFLOW_DECISIVE_CONFIDENCE = 0.95
_CROSS_WORKFLOW_DECISIVE_MARGIN = 0.06
_EXCLUSIVE_LONGBOW_PAGES = {
    GamePageId.MAP_LONGBOW_START,
    GamePageId.MAP_LONGBOW_DOWNLOAD,
}
_TRADING_PAGES = {
    GamePageId.FAVORITES,
    GamePageId.MARKET_DETAIL,
    GamePageId.WAREHOUSE,
    GamePageId.MARKET_SELL,
    GamePageId.LISTING_EDITOR,
    GamePageId.MARKET_OTHER,
}


def resolve_page_observations(
    *observations: UnifiedPageObservation,
) -> UnifiedPageObservation:
    """Resolve same-frame recognizers without guessing across conflicts."""

    if not observations:
        raise ValueError("至少需要一个页面观察")
    generations = {item.window_generation for item in observations}
    if len(generations) > 1:
        raise ValueError("不能合并不同窗口代次的页面观察")

    strong = [item for item in observations if item.page_id not in _WEAK_PAGES]
    known = strong or [
        item for item in observations if item.page_id not in _META_PAGES
    ]
    if not known:
        invalid = next(
            (
                item
                for item in observations
                if item.page_id == GamePageId.INVALID_FRAME
            ),
            None,
        )
        return invalid or max(observations, key=lambda item: item.confidence)

    page_ids = {item.page_id for item in known}
    if len(page_ids) == 1:
        return max(known, key=lambda item: item.confidence)

    ranked = sorted(known, key=lambda item: item.confidence, reverse=True)
    winner, runner_up = ranked[:2]
    crosses_workflows = (
        (winner.page_id in _TRADING_PAGES)
        != (runner_up.page_id in _TRADING_PAGES)
    )
    if (
        winner.page_id in _EXCLUSIVE_LONGBOW_PAGES
        and winner.confidence >= 0.99
        and runner_up.page_id == GamePageId.MARKET_DETAIL
        and runner_up.confidence <= 0.98
    ):
        # Longbow pages require two dedicated anchors in the Mail recognizer.
        # A trading detail result is a generic geometry heuristic, so it must
        # not veto those page-specific anchors when both inspect one frame.
        return replace(
            winner,
            evidence=winner.evidence
            + (
                f"统一上下文门控：{winner.chinese_name}专用锚点覆盖"
                f" {runner_up.chinese_name}通用结构命中",
            ),
        )
    if (
        winner.confidence >= _DECISIVE_CONFIDENCE
        and winner.confidence - runner_up.confidence >= _DECISIVE_MARGIN
    ) or (
        crosses_workflows
        and winner.confidence >= _CROSS_WORKFLOW_DECISIVE_CONFIDENCE
        and winner.confidence - runner_up.confidence
        >= _CROSS_WORKFLOW_DECISIVE_MARGIN
    ):
        return replace(
            winner,
            evidence=winner.evidence
            + (
                f"统一仲裁：{winner.chinese_name} {winner.confidence:.0%} 明显高于"
                f" {runner_up.chinese_name} {runner_up.confidence:.0%}",
            ),
        )

    modals = [
        item
        for item in known
        if page_descriptor(item.page_id).kind == PageKind.MODAL
    ]
    if modals:
        return max(modals, key=lambda item: item.confidence)
    evidence = tuple(
        f"识别冲突：{item.chinese_name} {item.confidence:.0%}"
        for item in known
    )
    return UnifiedPageObservation(
        page_id=GamePageId.UNKNOWN,
        confidence=0.0,
        chinese_name=page_label(GamePageId.UNKNOWN),
        surface=PageSurface.GAME,
        evidence=evidence,
        stable=False,
        captured_at=max(item.captured_at for item in observations),
        window_generation=observations[0].window_generation,
    )


class PageStabilityTracker:
    def __init__(self, required_observations: int = 1) -> None:
        if required_observations < 1:
            raise ValueError("页面稳定确认次数必须至少为 1")
        self._required = required_observations
        self._candidate: tuple[PageSurface, GamePageId, int] | None = None
        self._count = 0

    def reset(self) -> None:
        self._candidate = None
        self._count = 0

    def update(
        self,
        observation: UnifiedPageObservation,
    ) -> UnifiedPageObservation:
        descriptor = page_descriptor(observation.page_id)
        if observation.page_id in _META_PAGES or descriptor.kind == PageKind.TRANSITION:
            self.reset()
            return replace(observation, stable=False)
        candidate = (
            observation.surface,
            observation.page_id,
            observation.window_generation,
        )
        if candidate == self._candidate:
            self._count += 1
        else:
            self._candidate = candidate
            self._count = 1
        return replace(
            observation,
            stable=self._count >= self._required,
        )
