from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from bulletbot.domain.models import (
    PageObservation as TradingPageObservation,
    PageState,
)

if TYPE_CHECKING:
    from bulletbot.mail_storage.models import (
        PageObservation as MailPageObservation,
        PageType,
    )

from .catalog import format_page, page_descriptor, page_label
from .models import GamePageId, PageSurface, UnifiedPageObservation


TRADING_PAGE_MAP = MappingProxyType(
    {
        PageState.FAVORITES: GamePageId.FAVORITES,
        PageState.DETAIL: GamePageId.MARKET_DETAIL,
        PageState.WAREHOUSE: GamePageId.WAREHOUSE,
        PageState.WAREHOUSE_SELL_DIALOG: GamePageId.WAREHOUSE_SELL_DIALOG,
        PageState.MARKET_SELL: GamePageId.MARKET_SELL,
        PageState.MARKET_CLAIM_COMPLETE: GamePageId.MARKET_CLAIM_COMPLETE,
        PageState.LISTING_EDITOR: GamePageId.LISTING_EDITOR,
        PageState.MARKET_OTHER: GamePageId.MARKET_OTHER,
        PageState.UNKNOWN: GamePageId.UNKNOWN,
    }
)


def trading_page_id(page_state: PageState) -> GamePageId:
    return TRADING_PAGE_MAP[page_state]


def mail_page_id(page_type: PageType) -> GamePageId:
    return GamePageId(page_type.value)


def format_trading_page(page_state: PageState) -> str:
    return format_page(trading_page_id(page_state))


def format_mail_page(page_type: PageType) -> str:
    return format_page(mail_page_id(page_type))


def adapt_trading_observation(
    observation: TradingPageObservation,
    *,
    stable: bool = False,
    captured_at: float = 0.0,
    window_generation: int = 0,
) -> UnifiedPageObservation:
    page_id = trading_page_id(observation.state)
    return UnifiedPageObservation(
        page_id=page_id,
        confidence=observation.confidence,
        chinese_name=page_label(page_id),
        surface=PageSurface.GAME,
        evidence=observation.evidence,
        stable=stable,
        captured_at=captured_at,
        window_generation=window_generation,
        screenshot_path=observation.screenshot_path,
    )


def adapt_mail_observation(
    observation: MailPageObservation,
    *,
    target: str,
    stable: bool = False,
    window_generation: int = 0,
) -> UnifiedPageObservation:
    page_id = mail_page_id(observation.page_type)
    if target == "game":
        surface = PageSurface.GAME
    elif target == "launcher_settings":
        surface = PageSurface.LAUNCHER_SETTINGS
    elif page_id in {GamePageId.UNKNOWN, GamePageId.INVALID_FRAME}:
        surface = PageSurface.LAUNCHER
    else:
        surface = page_descriptor(page_id).default_surface
    evidence = [
        f"模板 {name} {match.score:.0%}"
        for name, match in sorted(observation.matches.items())
    ]
    if observation.ocr_texts:
        evidence.append(f"OCR 文本 {len(observation.ocr_texts)} 项")
    return UnifiedPageObservation(
        page_id=page_id,
        confidence=observation.confidence,
        chinese_name=page_label(page_id),
        surface=surface,
        evidence=tuple(evidence),
        stable=stable,
        captured_at=observation.timestamp,
        window_generation=window_generation,
    )
