from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import Mapping
import unicodedata

from bulletbot.domain.models import CardRegion, MarketObservation
from bulletbot.trading.models import TradingCandidate, TradingRule


@dataclass(frozen=True)
class FuzzyFavoriteMatch:
    rule: TradingRule
    observation: MarketObservation


@dataclass(frozen=True)
class FavoritesMatchResult:
    observations_by_name: Mapping[str, MarketObservation]
    fuzzy_matches: tuple[FuzzyFavoriteMatch, ...] = ()
    missing_rules: tuple[TradingRule, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing_rules


def normalize_product_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("×", "x")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", normalized)


def is_single_character_ocr_variant(expected: str, actual: str) -> bool:
    if abs(len(expected) - len(actual)) != 1:
        return False
    if actual.startswith(expected) or expected.startswith(actual):
        return True

    # A leading dot on calibers such as ".50 AE" is occasionally read as an
    # extra L/I. Only accept it when the remaining text is an exact match.
    if (
        len(actual) == len(expected) + 1
        and expected[:1].isdigit()
        and actual[:1] in {"l", "i"}
    ):
        return actual[1:] == expected
    return False


def product_name_similarity(expected: str, actual: str) -> float:
    expected = normalize_product_name(expected)
    actual = normalize_product_name(actual)
    if not expected or not actual:
        return 0.0
    return SequenceMatcher(
        None,
        expected,
        actual,
        autojunk=False,
    ).ratio()


def is_fuzzy_product_name_match(
    expected: str,
    actual: str,
    known_product_names: tuple[str, ...] = (),
) -> bool:
    """Accept OCR damage only when the expected product is the clear best match."""

    expected = normalize_product_name(expected)
    actual = normalize_product_name(actual)
    if not expected or len(actual) < 4:
        return False
    if actual == expected:
        return True

    expected_score = product_name_similarity(expected, actual)
    legacy_variant = is_single_character_ocr_variant(expected, actual)
    if not legacy_variant and expected_score < 0.60:
        return False

    alternative_scores = [
        product_name_similarity(candidate, actual)
        for candidate in {
            normalize_product_name(name)
            for name in known_product_names
        }
        if candidate and candidate != expected
    ]
    best_alternative = max(alternative_scores, default=0.0)
    return expected_score >= best_alternative + 0.12


def match_favorite_observations(
    observations: tuple[MarketObservation, ...],
    rules: tuple[TradingRule, ...],
) -> FavoritesMatchResult:
    named_observations = [
        (normalize_product_name(observation.name.text), observation)
        for observation in observations
        if observation.name is not None
    ]
    matched: dict[str, MarketObservation] = {}
    fuzzy_matches: list[FuzzyFavoriteMatch] = []
    used_cards: set[int] = set()

    # Reserve exact matches first so tolerance cannot steal a card from a
    # similar enabled product.
    for rule in rules:
        exact = next(
            (
                observation
                for normalized_name, observation in named_observations
                if normalized_name == rule.normalized_name
                and observation.card.index not in used_cards
            ),
            None,
        )
        if exact is not None:
            matched[rule.normalized_name] = exact
            used_cards.add(exact.card.index)

    for rule in rules:
        if rule.normalized_name in matched:
            continue
        candidates = [
            observation
            for normalized_name, observation in named_observations
            if observation.card.index not in used_cards
            and is_single_character_ocr_variant(
                rule.normalized_name,
                normalized_name,
            )
        ]
        if len(candidates) != 1:
            continue
        observation = candidates[0]
        matched[rule.normalized_name] = observation
        fuzzy_matches.append(FuzzyFavoriteMatch(rule, observation))
        used_cards.add(observation.card.index)

    missing_rules = tuple(rule for rule in rules if rule.normalized_name not in matched)
    return FavoritesMatchResult(
        observations_by_name=matched,
        fuzzy_matches=tuple(fuzzy_matches),
        missing_rules=missing_rules,
    )


@dataclass
class FavoritesScanSession:
    rules: tuple[TradingRule, ...] = ()
    observations: tuple[MarketObservation, ...] = ()
    candidates: tuple[TradingCandidate, ...] = ()
    index: int = 0
    cycle: int = 0
    retry_count: int = 0
    card_cache: dict[str, CardRegion] = field(default_factory=dict)

    def reset(self, rules: tuple[TradingRule, ...] = ()) -> None:
        self.rules = tuple(rules)
        self.observations = ()
        self.candidates = ()
        self.index = 0
        self.cycle = 0
        self.retry_count = 0
        self.card_cache.clear()

    def clear_candidates(self) -> None:
        self.observations = ()
        self.candidates = ()
        self.index = 0
        self.card_cache.clear()

    def retry_available(self, max_retries: int) -> bool:
        if self.retry_count >= max_retries:
            return False
        self.retry_count += 1
        return True

    def accept_scan(
        self,
        observations: tuple[MarketObservation, ...],
    ) -> FavoritesMatchResult:
        result = match_favorite_observations(observations, self.rules)
        if not result.complete:
            return result
        self.observations = tuple(observations)
        self.candidates = tuple(
            TradingCandidate(rule=rule, observation=result.observations_by_name[rule.normalized_name])
            for rule in self.rules
        )
        self.card_cache = {
            candidate.rule.normalized_name: candidate.observation.card
            for candidate in self.candidates
        }
        self.index = 0
        self.cycle += 1
        self.retry_count = 0
        return result

    def refresh_current_candidate(
        self,
        observation: MarketObservation,
    ) -> TradingCandidate | None:
        if not self.candidates or self.index >= len(self.candidates):
            return None
        current = self.candidates[self.index]
        refreshed = TradingCandidate(rule=current.rule, observation=observation)
        candidates = list(self.candidates)
        candidates[self.index] = refreshed
        self.candidates = tuple(candidates)
        self.observations = tuple(
            observation if item.card.index == current.observation.card.index else item
            for item in self.observations
        )
        self.card_cache[current.rule.normalized_name] = observation.card
        self.cycle += 1
        return refreshed

    def advance(self) -> None:
        self.index += 1

    def begin_cached_cycle(self) -> bool:
        """Reuse the established rule-to-card mapping without another scan."""

        if not self.candidates or len(self.card_cache) != len(self.rules):
            return False
        self.index = 0
        self.cycle += 1
        self.retry_count = 0
        return True

    def card_for(
        self,
        product_key: str,
        fallback: CardRegion | None = None,
    ) -> CardRegion | None:
        return self.card_cache.get(product_key, fallback)
