from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import cv2
import numpy as np

from bulletbot.domain.models import Rect


@dataclass(frozen=True)
class InventoryTemplateMatch:
    confidence: float
    bounds: Rect
    row: int
    column: int


@dataclass(frozen=True)
class _TemplateImage:
    product_key: str
    image: np.ndarray


class InventoryTemplateMatcher:
    """Find a known inventory item by its cell image before falling back to OCR."""

    # Templates only nominate a cell. Existing listing-title validation and
    # the warehouse flow's hover-name check provide the final exact match.
    _MIN_SCORE = 0.82
    _MIN_SCORE_WITHOUT_VARIANT = 0.86
    # Cell backgrounds dominate normalized scores, so visually adjacent ammo
    # variants can differ by only a few thousandths. The winning-template
    # requirement plus the later full-name check remain the safety boundary.
    _MIN_MARGIN = 0.0025
    _LOW_SCORE_BOUNDARY = 0.88
    _LOW_SCORE_MIN_MARGIN = 0.025
    _CELL_SEARCH_MARGIN = 0.16
    # Stack counts are rendered in the bottom band of an inventory cell. Keep
    # only the stable label/icon region so a one-round sample remains useful
    # for live stacks while correlation still rejects background-only matches.
    _STABLE_FEATURE_HEIGHT_RATIO = 0.72

    def __init__(self, template_directory: Path | None = None) -> None:
        self._template_directory = template_directory or self.default_template_directory()
        self._templates: tuple[_TemplateImage, ...] | None = None

    @staticmethod
    def default_template_directory() -> Path:
        if getattr(sys, "frozen", False):
            root = Path(getattr(sys, "_MEIPASS"))
        else:
            root = Path(__file__).resolve().parents[2]
        return root / "assets" / "inventory_templates"

    def has_template_for(self, product_name: str) -> bool:
        expected_key = self.normalize_product_name(product_name)
        return any(template.product_key == expected_key for template in self._load_templates())

    def find_expected(
        self,
        image: np.ndarray,
        product_name: str,
        grid_bounds: Rect,
        columns: int,
        excluded_cells: set[tuple[int, int, int]] | None = None,
        tab_index: int = 0,
    ) -> InventoryTemplateMatch | None:
        templates = self._load_templates()
        expected_key = self.normalize_product_name(product_name)
        family = self._product_family(expected_key)
        family_templates = [
            template
            for template in templates
            if self._product_family(template.product_key) == family
        ]
        if not any(template.product_key == expected_key for template in family_templates):
            return None

        cell_size = grid_bounds.width / columns
        rows = max(1, int(grid_bounds.height // cell_size))
        candidates: list[tuple[float, int, int, Rect]] = []
        for row in range(rows):
            for column in range(columns):
                if excluded_cells and (tab_index, row, column) in excluded_cells:
                    continue
                bounds = self._cell_bounds(grid_bounds, cell_size, row, column)
                scores = self._scores_for_cell(image, bounds, family_templates)
                expected_score = scores.get(expected_key)
                if expected_score is None:
                    continue
                winner_key, winner_score = max(
                    scores.items(),
                    key=lambda item: item[1],
                )
                alternative_score = max(
                    (
                        score
                        for key, score in scores.items()
                        if key != winner_key
                    ),
                    default=None,
                )
                minimum_score = (
                    self._MIN_SCORE
                    if alternative_score is not None
                    else self._MIN_SCORE_WITHOUT_VARIANT
                )
                margin = (
                    winner_score - alternative_score
                    if alternative_score is not None
                    else 1.0
                )
                required_margin = (
                    self._LOW_SCORE_MIN_MARGIN
                    if winner_score < self._LOW_SCORE_BOUNDARY
                    else self._MIN_MARGIN
                )
                if (
                    winner_key == expected_key
                    and winner_score >= minimum_score
                    and margin >= required_margin
                ):
                    candidates.append((winner_score, row, column, bounds))
        if not candidates:
            return None
        confidence, row, column, bounds = max(
            candidates,
            key=lambda item: (item[0], -item[1], -item[2]),
        )
        return InventoryTemplateMatch(confidence, bounds, row, column)

    @classmethod
    def normalize_product_name(cls, value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value.casefold().replace("mm", ""))

    @staticmethod
    def _product_family(product_key: str) -> str:
        match = re.match(r"\d+x\d+", product_key)
        if match:
            return match.group(0)
        for prefix in (
            "300blackout",
            "357magnum",
            "45acp",
            "50ae",
            "12gauge",
            "4570govt",
        ):
            if product_key.startswith(prefix):
                return prefix
        if product_key.endswith("箭矢"):
            return "箭矢"
        return product_key

    @staticmethod
    def _cell_bounds(
        grid_bounds: Rect,
        cell_size: float,
        row: int,
        column: int,
    ) -> Rect:
        left = round(grid_bounds.left + column * cell_size)
        top = round(grid_bounds.top + row * cell_size)
        right = round(grid_bounds.left + (column + 1) * cell_size)
        bottom = min(
            grid_bounds.bottom,
            round(grid_bounds.top + (row + 1) * cell_size),
        )
        return Rect(left, top, right - left, bottom - top)

    def _scores_for_cell(
        self,
        image: np.ndarray,
        bounds: Rect,
        templates: list[_TemplateImage],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for template in templates:
            score = self._template_score(image, bounds, template.image)
            scores[template.product_key] = max(scores.get(template.product_key, -1.0), score)
        return scores

    def _template_score(
        self,
        image: np.ndarray,
        bounds: Rect,
        template: np.ndarray,
    ) -> float:
        margin_x = round(bounds.width * self._CELL_SEARCH_MARGIN)
        margin_y = round(bounds.height * self._CELL_SEARCH_MARGIN)
        left = max(0, bounds.left - margin_x)
        top = max(0, bounds.top - margin_y)
        right = min(image.shape[1], bounds.right + margin_x)
        bottom = min(image.shape[0], bounds.bottom + margin_y)
        search = image[top:bottom, left:right]
        if search.size == 0:
            return -1.0

        stable_height = max(
            8,
            round(template.shape[0] * self._STABLE_FEATURE_HEIGHT_RATIO),
        )
        stable_template = template[:stable_height, :]

        # A template captured at the active resolution is more precise than a
        # resized copy. Resize only when the inventory cell is materially
        # different from the template's native scale.
        scale_ratio = bounds.width / template.shape[1]
        if 0.90 <= scale_ratio <= 1.10:
            resized = stable_template
        else:
            target_width = max(1, round(bounds.width * 0.98))
            target_height = max(
                1,
                round(
                    stable_template.shape[0]
                    * target_width
                    / stable_template.shape[1]
                ),
            )
            resized = cv2.resize(
                stable_template,
                (target_width, target_height),
                interpolation=(
                    cv2.INTER_AREA
                    if target_width < stable_template.shape[1]
                    else cv2.INTER_CUBIC
                ),
            )
        if resized.shape[1] > search.shape[1] or resized.shape[0] > search.shape[0]:
            return -1.0
        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(
            search_gray,
            template_gray,
            cv2.TM_CCOEFF_NORMED,
        )
        finite = result[np.isfinite(result)]
        if finite.size == 0:
            return -1.0
        return float(finite.max())

    def _load_templates(self) -> list[_TemplateImage]:
        if self._templates is not None:
            return list(self._templates)
        if not self._template_directory.exists():
            return []
        templates: list[_TemplateImage] = []
        for path in sorted(self._template_directory.glob("*.png")):
            product_name = path.stem.split("__", 1)[0].replace("_", " ")
            product_key = self.normalize_product_name(product_name)
            try:
                image = cv2.imdecode(
                    np.fromfile(path, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            except OSError:
                image = None
            if product_key and image is not None and image.size:
                templates.append(_TemplateImage(product_key, image))
        self._templates = tuple(templates)
        return list(self._templates)
