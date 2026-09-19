from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Protocol
import unicodedata

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import InventoryMatch, Rect
from bulletbot.loadout_purchase.ammunition_catalog import (
    ammunition_definition,
    market_ammunition_name,
)
from bulletbot.mail_storage.models import OcrTextRegion
from bulletbot.mail_storage.vision import OcrEngine
from bulletbot.ocr.numbers import parse_ocr_integer
from bulletbot.vision.reference_layout import ReferenceLayout
from bulletbot.vision.warehouse_inventory import WarehouseInventoryDetector


class _RegionOcr(Protocol):
    def detect(self, image: np.ndarray) -> list[OcrTextRegion]: ...


@dataclass(frozen=True)
class WarehouseTextMatch:
    text: str
    confidence: float
    bounds: Rect
    price: int | None = None


@dataclass(frozen=True)
class WarehouseSellDialogSnapshot:
    title: WarehouseTextMatch | None
    military_price_label: WarehouseTextMatch | None
    military_sell_action: WarehouseTextMatch | None
    market_price_label: WarehouseTextMatch | None
    market_list_action: WarehouseTextMatch | None

    @property
    def confirmed(self) -> bool:
        return all(
            value is not None
            for value in (
                self.title,
                self.military_price_label,
                self.military_sell_action,
                self.market_price_label,
                self.market_list_action,
            )
        )


class WarehouseOcr:
    """Read the warehouse hover, context menu, and sell-choice dialog."""

    _SELL_DIALOG_CROP = (0.160, 0.185, 0.840, 0.770)
    _INVENTORY_NAME_MATCH_THRESHOLD = 0.68
    _INVENTORY_OCR_SCALE = 2
    _CONTEXT_MENU_WIDTH_CELLS = 1.85
    _CONTEXT_MENU_ROW_HEIGHT_CELLS = 0.477
    _CONTEXT_MENU_ROWS = (6, 7)
    _CONTEXT_MENU_ANCHOR_OFFSETS = (-8, -4, 0, 4)
    _CONTEXT_MENU_DARK_LUMA = 42
    _CONTEXT_MENU_MIN_DARK_FRACTION = 0.72

    def __init__(self, engine: _RegionOcr | None = None) -> None:
        self._engine = engine or OcrEngine()

    def read_hover_name(
        self,
        image: Image.Image,
        expected_name: str,
        cell_bounds: Rect,
    ) -> WarehouseTextMatch | None:
        if not WarehouseInventoryDetector.is_supported_ammunition(expected_name):
            return None
        search = self._expanded_cell_region(
            cell_bounds,
            image,
            left_cells=2.7,
            right_cells=2.7,
            top_cells=0.8,
            bottom_cells=1.0,
        )
        expected = self.normalize_product_name(
            market_ammunition_name(expected_name)
        )
        candidates = self._detect_region(image, search)
        outside_cell = [
            candidate
            for candidate in candidates
            if not self._center_inside(candidate.bounds, cell_bounds)
        ]
        exact = [
            candidate
            for candidate in outside_cell
            if self.normalize_product_name(candidate.text) == expected
        ]
        exact.extend(
            self._joined_same_line_matches(
                outside_cell,
                expected,
                max_gap=max(8, round(cell_bounds.width * 0.22)),
            )
        )
        name_match = max(exact, key=lambda value: value.confidence, default=None)
        if name_match is None:
            return None

        definition = ammunition_definition(expected_name)
        if definition is None or (
            definition.market_price_min is None
            and definition.market_price_max is None
        ):
            return name_match

        price_matches: list[tuple[float, int]] = []
        for candidate in outside_cell:
            if not re.fullmatch(r"\s*[\d,.]+\s*", candidate.text):
                continue
            if candidate.bounds.top < name_match.bounds.top:
                continue
            if candidate.bounds.top > name_match.bounds.bottom + cell_bounds.height:
                continue
            if candidate.bounds.right < name_match.bounds.left - cell_bounds.width:
                continue
            if candidate.bounds.left > name_match.bounds.right + cell_bounds.width:
                continue
            price = parse_ocr_integer(candidate.text)
            if price is not None and definition.accepts_market_price(price):
                price_matches.append((candidate.confidence, price))
        if not price_matches:
            return None
        price_confidence, price = max(price_matches)
        return WarehouseTextMatch(
            text=name_match.text,
            confidence=min(name_match.confidence, price_confidence),
            bounds=name_match.bounds,
            price=price,
        )

    def find_inventory_name_candidate(
        self,
        image: Image.Image,
        expected_name: str,
        grid_bounds: Rect,
        *,
        box_index: int,
        columns: int = WarehouseInventoryDetector.INVENTORY_COLUMNS,
        excluded_cells: set[tuple[int, int, int]] | None = None,
    ) -> InventoryMatch | None:
        """Nominate a warehouse cell when the visual template did not match.

        Inventory labels are deliberately treated as a weak signal.  The
        caller must still verify the complete hover name before opening the
        context menu.
        """

        if (
            not WarehouseInventoryDetector.is_supported_ammunition(expected_name)
            or columns < 1
        ):
            return None
        crop = np.asarray(
            image.crop(
                (
                    grid_bounds.left,
                    grid_bounds.top,
                    grid_bounds.right,
                    grid_bounds.bottom,
                )
            ).convert("RGB")
        )
        if crop.size == 0:
            return None
        scale = self._INVENTORY_OCR_SCALE
        enlarged = cv2.resize(
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        cell_size = grid_bounds.width / columns
        scaled_cell_size = cell_size * scale
        grouped: dict[tuple[int, int], list[tuple[float, float, str, float]]] = (
            defaultdict(list)
        )
        for region in self._engine.detect(enlarged):
            text = region.text.strip()
            if not text:
                continue
            left, top, right, bottom = region.bounds
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            column = int(center_x // scaled_cell_size)
            row = int(center_y // scaled_cell_size)
            if not (0 <= column < columns) or row < 0:
                continue
            relative_y = (center_y - row * scaled_cell_size) / scaled_cell_size
            if relative_y > 0.58:
                # The bottom-right count changes as stacks are combined.  It
                # must not contribute to product identity.
                continue
            grouped[(row, column)].append(
                (center_y, center_x, text, float(region.confidence))
            )

        expected = self.normalize_product_name(expected_name)
        expected_caliber = self._caliber_prefix(expected)
        candidates: list[tuple[float, float, int, int, str]] = []
        for (row, column), fragments in grouped.items():
            if excluded_cells and (box_index, row, column) in excluded_cells:
                continue
            fragments.sort(key=lambda value: (value[0], value[1]))
            raw_name = "".join(value[2] for value in fragments)
            normalized = self.normalize_product_name(raw_name)
            if not normalized or self._caliber_prefix(normalized) != expected_caliber:
                continue
            similarity = SequenceMatcher(None, expected, normalized).ratio()
            confidence = sum(value[3] for value in fragments) / len(fragments)
            candidates.append(
                (similarity, confidence, row, column, raw_name)
            )
        if not candidates:
            return None
        similarity, confidence, row, column, raw_name = max(
            candidates,
            key=lambda value: (value[0], value[1], -value[2], -value[3]),
        )
        if similarity < self._INVENTORY_NAME_MATCH_THRESHOLD:
            return None

        left = round(grid_bounds.left + column * cell_size)
        top = round(grid_bounds.top + row * cell_size)
        right = round(grid_bounds.left + (column + 1) * cell_size)
        bottom = min(
            grid_bounds.bottom,
            round(grid_bounds.top + (row + 1) * cell_size),
        )
        return InventoryMatch(
            name=raw_name,
            confidence=min(similarity, confidence),
            bounds=Rect(left, top, max(1, right - left), max(1, bottom - top)),
            tab_index=box_index,
            row=row,
            column=column,
            source="warehouse_ocr_candidate",
        )

    def find_context_sell_action(
        self,
        image: Image.Image,
        cell_bounds: Rect,
    ) -> WarehouseTextMatch | None:
        current = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        cell_size = max(1, round((cell_bounds.width + cell_bounds.height) / 2))
        menu_width = max(1, round(cell_size * self._CONTEXT_MENU_WIDTH_CELLS))
        row_height = max(1, round(cell_size * self._CONTEXT_MENU_ROW_HEIGHT_CELLS))
        anchor_x = round(cell_bounds.left + cell_bounds.width / 2)
        anchor_y = round(cell_bounds.top + cell_bounds.height / 2)
        frame_height, frame_width = current.shape
        edge_width = max(3, round(cell_size * 0.10))
        candidates: list[tuple[float, float, int, Rect]] = []

        for rows in self._CONTEXT_MENU_ROWS:
            menu_height = row_height * rows
            for horizontal in (-1, 1):
                for vertical in (-1, 1):
                    for x_offset in self._CONTEXT_MENU_ANCHOR_OFFSETS:
                        for y_offset in self._CONTEXT_MENU_ANCHOR_OFFSETS:
                            left = (
                                anchor_x + x_offset
                                if horizontal > 0
                                else anchor_x - menu_width + x_offset
                            )
                            top = (
                                anchor_y + y_offset
                                if vertical > 0
                                else anchor_y - menu_height + y_offset
                            )
                            right = left + menu_width
                            bottom = top + menu_height
                            if (
                                left < 0
                                or top < 0
                                or right > frame_width
                                or bottom > frame_height
                            ):
                                continue

                            region = current[top:bottom, left:right]
                            dark_fraction = float(
                                np.mean(region <= self._CONTEXT_MENU_DARK_LUMA)
                            )
                            if dark_fraction < self._CONTEXT_MENU_MIN_DARK_FRACTION:
                                continue

                            if vertical > 0:
                                edge = current[
                                    bottom : min(frame_height, bottom + edge_width),
                                    left:right,
                                ]
                            else:
                                edge = current[
                                    max(0, top - edge_width) : top,
                                    left:right,
                                ]
                            edge_dark = (
                                float(np.mean(edge <= self._CONTEXT_MENU_DARK_LUMA))
                                if edge.size
                                else 0.0
                            )
                            boundary_score = max(0.0, dark_fraction - edge_dark)

                            score = dark_fraction + boundary_score
                            candidates.append(
                                (
                                    score,
                                    dark_fraction,
                                    rows,
                                    Rect(left, top, menu_width, menu_height),
                                )
                            )

        if not candidates:
            return None
        _, dark_fraction, rows, menu = max(candidates, key=lambda value: value[0])
        sell_row = rows - 4
        return WarehouseTextMatch(
            text="出售",
            confidence=dark_fraction,
            bounds=Rect(
                menu.left + round(menu.width * 0.20),
                menu.top + sell_row * row_height,
                max(1, round(menu.width * 0.60)),
                row_height,
            ),
        )

    def read_sell_dialog(self, image: Image.Image) -> WarehouseSellDialogSnapshot:
        crop = self._relative_top_rect(image, self._SELL_DIALOG_CROP)
        candidates = self._detect_region(image, crop)
        width, height = image.size

        title = self._best(
            candidates,
            lambda value: self.normalize_ui_text(value.text) == "出售"
            and value.bounds.top < height * 0.32
            and value.bounds.left < width * 0.42,
        )
        military_label = self._best(
            candidates,
            lambda value: "军需处回收价" in self.normalize_ui_text(value.text),
        )
        market_label = self._best(
            candidates,
            lambda value: "交易行税后价" in self.normalize_ui_text(value.text),
        )
        military_action = self._best(
            candidates,
            lambda value: self.normalize_ui_text(value.text) == "出售"
            and value.bounds.left > width * 0.60
            and military_label is not None
            and value.bounds.top > military_label.bounds.top
            and (market_label is None or value.bounds.bottom < market_label.bounds.top),
        )
        market_action = self._best(
            candidates,
            lambda value: self.normalize_ui_text(value.text) == "上架"
            and value.bounds.left > width * 0.60
            and market_label is not None
            and value.bounds.top > market_label.bounds.top,
        )
        return WarehouseSellDialogSnapshot(
            title=title,
            military_price_label=military_label,
            military_sell_action=military_action,
            market_price_label=market_label,
            market_list_action=market_action,
        )

    def _detect_region(
        self,
        image: Image.Image,
        bounds: Rect,
    ) -> list[WarehouseTextMatch]:
        crop = np.asarray(
            image.crop(
                (bounds.left, bounds.top, bounds.right, bounds.bottom)
            ).convert("RGB")
        )
        if crop.size == 0:
            return []
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        matches: list[WarehouseTextMatch] = []
        for region in self._engine.detect(bgr):
            left, top, right, bottom = region.bounds
            matches.append(
                WarehouseTextMatch(
                    text=region.text,
                    confidence=region.confidence,
                    bounds=Rect(
                        bounds.left + left,
                        bounds.top + top,
                        max(1, right - left),
                        max(1, bottom - top),
                    ),
                )
            )
        return matches

    @staticmethod
    def normalize_product_name(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        normalized = normalized.replace("×", "x").replace("*", "x")
        normalized = normalized.replace("mm", "")
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", normalized)
        ply_match = re.search(r"ply([il]+)$", normalized)
        if ply_match is not None:
            normalized = (
                normalized[: ply_match.start(1)]
                + ply_match.group(1).replace("l", "i")
            )
        return normalized

    @staticmethod
    def normalize_ui_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", normalized)

    @staticmethod
    def _caliber_prefix(normalized: str) -> str | None:
        # Decimal calibers lose their dot during normalization, while shotgun
        # names may only start with a gauge number.  A leading numeric family
        # is sufficient here because hover OCR performs the exact final check.
        match = re.match(r"\d+(?:x\d+)*", normalized)
        return match.group(0) if match else None

    @staticmethod
    def _best(
        candidates: list[WarehouseTextMatch],
        predicate,
    ) -> WarehouseTextMatch | None:
        return max(
            (candidate for candidate in candidates if predicate(candidate)),
            key=lambda value: value.confidence,
            default=None,
        )

    @classmethod
    def _joined_same_line_matches(
        cls,
        candidates: list[WarehouseTextMatch],
        expected: str,
        *,
        max_gap: int,
    ) -> list[WarehouseTextMatch]:
        """Join OCR fragments that belong to one hover-title line.

        PaddleOCR occasionally returns decimal ammunition names as separate
        ``.50 AE`` and ``JHP`` regions.  Inventory labels use two physical
        rows, so strict vertical alignment keeps those labels from being
        mistaken for the single-line hover title.
        """

        ordered = sorted(
            candidates,
            key=lambda value: (value.bounds.top, value.bounds.left),
        )
        matches: list[WarehouseTextMatch] = []
        for start, first in enumerate(ordered):
            fragments = [first]
            for candidate in sorted(
                ordered[start + 1 :],
                key=lambda value: value.bounds.left,
            ):
                previous = fragments[-1]
                if candidate.bounds.left < previous.bounds.left:
                    continue
                center_delta = abs(
                    (candidate.bounds.top + candidate.bounds.height / 2)
                    - (first.bounds.top + first.bounds.height / 2)
                )
                alignment_limit = max(
                    3.0,
                    min(candidate.bounds.height, first.bounds.height) * 0.35,
                )
                if center_delta > alignment_limit:
                    continue
                gap = candidate.bounds.left - previous.bounds.right
                if gap < -max_gap or gap > max_gap:
                    continue
                fragments.append(candidate)
                joined_text = "".join(fragment.text for fragment in fragments)
                if cls.normalize_product_name(joined_text) != expected:
                    continue
                left = min(fragment.bounds.left for fragment in fragments)
                top = min(fragment.bounds.top for fragment in fragments)
                right = max(fragment.bounds.right for fragment in fragments)
                bottom = max(fragment.bounds.bottom for fragment in fragments)
                matches.append(
                    WarehouseTextMatch(
                        text=joined_text,
                        confidence=sum(
                            fragment.confidence for fragment in fragments
                        )
                        / len(fragments),
                        bounds=Rect(left, top, right - left, bottom - top),
                    )
                )
        return matches

    @staticmethod
    def _center_inside(inner: Rect, outer: Rect) -> bool:
        center_x = inner.left + inner.width / 2
        center_y = inner.top + inner.height / 2
        return (
            outer.left <= center_x <= outer.right
            and outer.top <= center_y <= outer.bottom
        )

    @staticmethod
    def _expanded_cell_region(
        cell: Rect,
        image: Image.Image,
        *,
        left_cells: float,
        right_cells: float,
        top_cells: float,
        bottom_cells: float,
    ) -> Rect:
        left = max(0, round(cell.left - cell.width * left_cells))
        top = max(0, round(cell.top - cell.height * top_cells))
        right = min(image.width, round(cell.right + cell.width * right_cells))
        bottom = min(image.height, round(cell.bottom + cell.height * bottom_cells))
        return Rect(left, top, max(1, right - left), max(1, bottom - top))

    @staticmethod
    def _relative_top_rect(
        image: Image.Image,
        bounds: tuple[float, float, float, float],
    ) -> Rect:
        left, top, right, bottom = ReferenceLayout(
            image.width,
            image.height,
        ).top_box(bounds)
        return Rect(left, top, right - left, bottom - top)
