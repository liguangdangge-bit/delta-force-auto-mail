from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from statistics import median
import unicodedata

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import (
    InventoryMatch,
    MarketDepthLevel,
    OcrText,
    Rect,
    SellListingSnapshot,
    SellOrderSnapshot,
    SellSlotsSnapshot,
)
from bulletbot.ocr.favorites_ocr import preload_onnx_runtime
from bulletbot.ocr.numeric_ocr import NumericOcrEngine
from bulletbot.ocr.numbers import parse_ocr_integer, parse_ocr_integer_pair
from bulletbot.ocr.preprocessing import ProductNamePreprocessor
from bulletbot.vision.inventory_templates import InventoryTemplateMatcher
from bulletbot.vision.reference_layout import ReferenceLayout


@dataclass(frozen=True)
class SellPriceDecision:
    price: int | None
    tick: int | None
    lowest_price: int | None
    main_price: int | None
    lowest_to_main_ratio: float | None
    reason: str


@dataclass(frozen=True)
class _PositionedOcrText:
    text: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


class SellOcr:
    """Read the market sell page and listing editor with local OCR crops."""

    _INVENTORY_GRID = (0.642, 0.164, 0.950, 0.912)
    _INVENTORY_COLUMNS = 9
    _INVENTORY_MATCH_THRESHOLD = 0.74
    _INVENTORY_FOCUSED_SCALE = 6
    _INVENTORY_FOCUSED_HEIGHT_RATIO = 0.52
    _INVENTORY_FOCUSED_MAX_CELLS = 12
    _INVENTORY_FOCUSED_CONFIRMATIONS = 2
    _DEPTH_CENTERS = (0.255, 0.322, 0.389, 0.456, 0.522)
    _DEPTH_VOLUME_BAND = (0.220, 0.340, 0.555, 0.665)
    _DEPTH_COLUMN_MAX_DISTANCE = 0.040
    _DEPTH_PRICE_HALF_WIDTH = 0.027
    _DEPTH_PRICE_VERTICAL_BOUNDS = (0.671, 0.704)
    _TOP_ORDER_QUANTITY_CROP = (0.260, 0.160, 0.295, 0.193)
    _TOP_ORDER_PRICE_CROP = (0.260, 0.272, 0.295, 0.302)
    # Include the label, coin icon, and value. The previous 217x45 crop only
    # covered the value and was too sensitive to small layout shifts.
    _LISTING_LOWEST_PRICE_CROP = (0.210, 0.285, 0.455, 0.345)
    _LISTING_EDITOR_PANEL_CROP = (0.185, 0.205, 0.815, 0.750)
    _LISTING_LOWEST_PRICE_FALLBACK_REGION = (0.205, 0.275, 0.460, 0.355)
    _LISTING_LOWEST_PRICE_LABELS = ("在售最低价", "最低价")
    _LISTING_PRICE_CROP = (0.645, 0.576, 0.715, 0.615)
    _LISTING_EXPECTED_INCOME_CROP = (0.675, 0.635, 0.728, 0.665)
    _NUMERIC_MIN_CONFIDENCE = 0.75
    # The listing title is a one-line label. Keep the crop tight so the item
    # preview and panel decorations do not become competing OCR detections.
    _LISTING_TITLE_CROP = (0.575, 0.285, 0.790, 0.330)
    _LISTING_EDITOR_HEADER_CROP = (0.170, 0.185, 0.350, 0.275)
    _LISTING_EDITOR_HEADER_ALIASES = ("上架物品", "上架道具")

    def __init__(self) -> None:
        self._engine = None
        self._numeric_engine = None
        self._template_matcher = InventoryTemplateMatcher()

    def read_slots(self, image: Image.Image) -> SellSlotsSnapshot:
        crop = self._top_crop(image, (0.035, 0.112, 0.180, 0.174), scale=3)
        candidates = self._recognize(crop)
        for candidate in sorted(candidates, key=lambda value: value.confidence, reverse=True):
            slot_pair = parse_ocr_integer_pair(candidate.text)
            if slot_pair is not None:
                return SellSlotsSnapshot(
                    used=slot_pair[0],
                    total=slot_pair[1],
                    confidence=candidate.confidence,
                )
        return SellSlotsSnapshot(None, None, None)

    def read_listing_editor_header(self, image: Image.Image) -> OcrText | None:
        """Read the large editor heading from a small, stable screen region."""
        crop = self._relative_crop(
            image,
            self._LISTING_EDITOR_HEADER_CROP,
            scale=3,
        )
        readings: list[OcrText] = []
        variants = (crop,) + ProductNamePreprocessor.fallback_variants(crop)
        for variant in variants:
            reading = self._recognize_single(variant)
            if reading is None:
                continue
            readings.append(reading)
            if self.is_listing_editor_header(reading.text):
                return reading
        return max(readings, key=lambda value: value.confidence, default=None)

    @classmethod
    def is_listing_editor_header(cls, text: str | None) -> bool:
        normalized = re.sub(r"\s+", "", str(text or ""))
        return any(alias in normalized for alias in cls._LISTING_EDITOR_HEADER_ALIASES)

    def inventory_grid_bounds(self, image: Image.Image) -> Rect:
        left, top, right, bottom = ReferenceLayout(
            *image.size
        ).stretched_box(self._INVENTORY_GRID)
        return Rect(left, top, right - left, bottom - top)

    def read_top_order(self, image: Image.Image) -> SellOrderSnapshot:
        name = self._best_text(
            self._recognize(
                self._top_crop(image, (0.045, 0.160, 0.245, 0.205), scale=3)
            ),
            require_digit=True,
        )
        quantity_text = self._read_numeric_crop(
            image,
            self._TOP_ORDER_QUANTITY_CROP,
            scale=4,
            top_anchored=True,
        )
        price_text = self._read_numeric_crop(
            image,
            self._TOP_ORDER_PRICE_CROP,
            scale=4,
            top_anchored=True,
        )
        return SellOrderSnapshot(
            product_name=name.text if name else None,
            quantity=self._parse_number(quantity_text.text) if quantity_text else None,
            unit_price=self._parse_number(price_text.text) if price_text else None,
            name_confidence=name.confidence if name else None,
            quantity_confidence=quantity_text.confidence if quantity_text else None,
            price_confidence=price_text.confidence if price_text else None,
        )

    def find_inventory_match(
        self,
        image: Image.Image,
        expected_name: str,
        tab_index: int,
        excluded_cells: set[tuple[int, int, int]] | None = None,
    ) -> InventoryMatch | None:
        width, _height = image.size
        grid_bounds = self.inventory_grid_bounds(image)
        left, top = grid_bounds.left, grid_bounds.top
        right, bottom = grid_bounds.right, grid_bounds.bottom
        has_visual_template = self._template_matcher.has_template_for(expected_name)
        template_match = self._template_matcher.find_expected(
            np.asarray(image.convert("RGB"))[:, :, ::-1],
            expected_name,
            grid_bounds,
            self._INVENTORY_COLUMNS,
            excluded_cells=excluded_cells,
            tab_index=tab_index,
        )
        if template_match is not None:
            return InventoryMatch(
                name=expected_name,
                confidence=template_match.confidence,
                bounds=template_match.bounds,
                tab_index=tab_index,
                row=template_match.row,
                column=template_match.column,
                source="template",
            )
        raw_crop = np.asarray(image.crop((left, top, right, bottom)))
        scale = 2
        crop = cv2.resize(raw_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result, _ = self._get_engine()(crop)
        cell_size = (right - left) / self._INVENTORY_COLUMNS
        scaled_cell_size = cell_size * scale
        grouped: dict[tuple[int, int], list[tuple[float, str, float]]] = defaultdict(list)
        for box, text, confidence in result or []:
            normalized = str(text).strip()
            if not normalized:
                continue
            center_x = sum(float(point[0]) for point in box) / len(box)
            center_y = sum(float(point[1]) for point in box) / len(box)
            column = int(center_x // scaled_cell_size)
            row = int(center_y // scaled_cell_size)
            if not (0 <= column < self._INVENTORY_COLUMNS) or row < 0:
                continue
            relative_y = (center_y - row * scaled_cell_size) / scaled_cell_size
            if relative_y > 0.58:
                continue
            grouped[(row, column)].append((center_y, normalized, float(confidence)))

        expected = self.normalize_product_name(expected_name)
        matches: list[tuple[float, float, int, int, str]] = []
        for (row, column), values in grouped.items():
            if excluded_cells and (tab_index, row, column) in excluded_cells:
                continue
            values.sort(key=lambda value: value[0])
            candidate_text = "".join(value[1] for value in values)
            candidate = self.normalize_product_name(candidate_text)
            if not candidate:
                continue
            score = self.inventory_name_similarity(expected, candidate)
            confidence = sum(value[2] for value in values) / len(values)
            matches.append((score, confidence, row, column, candidate_text))
        if not matches:
            return None

        score, confidence, row, column, name = max(
            matches,
            key=lambda value: (value[0], value[1], -value[2], -value[3]),
        )
        source = "ocr_template_fallback" if has_visual_template else "ocr"
        if score < self._INVENTORY_MATCH_THRESHOLD:
            focused_match = self._find_focused_inventory_match(
                image,
                expected_name,
                tab_index,
                left,
                top,
                bottom,
                cell_size,
                matches,
                excluded_cells,
            )
            if focused_match is None:
                return None
            row, column, name, confidence = focused_match
            score = self.inventory_name_similarity(expected, name)
            source = (
                "ocr_template_fallback_focused"
                if has_visual_template
                else "ocr_focused"
            )
        cell_left = round(left + column * cell_size)
        cell_top = round(top + row * cell_size)
        cell_right = round(left + (column + 1) * cell_size)
        cell_bottom = min(bottom, round(top + (row + 1) * cell_size))
        return InventoryMatch(
            name=name,
            confidence=min(confidence, score),
            bounds=Rect(cell_left, cell_top, cell_right - cell_left, cell_bottom - cell_top),
            tab_index=tab_index,
            row=row,
            column=column,
            source=source,
        )

    def _find_focused_inventory_match(
        self,
        image: Image.Image,
        expected_name: str,
        tab_index: int,
        left: int,
        top: int,
        bottom: int,
        cell_size: float,
        candidates: list[tuple[float, float, int, int, str]],
        excluded_cells: set[tuple[int, int, int]] | None,
    ) -> tuple[int, int, str, float] | None:
        expected = self.normalize_product_name(expected_name)
        caliber = self._inventory_caliber_prefix(expected)
        if not caliber:
            return None
        same_caliber = [
            candidate
            for candidate in candidates
            if self.normalize_product_name(candidate[4]).startswith(caliber)
            and not (
                excluded_cells
                and (tab_index, candidate[2], candidate[3]) in excluded_cells
            )
        ]
        same_caliber.sort(
            key=lambda value: (-value[0], -value[1], value[2], value[3])
        )
        for _score, _confidence, row, column, _name in same_caliber[
            : self._INVENTORY_FOCUSED_MAX_CELLS
        ]:
            cell_left = round(left + column * cell_size)
            cell_top = round(top + row * cell_size)
            cell_right = round(left + (column + 1) * cell_size)
            cell_bottom = min(bottom, round(top + (row + 1) * cell_size))
            label_bottom = min(
                cell_bottom,
                cell_top + round(cell_size * self._INVENTORY_FOCUSED_HEIGHT_RATIO),
            )
            label = np.asarray(
                image.crop((cell_left, cell_top, cell_right, label_bottom))
            )
            if label.size == 0:
                continue
            scaled = cv2.resize(
                label,
                None,
                fx=self._INVENTORY_FOCUSED_SCALE,
                fy=self._INVENTORY_FOCUSED_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )
            confirmations: list[tuple[float, str]] = []
            variants = (scaled,) + ProductNamePreprocessor.fallback_variants(scaled)
            for variant in variants:
                result, _ = self._get_engine()(variant)
                readings: list[tuple[float, float, str, float]] = []
                for box, text, confidence in result or []:
                    normalized = str(text).strip()
                    if not normalized:
                        continue
                    center_x = sum(float(point[0]) for point in box) / len(box)
                    center_y = sum(float(point[1]) for point in box) / len(box)
                    readings.append(
                        (center_y, center_x, normalized, float(confidence))
                    )
                readings.sort(key=lambda value: (value[0], value[1]))
                candidate_text = "".join(value[2] for value in readings)
                if not self._focused_inventory_name_matches(
                    expected_name, candidate_text
                ):
                    continue
                confidence = sum(value[3] for value in readings) / len(readings)
                confirmations.append((confidence, candidate_text))
            if len(confirmations) < self._INVENTORY_FOCUSED_CONFIRMATIONS:
                continue
            confidence, name = max(confirmations)
            return row, column, name, confidence
        return None

    def read_listing(
        self,
        image: Image.Image,
        screenshot_path: str | None = None,
        include_depth: bool = True,
        expected_name: str | None = None,
    ) -> SellListingSnapshot:
        product = self._read_listing_product_name(image, expected_name=expected_name)
        quantity, inventory_total, used_slots, available_slots = (
            self.read_listing_quantity_and_slots(image)
        )
        lowest_price = self.read_listing_lowest_price(image)
        listing_price = self.read_listing_price(image)
        expected_income = self.read_listing_expected_income(image)
        levels = (
            self._read_listing_depth_levels(image, include_volumes=True)
            if include_depth
            else ()
        )

        return SellListingSnapshot(
            product_name=product.text if product else None,
            product_confidence=product.confidence if product else None,
            quantity=quantity,
            inventory_total=inventory_total,
            used_slots=used_slots,
            available_slots=available_slots,
            lowest_price=lowest_price,
            depth_levels=levels,
            listing_price=listing_price,
            expected_income=expected_income,
            screenshot_path=screenshot_path,
        )

    def read_listing_identity(
        self,
        image: Image.Image,
        *,
        expected_name: str | None = None,
    ) -> tuple[str | None, float | None]:
        product = self._read_listing_product_name(image, expected_name=expected_name)
        return (
            product.text if product else None,
            product.confidence if product else None,
        )

    def read_listing_quantity_and_slots(
        self,
        image: Image.Image,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        controls = self._recognize(
            self._relative_crop(image, (0.550, 0.440, 0.810, 0.510), scale=2)
        )
        quantity = inventory_total = used_slots = available_slots = None
        fractions: list[tuple[int, int]] = []
        for candidate in controls:
            normalized = candidate.text.replace("：", ":")
            quantity_pair = parse_ocr_integer_pair(normalized)
            if quantity_pair is None:
                continue
            first, second = quantity_pair
            fractions.append((first, second))
            if "售位" in normalized:
                used_slots, available_slots = first, second
            elif "数量" in normalized:
                quantity, inventory_total = first, second
        if quantity is None and fractions:
            quantity, inventory_total = fractions[0]
        if used_slots is None and len(fractions) >= 2:
            used_slots, available_slots = fractions[1]
        return quantity, inventory_total, used_slots, available_slots

    def read_listing_lowest_price(self, image: Image.Image) -> int | None:
        candidates = self._recognize(
            self._relative_crop(
                image,
                self._LISTING_LOWEST_PRICE_CROP,
                scale=3,
            )
        )
        valid = [
            (candidate, value)
            for candidate in candidates
            if candidate.confidence >= self._NUMERIC_MIN_CONFIDENCE
            and (value := self._parse_number(candidate.text)) is not None
            and value > 0
            and len(str(value)) >= 2
        ]
        if not valid:
            return None
        return max(
            valid,
            key=lambda item: (len(str(item[1])), item[0].confidence),
        )[1]

    def read_listing_lowest_price_full_editor(self, image: Image.Image) -> int | None:
        """Find the lowest price in the full editor by label and geometry."""

        readings = self._recognize_positioned(
            image,
            self._LISTING_EDITOR_PANEL_CROP,
            scale=1,
        )
        labels = [
            reading
            for reading in readings
            if self._is_lowest_price_label(reading.text)
        ]
        for label in sorted(labels, key=lambda value: value.confidence, reverse=True):
            embedded = self._valid_positioned_integer(label)
            if embedded is not None:
                return embedded
            aligned: list[tuple[float, float, int]] = []
            row_tolerance = max(0.025, (label.bottom - label.top) * 1.5)
            for reading in readings:
                value = self._valid_positioned_integer(reading)
                if value is None:
                    continue
                if reading.left < label.right - 0.01:
                    continue
                if reading.left > label.right + 0.16:
                    continue
                if abs(reading.center_y - label.center_y) > row_tolerance:
                    continue
                aligned.append(
                    (
                        reading.confidence,
                        -abs(reading.left - label.right),
                        value,
                    )
                )
            if aligned:
                return max(aligned)[2]

        left, top, right, bottom = self._LISTING_LOWEST_PRICE_FALLBACK_REGION
        spatial = [
            (reading.confidence, value)
            for reading in readings
            if left <= reading.center_x <= right
            and top <= reading.center_y <= bottom
            and (value := self._valid_positioned_integer(reading)) is not None
        ]
        return max(spatial)[1] if spatial else None

    def read_listing_price(self, image: Image.Image) -> int | None:
        text = self._read_numeric_crop(
            image,
            self._LISTING_PRICE_CROP,
            scale=5,
        )
        return self._parse_number(text.text) if text else None

    def read_listing_expected_income(self, image: Image.Image) -> int | None:
        text = self._read_numeric_crop(
            image,
            self._LISTING_EXPECTED_INCOME_CROP,
            scale=4,
        )
        return self._parse_number(text.text) if text else None

    def read_listing_depth_prices(
        self,
        image: Image.Image,
    ) -> tuple[MarketDepthLevel, ...]:
        return self._read_listing_depth_levels(image, include_volumes=False)

    def _read_listing_depth_levels(
        self,
        image: Image.Image,
        *,
        include_volumes: bool,
    ) -> tuple[MarketDepthLevel, ...]:
        price_texts = self._read_depth_price_columns(image)
        volume_texts: dict[int, OcrText] = {}
        if include_volumes:
            volume_texts = self._read_depth_band(
                image,
                self._DEPTH_VOLUME_BAND,
                scale=2,
                minimum_digits=2,
            )
        levels: list[MarketDepthLevel] = []
        for column in range(len(self._DEPTH_CENTERS)):
            price_text = price_texts.get(column)
            volume_text = volume_texts.get(column)
            price = self._parse_number(price_text.text) if price_text else None
            if price is None:
                continue
            volume = self._parse_number(volume_text.text) if volume_text else None
            levels.append(MarketDepthLevel(price=price, volume=volume))
        return tuple(levels)

    def _read_depth_price_columns(self, image: Image.Image) -> dict[int, OcrText]:
        top, bottom = self._DEPTH_PRICE_VERTICAL_BOUNDS
        readings: dict[int, OcrText] = {}
        for column, center in enumerate(self._DEPTH_CENTERS):
            reading = self._read_numeric_crop(
                image,
                (
                    center - self._DEPTH_PRICE_HALF_WIDTH,
                    top,
                    center + self._DEPTH_PRICE_HALF_WIDTH,
                    bottom,
                ),
                scale=5,
            )
            if reading is None:
                continue
            value = parse_ocr_integer(reading.text)
            if value is not None and len(str(value)) >= 2:
                readings[column] = reading
        return readings

    def _read_depth_band(
        self,
        image: Image.Image,
        bounds: tuple[float, float, float, float],
        *,
        scale: int,
        minimum_digits: int,
    ) -> dict[int, OcrText]:
        crop = self._relative_crop(image, bounds, scale=scale)
        result, _ = self._get_engine()(crop)
        width, _height = image.size
        left_px = round(width * bounds[0])
        candidates: list[tuple[float, OcrText]] = []
        for box, text, confidence in result or []:
            normalized = str(text).strip()
            center_x = sum(float(point[0]) for point in box) / len(box)
            relative_x = (left_px + center_x / scale) / width
            candidates.append((relative_x, OcrText(normalized, float(confidence))))
        return self._assign_depth_texts(candidates, minimum_digits=minimum_digits)

    @classmethod
    def _assign_depth_texts(
        cls,
        candidates: list[tuple[float, OcrText]],
        *,
        minimum_digits: int = 1,
    ) -> dict[int, OcrText]:
        assigned: dict[int, OcrText] = {}
        for relative_x, candidate in candidates:
            value = parse_ocr_integer(candidate.text)
            if value is None or len(str(value)) < minimum_digits:
                continue
            column = min(
                range(len(cls._DEPTH_CENTERS)),
                key=lambda index: abs(cls._DEPTH_CENTERS[index] - relative_x),
            )
            if abs(cls._DEPTH_CENTERS[column] - relative_x) > cls._DEPTH_COLUMN_MAX_DISTANCE:
                continue
            current = assigned.get(column)
            if current is None or (candidate.confidence, len(candidate.text)) > (
                current.confidence,
                len(current.text),
            ):
                assigned[column] = candidate
        return assigned

    @staticmethod
    def choose_listing_price(
        levels: tuple[MarketDepthLevel, ...],
        trigger_price: int,
        thin_ratio: float = 0.05,
        minimum_gap_ticks: int = 2,
        mode: str = "auto",
        allow_below_trigger: bool = False,
        listing_lowest_price: int | None = None,
    ) -> SellPriceDecision:
        # The title's lowest-price field is clearer than the small labels above
        # the depth bars. A direct-lowest listing needs no tick calculation.
        if mode == "lowest" and listing_lowest_price is not None:
            if listing_lowest_price < trigger_price and not allow_below_trigger:
                return SellPriceDecision(
                    None,
                    None,
                    listing_lowest_price,
                    None,
                    None,
                    "当前标题最低价已经低于自动卖出触发价",
                )
            return SellPriceDecision(
                listing_lowest_price,
                None,
                listing_lowest_price,
                None,
                None,
                "按最低柱直接卖出，使用上架页标题最低价，未依赖价格柱 OCR",
            )

        ordered = sorted(levels, key=lambda level: level.price)
        if len(ordered) < 3:
            return SellPriceDecision(None, None, None, None, None, "有效价格柱不足 3 根")
        differences = [
            current.price - previous.price
            for previous, current in zip(ordered, ordered[1:])
            if current.price > previous.price
        ]
        if not differences:
            return SellPriceDecision(None, None, ordered[0].price, None, None, "无法计算价格档位")
        tick = round(median(differences))
        if tick <= 0 or any(abs(value - tick) > max(1, tick * 0.15) for value in differences):
            return SellPriceDecision(None, tick, ordered[0].price, None, None, "价格柱档位不一致")
        lowest = ordered[0]
        volume_levels = [level for level in ordered if level.volume is not None and level.volume > 0]
        if not volume_levels:
            return SellPriceDecision(None, tick, lowest.price, None, None, "价格柱数量未识别")
        main = max(volume_levels, key=lambda level: level.volume or 0)
        ratio = (lowest.volume or 0) / max(1, main.volume or 0)
        gap_ticks = (main.price - lowest.price) / tick
        thin_tail = ratio <= thin_ratio and gap_ticks >= minimum_gap_ticks
        if mode == "lowest":
            selected = lowest.price
            reason = "按试单策略使用当前最低柱"
        elif mode == "undercut":
            selected = lowest.price - tick
            reason = "试单超时，按商品规则降低一个价格档位"
        else:
            selected = lowest.price if thin_tail else lowest.price - tick
            reason = (
                f"最低柱仅为主力柱的 {ratio:.2%}，直接使用最低价"
                if thin_tail
                else "最低柱接近主力柱，降低一个价格档位"
            )
        if selected < trigger_price and not allow_below_trigger:
            if lowest.price >= trigger_price:
                selected = lowest.price
                reason += "；降档会低于触发价，改用最低价"
            else:
                return SellPriceDecision(
                    None,
                    tick,
                    lowest.price,
                    main.price,
                    ratio,
                    "当前最低价已经低于自动卖出触发价",
                )
        return SellPriceDecision(selected, tick, lowest.price, main.price, ratio, reason)

    @classmethod
    def top_order_matches(
        cls,
        order: SellOrderSnapshot,
        product_name: str,
        quantity: int,
        unit_price: int,
    ) -> bool:
        return (
            cls.normalize_product_name(order.product_name or "")
            == cls.normalize_product_name(product_name)
            and order.quantity == quantity
            and order.unit_price == unit_price
        )

    @staticmethod
    def normalize_product_name(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        normalized = normalized.replace("×", "x").replace("*", "x").replace("mm", "")
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", normalized)

    @staticmethod
    def _inventory_caliber_prefix(normalized: str) -> str | None:
        match = re.match(r"\d+(?:x\d+)+", normalized)
        return match.group(0) if match else None

    @classmethod
    def _focused_inventory_name_matches(cls, expected: str, candidate: str) -> bool:
        expected_name = cls.normalize_product_name(expected)
        candidate_name = cls.normalize_product_name(candidate)
        if expected_name == candidate_name:
            return True
        expected_components = cls._name_components(expected_name)
        candidate_components = cls._name_components(candidate_name)
        if expected_components is None or candidate_components is None:
            return False
        expected_caliber, expected_variant = expected_components
        candidate_caliber, candidate_variant = candidate_components
        return (
            candidate_caliber == expected_caliber
            and any(character.isdigit() for character in expected_variant)
            and candidate_variant.startswith(expected_variant)
            and len(candidate_variant) - len(expected_variant) <= 3
        )

    @classmethod
    def inventory_name_similarity(cls, expected: str, candidate: str) -> float:
        """Score noisy inventory labels before the listing-editor confirmation."""

        expected_name = cls.normalize_product_name(expected)
        candidate_name = cls.normalize_product_name(candidate)
        if not expected_name or not candidate_name:
            return 0.0
        if expected_name == candidate_name:
            return 1.0
        return SequenceMatcher(None, expected_name, candidate_name).ratio()

    @classmethod
    def listing_name_match_state(cls, expected: str, candidate: str) -> str:
        """Return ``match``, ``incomplete``, or ``mismatch`` for a title."""

        expected_name = cls.normalize_product_name(expected)
        candidate_name = cls.normalize_product_name(candidate)
        if not expected_name or not candidate_name:
            return "incomplete"
        if expected_name == candidate_name:
            return "match"

        expected_components = cls._name_components(expected_name)
        candidate_components = cls._name_components(candidate_name)
        if expected_components and candidate_components:
            if expected_components == candidate_components:
                return "match"
            if expected_components[0] == candidate_components[0]:
                return "mismatch"

        if len(candidate_name) < max(4, round(len(expected_name) * 0.65)):
            return "incomplete"
        return "mismatch"

    @staticmethod
    def _name_components(normalized: str) -> tuple[str, str] | None:
        match = re.match(r"(\d+(?:x\d+)?)([a-z].*)$", normalized)
        if not match:
            return None
        return match.group(1), match.group(2)

    @staticmethod
    def _parse_number(text: str) -> int | None:
        return parse_ocr_integer(text)

    def _read_listing_product_name(
        self,
        image: Image.Image,
        *,
        expected_name: str | None = None,
    ) -> OcrText | None:
        """Read the whole listing title instead of selecting one OCR word."""

        crop = self._relative_crop(image, self._LISTING_TITLE_CROP, scale=4)
        variants = (crop,) + ProductNamePreprocessor.fallback_variants(crop)
        readings: list[OcrText] = []
        for variant in variants:
            candidates = self._recognize(variant)
            if not candidates:
                continue
            text = "".join(candidate.text for candidate in candidates).strip()
            if text:
                readings.append(
                    OcrText(text, min(candidate.confidence for candidate in candidates))
                )
            if expected_name and self.listing_name_match_state(expected_name, text) == "match":
                return readings[-1]
        if not readings:
            return None
        if expected_name:
            state_rank = {"mismatch": 0, "incomplete": 1, "match": 2}
            return max(
                readings,
                key=lambda value: (
                    state_rank[self.listing_name_match_state(expected_name, value.text)],
                    self.inventory_name_similarity(expected_name, value.text),
                    value.confidence,
                ),
            )
        return max(
            readings,
            key=lambda value: (
                len(self.normalize_product_name(value.text)),
                value.confidence,
            ),
        )

    @staticmethod
    def _relative_crop(
        image: Image.Image,
        bounds: tuple[float, float, float, float],
        scale: int,
    ) -> np.ndarray:
        width, height = image.size
        crop = np.asarray(
            image.crop(
                (
                    round(width * bounds[0]),
                    round(height * bounds[1]),
                    round(width * bounds[2]),
                    round(height * bounds[3]),
                )
            )
        )
        return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _top_crop(
        image: Image.Image,
        bounds: tuple[float, float, float, float],
        scale: int,
    ) -> np.ndarray:
        crop = np.asarray(image.crop(ReferenceLayout(*image.size).top_box(bounds)))
        return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _recognize(self, image: np.ndarray) -> list[OcrText]:
        result, _ = self._get_engine()(image)
        return [
            OcrText(str(text).strip(), float(confidence))
            for _box, text, confidence in result or []
            if str(text).strip()
        ]

    def _recognize_positioned(
        self,
        image: Image.Image,
        bounds: tuple[float, float, float, float],
        *,
        scale: int,
    ) -> list[_PositionedOcrText]:
        prepared = self._relative_crop(image, bounds, scale)
        result, _ = self._get_engine()(prepared)
        width, height = image.size
        crop_left = round(width * bounds[0])
        crop_top = round(height * bounds[1])
        readings: list[_PositionedOcrText] = []
        for box, text, confidence in result or []:
            normalized = str(text).strip()
            if not normalized:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            readings.append(
                _PositionedOcrText(
                    text=normalized,
                    confidence=float(confidence),
                    left=(crop_left + min(xs) / scale) / width,
                    top=(crop_top + min(ys) / scale) / height,
                    right=(crop_left + max(xs) / scale) / width,
                    bottom=(crop_top + max(ys) / scale) / height,
                )
            )
        return readings

    @classmethod
    def _is_lowest_price_label(cls, text: str) -> bool:
        normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
        return any(label in normalized for label in cls._LISTING_LOWEST_PRICE_LABELS)

    @classmethod
    def _valid_positioned_integer(cls, reading: _PositionedOcrText) -> int | None:
        if reading.confidence < cls._NUMERIC_MIN_CONFIDENCE:
            return None
        value = parse_ocr_integer(reading.text)
        if value is None or value <= 0 or len(str(value)) < 2:
            return None
        return value

    def _recognize_single(self, image: np.ndarray) -> OcrText | None:
        result, _ = self._get_engine()(image, use_det=False, use_cls=False)
        if not result:
            return None
        text, confidence = result[0]
        normalized = str(text).strip()
        return OcrText(normalized, float(confidence)) if normalized else None

    def _read_numeric_crop(
        self,
        image: Image.Image,
        bounds: tuple[float, float, float, float],
        *,
        scale: int,
        top_anchored: bool = False,
    ) -> OcrText | None:
        prepared = (
            self._top_crop(image, bounds, scale)
            if top_anchored
            else self._relative_crop(image, bounds, scale)
        )
        recognized = self._get_numeric_engine().recognize_line(prepared)
        if recognized is None:
            return None
        text, confidence = recognized
        if (
            confidence < self._NUMERIC_MIN_CONFIDENCE
            or parse_ocr_integer(text) is None
        ):
            return None
        return OcrText(text, confidence)

    @staticmethod
    def _best_text(candidates: list[OcrText], require_digit: bool = False) -> OcrText | None:
        filtered = [
            candidate
            for candidate in candidates
            if not require_digit or re.search(r"\d", candidate.text)
        ]
        if not filtered:
            return None
        return max(filtered, key=lambda value: (value.confidence, len(value.text)))

    @staticmethod
    def _best_integer_text(candidates: list[OcrText]) -> OcrText | None:
        valid = [
            candidate
            for candidate in candidates
            if parse_ocr_integer(candidate.text) is not None
        ]
        if not valid:
            return None
        return max(valid, key=lambda value: (value.confidence, len(value.text)))

    def _get_engine(self):
        if self._engine is None:
            self._engine = preload_onnx_runtime()()
        return self._engine

    def _get_numeric_engine(self) -> NumericOcrEngine:
        if self._numeric_engine is None:
            self._numeric_engine = NumericOcrEngine()
        return self._numeric_engine
