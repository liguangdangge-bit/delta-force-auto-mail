from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bulletbot.domain.models import PageState, Rect
from bulletbot.ocr.detail_ocr import DetailOcr
from bulletbot.ocr.favorites_ocr import preload_onnx_runtime
from bulletbot.ocr.preprocessing import ProductNamePreprocessor
from bulletbot.vision.favorites_grid import FavoritesGridDetector
from bulletbot.vision.page_classifier import PageClassifier
from bulletbot.vision.page_templates import PageTemplateMatcher


@dataclass(frozen=True)
class DetectedControl:
    name: str
    bounds: Rect
    confidence: float
    source: str = "ocr"


@dataclass(frozen=True)
class LayoutAnalysis:
    state: PageState
    confidence: float
    controls: dict[str, DetectedControl]
    evidence: tuple[str, ...]
    diagnostic_path: Path | None = None


class LayoutCalibrator:
    """Locate stable game controls from the current full-resolution page."""

    _LABELS = {
        "warehouse_tab": ("仓库",),
        "market_tab": ("交易行",),
        "buy_tab": ("购买",),
        "sell_tab": ("出售",),
        "trade_history_tab": ("交易记录",),
    }
    def __init__(self, page_templates: PageTemplateMatcher | None = None) -> None:
        self._engine = None
        self._detail = DetailOcr()
        self._favorites = FavoritesGridDetector()
        self._page_templates = page_templates or PageTemplateMatcher()

    def analyze(self, image: Image.Image) -> LayoutAnalysis:
        controls: dict[str, DetectedControl] = {}
        template_matches = self._page_templates.match(image)
        for template in template_matches:
            for name, bounds in template.controls:
                controls.setdefault(
                    name,
                    DetectedControl(
                        name,
                        bounds,
                        template.score * 0.86,
                        "template_anchor",
                    ),
                )
        anchor_bounds = (0, 0, image.width, round(image.height * 0.24))
        texts = self._recognize_region(image, anchor_bounds)
        navigation_found = all(
            self._find_text(
                texts,
                self._LABELS[name],
                max_top=image.height * 0.10,
                prefer_top=True,
            )
            is not None
            for name in ("warehouse_tab", "market_tab")
        )
        if not navigation_found:
            texts = self._merge_recognized_texts(
                texts,
                self._recognize_region(image, anchor_bounds, enhanced=True),
            )
        for name, aliases in self._LABELS.items():
            max_top = image.height * (0.085 if name in {"warehouse_tab", "market_tab"} else 0.16)
            match = self._find_text(
                texts,
                aliases,
                max_top=max_top,
                prefer_top=name in {"warehouse_tab", "market_tab"},
            )
            if match is not None:
                text, confidence, bounds = match
                del text
                controls[name] = DetectedControl(
                    name,
                    self._expand(bounds, image.width, image.height),
                    confidence,
                )
        if "sell_tab" not in controls:
            buy = controls.get("buy_tab")
            history = controls.get("trade_history_tab")
            if buy is not None and history is not None:
                buy_center = buy.bounds.left + buy.bounds.width / 2
                history_center = history.bounds.left + history.bounds.width / 2
                center_x = round((buy_center + history_center) / 2)
                width = max(buy.bounds.width, round((history_center - buy_center) * 0.42))
                controls["sell_tab"] = DetectedControl(
                    "sell_tab",
                    Rect(
                        center_x - width // 2,
                        min(buy.bounds.top, history.bounds.top),
                        width,
                        max(buy.bounds.height, history.bounds.height),
                    ),
                    min(buy.confidence, history.confidence) * 0.85,
                    "anchor_inference",
                )

        listing_controls = self._detect_listing_controls(image)
        listing_required = {
            "listing_confirm",
            "listing_quantity_minus",
            "listing_quantity_plus",
            "listing_price_minus",
            "listing_price_plus",
        }
        market_navigation = controls.get("market_tab")
        market_navigation_active = (
            market_navigation is not None
            and PageClassifier.market_tab_score(image) >= 0.12
        )
        listing_title_visible = self._find_text(
            texts,
            ("上架物品", "上架道具"),
        ) is not None
        if (
            (market_navigation_active or listing_title_visible)
            and listing_required.issubset(listing_controls)
        ):
            controls.update(listing_controls)
            identity_evidence = (
                "检测到上架编辑器标题和操作区"
                if listing_title_visible
                else "检测到交易行高亮和上架编辑器操作区"
            )
            return LayoutAnalysis(
                PageState.LISTING_EDITOR,
                min(0.99, 0.88 + listing_controls["listing_confirm"].confidence * 0.1),
                controls,
                (identity_evidence,),
            )

        purchase_controls = self._detail.read_purchase_controls(image)
        detail_controls = self._detail_controls(purchase_controls)
        detail_required = {
            "detail_price_button",
            "detail_slider",
            "detail_minus",
            "detail_plus",
        }
        if detail_required.issubset(detail_controls):
            controls.update(detail_controls)
            return LayoutAnalysis(
                PageState.DETAIL,
                0.98,
                controls,
                ("检测到详情页价格按钮和数量控件",),
            )

        normalized_texts = tuple(self._normalize(text) for text, _confidence, _bounds in texts)
        is_sell_page = any("上架数量" in text for text in normalized_texts) and any(
            "上架" in text and ("道具" in text or "物品" in text)
            for text in normalized_texts
        )
        if is_sell_page:
            controls.update(self._detect_sell_inventory_controls(image, texts))
            slots = self._find_text(texts, ("上架数量",))
            if slots is not None:
                _text, confidence, bounds = slots
                controls["sell_slots_region"] = DetectedControl(
                    "sell_slots_region",
                    self._expand(bounds, image.width, image.height),
                    confidence,
                )
            return LayoutAnalysis(
                PageState.MARKET_SELL,
                0.96,
                controls,
                ("OCR 检测到出售页上架数量和库存标题",),
            )

        warehouse = controls.get("warehouse_tab")
        market = controls.get("market_tab")
        warehouse_score = 0.0
        market_score = 0.0
        if warehouse is not None and market is not None:
            warehouse_score = PageClassifier.warehouse_tab_score(image)
            market_score = PageClassifier.market_tab_score(image)
            if warehouse_score >= 0.12 and warehouse_score >= market_score + 0.04:
                return LayoutAnalysis(
                    PageState.WAREHOUSE,
                    min(0.98, 0.72 + warehouse_score),
                    controls,
                    (f"仓库标签高亮 {warehouse_score:.0%}",),
                )

        cards = self._favorites.detect(image)
        if len(cards) >= 9 and (
            "buy_tab" in controls
            or "sell_tab" in controls
            or market_score >= 0.12
        ):
            first = cards[0].bounds
            controls["favorites_first_card"] = DetectedControl(
                "favorites_first_card", first, 0.95, "grid"
            )
            return LayoutAnalysis(
                PageState.FAVORITES,
                0.98,
                controls,
                (f"检测到 {len(cards)} 个收藏卡片",),
            )

        if warehouse is not None and market is not None:
            if market_score >= 0.12:
                return LayoutAnalysis(
                    PageState.MARKET_OTHER,
                    min(0.95, 0.65 + market_score),
                    controls,
                    (f"交易行标签高亮 {market_score:.0%}",),
                )

        return LayoutAnalysis(
            PageState.UNKNOWN,
            0.0,
            controls,
            ("自动校准未识别当前页面",),
        )

    def annotate(self, image: Image.Image, analysis: LayoutAnalysis) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for control in analysis.controls.values():
            bounds = control.bounds
            draw.rectangle(
                (bounds.left, bounds.top, bounds.right, bounds.bottom),
                outline="#36d399",
                width=3,
            )
            draw.text(
                (bounds.left + 3, max(0, bounds.top - 18)),
                control.name,
                fill="#f9fafb",
            )
        return annotated

    @staticmethod
    def _detail_controls(
        purchase_controls: tuple[
            int | None,
            int | None,
            int | None,
            float | None,
            Rect | None,
            Rect | None,
            Rect | None,
            Rect | None,
        ],
    ) -> dict[str, DetectedControl]:
        names = (
            "detail_price_button",
            "detail_slider",
            "detail_minus",
            "detail_plus",
        )
        rects = (
            purchase_controls[4],
            purchase_controls[5],
            purchase_controls[6],
            purchase_controls[7],
        )
        return {
            name: DetectedControl(name, bounds, 0.95, "structure")
            for name, bounds in zip(names, rects)
            if bounds is not None
        }

    def _detect_sell_inventory_controls(
        self,
        image: Image.Image,
        texts: list[tuple[str, float, Rect]],
    ) -> dict[str, DetectedControl]:
        title = self._find_text(texts, ("请选择要上架的道具", "请选择要上架的物品"))
        if title is None:
            return {}
        _text, confidence, title_bounds = title
        frame = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 25, 80)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Rect] = []
        min_size = max(18, round(min(image.size) * 0.018))
        max_size = max(min_size + 1, round(min(image.size) * 0.075))
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if not (min_size <= width <= max_size and min_size <= height <= max_size):
                continue
            if not (0.68 <= width / max(1, height) <= 1.35):
                continue
            center_x = left + width / 2
            if not (
                title_bounds.left - min_size
                <= center_x
                <= title_bounds.left + max_size * 0.65
            ):
                continue
            if top < title_bounds.bottom or top > image.height * 0.75:
                continue
            candidates.append(Rect(left, top, width, height))
        deduped = self._deduplicate_rects(candidates, tolerance=max(10, min_size))
        aligned = sorted(deduped, key=lambda rect: rect.top)
        result: dict[str, DetectedControl] = {}
        for index, bounds in enumerate(aligned[:3]):
            result[f"sell_inventory_tab_{index}"] = DetectedControl(
                f"sell_inventory_tab_{index}", bounds, confidence, "structure"
            )
        for index, candidate in enumerate(
            self._best_inventory_cells(image, title_bounds, limit=5)
        ):
            name = f"sell_inventory_candidate_{index}"
            result[name] = DetectedControl(name, candidate, 0.80, "content")
        return result

    @staticmethod
    def _best_inventory_cell(image: Image.Image, title_bounds: Rect) -> Rect | None:
        candidates = LayoutCalibrator._best_inventory_cells(
            image,
            title_bounds,
            limit=1,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _best_inventory_cells(
        image: Image.Image,
        title_bounds: Rect,
        *,
        limit: int,
    ) -> list[Rect]:
        frame = np.asarray(image.convert("RGB"))
        grid_left = title_bounds.left + max(title_bounds.height * 2, round(image.width * 0.025))
        grid_right = image.width - max(20, round(image.width * 0.045))
        grid_top = title_bounds.bottom + max(4, round(image.height * 0.006))
        if grid_right <= grid_left:
            return []
        columns = 9
        cell_width = (grid_right - grid_left) // columns
        cell_height = max(20, round(cell_width * 0.82))
        scored: list[tuple[float, Rect]] = []
        for row in range(0, 6):
            top = grid_top + row * cell_height
            bottom = min(image.height, top + cell_height)
            if bottom - top < cell_height * 0.70:
                break
            for column in range(columns):
                left = grid_left + column * cell_width
                right = min(grid_right, left + cell_width)
                crop = frame[top:bottom, left:right]
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                edge_density = float(np.mean(cv2.Canny(gray, 25, 80) > 0))
                contrast = float(np.std(gray)) / 255.0
                score = edge_density + contrast
                bounds = Rect(left, top, right - left, bottom - top)
                if score >= 0.08:
                    scored.append((score, bounds))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [bounds for _score, bounds in scored[: max(1, limit)]]

    def _detect_listing_controls(self, image: Image.Image) -> dict[str, DetectedControl]:
        frame = np.asarray(image.convert("RGB"))
        red = frame[:, :, 0].astype(np.int16)
        green = frame[:, :, 1].astype(np.int16)
        blue = frame[:, :, 2].astype(np.int16)
        mask = (
            (green >= 70)
            & (green >= red + 18)
            & (green >= blue + 8)
        ).astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        action_candidates: list[Rect] = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if (
                width >= image.width * 0.10
                and height >= image.height * 0.025
                and top >= image.height * 0.45
                and width / max(1, height) >= 3.0
            ):
                action_candidates.append(Rect(left, top, width, height))
        if not action_candidates:
            return {}
        action = max(action_candidates, key=lambda rect: rect.width * rect.height)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 25, 90)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        squares: list[Rect] = []
        min_size = max(18, round(min(image.size) * 0.018))
        max_size = max(min_size + 1, round(min(image.size) * 0.060))
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if not (min_size <= width <= max_size and min_size <= height <= max_size):
                continue
            if not (0.65 <= width / max(1, height) <= 1.40):
                continue
            if left < image.width * 0.45 or top < image.height * 0.35 or top >= action.top:
                continue
            if (
                left < action.left - action.width * 0.30
                or left + width > action.right + action.width * 0.30
            ):
                continue
            squares.append(Rect(left, top, width, height))
        squares = self._deduplicate_rects(squares, tolerance=max(4, min_size // 3))
        rows: list[list[Rect]] = []
        for rect in sorted(squares, key=lambda value: (value.top, value.left)):
            center_y = rect.top + rect.height / 2
            row = next(
                (
                    existing
                    for existing in rows
                    if abs(
                        center_y
                        - sum(item.top + item.height / 2 for item in existing) / len(existing)
                    )
                    <= max(rect.height, existing[0].height) * 0.45
                ),
                None,
            )
            if row is None:
                rows.append([rect])
            else:
                row.append(rect)
        paired_rows = [
            sorted(row, key=lambda value: value.left)
            for row in rows
            if len(row) >= 2
            and max(item.left for item in row) - min(item.left for item in row)
            >= action.width * 0.80
        ]
        paired_rows = sorted(paired_rows, key=lambda row: row[0].top)
        result = {
            "listing_confirm": DetectedControl(
                "listing_confirm", action, 0.95, "color"
            )
        }
        if len(paired_rows) >= 2:
            quantity_row, price_row = paired_rows[-2:]
            for name, bounds in (
                ("listing_quantity_minus", quantity_row[0]),
                ("listing_quantity_plus", quantity_row[-1]),
                ("listing_price_minus", price_row[0]),
                ("listing_price_plus", price_row[-1]),
            ):
                result[name] = DetectedControl(name, bounds, 0.90, "structure")
        return result

    def _recognize_region(
        self,
        image: Image.Image,
        bounds: tuple[int, int, int, int],
        *,
        enhanced: bool = False,
    ) -> list[tuple[str, float, Rect]]:
        crop = np.asarray(image.crop(bounds))
        if enhanced:
            variants = ProductNamePreprocessor.fallback_variants(crop)
            if variants:
                crop = variants[0]
        result, _ = self._get_engine()(crop)
        left_offset, top_offset = bounds[:2]
        recognized: list[tuple[str, float, Rect]] = []
        for box, text, confidence in result or []:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            left = left_offset + round(min(xs))
            top = top_offset + round(min(ys))
            right = left_offset + round(max(xs))
            bottom = top_offset + round(max(ys))
            recognized.append(
                (
                    str(text).strip(),
                    float(confidence),
                    Rect(left, top, max(1, right - left), max(1, bottom - top)),
                )
            )
        return recognized

    @staticmethod
    def _merge_recognized_texts(
        primary: list[tuple[str, float, Rect]],
        fallback: list[tuple[str, float, Rect]],
    ) -> list[tuple[str, float, Rect]]:
        merged = list(primary)
        for candidate in fallback:
            center_x = candidate[2].left + candidate[2].width / 2
            center_y = candidate[2].top + candidate[2].height / 2
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if abs(center_x - (existing[2].left + existing[2].width / 2))
                    <= max(candidate[2].width, existing[2].width) * 0.35
                    and abs(center_y - (existing[2].top + existing[2].height / 2))
                    <= max(candidate[2].height, existing[2].height) * 0.55
                ),
                None,
            )
            if duplicate_index is None:
                merged.append(candidate)
            elif candidate[1] > merged[duplicate_index][1]:
                merged[duplicate_index] = candidate
        return merged

    @classmethod
    def _find_text(
        cls,
        texts: list[tuple[str, float, Rect]],
        aliases: tuple[str, ...],
        *,
        max_top: float | None = None,
        prefer_top: bool = False,
    ) -> tuple[str, float, Rect] | None:
        normalized_aliases = tuple(cls._normalize(alias) for alias in aliases)
        matches = [
            item
            for item in texts
            if any(alias in cls._normalize(item[0]) for alias in normalized_aliases)
            and (max_top is None or item[2].top <= max_top)
        ]
        if prefer_top:
            return min(
                matches,
                key=lambda item: (item[2].top, -item[1]),
                default=None,
            )
        return max(matches, key=lambda item: item[1], default=None)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", str(text)).casefold()

    @staticmethod
    def _expand(bounds: Rect, width: int, height: int) -> Rect:
        horizontal = max(8, round(width * 0.006))
        vertical = max(6, round(height * 0.006))
        left = max(0, bounds.left - horizontal)
        top = max(0, bounds.top - vertical)
        right = min(width, bounds.right + horizontal)
        bottom = min(height, bounds.bottom + vertical)
        return Rect(left, top, right - left, bottom - top)

    @staticmethod
    def _deduplicate_rects(rects: list[Rect], tolerance: int) -> list[Rect]:
        result: list[Rect] = []
        for rect in sorted(rects, key=lambda value: value.width * value.height, reverse=True):
            center_x = rect.left + rect.width / 2
            center_y = rect.top + rect.height / 2
            if any(
                abs(center_x - (existing.left + existing.width / 2)) <= tolerance
                and abs(center_y - (existing.top + existing.height / 2)) <= tolerance
                for existing in result
            ):
                continue
            result.append(rect)
        return result

    def _get_engine(self):
        if self._engine is None:
            self._engine = preload_onnx_runtime()()
        return self._engine
