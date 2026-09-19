from __future__ import annotations

import hashlib
import re
from threading import Lock
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import OcrText, Rect
from bulletbot.ocr.favorites_ocr import preload_onnx_runtime
from bulletbot.ocr.numbers import parse_ocr_integer, parse_ocr_integer_pair
from bulletbot.ocr.numeric_ocr import NumericOcrEngine
from bulletbot.ocr.preprocessing import ProductNamePreprocessor
from bulletbot.vision.reference_layout import ReferenceLayout


@dataclass(frozen=True)
class BalanceOcrRead:
    balance: int | None
    confidence: float | None
    selected_text: str | None
    candidates: tuple[OcrText, ...]


@dataclass(frozen=True)
class AveragePriceRead:
    unit_price: int | None
    confidence: float | None
    signature: str
    changed: bool


class DetailOcr:
    """Read prices, quantity, and the expanded balance on the detail page."""

    _LOWEST_PRICE_FAST_CROP = (0.148, 0.145, 0.22, 0.20)
    _PRODUCT_NAME_CROP = (0.765, 0.135, 0.955, 0.195)
    _BALANCE_CROP = (0.74, 0.235, 0.93, 0.305)
    _AVERAGE_PRICE_HEIGHT_RATIO = 0.95

    def __init__(self) -> None:
        self._engine = None
        self._numeric_engine = None
        self._balance_label_engine = None
        self._balance_engine_lock = Lock()

    def is_detail_page(self, image: Image.Image) -> bool:
        """Identify the purchase detail layout without running OCR."""

        frame = np.asarray(image.convert("RGB"))
        price_button = self._find_price_button(frame)
        if price_button is None:
            return False
        slider = self._find_slider(frame, price_button)
        if slider is None:
            return False
        minus_button, plus_button = self._find_quantity_step_buttons(
            frame,
            slider,
            price_button,
        )
        return minus_button is not None and plus_button is not None

    def read_lowest_price(self, image: Image.Image) -> tuple[int | None, float | None]:
        layout = ReferenceLayout(*image.size)
        fast_crop = image.crop(layout.top_box(self._LOWEST_PRICE_FAST_CROP))
        fast_prepared = cv2.resize(
            np.asarray(fast_crop), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
        )
        fast_result = self._get_numeric_engine().recognize_line(fast_prepared)
        if fast_result is not None:
            text, confidence = fast_result
            value = self._parse_number(text)
            if value is not None and confidence >= 0.75:
                return value, confidence
        return None, None

    def read_product_name(self, image: Image.Image) -> tuple[str | None, float | None]:
        """Read the single-line product title without running text detection."""

        layout = ReferenceLayout(*image.size)
        crop = image.crop(layout.top_box(self._PRODUCT_NAME_CROP))
        prepared = ProductNamePreprocessor.tighten_label(np.asarray(crop))
        primary = self._read_single_line(prepared)
        if (
            primary is not None
            and primary.confidence >= ProductNamePreprocessor.MINIMUM_CONFIDENCE
        ):
            return primary.text, primary.confidence

        candidates = [primary] if primary is not None else []
        for variant in ProductNamePreprocessor.fallback_variants(prepared):
            candidate = self._read_single_line(variant)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None, None
        best = max(candidates, key=lambda value: (value.confidence, len(value.text)))
        return best.text, best.confidence

    def _read_single_line(self, image: np.ndarray) -> OcrText | None:
        result, _ = self._get_engine()(image, use_det=False, use_cls=False)
        if not result:
            return None
        text, confidence = str(result[0][0]).strip(), float(result[0][1])
        return OcrText(text, confidence) if text else None

    def read_purchase_controls(
        self, image: Image.Image
    ) -> tuple[
        int | None,
        int | None,
        int | None,
        float | None,
        Rect | None,
        Rect | None,
        Rect | None,
        Rect | None,
    ]:
        frame = np.asarray(image.convert("RGB"))
        price_button = self._find_price_button(frame)
        if price_button is None:
            return None, None, None, None, None, None, None, None

        slider = self._find_slider(frame, price_button)
        if slider is None:
            return None, None, None, None, None, None, None, None
        minus_button, plus_button = self._find_quantity_step_buttons(frame, slider, price_button)
        quantity, max_quantity, average_price, confidence = self._read_quantity_and_price(
            image, price_button
        )
        return (
            quantity,
            max_quantity,
            average_price,
            confidence,
            price_button,
            slider,
            minus_button,
            plus_button,
        )

    def read_expanded_balance(self, image: Image.Image) -> tuple[int | None, float | None]:
        read = self.read_expanded_balance_details(image)
        return read.balance, read.confidence

    def read_average_unit_price(
        self,
        image: Image.Image,
        button: Rect,
        previous_signature: str | None = None,
    ) -> AveragePriceRead:
        """Read the average unit-price row after a cheap visual change check."""

        crop = image.crop(self.average_price_box(image, button)).convert("RGB")
        return self.read_average_unit_price_crop(crop, previous_signature)

    def read_average_unit_price_crop(
        self,
        crop: Image.Image,
        previous_signature: str | None = None,
    ) -> AveragePriceRead:
        """Read a caller-supplied average-price crop without locating controls."""

        crop_array = np.asarray(crop.convert("RGB"))
        gray = cv2.cvtColor(crop_array, cv2.COLOR_RGB2GRAY)
        signature = self._average_price_signature(gray)
        changed = previous_signature is None or signature != previous_signature
        if not changed:
            return AveragePriceRead(None, None, signature, False)

        prepared = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )
        result, _ = self._get_engine()(prepared, use_det=False)
        candidates: list[tuple[int, float]] = []
        for candidate in result or []:
            if len(candidate) < 2:
                continue
            text, confidence = str(candidate[0]).strip(), float(candidate[1])
            value = parse_ocr_integer(text)
            if value is None or len(str(value)) < 2:
                continue
            candidates.append((value, confidence))
        if not candidates:
            return AveragePriceRead(None, None, signature, True)
        unit_price, confidence = max(
            candidates,
            key=lambda candidate: (candidate[1], len(str(candidate[0]))),
        )
        return AveragePriceRead(unit_price, confidence, signature, True)

    def read_single_line_crop(self, crop: Image.Image) -> OcrText | None:
        """Read one small caller-supplied text row without text detection."""

        crop_array = np.asarray(crop.convert("RGB"))
        gray = cv2.cvtColor(crop_array, cv2.COLOR_RGB2GRAY)
        prepared = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )
        result, _ = self._get_engine()(prepared, use_det=False)
        candidates = [
            OcrText(str(candidate[0]).strip(), float(candidate[1]))
            for candidate in (result or [])
            if len(candidate) >= 2 and str(candidate[0]).strip()
        ]
        return max(candidates, key=lambda candidate: candidate.confidence, default=None)

    @classmethod
    def average_price_box(
        cls,
        image: Image.Image,
        button: Rect,
    ) -> tuple[int, int, int, int]:
        return cls.average_price_box_for_size(image.width, image.height, button)

    @classmethod
    def average_price_box_for_size(
        cls,
        width: int,
        height: int,
        button: Rect,
    ) -> tuple[int, int, int, int]:
        return (
            max(0, button.left),
            max(0, button.top - round(button.height * cls._AVERAGE_PRICE_HEIGHT_RATIO)),
            min(width, button.right),
            min(height, button.top),
        )

    def average_price_signature(self, image: Image.Image, button: Rect) -> str:
        crop = np.asarray(
            image.crop(self.average_price_box(image, button)).convert("RGB")
        )
        return self._average_price_signature(
            cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        )

    @staticmethod
    def _average_price_signature(gray: np.ndarray) -> str:
        signature_mask = np.where(gray > 55, 255, 0).astype(np.uint8)
        signature_image = cv2.resize(
            signature_mask,
            (128, 24),
            interpolation=cv2.INTER_AREA,
        )
        return hashlib.sha1(signature_image.tobytes()).hexdigest()

    def read_expanded_balance_details(self, image: Image.Image) -> BalanceOcrRead:
        """Read the exact Half Coin value from the hover card, not the rounded K header."""

        crop = np.asarray(image.crop(self.balance_crop_box(image)))
        result, _ = self._get_engine()(crop)
        recognized: list[OcrText] = []
        candidates: list[tuple[int, OcrText]] = []
        for _box, text, confidence in result or []:
            normalized = str(text).strip()
            recognized_text = OcrText(normalized, float(confidence))
            if normalized:
                recognized.append(recognized_text)
            value = parse_ocr_integer(normalized)
            if value is None or len(str(value)) < 6:
                continue
            candidates.append((value, recognized_text))
        if not candidates:
            return BalanceOcrRead(None, None, None, tuple(recognized))
        value, selected = max(
            candidates,
            key=lambda candidate: (len(str(candidate[0])), candidate[1].confidence),
        )
        return BalanceOcrRead(value, selected.confidence, selected.text, tuple(recognized))

    @classmethod
    def balance_crop_box(cls, image: Image.Image) -> tuple[int, int, int, int]:
        return ReferenceLayout(*image.size).top_box(cls._BALANCE_CROP)

    def _read_quantity_and_price(
        self, image: Image.Image, button: Rect
    ) -> tuple[int | None, int | None, int | None, float | None]:
        top = max(0, button.top - round(button.height * 2.05))
        crop = np.asarray(image.crop((button.left, top, button.right, button.bottom)))
        result, _ = self._get_engine()(crop)
        quantity: int | None = None
        max_quantity: int | None = None
        average_price_candidates: list[OcrText] = []
        button_top = button.top - top
        for box, text, confidence in result or []:
            normalized = str(text).strip()
            quantity_pair = parse_ocr_integer_pair(normalized)
            if quantity_pair is not None:
                quantity, max_quantity = quantity_pair
                continue
            box_center_y = sum(float(point[1]) for point in box) / len(box)
            if "平均单价" in normalized and re.search(r"\d", normalized):
                average_price_candidates.append(OcrText(normalized, float(confidence)))
            elif (
                box_center_y < button_top
                and box_center_y >= button_top - button.height * 0.85
                and re.search(r"\d", normalized)
            ):
                average_price_candidates.append(OcrText(normalized, float(confidence)))
        if not average_price_candidates:
            return quantity, max_quantity, None, None
        best = max(
            average_price_candidates,
            key=lambda value: ("平均单价" in value.text, value.confidence, len(value.text)),
        )
        return quantity, max_quantity, self._parse_number(best.text), best.confidence

    @staticmethod
    def _find_price_button(frame: np.ndarray) -> Rect | None:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 20, 70)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Rect] = []
        for contour in contours:
            left, top, rect_width, rect_height = cv2.boundingRect(contour)
            aspect = rect_width / max(1, rect_height)
            if (
                left > width * 0.68
                and top > height * 0.55
                and width * 0.11 <= rect_width <= width * 0.23
                and height * 0.035 <= rect_height <= height * 0.09
                and 3.4 <= aspect <= 8.5
            ):
                candidates.append(Rect(left, top, rect_width, rect_height))

        deduplicated: list[Rect] = []
        for candidate in sorted(candidates, key=lambda rect: rect.width * rect.height, reverse=True):
            if any(
                abs(candidate.left + candidate.width // 2 - existing.left - existing.width // 2)
                < width * 0.01
                and abs(candidate.top + candidate.height // 2 - existing.top - existing.height // 2)
                < height * 0.01
                for existing in deduplicated
            ):
                continue
            deduplicated.append(candidate)

        # Some item pages show only the green price button and omit the
        # secondary exchange button below it. Color is the strongest signal
        # for the actionable price control in both one- and two-button layouts.
        green_candidates = [
            (candidate, DetailOcr._green_emphasis_score(frame, candidate))
            for candidate in deduplicated
        ]
        green_candidates = [
            candidate for candidate in green_candidates if candidate[1] >= 0.02
        ]
        if green_candidates:
            return max(
                green_candidates,
                key=lambda candidate: (
                    candidate[1],
                    candidate[0].width * candidate[0].height,
                ),
            )[0]

        pairs: list[tuple[Rect, Rect]] = []
        for upper in deduplicated:
            for lower in deduplicated:
                if lower.top <= upper.top:
                    continue
                if abs(upper.left - lower.left) > width * 0.015:
                    continue
                if abs(upper.width - lower.width) > width * 0.02:
                    continue
                gap = lower.top - upper.bottom
                if -height * 0.01 <= gap <= height * 0.04:
                    pairs.append((upper, lower))
        if not pairs:
            return None
        return max(pairs, key=lambda pair: pair[1].bottom)[0]

    @staticmethod
    def _green_emphasis_score(frame: np.ndarray, bounds: Rect) -> float:
        crop = frame[bounds.top : bounds.bottom, bounds.left : bounds.right].astype(np.int16)
        if crop.size == 0:
            return 0.0
        red = crop[:, :, 0]
        green = crop[:, :, 1]
        blue = crop[:, :, 2]
        emphasized = (green >= 70) & (green >= red + 18) & (green >= blue + 8)
        return float(np.mean(emphasized))

    @staticmethod
    def _find_slider(frame: np.ndarray, button: Rect) -> Rect | None:
        """Locate the expected quantity track and verify that it is visible."""

        left = button.left + round(button.width * 0.11)
        right = button.right - round(button.width * 0.11)
        expected_y = button.top - round(button.height * 0.90)
        search_radius = max(6, round(button.height * 0.25))
        top = max(0, expected_y - search_radius)
        bottom = min(frame.shape[0], expected_y + search_radius + 1)
        left = max(0, left)
        right = min(frame.shape[1], right)
        if right <= left or bottom <= top:
            return None

        gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
        bright = gray >= 75
        required_run = max(1, round((right - left) * 0.55))
        qualifying_rows = [
            row_index
            for row_index, row in enumerate(bright)
            if DetailOcr._longest_true_run(row) >= required_run
        ]
        if not qualifying_rows:
            return None

        center_row = round(sum(qualifying_rows) / len(qualifying_rows))
        slider_height = max(4, round(button.height * 0.13))
        slider_top = max(0, top + center_row - slider_height // 2)
        return Rect(left, slider_top, right - left, slider_height)

    @staticmethod
    def _longest_true_run(values: np.ndarray) -> int:
        longest = 0
        current = 0
        for value in values:
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    @staticmethod
    def _find_quantity_step_buttons(
        frame: np.ndarray, slider: Rect, price_button: Rect
    ) -> tuple[Rect | None, Rect | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 20, 70)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Rect] = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            aspect = width / max(1, height)
            center_y = top + height / 2
            if (
                price_button.width * 0.07 <= width <= price_button.width * 0.28
                and price_button.height * 0.35 <= height <= price_button.height * 1.25
                and 0.68 <= aspect <= 1.35
                and abs(center_y - (slider.top + slider.height / 2)) <= price_button.height * 0.28
            ):
                candidates.append(Rect(left, top, width, height))

        deduplicated: list[Rect] = []
        for candidate in sorted(candidates, key=lambda rect: rect.width * rect.height, reverse=True):
            candidate_center = (candidate.left + candidate.width / 2, candidate.top + candidate.height / 2)
            if any(
                abs(candidate_center[0] - (existing.left + existing.width / 2)) <= 3
                and abs(candidate_center[1] - (existing.top + existing.height / 2)) <= 3
                for existing in deduplicated
            ):
                continue
            deduplicated.append(candidate)

        center_y = slider.top + slider.height / 2
        left_candidates = [rect for rect in deduplicated if rect.right <= slider.left]
        right_candidates = [rect for rect in deduplicated if rect.left >= slider.right]
        minus = min(
            left_candidates,
            key=lambda rect: (abs(rect.top + rect.height / 2 - center_y), slider.left - rect.right),
            default=None,
        )
        plus = min(
            right_candidates,
            key=lambda rect: (abs(rect.top + rect.height / 2 - center_y), rect.left - slider.right),
            default=None,
        )
        return minus, plus

    @staticmethod
    def _parse_number(text: str) -> int | None:
        return parse_ocr_integer(text)

    def _get_engine(self):
        if self._engine is None:
            self._engine = preload_onnx_runtime()()
        return self._engine

    def _get_numeric_engine(self) -> NumericOcrEngine:
        with self._balance_engine_lock:
            if self._numeric_engine is None:
                self._numeric_engine = NumericOcrEngine()
            return self._numeric_engine

    def _get_balance_label_engine(self):
        from bulletbot.ocr.balance_label import BalanceLabelOcr
        with self._balance_engine_lock:
            if self._balance_label_engine is None:
                self._balance_label_engine = BalanceLabelOcr()
            return self._balance_label_engine
