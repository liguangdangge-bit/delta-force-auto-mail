from __future__ import annotations

from bulletbot.ocr.local_models import general_model_params

from contextlib import nullcontext
import os
import re
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import CardRegion, MarketObservation, OcrText
from bulletbot.ocr.numbers import parse_ocr_integer
from bulletbot.ocr.preprocessing import ProductNamePreprocessor
from bulletbot.ocr.runtime_backend import (
    OCR_RUNTIME_LOCK,
    directml_enabled,
    provider_label,
    report_ocr_runtime,
)
from bulletbot.runtime.performance import PerformanceStats


_DLL_DIRECTORY_HANDLES = []


class _RapidOcrV6Compat:
    """Expose RapidOCR 3.x through the tuple API used by the application."""

    def __init__(self, *, prefer_directml: bool = True) -> None:
        self._inference_lock = threading.Lock()
        self._using_directml = prefer_directml and directml_enabled()
        try:
            with OCR_RUNTIME_LOCK:
                self._engine = self._build_engine(self._using_directml)
        except Exception as exc:
            if not self._using_directml:
                raise
            report_ocr_runtime(
                f"交易 OCR 的 DirectML 初始化失败，已回退 CPU：{exc}",
                "WARNING",
            )
            self._using_directml = False
            with OCR_RUNTIME_LOCK:
                self._engine = self._build_engine(False)
        report_ocr_runtime(
            f"交易 OCR 实际后端：{provider_label(self._using_directml)}"
        )

    @staticmethod
    def _build_engine(use_directml: bool):
        if use_directml:
            from bulletbot.ocr.isolated_runtime import IsolatedRapidOcr

            return IsolatedRapidOcr("trade_v6")

        from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

        return RapidOCR(
            params={
                **general_model_params(),
                "Global.log_level": "error",
                "Global.use_cls": False,
                "EngineConfig.onnxruntime.use_dml": use_directml,
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.model_type": ModelType.SMALL,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.model_type": ModelType.MOBILE,
                "Cls.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.model_type": ModelType.SMALL,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
            }
        )

    def __call__(
        self,
        image: np.ndarray,
        use_det: bool = True,
        use_cls: bool = False,
        **_kwargs,
    ):
        with self._inference_lock, OCR_RUNTIME_LOCK:
            try:
                result = self._run(image, use_det, use_cls)
            except Exception as exc:
                if not self._using_directml:
                    raise
                report_ocr_runtime(
                    f"交易 OCR 的 DirectML 推理失败，已在本次运行回退 CPU：{exc}",
                    "WARNING",
                )
                self._using_directml = False
                self._engine = self._build_engine(False)
                result = self._run(image, use_det, use_cls)
        texts_value = getattr(result, "txts", None)
        scores_value = getattr(result, "scores", None)
        texts = list(texts_value) if texts_value is not None else []
        scores = list(scores_value) if scores_value is not None else []
        if not use_det:
            return [
                [str(text), float(score)]
                for text, score in zip(texts, scores)
            ], None

        boxes_value = getattr(result, "boxes", None)
        boxes = list(boxes_value) if boxes_value is not None else []
        return [
            [
                box.tolist() if hasattr(box, "tolist") else box,
                str(text),
                float(score),
            ]
            for box, text, score in zip(boxes, texts, scores)
        ], None

    def _run(self, image: np.ndarray, use_det: bool, use_cls: bool):
        return self._engine(
            image,
            use_det=use_det,
            use_cls=use_cls,
            use_rec=True,
        )

    @property
    def execution_provider(self) -> str:
        return provider_label(self._using_directml)


def _configure_frozen_dll_search_path() -> None:
    """Let ONNX Runtime resolve its sibling DLLs in a PyInstaller build."""

    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    internal_root = Path(
        getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
    )
    for directory in (
        internal_root,
        internal_root / "onnxruntime" / "capi",
        internal_root / "numpy.libs",
    ):
        if not directory.is_dir():
            continue
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
        except OSError:
            # PyInstaller's bootloader already adds the common directory on
            # most systems. Keep the normal import error if registration fails.
            continue


def preload_onnx_runtime():
    """Load the PP-OCRv6 small ONNX stack before Qt initializes."""

    _configure_frozen_dll_search_path()
    try:
        import onnxruntime  # noqa: F401
        import rapidocr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "无法加载 PP-OCRv6 ONNX OCR。"
            f" Python: {sys.executable}; 原因: {exc}"
        ) from exc
    return _RapidOcrV6Compat


class FavoritesOcr:
    """Read fixed text regions with ONNX OCR in the configured backend."""

    def __init__(self, timings: PerformanceStats | None = None) -> None:
        self._engine = None
        self._timings = timings

    def recognize(
        self,
        image: Image.Image,
        cards: list[CardRegion],
        known_names: dict[int, OcrText],
        read_names: bool,
        read_prices: bool,
        strict_prices: bool,
    ) -> list[MarketObservation]:
        observations: list[MarketObservation] = []
        for card in cards:
            name = known_names.get(card.index)
            if read_names:
                with self._measure("name_ocr"):
                    name = self._read_name(image, card)
                if name is not None and self._is_recent_purchase_section(name.text):
                    break
            price, price_confidence = (None, None)
            if read_prices:
                with self._measure("price_ocr"):
                    price, price_confidence = self._read_price(
                        image, card, strict_prices
                    )
            if name is None and price is None:
                continue
            observations.append(
                MarketObservation(
                    card=card,
                    name=name,
                    price=price,
                    price_confidence=price_confidence,
                )
            )
        return observations

    def _measure(self, phase: str):
        if self._timings is None:
            return nullcontext()
        return self._timings.measure(phase)

    @staticmethod
    def _is_recent_purchase_section(text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return "最近购买" in normalized

    def _read_name(self, image: Image.Image, card: CardRegion) -> OcrText | None:
        crop = image.crop(
            (
                card.name_bounds.left,
                card.name_bounds.top,
                card.name_bounds.right,
                card.name_bounds.bottom,
            )
        )
        crop_array = ProductNamePreprocessor.tighten_label(np.asarray(crop))
        primary = self._recognize_single(crop_array)
        if (
            primary is not None
            and primary.confidence >= ProductNamePreprocessor.MINIMUM_CONFIDENCE
        ):
            return primary

        candidates = [primary] if primary is not None else []
        for variant in ProductNamePreprocessor.fallback_variants(crop_array):
            candidate = self._recognize_single(variant)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None
        return max(candidates, key=lambda value: value.confidence)

    def _read_price(
        self, image: Image.Image, card: CardRegion, strict: bool
    ) -> tuple[int | None, float | None]:
        crop = self._price_crop(image, card)
        # Detection isolates the right-aligned price text from the silver coin
        # icon. Recognition-only OCR can interpret that icon as a leading "1"
        # (for example, 2,121 becomes 12,121), while still returning a valid
        # looking number. Prefer the localized text result and retain the fast
        # single-line recognizer only as a fallback when detection finds none.
        detected_result = self._recognize_price_with_detection(crop)
        detected_value = (
            self._parse_price(detected_result.text) if detected_result else None
        )
        if detected_result and detected_value is not None and (
            not strict
            or (
                self._is_valid_price_text(detected_result.text)
                and detected_result.confidence >= 0.75
            )
        ):
            return detected_value, detected_result.confidence

        fast_result = self._recognize_single(crop)
        fast_value = self._parse_price(fast_result.text) if fast_result else None
        if fast_result and fast_value is not None and (
            not strict
            or (
                self._is_valid_price_text(fast_result.text)
                and fast_result.confidence >= 0.75
            )
        ):
            return fast_value, fast_result.confidence
        return fast_value, fast_result.confidence if fast_result else None

    @staticmethod
    def _price_crop(image: Image.Image, card: CardRegion) -> np.ndarray:
        image_width, image_height = image.size
        left = card.bounds.left + round(card.bounds.width * 0.62)
        top = card.bounds.top + round(card.bounds.height * 0.66)
        # Prices are right-aligned and their final digit can extend beyond the
        # detected card edge. A 32 px safety margin preserved all digits in
        # the 1920x1080 reference capture, including three-digit values.
        right = min(image_width, card.bounds.right + max(32, round(card.bounds.width * 0.07)))
        bottom = min(image_height, card.bounds.bottom - 2)
        crop = np.asarray(image.crop((left, top, right, bottom)))
        return cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    def _recognize_single(self, image: np.ndarray) -> OcrText | None:
        result, _ = self._get_engine()(image, use_det=False, use_cls=False)
        if not result:
            return None
        text, confidence = result[0]
        text = str(text).strip()
        return OcrText(text=text, confidence=float(confidence)) if text else None

    def _recognize_price_with_detection(self, image: np.ndarray) -> OcrText | None:
        result, _ = self._get_engine()(image)
        if not result:
            return None
        candidates: list[OcrText] = []
        for _box, text, confidence in result:
            normalized = str(text).strip()
            if self._parse_price(normalized) is not None:
                candidates.append(OcrText(normalized, float(confidence)))
        if not candidates:
            return None
        return max(candidates, key=lambda value: (value.confidence, len(value.text)))

    @staticmethod
    def _parse_price(text: str) -> int | None:
        return parse_ocr_integer(text)

    @classmethod
    def _is_valid_price_text(cls, text: str) -> bool:
        return parse_ocr_integer(text) is not None

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        RapidOCR = preload_onnx_runtime()
        self._engine = RapidOCR()
        return self._engine
