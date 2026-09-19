from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

from bulletbot.ocr.runtime_backend import (
    OCR_RUNTIME_LOCK,
    directml_enabled,
    provider_label,
    report_ocr_runtime,
)


def _default_model_root() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        root = Path(__file__).resolve().parents[2]
    return root / "models" / "mail_storage" / "ocr" / "rapidocr"


class NumericOcrEngine:
    """Recognition-only PP-OCRv5 engine for tightly cropped numeric lines."""

    MODEL_VERSION = "PP-OCRv5 English mobile"

    def __init__(self, model_root: Path | None = None) -> None:
        self.model_root = model_root or _default_model_root()
        self._engine = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._using_directml = False

    def _get_engine(self):
        with self._lock:
            if self._engine is not None:
                return self._engine
            self._using_directml = directml_enabled()
            try:
                self._engine = self._build_engine(self._using_directml)
            except Exception as exc:
                if not self._using_directml:
                    raise
                report_ocr_runtime(
                    f"数字 OCR 的 DirectML 初始化失败，已回退 CPU：{exc}",
                    "WARNING",
                )
                self._using_directml = False
                self._engine = self._build_engine(False)
            report_ocr_runtime(
                f"数字 OCR 实际后端：{provider_label(self._using_directml)}；"
                "仅加载 en_PP-OCRv5 mobile 文字识别模型"
            )
            return self._engine

    def _build_engine(self, use_directml: bool):
        if use_directml:
            from bulletbot.ocr.isolated_runtime import IsolatedRapidOcr

            return IsolatedRapidOcr(
                "fast_price_v5",
                model_root=self.model_root,
                numeric_worker=True,
            )
        from bulletbot.ocr.recognition_only import RecognitionOnlyOcr

        return RecognitionOnlyOcr(
            self.model_root / "en_PP-OCRv5_rec_mobile.onnx",
            use_directml=False,
        )

    def recognize_line(self, image: np.ndarray) -> tuple[str, float] | None:
        engine = self._get_engine()
        with self._inference_lock, OCR_RUNTIME_LOCK:
            try:
                result = engine(
                    image,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
            except Exception as exc:
                if not self._using_directml:
                    raise
                report_ocr_runtime(
                    f"数字 OCR 的 DirectML 推理失败，已在本次运行回退 CPU：{exc}",
                    "WARNING",
                )
                self._using_directml = False
                engine = self._build_engine(False)
                self._engine = engine
                result = engine(
                    image,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
        texts_value = getattr(result, "txts", None)
        scores_value = getattr(result, "scores", None)
        if texts_value is None:
            return None
        scores = list(scores_value) if scores_value is not None else []
        candidates = [
            (str(text).strip(), float(score))
            for text, score in zip(list(texts_value), scores)
            if str(text).strip()
        ]
        return max(candidates, key=lambda item: item[1]) if candidates else None
