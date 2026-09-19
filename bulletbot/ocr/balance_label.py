"""Recognize the currency label without requiring a rectangular tooltip border."""
import re

from bulletbot.domain.models import OcrText
from bulletbot.ocr.runtime_backend import OCR_RUNTIME_LOCK, directml_enabled, report_ocr_runtime


LABEL_MIN_CONFIDENCE = 0.60
LABEL_ALIASES = {"哈夫币", "哈弗币", "哈佛币"}


def label_failure(read):
    if read is None:
        return "标签OCR无结果：没有读取到哈夫币/哈弗币三个字"
    text = re.sub(r"[^\u4e00-\u9fff]", "", read.text)
    detail = f"标签原文={read.text!r}，置信度={read.confidence:.3f}"
    if text not in LABEL_ALIASES:
        return f"标签不匹配：没有确认哈夫币/哈弗币三个字；{detail}"
    if read.confidence < LABEL_MIN_CONFIDENCE:
        return f"标签置信度不足（要求≥{LABEL_MIN_CONFIDENCE:.3f}）；{detail}"
    return None


class BalanceLabelOcr:
    def __init__(self):
        self._engine = None
        self._directml = directml_enabled()

    def read(self, image):
        with OCR_RUNTIME_LOCK:
            try:
                return self._read(image)
            except Exception as exc:
                if not self._directml:
                    raise
                report_ocr_runtime(f"余额标签OCR DirectML异常，回退CPU：{type(exc).__name__}: {exc}", "WARNING")
                self._engine = None
                self._directml = False
                return self._read(image)

    def _read(self, image):
        if self._engine is None:
            if self._directml:
                from bulletbot.ocr.isolated_runtime import IsolatedRapidOcr
                self._engine = IsolatedRapidOcr("trade_v6", balance_label_worker=True)
            else:
                from bulletbot.ocr.favorites_ocr import _RapidOcrV6Compat
                self._engine = _RapidOcrV6Compat(prefer_directml=False)
            report_ocr_runtime("余额标签OCR：PP-OCRv6中文识别；" +
                               ("独立常驻DirectML进程" if self._directml else "CPU"))
        result = self._engine(image, use_det=False, use_cls=False)
        if self._directml:
            rows = list(zip(result.txts, result.scores))
        else:
            rows = result[0] or []
        return max((OcrText(str(text), float(score)) for text, score in rows),
                   key=lambda read: read.confidence, default=None)
