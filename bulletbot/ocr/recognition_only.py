from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


class RecognitionOnlyOcr:
    """Load one recognition model without detector or classifier sessions."""

    def __init__(self, model_path: Path, *, use_directml: bool) -> None:
        from rapidocr.ch_ppocr_rec import TextRecognizer
        from rapidocr.main import DEFAULT_CFG_PATH
        from rapidocr.utils.parse_parameters import ParseParams

        model = Path(model_path).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"OCR recognition model does not exist: {model}")

        cfg = ParseParams.load(DEFAULT_CFG_PATH)
        cfg.EngineConfig.onnxruntime.use_dml = bool(use_directml)
        cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
        cfg.Rec.model_path = str(model)
        cfg.Rec.model_root_dir = str(model.parent)
        cfg.Rec.font_path = cfg.Global.font_path
        cfg.Rec.rec_batch_num = 1
        self._recognizer = TextRecognizer(cfg.Rec)
        # IsolatedRapidOcr exposes this attribute for API compatibility.
        self.text_det = SimpleNamespace(limit_side_len=736)

    def __call__(
        self,
        image: np.ndarray,
        *,
        use_det: bool = False,
        use_cls: bool = False,
        use_rec: bool = True,
        **_kwargs,
    ):
        if use_det or use_cls or not use_rec:
            raise ValueError("RecognitionOnlyOcr only supports recognition-only calls")

        from rapidocr.ch_ppocr_rec import TextRecInput

        return self._recognizer(
            TextRecInput(img=np.asarray(image), return_word_box=False)
        )
