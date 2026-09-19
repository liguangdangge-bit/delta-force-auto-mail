"""Resolve copied models in this project, including OCR worker processes."""
import sys
from pathlib import Path


def general_model_params():
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
    folder = root / 'models' / 'general_ocr'
    models = {
        'Det.model_path': folder / 'PP-OCRv6_det_small.onnx',
        'Rec.model_path': folder / 'PP-OCRv6_rec_small.onnx',
        'Cls.model_path': folder / 'ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx',
    }
    for path in models.values():
        if not path.is_file():
            raise FileNotFoundError(f'独立卡邮件模型缺失：{path}')
    return {key: str(path) for key, path in models.items()}
