"""Explain private-resource omissions before starting any game operations."""
from bulletbot.mail_storage.paths import resource_path


def require_local_resources():
    required = (
        'assets/mail_storage/templates/manifest.json',
        'assets/mail_storage/templates/real_flow_manifest.json',
        'assets/ammunition_names.csv',
        'models/general_ocr/PP-OCRv6_det_small.onnx',
        'models/general_ocr/PP-OCRv6_rec_small.onnx',
        'models/general_ocr/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx',
        'models/mail_storage/ocr/rapidocr/en_PP-OCRv5_rec_mobile.onnx',
    )
    missing = [p for p in required if not resource_path(*p.split('/')).is_file()]
    if missing:
        raise ValueError('这是纯源码公开版，未附带模板或模型。请按 RESOURCES.md 自行准备资源。缺少：' + '；'.join(missing))
