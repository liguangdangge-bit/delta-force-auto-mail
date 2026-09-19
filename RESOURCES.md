# 本地资源准备

这个仓库没有作者的模板图片、模板清单和模型，也不会自动下载作者的资源。以下目录属于本地运行依赖，均已加入 `.gitignore`。

| 本地路径 | 用途 |
| --- | --- |
| `assets/mail_storage/templates/manifest.json` | 卡邮件页面和模板定义 |
| `assets/mail_storage/templates/real_flow_manifest.json` | 实际流程的补充模板定义 |
| `assets/mail_storage/templates/` 下的图片 | 模板清单引用的游戏、启动器和资源管理画面 |
| `assets/page_templates/` | 公共页面识别 |
| `assets/inventory_templates/`、`assets/warehouse_listing_templates/`、`assets/balance_tooltip/` | 共用识别和恢复模块依赖 |
| `assets/ammunition_names.csv` | 公共名称识别数据 |
| `models/general_ocr/PP-OCRv6_det_small.onnx` | 通用文字检测 |
| `models/general_ocr/PP-OCRv6_rec_small.onnx` | 通用文字识别 |
| `models/general_ocr/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx` | 方向分类 |
| `models/mail_storage/ocr/rapidocr/en_PP-OCRv5_rec_mobile.onnx` | 数字行识别 |
| `config/` | 自己电脑上的窗口指纹、PAK 路径和运行参数 |

自行取得模型时，应核对模型提供方的使用条款和版本。私有图片、模型及其他第三方资源不因本仓库公开而获准使用。

模板清单的读取逻辑见 `bulletbot/mail_storage/vision.py` 中的 `TemplateCatalog`。清单包含 `pages` 和 `templates`；每个模板定义名称、页面、图片文件路径、参考分辨率、搜索范围与匹配阈值，可选点击位置。实际必需的模板名称和页面分支由工作流代码决定。

仅提供名称或占位清单无法复原作者的模板。未完成这些资源的准备与适配前，完整游戏流程不能运行。

本地环境保存后配置写入本目录；PAK 暂存、恢复等操作需要实际游戏文件。不要同时使用其他程序操作同一个游戏和 PAK 文件。此公开版本未在每位使用者的机器与游戏环境中验证。
