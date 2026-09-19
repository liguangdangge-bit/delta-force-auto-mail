# 三角洲全自动卡邮件

**Delta Force Auto Mail** · `delta-force-auto-mail` · 纯源码公开版

**抖音号：86599533954**

用于个人学习和研究，**禁止商用**。本项目采用 [PolyForm Noncommercial License 1.0.0](LICENSE)，属于附带非商业限制的源码公开项目。使用、修改或分发时请遵守完整许可，并保留 [NOTICE](NOTICE)。

## 公开内容

本仓库只提供代码、依赖清单和说明。功能包括独立卡邮件流程、窗口与 PAK 配置、暂停／停止、异常恢复、日志和运行画面预览。程序为本地版本，没有登录、授权校验、心跳或公告服务。

使用前，请根据自己的运行环境准备模板图片、模板清单、OCR 模型和本机配置，具体可参考 [资源准备说明](RESOURCES.md)。

## 流程原理与来源

本项目的卡邮件流程参考网上广为流传的“删除长工地图卡邮件”办法，将相关手动操作整理为自动化流程，供个人学习与研究。

上述说明用于交代流程来源，不代表游戏官方认可，也不对该方法是否属于漏洞利用作出结论。

## 环境与启动

Windows，Python 3.12。先安装依赖：

```powershell
.\setup_environment.ps1
```

准备好模板、识别模型和本机配置后，按 [资源准备说明](RESOURCES.md) 放入相应目录，再运行：

```powershell
.\start.ps1
```

程序需要管理员权限执行游戏输入和 PAK 文件操作。卡邮件工具自身不要求联网授权；游戏与游戏启动器仍按其自身方式联网。

首次使用时，请按代码要求配置模板名称、画面布局和参考分辨率，并完成本机适配。程序会在启动自动流程时检查所需资源，提示需要补充的内容。

## 代码结构

- `bulletbot/mail_storage`：卡邮件主流程、截图识别、窗口指纹和 PAK 事务恢复。
- `bulletbot/ui/mail_window.py`：本地卡邮件窗口、日志、预览和控制按钮。
- `bulletbot/orchestration`：启动、暂停／停止、结果与恢复状态。
- `bulletbot/ocr`、`vision`、`navigation`：识别、模型加载和页面导航。
- `bulletbot/platform`、`window`：Windows 输入、窗口和进程控制。
- `bulletbot/reporting`：运行日志、报告和诊断导出。

保留部分配装、出售等公共后端模块，因为卡邮件原流程仍导入这些依赖；没有提供相应业务页面。此仓库不需要作者原项目目录参与导入或运行。

## 开发与维护

模板、模型、个人配置和构建产物由使用者在本地管理，`.gitignore` 已配置相应规则，便于维护源码。

提交前可运行检查：

```powershell
python tools/check_public_files.py
git status --short
git diff --cached --name-only
```

请使用仓库中现成的 `LICENSE`，不要在 GitHub 创建仓库时另选 MIT、Apache 或 GPL 来替换非商业许可。
