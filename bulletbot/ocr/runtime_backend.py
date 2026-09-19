from __future__ import annotations

import os
import threading
from collections.abc import Callable


DIRECTML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
DIRECTML_FORCE_CPU_ENV = "DELTAFORCE_BULLETBOT_FORCE_CPU"

_report_lock = threading.Lock()
_reporter: Callable[[str, str], None] | None = None
_reported_messages: set[str] = set()
OCR_RUNTIME_LOCK = threading.RLock()
_rapidocr_options_lock = threading.Lock()
_rapidocr_options_configured = False


def available_onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort

        return tuple(ort.get_available_providers())
    except Exception:
        return ()


def directml_available() -> bool:
    return DIRECTML_PROVIDER in available_onnx_providers()


def directml_enabled() -> bool:
    """Prefer DirectML unless CPU mode was explicitly requested."""

    force_cpu = os.environ.get(DIRECTML_FORCE_CPU_ENV, "").strip().casefold()
    return force_cpu not in {"1", "true", "yes", "on"} and directml_available()


def apply_directml_session_options(session_options, *, use_directml: bool) -> None:
    """Apply the session settings required by the DirectML provider."""

    if not use_directml:
        return
    from onnxruntime import ExecutionMode

    session_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_mem_pattern = False


def configure_rapidocr_directml_session_options() -> None:
    """Add DirectML-safe SessionOptions missing from RapidOCR 3.9.x."""

    global _rapidocr_options_configured
    with _rapidocr_options_lock:
        if _rapidocr_options_configured:
            return

        from rapidocr.inference_engine.onnxruntime.main import OrtInferSession

        original = OrtInferSession._init_sess_opts
        if getattr(original, "_bulletbot_directml_safe", False):
            _rapidocr_options_configured = True
            return

        def initialize_session_options(config):
            session_options = original(config)
            apply_directml_session_options(
                session_options,
                use_directml=bool(config.get("use_dml", False)),
            )
            return session_options

        initialize_session_options._bulletbot_directml_safe = True
        OrtInferSession._init_sess_opts = staticmethod(initialize_session_options)
        _rapidocr_options_configured = True


def configure_ocr_runtime_reporter(
    reporter: Callable[[str, str], None] | None,
) -> None:
    global _reporter
    with _report_lock:
        _reporter = reporter


def report_ocr_runtime(message: str, level: str = "INFO") -> None:
    with _report_lock:
        if message in _reported_messages:
            return
        _reported_messages.add(message)
        reporter = _reporter
    if reporter is not None:
        reporter(message, level)


def report_ocr_capabilities() -> None:
    providers = available_onnx_providers()
    if DIRECTML_PROVIDER in providers:
        if not directml_enabled():
            report_ocr_runtime(
                "OCR 对照测试模式：已按环境设置强制使用 CPU",
                "WARNING",
            )
            return
        report_ocr_runtime(
            "OCR 加速：检测到 DirectML，自动优先使用 GPU；"
            "独立 OCR 进程保护已启用；初始化或推理失败时自动回退 CPU"
        )
        return
    provider_text = ", ".join(providers) if providers else "未检测到 ONNX Runtime"
    report_ocr_runtime(
        f"OCR 加速：DirectML 不可用，使用 CPU（可用后端：{provider_text}）",
        "WARNING",
    )


def provider_label(use_directml: bool) -> str:
    return "DirectML GPU" if use_directml else "CPU"
