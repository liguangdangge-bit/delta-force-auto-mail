from __future__ import annotations

from bulletbot.ocr.local_models import general_model_params

import atexit
import multiprocessing
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


WORKER_TIMEOUT_SECONDS = 60.0


class OcrWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrWorkerStatus:
    process_id: int | None
    restart_count: int
    last_exit_code: int | None


def _normalize_result(result: Any) -> dict[str, object]:
    if isinstance(result, tuple):
        result = result[0]
    if result is None:
        return {"txts": [], "scores": [], "boxes": []}

    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    boxes = getattr(result, "boxes", None)
    if txts is not None:
        return {
            "txts": list(txts),
            "scores": [
                float(value) for value in (scores if scores is not None else [])
            ],
            "boxes": [
                value.tolist() if hasattr(value, "tolist") else value
                for value in (boxes if boxes is not None else [])
            ],
        }

    rows = list(result) if isinstance(result, (list, tuple)) else []
    return {
        "txts": [str(row[1]) for row in rows],
        "scores": [float(row[2]) for row in rows],
        "boxes": [row[0] for row in rows],
    }


def _build_engine(profile: str, model_root: str | None):
    from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

    from bulletbot.ocr.runtime_backend import (
        configure_rapidocr_directml_session_options,
    )

    configure_rapidocr_directml_session_options()
    if profile == "fast_price_v5":
        if not model_root:
            raise ValueError("fast_price_v5 requires model_root")
        from bulletbot.ocr.recognition_only import RecognitionOnlyOcr

        return RecognitionOnlyOcr(
            Path(model_root) / "en_PP-OCRv5_rec_mobile.onnx",
            use_directml=True,
        )
    if profile == "trade_v6":
        return RapidOCR(
            params={
                **general_model_params(),
                "Global.log_level": "error",
                "Global.use_cls": False,
                "EngineConfig.onnxruntime.use_dml": True,
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
    raise ValueError(f"Unknown OCR worker profile: {profile}")


def _worker_main(connection) -> None:
    engine_key: tuple[str, str | None] | None = None
    engine = None
    crash_next_inference = False
    try:
        while True:
            request = connection.recv()
            if request.get("command") == "stop":
                return
            if request.get("command") == "crash_next_for_diagnostic":
                crash_next_inference = True
                connection.send({"ok": True})
                continue
            if request.get("command") != "infer":
                raise ValueError("Unknown OCR worker command")
            if crash_next_inference:
                import os

                os._exit(-1073741819)

            profile = str(request["profile"])
            model_root_value = request.get("model_root")
            model_root = str(model_root_value) if model_root_value else None
            key = (profile, model_root)
            if engine is None or engine_key != key:
                engine = _build_engine(*key)
                engine_key = key

            engine.text_det.limit_side_len = int(request["detector_min_side_len"])
            call_options = {
                "use_det": bool(request["use_det"]),
                "use_rec": True,
            }
            use_cls = request.get("use_cls")
            if use_cls is not None:
                call_options["use_cls"] = bool(use_cls)
            result = engine(request["image"], **call_options)
            connection.send({"ok": True, "result": _normalize_result(result)})
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


class OcrWorkerClient:
    def __init__(self, *, timeout_seconds: float = WORKER_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process = None
        self._connection = None
        self._restart_count = 0
        self._last_exit_code = None
        self._active_key: tuple[str, str | None] | None = None

    @property
    def status(self) -> OcrWorkerStatus:
        with self._lock:
            process = self._process
            return OcrWorkerStatus(
                process_id=process.pid if process is not None else None,
                restart_count=self._restart_count,
                last_exit_code=self._last_exit_code,
            )

    def infer(self, request: dict[str, object]) -> dict[str, object]:
        with self._lock:
            model_root_value = request.get("model_root")
            requested_key = (
                str(request["profile"]),
                str(model_root_value) if model_root_value else None,
            )
            if (
                self._active_key is not None
                and self._active_key != requested_key
            ):
                # DirectML does not reliably return all allocations when an
                # ONNX session is deleted. Restarting at a profile boundary is
                # the only dependable way to discard the previous model.
                self._restart_locked(count_restart=False)
            first_error: BaseException | None = None
            for attempt in range(2):
                try:
                    result = self._request_locked(request)
                    self._active_key = requested_key
                    return result
                except (BrokenPipeError, EOFError, OSError, OcrWorkerError) as exc:
                    first_error = first_error or exc
                    self._restart_locked()
                    if attempt == 0:
                        continue
            raise OcrWorkerError(
                f"DirectML OCR worker failed twice: {first_error}"
            ) from first_error

    def _request_locked(self, request: dict[str, object]) -> dict[str, object]:
        self._ensure_started_locked()
        connection = self._connection
        process = self._process
        if connection is None or process is None:
            raise OcrWorkerError("DirectML OCR worker did not start")
        connection.send({"command": "infer", **request})
        if not connection.poll(self._timeout_seconds):
            raise OcrWorkerError(
                f"DirectML OCR worker timed out after {self._timeout_seconds:g}s"
            )
        response = connection.recv()
        if not response.get("ok"):
            raise OcrWorkerError(str(response.get("error", "unknown worker error")))
        return dict(response["result"])

    def _ensure_started_locked(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._restart_locked(count_restart=self._process is not None)

    def _restart_locked(self, *, count_restart: bool = True) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        self._active_key = None
        if connection is not None:
            connection.close()
        if process is not None:
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.exitcode not in (None, 0):
                self._last_exit_code = process.exitcode
        if count_restart:
            self._restart_count += 1

        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child_connection,),
            name="DeltaForceBulletBot-DirectML-OCR",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def crash_next_inference_for_diagnostic(self) -> None:
        with self._lock:
            self._ensure_started_locked()
            connection = self._connection
            process = self._process
            if connection is None or process is None:
                raise OcrWorkerError("DirectML OCR worker did not start")
            connection.send({"command": "crash_next_for_diagnostic"})
            if not connection.poll(5.0):
                raise OcrWorkerError("Could not arm diagnostic OCR worker crash")
            response = connection.recv()
            if not response.get("ok"):
                raise OcrWorkerError("Could not arm diagnostic OCR worker crash")

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            process = self._process
            self._connection = None
            self._process = None
            if connection is not None:
                try:
                    connection.send({"command": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                connection.close()
            if process is not None:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)


_client_lock = threading.Lock()
_client: OcrWorkerClient | None = None
_balance_label_client: OcrWorkerClient | None = None
_numeric_client: OcrWorkerClient | None = None


def get_numeric_worker_client() -> OcrWorkerClient:
    """Navigation OCR must not discard the prewarmed balance numeric model."""
    global _numeric_client
    with _client_lock:
        if _numeric_client is None:
            _numeric_client = OcrWorkerClient()
        return _numeric_client


def get_balance_label_worker_client() -> OcrWorkerClient:
    """Keep the label model resident without replacing the numeric worker."""
    global _balance_label_client
    with _client_lock:
        if _balance_label_client is None:
            _balance_label_client = OcrWorkerClient()
        return _balance_label_client


def get_ocr_worker_client() -> OcrWorkerClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = OcrWorkerClient()
        return _client


def close_ocr_worker_client() -> None:
    global _client, _balance_label_client, _numeric_client
    with _client_lock:
        client = _client
        _client = None
        label_client = _balance_label_client
        _balance_label_client = None
        numeric_client = _numeric_client
        _numeric_client = None
    if client is not None:
        client.close()
    if label_client is not None:
        label_client.close()
    if numeric_client is not None:
        numeric_client.close()


class IsolatedRapidOcr:
    def __init__(
        self,
        profile: str,
        *,
        model_root: Path | None = None,
        detector_min_side_len: int = 736,
        balance_label_worker: bool = False,
        numeric_worker: bool = False,
    ) -> None:
        self._profile = profile
        self._balance_label_worker = balance_label_worker
        self._numeric_worker = numeric_worker
        self._model_root = str(model_root) if model_root is not None else None
        self.text_det = SimpleNamespace(limit_side_len=detector_min_side_len)

    def __call__(
        self,
        image: np.ndarray,
        *,
        use_det: bool = True,
        use_cls: bool | None = None,
        use_rec: bool = True,
        **_kwargs,
    ):
        if not use_rec:
            raise ValueError("OCR worker requires use_rec=True")
        client = (get_numeric_worker_client() if self._numeric_worker else
                  get_balance_label_worker_client() if self._balance_label_worker else
                  get_ocr_worker_client())
        restart_count_before = client.status.restart_count
        result = client.infer(
            {
                "profile": self._profile,
                "model_root": self._model_root,
                "image": np.asarray(image),
                "use_det": use_det,
                "use_cls": use_cls,
                "detector_min_side_len": self.text_det.limit_side_len,
            }
        )
        restart_count_after = client.status.restart_count
        if restart_count_after > restart_count_before:
            from bulletbot.ocr.runtime_backend import report_ocr_runtime

            report_ocr_runtime(
                "DirectML OCR 工作进程异常退出，已自动重建并完成本次重试",
                "WARNING",
            )
        return SimpleNamespace(
            txts=np.asarray(result["txts"]),
            scores=np.asarray(result["scores"], dtype=float),
            boxes=np.asarray(result["boxes"]),
        )


atexit.register(close_ocr_worker_client)
