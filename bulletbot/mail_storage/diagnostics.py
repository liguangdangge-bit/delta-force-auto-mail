from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bulletbot.reporting.run_storage import rotate_screenshots

from .paths import (
    MAX_RETAINED_RUNS,
    RUN_RETENTION_DAYS,
    cleanup_run_data,
    run_data_dir,
)


MAX_PRICE_TRACE_IMAGES = 100


class RunDiagnostics:
    def __init__(self, root: Path | None = None, *, save_images: bool = True) -> None:
        """Create diagnostics in ``root`` when the host owns a run directory.

        Standalone MailStorageWorkflow users keep the historical per-run data
        location. The main application passes its RunRecorder directory so
        structured events and diagnostic screenshots can be opened together.
        In that hosted mode, RunRecorder remains the only writer of run.log.
        """

        self.save_images = save_images
        standalone = root is None
        removed_runs = cleanup_run_data() if standalone else ()
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = (
            run_data_dir(self.run_id)
            if standalone
            else Path(root).expanduser().resolve()
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.price_trace_root = self.root / "price_trace"
        if save_images:
            self.price_trace_root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self._lock = threading.Lock()
        self.logger: logging.Logger | None = None
        if standalone:
            self.logger = logging.getLogger(
                f"bulletbot.mail_storage.{self.run_id}.{id(self)}"
            )
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            handler = logging.FileHandler(self.root / "run.log", encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            self.logger.addHandler(handler)
        if removed_runs:
            self.event(
                "retention_cleanup",
                removed_count=len(removed_runs),
                max_age_days=RUN_RETENTION_DAYS,
                max_runs=MAX_RETAINED_RUNS,
            )

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "type": event_type,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.logger is not None:
            self.logger.info("%s %s", event_type, payload)

    def save_frame(self, name: str, image: np.ndarray) -> Path:
        if not self.save_images:
            return self.root / "images-disabled"
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
        path = self.root / f"{datetime.now().strftime('%H%M%S-%f')}_{safe_name}.png"
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("无法编码诊断截图")
        encoded.tofile(path)
        rotate_screenshots(self.root, protected=(path,))
        return path

    def save_price_trace_frame(self, name: str, image: np.ndarray) -> Path:
        """Save a small price-region frame in its bounded trace directory."""

        if not self.save_images:
            return self.root / "images-disabled"
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("价格追踪截图为空")
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in name
        )
        path = self.price_trace_root / (
            f"{datetime.now().strftime('%H%M%S-%f')}_{safe_name}.png"
        )
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("无法编码价格追踪截图")
        encoded.tofile(path)
        try:
            trace_count = sum(
                1
                for candidate in self.price_trace_root.glob("*.png")
                if candidate.is_file() and not candidate.is_symlink()
            )
        except OSError:
            trace_count = MAX_PRICE_TRACE_IMAGES + 1
        if trace_count > MAX_PRICE_TRACE_IMAGES:
            rotate_screenshots(
                self.price_trace_root,
                max_images=MAX_PRICE_TRACE_IMAGES,
                protected=(path,),
            )
        return path

    def append_demo_index(self, line: str) -> Path:
        path = self.root / "演示节点.txt"
        with self._lock:
            with path.open("a", encoding="utf-8-sig") as handle:
                handle.write(line.rstrip() + "\n")
        return path

    def close(self) -> None:
        if self.logger is None:
            return
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
