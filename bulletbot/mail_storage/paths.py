from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path


APP_NAME = "StandaloneMailBot"
RUN_RETENTION_DAYS = 14
MAX_RETAINED_RUNS = 50
_RUN_DIRECTORY_PATTERN = re.compile(r"\d{8}-\d{6}$")


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def user_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    path = base / APP_NAME / "mail_storage"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_data_dir(run_id: str) -> Path:
    path = user_data_dir() / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_run_data(
    runs_root: Path | None = None,
    *,
    max_age_days: int = RUN_RETENTION_DAYS,
    max_runs: int = MAX_RETAINED_RUNS,
) -> tuple[Path, ...]:
    """Remove old bounded diagnostic runs without touching unrelated folders."""

    root = (runs_root or (user_data_dir() / "runs")).resolve()
    if not root.exists():
        return ()
    cutoff = time.time() - max(0, max_age_days) * 24 * 60 * 60
    try:
        runs_with_mtime = [
            (path, path.stat().st_mtime)
            for path in root.iterdir()
            if path.is_dir() and _RUN_DIRECTORY_PATTERN.fullmatch(path.name)
        ]
        runs_with_mtime.sort(key=lambda item: item[1], reverse=True)
    except OSError:
        return ()

    expired = {
        path
        for path, modified_at in runs_with_mtime
        if max_age_days >= 0 and modified_at < cutoff
    }
    retained = [path for path, _modified_at in runs_with_mtime if path not in expired]
    if max_runs >= 0:
        expired.update(retained[max_runs:])

    removed: list[Path] = []
    for path in sorted(expired, key=lambda item: item.name):
        try:
            resolved = path.resolve()
            if resolved.parent != root:
                continue
            shutil.rmtree(resolved)
            removed.append(path)
        except OSError:
            continue
    return tuple(removed)
