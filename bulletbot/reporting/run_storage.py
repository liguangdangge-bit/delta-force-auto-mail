from __future__ import annotations

from pathlib import Path
import re
import shutil
import threading


MAX_RUN_DIRECTORIES = 15
MAX_SCREENSHOTS_PER_RUN = 30
_RUN_DIRECTORY_PATTERN = re.compile(r"\d{8}_\d{6}_\d{6}")
_SCREENSHOT_RETENTION_LOCK = threading.RLock()


def cleanup_run_directories(
    runs_root: Path,
    *,
    max_runs: int = MAX_RUN_DIRECTORIES,
    protected: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Keep only recent timestamped run directories under ``runs_root``."""

    root = runs_root.expanduser().resolve()
    if not root.is_dir() or max_runs < 0:
        return ()
    protected_paths = {path.expanduser().resolve() for path in protected}
    try:
        runs = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and _RUN_DIRECTORY_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: path.name,
            reverse=True,
        )
    except OSError:
        return ()

    retained = set(protected_paths)
    for path in runs:
        if len(retained) >= max_runs:
            break
        retained.add(path.resolve())
    removed: list[Path] = []
    for path in reversed(runs):
        try:
            resolved = path.resolve()
            if resolved in retained or resolved.parent != root:
                continue
            shutil.rmtree(resolved)
            removed.append(path)
        except OSError:
            continue
    return tuple(removed)


def rotate_screenshots(
    run_directory: Path,
    *,
    max_images: int = MAX_SCREENSHOTS_PER_RUN,
    protected: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Replace the oldest PNGs once a run reaches its screenshot limit."""

    if max_images < 0:
        return ()
    with _SCREENSHOT_RETENTION_LOCK:
        root = run_directory.expanduser().resolve()
        if not root.is_dir():
            return ()
        try:
            images = [
                (path, path.stat().st_mtime_ns)
                for path in root.glob("*.png")
                if path.is_file() and not path.is_symlink()
            ]
            images.sort(key=lambda item: (item[1], item[0].name), reverse=True)
        except OSError:
            return ()

        protected_paths = {
            path.expanduser().resolve()
            for path in protected
            if path is not None
        }
        retained = set(protected_paths)
        for path, _modified_at in images:
            if len(retained) >= max_images:
                break
            retained.add(path.resolve())

        removed: list[Path] = []
        for path, _modified_at in reversed(images):
            try:
                resolved = path.resolve()
                if resolved in retained or resolved.parent != root:
                    continue
                resolved.unlink()
                removed.append(path)
            except OSError:
                continue
        return tuple(removed)
