from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED


class DiagnosticPackage:
    def __init__(self, recorder):
        self._temporary = tempfile.TemporaryDirectory(prefix="bulletbot-diagnostics-")
        self.path = Path(self._temporary.name) / f"{recorder.directory.name}.zip"
        try:
            snapshot = Path(self._temporary.name) / recorder.directory.name
            required = recorder.snapshot_for_export(snapshot)
            # Snapshot each additional file at its initial size: a growing event
            # log must not keep the exporter copying indefinitely.
            for source in sorted(recorder.directory.rglob("*")):
                if source.is_symlink() or not source.is_file():
                    continue
                if source.suffix.lower() not in {".log", ".jsonl", ".csv", ".html", ".png", ".jpg", ".jpeg"}:
                    continue
                if not source.resolve().is_relative_to(recorder.directory.resolve()):
                    raise OSError(f"诊断文件位于运行目录之外：{source.name}")
                relative = source.relative_to(recorder.directory)
                if relative.as_posix() in required or source.suffix == ".tmp":
                    continue
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with source.open("rb") as reader, destination.open("wb") as writer:
                        remaining = os.fstat(reader.fileno()).st_size
                        while remaining:
                            chunk = reader.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise OSError(f"导出时文件发生截断：{relative}")
                            writer.write(chunk)
                            remaining -= len(chunk)
                except OSError as exc:
                    raise OSError(f"无法复制诊断文件 {relative}：{exc}") from exc
                if source.name == "events.jsonl":
                    # An event may still be writing its last line. Keep all
                    # completed records and exclude only that partial tail.
                    with destination.open("rb+") as handle:
                        data = handle.read()
                        handle.truncate(data.rfind(b"\n") + 1)
            files = sorted(p for p in snapshot.rglob("*") if p.is_file())
            manifest = {
                "run_id": recorder.directory.name,
                "required_files": list(required),
                "events_present": (snapshot / "events.jsonl").exists(),
                "note": "运行中快照；尚未产生事件或截图时不会包含这些文件。",
                "files": {p.relative_to(snapshot).as_posix(): p.stat().st_size for p in files},
            }
            (snapshot / "diagnostic_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            with ZipFile(self.path, "w", ZIP_DEFLATED) as archive:
                for path in sorted(snapshot.rglob("*")):
                    if path.is_file():
                        archive.write(path, f"{snapshot.name}/{path.relative_to(snapshot).as_posix()}")
            with ZipFile(self.path) as archive:
                names = set(archive.namelist())
                missing = [name for name in required if f"{snapshot.name}/{name}" not in names]
                if missing or archive.testzip() is not None:
                    raise RuntimeError(f"诊断包完整性检查失败：{missing}")
        except BaseException:
            self.cleanup()
            raise

    def save(self, destination: Path) -> None:
        """Replace the chosen destination only after a complete copy succeeds."""
        descriptor, temporary = tempfile.mkstemp(prefix=".diagnostics-", suffix=".tmp",
                                                  dir=destination.parent)
        os.close(descriptor)
        try:
            shutil.copyfile(self.path, temporary)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def cleanup(self) -> None:
        self._temporary.cleanup()
