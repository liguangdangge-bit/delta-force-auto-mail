from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .paths import user_data_dir


@dataclass
class PakFileRecord:
    source: str
    staged: str
    size: int


@dataclass
class PakRecord:
    files: list[PakFileRecord]
    phase: str
    updated_at: float


@dataclass(frozen=True)
class PakMoveTestResult:
    size: int
    elapsed_seconds: float


class PakTransactionError(RuntimeError):
    pass


class PakTransaction:
    def __init__(
        self,
        source: Path | Sequence[Path],
        staged: Path | Sequence[Path],
        enabled: bool = False,
        record_path: Path | None = None,
    ) -> None:
        self.sources = self._coerce_paths(source)
        self.staged_paths = self._coerce_paths(staged)
        if not self.sources or len(self.sources) > 2:
            raise PakTransactionError("PAK 文件数量必须为 1 或 2")
        if len(self.sources) != len(self.staged_paths):
            raise PakTransactionError("PAK 原路径与暂存路径数量不一致")
        if len(set(self.sources)) != len(self.sources):
            raise PakTransactionError("PAK 原路径不能重复")
        if len(set(self.staged_paths)) != len(self.staged_paths):
            raise PakTransactionError("PAK 暂存路径不能重复")
        self.source = self.sources[0]
        self.staged = self.staged_paths[0]
        self.enabled = enabled
        self.record_path = record_path or user_data_dir() / "recovery.json"

    @property
    def source_display(self) -> str:
        return "；".join(str(path) for path in self.sources)

    @property
    def staged_display(self) -> str:
        return "；".join(str(path) for path in self.staged_paths)

    def inspect(self) -> str:
        states = [
            (source.exists(), staged.exists())
            for source, staged in zip(self.sources, self.staged_paths)
        ]
        if all(source_exists and not staged_exists for source_exists, staged_exists in states):
            return "ready"
        if all(staged_exists and not source_exists for source_exists, staged_exists in states):
            return "staged"
        if any(source_exists and staged_exists for source_exists, staged_exists in states):
            return "conflict"
        if any(staged_exists for _source_exists, staged_exists in states):
            return "partial"
        return "missing"

    def stage(self) -> None:
        self._require_enabled()
        status = self.inspect()
        if status == "staged":
            return
        if status != "ready":
            raise PakTransactionError(f"无法暂存 PAK，当前状态：{status}")
        for source, staged in zip(self.sources, self.staged_paths):
            if source.parent.drive.casefold() != staged.parent.drive.casefold():
                raise PakTransactionError("PAK 原路径与暂存路径必须位于同一磁盘")
            staged.parent.mkdir(parents=True, exist_ok=True)
        record = self._current_record("staging")
        self._write_record(record)
        moved: list[tuple[Path, Path]] = []
        try:
            for source, staged in zip(self.sources, self.staged_paths):
                self._move_with_retry(source, staged)
                moved.append((source, staged))
        except Exception as exc:
            rollback_errors: list[str] = []
            for source, staged in reversed(moved):
                try:
                    self._move_with_retry(staged, source)
                except Exception as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            if not rollback_errors and self.inspect() == "ready":
                self._clear_record()
            else:
                record.phase = "rollback_required"
                record.updated_at = time.time()
                self._write_record(record)
            detail = (
                f"；回滚失败：{'；'.join(rollback_errors)}"
                if rollback_errors
                else "；已回滚已移动文件"
            )
            raise PakTransactionError(f"暂存 PAK 文件组失败：{exc}{detail}") from exc
        record.phase = "staged"
        record.updated_at = time.time()
        self._write_record(record)

    def restore(self) -> None:
        self._require_enabled()
        status = self.inspect()
        if status == "ready":
            self._clear_record()
            return
        if status not in {"staged", "partial"}:
            raise PakTransactionError(f"无法恢复 PAK，当前状态：{status}")
        self._write_record(self._current_record("restoring"))
        for source, staged in zip(self.sources, self.staged_paths):
            source_exists = source.exists()
            staged_exists = staged.exists()
            if source_exists and not staged_exists:
                continue
            if staged_exists and not source_exists:
                source.parent.mkdir(parents=True, exist_ok=True)
                self._move_with_retry(staged, source)
                continue
            state = "conflict" if source_exists and staged_exists else "missing"
            raise PakTransactionError(
                f"无法恢复 PAK 文件 {source.name}，当前状态：{state}"
            )
        if self.inspect() != "ready":
            raise PakTransactionError("PAK 文件组恢复后状态异常")
        self._clear_record()

    def recover_if_needed(self) -> bool:
        record = self._read_record()
        if record is not None:
            recorded = PakTransaction(
                [Path(item.source) for item in record.files],
                [Path(item.staged) for item in record.files],
                enabled=self.enabled,
                record_path=self.record_path,
            )
            if recorded.inspect() == "ready":
                recorded._clear_record()
                return False
            recorded.restore()
            return True
        if self.inspect() not in {"staged", "partial"}:
            return False
        self.restore()
        return True

    def self_test(self, hold_seconds: float = 0.0) -> PakMoveTestResult:
        self._require_enabled()
        status = self.inspect()
        if status != "ready":
            raise PakTransactionError(f"无法开始 PAK 移动自检，当前状态：{status}")

        expected_sizes = tuple(source.stat().st_size for source in self.sources)
        started = time.monotonic()
        try:
            self.stage()
            staged_sizes = tuple(path.stat().st_size for path in self.staged_paths)
            if self.inspect() != "staged" or staged_sizes != expected_sizes:
                raise PakTransactionError("PAK 暂存后的文件状态或大小不正确")
            if hold_seconds > 0:
                time.sleep(hold_seconds)
        finally:
            current = self.inspect()
            if current in {"staged", "partial"}:
                self.restore()
            elif current == "ready" and self.record_path.exists():
                self._clear_record()

        restored_sizes = tuple(source.stat().st_size for source in self.sources)
        if self.inspect() != "ready" or restored_sizes != expected_sizes:
            raise PakTransactionError("PAK 自检结束后未正确恢复到原路径")
        return PakMoveTestResult(
            size=sum(expected_sizes),
            elapsed_seconds=time.monotonic() - started,
        )

    @staticmethod
    def _coerce_paths(value: Path | Sequence[Path]) -> tuple[Path, ...]:
        if isinstance(value, (str, os.PathLike)):
            return (Path(value),)
        return tuple(Path(path) for path in value)

    def _current_record(self, phase: str) -> PakRecord:
        files = []
        for source, staged in zip(self.sources, self.staged_paths):
            size = (
                source.stat().st_size
                if source.exists()
                else staged.stat().st_size
                if staged.exists()
                else 0
            )
            files.append(PakFileRecord(str(source), str(staged), size))
        return PakRecord(files=files, phase=phase, updated_at=time.time())

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise PakTransactionError("真实 PAK 文件操作尚未启用")

    @staticmethod
    def _move_with_retry(source: Path, target: Path, attempts: int = 10) -> None:
        last_error: OSError | None = None
        for attempt in range(attempts):
            try:
                os.replace(source, target)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.25 * (attempt + 1), 2.0))
        raise PakTransactionError(f"移动文件失败：{last_error}")

    def _write_record(self, record: PakRecord) -> None:
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.record_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(asdict(record), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.record_path)

    def _read_record(self) -> PakRecord | None:
        if not self.record_path.exists():
            return None
        try:
            with self.record_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            raw_files = value.get("files")
            if raw_files is None:
                raw_files = [value]
            return PakRecord(
                files=[
                    PakFileRecord(
                        source=str(item["source"]),
                        staged=str(item["staged"]),
                        size=int(item["size"]),
                    )
                    for item in raw_files
                ],
                phase=str(value["phase"]),
                updated_at=float(value["updated_at"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PakTransactionError(f"恢复记录损坏：{exc}") from exc

    def _clear_record(self) -> None:
        try:
            self.record_path.unlink()
        except FileNotFoundError:
            pass
