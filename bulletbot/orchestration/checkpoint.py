from __future__ import annotations

import json
import os
from pathlib import Path

from .models import SessionCheckpoint


class SessionCheckpointStore:
    """Persist one orchestration checkpoint with an atomic file replacement."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> SessionCheckpoint | None:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取会话检查点：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("会话检查点根节点必须是对象")
        return SessionCheckpoint.from_dict(value)

    def save(self, checkpoint: SessionCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    checkpoint.to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def delete(self) -> None:
        """Remove the durable session after an explicit operator reset."""

        self.path.unlink(missing_ok=True)
