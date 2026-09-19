from __future__ import annotations

from collections import deque
from collections.abc import Callable

from .models import (
    GamePageId,
    NavigationAction,
    NavigationTraceEntry,
)


FailureCapture = Callable[[str], str | None]


class NavigationJournal:
    """Bounded history shared by successful and failed navigation results."""

    def __init__(
        self,
        *,
        max_entries: int = 64,
        failure_capture: FailureCapture | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("导航历史容量必须至少为 1")
        self._entries: deque[NavigationTraceEntry] = deque(maxlen=max_entries)
        self._sequence = 0
        self._failure_capture = failure_capture

    @property
    def entries(self) -> tuple[NavigationTraceEntry, ...]:
        return tuple(self._entries)

    @property
    def sequence(self) -> int:
        return self._sequence

    def entries_since(self, sequence: int) -> tuple[NavigationTraceEntry, ...]:
        return tuple(entry for entry in self._entries if entry.sequence > sequence)

    def record(
        self,
        event: str,
        message: str,
        *,
        page_id: GamePageId | None = None,
        action: NavigationAction | None = None,
        attempt: int = 0,
        window_generation: int = 0,
    ) -> NavigationTraceEntry:
        self._sequence += 1
        entry = NavigationTraceEntry(
            sequence=self._sequence,
            event=event,
            message=message,
            page_id=page_id,
            action=action,
            attempt=attempt,
            window_generation=window_generation,
        )
        self._entries.append(entry)
        return entry

    def capture_failure(self, label: str) -> str | None:
        if self._failure_capture is None:
            return None
        try:
            return self._failure_capture(label)
        except Exception as exc:
            self.record("diagnostic_error", f"保存导航失败截图时出错：{exc}")
            return None
