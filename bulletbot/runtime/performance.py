from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator


@dataclass(frozen=True)
class TimingSummary:
    """Aggregated measurements for one runtime phase."""

    calls: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    max_ms: float = 0.0

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class PerformanceStats:
    """Small, dependency-free accumulator for runtime phase timings."""

    PHASES = (
        "capture_frame",
        "grid_detection",
        "name_ocr",
        "price_ocr",
        "detail_ocr",
        "balance_ocr",
        "fixed_wait",
        "debug_output",
    )

    def __init__(self) -> None:
        self._values: dict[str, TimingSummary] = {
            phase: TimingSummary() for phase in self.PHASES
        }

    def record(self, phase: str, elapsed_ms: float) -> None:
        """Record one completed measurement, ignoring unknown phase names."""

        if phase not in self._values:
            return
        elapsed = max(0.0, float(elapsed_ms))
        previous = self._values[phase]
        self._values[phase] = TimingSummary(
            calls=previous.calls + 1,
            total_ms=previous.total_ms + elapsed,
            last_ms=elapsed,
            max_ms=max(previous.max_ms, elapsed),
        )

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.record(phase, (perf_counter() - started) * 1000.0)

    def snapshot(self, *, reset: bool = False) -> dict[str, TimingSummary]:
        result = dict(self._values)
        if reset:
            self._values = {phase: TimingSummary() for phase in self.PHASES}
        return result

    @classmethod
    def format_snapshot(cls, snapshot: dict[str, TimingSummary]) -> str:
        parts: list[str] = []
        for phase in cls.PHASES:
            summary = snapshot.get(phase, TimingSummary())
            if not summary.calls:
                continue
            parts.append(
                f"{phase} n={summary.calls} avg={summary.average_ms:.1f}ms "
                f"last={summary.last_ms:.1f}ms max={summary.max_ms:.1f}ms"
            )
        return "; ".join(parts) if parts else "无计时样本"
