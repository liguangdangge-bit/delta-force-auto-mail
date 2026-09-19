from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ReferenceLayout:
    """Map the game's 1920x1080 UI anchors into the live client area."""

    width: int
    height: int

    REFERENCE_WIDTH: ClassVar[int] = 1920
    REFERENCE_HEIGHT: ClassVar[int] = 1080

    @property
    def ui_scale(self) -> float:
        return min(
            self.width / self.REFERENCE_WIDTH,
            self.height / self.REFERENCE_HEIGHT,
        )

    def x(self, relative_x: float) -> int:
        return round(self.width * relative_x)

    def top_y(self, reference_relative_y: float) -> int:
        return round(
            self.REFERENCE_HEIGHT * reference_relative_y * self.ui_scale
        )

    def bottom_y(self, reference_relative_y: float) -> int:
        bottom_margin = (
            self.REFERENCE_HEIGHT
            * (1.0 - reference_relative_y)
            * self.ui_scale
        )
        return round(self.height - bottom_margin)

    def top_point(
        self,
        relative_x: float,
        reference_relative_y: float,
    ) -> tuple[int, int]:
        return self.x(relative_x), self.top_y(reference_relative_y)

    def top_box(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = bounds
        return (
            self.x(left),
            self.top_y(top),
            self.x(right),
            self.top_y(bottom),
        )

    def stretched_box(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = bounds
        return (
            self.x(left),
            self.top_y(top),
            self.x(right),
            self.bottom_y(bottom),
        )
