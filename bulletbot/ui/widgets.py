from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    """Keep mouse-wheel scrolling from changing a selected option."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Keep mouse-wheel scrolling from changing a numeric value."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Keep mouse-wheel scrolling from changing a decimal value."""

    def wheelEvent(self, event) -> None:
        event.ignore()
