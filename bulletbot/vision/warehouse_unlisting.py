from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import Rect
from bulletbot.vision.reference_layout import ReferenceLayout


@dataclass(frozen=True)
class UnlistControlMatch:
    bounds: Rect
    confidence: float


@dataclass(frozen=True)
class UnlistDialogSnapshot:
    header: UnlistControlMatch | None
    confirm_action: UnlistControlMatch | None

    @property
    def confirmed(self) -> bool:
        return self.header is not None and self.confirm_action is not None


@dataclass(frozen=True)
class _TemplateSpec:
    filename: str
    search_bounds: tuple[float, float, float, float]
    threshold: float


class WarehouseUnlistingDetector:
    """Locate the first sell order and the modal's irreversible action."""

    _FIRST_ORDER = _TemplateSpec(
        "first_order_unlist_button.png",
        (0.32, 0.22, 0.46, 0.31),
        0.86,
    )
    _DIALOG_HEADER = _TemplateSpec(
        "unlist_dialog_header.png",
        (0.22, 0.22, 0.48, 0.34),
        0.84,
    )
    _DIALOG_CONFIRM = _TemplateSpec(
        "unlist_confirm_button.png",
        (0.48, 0.60, 0.69, 0.72),
        0.86,
    )

    def __init__(self, template_directory: Path | None = None) -> None:
        self._template_directory = template_directory or self.default_template_directory()
        self._templates: dict[str, np.ndarray | None] = {}

    @staticmethod
    def default_template_directory() -> Path:
        if getattr(sys, "frozen", False):
            root = Path(getattr(sys, "_MEIPASS"))
        else:
            root = Path(__file__).resolve().parents[2]
        return root / "assets" / "warehouse_listing_templates"

    def find_first_unlist_action(
        self,
        image: Image.Image,
    ) -> UnlistControlMatch | None:
        return self._match(image, self._FIRST_ORDER)

    def read_unlist_dialog(self, image: Image.Image) -> UnlistDialogSnapshot:
        return UnlistDialogSnapshot(
            header=self._match(image, self._DIALOG_HEADER),
            confirm_action=self._match(image, self._DIALOG_CONFIRM),
        )

    def _match(
        self,
        image: Image.Image,
        spec: _TemplateSpec,
    ) -> UnlistControlMatch | None:
        template = self._load_template(spec.filename)
        if template is None:
            return None
        layout = ReferenceLayout(image.width, image.height)
        scale = layout.ui_scale
        scaled_width = max(1, round(template.shape[1] * scale))
        scaled_height = max(1, round(template.shape[0] * scale))
        if (scaled_width, scaled_height) != (template.shape[1], template.shape[0]):
            template = cv2.resize(
                template,
                (scaled_width, scaled_height),
                interpolation=(cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC),
            )

        left, top, right, bottom = layout.top_box(spec.search_bounds)
        frame = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        search = frame[top:bottom, left:right]
        if (
            search.size == 0
            or search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return None
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _minimum, maximum, _minimum_at, maximum_at = cv2.minMaxLoc(scores)
        if maximum < spec.threshold:
            return None
        match_left = left + maximum_at[0]
        match_top = top + maximum_at[1]
        return UnlistControlMatch(
            bounds=Rect(
                match_left,
                match_top,
                template.shape[1],
                template.shape[0],
            ),
            confidence=float(maximum),
        )

    def _load_template(self, filename: str) -> np.ndarray | None:
        if filename not in self._templates:
            path = self._template_directory / filename
            try:
                payload = np.fromfile(path, dtype=np.uint8)
            except OSError:
                payload = np.empty(0, dtype=np.uint8)
            template = (
                cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
                if payload.size
                else None
            )
            self._templates[filename] = template
        return self._templates[filename]
