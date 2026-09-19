from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import InventoryMatch, Rect
from bulletbot.vision.inventory_templates import InventoryTemplateMatcher
from bulletbot.vision.reference_layout import ReferenceLayout


@dataclass(frozen=True)
class WarehouseBoxButton:
    index: int
    bounds: Rect
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.bounds.left + self.bounds.width // 2,
            self.bounds.top + self.bounds.height // 2,
        )


class WarehouseInventoryDetector:
    """Detect warehouse boxes and nominate known ammunition cells."""

    MAX_BOXES = 7
    INVENTORY_COLUMNS = 9

    _FIRST_BOX_CENTER = (1206, 202)
    _BOX_BUTTON_SIZE = (50, 48)
    _BOX_STEP = 50
    _INVENTORY_GRID = (0.6435, 0.121, 0.949, 0.910)
    _SETTINGS_SEARCH = (0.606, 0.150, 0.650, 0.720)
    _SETTINGS_TEMPLATE_THRESHOLD = 0.74
    _SETTINGS_TEMPLATE_SCALES = (0.90, 0.96, 1.0, 1.04, 1.10)
    _MIN_ICON_EDGE_COVERAGE = 0.045

    def __init__(
        self,
        *,
        template_matcher: InventoryTemplateMatcher | None = None,
        control_template_directory: Path | None = None,
    ) -> None:
        self._template_matcher = template_matcher or InventoryTemplateMatcher()
        self._control_template_directory = (
            control_template_directory or self.default_control_template_directory()
        )
        self._settings_template: np.ndarray | None | bool = False

    @staticmethod
    def default_control_template_directory() -> Path:
        if getattr(sys, "frozen", False):
            root = Path(getattr(sys, "_MEIPASS"))
        else:
            root = Path(__file__).resolve().parents[2]
        return root / "assets" / "warehouse_listing_templates"

    @staticmethod
    def is_supported_ammunition(product_name: str) -> bool:
        normalized = InventoryTemplateMatcher.normalize_product_name(product_name)
        return bool(normalized)

    def has_supported_template(self, product_name: str) -> bool:
        return bool(
            self.is_supported_ammunition(product_name)
            and self._template_matcher.has_template_for(product_name)
        )

    def detect_box_buttons(self, image: Image.Image) -> tuple[WarehouseBoxButton, ...]:
        frame = np.asarray(image.convert("RGB"))[:, :, ::-1]
        layout = ReferenceLayout(image.width, image.height)
        scale = layout.ui_scale
        first_center_x = round(self._FIRST_BOX_CENTER[0] * image.width / 1920)
        first_center_y = round(self._FIRST_BOX_CENTER[1] * scale)
        step = max(1, round(self._BOX_STEP * scale))

        settings = self._find_settings_button(frame, layout)
        if settings is not None:
            settings_center_y = settings[0].top + settings[0].height / 2
            detected_count = round((settings_center_y - first_center_y) / step)
            maximum_candidates = min(self.MAX_BOXES, max(0, detected_count))
            settings_confidence = settings[1]
        else:
            maximum_candidates = self.MAX_BOXES
            settings_confidence = 0.72

        button_width = max(20, round(self._BOX_BUTTON_SIZE[0] * scale))
        button_height = max(20, round(self._BOX_BUTTON_SIZE[1] * scale))
        buttons: list[WarehouseBoxButton] = []
        for index in range(maximum_candidates):
            center_y = first_center_y + index * step
            bounds = self._clipped_rect(
                first_center_x - button_width // 2,
                center_y - button_height // 2,
                button_width,
                button_height,
                image.width,
                image.height,
            )
            icon_score = self._button_icon_score(frame, bounds)
            if icon_score < self._MIN_ICON_EDGE_COVERAGE:
                break
            buttons.append(
                WarehouseBoxButton(
                    index=index,
                    bounds=bounds,
                    confidence=min(
                        0.99,
                        settings_confidence * 0.70
                        + min(1.0, icon_score / 0.13) * 0.30,
                    ),
                )
            )
        return tuple(buttons)

    def inventory_grid_bounds(self, image: Image.Image) -> Rect:
        left, top, right, bottom = ReferenceLayout(
            image.width,
            image.height,
        ).stretched_box(self._INVENTORY_GRID)
        return Rect(left, top, right - left, bottom - top)

    def find_ammunition_candidate(
        self,
        image: Image.Image,
        product_name: str,
        *,
        box_index: int,
        excluded_cells: set[tuple[int, int, int]] | None = None,
    ) -> InventoryMatch | None:
        if not self.has_supported_template(product_name):
            return None
        grid = self.inventory_grid_bounds(image)
        match = self._template_matcher.find_expected(
            np.asarray(image.convert("RGB"))[:, :, ::-1],
            product_name,
            grid,
            self.INVENTORY_COLUMNS,
            excluded_cells=excluded_cells,
            tab_index=box_index,
        )
        if match is None:
            return None
        return InventoryMatch(
            name=product_name,
            confidence=match.confidence,
            bounds=match.bounds,
            tab_index=box_index,
            row=match.row,
            column=match.column,
            source="warehouse_template",
        )

    def _find_settings_button(
        self,
        frame: np.ndarray,
        layout: ReferenceLayout,
    ) -> tuple[Rect, float] | None:
        template = self._load_settings_template()
        if template is None:
            return None
        left, top, right, bottom = layout.top_box(self._SETTINGS_SEARCH)
        search = frame[top:bottom, left:right]
        if search.size == 0:
            return None
        best: tuple[float, Rect] | None = None
        for factor in self._SETTINGS_TEMPLATE_SCALES:
            scale = layout.ui_scale * factor
            width = max(8, round(template.shape[1] * scale))
            height = max(8, round(template.shape[0] * scale))
            if width > search.shape[1] or height > search.shape[0]:
                continue
            resized = cv2.resize(
                template,
                (width, height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            result = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
            _minimum, score, _min_location, location = cv2.minMaxLoc(result)
            bounds = Rect(left + location[0], top + location[1], width, height)
            if best is None or score > best[0]:
                best = float(score), bounds
        if best is None or best[0] < self._SETTINGS_TEMPLATE_THRESHOLD:
            return None
        return best[1], best[0]

    def _load_settings_template(self) -> np.ndarray | None:
        if self._settings_template is not False:
            return self._settings_template
        path = self._control_template_directory / "warehouse_sidebar_settings.png"
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            image = None
        self._settings_template = image if image is not None and image.size else None
        return self._settings_template

    @staticmethod
    def _button_icon_score(frame: np.ndarray, bounds: Rect) -> float:
        margin_x = max(2, round(bounds.width * 0.15))
        margin_y = max(2, round(bounds.height * 0.15))
        crop = frame[
            bounds.top + margin_y : bounds.bottom - margin_y,
            bounds.left + margin_x : bounds.right - margin_x,
        ]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.mean(cv2.Canny(gray, 35, 90) > 0))

    @staticmethod
    def _clipped_rect(
        left: int,
        top: int,
        width: int,
        height: int,
        frame_width: int,
        frame_height: int,
    ) -> Rect:
        left = max(0, min(left, frame_width - 1))
        top = max(0, min(top, frame_height - 1))
        right = max(left + 1, min(frame_width, left + width))
        bottom = max(top + 1, min(frame_height, top + height))
        return Rect(left, top, right - left, bottom - top)
