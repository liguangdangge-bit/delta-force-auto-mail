from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from bulletbot.domain.models import PageState, Rect


@dataclass(frozen=True)
class PageTemplateEvidence:
    state: PageState
    name: str
    score: float
    bounds: Rect
    controls: tuple[tuple[str, Rect], ...] = ()


@dataclass(frozen=True)
class _PageTemplate:
    state: PageState
    name: str
    image: np.ndarray
    source_width: int
    source_height: int
    search_bounds: tuple[float, float, float, float]
    threshold: float
    active_highlight_threshold: float | None
    controls: tuple[tuple[str, tuple[float, float, float, float]], ...]


class PageTemplateMatcher:
    """Find small, stable UI anchors at the scale of the current client frame."""

    _SCALE_FACTORS = (0.82, 0.90, 0.96, 1.0, 1.04, 1.10, 1.18)

    def __init__(self, template_directory: Path | None = None) -> None:
        self._template_directory = template_directory or self.default_template_directory()
        self._templates: tuple[_PageTemplate, ...] | None = None

    @staticmethod
    def default_template_directory() -> Path:
        if getattr(sys, "frozen", False):
            root = Path(getattr(sys, "_MEIPASS"))
        else:
            root = Path(__file__).resolve().parents[2]
        return root / "assets" / "page_templates"

    def match(
        self,
        image: Image.Image,
        *,
        state: PageState | None = None,
    ) -> tuple[PageTemplateEvidence, ...]:
        templates = self._load_templates()
        if not templates:
            return ()
        frame = np.asarray(image.convert("RGB"))[:, :, ::-1]
        height, width = frame.shape[:2]
        matches: list[PageTemplateEvidence] = []
        for template in templates:
            if state is not None and template.state is not state:
                continue
            left = max(0, round(width * template.search_bounds[0]))
            top = max(0, round(height * template.search_bounds[1]))
            right = min(width, round(width * template.search_bounds[2]))
            bottom = min(height, round(height * template.search_bounds[3]))
            search = frame[top:bottom, left:right]
            if search.size == 0:
                continue
            base_scale = min(
                width / max(1, template.source_width),
                height / max(1, template.source_height),
            )
            best: tuple[float, Rect] | None = None
            for factor in self._SCALE_FACTORS:
                scale = base_scale * factor
                target_width = max(8, round(template.image.shape[1] * scale))
                target_height = max(8, round(template.image.shape[0] * scale))
                if target_width > search.shape[1] or target_height > search.shape[0]:
                    continue
                resized = cv2.resize(
                    template.image,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
                )
                result = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
                _minimum, score, _minimum_location, location = cv2.minMaxLoc(result)
                bounds = Rect(
                    left + int(location[0]),
                    top + int(location[1]),
                    target_width,
                    target_height,
                )
                if best is None or score > best[0]:
                    best = float(score), bounds
            active_score = (
                self._active_highlight_score(frame, best[1])
                if best is not None and template.active_highlight_threshold is not None
                else None
            )
            if (
                best is not None
                and best[0] >= template.threshold
                and (
                    template.active_highlight_threshold is None
                    or (
                        active_score is not None
                        and active_score >= template.active_highlight_threshold
                    )
                )
            ):
                matches.append(
                    PageTemplateEvidence(
                        template.state,
                        template.name,
                        best[0],
                        best[1],
                        tuple(
                            (
                                name,
                                self._relative_control_bounds(
                                    best[1],
                                    relative_bounds,
                                    width,
                                    height,
                                ),
                            )
                            for name, relative_bounds in template.controls
                        ),
                    )
                )
        return tuple(sorted(matches, key=lambda value: value.score, reverse=True))

    def _load_templates(self) -> tuple[_PageTemplate, ...]:
        if self._templates is not None:
            return self._templates
        manifest_path = self._template_directory / "manifest.json"
        if not manifest_path.is_file():
            self._templates = ()
            return self._templates
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._templates = ()
            return self._templates
        loaded: list[_PageTemplate] = []
        for item in payload.get("templates", []):
            try:
                state = PageState(str(item["state"]))
                path = self._template_directory / str(item["file"])
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                source_size = item["source_size"]
                raw_search = item.get("search_bounds", (0.0, 0.0, 1.0, 1.0))
                search_bounds = tuple(float(value) for value in raw_search)
                raw_controls = item.get("controls", {})
                if image is None or image.size == 0 or len(search_bounds) != 4:
                    continue
                loaded.append(
                    _PageTemplate(
                        state=state,
                        name=str(item.get("name") or path.stem),
                        image=image,
                        source_width=int(source_size[0]),
                        source_height=int(source_size[1]),
                        search_bounds=search_bounds,
                        threshold=float(item.get("threshold", 0.78)),
                        active_highlight_threshold=(
                            float(item["active_highlight_threshold"])
                            if item.get("active_highlight_threshold") is not None
                            else None
                        ),
                        controls=tuple(
                            (
                                str(name),
                                tuple(float(value) for value in relative_bounds),
                            )
                            for name, relative_bounds in raw_controls.items()
                            if isinstance(relative_bounds, (list, tuple))
                            and len(relative_bounds) == 4
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._templates = tuple(loaded)
        return self._templates

    @staticmethod
    def _active_highlight_score(frame: np.ndarray, bounds: Rect) -> float:
        crop = frame[bounds.top : bounds.bottom, bounds.left : bounds.right].astype(
            np.int16
        )
        if crop.size == 0:
            return 0.0
        blue = crop[:, :, 0]
        green = crop[:, :, 1]
        red = crop[:, :, 2]
        turquoise = (
            (green >= 105)
            & (green >= red + 18)
            & (blue >= 75)
            & (blue >= red + 8)
        )
        coverage = np.mean(turquoise, axis=1)
        strongest = np.sort(coverage)[-min(3, len(coverage)) :]
        return float(np.mean(strongest))

    @staticmethod
    def _relative_control_bounds(
        anchor: Rect,
        relative: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Rect:
        left = max(0, anchor.left + round(anchor.width * relative[0]))
        top = max(0, anchor.top + round(anchor.height * relative[1]))
        right = min(
            frame_width,
            anchor.left + round(anchor.width * relative[2]),
        )
        bottom = min(
            frame_height,
            anchor.top + round(anchor.height * relative[3]),
        )
        return Rect(left, top, max(1, right - left), max(1, bottom - top))
