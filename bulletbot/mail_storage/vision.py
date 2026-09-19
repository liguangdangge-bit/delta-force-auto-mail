from __future__ import annotations

from bulletbot.ocr.local_models import general_model_params

import json
from concurrent.futures import Future
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .capture import CapturedFrame
from .models import OcrTextRegion, PageObservation, PageType, TemplateMatch
from .paths import resource_path
from bulletbot.ocr.runtime_backend import (
    OCR_RUNTIME_LOCK,
    directml_enabled,
    provider_label,
    report_ocr_runtime,
)
from bulletbot.ocr.numeric_ocr import NumericOcrEngine


FastPriceOcrEngine = NumericOcrEngine


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    page: str
    file: Path
    reference_size: tuple[int, int]
    search_rect: tuple[int, int, int, int]
    threshold: float
    click_normalized: tuple[float, float] | None
    click_offset: tuple[int, int] | None


@dataclass(frozen=True)
class _LauncherResourceCard:
    bounds: tuple[int, int, int, int]
    text_regions: tuple[OcrTextRegion, ...]


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


class TemplateCatalog:
    def __init__(self, manifest_path: Path | None = None) -> None:
        self.root = resource_path("assets", "mail_storage", "templates")
        self.manifest_path = manifest_path or self.root / "manifest.json"
        manifest_paths = [self.manifest_path]
        real_flow_manifest = self.root / "real_flow_manifest.json"
        if manifest_path is None and real_flow_manifest.exists():
            manifest_paths.append(real_flow_manifest)
        self.pages: dict[str, dict[str, Any]] = {}
        self.specs: dict[str, TemplateSpec] = {}
        self.images: dict[str, np.ndarray] = {}
        for current_manifest in manifest_paths:
            raw = json.loads(current_manifest.read_text(encoding="utf-8"))
            self.pages.update(raw["pages"])
            for value in raw["templates"]:
                click = value.get("click_normalized")
                click_offset = value.get("click_offset")
                spec = TemplateSpec(
                    name=value["name"],
                    page=value["page"],
                    file=self.root / value["file"],
                    reference_size=tuple(value["reference_size"]),
                    search_rect=tuple(value["search_rect"]),
                    threshold=float(value["threshold"]),
                    click_normalized=tuple(click) if click else None,
                    click_offset=tuple(click_offset) if click_offset else None,
                )
                self.specs[spec.name] = spec
                self.images[spec.name] = _read_image(spec.file)


class OcrEngine:
    GENERAL_MODEL_VERSION = "PP-OCRv6 Small"

    def __init__(self, model_root: Path | None = None) -> None:
        self.model_root = model_root or resource_path(
            "models", "mail_storage", "ocr", "rapidocr"
        )
        self._engine = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._using_directml = False

    def _get_engine(self):
        with self._lock:
            if self._engine is None:
                from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

                self._using_directml = directml_enabled()
                try:
                    with OCR_RUNTIME_LOCK:
                        self._engine = self._build_engine(
                            RapidOCR,
                            EngineType,
                            ModelType,
                            OCRVersion,
                            use_directml=self._using_directml,
                        )
                except Exception as exc:
                    if not self._using_directml:
                        raise
                    report_ocr_runtime(
                        f"配装 OCR 的 DirectML 初始化失败，已回退 CPU：{exc}",
                        "WARNING",
                    )
                    self._using_directml = False
                    with OCR_RUNTIME_LOCK:
                        self._engine = self._build_engine(
                            RapidOCR,
                            EngineType,
                            ModelType,
                            OCRVersion,
                            use_directml=False,
                        )
                report_ocr_runtime(
                    f"配装 OCR 实际后端：{provider_label(self._using_directml)}"
                )
            return self._engine

    def _build_engine(
        self,
        rapid_ocr,
        engine_type,
        model_type,
        ocr_version,
        *,
        use_directml: bool,
    ):
        if use_directml:
            from bulletbot.ocr.isolated_runtime import IsolatedRapidOcr

            return IsolatedRapidOcr("trade_v6")

        return rapid_ocr(
            params={
                **general_model_params(),
                "Global.log_level": "error",
                "Global.use_cls": False,
                "EngineConfig.onnxruntime.use_dml": use_directml,
                "Det.engine_type": engine_type.ONNXRUNTIME,
                "Det.model_type": model_type.SMALL,
                "Det.ocr_version": ocr_version.PPOCRV6,
                "Cls.engine_type": engine_type.ONNXRUNTIME,
                "Cls.model_type": model_type.MOBILE,
                "Cls.ocr_version": ocr_version.PPOCRV5,
                "Rec.engine_type": engine_type.ONNXRUNTIME,
                "Rec.model_type": model_type.SMALL,
                "Rec.ocr_version": ocr_version.PPOCRV6,
            }
        )

    @staticmethod
    def _run_engine(engine, image: np.ndarray, **kwargs):
        result = engine(image, **kwargs)
        # RapidOCR 3.x returns OcrResult; retain compatibility with older
        # adapters and test doubles that return (result, elapsed).
        return result[0] if isinstance(result, tuple) else result

    def detect(
        self,
        image: np.ndarray,
        *,
        detector_min_side_len: int | None = None,
    ) -> list[OcrTextRegion]:
        if detector_min_side_len is not None and detector_min_side_len <= 0:
            raise ValueError("detector_min_side_len must be positive")
        engine = self._get_engine()
        with self._inference_lock, OCR_RUNTIME_LOCK:
            original_min_side_len = engine.text_det.limit_side_len
            if detector_min_side_len is not None:
                engine.text_det.limit_side_len = detector_min_side_len
            try:
                try:
                    result = self._run_engine(engine, image)
                except Exception as exc:
                    if not self._using_directml:
                        raise
                    report_ocr_runtime(
                        f"配装 OCR 的 DirectML 推理失败，已在本次运行回退 CPU：{exc}",
                        "WARNING",
                    )
                    from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

                    self._using_directml = False
                    fallback_engine = self._build_engine(
                        RapidOCR,
                        EngineType,
                        ModelType,
                        OCRVersion,
                        use_directml=False,
                    )
                    self._engine = fallback_engine
                    fallback_min_side_len = fallback_engine.text_det.limit_side_len
                    if detector_min_side_len is not None:
                        fallback_engine.text_det.limit_side_len = detector_min_side_len
                    try:
                        result = self._run_engine(fallback_engine, image)
                    finally:
                        fallback_engine.text_det.limit_side_len = (
                            fallback_min_side_len
                        )
            finally:
                engine.text_det.limit_side_len = original_min_side_len
        if not result:
            return []
        if isinstance(result, tuple):
            result = result[0]
        texts_value = getattr(result, "txts", None)
        scores_value = getattr(result, "scores", None)
        boxes_value = getattr(result, "boxes", None)
        texts = list(texts_value) if texts_value is not None else []
        scores = list(scores_value) if scores_value is not None else []
        boxes = list(boxes_value) if boxes_value is not None else []
        if texts:
            return [
                OcrTextRegion(
                    text=str(text).strip(),
                    confidence=float(score),
                    bounds=(
                        int(np.floor(np.asarray(box)[:, 0].min())),
                        int(np.floor(np.asarray(box)[:, 1].min())),
                        int(np.ceil(np.asarray(box)[:, 0].max())),
                        int(np.ceil(np.asarray(box)[:, 1].max())),
                    ),
                )
                for box, text, score in zip(boxes, texts, scores)
                if str(text).strip()
            ]
        if not isinstance(result, (list, tuple)):
            return []
        regions: list[OcrTextRegion] = []
        for item in result:
            text = str(item[1]).strip()
            if not text:
                continue
            points = np.asarray(item[0], dtype=np.float32)
            left = int(np.floor(points[:, 0].min()))
            top = int(np.floor(points[:, 1].min()))
            right = int(np.ceil(points[:, 0].max()))
            bottom = int(np.ceil(points[:, 1].max()))
            regions.append(
                OcrTextRegion(
                    text=text,
                    confidence=float(item[2]),
                    bounds=(left, top, right, bottom),
                )
            )
        return regions

    def recognize(self, image: np.ndarray) -> list[str]:
        return [region.text for region in self.detect(image)]

    def recognize_line(self, image: np.ndarray) -> tuple[str, float] | None:
        """Recognize one tightly cropped line without text detection."""

        engine = self._get_engine()
        with self._inference_lock, OCR_RUNTIME_LOCK:
            try:
                result = self._run_engine(
                    engine,
                    image,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
            except Exception as exc:
                if not self._using_directml:
                    raise
                report_ocr_runtime(
                    f"配装 OCR 的 DirectML 推理失败，已在本次运行回退 CPU：{exc}",
                    "WARNING",
                )
                from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

                self._using_directml = False
                engine = self._build_engine(
                    RapidOCR,
                    EngineType,
                    ModelType,
                    OCRVersion,
                    use_directml=False,
                )
                self._engine = engine
                result = self._run_engine(
                    engine,
                    image,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
        if not result:
            return None
        texts_value = getattr(result, "txts", None)
        scores_value = getattr(result, "scores", None)
        if texts_value is not None:
            scores = list(scores_value) if scores_value is not None else []
            candidates = [
                (str(text).strip(), float(score))
                for text, score in zip(list(texts_value), scores)
                if str(text).strip()
            ]
            return max(candidates, key=lambda item: item[1]) if candidates else None
        if not isinstance(result, (list, tuple)):
            return None
        candidates: list[tuple[str, float]] = []
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            if isinstance(item[0], str):
                text, score = item[0], item[1]
            elif len(item) >= 3:
                text, score = item[1], item[2]
            else:
                continue
            if str(text).strip():
                candidates.append((str(text).strip(), float(score)))
        return max(candidates, key=lambda item: item[1]) if candidates else None


class VisionEngine:
    _LAUNCHER_LONGBOW_TEXT = "烽火地带长弓溪谷"
    _LAUNCHER_LONGBOW_MIN_SIMILARITY = 0.72
    _TRASH_ICON_MIN_SCORE = 0.76
    _TRASH_ICON_SCALES = tuple(0.75 + step * 0.05 for step in range(18))
    _MODE_MENU_LABEL_RECTS = {
        "firestorm": (0.040, 0.289, 0.141, 0.328),
        "warfare": (0.040, 0.460, 0.141, 0.511),
        "black_hawk": (0.040, 0.634, 0.141, 0.681),
    }
    _SEASON_2026_MODE_MENU_LABEL_RECTS = {
        "firestorm": (0.040, 0.254, 0.141, 0.294),
        "warfare": (0.040, 0.393, 0.141, 0.433),
        "black_tide": (0.040, 0.532, 0.141, 0.572),
        "black_hawk": (0.040, 0.671, 0.141, 0.711),
    }
    _MODE_MENU_TEMPLATE_LAYOUTS = {
        "season_2026_mode_menu": "season_2026",
        "season_2026_mode_menu_warfare": "season_2026",
    }
    _MODE_MENU_BRIGHTNESS_THRESHOLD = 150
    _MODE_MENU_MIN_SELECTED_RATIO = 0.04
    _MODE_MENU_MIN_SELECTION_MARGIN = 0.025

    def __init__(self, catalog: TemplateCatalog | None = None) -> None:
        self.catalog = catalog or TemplateCatalog()
        self.ocr = OcrEngine()
        self.fast_price_ocr = FastPriceOcrEngine(self.ocr.model_root)

    def observe(
        self,
        frame: CapturedFrame,
        include_ocr: bool = False,
        template_names: Collection[str] | None = None,
        ocr_crop: tuple[float, float, float, float] | None = None,
        target: str | None = None,
        accept_template: Callable[[PageObservation], bool] | None = None,
    ) -> PageObservation:
        if not frame.healthy:
            return PageObservation(
                page_type=PageType.INVALID_FRAME,
                confidence=0.0,
                frame_mean=frame.mean,
                frame_std=frame.std,
                timestamp=frame.timestamp,
            )

        resized_cache: dict[tuple[int, int], np.ndarray] = {}
        matches: dict[str, TemplateMatch] = {}
        for name, spec in self.catalog.specs.items():
            page_spec = self.catalog.pages.get(spec.page, {})
            page_name = page_spec.get("page_type", spec.page)
            launcher_template = page_name.startswith("launcher_")
            if target is not None and launcher_template != (target != "game"):
                continue
            if template_names is not None and name not in template_names:
                continue
            normalized = resized_cache.get(spec.reference_size)
            if normalized is None:
                normalized = cv2.resize(frame.image, spec.reference_size, interpolation=cv2.INTER_AREA)
                resized_cache[spec.reference_size] = normalized
            search_left, search_top, search_right, search_bottom = spec.search_rect
            search = normalized[search_top:search_bottom, search_left:search_right]
            template = self.catalog.images[name]
            if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(
                cv2.cvtColor(search, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                cv2.TM_CCOEFF_NORMED,
            )
            _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(result)
            if maximum >= spec.threshold:
                x = search_left + max_location[0]
                y = search_top + max_location[1]
                click_normalized = spec.click_normalized
                if spec.click_offset is not None:
                    click_normalized = (
                        (x + spec.click_offset[0]) / spec.reference_size[0],
                        (y + spec.click_offset[1]) / spec.reference_size[1],
                    )
                matches[name] = TemplateMatch(
                    name=name,
                    score=float(maximum),
                    location=(x, y),
                    size=(template.shape[1], template.shape[0]),
                    click_normalized=click_normalized,
                )

        mode_menu_selection = None
        mode_menu_layouts = [
            layout_name
            for template_name, layout_name in self._MODE_MENU_TEMPLATE_LAYOUTS.items()
            if template_name in matches
        ]
        if mode_menu_layouts:
            mode_menu_selection = self.detect_mode_menu_selection(
                frame.image,
                layout_names=mode_menu_layouts,
            )
            if mode_menu_selection is not None:
                _selected_mode, selection_match = mode_menu_selection
                matches[selection_match.name] = selection_match

        page_type, confidence = self._classify(matches)
        if mode_menu_selection is not None:
            _selected_mode, selection_match = mode_menu_selection
            page_type = PageType.MODE_MENU
            confidence = selection_match.score
        regions: list[OcrTextRegion] = []
        weak_activity = (
            target == "game"
            and page_type == PageType.ACTIVITY_REMINDER
            and "real_activity_header" not in matches
        )
        if accept_template is not None:
            # This gate only decides whether OCR can be skipped. It must not
            # rewrite the existing classifier's input or OCR fallback result.
            candidates = self._template_candidates(matches)
            remaining = {candidate[2] for candidate in candidates}
            # Only explicit page relationships can remove a background
            # candidate. Anchor-set inclusion alone does not prove equivalence.
            refinements = {
                PageType.MAIL_SCHEME_SELECTED: {PageType.LOADOUT_SCHEMES, PageType.LOADOUT},
                PageType.LOADOUT_SCHEMES: {PageType.LOADOUT},
                PageType.MAP_ZERO_DAM_START: {PageType.MAP_OTHER, PageType.MAP_OVERVIEW},
            }
            for specific, backgrounds in refinements.items():
                if specific in remaining:
                    remaining -= backgrounds
            weak_loadout = (
                page_type == PageType.LOADOUT
                and "real_loadout_scheme_shortcut" not in matches
            )
            if len(remaining) == 1 and not weak_activity and not weak_loadout:
                selected = next(iter(remaining))
                selected_confidence = max(c[1] for c in candidates if c[2] == selected)
                template_result = PageObservation(
                    page_type=selected, confidence=selected_confidence, matches=matches,
                    frame_mean=frame.mean, frame_std=frame.std, timestamp=frame.timestamp,
                )
                if accept_template(template_result):
                    return template_result
            if not include_ocr and (len(remaining) != 1 or weak_activity or weak_loadout):
                # Wait for the scheduled OCR scan, retaining template evidence
                # for diagnostics. Do not discard it before OCR adjudication.
                page_type, confidence = PageType.UNKNOWN, 0.0
        # Preserve the existing specialist fallback for legacy/content readers.
        # Page waits explicitly opt in above and control when OCR may run.
        run_ocr = include_ocr or (accept_template is None and weak_activity)
        if run_ocr:
            ocr_image, offset = self._ocr_image(frame.image, ocr_crop)
            agent_only = (ocr_crop is not None and template_names is not None
                          and "real_agent_select_header" in template_names
                          and set(template_names) <= {
                              "real_agent_select_header", "real_agent_select_view_ability",
                              "real_entry_warning_header", "real_entry_warning_continue",
                              "real_game_loading_art",
                          })
            regions = (self._detect_agent_ocr_nonblocking(ocr_image)
                       if agent_only else self.ocr.detect(ocr_image))
            if offset != (0, 0):
                offset_x, offset_y = offset
                regions = [
                    OcrTextRegion(
                        region.text,
                        region.confidence,
                        (
                            region.bounds[0] + offset_x,
                            region.bounds[1] + offset_y,
                            region.bounds[2] + offset_x,
                            region.bounds[3] + offset_y,
                        ),
                    )
                    for region in regions
                ]
        texts = [region.text for region in regions]
        if texts:
            text_page, text_confidence = self._classify_text(
                texts, allow_transition=self._transition_text_matches(regions, frame.image.shape),
            )
            # Only names inside the left mode cards can identify this menu.
            if text_page == PageType.MODE_MENU:
                text_page = PageType.UNKNOWN
            if self._mode_menu_text_matches(regions, frame.image.shape):
                text_page, text_confidence = PageType.MODE_MENU, 0.75
            text_allowed = target is None or (
                text_page.value.startswith("launcher_") == (target != "game")
            )
            if text_allowed and text_page != PageType.UNKNOWN and (
                page_type == PageType.UNKNOWN or text_page in {
                    PageType.GAME_LOADING,
                    PageType.MATCHING, PageType.MISSING_RESOURCE,
                    PageType.RECONNECT_PROMPT, PageType.ABANDON_PROMPT,
                    PageType.ENTRY_WARNING, PageType.INITIAL_MODE_SELECTION,
                    PageType.LAUNCHER_RESOURCES, PageType.LAUNCHER_DELETE_CONFIRM,
                }
            ):
                page_type, confidence = text_page, text_confidence
            self._add_ocr_matches(matches, regions, frame.image)
        # Read control text in its expected region to distinguish a promotional
        # CTA from the visually similar loadout confirmation button.
        if target == "game" and regions:
            continuation = any(
                r.confidence >= 0.70 and "".join(r.text.split()) == "继续"
                and 0.35 <= (r.bounds[0] + r.bounds[2]) / 2 / frame.image.shape[1] <= 0.65
                and (r.bounds[1] + r.bounds[3]) / 2 / frame.image.shape[0] >= 0.90
                for r in regions
            )
            compact = "".join("".join(r.text.split()) for r in regions if r.confidence >= 0.70)
            button_texts = {
                "".join(r.text.split()) for r in regions
                if r.confidence >= 0.70
                and 0.75 <= (r.bounds[0] + r.bounds[2]) / 2 / frame.image.shape[1] <= 0.99
                and 0.78 <= (r.bounds[1] + r.bounds[3]) / 2 / frame.image.shape[0] <= 0.96
            }
            promotion = bool(button_texts & {"前往观看", "前往获取", "前往参与"})
            loadout_confirmation = bool(button_texts & {"确认配装", "确定配装", "确认装配"})
            if loadout_confirmation and page_type == PageType.UNKNOWN:
                page_type, confidence = PageType.LOADOUT, 0.85
            if (continuation or promotion) and page_type in {
                PageType.UNKNOWN, PageType.LOADOUT, PageType.ACTIVITY_REMINDER,
                PageType.GAME_TRANSITION, PageType.SETTLEMENT,
            }:
                if continuation and "赛季通行证进度" in compact:
                    page_type, confidence = PageType.SETTLEMENT, 0.85
                elif (not loadout_confirmation and page_type != PageType.SETTLEMENT):
                    page_type, confidence = PageType.ACTIVITY_REMINDER, 0.85
        if target in {"launcher", "launcher_settings"}:
            running = self.launcher_running_text_match(regions, frame.image.shape)
            if running is not None:
                matches[running.name] = running
            # Text is spatially constrained; words on background advertisements
            # must not authorize a launcher action through a settings dialog.
            home_match = self._launcher_home_text_match(regions, frame.image.shape)
            if home_match is not None and page_type not in {
                PageType.LAUNCHER_RESOURCES, PageType.LAUNCHER_DELETE_CONFIRM,
            } and not any("设置" in r.text for r in regions):
                matches[home_match.name] = home_match
                page_type, confidence = PageType.LAUNCHER_HOME, home_match.score
        # The map card may lose template contrast against a bright background.
        # Only this card's search area can supply the alternative text evidence.
        map_spec = self.catalog.specs.get("real_map_selection_mode")
        if ((include_ocr or accept_template is None) and target == "game" and map_spec is not None
                and page_type in {PageType.UNKNOWN, PageType.MAP_OVERVIEW}
                and (template_names is None or "real_map_selection_mode" in template_names)):
            rw, rh = map_spec.reference_size
            left, top, right, bottom = map_spec.search_rect
            height, width = frame.image.shape[:2]
            bounds = (round(left * width / rw), round(top * height / rh),
                      round(right * width / rw), round(bottom * height / rh))
            x1, y1, x2, y2 = bounds
            if include_ocr and ocr_crop is None:
                card_regions = [r for r in regions
                                if x1 <= (r.bounds[0] + r.bounds[2]) / 2 <= x2
                                and y1 <= (r.bounds[1] + r.bounds[3]) / 2 <= y2]
            else:
                card_image = frame.image[y1:y2, x1:x2]
                card_regions = self.ocr.detect(card_image) if card_image.size else []
            if any(r.confidence >= 0.70 and "危险行动" in "".join(r.text.split())
                   for r in card_regions):
                page_type, confidence = PageType.MAP_SELECTION, 0.75
                matches["ocr_map_selection_mode"] = TemplateMatch(
                    name="ocr_map_selection_mode", score=confidence,
                    location=(x1, y1), size=(x2 - x1, y2 - y1),
                )
        return PageObservation(
            page_type=page_type,
            confidence=confidence,
            matches=matches,
            ocr_texts=texts,
            ocr_regions=regions,
            frame_mean=frame.mean,
            frame_std=frame.std,
            timestamp=frame.timestamp or time.time(),
        )

    def _detect_agent_ocr_nonblocking(self, image: np.ndarray) -> list[OcrTextRegion]:
        """One bounded background crop; stale/different screenshots never authorize actions."""
        pending = getattr(self, "_agent_ocr_pending", None)
        result = []
        if pending is not None:
            future, source, started = pending
            if not future.done():
                return []
            self._agent_ocr_pending = None
            try:
                regions = future.result()
                if (time.monotonic() - started <= 2.0 and source.shape == image.shape
                        and np.mean(cv2.absdiff(source, image)) <= 3.0):
                    result = regions
            except Exception:
                # Template checks continue even when the optional OCR worker fails.
                pass
        if not result:
            future = Future()
            source = image.copy()
            self._agent_ocr_pending = (future, source, time.monotonic())
            def run():
                try:
                    future.set_result(self.ocr.detect(source))
                except Exception as exc:
                    future.set_exception(exc)
            threading.Thread(target=run, name="mail-agent-ocr", daemon=True).start()
        return result

    @staticmethod
    def _mode_menu_text_matches(
        regions: list[OcrTextRegion], shape: tuple[int, ...],
    ) -> bool:
        height, width = shape[:2]
        labels = set()
        for region in regions:
            if region.confidence < 0.70:
                continue
            left, top, right, bottom = region.bounds
            x, y = (left + right) / (2 * width), (top + bottom) / (2 * height)
            if not (0.015 <= x <= 0.28 and 0.13 <= y <= 0.78):
                continue
            text = "".join(region.text.split())
            labels.update(name for name in (
                "烽火地带", "全面战场", "黑潮爆破", "黑鹰坠落",
            ) if name in text)
        return len(labels) >= 3

    @staticmethod
    def launcher_running_text_match(regions, shape) -> TemplateMatch | None:
        height, width = shape[:2]
        for region in regions:
            text = "".join(region.text.split())
            left, top, right, bottom = region.bounds
            x, y = (left + right) / (2 * width), (top + bottom) / (2 * height)
            if (region.confidence >= .70 and text in {"游戏运行中", "游戏正在运行中", "游戏正在运行"}
                    and .65 <= x <= .98 and .72 <= y <= .98):
                return TemplateMatch("ocr_launcher_game_running", region.confidence,
                                     (left, top), (right - left, bottom - top))
        return None

    @staticmethod
    def _launcher_home_text_match(
        regions: list[OcrTextRegion], shape: tuple[int, ...],
    ) -> TemplateMatch | None:
        height, width = shape[:2]
        home = []
        start = []
        for region in regions:
            if region.confidence < 0.70:
                continue
            text = "".join(region.text.split())
            left, top, right, bottom = region.bounds
            x, y = (left + right) / (2 * width), (top + bottom) / (2 * height)
            if text == "首页" and 0.04 <= x <= 0.4 and y <= 0.20:
                home.append(region)
            if text == "开始游戏" and 0.65 <= x <= 0.98 and 0.72 <= y <= 0.98:
                start.append(region)
        if not home or not start:
            return None
        button = max(start, key=lambda r: r.confidence)
        left, top, right, bottom = button.bounds
        return TemplateMatch(
            name="ocr_launcher_start_button",
            score=min(max(r.confidence for r in home), button.confidence),
            location=(left, top), size=(right - left, bottom - top),
            click_normalized=((left + right) / (2 * width), (top + bottom) / (2 * height)),
        )

    @staticmethod
    def _ocr_image(
        image: np.ndarray,
        crop: tuple[float, float, float, float] | None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if crop is None:
            return image, (0, 0)
        height, width = image.shape[:2]
        left, top, right, bottom = crop
        x1 = max(0, min(width - 1, round(left * width)))
        y1 = max(0, min(height - 1, round(top * height)))
        x2 = max(x1 + 1, min(width, round(right * width)))
        y2 = max(y1 + 1, min(height, round(bottom * height)))
        return image[y1:y2, x1:x2], (x1, y1)

    @classmethod
    def detect_mode_menu_selection(
        cls,
        image: np.ndarray,
        *,
        layout_names: Collection[str] | None = None,
    ) -> tuple[str, TemplateMatch] | None:
        """Identify the active mode from the one bright label in the left menu."""

        if image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
            return None
        layouts = {
            "legacy": cls._MODE_MENU_LABEL_RECTS,
            "season_2026": cls._SEASON_2026_MODE_MENU_LABEL_RECTS,
        }
        selected_layouts = (
            (layouts["legacy"],)
            if layout_names is None
            else (layouts[name] for name in layout_names if name in layouts)
        )
        candidates = [
            candidate
            for rects in selected_layouts
            if (candidate := cls._detect_mode_menu_selection_in_layout(image, rects))
            is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1].score)

    @classmethod
    def _detect_mode_menu_selection_in_layout(
        cls,
        image: np.ndarray,
        rects: dict[str, tuple[float, float, float, float]],
    ) -> tuple[str, TemplateMatch] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        ratios: dict[str, float] = {}
        pixel_rects: dict[str, tuple[int, int, int, int]] = {}
        for mode, (left, top, right, bottom) in rects.items():
            x1 = max(0, min(width - 1, round(left * width)))
            y1 = max(0, min(height - 1, round(top * height)))
            x2 = max(x1 + 1, min(width, round(right * width)))
            y2 = max(y1 + 1, min(height, round(bottom * height)))
            region = gray[y1:y2, x1:x2]
            ratios[mode] = float(
                np.count_nonzero(region >= cls._MODE_MENU_BRIGHTNESS_THRESHOLD)
                / region.size
            )
            pixel_rects[mode] = (x1, y1, x2, y2)

        ranked = sorted(ratios.items(), key=lambda item: item[1], reverse=True)
        (selected_mode, selected_ratio), (_runner_up, runner_up_ratio) = ranked[:2]
        margin = selected_ratio - runner_up_ratio
        if (
            selected_ratio < cls._MODE_MENU_MIN_SELECTED_RATIO
            or margin < cls._MODE_MENU_MIN_SELECTION_MARGIN
        ):
            return None

        x1, y1, x2, y2 = pixel_rects[selected_mode]
        confidence = min(0.99, 0.70 + selected_ratio + margin)
        name = f"mode_menu_selected_{selected_mode}"
        return selected_mode, TemplateMatch(
            name=name,
            score=confidence,
            location=(x1, y1),
            size=(x2 - x1, y2 - y1),
        )

    @staticmethod
    def selected_mode(observation: PageObservation) -> str | None:
        prefix = "mode_menu_selected_"
        return next(
            (
                name.removeprefix(prefix)
                for name in observation.matches
                if name.startswith(prefix)
            ),
            None,
        )

    def _add_ocr_matches(
        self,
        matches: dict[str, TemplateMatch],
        regions: list[OcrTextRegion],
        image: np.ndarray,
    ) -> None:
        frame_height, frame_width = image.shape[:2]
        delete_message = next(
            (region for region in regions if "确定删除资源包" in region.text.replace(" ", "")),
            None,
        )
        if delete_message is not None:
            candidates = [
                region
                for region in regions
                if region.text.replace(" ", "") in {"确定", "确认"}
                and region.bounds[1] > delete_message.bounds[3]
            ]
            if candidates:
                confirm = max(candidates, key=lambda region: (region.bounds[0], region.confidence))
                left, top, right, bottom = confirm.bounds
                matches["ocr_launcher_delete_confirm"] = TemplateMatch(
                    name="ocr_launcher_delete_confirm",
                    score=confirm.confidence,
                    location=(left, top),
                    size=(right - left, bottom - top),
                    click_normalized=(
                        ((left + right) / 2) / frame_width,
                        ((top + bottom) / 2) / frame_height,
                    ),
                )

        cards = self._launcher_downloaded_resource_cards(image.shape, regions)
        resource_candidates: list[
            tuple[float, OcrTextRegion, tuple[int, int, int, int] | None]
        ] = []
        if cards:
            for card in cards:
                for region in self._launcher_resource_text_candidates(
                    list(card.text_regions)
                ):
                    resource_candidates.append(
                        (
                            self._launcher_longbow_similarity(region.text),
                            region,
                            card.bounds,
                        )
                    )
        else:
            resource_candidates.extend(
                (
                    self._launcher_longbow_similarity(region.text),
                    region,
                    None,
                )
                for region in self._launcher_resource_text_candidates(regions)
            )
        resource_candidates.sort(
            key=lambda candidate: (candidate[0], candidate[1].confidence),
            reverse=True,
        )
        for identity_score, region, card_bounds in resource_candidates:
            if identity_score < self._LAUNCHER_LONGBOW_MIN_SIMILARITY:
                continue
            trash_match = self._find_launcher_trash_icon(
                image,
                region,
                card_bounds=card_bounds,
            )
            if trash_match is None:
                continue
            matches["ocr_launcher_longbow_resource"] = TemplateMatch(
                name="ocr_launcher_longbow_resource",
                score=min(identity_score, region.confidence, trash_match.score),
                location=trash_match.location,
                size=trash_match.size,
                click_normalized=trash_match.click_normalized,
            )
            return

    @classmethod
    def _launcher_downloaded_resource_cards(
        cls,
        image_shape: tuple[int, ...],
        regions: list[OcrTextRegion],
    ) -> list[_LauncherResourceCard]:
        """Divide the launcher's downloaded-resource section into card regions."""
        frame_height, frame_width = image_shape[:2]
        downloaded_headers = [
            region
            for region in regions
            if cls._normalize_ocr_text(region.text).startswith("已下载")
        ]
        if not downloaded_headers:
            return []
        header = min(downloaded_headers, key=lambda region: region.bounds[1])
        header_left, _header_top, _header_right, header_bottom = header.bounds
        lower_anchors = [
            region
            for region in regions
            if "安装至" in cls._normalize_ocr_text(region.text)
            and region.bounds[1] > header_bottom
        ]
        list_top = header_bottom
        list_bottom = (
            min(region.bounds[1] for region in lower_anchors)
            if lower_anchors
            else min(frame_height, list_top + round(frame_height * 0.32))
        )
        list_left = max(round(frame_width * 0.18), header_left)
        list_right = min(frame_width, round(frame_width * 0.95))
        if list_bottom <= list_top or list_right <= list_left:
            return []

        section_regions = [
            region
            for region in regions
            if list_left <= (region.bounds[0] + region.bounds[2]) / 2 <= list_right
            and list_top <= (region.bounds[1] + region.bounds[3]) / 2 <= list_bottom
        ]
        title_regions = [
            region
            for region in cls._launcher_resource_text_candidates(section_regions)
            if cls._looks_like_launcher_resource_title(region.text)
        ]
        if not title_regions:
            return []

        row_tolerance = max(8, round(frame_height * 0.025))
        row_clusters: list[list[float]] = []
        for center in sorted(
            (region.bounds[1] + region.bounds[3]) / 2 for region in title_regions
        ):
            if not row_clusters or center - sum(row_clusters[-1]) / len(row_clusters[-1]) > row_tolerance:
                row_clusters.append([center])
            else:
                row_clusters[-1].append(center)
        row_centers = [round(sum(cluster) / len(cluster)) for cluster in row_clusters]
        row_boundaries = [list_top]
        row_boundaries.extend(
            round((left + right) / 2)
            for left, right in zip(row_centers, row_centers[1:])
        )
        row_boundaries.append(list_bottom)

        column_boundary = round((list_left + list_right) / 2)
        column_bounds = (
            (list_left, column_boundary),
            (column_boundary, list_right),
        )
        cards: list[_LauncherResourceCard] = []
        for row_index, row_center in enumerate(row_centers):
            row_top = max(
                row_boundaries[row_index],
                row_center - max(20, round(frame_height * 0.045)),
            )
            row_bottom = min(
                row_boundaries[row_index + 1],
                row_center + max(28, round(frame_height * 0.06)),
            )
            for column_left, column_right in column_bounds:
                card_regions = tuple(
                    region
                    for region in section_regions
                    if column_left
                    <= (region.bounds[0] + region.bounds[2]) / 2
                    < column_right
                    and row_top
                    <= (region.bounds[1] + region.bounds[3]) / 2
                    <= row_bottom
                )
                if not card_regions:
                    continue
                if not any(
                    cls._looks_like_launcher_resource_title(region.text)
                    for region in cls._launcher_resource_text_candidates(
                        list(card_regions)
                    )
                ):
                    continue
                cards.append(
                    _LauncherResourceCard(
                        bounds=(column_left, row_top, column_right, row_bottom),
                        text_regions=card_regions,
                    )
                )
        return cards

    @staticmethod
    def _launcher_resource_text_candidates(
        regions: list[OcrTextRegion],
    ) -> list[OcrTextRegion]:
        """Return original OCR regions plus nearby fragments merged by row."""
        candidates = list(regions)
        ordered = sorted(regions, key=lambda region: (region.bounds[0], region.bounds[1]))
        for index, first in enumerate(ordered):
            left, top, right, bottom = first.bounds
            text = first.text
            confidence = first.confidence
            for following in ordered[index + 1 :]:
                next_left, next_top, next_right, next_bottom = following.bounds
                overlap = max(0, min(bottom, next_bottom) - max(top, next_top))
                minimum_height = max(1, min(bottom - top, next_bottom - next_top))
                if overlap / minimum_height < 0.5:
                    continue
                gap = next_left - right
                maximum_gap = max(24, round(minimum_height * 2.5))
                if gap < -round(minimum_height * 0.5) or gap > maximum_gap:
                    continue
                text += following.text
                right = max(right, next_right)
                top = min(top, next_top)
                bottom = max(bottom, next_bottom)
                confidence = min(confidence, following.confidence)
                candidates.append(
                    OcrTextRegion(
                        text=text,
                        confidence=confidence,
                        bounds=(left, top, right, bottom),
                    )
                )
        return candidates

    def _find_launcher_trash_icon(
        self,
        image: np.ndarray,
        resource_text: OcrTextRegion,
        *,
        card_bounds: tuple[int, int, int, int] | None = None,
    ) -> TemplateMatch | None:
        """Find a delete icon only inside the card that contains the OCR text."""
        template = self.catalog.images.get("launcher_trash_icon")
        if template is None:
            return None

        frame_height, frame_width = image.shape[:2]
        left, top, right, bottom = resource_text.bounds
        text_width = max(1, right - left)
        text_height = max(1, bottom - top)

        if card_bounds is not None:
            card_left, card_top, card_right, card_bottom = card_bounds
            card_width = max(1, card_right - card_left)
            search_left = max(
                0,
                right + max(4, round(text_width * 0.04)),
                card_left + round(card_width * 0.70),
            )
            search_right = min(frame_width, card_right)
            search_top = max(0, card_top)
            search_bottom = min(frame_height, card_bottom)
        else:
            # Fallback for launchers whose section headings were not readable.
            search_left = max(0, right + max(8, round(text_width * 0.08)))
            search_right = min(
                frame_width,
                right + max(round(text_width * 1.6), round(frame_width * 0.18)),
            )
            search_top = max(0, top - max(8, round(text_height * 0.8)))
            search_bottom = min(
                frame_height,
                bottom + max(20, round(text_height * 1.4)),
            )
        if search_right <= search_left or search_bottom <= search_top:
            return None

        search = image[search_top:search_bottom, search_left:search_right]
        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        best_score = -1.0
        best_location: tuple[int, int] | None = None
        best_template: np.ndarray | None = None

        # A standalone settings window may be DPI-scaled independently from
        # the launcher. Match a small, bounded scale range for its trash icon.
        for scale in self._TRASH_ICON_SCALES:
            scaled_template = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            if (
                search_gray.shape[0] < scaled_template.shape[0]
                or search_gray.shape[1] < scaled_template.shape[1]
            ):
                continue
            result = cv2.matchTemplate(
                search_gray,
                cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY),
                cv2.TM_CCOEFF_NORMED,
            )
            _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(result)
            if maximum > best_score:
                best_score = float(maximum)
                best_location = max_location
                best_template = scaled_template

        if (
            best_location is None
            or best_template is None
            or best_score < self._TRASH_ICON_MIN_SCORE
        ):
            return None

        icon_x = search_left + best_location[0]
        icon_y = search_top + best_location[1]
        icon_width = best_template.shape[1]
        icon_height = best_template.shape[0]
        return TemplateMatch(
            name="ocr_launcher_trash_icon",
            score=best_score,
            location=(icon_x, icon_y),
            size=(icon_width, icon_height),
            click_normalized=(
                (icon_x + icon_width / 2) / frame_width,
                (icon_y + icon_height / 2) / frame_height,
            ),
        )

    def _classify(self, matches: dict[str, TemplateMatch]) -> tuple[PageType, float]:
        candidates = self._template_candidates(matches)
        if not candidates:
            return PageType.UNKNOWN, 0.0
        _ranking, confidence, page = max(candidates, key=lambda item: item[0])
        return page, confidence

    def _template_candidates(
        self, matches: dict[str, TemplateMatch],
    ) -> list[tuple[float, float, PageType]]:
        candidates: list[tuple[float, float, PageType]] = []
        for page_name, page_spec in self.catalog.pages.items():
            anchors = page_spec["anchors"]
            present = [matches[name] for name in anchors if name in matches]
            if len(present) < int(page_spec["minimum_matches"]):
                continue
            average = sum(item.score for item in present) / len(anchors)
            # More independent anchors are stronger evidence than a single
            # background control that remains visible behind a modal.
            confidence = min(1.0, average + 0.04 * (len(present) - 1))
            try:
                page = PageType(page_spec.get("page_type", page_name))
            except ValueError:
                continue
            priority = float(page_spec.get("priority", 0))
            candidates.append((confidence + min(priority, 100.0) / 1000.0, confidence, page))
        if not candidates:
            return []

        # Top navigation identifies the game mode, not the page within that
        # mode. It remains visible on Firestorm map and loadout pages. Use it
        # only to reject the old shared warfare anchor; a page-specific anchor
        # must still identify the actual Firestorm page. Home action buttons
        # also override the tiny Space prompt, which can correlate with icons
        # in the persistent bottom bar.
        home_page_anchors = {
            "home_prepare_button",
            "season_2026_home_depart_buttons",
            "season_2026_home_depart_buttons",
        } & matches.keys()
        firestorm_context = "season_2026_firestorm_home_nav" in matches
        rejected_pages = set()
        longbow_action_anchors = {
            "map_start_button",
            "map_download_button",
            "map_download_icon",
            "map_download_progress_icon",
        } & matches.keys()
        if "map_longbow_title" in matches and longbow_action_anchors:
            # The generic map tabs remain visible on the Longbow detail panel.
            # A concrete action/download state is stronger evidence than them.
            rejected_pages.add(PageType.MAP_OVERVIEW)
        if home_page_anchors or firestorm_context:
            rejected_pages.add(PageType.WARFARE_HOME)
        if home_page_anchors:
            rejected_pages.add(PageType.ACTIVITY_REMINDER)
        if rejected_pages:
            candidates = [
                candidate
                for candidate in candidates
                if candidate[2] not in rejected_pages
            ]
            if not candidates:
                return []
        return candidates

    @staticmethod
    def _transition_text_matches(
        regions: list[OcrTextRegion], image_shape: tuple[int, ...],
    ) -> bool:
        # The real prompt is at the bottom center. The persistent top-left
        # "开始游戏" tab and bottom-left "Tab 进入特勤处" are not this prompt.
        height, width = image_shape[:2]
        prompt = "".join(
            "".join(region.text.split()).lower()
            for region in regions
            if region.confidence >= 0.70
            and 0.35 <= (region.bounds[0] + region.bounds[2]) / 2 / width <= 0.65
            and 0.85 <= (region.bounds[1] + region.bounds[3]) / 2 / height <= 0.98
        )
        return "tab" in prompt and "开始游戏" in prompt

    @staticmethod
    def _classify_text(
        texts: list[str], *, allow_transition: bool = True,
    ) -> tuple[PageType, float]:
        combined = " ".join(texts)
        compact = "".join(combined.split())
        # A delete confirmation modal is rendered on top of the resource list.
        # OCR therefore contains both the modal text and map names from the
        # obscured background.  Give the modal marker highest priority so
        # background content cannot steal the page classification.
        if "确定删除资源包" in compact:
            return PageType.LAUNCHER_DELETE_CONFIRM, 0.65

        # Settings navigation alone is shared with other tabs. Require resource
        # content as well, without depending on any particular map being visible.
        if ("设置" in compact and "资源管理" in compact
                and ("已下载" in compact or "资源包" in compact)):
            return PageType.LAUNCHER_RESOURCES, 0.75

        # A modal can overlay a fully recognizable mode/home page. Its
        # explicit message must win over navigation labels in the background.
        for modal, terms in (
            (PageType.MISSING_RESOURCE, ("存在未下载的资源", "无法重连入局")),
            (PageType.RECONNECT_PROMPT, ("取消重连", "重进入局")),
            (PageType.ABANDON_PROMPT, ("放弃对局", "确定放弃")),
            (PageType.ENTRY_WARNING, ("入局提醒明细", "仍要继续")),
        ):
            if all(term in compact for term in terms):
                return modal, 0.75

        if any(marker in compact for marker in (
            "正在初始化", "正在预加载资源", "正在加载资源",
        )):
            return PageType.GAME_LOADING, 0.80

        if allow_transition and "开始游戏" in combined and "tab" in combined.lower():
            return PageType.GAME_TRANSITION, 0.65
        if all(word in compact for word in ("常驻", "精选", "战役", "烽火地带", "全面战场")):
            return PageType.INITIAL_MODE_SELECTION, 0.75
        if (
            "烽火地带" in compact
            and "全面战场" in compact
            and ("黑潮爆破" in compact or "黑鹰坠落" in compact)
        ):
            return PageType.MODE_MENU, 0.75
        if "长弓溪谷" in compact and "MB" in combined:
            return PageType.MAP_LONGBOW_DOWNLOAD, 0.75
        if "长弓溪谷" in compact and "开始行动" in compact:
            return PageType.MAP_LONGBOW_START, 0.75
        mode_markers = ("危险行动", "黑夜行动", "红鼠窝竞技场")
        map_markers = (
            "零号大坝",
            "长弓溪谷",
            "巴克什",
            "特勤处",
            "潮汐监狱",
            "航天基地",
            "AZ3",
        )
        mode_count = sum(marker in compact for marker in mode_markers)
        map_count = sum(marker in compact for marker in map_markers)
        if (mode_count >= 2 and map_count >= 2) or map_count >= 4:
            return PageType.MAP_SELECTION, 0.75
        if "端游对局" in compact:
            if all(word in compact for word in ("配装", "载具", "军械库")):
                return PageType.WARFARE_HOME, 0.75
            if all(word in compact for word in ("仓库", "交易行", "特勤处")):
                return PageType.GAME_HOME_PREPARE, 0.75
        if "选择干员" in compact:
            return PageType.AGENT_SELECT, 0.65
        if (
            "资源管理" in combined
            and VisionEngine._contains_launcher_longbow_text(combined)
        ):
            return PageType.LAUNCHER_RESOURCES, 0.65
        rules = [
            (PageType.LOADOUT_PURCHASE_CONFIRM, ("确认实现方案",)),
            (PageType.LOADOUT_PRICE_CHANGE, ("价格变动提醒",)),
            (PageType.MATCHING, ("取消行动",)),
            (PageType.ENTRY_WARNING, ("入局提醒明细", "仍要继续")),
            (PageType.MISSING_RESOURCE, ("存在未下载的资源", "无法重连入局")),
            (PageType.RECONNECT_PROMPT, ("取消重连", "重进入局")),
            (PageType.ABANDON_PROMPT, ("放弃对局", "确定放弃")),
            (PageType.MAP_LONGBOW_DOWNLOAD, ("长弓溪谷", "MB")),
            (PageType.MAP_LONGBOW_START, ("长弓溪谷", "开始行动")),
            (PageType.GAME_HOME_READY, ("配装", "出发")),
            (PageType.LAUNCHER_RESOURCES, ("资源管理", "长弓溪谷")),
            (PageType.MAP_SELECTION, ("危险行动", "零号大坝", "长弓溪谷", "航天基地")),
        ]
        for page, required in rules:
            if all(word in combined for word in required):
                return page, 0.65
        return PageType.UNKNOWN, 0.0

    @staticmethod
    def _contains_launcher_longbow_text(text: str) -> bool:
        compact = VisionEngine._normalize_ocr_text(text)
        if "烽火地带" in compact and "长弓" in compact:
            return True
        return (
            VisionEngine._launcher_longbow_similarity(compact)
            >= VisionEngine._LAUNCHER_LONGBOW_MIN_SIMILARITY
        )

    @staticmethod
    def _launcher_longbow_similarity(text: str) -> float:
        compact = VisionEngine._normalize_ocr_text(text)
        if not compact:
            return 0.0
        return SequenceMatcher(
            None,
            VisionEngine._LAUNCHER_LONGBOW_TEXT,
            compact,
            autojunk=False,
        ).ratio()

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        return "".join(character for character in text if character.isalnum())

    @staticmethod
    def _looks_like_launcher_resource_title(text: str) -> bool:
        compact = VisionEngine._normalize_ocr_text(text)
        if compact.startswith("已下载") or "安装至" in compact:
            return False
        chinese_count = sum("\u4e00" <= character <= "\u9fff" for character in compact)
        return chinese_count >= 3


