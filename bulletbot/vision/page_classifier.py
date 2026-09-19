from __future__ import annotations

import numpy as np
from PIL import Image

from bulletbot.domain.models import PageObservation, PageState
from bulletbot.ocr.detail_ocr import DetailOcr
from bulletbot.vision.favorites_grid import FavoritesGridDetector
from bulletbot.vision.reference_layout import ReferenceLayout


class PageClassifier:
    """Classify the current game page from structural, low-cost visual cues."""

    _WAREHOUSE_TAB_RANGE = (0.134, 0.202)
    _MARKET_TAB_RANGE = (0.337, 0.405)
    _TAB_LINE_Y_RANGE = (0.058, 0.080)
    # Real active navigation underlines score at least about 28%. Background
    # decorations on the home screen can reach 10-12% in the same region.
    _ACTIVE_TAB_THRESHOLD = 0.18
    _FAVORITES_CATEGORY_X_RANGE = (0.045, 0.205)
    _FAVORITES_CATEGORY_Y_RANGE = (0.190, 0.255)
    _FAVORITES_CATEGORY_THRESHOLD = 0.70
    _SELL_SUBTAB_X_RANGE = (0.134, 0.203)
    _SUBTAB_LINE_Y_RANGE = (0.102, 0.125)
    _ACTIVE_SUBTAB_THRESHOLD = 0.03
    _LISTING_ACTION_BOUNDS = (0.600, 0.660, 0.770, 0.730)
    _LISTING_ACTION_THRESHOLD = 0.02
    _WAREHOUSE_SELL_HEADER_LINE_BOUNDS = (0.180, 0.252, 0.820, 0.267)
    _WAREHOUSE_SELL_UPPER_ACTION_BOUNDS = (0.675, 0.402, 0.785, 0.465)
    _WAREHOUSE_SELL_LOWER_ACTION_BOUNDS = (0.675, 0.618, 0.785, 0.680)
    _WAREHOUSE_SELL_ACTION_THRESHOLD = 0.045
    _WAREHOUSE_SELL_HEADER_THRESHOLD = 0.60

    def __init__(self) -> None:
        self._favorites_detector = FavoritesGridDetector()
        self._detail_detector = DetailOcr()

    def classify(self, image: Image.Image) -> PageObservation:
        frame = np.asarray(image.convert("RGB"))
        warehouse_sell_evidence = self._warehouse_sell_dialog_evidence(frame)
        if warehouse_sell_evidence is not None:
            header_score, upper_score, lower_score = warehouse_sell_evidence
            return PageObservation(
                state=PageState.WAREHOUSE_SELL_DIALOG,
                confidence=min(
                    0.99,
                    0.86 + min(upper_score, lower_score) + header_score * 0.04,
                ),
                evidence=(
                    f"出售页标题分隔线 {header_score:.0%}",
                    f"军需处出售按钮 {upper_score:.0%}",
                    f"交易行上架按钮 {lower_score:.0%}",
                ),
            )
        listing_score = self._green_score(frame, self._LISTING_ACTION_BOUNDS)
        if self._detail_detector.is_detail_page(image):
            return PageObservation(
                state=PageState.DETAIL,
                confidence=0.98,
                evidence=("检测到详情页绿色价格按钮",),
            )

        warehouse_score = self._active_tab_score(frame, self._WAREHOUSE_TAB_RANGE)
        market_score = self._active_tab_score(frame, self._MARKET_TAB_RANGE)
        if (
            warehouse_score >= self._ACTIVE_TAB_THRESHOLD
            and warehouse_score >= market_score + 0.04
        ):
            return PageObservation(
                state=PageState.WAREHOUSE,
                confidence=min(0.99, 0.70 + warehouse_score),
                evidence=(f"仓库导航高亮 {warehouse_score:.0%}",),
            )

        # A populated sell page contains enough repeated order and inventory
        # rectangles to resemble the favorites grid.  The active sell subtab
        # is the stronger page-specific signal and must win before grid scan.
        sell_subtab_score = self._green_score(
            frame,
            (
                self._SELL_SUBTAB_X_RANGE[0],
                self._SUBTAB_LINE_Y_RANGE[0],
                self._SELL_SUBTAB_X_RANGE[1],
                self._SUBTAB_LINE_Y_RANGE[1],
            ),
        )
        if (
            market_score >= self._ACTIVE_TAB_THRESHOLD
            and sell_subtab_score >= self._ACTIVE_SUBTAB_THRESHOLD
        ):
            return PageObservation(
                state=PageState.MARKET_SELL,
                confidence=min(0.99, 0.82 + sell_subtab_score),
                evidence=(
                    f"交易行导航高亮 {market_score:.0%}",
                    f"出售页签高亮 {sell_subtab_score:.0%}",
                ),
            )

        cards = self._favorites_detector.detect(image)
        favorites_category_score = self._horizontal_highlight_score(
            frame,
            self._FAVORITES_CATEGORY_X_RANGE,
            self._FAVORITES_CATEGORY_Y_RANGE,
        )
        if len(cards) >= 9 or (
            market_score >= self._ACTIVE_TAB_THRESHOLD
            and favorites_category_score >= self._FAVORITES_CATEGORY_THRESHOLD
        ):
            confidence = 0.98 if market_score >= self._ACTIVE_TAB_THRESHOLD else 0.90
            return PageObservation(
                state=PageState.FAVORITES,
                confidence=confidence,
                evidence=(
                    f"检测到 {len(cards)} 个收藏卡片",
                    f"交易行导航高亮 {market_score:.0%}",
                    f"收藏分类边框 {favorites_category_score:.0%}",
                ),
            )

        # Product cards contain green purchase-price regions in the same broad
        # area as the listing action button. Only treat that color as a listing
        # editor after the repeated favorites grid has been ruled out.
        if (
            market_score >= self._ACTIVE_TAB_THRESHOLD
            and listing_score >= self._LISTING_ACTION_THRESHOLD
        ):
            return PageObservation(
                state=PageState.LISTING_EDITOR,
                confidence=min(0.99, 0.88 + listing_score),
                evidence=(f"检测到上架编辑器绿色按钮 {listing_score:.0%}",),
            )

        if market_score >= self._ACTIVE_TAB_THRESHOLD:
            return PageObservation(
                state=PageState.MARKET_OTHER,
                confidence=min(0.95, 0.65 + market_score),
                evidence=(f"交易行导航高亮 {market_score:.0%}，未检测到收藏网格",),
            )

        return PageObservation(
            state=PageState.UNKNOWN,
            confidence=0.0,
            evidence=("未检测到已知页面结构",),
        )

    @classmethod
    def _warehouse_sell_dialog_evidence(
        cls,
        frame: np.ndarray,
    ) -> tuple[float, float, float] | None:
        header_score = cls._top_cyan_line_score(
            frame,
            cls._WAREHOUSE_SELL_HEADER_LINE_BOUNDS,
        )
        upper_score = cls._top_green_score(
            frame,
            cls._WAREHOUSE_SELL_UPPER_ACTION_BOUNDS,
        )
        lower_score = cls._top_green_score(
            frame,
            cls._WAREHOUSE_SELL_LOWER_ACTION_BOUNDS,
        )
        if (
            header_score < cls._WAREHOUSE_SELL_HEADER_THRESHOLD
            or upper_score < cls._WAREHOUSE_SELL_ACTION_THRESHOLD
            or lower_score < cls._WAREHOUSE_SELL_ACTION_THRESHOLD
        ):
            return None
        return header_score, upper_score, lower_score

    @classmethod
    def warehouse_tab_score(cls, image: Image.Image) -> float:
        return cls._active_tab_score(
            np.asarray(image.convert("RGB")),
            cls._WAREHOUSE_TAB_RANGE,
        )

    @classmethod
    def market_tab_score(cls, image: Image.Image) -> float:
        return cls._active_tab_score(
            np.asarray(image.convert("RGB")),
            cls._MARKET_TAB_RANGE,
        )

    @classmethod
    def listing_action_score(cls, image: Image.Image) -> float:
        """Return the green coverage of the editor's fixed listing button."""
        return cls._green_score(
            np.asarray(image.convert("RGB")),
            cls._LISTING_ACTION_BOUNDS,
        )

    @classmethod
    def has_listing_action(cls, image: Image.Image) -> bool:
        return cls.listing_action_score(image) >= cls._LISTING_ACTION_THRESHOLD

    @classmethod
    def has_fast_favorites_marker(cls, image: Image.Image) -> bool:
        """Check two small fixed regions that only coexist on the favorites page."""

        frame = np.asarray(image.convert("RGB"))
        market_score = cls._active_tab_score(frame, cls._MARKET_TAB_RANGE)
        category_score = cls._horizontal_highlight_score(
            frame,
            cls._FAVORITES_CATEGORY_X_RANGE,
            cls._FAVORITES_CATEGORY_Y_RANGE,
        )
        return bool(
            market_score >= cls._ACTIVE_TAB_THRESHOLD
            and category_score >= cls._FAVORITES_CATEGORY_THRESHOLD
        )

    @classmethod
    def has_fast_favorites_marker_crops(
        cls,
        market_tab: Image.Image,
        favorites_category: Image.Image,
    ) -> bool:
        """Check already-cropped fixed markers without processing a full frame."""

        market_frame = np.asarray(market_tab.convert("RGB")).astype(np.int16)
        category_frame = np.asarray(favorites_category.convert("RGB")).astype(np.int16)
        if market_frame.size == 0 or category_frame.size == 0:
            return False

        red = market_frame[:, :, 0]
        green = market_frame[:, :, 1]
        blue = market_frame[:, :, 2]
        turquoise = (
            (green >= 110)
            & (green >= red + 20)
            & (blue >= 80)
            & (blue >= red + 10)
        )
        market_rows = np.mean(turquoise, axis=1)
        market_score = float(
            np.mean(np.sort(market_rows)[-min(3, len(market_rows)) :])
        )

        channel_min = category_frame.min(axis=2)
        channel_max = category_frame.max(axis=2)
        neutral_bright = (channel_min >= 120) & ((channel_max - channel_min) <= 25)
        category_score = float(np.max(np.mean(neutral_bright, axis=1)))
        return bool(
            market_score >= cls._ACTIVE_TAB_THRESHOLD
            and category_score >= cls._FAVORITES_CATEGORY_THRESHOLD
        )

    @classmethod
    def _active_tab_score(
        cls,
        frame: np.ndarray,
        x_range: tuple[float, float],
        y_range: tuple[float, float] | None = None,
    ) -> float:
        height, width = frame.shape[:2]
        active_y_range = y_range or cls._TAB_LINE_Y_RANGE
        layout = ReferenceLayout(width, height)
        top = layout.top_y(active_y_range[0])
        bottom = max(top + 1, layout.top_y(active_y_range[1]))
        left = round(width * x_range[0])
        right = max(left + 1, round(width * x_range[1]))
        crop = frame[top:bottom, left:right].astype(np.int16)
        if crop.size == 0:
            return 0.0
        red = crop[:, :, 0]
        green = crop[:, :, 1]
        blue = crop[:, :, 2]
        turquoise = (
            (green >= 110)
            & (green >= red + 20)
            & (blue >= 80)
            & (blue >= red + 10)
        )
        # The game keeps tab underlines only a few physical pixels thick even
        # at high resolutions. Averaging the whole proportional ROI therefore
        # dilutes a valid line as the window gets taller. Score the strongest
        # three rows instead, which remains stable across aspect ratios.
        row_coverage = np.mean(turquoise, axis=1)
        strongest_rows = np.sort(row_coverage)[-min(3, len(row_coverage)) :]
        return float(np.mean(strongest_rows))

    @staticmethod
    def _green_score(
        frame: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> float:
        height, width = frame.shape[:2]
        left = round(width * bounds[0])
        top = round(height * bounds[1])
        right = max(left + 1, round(width * bounds[2]))
        bottom = max(top + 1, round(height * bounds[3]))
        crop = frame[top:bottom, left:right].astype(np.int16)
        if crop.size == 0:
            return 0.0
        red = crop[:, :, 0]
        green = crop[:, :, 1]
        blue = crop[:, :, 2]
        emphasized = (green >= 65) & (green >= red + 15) & (green >= blue + 5)
        return float(np.mean(emphasized))

    @staticmethod
    def _top_crop(
        frame: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        left, top, right, bottom = ReferenceLayout(width, height).top_box(bounds)
        return frame[top:bottom, left:right].astype(np.int16)

    @classmethod
    def _top_green_score(
        cls,
        frame: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> float:
        crop = cls._top_crop(frame, bounds)
        if crop.size == 0:
            return 0.0
        red = crop[:, :, 0]
        green = crop[:, :, 1]
        blue = crop[:, :, 2]
        emphasized = (green >= 65) & (green >= red + 15) & (green >= blue + 5)
        return float(np.mean(emphasized))

    @classmethod
    def _top_cyan_line_score(
        cls,
        frame: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> float:
        crop = cls._top_crop(frame, bounds)
        if crop.size == 0:
            return 0.0
        red = crop[:, :, 0]
        green = crop[:, :, 1]
        blue = crop[:, :, 2]
        turquoise = (
            (green >= 80)
            & (green >= red + 8)
            & (blue >= red + 5)
        )
        row_coverage = np.mean(turquoise, axis=1)
        strongest_rows = np.sort(row_coverage)[-min(3, len(row_coverage)) :]
        return float(np.mean(strongest_rows))

    @staticmethod
    def _horizontal_highlight_score(
        frame: np.ndarray,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> float:
        """Return the strongest neutral-bright horizontal line in an ROI."""

        height, width = frame.shape[:2]
        top = round(height * y_range[0])
        bottom = max(top + 1, round(height * y_range[1]))
        left = round(width * x_range[0])
        right = max(left + 1, round(width * x_range[1]))
        crop = frame[top:bottom, left:right].astype(np.int16)
        if crop.size == 0:
            return 0.0
        channel_min = crop.min(axis=2)
        channel_max = crop.max(axis=2)
        neutral_bright = (channel_min >= 120) & ((channel_max - channel_min) <= 25)
        return float(np.max(np.mean(neutral_bright, axis=1)))
