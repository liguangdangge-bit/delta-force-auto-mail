from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bulletbot.domain.models import CardRegion, Rect


@dataclass(frozen=True)
class _GridGeometry:
    left: int
    right: int
    row_edges: tuple[int, ...]
    columns: int


class FavoritesGridDetector:
    """Detect the visible favorites grid from the game frame itself.

    The detector intentionally does not use desktop coordinates or a fixed
    1920x1080 profile. It finds the sidebar, the page scrollbar, and repeated
    horizontal card boundaries, then derives the card rectangles from those
    measured elements.
    """

    _MIN_FRAME_WIDTH = 960
    _MIN_FRAME_HEIGHT = 540
    _CARD_WIDTH_TO_HEIGHT = 3.0

    def detect(self, image: Image.Image) -> list[CardRegion]:
        frame = np.asarray(image.convert("RGB"))
        geometry = self._find_geometry(frame)
        if geometry is None:
            return []
        return self._build_cards(geometry, frame.shape[1], frame.shape[0])

    def _find_geometry(self, frame: np.ndarray) -> _GridGeometry | None:
        height, width = frame.shape[:2]
        if width < self._MIN_FRAME_WIDTH or height < self._MIN_FRAME_HEIGHT:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 12, 35)
        vertical_lines = self._find_vertical_lines(edges, width, height)
        sidebar = self._find_sidebar(vertical_lines, width, height)
        scrollbar = self._find_scrollbar(vertical_lines, width, height)
        if sidebar is None or scrollbar is None:
            return None

        sidebar_left, sidebar_right, sidebar_top = sidebar
        del sidebar_left
        horizontal_lines = self._find_horizontal_lines(edges, width, height)
        grid_left = sidebar_right + max(16, round(width * 0.021))
        grid_right = scrollbar - max(16, round(width * 0.021))
        if grid_right - grid_left < width * 0.35:
            return None

        row_edges = self._find_row_edges(
            horizontal_lines=horizontal_lines,
            grid_left=grid_left,
            grid_right=grid_right,
            top_hint=sidebar_top,
            frame_height=height,
        )
        if row_edges is None:
            return None

        median_height = int(np.median(np.diff(np.asarray(row_edges))))
        columns = int(round((grid_right - grid_left) / (median_height * self._CARD_WIDTH_TO_HEIGHT)))
        columns = max(1, min(columns, 5))
        return _GridGeometry(
            left=grid_left,
            right=grid_right,
            row_edges=row_edges,
            columns=columns,
        )

    @staticmethod
    def _find_vertical_lines(edges: np.ndarray, width: int, height: int) -> list[tuple[int, int, int]]:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(60, round(width * 0.04)),
            minLineLength=round(height * 0.25),
            maxLineGap=round(height * 0.02),
        )
        if lines is None:
            return []
        result: list[tuple[int, int, int]] = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(int(x2) - int(x1)) <= 2:
                result.append((round((int(x1) + int(x2)) / 2), min(int(y1), int(y2)), abs(int(y2) - int(y1))))
        return result

    @staticmethod
    def _find_horizontal_lines(edges: np.ndarray, width: int, height: int) -> list[tuple[int, int, int]]:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(75, round(width * 0.07)),
            minLineLength=round(width * 0.18),
            maxLineGap=round(width * 0.014),
        )
        if lines is None:
            return []
        result: list[tuple[int, int, int]] = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(int(y2) - int(y1)) <= 2:
                result.append((min(int(x1), int(x2)), round((int(y1) + int(y2)) / 2), abs(int(x2) - int(x1))))
        return result

    @staticmethod
    def _find_sidebar(
        lines: list[tuple[int, int, int]], width: int, height: int
    ) -> tuple[int, int, int] | None:
        candidates = [line for line in lines if line[0] < width * 0.45 and line[2] > height * 0.30]
        best: tuple[int, int, int] | None = None
        best_score = -1
        for left_x, left_y, left_length in candidates:
            for right_x, right_y, right_length in candidates:
                sidebar_width = right_x - left_x
                if not (width * 0.10 <= sidebar_width <= width * 0.32):
                    continue
                overlap_top = max(left_y, right_y)
                overlap_bottom = min(left_y + left_length, right_y + right_length)
                overlap = overlap_bottom - overlap_top
                if overlap < height * 0.28:
                    continue
                score = overlap + min(left_length, right_length)
                if score > best_score:
                    best = (left_x, right_x, overlap_top)
                    best_score = score
        return best

    @staticmethod
    def _find_scrollbar(lines: list[tuple[int, int, int]], width: int, height: int) -> int | None:
        candidates = [line for line in lines if line[0] > width * 0.65 and line[2] > height * 0.50]
        if not candidates:
            return None
        return max(candidates, key=lambda line: line[2])[0]

    @staticmethod
    def _find_row_edges(
        horizontal_lines: list[tuple[int, int, int]],
        grid_left: int,
        grid_right: int,
        top_hint: int,
        frame_height: int,
    ) -> tuple[int, ...] | None:
        min_overlap = (grid_right - grid_left) * 0.24
        candidates = [
            y
            for x, y, line_width in horizontal_lines
            if y >= top_hint - frame_height * 0.08
            and x + line_width >= grid_left
            and x <= grid_right
            and min(x + line_width, grid_right) - max(x, grid_left) >= min_overlap
        ]
        if not candidates:
            return None
        clusters: list[list[int]] = []
        tolerance = max(4, round(frame_height * 0.008))
        for y in sorted(candidates):
            if not clusters or y - clusters[-1][-1] > tolerance:
                clusters.append([y])
            else:
                clusters[-1].append(y)
        lines = [round(float(np.median(cluster))) for cluster in clusters]
        return FavoritesGridDetector._longest_regular_sequence(
            lines,
            frame_height,
            top_hint,
        )

    @staticmethod
    def _longest_regular_sequence(
        lines: list[int],
        frame_height: int,
        top_hint: int,
    ) -> tuple[int, ...] | None:
        min_gap = frame_height * 0.07
        max_gap = frame_height * 0.22
        best: list[int] = []
        for start in range(len(lines)):
            sequence = [lines[start]]
            expected_gap: float | None = None
            for line in lines[start + 1 :]:
                gap = line - sequence[-1]
                if expected_gap is None and min_gap <= gap <= max_gap:
                    sequence.append(line)
                    expected_gap = float(np.median(np.diff(np.asarray(sequence))))
                    continue
                if expected_gap is None:
                    continue

                multiple = max(1, round(gap / expected_gap))
                matches_expected_gap = abs(gap - multiple * expected_gap) <= expected_gap * 0.24
                if multiple <= 3 and matches_expected_gap:
                    for _ in range(1, multiple):
                        sequence.append(round(sequence[-1] + expected_gap))
                    sequence.append(line)
                    expected_gap = float(np.median(np.diff(np.asarray(sequence))))
                elif gap > expected_gap * 1.24:
                    break
            if len(sequence) > len(best):
                best = sequence
        # A dark bottom row can hide one horizontal card boundary from the
        # edge detector. Three regular boundaries still define the row pitch;
        # the leading/trailing edges below can reconstruct the full grid.
        if len(best) < 3:
            return None
        gaps = np.diff(np.asarray(best))
        median_gap = int(round(float(np.median(gaps))))
        inferred_leading_edge = best[0] - median_gap
        leading_min = max(0, top_hint - round(frame_height * 0.08))
        leading_max = top_hint + round(frame_height * 0.03)
        if leading_min <= inferred_leading_edge <= leading_max:
            best.insert(0, inferred_leading_edge)
        if len(best) < 6:
            best.append(best[-1] + median_gap)
        if len(best) < 5 or best[-1] > frame_height * 0.97:
            return None
        return tuple(best)

    @staticmethod
    def _build_cards(geometry: _GridGeometry, width: int, height: int) -> list[CardRegion]:
        del height
        column_gap = max(4, round(width * 0.006))
        card_width = (geometry.right - geometry.left - column_gap * (geometry.columns - 1)) // geometry.columns
        cards: list[CardRegion] = []
        for row, (top, bottom) in enumerate(zip(geometry.row_edges, geometry.row_edges[1:])):
            card_height = bottom - top
            for column in range(geometry.columns):
                left = geometry.left + column * (card_width + column_gap)
                bounds = Rect(left, top, card_width, card_height)
                name_bounds = Rect(
                    left + round(card_width * 0.02),
                    top + round(card_height * 0.02),
                    round(card_width * 0.55),
                    round(card_height * 0.22),
                )
                price_left = left + round(card_width * 0.62)
                price_right = min(
                    width,
                    left + card_width + max(32, round(card_width * 0.07)),
                )
                price_bounds = Rect(
                    price_left,
                    top + round(card_height * 0.66),
                    price_right - price_left,
                    round(card_height * 0.32),
                )
                cards.append(
                    CardRegion(
                        index=row * geometry.columns + column + 1,
                        column=column,
                        row=row,
                        bounds=bounds,
                        name_bounds=name_bounds,
                        price_bounds=price_bounds,
                    )
                )
        return cards

    @staticmethod
    def annotate(image: Image.Image, cards: list[CardRegion]) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for card in cards:
            draw.rectangle(
                (card.bounds.left, card.bounds.top, card.bounds.right, card.bounds.bottom),
                outline="#36d399",
                width=2,
            )
            draw.rectangle(
                (
                    card.name_bounds.left,
                    card.name_bounds.top,
                    card.name_bounds.right,
                    card.name_bounds.bottom,
                ),
                outline="#facc15",
                width=2,
            )
            draw.rectangle(
                (
                    card.price_bounds.left,
                    card.price_bounds.top,
                    card.price_bounds.right,
                    card.price_bounds.bottom,
                ),
                outline="#38bdf8",
                width=2,
            )
            draw.text((card.bounds.left + 4, card.bounds.top + 4), f"C{card.index}", fill="#ffffff")
        return annotated
