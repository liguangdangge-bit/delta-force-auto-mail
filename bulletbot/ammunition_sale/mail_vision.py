from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080
MAIL_LIST_RECT = (90, 135, 490, 895)
MAIL_TAB_RECTS = (
    (127, 52, 257, 102),
    (257, 52, 387, 102),
    (387, 52, 518, 102),
    (518, 52, 648, 102),
)
MAIL_TAB_SELECTED_GREEN_RATIO = 0.04
MAIL_CLAIM_ACTION_RECT = (1534, 917, 1833, 977)
MAIL_ATTACHMENT_CARD_TOP = 890
MAIL_ATTACHMENT_CARD_BOTTOM = 977
MAIL_ATTACHMENT_CARD_MIN_WIDTH = 68
MAIL_ATTACHMENT_CARD_MAX_WIDTH = 105
MAIL_ATTACHMENT_EDGE_GROUP_DISTANCE = 12


@dataclass(frozen=True)
class MailListSelection:
    top: int
    bottom: int

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_normalized(self) -> tuple[float, float]:
        return (0.15, self.center_y / REFERENCE_HEIGHT)

    @property
    def next_row_normalized(self) -> tuple[float, float]:
        gap = 4
        return (
            0.15,
            (self.bottom + gap + self.height / 2) / REFERENCE_HEIGHT,
        )


def normalize_mail_frame(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] == (REFERENCE_HEIGHT, REFERENCE_WIDTH):
        return image
    return cv2.resize(
        image,
        (REFERENCE_WIDTH, REFERENCE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def selected_mail_row(image: np.ndarray) -> MailListSelection | None:
    """Locate the white border around the selected mail list row."""

    normalized = normalize_mail_frame(image)
    left, top, right, bottom = MAIL_LIST_RECT
    gray = cv2.cvtColor(normalized[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    bright_counts = np.sum(gray > 180, axis=1)
    line_indices = np.flatnonzero(bright_counts >= 250)
    groups: list[tuple[int, int]] = []
    for raw_index in line_indices:
        index = int(raw_index)
        if groups and index <= groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], index)
        else:
            groups.append((index, index))

    candidates: list[tuple[float, MailListSelection]] = []
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1 :]:
            separation = second[0] - first[1]
            if separation > 90:
                break
            if not 50 <= separation <= 85:
                continue
            first_strength = float(np.mean(bright_counts[first[0] : first[1] + 1]))
            second_strength = float(
                np.mean(bright_counts[second[0] : second[1] + 1])
            )
            candidates.append(
                (
                    min(first_strength, second_strength),
                    MailListSelection(
                        top=top + first[0],
                        bottom=top + second[1],
                    ),
                )
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def mail_list_signature(image: np.ndarray) -> np.ndarray:
    normalized = normalize_mail_frame(image)
    left, top, right, bottom = MAIL_LIST_RECT
    gray = cv2.cvtColor(normalized[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (200, 380), interpolation=cv2.INTER_AREA)


def selected_mail_tab(image: np.ndarray) -> int | None:
    """Return the green-highlighted mail tab: system, notice, market, or security."""

    normalized = normalize_mail_frame(image)
    ratios: list[float] = []
    for left, top, right, bottom in MAIL_TAB_RECTS:
        hsv = cv2.cvtColor(normalized[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        green = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 70)
        )
        ratios.append(float(np.mean(green)))
    if not ratios or max(ratios) < MAIL_TAB_SELECTED_GREEN_RATIO:
        return None
    return int(np.argmax(ratios))


def mail_claim_available(image: np.ndarray) -> bool:
    """Distinguish the green claim action from a grey claimed-mail delete action."""

    normalized = normalize_mail_frame(image)
    left, top, right, bottom = MAIL_CLAIM_ACTION_RECT
    crop = normalized[top:bottom, left:right]
    blue, green, red = cv2.split(crop.astype(np.float32))
    green_pixels = (
        (green > 90)
        & (green > red * 1.25)
        & (green > blue * 1.05)
    )
    return float(np.mean(green_pixels)) >= 0.02


def numbered_attachment_points(
    image: np.ndarray,
    card_bounds: list[tuple[int, int, int, int]],
) -> list[tuple[float, float]]:
    """Return cards whose lower-right corner contains a bright stack count."""

    normalized = normalize_mail_frame(image)
    points: list[tuple[float, float]] = []
    for left, top, right, bottom in card_bounds:
        quantity = normalized[
            top + round((bottom - top) * 0.52) : bottom - 3,
            max(left + 5, right - round((right - left) * 0.40)) : right - 4,
        ]
        if quantity.size == 0:
            continue
        gray = cv2.cvtColor(quantity, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray > 150)) < 0.035:
            continue
        points.append(
            (
                ((left + right) / 2) / REFERENCE_WIDTH,
                ((top + bottom) / 2) / REFERENCE_HEIGHT,
            )
        )
    return points


def attachment_card_bounds(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    normalized = normalize_mail_frame(image)
    gray = cv2.cvtColor(
        normalized[
            MAIL_ATTACHMENT_CARD_TOP:MAIL_ATTACHMENT_CARD_BOTTOM,
            570:1530,
        ],
        cv2.COLOR_BGR2GRAY,
    )
    edge_strength = np.mean(
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)),
        axis=0,
    )
    threshold = max(100.0, float(np.percentile(edge_strength, 97)))
    peaks = [
        x
        for x in range(2, len(edge_strength) - 2)
        if edge_strength[x] >= threshold
        and edge_strength[x] >= np.max(edge_strength[x - 2 : x + 3])
    ]
    groups: list[list[int]] = []
    for peak in peaks:
        if groups and peak - groups[-1][-1] <= MAIL_ATTACHMENT_EDGE_GROUP_DISTANCE:
            groups[-1].append(peak)
        else:
            groups.append([peak])
    candidates = [
        max(group, key=lambda x: edge_strength[x]) + 570
        for group in groups
    ]
    paths: list[list[int]] = []
    for index, boundary in enumerate(candidates):
        path = [boundary]
        for previous_index in range(index):
            distance = boundary - candidates[previous_index]
            if not MAIL_ATTACHMENT_CARD_MIN_WIDTH <= distance <= MAIL_ATTACHMENT_CARD_MAX_WIDTH:
                continue
            candidate_path = paths[previous_index] + [boundary]
            if len(candidate_path) > len(path):
                path = candidate_path
        paths.append(path)
    boundaries = max(paths, key=len, default=[])
    return [
        (left, MAIL_ATTACHMENT_CARD_TOP, right, MAIL_ATTACHMENT_CARD_BOTTOM)
        for left, right in zip(boundaries, boundaries[1:])
    ]
