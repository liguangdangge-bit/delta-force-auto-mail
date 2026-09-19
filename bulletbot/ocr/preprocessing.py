from __future__ import annotations

import cv2
import numpy as np


class ProductNamePreprocessor:
    """Prepare narrow game UI labels for recognition without losing glyphs."""

    MINIMUM_CONFIDENCE = 0.75

    @staticmethod
    def tighten_label(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[0] < 4 or image.shape[1] < 4:
            return image
        brightness = image.max(axis=2)
        foreground = brightness >= 75
        columns = np.flatnonzero(foreground.sum(axis=0) >= 2)
        rows = np.flatnonzero(foreground.sum(axis=1) >= 2)
        if columns.size == 0 or rows.size == 0:
            return image
        left = max(0, int(columns[0]) - 10)
        right = min(image.shape[1], int(columns[-1]) + 11)
        top = max(0, int(rows[0]) - 6)
        bottom = min(image.shape[0], int(rows[-1]) + 7)
        if right - left < 20 or bottom - top < 8:
            return image
        crop = image[top:bottom, left:right]
        if crop.shape[0] < 32:
            scale = 32 / crop.shape[0]
            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        return crop

    @staticmethod
    def fallback_variants(image: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return conservative retries used only after a weak raw result."""

        if image.ndim != 3:
            return ()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(clahe, 1.65, blurred, -0.65, 0)
        _threshold, binary = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return (
            cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB),
        )
