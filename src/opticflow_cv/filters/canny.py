"""Canny multi-stage edge detection operator."""

from __future__ import annotations

import cv2
import numpy as np


class CannyEdgeFilter:
    """Detects edges using the multi-stage Canny algorithm with hysteresis thresholding."""

    def __init__(
        self,
        low_threshold: float = 50.0,
        high_threshold: float = 150.0,
        aperture_size: int = 3,
        l2_gradient: bool = True,
    ) -> None:
        if low_threshold < 0 or high_threshold < 0:
            raise ValueError("Thresholds must be non-negative")
        if low_threshold > high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must not exceed high_threshold ({high_threshold})"
            )
        if aperture_size not in (3, 5, 7):
            raise ValueError(f"aperture_size must be 3, 5, or 7, got {aperture_size}")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Execute Canny edge detection returning binary edge map."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if gray.dtype != np.uint8:
            gray = gray.clip(0, 255).astype(np.uint8)

        edges = cv2.Canny(
            gray,
            threshold1=self.low_threshold,
            threshold2=self.high_threshold,
            apertureSize=self.aperture_size,
            L2gradient=self.l2_gradient,
        )
        return edges

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Alias for detect() satisfying unified filter interface."""
        return self.detect(image)
