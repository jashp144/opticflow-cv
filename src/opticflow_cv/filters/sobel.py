"""Sobel spatial gradient filtering operator."""

from __future__ import annotations

import cv2
import numpy as np


class SobelFilter:
    """Computes spatial derivatives and gradient magnitude/orientation using Sobel kernels."""

    def __init__(
        self,
        ksize: int = 3,
        scale: float = 1.0,
        delta: float = 0.0,
    ) -> None:
        if ksize not in (1, 3, 5, 7):
            raise ValueError(f"Sobel kernel size must be 1, 3, 5, or 7, got {ksize}")
        self.ksize = ksize
        self.scale = scale
        self.delta = delta

    def compute_gradients(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute horizontal, vertical gradients, magnitude, and orientation.

        Returns:
            Tuple of (gx, gy, magnitude, angle_radians).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=self.ksize,
            scale=self.scale,
            delta=self.delta,
        )
        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=self.ksize,
            scale=self.scale,
            delta=self.delta,
        )

        magnitude = np.hypot(gx, gy)
        angle = np.arctan2(gy, gx)
        return (gx, gy, magnitude, angle)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel filter and return 8-bit normalized gradient magnitude image."""
        _, _, magnitude, _ = self.compute_gradients(image)
        max_val = np.max(magnitude)
        if max_val > 0:
            norm = (magnitude / max_val * 255.0).clip(0, 255).astype(np.uint8)
        else:
            norm = np.zeros_like(magnitude, dtype=np.uint8)
        return norm
