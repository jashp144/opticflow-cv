"""Gaussian smoothing and Laplacian edge detection operators."""

from __future__ import annotations

import cv2
import numpy as np


class GaussianFilter:
    """Applies 2D Gaussian spatial smoothing to suppress high-frequency noise."""

    def __init__(
        self,
        kernel_size: tuple[int, int] = (5, 5),
        sigma_x: float = 1.0,
        sigma_y: float = 0.0,
    ) -> None:
        kx, ky = kernel_size
        if kx <= 0 or kx % 2 == 0 or ky <= 0 or ky % 2 == 0:
            raise ValueError(
                f"kernel_size elements must be positive odd integers, got {kernel_size}"
            )
        if sigma_x < 0 or sigma_y < 0:
            raise ValueError("sigma parameters must be non-negative")

        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to image."""
        return cv2.GaussianBlur(
            image,
            self.kernel_size,
            sigmaX=self.sigma_x,
            sigmaY=self.sigma_y,
        )


class LaplacianFilter:
    """Applies second-order spatial derivative (Laplacian) to detect rapid intensity transitions."""

    def __init__(
        self,
        ksize: int = 3,
        scale: float = 1.0,
        delta: float = 0.0,
    ) -> None:
        if ksize not in (1, 3, 5, 7):
            raise ValueError(f"Laplacian kernel size must be 1, 3, 5, or 7, got {ksize}")
        self.ksize = ksize
        self.scale = scale
        self.delta = delta

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian operator and return normalized 8-bit image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        lap = cv2.Laplacian(
            gray,
            cv2.CV_32F,
            ksize=self.ksize,
            scale=self.scale,
            delta=self.delta,
        )
        abs_lap = np.abs(lap)
        max_val = np.max(abs_lap)
        if max_val > 0:
            return (abs_lap / max_val * 255.0).clip(0, 255).astype(np.uint8)
        return np.zeros_like(gray, dtype=np.uint8)
