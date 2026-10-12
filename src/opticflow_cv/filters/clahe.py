"""Contrast Limited Adaptive Histogram Equalization (CLAHE) operator."""

from __future__ import annotations

import cv2
import numpy as np


class CLAHEFilter:
    """Applies Contrast Limited Adaptive Histogram Equalization to prevent over-amplification."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}")
        if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
            raise ValueError(f"tile_grid_size elements must be positive, got {tile_grid_size}")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tile_grid_size,
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement to grayscale or BGR color image."""
        if image.ndim == 2:
            uint8_img = image if image.dtype == np.uint8 else image.clip(0, 255).astype(np.uint8)
            return self._clahe.apply(uint8_img)

        if image.ndim == 3 and image.shape[2] == 3:
            uint8_img = image if image.dtype == np.uint8 else image.clip(0, 255).astype(np.uint8)
            lab = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = self._clahe.apply(lab_planes[0])
            merged = cv2.merge(lab_planes)
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        raise ValueError(f"Unsupported image shape for CLAHE: {image.shape}")
