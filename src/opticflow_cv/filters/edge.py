"""Unified EdgeFilter facade and edge detection dispatcher."""

from __future__ import annotations

from typing import Any

import numpy as np

from opticflow_cv.filters.canny import CannyEdgeFilter
from opticflow_cv.filters.clahe import CLAHEFilter
from opticflow_cv.filters.gaussian import GaussianFilter, LaplacianFilter
from opticflow_cv.filters.sobel import SobelFilter


class EdgeFilter:
    """Unified Edge and Gradient filtering facade for computer vision pipelines.

    Supports Canny edge detection, Sobel gradient estimation, Laplacian of Gaussian,
    Gaussian smoothing, and CLAHE adaptive contrast equalization.
    """

    def __init__(
        self,
        method: str = "canny",
        **params: Any,
    ) -> None:
        self.method = method.lower()
        self.params = params

        if self.method == "canny":
            low = params.get("low_threshold", 50.0)
            high = params.get("high_threshold", 150.0)
            aperture = params.get("aperture_size", 3)
            self._filter: Any = CannyEdgeFilter(
                low_threshold=low,
                high_threshold=high,
                aperture_size=aperture,
            )
        elif self.method == "sobel":
            ksize = params.get("ksize", 3)
            scale = params.get("scale", 1.0)
            self._filter = SobelFilter(ksize=ksize, scale=scale)
        elif self.method == "laplacian":
            ksize = params.get("ksize", 3)
            scale = params.get("scale", 1.0)
            self._filter = LaplacianFilter(ksize=ksize, scale=scale)
        elif self.method == "gaussian":
            ksize = params.get("kernel_size", (5, 5))
            sigma = params.get("sigma", 1.0)
            self._filter = GaussianFilter(kernel_size=ksize, sigma_x=sigma)
        elif self.method == "clahe":
            clip = params.get("clip_limit", 2.0)
            grid = params.get("tile_grid_size", (8, 8))
            self._filter = CLAHEFilter(clip_limit=clip, tile_grid_size=grid)
        else:
            valid = {"canny", "sobel", "laplacian", "gaussian", "clahe"}
            raise ValueError(f"Unknown EdgeFilter method '{method}'. Must be one of {valid}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply configured filter to target image."""
        return self._filter.apply(image)

    @staticmethod
    def canny(
        image: np.ndarray,
        low_threshold: float = 50.0,
        high_threshold: float = 150.0,
        aperture_size: int = 3,
    ) -> np.ndarray:
        """Execute Canny edge detection."""
        f = CannyEdgeFilter(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            aperture_size=aperture_size,
        )
        return f.detect(image)

    @staticmethod
    def sobel(
        image: np.ndarray,
        ksize: int = 3,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Execute Sobel gradient magnitude filtering."""
        f = SobelFilter(ksize=ksize, scale=scale)
        return f.apply(image)

    @staticmethod
    def laplacian(
        image: np.ndarray,
        ksize: int = 3,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Execute Laplacian edge detection."""
        f = LaplacianFilter(ksize=ksize, scale=scale)
        return f.apply(image)

    @staticmethod
    def gaussian_blur(
        image: np.ndarray,
        kernel_size: tuple[int, int] = (5, 5),
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Execute Gaussian spatial smoothing."""
        f = GaussianFilter(kernel_size=kernel_size, sigma_x=sigma)
        return f.apply(image)

    @staticmethod
    def clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Execute CLAHE contrast equalization."""
        f = CLAHEFilter(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        return f.apply(image)
