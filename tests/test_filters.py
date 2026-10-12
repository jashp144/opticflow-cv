"""Unit tests for edge, gradient, and contrast enhancement filters."""

from __future__ import annotations

import numpy as np
import pytest

from opticflow_cv.filters.canny import CannyEdgeFilter
from opticflow_cv.filters.clahe import CLAHEFilter
from opticflow_cv.filters.edge import EdgeFilter
from opticflow_cv.filters.gaussian import GaussianFilter, LaplacianFilter
from opticflow_cv.filters.sobel import SobelFilter


class TestSobelFilter:
    """Tests for SobelFilter spatial gradient calculations."""

    def test_horizontal_and_vertical_edges(self):
        # Vertical step edge: columns 0..24 are 0, columns 25..49 are 255
        img_vert = np.zeros((50, 50), dtype=np.uint8)
        img_vert[:, 25:] = 255

        sobel = SobelFilter(ksize=3)
        gx, gy, mag, angle = sobel.compute_gradients(img_vert)

        assert gx.shape == (50, 50)
        assert gy.shape == (50, 50)
        assert mag.shape == (50, 50)
        assert angle.shape == (50, 50)

        # Along the vertical boundary (column 24-25), gx magnitude is high
        assert np.max(np.abs(gx[:, 24:26])) > 500.0
        # Horizontal gradient should dominate over vertical gradient
        assert np.mean(np.abs(gx[:, 24:26])) > np.mean(np.abs(gy[:, 24:26])) * 5.0

    def test_apply_produces_normalized_uint8(self):
        img = np.zeros((40, 40), dtype=np.uint8)
        img[10:30, 10:30] = 200
        sobel = SobelFilter()
        out = sobel.apply(img)
        assert out.dtype == np.uint8
        assert out.shape == (40, 40)
        assert np.max(out) > 0

    def test_blank_image_safe(self):
        blank = np.zeros((30, 30), dtype=np.uint8)
        sobel = SobelFilter()
        out = sobel.apply(blank)
        assert np.all(out == 0)

    def test_color_image_handled(self):
        color = np.zeros((30, 30, 3), dtype=np.uint8)
        color[10:20, 10:20, 0] = 255
        sobel = SobelFilter()
        gx, _, _, _ = sobel.compute_gradients(color)
        assert gx.shape == (30, 30)

    def test_invalid_kernel_size(self):
        with pytest.raises(ValueError, match="Sobel kernel size"):
            SobelFilter(ksize=4)


class TestCannyEdgeFilter:
    """Tests for CannyEdgeFilter hysteresis edge detection."""

    def test_edge_detection_on_rectangle(self):
        img = np.zeros((80, 80), dtype=np.uint8)
        img[20:60, 20:60] = 255

        canny = CannyEdgeFilter(low_threshold=50.0, high_threshold=150.0)
        edges = canny.detect(img)

        assert edges.dtype == np.uint8
        assert edges.shape == (80, 80)
        # Edges should exist along boundary
        assert np.sum(edges > 0) > 0
        # Center of rectangle should be clean of edges
        assert edges[40, 40] == 0

    def test_apply_alias(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 255
        canny = CannyEdgeFilter()
        assert np.array_equal(canny.apply(img), canny.detect(img))

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="non-negative"):
            CannyEdgeFilter(low_threshold=-1.0)
        with pytest.raises(ValueError, match="must not exceed"):
            CannyEdgeFilter(low_threshold=200.0, high_threshold=100.0)
        with pytest.raises(ValueError, match="aperture_size"):
            CannyEdgeFilter(aperture_size=4)


class TestCLAHEFilter:
    """Tests for CLAHE adaptive histogram equalization."""

    def test_contrast_enhancement(self):
        # Low-contrast image with narrow pixel range [100, 120]
        low_contrast = np.random.randint(100, 121, size=(64, 64), dtype=np.uint8)
        clahe = CLAHEFilter(clip_limit=3.0, tile_grid_size=(8, 8))
        enhanced = clahe.apply(low_contrast)

        assert enhanced.shape == (64, 64)
        assert enhanced.dtype == np.uint8
        # Dynamic range should be broadened
        assert (int(np.max(enhanced)) - int(np.min(enhanced))) >= (
            int(np.max(low_contrast)) - int(np.min(low_contrast))
        )

    def test_color_clahe(self):
        color = np.random.randint(50, 150, size=(48, 48, 3), dtype=np.uint8)
        clahe = CLAHEFilter()
        enhanced = clahe.apply(color)
        assert enhanced.shape == (48, 48, 3)
        assert enhanced.dtype == np.uint8

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="clip_limit"):
            CLAHEFilter(clip_limit=0.0)
        with pytest.raises(ValueError, match="tile_grid_size"):
            CLAHEFilter(tile_grid_size=(0, 8))
        with pytest.raises(ValueError, match="Unsupported image shape"):
            CLAHEFilter().apply(np.zeros((10, 10, 5)))


class TestGaussianAndLaplacian:
    """Tests for GaussianFilter and LaplacianFilter."""

    def test_gaussian_smoothing(self):
        noisy = np.random.normal(128, 20, size=(50, 50)).clip(0, 255).astype(np.uint8)
        gauss = GaussianFilter(kernel_size=(5, 5), sigma_x=1.5)
        smoothed = gauss.apply(noisy)

        assert smoothed.shape == (50, 50)
        # Smoothing reduces standard deviation of white noise
        assert np.std(smoothed) < np.std(noisy)

    def test_gaussian_validation(self):
        with pytest.raises(ValueError, match="positive odd"):
            GaussianFilter(kernel_size=(4, 4))
        with pytest.raises(ValueError, match="non-negative"):
            GaussianFilter(sigma_x=-1.0)

    def test_laplacian_edge_detection(self):
        img = np.zeros((60, 60), dtype=np.uint8)
        img[20:40, 20:40] = 255
        lap = LaplacianFilter(ksize=3)
        edges = lap.apply(img)

        assert edges.shape == (60, 60)
        assert edges.dtype == np.uint8
        assert np.max(edges) > 0

    def test_laplacian_validation(self):
        with pytest.raises(ValueError, match="Laplacian kernel size"):
            LaplacianFilter(ksize=2)


class TestEdgeFilterFacade:
    """Tests for unified EdgeFilter facade and static dispatchers."""

    def test_facade_methods(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        img[15:35, 15:35] = 255

        for method in ["canny", "sobel", "laplacian", "gaussian", "clahe"]:
            f = EdgeFilter(method=method)
            res = f.apply(img)
            assert res.shape == (50, 50)
            assert res.dtype == np.uint8

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown EdgeFilter method"):
            EdgeFilter(method="unsupported")

    def test_static_methods(self):
        img = np.zeros((40, 40), dtype=np.uint8)
        img[10:30, 10:30] = 200

        assert EdgeFilter.canny(img).shape == (40, 40)
        assert EdgeFilter.sobel(img).shape == (40, 40)
        assert EdgeFilter.laplacian(img).shape == (40, 40)
        assert EdgeFilter.gaussian_blur(img).shape == (40, 40)
        assert EdgeFilter.clahe(img).shape == (40, 40)
