"""Shared pytest fixtures and synthetic frame generators for opticflow-cv test suite."""

from __future__ import annotations

import numpy as np
import pytest

from opticflow_cv.core.models import Frame


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Ensure all tests run with deterministic random state."""
    np.random.seed(42)


@pytest.fixture
def synthetic_translating_pair() -> tuple[np.ndarray, np.ndarray]:
    """Generate pair of frames with a textured square translated by dx=4, dy=3."""
    h, w = 120, 160
    f1 = np.zeros((h, w), dtype=np.uint8)
    f2 = np.zeros((h, w), dtype=np.uint8)

    # Base square at (35, 35) to (85, 85)
    # Texture with smooth 2D gradient pattern preventing high-frequency phase aliasing
    y_grid, x_grid = np.ogrid[0:50, 0:50]
    pattern = ((np.sin(x_grid * 0.15) * np.cos(y_grid * 0.15) + 1.0) * 127).astype(np.uint8)

    f1[35:85, 35:85] = pattern
    f2[38:88, 39:89] = pattern  # dx = +4, dy = +3

    return (f1, f2)


@pytest.fixture
def synthetic_color_translating_pair() -> tuple[np.ndarray, np.ndarray]:
    """Generate 3-channel BGR translating frame pair."""
    h, w = 120, 160
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    f2 = np.zeros((h, w, 3), dtype=np.uint8)

    y_grid, x_grid = np.ogrid[0:50, 0:50]
    pattern = ((np.sin(x_grid * 0.15) * np.cos(y_grid * 0.15) + 1.0) * 127).astype(np.uint8)

    for c in range(3):
        f1[35:85, 35:85, c] = pattern
        f2[38:88, 39:89, c] = pattern

    return (f1, f2)


@pytest.fixture
def blank_frame_pair() -> tuple[np.ndarray, np.ndarray]:
    """Generate pair of uniform blank frames (zero motion)."""
    h, w = 100, 100
    f1 = np.full((h, w), 128, dtype=np.uint8)
    f2 = np.full((h, w), 128, dtype=np.uint8)
    return (f1, f2)


@pytest.fixture
def frame_sequence() -> list[Frame]:
    """Generate a 5-frame sequence of a translating square."""
    seq = []
    h, w = 100, 120
    for i in range(5):
        img = np.zeros((h, w), dtype=np.uint8)
        # Shift square across frames
        x0 = 20 + i * 4
        y0 = 30 + i * 2
        # Checkerboard patch inside square
        patch = np.indices((30, 30)).sum(axis=0) % 2 * 255
        img[y0 : y0 + 30, x0 : x0 + 30] = patch.astype(np.uint8)
        seq.append(Frame.from_array(img, frame_id=i, timestamp=i * 0.033))
    return seq
