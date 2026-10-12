"""Pure-NumPy optical flow solver using gradient constraint and box filtering."""

from __future__ import annotations

import numpy as np

from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.core.models import FlowField, Frame


def _box_sum_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Compute 2D sliding box sum of size (2*radius + 1) using fast cumulative sums."""
    h, w = arr.shape
    k = 2 * radius + 1

    # Pad array symmetrically
    padded = np.pad(arr, radius, mode="edge")

    # Cumulative sum along rows
    cs_y = np.cumsum(padded, axis=0)
    # Window sum along y: cs_y[k:] - cs_y[:-k]
    y_sum = np.empty((h, padded.shape[1]), dtype=np.float32)
    y_sum[0] = cs_y[k - 1]
    y_sum[1:] = cs_y[k:] - cs_y[:-k]

    # Cumulative sum along columns
    cs_x = np.cumsum(y_sum, axis=1)
    out = np.empty((h, w), dtype=np.float32)
    out[:, 0] = cs_x[:, k - 1]
    out[:, 1:] = cs_x[:, k:] - cs_x[:, :-k]

    return out


class NumPyGradientFlow:
    """Pure-NumPy implementation of Lucas-Kanade optical flow.

    Provides a zero-C++-dependency fallback solver with eigenvalue conditioning.
    """

    def __init__(
        self,
        config: OpticalFlowConfig | None = None,
        min_eigenvalue: float = 1e-3,
    ) -> None:
        self.config = config or OpticalFlowConfig()
        self.min_eigenvalue = min_eigenvalue
        self.radius = max(1, self.config.winsize // 2)

    def _to_gray_float(self, frame_input: np.ndarray | Frame) -> np.ndarray:
        if isinstance(frame_input, Frame):
            gray = frame_input.to_grayscale()
        elif isinstance(frame_input, np.ndarray):
            if frame_input.ndim == 2:
                gray = frame_input
            elif frame_input.ndim == 3:
                # Standard Rec.601 luma weights
                gray = (
                    0.299 * frame_input[..., 0]
                    + 0.587 * frame_input[..., 1]
                    + 0.114 * frame_input[..., 2]
                )
            else:
                raise ValueError(f"Invalid frame dimensions: {frame_input.ndim}")
        else:
            raise TypeError(f"Expected np.ndarray or Frame, got {type(frame_input)}")

        return np.asarray(gray, dtype=np.float32)

    def compute(
        self,
        frame_prev: np.ndarray | Frame,
        frame_curr: np.ndarray | Frame,
    ) -> FlowField:
        """Compute dense optical flow using pure-NumPy vectorized normal equations."""
        i1 = self._to_gray_float(frame_prev)
        i2 = self._to_gray_float(frame_curr)

        if i1.shape != i2.shape:
            raise ValueError(f"Shape mismatch: {i1.shape} vs {i2.shape}")

        # Compute spatio-temporal derivatives
        # Temporal gradient: I_t = I2 - I1
        it = i2 - i1

        # Spatial gradients on temporal average frame
        i_avg = 0.5 * (i1 + i2)
        iy, ix = np.gradient(i_avg)

        # Elements of structural tensor M = [[A, B], [B, C]]
        # A = sum(Ix^2), B = sum(Ix * Iy), C = sum(Iy^2)
        a = _box_sum_2d(ix * ix, self.radius)
        b = _box_sum_2d(ix * iy, self.radius)
        c = _box_sum_2d(iy * iy, self.radius)

        # Right-hand side temporal projections b_vec = [-sum(Ix * It), -sum(Iy * It)]
        b1 = -_box_sum_2d(ix * it, self.radius)
        b2 = -_box_sum_2d(iy * it, self.radius)

        # Determinant and trace
        det = a * c - b * b
        trace = a + c

        # Smallest eigenvalue lambda2 = (trace - sqrt((A - C)^2 + 4*B^2)) / 2
        diff = a - c
        discriminant = np.maximum(0.0, diff * diff + 4.0 * b * b)
        lambda2 = 0.5 * (trace - np.sqrt(discriminant))

        # Condition mask: valid where smallest eigenvalue is sufficiently positive
        valid_mask = (lambda2 >= self.min_eigenvalue) & (det > 1e-7)

        safe_det = np.where(valid_mask, det, 1.0)

        # Solve system: u = (C*b1 - B*b2) / det, v = (-B*b1 + A*b2) / det
        u = np.where(valid_mask, (c * b1 - b * b2) / safe_det, 0.0).astype(np.float32)
        v = np.where(valid_mask, (-b * b1 + a * b2) / safe_det, 0.0).astype(np.float32)

        # Normalize confidence to [0, 1] based on lambda2
        max_lambda = np.max(lambda2)
        if max_lambda > 0:
            confidence = (np.clip(lambda2 / max_lambda, 0.0, 1.0)).astype(np.float32)
        else:
            confidence = np.zeros_like(u, dtype=np.float32)

        return FlowField(u=u, v=v, confidence=confidence)
