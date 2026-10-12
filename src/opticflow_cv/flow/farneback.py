"""Farneback dense optical flow polynomial expansion solver."""

from __future__ import annotations

import cv2
import numpy as np

from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.core.models import FlowField, Frame


class FarnebackDenseFlow:
    """Computes dense optical flow field across all pixels using Farneback's algorithm."""

    def __init__(self, config: OpticalFlowConfig | None = None) -> None:
        self.config = config or OpticalFlowConfig()

    def _extract_gray(self, frame_input: np.ndarray | Frame) -> np.ndarray:
        if isinstance(frame_input, Frame):
            return frame_input.to_grayscale()
        if isinstance(frame_input, np.ndarray):
            if frame_input.ndim == 2:
                return (
                    frame_input.astype(np.uint8) if frame_input.dtype != np.uint8 else frame_input
                )
            if frame_input.ndim == 3 and frame_input.shape[2] in (3, 4):
                return cv2.cvtColor(frame_input, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Invalid frame input type {type(frame_input)}")

    def compute(
        self,
        frame_prev: np.ndarray | Frame,
        frame_curr: np.ndarray | Frame,
    ) -> FlowField:
        """Calculate dense optical flow between two frames."""
        gray_prev = self._extract_gray(frame_prev)
        gray_curr = self._extract_gray(frame_curr)

        if gray_prev.shape != gray_curr.shape:
            raise ValueError(
                f"Frame shape mismatch: prev is {gray_prev.shape}, curr is {gray_curr.shape}"
            )

        flow = cv2.calcOpticalFlowFarneback(
            prev=gray_prev,
            next=gray_curr,
            flow=None,
            pyr_scale=self.config.pyr_scale,
            levels=self.config.levels,
            winsize=self.config.winsize,
            iterations=self.config.iterations,
            poly_n=self.config.poly_n,
            poly_sigma=self.config.poly_sigma,
            flags=self.config.flags,
        )

        u = flow[..., 0].astype(np.float32)
        v = flow[..., 1].astype(np.float32)

        return FlowField(u=u, v=v)
