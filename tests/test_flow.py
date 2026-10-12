"""Unit and integration tests for optical flow solvers (Lucas-Kanade, Farneback, NumPy)."""

from __future__ import annotations

import numpy as np
import pytest

from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.flow.farneback import FarnebackDenseFlow
from opticflow_cv.flow.lucas_kanade import LucasKanadeTracker
from opticflow_cv.flow.numpy_flow import NumPyGradientFlow


class TestLucasKanadeTracker:
    """Tests for Lucas-Kanade sparse feature tracking."""

    def test_process_frames_synthetic_translation(self, synthetic_translating_pair):
        f1, f2 = synthetic_translating_pair
        tracker = LucasKanadeTracker()
        vectors = tracker.process_frames(f1, f2)

        assert len(vectors) > 0
        # The true synthetic shift was dx=+4, dy=+3
        mean_vx = np.mean([v.vx for v in vectors])
        mean_vy = np.mean([v.vy for v in vectors])

        assert mean_vx == pytest.approx(4.0, abs=1.5)
        assert mean_vy == pytest.approx(3.0, abs=1.5)

    def test_blank_frames_return_empty(self, blank_frame_pair):
        f1, f2 = blank_frame_pair
        tracker = LucasKanadeTracker()
        vectors = tracker.process_frames(f1, f2)
        assert vectors == []

    def test_step_streaming_and_trajectories(self, frame_sequence):
        tracker = LucasKanadeTracker()
        all_vectors = []

        for frame in frame_sequence:
            vecs, trajs = tracker.step(frame, timestamp=frame.timestamp)
            all_vectors.extend(vecs)

        # After 5 frames, trajectories should have multiple points
        assert len(tracker.trajectories) > 0
        multi_point_trajs = [t for t in tracker.trajectories if len(t) >= 2]
        assert len(multi_point_trajs) > 0

        # Check reset
        tracker.reset()
        assert len(tracker.trajectories) == 0

    def test_invalid_input_type(self):
        tracker = LucasKanadeTracker()
        with pytest.raises(ValueError, match="Invalid frame input"):
            tracker.process_frames("invalid", "invalid")  # type: ignore


class TestFarnebackDenseFlow:
    """Tests for Farneback polynomial expansion optical flow."""

    def test_compute_synthetic_translation(self, synthetic_translating_pair):
        f1, f2 = synthetic_translating_pair
        solver = FarnebackDenseFlow()
        flow = solver.compute(f1, f2)

        assert flow.shape == f1.shape
        # True shift is dx=+4, dy=+3 inside moving region (45:75, 45:75)
        region_u = flow.u[45:75, 45:75]
        region_v = flow.v[45:75, 45:75]

        assert np.mean(region_u) == pytest.approx(4.0, abs=1.5)
        assert np.mean(region_v) == pytest.approx(3.0, abs=1.5)

    def test_blank_frames_yield_zero_flow(self, blank_frame_pair):
        f1, f2 = blank_frame_pair
        solver = FarnebackDenseFlow()
        flow = solver.compute(f1, f2)

        assert np.all(flow.magnitude == 0.0)

    def test_shape_mismatch_raises(self):
        f1 = np.zeros((50, 50), dtype=np.uint8)
        f2 = np.zeros((60, 60), dtype=np.uint8)
        solver = FarnebackDenseFlow()
        with pytest.raises(ValueError, match="Frame shape mismatch"):
            solver.compute(f1, f2)

    def test_color_frames_support(self, synthetic_color_translating_pair):
        f1, f2 = synthetic_color_translating_pair
        solver = FarnebackDenseFlow()
        flow = solver.compute(f1, f2)
        assert flow.shape == (120, 160)
        assert flow.mean_speed() > 0.0


class TestNumPyGradientFlow:
    """Tests for pure-NumPy gradient optical flow solver."""

    def test_numpy_flow_translation(self, synthetic_translating_pair):
        f1, f2 = synthetic_translating_pair
        config = OpticalFlowConfig(winsize=15)
        solver = NumPyGradientFlow(config=config, min_eigenvalue=1e-3)
        flow = solver.compute(f1, f2)

        assert flow.shape == f1.shape
        assert flow.confidence is not None
        assert not np.isnan(flow.u).any()
        assert not np.isnan(flow.v).any()

        # Motion should be detected in textured moving region
        region_mag = flow.magnitude[45:75, 45:75]
        assert np.max(region_mag) > 1.0

    def test_numpy_blank_frame_zero_flow(self, blank_frame_pair):
        f1, f2 = blank_frame_pair
        solver = NumPyGradientFlow()
        flow = solver.compute(f1, f2)

        assert np.all(flow.u == 0.0)
        assert np.all(flow.v == 0.0)
        assert np.all(flow.magnitude == 0.0)

    def test_numpy_shape_mismatch(self):
        f1 = np.zeros((40, 40), dtype=np.uint8)
        f2 = np.zeros((50, 50), dtype=np.uint8)
        solver = NumPyGradientFlow()
        with pytest.raises(ValueError, match="Shape mismatch"):
            solver.compute(f1, f2)

    def test_numpy_color_handling(self, synthetic_color_translating_pair):
        f1, f2 = synthetic_color_translating_pair
        solver = NumPyGradientFlow()
        flow = solver.compute(f1, f2)
        assert flow.shape == (120, 160)
