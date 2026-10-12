"""Boundary conditions, stress edge cases, and stability tests."""

from __future__ import annotations

import numpy as np
import pytest

from opticflow_cv import EdgeFilter, OpticalFlowPipeline
from opticflow_cv.core.models import Trajectory


class TestEdgeCases:
    """Stress testing edge cases and boundary conditions."""

    @pytest.mark.parametrize("method", ["lucas_kanade", "farneback", "numpy_gradient"])
    def test_identical_frames_zero_motion(self, method):
        # 80x80 textured frame against itself -> zero motion expected
        y, x = np.mgrid[0:80, 0:80]
        pattern = ((np.sin(x * 0.2) + np.cos(y * 0.2) + 2.0) * 60).astype(np.uint8)

        pipeline = OpticalFlowPipeline(method=method)
        motion = pipeline.process_frames(pattern, pattern)

        assert motion.velocity_vectors is not None
        # Either no vectors or all near-zero magnitude
        for v in motion.velocity_vectors:
            assert v.magnitude < 0.5

    @pytest.mark.parametrize("method", ["farneback", "numpy_gradient"])
    def test_uniform_frames_no_crash(self, method):
        white = np.full((64, 64), 255, dtype=np.uint8)
        black = np.zeros((64, 64), dtype=np.uint8)

        pipeline = OpticalFlowPipeline(method=method)
        # All white
        m1 = pipeline.process_frames(white, white)
        assert m1.dominant_speed == 0.0

        # All black
        m2 = pipeline.process_frames(black, black)
        assert m2.dominant_speed == 0.0

    def test_small_dimension_frames(self):
        small1 = np.random.randint(0, 255, size=(24, 24), dtype=np.uint8)
        small2 = np.random.randint(0, 255, size=(24, 24), dtype=np.uint8)

        pipeline = OpticalFlowPipeline(method="numpy_gradient", winsize=5)
        motion = pipeline.process_frames(small1, small2)
        assert motion.frame_shape == (24, 24)

    def test_non_square_aspect_ratios(self):
        wide1 = np.zeros((32, 128), dtype=np.uint8)
        wide2 = np.zeros((32, 128), dtype=np.uint8)
        wide1[:, 20:40] = 200
        wide2[:, 25:45] = 200

        pipeline = OpticalFlowPipeline(method="farneback")
        motion = pipeline.process_frames(wide1, wide2)
        assert motion.frame_shape == (32, 128)

        tall1 = np.zeros((128, 32), dtype=np.uint8)
        tall2 = np.zeros((128, 32), dtype=np.uint8)
        tall1[20:40, :] = 200
        tall2[25:45, :] = 200

        motion_tall = pipeline.process_frames(tall1, tall2)
        assert motion_tall.frame_shape == (128, 32)

    def test_trajectory_zero_dt(self):
        traj = Trajectory(track_id=1)
        traj.add_point(10.0, 10.0, timestamp=1.0)
        traj.add_point(15.0, 15.0, timestamp=1.0)  # dt = 0.0
        # Should not raise ZeroDivisionError
        speed = traj.average_speed()
        assert speed > 0.0
        vx, vy = traj.latest_velocity()
        assert (vx, vy) == (5.0, 5.0)

    def test_edge_filter_different_dtypes(self):
        float_img = np.random.rand(40, 40).astype(np.float32)
        edges = EdgeFilter.canny(float_img)
        assert edges.shape == (40, 40)

        clahe_out = EdgeFilter.clahe(float_img)
        assert clahe_out.shape == (40, 40)
