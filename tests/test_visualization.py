"""Unit tests for FlowVisualizer rendering methods."""

from __future__ import annotations

import numpy as np
import pytest

from opticflow_cv.core.models import FlowField, MotionVector, Trajectory
from opticflow_cv.motion.clustering import MotionCluster
from opticflow_cv.motion.motion_field import MotionField
from opticflow_cv.visualization.visualizer import FlowVisualizer


class TestFlowVisualizer:
    """Tests for drawing optical flow arrows, HSV color wheels, and tracks."""

    @pytest.fixture
    def visualizer(self) -> FlowVisualizer:
        return FlowVisualizer()

    def test_draw_flow_arrows_with_vectors(self, visualizer):
        img = np.zeros((100, 100), dtype=np.uint8)
        vecs = [
            MotionVector(x=20.0, y=20.0, vx=5.0, vy=5.0),
            MotionVector(x=50.0, y=50.0, vx=-3.0, vy=4.0),
        ]
        out = visualizer.draw_flow_arrows(img, vecs, scale=2.0)
        assert out.shape == (100, 100, 3)
        assert out.dtype == np.uint8
        # Pixels where arrows are drawn should not be purely black
        assert np.sum(out > 0) > 0

    def test_draw_flow_arrows_with_flow_field(self, visualizer):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        u = np.full((64, 64), 3.0, dtype=np.float32)
        v = np.full((64, 64), 2.0, dtype=np.float32)
        field = FlowField(u=u, v=v)

        out = visualizer.draw_flow_arrows(img, field, step=16)
        assert out.shape == (64, 64, 3)
        assert np.sum(out > 0) > 0

    def test_draw_flow_hsv(self, visualizer):
        u = np.ones((50, 60), dtype=np.float32) * 4.0
        v = np.ones((50, 60), dtype=np.float32) * -3.0
        field = FlowField(u=u, v=v)

        hsv_bgr = visualizer.draw_flow_hsv(field)
        assert hsv_bgr.shape == (50, 60, 3)
        assert hsv_bgr.dtype == np.uint8
        assert np.max(hsv_bgr) > 0

    def test_draw_trajectories(self, visualizer):
        img = np.zeros((80, 80), dtype=np.uint8)

        # 1-point trajectory
        t1 = Trajectory(track_id=1)
        t1.add_point(10.0, 10.0)

        # Multi-point trajectory
        t2 = Trajectory(track_id=2)
        t2.add_point(20.0, 20.0)
        t2.add_point(25.0, 30.0)
        t2.add_point(30.0, 40.0)

        out = visualizer.draw_trajectories(img, [t1, t2])
        assert out.shape == (80, 80, 3)
        assert np.sum(out > 0) > 0

    def test_draw_motion_clusters(self, visualizer):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        c = MotionCluster(
            cluster_id=1,
            centroid=(40.0, 40.0),
            average_velocity=(3.0, 0.0),
            average_speed=3.0,
            member_count=5,
            bounding_box=(20, 20, 60, 60),
        )
        out = visualizer.draw_motion_clusters(img, [c])
        assert out.shape == (100, 100, 3)
        assert np.sum(out > 0) > 0

    def test_draw_hud(self, visualizer):
        img = np.zeros((120, 120), dtype=np.uint8)
        field = MotionField(
            velocity_vectors=[MotionVector(x=10.0, y=10.0, vx=2.0, vy=2.0)],
            dominant_direction=45.0,
            dominant_speed=2.83,
            anomaly_score=0.8,
        )
        out = visualizer.draw_hud(img, field)
        assert out.shape == (120, 120, 3)
        assert np.sum(out > 0) > 0

    def test_bgra_conversion(self, visualizer):
        bgra = np.zeros((50, 50, 4), dtype=np.uint8)
        out = visualizer.draw_flow_arrows(bgra, [])
        assert out.shape == (50, 50, 3)
