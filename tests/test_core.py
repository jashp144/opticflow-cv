"""Unit tests for opticflow-cv core data models and configuration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.core.models import FlowField, Frame, MotionVector, Trajectory


class TestFrame:
    """Tests for Frame model, validation, and conversions."""

    def test_valid_frame_creation(self):
        arr = np.zeros((100, 100), dtype=np.uint8)
        frame = Frame.from_array(arr, frame_id=1, timestamp=0.5, source="camera_1")
        assert frame.height == 100
        assert frame.width == 100
        assert frame.shape == (100, 100)
        assert frame.frame_id == 1
        assert frame.timestamp == 0.5
        assert frame.metadata == {"source": "camera_1"}
        assert not frame.is_color

    def test_color_frame(self):
        arr = np.zeros((60, 80, 3), dtype=np.uint8)
        frame = Frame(data=arr)
        assert frame.is_color
        assert frame.height == 60
        assert frame.width == 80

    def test_invalid_frame_data_type(self):
        with pytest.raises(TypeError, match="numpy ndarray"):
            Frame(data=[[1, 2], [3, 4]])  # type: ignore

    def test_invalid_frame_dimensions(self):
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            Frame(data=np.zeros((10, 10, 10, 10)))
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            Frame(data=np.array([1, 2, 3]))

    def test_empty_frame(self):
        with pytest.raises(ValueError, match="empty"):
            Frame(data=np.empty((0, 0)))

    def test_to_grayscale_conversions(self):
        # 2D uint8
        g2d = np.ones((40, 40), dtype=np.uint8) * 120
        f2d = Frame(data=g2d)
        assert np.array_equal(f2d.to_grayscale(), g2d)

        # 3D BGR
        bgr = np.zeros((40, 40, 3), dtype=np.uint8)
        bgr[..., 0] = 255  # Blue channel
        f_bgr = Frame(data=bgr)
        gray_bgr = f_bgr.to_grayscale()
        assert gray_bgr.shape == (40, 40)
        assert gray_bgr.dtype == np.uint8
        assert gray_bgr[0, 0] > 0

        # Float in [0, 1]
        flt = np.ones((20, 20), dtype=np.float32) * 0.5
        f_flt = Frame(data=flt)
        gray_flt = f_flt.to_grayscale()
        assert gray_flt.dtype == np.uint8
        assert abs(int(gray_flt[0, 0]) - 127) <= 1

        # Float in [0, 255]
        flt255 = np.ones((20, 20), dtype=np.float32) * 200.0
        f_flt255 = Frame(data=flt255)
        gray_flt255 = f_flt255.to_grayscale()
        assert gray_flt255[0, 0] == 200


class TestMotionVector:
    """Tests for MotionVector representation and geometry."""

    def test_motion_vector_geometry(self):
        vec = MotionVector(x=10.0, y=20.0, vx=3.0, vy=4.0, confidence=0.9, track_id=42)
        assert vec.magnitude == pytest.approx(5.0)
        assert vec.target_x == pytest.approx(13.0)
        assert vec.target_y == pytest.approx(24.0)
        assert vec.angle == pytest.approx(math.atan2(4.0, 3.0))
        assert vec.angle_degrees == pytest.approx(math.degrees(math.atan2(4.0, 3.0)))
        assert vec.to_tuple() == (10.0, 20.0, 3.0, 4.0)

        data = vec.to_dict()
        assert data["magnitude"] == pytest.approx(5.0)
        assert data["track_id"] == 42
        assert data["confidence"] == pytest.approx(0.9)

    def test_negative_velocities_angle_normalization(self):
        vec = MotionVector(x=0.0, y=0.0, vx=-1.0, vy=-1.0)
        assert 0.0 <= vec.angle_degrees < 360.0
        assert vec.angle_degrees == pytest.approx(225.0)


class TestFlowField:
    """Tests for FlowField operations and vector calculus."""

    def test_flow_field_creation_and_shapes(self):
        u = np.ones((30, 40), dtype=np.float32) * 2.0
        v = np.ones((30, 40), dtype=np.float32) * 3.0
        field = FlowField(u=u, v=v)

        assert field.shape == (30, 40)
        assert field.height == 30
        assert field.width == 40
        assert field.mean_velocity() == (pytest.approx(2.0), pytest.approx(3.0))
        assert field.mean_speed() == pytest.approx(math.hypot(2.0, 3.0))
        assert field.max_speed() == pytest.approx(math.hypot(2.0, 3.0))

    def test_flow_field_validation(self):
        with pytest.raises(TypeError):
            FlowField(u=[1, 2], v=[1, 2])  # type: ignore
        with pytest.raises(ValueError, match="Shape mismatch"):
            FlowField(u=np.zeros((10, 10)), v=np.zeros((10, 12)))
        with pytest.raises(ValueError, match="2-dimensional"):
            FlowField(u=np.zeros((5, 5, 2)), v=np.zeros((5, 5, 2)))

    def test_divergence_and_curl(self):
        # Construct synthetic expansion field u = x, v = y
        y, x = np.mgrid[0:50, 0:50]
        u = (x - 25).astype(np.float32)
        v = (y - 25).astype(np.float32)
        field = FlowField(u=u, v=v)

        div = field.divergence()
        # In interior, du/dx = 1, dv/dy = 1 -> divergence ~ 2
        assert np.mean(div[10:40, 10:40]) == pytest.approx(2.0, abs=0.1)

        # In pure expansion, curl = dv/dx - du/dy = 0 - 0 = 0
        curl = field.curl()
        assert np.mean(np.abs(curl[10:40, 10:40])) == pytest.approx(0.0, abs=0.1)

    def test_to_motion_vectors_sampling(self):
        u = np.zeros((64, 64), dtype=np.float32)
        v = np.zeros((64, 64), dtype=np.float32)
        # Put motion in top-left region
        u[0:32, 0:32] = 4.0
        v[0:32, 0:32] = 3.0

        field = FlowField(u=u, v=v)
        vecs = field.to_motion_vectors(step=16, min_magnitude=1.0)
        assert len(vecs) > 0
        for vec in vecs:
            assert vec.magnitude >= 1.0

        with pytest.raises(ValueError, match="positive"):
            field.to_motion_vectors(step=0)


class TestTrajectory:
    """Tests for Trajectory temporal history and kinematics."""

    def test_trajectory_lifecycle(self):
        traj = Trajectory(track_id=1)
        assert len(traj) == 0
        assert traj.path_length() == 0.0
        assert traj.total_displacement() == (0.0, 0.0)
        assert traj.average_speed() == 0.0

        traj.add_point(0.0, 0.0, timestamp=0.0)
        assert traj.latest_position() == (0.0, 0.0)
        assert traj.latest_velocity() == (0.0, 0.0)

        traj.add_point(3.0, 4.0, timestamp=1.0)
        assert len(traj) == 2
        assert traj.path_length() == pytest.approx(5.0)
        assert traj.total_displacement() == (3.0, 4.0)
        assert traj.average_speed() == pytest.approx(5.0)
        assert traj.latest_velocity() == (3.0, 4.0)

        traj.add_point(6.0, 8.0, timestamp=2.0)
        assert traj.path_length() == pytest.approx(10.0)
        assert traj.total_displacement() == (6.0, 8.0)

    def test_empty_trajectory_error(self):
        traj = Trajectory(track_id=99)
        with pytest.raises(IndexError, match="no points"):
            traj.latest_position()


class TestOpticalFlowConfig:
    """Tests for OpticalFlowConfig validation."""

    def test_valid_default_config(self):
        cfg = OpticalFlowConfig()
        assert cfg.method == "lucas_kanade"
        assert cfg.winsize == 15
        assert cfg.motion_threshold == 0.5

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Invalid optical flow method"):
            OpticalFlowConfig(method="invalid_solver")

        with pytest.raises(ValueError, match="pyr_scale"):
            OpticalFlowConfig(pyr_scale=1.5)

        with pytest.raises(ValueError, match="positive odd"):
            OpticalFlowConfig(winsize=16)

        with pytest.raises(ValueError, match="poly_n"):
            OpticalFlowConfig(poly_n=9)

        with pytest.raises(ValueError, match="motion_threshold"):
            OpticalFlowConfig(motion_threshold=-1.0)

        with pytest.raises(ValueError, match="grid_step"):
            OpticalFlowConfig(grid_step=0)
