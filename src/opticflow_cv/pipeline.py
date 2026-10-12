"""OpticalFlowPipeline orchestrating filters, optical flow algorithms, and motion analytics."""

from __future__ import annotations

from typing import Any

import numpy as np

from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.core.models import FlowField, Frame, MotionVector, Trajectory
from opticflow_cv.flow.farneback import FarnebackDenseFlow
from opticflow_cv.flow.lucas_kanade import LucasKanadeTracker
from opticflow_cv.flow.numpy_flow import NumPyGradientFlow
from opticflow_cv.motion.anomaly import MotionAnomalyDetector
from opticflow_cv.motion.clustering import MotionClusterer
from opticflow_cv.motion.motion_field import MotionField


class OpticalFlowPipeline:
    """Production-grade pipeline executing edge filtering, optical flow, and motion tracking."""

    def __init__(
        self,
        method: str = "lucas_kanade",
        config: OpticalFlowConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self.config = config
            if method != "lucas_kanade":
                self.config.method = method
        else:
            self.config = OpticalFlowConfig(method=method, **kwargs)

        self.method = self.config.method.lower()

        # Instantiate solvers
        self._lk_tracker = LucasKanadeTracker(self.config)
        self._farneback_solver = FarnebackDenseFlow(self.config)
        self._numpy_solver = NumPyGradientFlow(self.config)

        # Motion analysis engines
        self._clusterer = MotionClusterer(
            eps=self.config.clustering_eps,
            min_samples=self.config.clustering_min_samples,
        )
        self._anomaly_detector = MotionAnomalyDetector(
            window_size=self.config.anomaly_window_size,
            z_threshold=self.config.anomaly_z_threshold,
        )

        # State for streaming processing
        self._prev_frame: Frame | np.ndarray | None = None
        self._frame_index: int = 0

    @property
    def trajectories(self) -> list[Trajectory]:
        """Return active feature trajectories tracked by Lucas-Kanade."""
        return self._lk_tracker.trajectories

    def reset(self) -> None:
        """Reset internal frame buffers, trajectory histories, and anomaly trackers."""
        self._prev_frame = None
        self._frame_index = 0
        self._lk_tracker.reset()
        self._anomaly_detector.reset()

    def process_frames(
        self,
        frame_prev: np.ndarray | Frame,
        frame_curr: np.ndarray | Frame,
        timestamp: float = 0.0,
    ) -> MotionField:
        """Execute optical flow and motion analysis across a pair of frames.

        Args:
            frame_prev: Reference/previous image frame (numpy array or Frame).
            frame_curr: Current target image frame (numpy array or Frame).
            timestamp: Optional timestamp for temporal velocity scaling.

        Returns:
            MotionField aggregating velocity vectors, flow grid, clusters, and anomaly score.
        """
        # Determine shape
        if isinstance(frame_curr, Frame):
            shape = (frame_curr.height, frame_curr.width)
        else:
            shape = (int(frame_curr.shape[0]), int(frame_curr.shape[1]))

        vectors: list[MotionVector] = []
        flow_field: FlowField | None = None

        if self.method == "lucas_kanade":
            vectors = self._lk_tracker.process_frames(frame_prev, frame_curr)
        elif self.method == "farneback":
            flow_field = self._farneback_solver.compute(frame_prev, frame_curr)
            vectors = flow_field.to_motion_vectors(
                step=self.config.grid_step,
                min_magnitude=self.config.motion_threshold,
            )
        elif self.method == "numpy_gradient":
            flow_field = self._numpy_solver.compute(frame_prev, frame_curr)
            vectors = flow_field.to_motion_vectors(
                step=self.config.grid_step,
                min_magnitude=self.config.motion_threshold,
            )
        else:
            raise ValueError(f"Unsupported pipeline optical flow method: {self.method}")

        # Cluster motion vectors into coherent objects
        clusters = self._clusterer.cluster(vectors)

        # Calculate average speed and direction for anomaly detection
        avg_speed = 0.0
        avg_dir_rad: float | None = None
        if vectors:
            avg_speed = sum(v.magnitude for v in vectors) / len(vectors)
            mean_vx = sum(v.vx for v in vectors) / len(vectors)
            mean_vy = sum(v.vy for v in vectors) / len(vectors)
            avg_dir_rad = float(np.arctan2(mean_vy, mean_vx))
        elif flow_field is not None:
            avg_speed = flow_field.mean_speed()
            u_m, v_m = flow_field.mean_velocity()
            avg_dir_rad = float(np.arctan2(v_m, u_m))

        anomaly_score = self._anomaly_detector.update(
            current_speed=avg_speed,
            current_direction_rad=avg_dir_rad,
        )

        return MotionField.from_vectors(
            vectors=vectors,
            flow_field=flow_field,
            clusters=clusters,
            anomaly_score=anomaly_score,
            timestamp=timestamp,
            frame_shape=shape,
        )

    def process(
        self,
        frame: np.ndarray | Frame,
        timestamp: float = 0.0,
    ) -> MotionField:
        """Stream a single frame into the pipeline, computing flow against the preceding frame."""
        self._frame_index += 1

        if self._prev_frame is None:
            self._prev_frame = frame
            shape = (
                (frame.height, frame.width)
                if isinstance(frame, Frame)
                else (int(frame.shape[0]), int(frame.shape[1]))
            )
            return MotionField.from_vectors(
                vectors=[],
                flow_field=None,
                clusters=[],
                anomaly_score=0.0,
                timestamp=timestamp,
                frame_shape=shape,
            )

        motion = self.process_frames(self._prev_frame, frame, timestamp=timestamp)
        self._prev_frame = frame
        return motion
