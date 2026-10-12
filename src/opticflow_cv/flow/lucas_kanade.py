"""Lucas-Kanade sparse feature tracking using Shi-Tomasi corners and pyramidal flow."""

from __future__ import annotations

import cv2
import numpy as np

from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.core.models import Frame, MotionVector, Trajectory


class LucasKanadeTracker:
    """Tracks sparse salient features across frames using Pyramidal Lucas-Kanade."""

    def __init__(self, config: OpticalFlowConfig | None = None) -> None:
        self.config = config or OpticalFlowConfig()
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._trajectories: dict[int, Trajectory] = {}
        self._next_track_id: int = 1
        self._current_track_ids: list[int] = []

        # OpenCV LK and Shi-Tomasi parameter dictionaries
        self._lk_params = {
            "winSize": self.config.lk_winsize,
            "maxLevel": self.config.lk_max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        }
        self._feature_params = {
            "maxCorners": self.config.max_corners,
            "qualityLevel": self.config.quality_level,
            "minDistance": self.config.min_distance,
            "blockSize": self.config.block_size,
        }

    @property
    def trajectories(self) -> list[Trajectory]:
        """Return list of all currently tracked trajectories."""
        return list(self._trajectories.values())

    def reset(self) -> None:
        """Reset internal tracking state and history."""
        self._prev_gray = None
        self._prev_pts = None
        self._trajectories.clear()
        self._next_track_id = 1
        self._current_track_ids.clear()

    def _extract_gray(self, frame_input: np.ndarray | Frame) -> np.ndarray:
        if isinstance(frame_input, Frame):
            return frame_input.to_grayscale()
        if isinstance(frame_input, np.ndarray):
            if frame_input.ndim == 2:
                return frame_input
            if frame_input.ndim == 3 and frame_input.shape[2] in (3, 4):
                return cv2.cvtColor(frame_input, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Invalid frame input type {type(frame_input)}")

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect good features to track (Shi-Tomasi corners)."""
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
        if pts is not None and len(pts) > 0:
            return pts.astype(np.float32)
        return np.empty((0, 1, 2), dtype=np.float32)

    def process_frames(
        self,
        frame_prev: np.ndarray | Frame,
        frame_curr: np.ndarray | Frame,
    ) -> list[MotionVector]:
        """Compute sparse optical flow between two discrete frames."""
        gray_prev = self._extract_gray(frame_prev)
        gray_curr = self._extract_gray(frame_curr)

        pts_prev = self._detect_features(gray_prev)
        if len(pts_prev) == 0:
            return []

        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, pts_prev, None, **self._lk_params
        )

        vectors: list[MotionVector] = []
        if pts_curr is None or status is None:
            return []

        h, w = gray_curr.shape
        status_flat = status.reshape(-1)
        err_flat = err.reshape(-1) if err is not None else np.ones(len(status_flat))

        for i, matched in enumerate(status_flat):
            if matched == 1:
                p0 = pts_prev[i].reshape(2)
                p1 = pts_curr[i].reshape(2)
                # Check bounds
                if 0 <= p1[0] < w and 0 <= p1[1] < h:
                    vx = float(p1[0] - p0[0])
                    vy = float(p1[1] - p0[1])
                    mag = float(np.hypot(vx, vy))
                    if mag >= self.config.motion_threshold:
                        error_val = float(err_flat[i])
                        conf = max(0.0, 1.0 - (error_val / 50.0))
                        vectors.append(
                            MotionVector(
                                x=float(p0[0]),
                                y=float(p0[1]),
                                vx=vx,
                                vy=vy,
                                confidence=conf,
                            )
                        )
        return vectors

    def step(
        self,
        frame: np.ndarray | Frame,
        timestamp: float = 0.0,
    ) -> tuple[list[MotionVector], list[Trajectory]]:
        """Process streaming frame in sequence, updating persistent trajectories."""
        gray_curr = self._extract_gray(frame)
        vectors: list[MotionVector] = []

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) == 0:
            # First frame or re-initialization
            new_pts = self._detect_features(gray_curr)
            self._prev_gray = gray_curr
            self._prev_pts = new_pts
            self._current_track_ids = []
            for pt in new_pts:
                tid = self._next_track_id
                self._next_track_id += 1
                t = Trajectory(track_id=tid)
                t.add_point(float(pt[0, 0]), float(pt[0, 1]), timestamp)
                self._trajectories[tid] = t
                self._current_track_ids.append(tid)
            return ([], list(self._trajectories.values()))

        # Track existing points
        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray_curr, self._prev_pts, None, **self._lk_params
        )

        h, w = gray_curr.shape
        surviving_pts: list[list[float]] = []
        surviving_ids: list[int] = []

        if pts_curr is not None and status is not None:
            status_flat = status.reshape(-1)
            err_flat = err.reshape(-1) if err is not None else np.ones(len(status_flat))

            for i, matched in enumerate(status_flat):
                if matched == 1:
                    p0 = self._prev_pts[i].reshape(2)
                    p1 = pts_curr[i].reshape(2)
                    tid = self._current_track_ids[i]

                    if 0 <= p1[0] < w and 0 <= p1[1] < h:
                        vx = float(p1[0] - p0[0])
                        vy = float(p1[1] - p0[1])
                        mag = float(np.hypot(vx, vy))

                        # Update trajectory
                        traj = self._trajectories[tid]
                        traj.add_point(float(p1[0]), float(p1[1]), timestamp)
                        if len(traj.points) > self.config.max_trajectory_length:
                            traj.points.pop(0)
                            traj.timestamps.pop(0)

                        if mag >= self.config.motion_threshold:
                            error_val = float(err_flat[i])
                            conf = max(0.0, 1.0 - (error_val / 50.0))
                            vectors.append(
                                MotionVector(
                                    x=float(p0[0]),
                                    y=float(p0[1]),
                                    vx=vx,
                                    vy=vy,
                                    confidence=conf,
                                    track_id=tid,
                                )
                            )

                        surviving_pts.append([[p1[0], p1[1]]])
                        surviving_ids.append(tid)

        # Replenish features if track count drops below threshold
        if len(surviving_pts) < self.config.max_corners // 2:
            new_pts = self._detect_features(gray_curr)
            if len(new_pts) > 0:
                for pt in new_pts:
                    px, py = float(pt[0, 0]), float(pt[0, 1])
                    # Check distance against surviving points
                    too_close = False
                    for s in surviving_pts:
                        if np.hypot(px - s[0][0], py - s[0][1]) < self.config.min_distance:
                            too_close = True
                            break
                    if not too_close:
                        tid = self._next_track_id
                        self._next_track_id += 1
                        t = Trajectory(track_id=tid)
                        t.add_point(px, py, timestamp)
                        self._trajectories[tid] = t
                        surviving_pts.append([[px, py]])
                        surviving_ids.append(tid)

        self._prev_gray = gray_curr
        self._prev_pts = (
            np.array(surviving_pts, dtype=np.float32).reshape(-1, 1, 2)
            if surviving_pts
            else np.empty((0, 1, 2), dtype=np.float32)
        )
        self._current_track_ids = surviving_ids

        return (vectors, list(self._trajectories.values()))
