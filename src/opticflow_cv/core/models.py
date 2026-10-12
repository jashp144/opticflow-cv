"""Core mathematical data models for opticflow-cv.

Provides Frame, MotionVector, FlowField, and Trajectory structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class Frame:
    """Represents a video frame with image data, metadata, and validation."""

    data: np.ndarray
    frame_id: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"Frame data must be a numpy ndarray, got {type(self.data).__name__}")
        if self.data.ndim not in (2, 3):
            raise ValueError(f"Frame data must have 2 or 3 dimensions, got shape {self.data.shape}")
        if self.data.size == 0:
            raise ValueError("Frame data cannot be empty")

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
        **metadata: Any,
    ) -> Frame:
        """Create a Frame instance from a raw numpy array."""
        return cls(data=array, frame_id=frame_id, timestamp=timestamp, metadata=metadata)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return dimensions of the underlying image array."""
        return self.data.shape

    @property
    def height(self) -> int:
        """Return height in pixels."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Return width in pixels."""
        return int(self.data.shape[1])

    @property
    def is_color(self) -> bool:
        """Return True if image has multiple color channels."""
        return self.data.ndim == 3 and self.data.shape[2] in (3, 4)

    def to_grayscale(self) -> np.ndarray:
        """Convert frame data to 2D uint8 grayscale array."""
        img = self.data
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        if img.ndim == 2:
            return img

        if img.ndim == 3:
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        raise ValueError(f"Cannot convert array of shape {img.shape} to grayscale")


@dataclass(slots=True)
class MotionVector:
    """Represents a 2D spatial motion vector at (x, y) with displacement (vx, vy)."""

    x: float
    y: float
    vx: float
    vy: float
    confidence: float = 1.0
    track_id: int | None = None

    @property
    def magnitude(self) -> float:
        """Euclidean velocity magnitude (speed)."""
        return float(math.hypot(self.vx, self.vy))

    @property
    def angle(self) -> float:
        """Motion direction in radians, in range [-pi, pi]."""
        return float(math.atan2(self.vy, self.vx))

    @property
    def angle_degrees(self) -> float:
        """Motion direction in degrees, normalized to [0, 360)."""
        deg = math.degrees(math.atan2(self.vy, self.vx))
        return float(deg % 360.0)

    @property
    def target_x(self) -> float:
        """Estimated destination coordinate x + vx."""
        return self.x + self.vx

    @property
    def target_y(self) -> float:
        """Estimated destination coordinate y + vy."""
        return self.y + self.vy

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return (x, y, vx, vy) tuple."""
        return (self.x, self.y, self.vx, self.vy)

    def to_dict(self) -> dict[str, Any]:
        """Serialize vector to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "magnitude": self.magnitude,
            "angle_degrees": self.angle_degrees,
            "confidence": self.confidence,
            "track_id": self.track_id,
        }


@dataclass
class FlowField:
    """Represents a dense 2D optical flow field containing horizontal and vertical velocity grids."""

    u: np.ndarray  # horizontal displacement
    v: np.ndarray  # vertical displacement
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.u, np.ndarray) or not isinstance(self.v, np.ndarray):
            raise TypeError("Flow components u and v must be numpy ndarrays")
        if self.u.shape != self.v.shape:
            raise ValueError(f"Shape mismatch: u is {self.u.shape}, v is {self.v.shape}")
        if self.u.ndim != 2:
            raise ValueError(f"Flow components must be 2-dimensional, got {self.u.ndim}D")

        self.u = np.asarray(self.u, dtype=np.float32)
        self.v = np.asarray(self.v, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        """Return grid dimensions (height, width)."""
        return (int(self.u.shape[0]), int(self.u.shape[1]))

    @property
    def height(self) -> int:
        """Return field height."""
        return int(self.u.shape[0])

    @property
    def width(self) -> int:
        """Return field width."""
        return int(self.u.shape[1])

    @property
    def magnitude(self) -> np.ndarray:
        """Elementwise flow magnitude grid sqrt(u^2 + v^2)."""
        return np.hypot(self.u, self.v)

    @property
    def angle(self) -> np.ndarray:
        """Elementwise flow orientation in radians [-pi, pi]."""
        return np.arctan2(self.v, self.u)

    @property
    def angle_degrees(self) -> np.ndarray:
        """Elementwise flow orientation in degrees [0, 360)."""
        deg = np.rad2deg(np.arctan2(self.v, self.u))
        return np.mod(deg, 360.0)

    def mean_velocity(self) -> tuple[float, float]:
        """Calculate mean velocity components (mean_u, mean_v)."""
        return (float(np.mean(self.u)), float(np.mean(self.v)))

    def mean_speed(self) -> float:
        """Calculate mean speed across all pixels."""
        return float(np.mean(self.magnitude))

    def max_speed(self) -> float:
        """Calculate maximum speed across all pixels."""
        return float(np.max(self.magnitude))

    def divergence(self) -> np.ndarray:
        """Compute spatial divergence nabla . v = du/dx + dv/dy.

        Positive values indicate fluid expansion/source, negative values indicate sink/compression.
        """
        # np.gradient returns [grad_y, grad_x] for 2D array
        _, du_dx = np.gradient(self.u)
        dv_dy, _ = np.gradient(self.v)
        return du_dx + dv_dy

    def curl(self) -> np.ndarray:
        """Compute spatial vorticity (2D scalar curl) nabla x v = dv/dx - du/dy.

        Measures local angular rotational velocity.
        """
        _, dv_dx = np.gradient(self.v)
        du_dy, _ = np.gradient(self.u)
        return dv_dx - du_dy

    def to_motion_vectors(self, step: int = 16, min_magnitude: float = 0.5) -> list[MotionVector]:
        """Subsample dense field into discrete MotionVector instances at regular grid points."""
        if step <= 0:
            raise ValueError(f"Step must be positive, got {step}")

        h, w = self.shape
        vectors: list[MotionVector] = []
        mag = self.magnitude

        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                m = mag[y, x]
                if m >= min_magnitude:
                    conf = 1.0 if self.confidence is None else float(self.confidence[y, x])
                    vectors.append(
                        MotionVector(
                            x=float(x),
                            y=float(y),
                            vx=float(self.u[y, x]),
                            vy=float(self.v[y, x]),
                            confidence=conf,
                        )
                    )
        return vectors


@dataclass
class Trajectory:
    """Represents a continuous motion path over time for a tracked feature or object."""

    track_id: int
    points: list[tuple[float, float]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def add_point(self, x: float, y: float, timestamp: float = 0.0) -> None:
        """Append spatial position coordinate and associated timestamp."""
        self.points.append((float(x), float(y)))
        self.timestamps.append(float(timestamp))

    def __len__(self) -> int:
        return len(self.points)

    def path_length(self) -> float:
        """Calculate cumulative Euclidean arc length traversed."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.points)):
            p0 = self.points[i - 1]
            p1 = self.points[i]
            total += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        return float(total)

    def total_displacement(self) -> tuple[float, float]:
        """Calculate net displacement vector (delta_x, delta_y) from start to finish."""
        if len(self.points) < 2:
            return (0.0, 0.0)
        p_start = self.points[0]
        p_end = self.points[-1]
        return (p_end[0] - p_start[0], p_end[1] - p_start[1])

    def average_speed(self) -> float:
        """Calculate average speed along path."""
        if len(self.points) < 2:
            return 0.0
        dist = self.path_length()
        if len(self.timestamps) == len(self.points):
            dt = self.timestamps[-1] - self.timestamps[0]
            if dt > 1e-6:
                return float(dist / dt)
        return float(dist / (len(self.points) - 1))

    def latest_position(self) -> tuple[float, float]:
        """Return the most recent tracked point."""
        if not self.points:
            raise IndexError("Trajectory contains no points")
        return self.points[-1]

    def latest_velocity(self) -> tuple[float, float]:
        """Return the instantaneous velocity between the last two positions."""
        if len(self.points) < 2:
            return (0.0, 0.0)
        p0 = self.points[-2]
        p1 = self.points[-1]
        dt = 1.0
        if len(self.timestamps) >= 2:
            diff = self.timestamps[-1] - self.timestamps[-2]
            if diff > 1e-6:
                dt = diff
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)
