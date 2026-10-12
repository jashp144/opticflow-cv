"""MotionField container aggregating velocity vectors, clusters, and motion metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from opticflow_cv.core.models import FlowField, MotionVector
from opticflow_cv.motion.clustering import MotionCluster


@dataclass
class MotionField:
    """Aggregates all motion vectors, flow fields, object clusters, and dynamics for a frame."""

    velocity_vectors: list[MotionVector] = field(default_factory=list)
    flow_field: FlowField | None = None
    dominant_direction: float = 0.0  # Angle in degrees [0, 360)
    dominant_speed: float = 0.0
    clusters: list[MotionCluster] = field(default_factory=list)
    anomaly_score: float = 0.0  # Normalized [0.0, 1.0]
    timestamp: float = 0.0
    frame_shape: tuple[int, int] = (0, 0)

    def filter_by_magnitude(self, min_speed: float) -> list[MotionVector]:
        """Filter velocity vectors exceeding a minimum speed threshold."""
        return [v for v in self.velocity_vectors if v.magnitude >= min_speed]

    def mean_velocity(self) -> tuple[float, float]:
        """Compute the average (vx, vy) velocity vector across all detected vectors."""
        if not self.velocity_vectors:
            return (0.0, 0.0)
        mean_vx = sum(v.vx for v in self.velocity_vectors) / len(self.velocity_vectors)
        mean_vy = sum(v.vy for v in self.velocity_vectors) / len(self.velocity_vectors)
        return (float(mean_vx), float(mean_vy))

    def total_motion_energy(self) -> float:
        """Sum of squared velocity magnitudes (kinetic energy proxy)."""
        return sum(v.magnitude * v.magnitude for v in self.velocity_vectors)

    @classmethod
    def from_vectors(
        cls,
        vectors: list[MotionVector],
        flow_field: FlowField | None = None,
        clusters: list[MotionCluster] | None = None,
        anomaly_score: float = 0.0,
        timestamp: float = 0.0,
        frame_shape: tuple[int, int] = (0, 0),
    ) -> MotionField:
        """Construct MotionField calculating dominant direction and speed automatically."""
        dom_speed = 0.0
        dom_angle_deg = 0.0

        if vectors:
            # Weighted average direction based on vector magnitudes
            vx_sum = sum(v.vx * v.magnitude for v in vectors)
            vy_sum = sum(v.vy * v.magnitude for v in vectors)
            mag_sum = sum(v.magnitude for v in vectors)

            if mag_sum > 1e-6:
                avg_rad = math.atan2(vy_sum / mag_sum, vx_sum / mag_sum)
                dom_angle_deg = math.degrees(avg_rad) % 360.0
                dom_speed = sum(v.magnitude for v in vectors) / len(vectors)
        elif flow_field is not None:
            mean_u, mean_v = flow_field.mean_velocity()
            avg_rad = math.atan2(mean_v, mean_u)
            dom_angle_deg = math.degrees(avg_rad) % 360.0
            dom_speed = flow_field.mean_speed()

        return cls(
            velocity_vectors=vectors,
            flow_field=flow_field,
            dominant_direction=float(dom_angle_deg),
            dominant_speed=float(dom_speed),
            clusters=clusters or [],
            anomaly_score=float(anomaly_score),
            timestamp=float(timestamp),
            frame_shape=frame_shape,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MotionField state to dictionary."""
        return {
            "num_vectors": len(self.velocity_vectors),
            "dominant_direction_deg": self.dominant_direction,
            "dominant_speed": self.dominant_speed,
            "num_clusters": len(self.clusters),
            "anomaly_score": self.anomaly_score,
            "timestamp": self.timestamp,
            "frame_shape": self.frame_shape,
            "clusters": [c.to_dict() for c in self.clusters],
        }

    def summary(self) -> str:
        """Formatted human-readable motion telemetry string."""
        return (
            f"MotionField(vectors={len(self.velocity_vectors)}, "
            f"clusters={len(self.clusters)}, "
            f"speed={self.dominant_speed:.2f}px/f, "
            f"direction={self.dominant_direction:.1f}°, "
            f"anomaly={self.anomaly_score:.2f})"
        )
