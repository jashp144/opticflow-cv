"""Visual rendering utilities for optical flow vectors, HSV color wheels, and trajectories."""

from __future__ import annotations

import cv2
import numpy as np

from opticflow_cv.core.models import FlowField, MotionVector, Trajectory
from opticflow_cv.motion.clustering import MotionCluster
from opticflow_cv.motion.motion_field import MotionField


class FlowVisualizer:
    """Renders motion vectors, dense optical flow wheels, and trajectory trails on frames."""

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def draw_flow_arrows(
        self,
        image: np.ndarray,
        vectors_or_flow: list[MotionVector] | FlowField,
        step: int = 16,
        color: tuple[int, int, int] = (0, 255, 0),
        scale: float = 2.0,
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw motion arrows on the image representing velocity vectors."""
        canvas = self._ensure_bgr(image)

        if isinstance(vectors_or_flow, FlowField):
            vectors = vectors_or_flow.to_motion_vectors(step=step, min_magnitude=0.5)
        else:
            vectors = vectors_or_flow

        for v in vectors:
            p_start = (int(round(v.x)), int(round(v.y)))
            p_end = (
                int(round(v.x + v.vx * scale)),
                int(round(v.y + v.vy * scale)),
            )
            # Only draw if start and end are distinct
            if p_start != p_end:
                cv2.arrowedLine(
                    canvas,
                    p_start,
                    p_end,
                    color=color,
                    thickness=thickness,
                    tipLength=0.3,
                )

        return canvas

    def draw_flow_hsv(self, flow: FlowField) -> np.ndarray:
        """Render dense optical flow as a color-coded HSV image (Hue=direction, Value=magnitude)."""
        h, w = flow.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Hue channel: angle mapped to [0, 180) for OpenCV 8-bit HSV
        # np.arctan2 returns [-pi, pi], angle_degrees is [0, 360)
        angle_deg = flow.angle_degrees
        hsv[..., 0] = (angle_deg / 2.0).astype(np.uint8)

        # Saturation channel: full saturation
        hsv[..., 1] = 255

        # Value channel: flow magnitude scaled to [0, 255]
        mag = flow.magnitude
        max_mag = float(np.max(mag))
        if max_mag > 0:
            val_norm = np.clip(mag / max_mag * 255.0, 0, 255).astype(np.uint8)
        else:
            val_norm = np.zeros_like(mag, dtype=np.uint8)
        hsv[..., 2] = val_norm

        # Convert HSV to BGR for standard rendering
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def draw_trajectories(
        self,
        image: np.ndarray,
        trajectories: list[Trajectory],
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw historical path lines and terminal circles for tracked points."""
        canvas = self._ensure_bgr(image)

        for traj in trajectories:
            pts = traj.points
            if len(pts) < 2:
                if pts:
                    cv2.circle(canvas, (int(pts[0][0]), int(pts[0][1])), 3, color, -1)
                continue

            # Draw lines between consecutive points
            for i in range(1, len(pts)):
                p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                cv2.line(canvas, p0, p1, color, thickness)

            # Draw endpoint circle
            latest = (int(round(pts[-1][0])), int(round(pts[-1][1])))
            cv2.circle(canvas, latest, 4, (0, 255, 255), -1)

        return canvas

    def draw_motion_clusters(
        self,
        image: np.ndarray,
        clusters: list[MotionCluster],
        color: tuple[int, int, int] = (255, 128, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and telemetry tags for detected motion clusters."""
        canvas = self._ensure_bgr(image)

        for c in clusters:
            min_x, min_y, max_x, max_y = c.bounding_box
            cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), color, thickness)
            label = f"ID:{c.cluster_id} v:{c.average_speed:.1f} n:{c.member_count}"
            cv2.putText(
                canvas,
                label,
                (min_x, max(12, min_y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        return canvas

    def draw_hud(
        self,
        image: np.ndarray,
        motion_field: MotionField,
    ) -> np.ndarray:
        """Render diagnostic heads-up display overlay with motion telemetry metrics."""
        canvas = self._ensure_bgr(image)

        hud_lines = [
            f"Vectors: {len(motion_field.velocity_vectors)}",
            f"Speed: {motion_field.dominant_speed:.2f} px/f",
            f"Direction: {motion_field.dominant_direction:.1f} deg",
            f"Clusters: {len(motion_field.clusters)}",
            f"Anomaly: {motion_field.anomaly_score:.2f}",
        ]

        # Draw semi-transparent background box in top-left corner
        h, w = canvas.shape[:2]
        box_w = min(220, w)
        box_h = min(120, h)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (5, 5), (box_w, box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        y_offset = 24
        for line in hud_lines:
            text_color = (0, 255, 0)
            if "Anomaly" in line and motion_field.anomaly_score > 0.5:
                text_color = (0, 0, 255)
            cv2.putText(
                canvas,
                line,
                (12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA,
            )
            y_offset += 20

        return canvas
