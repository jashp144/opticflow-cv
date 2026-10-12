"""Motion anomaly, burstiness, and sudden acceleration detector."""

from __future__ import annotations

import math
from collections import deque


class MotionAnomalyDetector:
    """Monitors temporal velocity dynamics to score sudden acceleration or anomalous motion bursts."""

    def __init__(
        self,
        window_size: int = 20,
        z_threshold: float = 2.5,
        min_history: int = 5,
    ) -> None:
        if window_size < 3:
            raise ValueError("window_size must be at least 3")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")

        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_history = min_history
        self._speed_history: deque[float] = deque(maxlen=window_size)
        self._prev_direction: float | None = None

    def reset(self) -> None:
        """Clear temporal history."""
        self._speed_history.clear()
        self._prev_direction = None

    def update(
        self,
        current_speed: float,
        current_direction_rad: float | None = None,
    ) -> float:
        """Evaluate new frame velocity statistics and compute anomaly score in [0.0, 1.0]."""
        speed = max(0.0, float(current_speed))

        if len(self._speed_history) < self.min_history:
            self._speed_history.append(speed)
            self._prev_direction = current_direction_rad
            return 0.0

        # Calculate baseline statistics over previous history window
        history_list = list(self._speed_history)
        n = len(history_list)
        mean_speed = sum(history_list) / n
        variance = sum((s - mean_speed) ** 2 for s in history_list) / n
        std_speed = math.sqrt(variance)

        # Append current speed to rolling history
        self._speed_history.append(speed)

        # Speed deviation Z-score
        eps = 1e-4
        z_score = abs(speed - mean_speed) / (std_speed + eps)

        # Normalized burst score from Z-score
        burst_score = min(1.0, z_score / self.z_threshold)

        # Direction reversal penalty
        reversal_penalty = 0.0
        if current_direction_rad is not None and self._prev_direction is not None:
            # Angle difference wrapped to [-pi, pi]
            diff = abs(current_direction_rad - self._prev_direction)
            diff = min(diff, 2.0 * math.pi - diff)
            if diff > 0.75 * math.pi and speed > 1.0:
                # Near 180-degree flip under high speed indicates sharp bounce or jerk
                reversal_penalty = 0.3

        self._prev_direction = current_direction_rad

        # Composite anomaly score clamped to [0.0, 1.0]
        anomaly_score = min(1.0, burst_score * 0.7 + reversal_penalty)
        return float(anomaly_score)
