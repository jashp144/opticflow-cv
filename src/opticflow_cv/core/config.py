"""Configuration models for opticflow-cv pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpticalFlowConfig:
    """Configuration parameters for edge filtering, optical flow, and motion tracking."""

    # Flow method selection: "lucas_kanade", "farneback", "numpy_gradient"
    method: str = "lucas_kanade"

    # Dense Farneback parameters
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    # Lucas-Kanade Shi-Tomasi feature tracking parameters
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: float = 10.0
    block_size: int = 7
    lk_winsize: tuple[int, int] = (15, 15)
    lk_max_level: int = 2

    # Vector sampling and filtering
    motion_threshold: float = 0.5
    grid_step: int = 16
    max_trajectory_length: int = 50

    # Motion clustering parameters
    clustering_eps: float = 30.0
    clustering_min_samples: int = 3

    # Anomaly scoring parameters
    anomaly_window_size: int = 20
    anomaly_z_threshold: float = 2.5

    # Edge filtering parameters
    canny_low: float = 50.0
    canny_high: float = 150.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    gaussian_kernel_size: tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0

    def __post_init__(self) -> None:
        valid_methods = {"lucas_kanade", "farneback", "numpy_gradient"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid optical flow method '{self.method}'. Must be one of {valid_methods}"
            )
        if not (0.0 < self.pyr_scale < 1.0):
            raise ValueError(f"pyr_scale must be between 0.0 and 1.0, got {self.pyr_scale}")
        if self.winsize <= 0 or self.winsize % 2 == 0:
            raise ValueError(f"winsize must be positive odd integer, got {self.winsize}")
        if self.poly_n not in (5, 7):
            raise ValueError(f"poly_n typically must be 5 or 7, got {self.poly_n}")
        if self.motion_threshold < 0:
            raise ValueError(f"motion_threshold cannot be negative, got {self.motion_threshold}")
        if self.grid_step <= 0:
            raise ValueError(f"grid_step must be positive, got {self.grid_step}")
