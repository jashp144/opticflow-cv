"""opticflow-cv: Production-grade edge filtering and optical flow motion estimation."""

from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.core.models import FlowField, Frame, MotionVector, Trajectory
from opticflow_cv.filters.canny import CannyEdgeFilter
from opticflow_cv.filters.clahe import CLAHEFilter
from opticflow_cv.filters.edge import EdgeFilter
from opticflow_cv.filters.gaussian import GaussianFilter, LaplacianFilter
from opticflow_cv.filters.sobel import SobelFilter
from opticflow_cv.flow.farneback import FarnebackDenseFlow
from opticflow_cv.flow.lucas_kanade import LucasKanadeTracker
from opticflow_cv.flow.numpy_flow import NumPyGradientFlow
from opticflow_cv.motion.anomaly import MotionAnomalyDetector
from opticflow_cv.motion.clustering import MotionCluster, MotionClusterer
from opticflow_cv.motion.motion_field import MotionField
from opticflow_cv.pipeline import OpticalFlowPipeline
from opticflow_cv.visualization.visualizer import FlowVisualizer

__version__ = "1.0.0"

__all__ = [
    # Top-level public interface contracts
    "EdgeFilter",
    "OpticalFlowPipeline",
    "MotionField",
    # Core models and configs
    "Frame",
    "MotionVector",
    "FlowField",
    "Trajectory",
    "OpticalFlowConfig",
    # Filtering operators
    "SobelFilter",
    "CannyEdgeFilter",
    "CLAHEFilter",
    "GaussianFilter",
    "LaplacianFilter",
    # Flow engines
    "LucasKanadeTracker",
    "FarnebackDenseFlow",
    "NumPyGradientFlow",
    # Motion and clustering
    "MotionCluster",
    "MotionClusterer",
    "MotionAnomalyDetector",
    # Visualization
    "FlowVisualizer",
    "__version__",
]
