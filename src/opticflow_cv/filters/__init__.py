"""Edge detection, gradient operators, and spatial filters."""

from opticflow_cv.filters.canny import CannyEdgeFilter
from opticflow_cv.filters.clahe import CLAHEFilter
from opticflow_cv.filters.edge import EdgeFilter
from opticflow_cv.filters.gaussian import GaussianFilter, LaplacianFilter
from opticflow_cv.filters.sobel import SobelFilter

__all__ = [
    "EdgeFilter",
    "SobelFilter",
    "CannyEdgeFilter",
    "CLAHEFilter",
    "GaussianFilter",
    "LaplacianFilter",
]
