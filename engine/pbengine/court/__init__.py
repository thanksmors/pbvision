"""Court detection geometry: pickleball reference model + homography."""

from pbengine.court.court_model import (
    REFERENCE_POINTS,
    is_in_bounds,
    service_side,
    side_of,
)
from pbengine.court.homography import (
    compute_homography,
    homography_from_named_points,
    project,
)

__all__ = [
    "REFERENCE_POINTS",
    "compute_homography",
    "homography_from_named_points",
    "is_in_bounds",
    "project",
    "service_side",
    "side_of",
]
