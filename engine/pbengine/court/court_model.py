"""Pickleball court reference geometry.

A regulation pickleball court is 20 ft wide x 44 ft long. The net splits it across the
middle; the non-volley zone ("kitchen") extends 7 ft from the net on each side. Service
courts sit behind the kitchen, split left/right by the centerline.

We express the reference court in **normalized top-down coordinates** ``[0, 1] x [0, 1]``:
``x`` runs across the 20 ft width, ``y`` runs along the 44 ft length, with the net at
``y = 0.5``. Side A occupies ``y < 0.5``; side B occupies ``y > 0.5``. Downstream geometry
(which side a bounce landed on, in/out, which service court) is done in these coordinates
so it is independent of video resolution and camera placement.
"""

from __future__ import annotations

WIDTH_FT = 20.0
LENGTH_FT = 44.0
KITCHEN_FT = 7.0  # non-volley zone depth from the net

# Normalized landmark positions. Keys are stable names used when matching a court-detector's
# keypoints to the reference template for the homography solve.
NET_Y = 0.5
KITCHEN_A_Y = (LENGTH_FT / 2 - KITCHEN_FT) / LENGTH_FT  # kitchen line on side A
KITCHEN_B_Y = (LENGTH_FT / 2 + KITCHEN_FT) / LENGTH_FT  # kitchen line on side B
CENTER_X = 0.5

REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    # Outer corners (baseline x sideline).
    "corner_a_left": (0.0, 0.0),
    "corner_a_right": (1.0, 0.0),
    "corner_b_left": (0.0, 1.0),
    "corner_b_right": (1.0, 1.0),
    # Kitchen line x sideline intersections.
    "kitchen_a_left": (0.0, KITCHEN_A_Y),
    "kitchen_a_right": (1.0, KITCHEN_A_Y),
    "kitchen_b_left": (0.0, KITCHEN_B_Y),
    "kitchen_b_right": (1.0, KITCHEN_B_Y),
    # Centerline meets baselines and kitchen lines (centerline does not cross the kitchen).
    "baseline_a_center": (CENTER_X, 0.0),
    "baseline_b_center": (CENTER_X, 1.0),
    "kitchen_a_center": (CENTER_X, KITCHEN_A_Y),
    "kitchen_b_center": (CENTER_X, KITCHEN_B_Y),
    # Net x sideline.
    "net_left": (0.0, NET_Y),
    "net_right": (1.0, NET_Y),
}


def side_of(court_xy: tuple[float, float]) -> str:
    """Return ``"A"`` or ``"B"`` for a normalized court point (net at ``y = 0.5``)."""
    return "A" if court_xy[1] < NET_Y else "B"


def is_in_bounds(court_xy: tuple[float, float], margin: float = 0.0) -> bool:
    """Whether a normalized point lies on the court, with an optional tolerance ``margin``."""
    x, y = court_xy
    return -margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin


def service_side(court_xy: tuple[float, float]) -> str:
    """Left/right service court for a server standing at ``court_xy`` (from their baseline)."""
    return "right" if court_xy[0] >= CENTER_X else "left"
