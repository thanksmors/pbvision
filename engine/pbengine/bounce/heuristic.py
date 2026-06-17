"""Heuristic bounce detection from a ball trajectory.

The homography flattens height onto the ground plane, so bounces are detected in **pixel**
space: as the ball falls it moves *down* the image (pixel-y increases) and a bounce is the
lowest point — a local maximum in pixel-y where vertical velocity flips from positive
(falling) to negative (rising). A v1.1 upgrade trains an ``sktime`` TimeSeriesForestClassifier
on (x, y, velocity) features; the heuristic is enough to drive rally/winner logic for v1.
"""

from __future__ import annotations

from pbengine.court.court_model import is_in_bounds, side_of
from pbengine.schema.models import BallSample, Bounce, Team


def detect_bounces(
    trajectory: list[BallSample], window: int = 2, min_separation: int = 4
) -> list[Bounce]:
    """Find bounces as local pixel-y maxima with a vertical-velocity sign flip.

    ``window`` is the half-width (in samples) used to confirm a local maximum, which
    suppresses jitter from noisy detections. A single physical bounce can clear the local-max
    test on two or three adjacent frames (a flat-ish peak), so ``min_separation`` (in frames)
    applies non-maximum suppression: within a cluster of near-frame candidates only the one
    lowest on screen (largest pixel-y) is kept.
    """
    samples = sorted(trajectory, key=lambda s: s.frame)
    if len(samples) < 2 * window + 1:
        return []

    # Collect raw local-maxima candidates as (frame, pixel_y, court_xy).
    candidates: list[tuple[int, float, tuple[float, float]]] = []
    for i in range(window, len(samples) - window):
        py = samples[i].px[1]
        left = samples[i - window].px[1]
        right = samples[i + window].px[1]
        # Local max in pixel-y (lowest on screen) => ground contact, with falling-then-rising.
        if py >= left and py >= right and (py - left) > 0 and (py - right) > 0:
            court_xy = samples[i].court_xy
            if court_xy is None:
                continue
            candidates.append((samples[i].frame, py, court_xy))

    # Non-maximum suppression: collapse clusters within ``min_separation`` frames to their peak.
    bounces: list[Bounce] = []
    peak_py: list[float] = []
    for frame, py, court_xy in candidates:
        if bounces and frame - bounces[-1].frame < min_separation:
            if py <= peak_py[-1]:
                continue  # keep the existing, lower-on-screen detection
            bounces.pop()  # this candidate is the stronger peak of the cluster
            peak_py.pop()
        bounces.append(
            Bounce(
                frame=frame,
                court_xy=court_xy,
                side=Team(side_of(court_xy)),
                in_bounds=is_in_bounds(court_xy, margin=0.03),
            )
        )
        peak_py.append(py)
    return bounces
