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


def detect_bounces(trajectory: list[BallSample], window: int = 2) -> list[Bounce]:
    """Find bounces as local pixel-y maxima with a vertical-velocity sign flip.

    ``window`` is the half-width (in samples) used to confirm a local maximum, which
    suppresses jitter from noisy detections.
    """
    samples = sorted(trajectory, key=lambda s: s.frame)
    if len(samples) < 2 * window + 1:
        return []

    bounces: list[Bounce] = []
    for i in range(window, len(samples) - window):
        py = samples[i].px[1]
        left = samples[i - window].px[1]
        right = samples[i + window].px[1]
        # Local max in pixel-y (lowest on screen) => ground contact, with falling-then-rising.
        if py >= left and py >= right and (py - left) > 0 and (py - right) > 0:
            court_xy = samples[i].court_xy
            if court_xy is None:
                continue
            bounces.append(
                Bounce(
                    frame=samples[i].frame,
                    court_xy=court_xy,
                    side=Team(side_of(court_xy)),
                    in_bounds=is_in_bounds(court_xy, margin=0.03),
                )
            )
    return bounces
