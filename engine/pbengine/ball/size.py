"""Apparent ball size as a depth cue.

A ball of known real diameter projects to a pixel radius that shrinks with distance from the camera:
for a pinhole camera ``r_px ≈ focal_px · R_real / depth``, so ``depth ≈ focal_px · R_real / r_px``.
This is an estimate of distance *along the optical axis* that is independent of the ground-plane
homography — useful exactly where the flat-ground projection of an airborne ball is meaningless.

Pure geometry only (no detector/heatmap here): the radius is extracted in :mod:`pbengine.ball.wasb`
and carried on :class:`~pbengine.schema.models.BallSample.radius_px`. Whether the signal is clean
enough to feed the 3D reconstruction is measured first (see :mod:`pbengine.ball.diag`); the blend into
``trajectory3d.ball_world_ft`` is a follow-up gated on that evidence.
"""

from __future__ import annotations

# Regulation pickleball: 2.87–2.97 in diameter -> ~2.92 in ≈ 0.243 ft. Radius ≈ 0.122 ft.
PICKLEBALL_DIAMETER_FT = 0.243


def depth_from_radius(radius_px: float, focal_px: float,
                      diameter_ft: float = PICKLEBALL_DIAMETER_FT) -> float | None:
    """Distance (ft) from the camera to a ball of ``diameter_ft`` imaged at ``radius_px``.

    ``depth = focal_px · (diameter_ft / 2) / radius_px``. Returns None for non-positive inputs
    (a zero/negative radius or focal carries no depth information).
    """
    if radius_px is None or focal_px is None or radius_px <= 0.0 or focal_px <= 0.0:
        return None
    return focal_px * (diameter_ft / 2.0) / radius_px
