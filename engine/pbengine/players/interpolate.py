"""Fill short per-track gaps in player positions so skeletons stay continuous.

The player stage emits a :class:`PlayerPosition` only on frames where the detector+tracker produced
a box, so a brief ByteTrack dropout or occlusion leaves a hole and the skeleton blinks out. This
linearly interpolates the missing frames between two detections **up to a cap** (``max_gap_frames``):
a gap longer than the cap means the player genuinely left the frame, so inventing a pose there would
be wrong and is left alone. Interpolated positions are flagged ``interpolated=True`` (mirrors
``BallSample.interpolated``), so any analysis can exclude them.
"""

from __future__ import annotations

from pbengine.schema.models import PlayerPosition


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _lerp_tuple(a, b, t):
    return tuple(_lerp(a[k], b[k], t) for k in range(len(a)))


def _lerp_points(a, b, t):
    """Per-point lerp for a list of coordinate tuples; None unless both sides are present + aligned."""
    if a is None or b is None or len(a) != len(b):
        return None
    return [tuple(_lerp(pa[k], pb[k], t) for k in range(len(pa))) for pa, pb in zip(a, b)]


def interpolate_positions(
    positions: list[PlayerPosition], fps: float, max_gap_frames: int = 30
) -> list[PlayerPosition]:
    """Return ``positions`` with linearly-interpolated samples filling gaps ``<= max_gap_frames``.

    ``max_gap_frames`` defaults to ~1 s at 30 fps. Longer gaps (real off-court absences) are left as
    holes. Fields are interpolated only when both bounding samples carry them; ``pose_conf`` carries
    the per-keypoint minimum so a low-confidence end keeps the bridged keypoints low-confidence too.
    """
    if len(positions) < 2:
        return list(positions)
    knots = sorted(positions, key=lambda p: p.frame)
    fills: list[PlayerPosition] = []
    for a, c in zip(knots, knots[1:]):
        gap = c.frame - a.frame
        if gap <= 1 or gap > max_gap_frames:
            continue
        conf = None
        if a.pose_conf is not None and c.pose_conf is not None and len(a.pose_conf) == len(c.pose_conf):
            conf = [min(ca, cc) for ca, cc in zip(a.pose_conf, c.pose_conf)]
        for f in range(a.frame + 1, c.frame):
            t = (f - a.frame) / gap
            fills.append(PlayerPosition(
                frame=f,
                court_xy=_lerp_tuple(a.court_xy, c.court_xy, t),
                bbox_px=(_lerp_tuple(a.bbox_px, c.bbox_px, t)
                         if a.bbox_px is not None and c.bbox_px is not None else None),
                pose_px=_lerp_points(a.pose_px, c.pose_px, t),
                pose_conf=conf,
                pose_world_ft=_lerp_points(a.pose_world_ft, c.pose_world_ft, t),
                paddle_px=(_lerp_tuple(a.paddle_px, c.paddle_px, t)
                           if a.paddle_px is not None and c.paddle_px is not None else None),
                paddle_world_ft=(_lerp_tuple(a.paddle_world_ft, c.paddle_world_ft, t)
                                 if a.paddle_world_ft is not None and c.paddle_world_ft is not None
                                 else None),
                interpolated=True,
            ))
    out = knots + fills
    out.sort(key=lambda p: p.frame)
    return out
