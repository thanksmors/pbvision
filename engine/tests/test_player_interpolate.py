"""Per-track player position interpolation: bridge short detection gaps, leave long absences."""

from __future__ import annotations

from pbengine.players.interpolate import interpolate_positions
from pbengine.schema.models import PlayerPosition


def _pos(frame, x, y, *, pose=True):
    return PlayerPosition(
        frame=frame,
        court_xy=(x, y),
        bbox_px=(x, y, x + 40, y + 100),
        pose_px=[(x + k, y + k) for k in range(17)] if pose else None,
        pose_conf=[0.9] * 17 if pose else None,
        pose_world_ft=[(x, y, float(k)) for k in range(17)] if pose else None,
    )


def test_short_gap_is_interpolated_and_flagged():
    positions = [_pos(0, 100.0, 200.0), _pos(10, 200.0, 240.0)]
    out = interpolate_positions(positions, fps=30.0, max_gap_frames=30)
    assert [p.frame for p in out] == list(range(11))  # gap fully bridged
    fills = [p for p in out if p.interpolated]
    assert {p.frame for p in fills} == set(range(1, 10))
    mid = next(p for p in out if p.frame == 5)
    assert mid.interpolated
    assert abs(mid.court_xy[0] - 150.0) < 1e-6           # halfway between 100 and 200
    assert mid.pose_px is not None and len(mid.pose_px) == 17
    assert mid.pose_world_ft is not None and abs(mid.pose_world_ft[0][0] - 150.0) < 1e-6
    # endpoints are untouched measurements
    assert not out[0].interpolated and not out[-1].interpolated


def test_long_gap_is_left_as_a_hole():
    positions = [_pos(0, 100.0, 200.0), _pos(100, 300.0, 260.0)]  # ~3.3 s gap at 30 fps
    out = interpolate_positions(positions, fps=30.0, max_gap_frames=30)
    assert [p.frame for p in out] == [0, 100]  # real absence: not invented
    assert not any(p.interpolated for p in out)


def test_missing_pose_fields_stay_none():
    positions = [_pos(0, 100.0, 200.0, pose=False), _pos(6, 160.0, 220.0, pose=False)]
    out = interpolate_positions(positions, fps=30.0)
    mid = next(p for p in out if p.frame == 3)
    assert mid.interpolated
    assert mid.pose_px is None and mid.pose_world_ft is None
    assert abs(mid.court_xy[0] - 130.0) < 1e-6  # court_xy still interpolated
