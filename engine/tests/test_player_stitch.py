"""Track stitching: merge ByteTrack fragments of one player, keep distinct players apart."""

from __future__ import annotations

from pbengine.players.stitch import max_stitch_frames, stitch_tracks

FPS = 30.0


def _track(start, end, x, y, step=0.0):
    """Frames [start, end] walking from (x, y) along +y at `step` per frame (normalized court)."""
    return [{"frame": f, "court_xy": (x, y + step * (f - start))} for f in range(start, end + 1)]


def _track_x(start, end, x0, y, xstep):
    """Frames [start, end] walking from (x0, y) along x at `xstep` per frame (normalized court)."""
    return [{"frame": f, "court_xy": (x0 + xstep * (f - start), y)} for f in range(start, end + 1)]


def _groups(raw, votes):
    return [sorted(g) for g in stitch_tracks(raw, votes, FPS)]


def test_merges_continuous_fragments():
    # One player walking +y; ByteTrack drops them for ~10 frames mid-walk, restarting as id 2 right
    # where 1 left off (same side, small gap, continuous position).
    raw = {1: _track(0, 20, 0.5, 0.20, step=0.004),
           2: _track(31, 50, 0.5, 0.244, step=0.004)}
    votes = {1: ["A"] * 21, 2: ["A"] * 20}
    assert _groups(raw, votes) == [[1, 2]]


def test_keeps_far_apart_players_separate():
    # Same side and time, but the second fragment is across the court — different player.
    raw = {1: _track(0, 20, 0.2, 0.20), 2: _track(31, 50, 0.9, 0.30)}
    votes = {1: ["A"] * 21, 2: ["A"] * 20}
    assert _groups(raw, votes) == [[1], [2]]


def test_keeps_opposite_sides_separate():
    raw = {1: _track(0, 20, 0.5, 0.20), 2: _track(31, 50, 0.5, 0.80)}
    votes = {1: ["A"] * 21, 2: ["B"] * 20}
    assert _groups(raw, votes) == [[1], [2]]


def test_does_not_bridge_too_long_a_gap():
    # Continuous in space but the gap exceeds the stitch window -> a genuine new appearance.
    gap = max_stitch_frames(FPS) + 10
    raw = {1: _track(0, 20, 0.5, 0.20), 2: _track(21 + gap, 40 + gap, 0.5, 0.20)}
    votes = {1: ["A"] * 21, 2: ["A"] * 20}
    assert _groups(raw, votes) == [[1], [2]]


def test_ambiguous_match_left_unmerged():
    # Two open fragments end at the same place and time; a new fragment could continue either ->
    # ambiguous (within MARGIN), so it is not merged into either.
    raw = {1: _track(0, 20, 0.5, 0.30), 2: _track(0, 20, 0.5, 0.30),
           3: _track(31, 50, 0.5, 0.30)}
    votes = {1: ["A"] * 21, 2: ["A"] * 21, 3: ["A"] * 20}
    groups = _groups(raw, votes)
    assert [3] in groups  # id 3 stands alone rather than teleporting onto 1 or 2
    assert len(groups) == 3


def test_does_not_merge_overlapping_tracks():
    # Two simultaneous players (overlapping frames) are never merged.
    raw = {1: _track(0, 30, 0.4, 0.30), 2: _track(0, 30, 0.6, 0.30)}
    votes = {1: ["A"] * 31, 2: ["A"] * 31}
    assert _groups(raw, votes) == [[1], [2]]


def test_velocity_cannot_fabricate_cross_court_merge():
    # The "chimera" bug: track 1 ends far-right (x~0.83) moving left fast; track 2 starts far-left
    # (x~0.39). Velocity extrapolation alone would pull the prediction near track 2 and merge them,
    # but the *raw* last->first distance (~8.8 ft) exceeds the match radius -> they stay two players.
    raw = {1: _track_x(0, 20, 1.05, 0.89, -0.011),   # ends ~(0.83, 0.89), vel ~ -0.22 ft/frame in x
           2: _track(31, 50, 0.39, 0.91)}
    votes = {1: ["B"] * 21, 2: ["B"] * 20}
    assert _groups(raw, votes) == [[1], [2]]
