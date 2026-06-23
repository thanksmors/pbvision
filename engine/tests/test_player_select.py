"""On-court roster selection: drop background people, keep deep/late players, cap per side."""

from __future__ import annotations

from pbengine.players.select import select_on_court_players
from pbengine.schema.models import Player, PlayerPosition, Team


def _player(track_id: int, team: Team | None, court_xy, n: int, start: int = 0) -> Player:
    return Player(track_id=track_id, team=team,
                  positions=[PlayerPosition(frame=start + f, court_xy=court_xy) for f in range(n)])


def test_drops_off_court_people_and_caps_two_per_side():
    players = [
        _player(1, Team.A, (0.3, 0.3), 100),    # real A
        _player(2, Team.A, (0.7, 0.3), 100),    # real A
        _player(3, Team.A, (0.5, 0.35), 40),    # extra on A (coach) -> capped out (least present)
        _player(4, Team.B, (0.3, 0.7), 100),    # real B
        _player(5, Team.B, (0.7, 0.7), 100),    # real B
        _player(6, Team.A, (5.0, 5.0), 100),    # spectator / adjacent court -> dropped (off-court)
    ]
    kept = select_on_court_players(players)
    assert {p.track_id for p in kept} == {1, 2, 4, 5}


def test_keeps_deep_and_late_real_players():
    # A deep player just behind the baseline (y slightly > 1) and a late entrant (fewer frames) both
    # have on-court medians and enough presence -> kept. Partners split left/right; the deep player is
    # the left of side A, the right is its own player.
    players = [
        _player(1, Team.A, (0.25, 1.06), 200),   # deep behind baseline, A-left
        _player(2, Team.A, (0.75, 0.30), 200),   # A-right
        _player(3, Team.B, (0.30, 0.70), 200),
        _player(4, Team.B, (0.75, 0.72), 60, start=300),  # entered late, B-right
    ]
    assert {p.track_id for p in select_on_court_players(players)} == {1, 2, 3, 4}


def test_merges_temporally_disjoint_fragments_of_one_player():
    # One physical player on each corner, but the left players are each split into two non-overlapping
    # ByteTrack fragments. The roster must MERGE each into one continuous player spanning both windows,
    # so the overlay shows the player across the whole match — not just the longest fragment.
    players = [
        _player(1, Team.A, (0.30, 0.20), 200, start=0),     # A-left, early
        _player(50, Team.A, (0.33, 0.25), 150, start=600),  # A-left, later (same lateral lane)
        _player(2, Team.A, (0.75, 0.20), 300),              # A-right
        _player(3, Team.B, (0.30, 0.80), 300),              # B-left
        _player(4, Team.B, (0.75, 0.80), 300),              # B-right
    ]
    kept = {p.track_id: p for p in select_on_court_players(players)}
    assert set(kept) == {1, 2, 3, 4}                         # 4 players, #50 merged into #1
    aleft = kept[1]
    frames = [pos.frame for pos in aleft.positions]
    assert frames == sorted(frames) and len(frames) == len(set(frames))  # sorted, deduped
    assert min(frames) == 0 and max(frames) == 749           # spans BOTH fragments (0..199, 600..749)


def test_merge_dedupes_overlapping_frames():
    # Two same-lane fragments that overlap in time merge without duplicate frames.
    players = [
        _player(1, Team.A, (0.30, 0.20), 100, start=0),    # frames 0..99
        _player(7, Team.A, (0.32, 0.22), 100, start=50),   # frames 50..149 (overlap 50..99)
        _player(2, Team.B, (0.50, 0.70), 100),
    ]
    kept = {p.track_id: p for p in select_on_court_players(players)}
    frames = [pos.frame for pos in kept[1].positions]
    assert frames == list(range(0, 150))                   # union 0..149, no duplicates


def test_keeps_distinct_partners_over_duplicate_fragments():
    # The "only 3 players" bug: a side's left player is split into two long fragments while the real
    # right partner has a shorter track. The roster must keep one left + the right (the two left
    # fragments merge into one player; the right is its own).
    players = [
        _player(1, Team.A, (0.3, 0.2), 100),     # near-left A
        _player(2, Team.A, (0.7, 0.2), 100),     # near-right A
        _player(10, Team.B, (0.34, 0.80), 900),  # far-left B  (long fragment)
        _player(11, Team.B, (0.39, 0.82), 700),  # far-left B  (another long fragment ~1 ft away)
        _player(12, Team.B, (0.80, 0.85), 400),  # far-right B (the real partner, fewer frames)
    ]
    kept = {p.track_id for p in select_on_court_players(players)}
    assert kept == {1, 2, 10, 12}              # one left (#10, with #11 merged in) + the right
    assert 11 not in kept                       # near-duplicate of #10 -> merged, not a separate id


def test_third_lateral_cluster_on_a_side_is_dropped():
    # Three laterally-distinct in-bounds tracks on side A: only the two most-present survive; a third
    # distinct lane is a stray (it can't be a doubles partner) and is dropped.
    players = [
        _player(1, Team.A, (0.15, 0.2), 300),   # far-left
        _player(2, Team.A, (0.85, 0.2), 250),   # far-right
        _player(3, Team.A, (0.50, 0.2), 100),   # middle lane, distinct from both -> dropped
        _player(4, Team.B, (0.5, 0.8), 100),
    ]
    assert {p.track_id for p in select_on_court_players(players)} == {1, 2, 4}


def test_ignores_blink_tracks():
    players = [
        _player(1, Team.A, (0.5, 0.3), 100),
        _player(2, Team.A, (0.5, 0.3), 5),       # 5-frame blink -> too brief
        _player(3, Team.B, (0.5, 0.7), 100),
    ]
    assert {p.track_id for p in select_on_court_players(players)} == {1, 3}


def test_singles_keeps_one_per_side():
    players = [_player(1, Team.A, (0.5, 0.3), 50), _player(2, Team.B, (0.5, 0.7), 50)]
    assert {p.track_id for p in select_on_court_players(players)} == {1, 2}


def test_no_team_returns_unchanged():
    # No homography -> team None -> sides are meaningless, so nothing is filtered.
    players = [_player(1, None, (9.0, 9.0), 10), _player(2, None, (0.5, 0.5), 10)]
    assert select_on_court_players(players) == players
