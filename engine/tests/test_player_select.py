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
    # have on-court medians and enough presence -> kept (the old <50%-in-bounds rule cut these).
    players = [
        _player(1, Team.A, (0.5, 0.30), 200),
        _player(2, Team.A, (0.5, 1.06), 200),    # deep behind baseline
        _player(3, Team.B, (0.5, 0.70), 200),
        _player(4, Team.B, (0.5, 0.72), 60, start=300),  # entered late
    ]
    assert {p.track_id for p in select_on_court_players(players)} == {1, 2, 3, 4}


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
