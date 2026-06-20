"""On-court roster selection: drop background people, cap each side to the doubles count."""

from __future__ import annotations

from pbengine.players.select import select_on_court_players
from pbengine.schema.models import Player, PlayerPosition, Team


def _player(track_id: int, team: Team | None, court_xy, n: int) -> Player:
    return Player(track_id=track_id, team=team,
                  positions=[PlayerPosition(frame=f, court_xy=court_xy) for f in range(n)])


def test_drops_off_court_people_and_caps_two_per_side():
    players = [
        _player(1, Team.A, (0.3, 0.3), 100),   # real A
        _player(2, Team.A, (0.7, 0.3), 100),   # real A
        _player(3, Team.A, (0.5, 0.35), 40),   # extra on A (e.g. a coach) -> capped out (least present)
        _player(4, Team.B, (0.3, 0.7), 100),   # real B
        _player(5, Team.B, (0.7, 0.7), 100),   # real B
        _player(6, Team.A, (5.0, 5.0), 100),   # spectator / adjacent court, way off-court -> dropped
    ]
    kept = select_on_court_players(players)
    assert {p.track_id for p in kept} == {1, 2, 4, 5}


def test_singles_keeps_one_per_side():
    players = [_player(1, Team.A, (0.5, 0.3), 50), _player(2, Team.B, (0.5, 0.7), 50)]
    assert {p.track_id for p in select_on_court_players(players)} == {1, 2}


def test_no_team_returns_unchanged():
    # No homography -> team None -> sides are meaningless, so nothing is filtered.
    players = [_player(1, None, (9.0, 9.0), 10), _player(2, None, (0.5, 0.5), 10)]
    assert select_on_court_players(players) == players
