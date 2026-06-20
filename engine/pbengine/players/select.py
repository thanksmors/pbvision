"""Keep only the on-court roster, dropping background people (refs, spectators, next court).

ByteTrack returns *everyone* it sees, so a raw match carries far more than the players — anyone
whose feet project (via the court homography) well outside ``[0,1]^2`` is not on this court. We keep
players that spend most of their time in bounds, then cap each side to the ``max_per_side`` most
present (2 for doubles, which naturally yields 1 per side for singles).
"""

from __future__ import annotations

from pbengine.court.court_model import is_in_bounds
from pbengine.schema.models import Player

# A few feet of slack so a player standing behind the baseline / outside the sideline still counts.
_MARGIN = 0.12
# Drop a track that is in bounds for less than this fraction of its life (clearly off-court person).
_MIN_INBOUNDS_FRAC = 0.5


def _inbounds_count(player: Player) -> int:
    return sum(1 for p in player.positions if is_in_bounds(p.court_xy, _MARGIN))


def select_on_court_players(players: list[Player], max_per_side: int = 2) -> list[Player]:
    """Return the on-court roster: in-bounds players, at most ``max_per_side`` per side.

    Scoring is by in-bounds presence (frame count), so a mostly-off-court person loses a tie to a
    real player. Players without a ``team`` (no homography ⇒ sides are meaningless) are returned
    unchanged — there's nothing reliable to filter on.
    """
    if any(p.team is None for p in players):
        return players

    kept: list[Player] = []
    for p in players:
        n = len(p.positions)
        if n and _inbounds_count(p) / n >= _MIN_INBOUNDS_FRAC:
            kept.append(p)

    roster: list[Player] = []
    for team in {p.team for p in kept}:
        side = [p for p in kept if p.team == team]
        side.sort(key=_inbounds_count, reverse=True)
        roster.extend(side[:max_per_side])
    roster.sort(key=lambda p: p.track_id)
    return roster
