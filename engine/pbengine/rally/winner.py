"""Point-winner heuristic from the ball trajectory and detected bounces.

This is the second-riskiest stage after ball tracking: it compounds ball-tracking,
bounce-detection and homography error. We therefore return a confidence and let the
pipeline flag low-confidence calls with ``needs_review`` rather than pretending to be a
perfect umpire. Kitchen-volley faults are out of scope for v1.

Rules, in priority order, all reasoned in normalized court coords (net at y = 0.5):

1. **Double bounce** — two consecutive in-bounds bounces on the same side with no contact
   between. That side failed to return; the other side wins.
2. **Ball out** — the final bounce lands out of bounds. The *hitter* (on the opposite side
   from the landing region) erred, so the side the ball landed toward wins.
3. **Net** — the ball dies near the net plane without crossing. The side that hit it into the
   net loses; the receiving side wins. Direction is read from the trailing trajectory.
4. **Unknown** — none of the above fire; low confidence, flagged for review.
"""

from __future__ import annotations

from pbengine.court.court_model import side_of
from pbengine.schema.models import BallSample, Bounce, Team, WinReason

NET_Y = 0.5
_NET_BAND = 0.08  # how close to the net plane counts as "at the net"


def _other(team: Team) -> Team:
    return Team.B if team is Team.A else Team.A


def determine_winner(
    bounces: list[Bounce],
    trajectory: list[BallSample],
) -> tuple[Team | None, WinReason, float]:
    """Return ``(winner_team, win_reason, confidence)`` for a finished rally."""
    # Rule 1: double bounce on one side.
    in_bounds = [b for b in bounces if b.in_bounds]
    if len(in_bounds) >= 2 and in_bounds[-1].side == in_bounds[-2].side:
        loser = in_bounds[-1].side
        return _other(loser), WinReason.double_bounce, 0.7

    # Rule 2: last bounce out of bounds.
    if bounces and not bounces[-1].in_bounds:
        # Landing region side == winner (the hitter on the far side put it out).
        winner = Team(side_of(bounces[-1].court_xy))
        return winner, WinReason.ball_out, 0.6

    # Rule 3: ball dies at the net.
    terminal = next(
        (s.court_xy for s in reversed(trajectory) if s.court_xy is not None), None
    )
    if terminal is not None and abs(terminal[1] - NET_Y) <= _NET_BAND:
        direction = _trajectory_direction(trajectory)
        if direction > 0:  # moving A -> B, A hit into the net
            return Team.B, WinReason.net, 0.55
        if direction < 0:  # moving B -> A, B hit into the net
            return Team.A, WinReason.net, 0.55

    # Rule 4: undetermined.
    return None, WinReason.unknown, 0.3


def _trajectory_direction(trajectory: list[BallSample]) -> float:
    """Sign of net-crossing direction from the trailing samples: +y is A->B, -y is B->A."""
    ys = [s.court_xy[1] for s in trajectory if s.court_xy is not None]
    if len(ys) < 2:
        return 0.0
    return ys[-1] - ys[max(0, len(ys) - 5)]
