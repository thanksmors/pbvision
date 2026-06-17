"""Serve detection within a rally (heuristic, no pose required for v1).

A rally starts with a serve. Without pose we approximate: the serve frame is the first
reliable ball sample of the rally, and the serving team/side is read from where that ball
originates in normalized court coordinates — which half (A/B) and which service court
(left/right). Pose (RTMPose) can later sharpen "underhand contact below the waist".
"""

from __future__ import annotations

from pbengine.court.court_model import service_side, side_of
from pbengine.schema.models import BallSample, Serve, Team


def detect_serve(trajectory: list[BallSample]) -> Serve | None:
    """Infer the serve from a rally's ball trajectory.

    Returns ``None`` if no sample carries court coordinates (homography unavailable).
    Confidence is modest by construction — this is a temporal/geometric heuristic.
    """
    first = next((s for s in trajectory if s.court_xy is not None), None)
    if first is None:
        return None

    origin = first.court_xy
    team = Team(side_of(origin))
    side = service_side(origin)

    # Confidence: higher when the serve clearly originates near a baseline (y close to 0 or 1)
    # rather than mid-court, which would suggest we missed the true rally start.
    depth = origin[1] if team is Team.A else 1.0 - origin[1]
    confidence = max(0.3, min(0.9, 1.0 - depth * 2.0)) * first.conf if first.conf else 0.5

    return Serve(
        frame=first.frame,
        server_team=team,
        server_side=side,
        confidence=round(confidence, 3),
    )
