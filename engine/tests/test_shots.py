"""Shot (player–ball contact) detection: ordering, attribution, classification, outcome."""

from __future__ import annotations

from pbengine.rally.shots import detect_shots, enrich_shots, shot_contacts
from pbengine.schema.models import (
    BallSample,
    Bounce,
    Player,
    PlayerPosition,
    Serve,
    ShotOutcome,
    ShotType,
    Team,
)

FPS = 30.0


def _player(track_id, team, court_xy, frames, paddle_z=2.5):
    px, py = court_xy[0] * 20.0, court_xy[1] * 44.0
    return Player(track_id=track_id, team=team, positions=[
        PlayerPosition(frame=f, court_xy=court_xy, paddle_world_ft=(px, py, paddle_z))
        for f in frames])


def _rally():
    """Synthesize a back-and-forth rally: ball y oscillates, each reversal is a hit."""
    # y: A baseline (0.15) -> over -> B baseline (0.85) -> back -> A (0.15) ... reversals at the ends.
    ys = []
    f = 0
    waypoints = [0.15, 0.85, 0.20, 0.80]  # 3 reversals after the first leg
    cur = waypoints[0]
    for nxt in waypoints[1:]:
        for k in range(1, 16):
            ys.append((f, cur + (nxt - cur) * k / 15))
            f += 1
        cur = nxt
    measured = [BallSample(frame=fr, px=(960.0, 540.0), court_xy=(0.5, y), conf=1.0) for fr, y in ys]
    return measured


def test_shots_are_ordered_attributed_and_classified():
    measured = _rally()
    frames = [s.frame for s in measured]
    a = _player(1, Team.A, (0.5, 0.12), frames)   # near A baseline
    b = _player(2, Team.B, (0.5, 0.88), frames)   # near B baseline
    bounces = [Bounce(frame=22, court_xy=(0.5, 0.7), side=Team.B, in_bounds=True)]
    serve = Serve(frame=0, server_team=Team.A, server_side="right", confidence=0.8)

    shots = detect_shots(measured, bounces, [a, b], serve, FPS)
    assert len(shots) >= 3
    assert [s.shot_index for s in shots] == list(range(len(shots)))
    # Reversal near B's baseline is attributed to B; near A's baseline to A.
    for s in shots[1:]:
        assert s.team in (Team.A, Team.B)
        assert s.player_track_id in (1, 2)

    # Enrich with a lifted trajectory (give it speed + height) and a winner; types come from here.
    for s in measured:
        s.world_ft = (10.0, s.court_xy[1] * 44.0, 2.0)
        s.speed_mph = 30.0
    enriched = enrich_shots(shots, measured, bounces, Team.A)
    assert enriched[0].shot_type == ShotType.serve
    assert enriched[1].shot_type == ShotType.return_
    assert enriched[-1].outcome in (ShotOutcome.winner, ShotOutcome.error)
    assert enriched[-1].outcome == (ShotOutcome.winner if enriched[-1].team == Team.A
                                    else ShotOutcome.error)
    assert all(s.speed_mph is not None for s in enriched)


def test_shot_contacts_feed_the_height_model():
    measured = _rally()
    shots = detect_shots(measured, [], [_player(1, Team.A, (0.5, 0.12), [s.frame for s in measured])],
                         None, FPS)
    contacts = shot_contacts(shots)
    assert contacts and all(len(c) == 2 for c in contacts)
    assert all(0.5 <= z <= 10.0 for _, z in contacts)


def test_degrades_without_players_or_camera():
    measured = _rally()
    shots = detect_shots(measured, [], [], None, FPS)  # no players -> unattributed, no crash
    assert all(s.player_track_id is None and s.team is None for s in shots)
    # No serve, still returns contact-driven shots in order.
    assert [s.shot_index for s in shots] == list(range(len(shots)))
