"""The fixture pipeline must produce a valid, varied MatchResult with no ML installed.

This guards the end-to-end demo path: scripted detectors -> segmentation -> serve/bounce/
winner -> JSON contract. It deliberately asserts the three distinct win-reason branches so a
regression in any heuristic (or the bounce NMS) is caught.
"""

from pbengine import fixtures
from pbengine.pipeline import analyze_match
from pbengine.schema.models import MatchResult, Team, WinReason


def test_scripted_ball_segments_three_points(tmp_path):
    video = tmp_path / "demo.mp4"
    fixtures.write_synthetic_video(video)
    court, players, ball = fixtures.fixture_detectors()

    result = analyze_match(
        video,
        court_detector=court,
        player_detector=players,
        ball_tracker=ball,
    )

    assert isinstance(result, MatchResult)
    assert len(result.points) == 3
    assert len(result.players) == 4
    assert result.court is not None and len(result.court.keypoints_px) == 6

    reasons = [p.win_reason for p in result.points]
    assert reasons == [WinReason.ball_out, WinReason.double_bounce, WinReason.net]
    winners = [p.winner_team for p in result.points]
    assert winners == [Team.B, Team.A, Team.B]

    # Every point carries a serve, a trajectory, and at least one detected bounce.
    for p in result.points:
        assert p.serve is not None
        assert p.ball_trajectory
        assert len(p.bounces) >= 1


def test_synthetic_video_is_probeable(tmp_path):
    video = tmp_path / "demo.mp4"
    meta = fixtures.write_synthetic_video(video)
    assert video.exists() and video.stat().st_size > 0
    assert meta.width == fixtures.WIDTH and meta.height == fixtures.HEIGHT
