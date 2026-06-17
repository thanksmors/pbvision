from pbengine.schema import MatchResult, Point, VideoMeta
from pbengine.schema.models import (
    BallSample,
    Bounce,
    CourtModel,
    Serve,
    Team,
    WinReason,
)


def test_match_result_roundtrip():
    result = MatchResult(
        match_id="abc",
        video=VideoMeta(fps=30.0, frames=900, width=1920, height=1080),
        court=CourtModel(homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        points=[
            Point(
                point_index=0,
                start_frame=10,
                end_frame=120,
                serve=Serve(frame=10, server_team=Team.A, server_side="right", confidence=0.7),
                bounces=[Bounce(frame=40, court_xy=(0.5, 0.7), side=Team.B, in_bounds=True)],
                ball_trajectory=[BallSample(frame=10, px=(960, 540), court_xy=(0.5, 0.1), conf=0.8)],
                winner_team=Team.A,
                win_reason=WinReason.double_bounce,
                confidence=0.7,
                needs_review=False,
            )
        ],
    )
    dumped = result.model_dump_json()
    back = MatchResult.model_validate_json(dumped)
    assert back.points[0].winner_team is Team.A
    assert back.points[0].win_reason is WinReason.double_bounce
    assert back.video.fps == 30.0


def test_confidence_bounds_enforced():
    try:
        Serve(frame=1, server_team=Team.A, server_side="left", confidence=1.5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("confidence > 1 should fail validation")
