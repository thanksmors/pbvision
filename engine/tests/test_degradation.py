"""The pipeline must run one model at a time: unavailable stages skip with a warning.

This is what lets a real run proceed with only the player detector installed — court and ball
raise ModelUnavailable and are skipped rather than crashing the whole match.
"""

from pbengine import fixtures
from pbengine.errors import ModelUnavailable
from pbengine.pipeline import analyze_match


class _Unavailable:
    """A detector whose every entry point reports its model isn't installed."""

    def detect(self, frame):
        raise ModelUnavailable("court weights missing")

    def solve(self, frame):
        raise ModelUnavailable("court weights missing")

    def track(self, *args, **kwargs):
        raise ModelUnavailable("ball weights missing")


def test_players_only_run_degrades_gracefully(tmp_path):
    video = tmp_path / "demo.mp4"
    fixtures.write_synthetic_video(video)
    _, players, _ = fixtures.fixture_detectors()  # real-ish players, unavailable court+ball

    result = analyze_match(
        video,
        court_detector=_Unavailable(),
        player_detector=players,
        ball_tracker=_Unavailable(),
    )

    # Court and ball were skipped; players still came through with bounding boxes.
    assert result.court is None
    assert result.points == []
    assert len(result.players) == 4
    assert all(p.positions[0].bbox_px is not None for p in result.players)

    stages = {w.split(":")[0] for w in result.warnings}
    assert stages == {"court", "ball"}
