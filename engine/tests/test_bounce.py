from pbengine.bounce.heuristic import detect_bounces
from pbengine.schema.models import BallSample


def _sample(frame, py, court_y):
    # x fixed; vary pixel-y (vertical screen motion) and court y.
    return BallSample(frame=frame, px=(500.0, py), court_xy=(0.5, court_y), conf=0.9)


def test_detects_single_bounce():
    # Pixel-y rises (ball falls) to a peak at frame 3, then drops (ball rises) = one bounce.
    traj = [
        _sample(1, 200, 0.3),
        _sample(2, 350, 0.4),
        _sample(3, 500, 0.5),  # lowest on screen -> bounce
        _sample(4, 360, 0.6),
        _sample(5, 210, 0.7),
    ]
    bounces = detect_bounces(traj, window=2)
    assert len(bounces) == 1
    assert bounces[0].frame == 3
    assert bounces[0].in_bounds is True


def test_no_bounce_on_monotonic():
    traj = [_sample(i, 100 + i * 10, 0.3 + i * 0.05) for i in range(6)]
    assert detect_bounces(traj, window=2) == []


def test_skips_bounce_at_nulled_court_xy():
    # The pixel-y local max sits on a sample whose court_xy was discarded as an off-court outlier
    # (airborne/vanishing-line projection). Without a court position it must not become a bounce.
    traj = [
        _sample(1, 200, 0.3),
        _sample(2, 350, 0.4),
        BallSample(frame=3, px=(500.0, 500.0), court_xy=None, conf=0.9),  # peak, but no court_xy
        _sample(4, 360, 0.6),
        _sample(5, 210, 0.7),
    ]
    assert detect_bounces(traj, window=2) == []
