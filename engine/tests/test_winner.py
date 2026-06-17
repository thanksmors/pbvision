from pbengine.rally.winner import determine_winner
from pbengine.schema.models import BallSample, Bounce, Team, WinReason


def _bounce(frame, xy, side, in_bounds):
    return Bounce(frame=frame, court_xy=xy, side=Team(side), in_bounds=in_bounds)


def test_double_bounce_loses():
    # Two in-bounds bounces on side B -> B failed to return -> A wins.
    bounces = [
        _bounce(10, (0.5, 0.3), "A", True),
        _bounce(20, (0.5, 0.7), "B", True),
        _bounce(28, (0.5, 0.75), "B", True),
    ]
    winner, reason, conf = determine_winner(bounces, [])
    assert winner is Team.A
    assert reason is WinReason.double_bounce
    assert conf >= 0.6


def test_ball_out_winner_is_landing_side():
    # Last bounce lands out beyond side B's baseline -> A hit it out -> B wins.
    bounces = [
        _bounce(10, (0.5, 0.3), "A", True),
        _bounce(22, (0.5, 1.08), "B", False),
    ]
    winner, reason, _ = determine_winner(bounces, [])
    assert winner is Team.B
    assert reason is WinReason.ball_out


def test_net_fault_direction():
    # Ball travelling A->B dies at the net -> A hit into the net -> B wins.
    traj = [
        BallSample(frame=1, px=(0, 0), court_xy=(0.5, 0.2), conf=0.9),
        BallSample(frame=2, px=(0, 0), court_xy=(0.5, 0.35), conf=0.9),
        BallSample(frame=3, px=(0, 0), court_xy=(0.5, 0.48), conf=0.9),
    ]
    winner, reason, _ = determine_winner([], traj)
    assert winner is Team.B
    assert reason is WinReason.net


def test_unknown_when_no_signal():
    winner, reason, conf = determine_winner([], [])
    assert winner is None
    assert reason is WinReason.unknown
    assert conf < 0.6
