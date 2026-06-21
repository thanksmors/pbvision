"""Ball tracker: the pure pieces (dotdict, postprocess) without torch/weights.

The full WASB model path is validated separately on CPU (it needs the ml extra + the
submodule + weights, none of which are present in CI).
"""

import numpy as np
import pytest

from pbengine.ball.tracker import BallTracker, _court_xy_or_none
from pbengine.ball.wasb import _DotDict, _wrap


def test_dotdict_supports_attr_and_item_access():
    # HRNet reads both cfg.MODEL.EXTRA and cfg['frames_in'].
    cfg = _wrap({"frames_in": 3, "MODEL": {"EXTRA": {"STEM": {"STRIDES": [1, 1]}}}})
    assert cfg["frames_in"] == 3
    assert cfg.MODEL.EXTRA.STEM.STRIDES == [1, 1]
    assert isinstance(cfg.MODEL, _DotDict)


def test_postprocess_gates_and_projects():
    bt = BallTracker(max_px_per_frame=150.0)
    # A clean diagonal track plus one impossible jump that gating should drop.
    raw = [(0, 100.0, 100.0, 0.9), (1, 110.0, 108.0, 0.8),
           (2, 5000.0, 5000.0, 0.7),  # teleport -> gated out
           (3, 130.0, 124.0, 0.85)]
    samples = bt.postprocess(raw, homography=None)
    frames = [s.frame for s in samples]
    assert 2 not in frames                      # impossible jump removed
    assert all(0.0 <= s.conf <= 1.0 for s in samples)
    assert all(s.court_xy is None for s in samples)  # no homography -> pixel-only


def test_postprocess_empty():
    assert BallTracker().postprocess([], None) == []


def test_court_xy_or_none_rejects_far_off_court_projections():
    # On court and just-out (a few ft past the lines) are kept; a far projection (8.0*20=160 ft) is
    # discarded — that's the airborne/vanishing-line explosion that produced court_xy ~7766.
    assert _court_xy_or_none(0.5, 0.5) == (0.5, 0.5)
    assert _court_xy_or_none(1.1, -0.05) == (1.1, -0.05)   # ~2 ft out / ~2 ft behind -> kept
    assert _court_xy_or_none(8.0, 0.5) is None             # 160 ft off court -> dropped
    assert _court_xy_or_none(0.5, -1.08) is None           # ~48 ft behind baseline -> dropped


def test_postprocess_nulls_off_court_ball_projection():
    # Homography maps pixel -> normalized court (px/1000). Use stable position clusters so the Kalman
    # smoother is a no-op and the projection is deterministic. An on-court cluster (x=500 -> 0.5) keeps
    # court_xy; an off-court cluster (x=9000 -> 9.0 -> 180 ft) has court_xy discarded, pixel preserved.
    H = np.array([[1 / 1000, 0, 0], [0, 1 / 1000, 0], [0, 0, 1]], dtype=float)
    bt = BallTracker(max_px_per_frame=1e9)  # disable the jump gate for this projection test

    on_court = bt.postprocess([(0, 500.0, 400.0, 0.9), (1, 500.0, 400.0, 0.9)], homography=H)
    assert all(s.court_xy is not None for s in on_court)
    assert on_court[0].court_xy[0] == pytest.approx(0.5, abs=0.05)

    off_court = bt.postprocess([(0, 9000.0, 400.0, 0.9), (1, 9000.0, 400.0, 0.9)], homography=H)
    assert all(s.court_xy is None for s in off_court)       # off-court projection -> discarded
    assert all(s.px[0] > 1000 for s in off_court)            # pixels still present (coverage preserved)


def test_fast_ball_survives_resolution_aware_gate():
    """A genuinely fast ball (~500 px/frame at 1080p) must be kept by the auto gate, but a
    full-frame teleport must still be dropped. The old fixed 150 px/frame would reject the fast ball.
    """
    # ~400 px/frame across the frame — a fast pickleball at 1080p/30fps (well over the old 150).
    fast = [(i, 100.0 + 400 * i, 500.0, 0.9) for i in range(5)]  # x: 100..1700
    teleport = (5, 100.0, 1050.0, 0.6)  # ~1690 px from the last fast point -> a true teleport

    bt = BallTracker(max_jump_frac=0.5)  # auto gate for a 1920-wide frame ~= 960 px/frame
    auto_gate = bt._gate_px(1920, 1080)
    samples = bt.postprocess(fast + [teleport], homography=None, max_px_per_frame=auto_gate)
    assert [s.frame for s in samples] == [0, 1, 2, 3, 4]  # fast ball kept, teleport rejected

    # The old fixed 150 px/frame would have discarded the fast ball after the first point.
    old = bt.postprocess(fast, homography=None, max_px_per_frame=150.0)
    assert [s.frame for s in old] == [0]
