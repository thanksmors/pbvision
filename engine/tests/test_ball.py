"""Ball tracker: the pure pieces (dotdict, postprocess) without torch/weights.

The full WASB model path is validated separately on CPU (it needs the ml extra + the
submodule + weights, none of which are present in CI).
"""

import numpy as np

from pbengine.ball.tracker import BallTracker
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
