"""Ball tracker: the pure pieces (dotdict, postprocess) without torch/weights.

The full WASB model path is validated separately on CPU (it needs the ml extra + the
submodule + weights, none of which are present in CI).
"""

import numpy as np
import pytest

from pbengine.ball.size import depth_from_radius
from pbengine.ball.tracker import BallTracker, _court_xy_or_none, _motion_peak
from pbengine.ball.wasb import _blob_radius_px, _DotDict, _wrap


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


def test_depth_from_radius_geometry():
    # depth = focal_px * (diameter/2) / radius_px
    assert depth_from_radius(5.0, 1000.0, 0.243) == pytest.approx(1000.0 * 0.1215 / 5.0)
    # a smaller (farther) ball implies a larger depth
    assert depth_from_radius(2.0, 1000.0) > depth_from_radius(8.0, 1000.0)
    # degenerate inputs carry no depth
    assert depth_from_radius(0.0, 1000.0) is None
    assert depth_from_radius(5.0, 0.0) is None
    assert depth_from_radius(None, 1000.0) is None


def test_blob_radius_from_synthetic_heatmap():
    # A gaussian blob (sigma=2) centred at (x=12, y=9); identity heatmap->source affine.
    h = w = 24
    yy, xx = np.mgrid[0:h, 0:w]
    hm = np.exp(-(((xx - 12) ** 2 + (yy - 9) ** 2) / (2 * 2.0 ** 2))).astype(np.float32)
    trans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = _blob_radius_px(hm, (12, 9), trans, 0.3)
    assert r is not None and 0.5 < r < 8.0          # ~the blob's spread, in (here equal) source px
    assert _blob_radius_px(hm, (1, 1), trans, 0.3) is None      # off the blob (below threshold)
    assert _blob_radius_px(hm, (100, 100), trans, 0.3) is None  # out of heatmap bounds


def test_postprocess_carries_radius_px():
    bt = BallTracker(max_px_per_frame=1e9)  # disable gating; stable positions so smoothing is a no-op
    with_r = bt.postprocess([(0, 100.0, 100.0, 0.9, 4.0), (1, 100.0, 100.0, 0.9, 3.0)], homography=None)
    assert [round(s.radius_px, 1) for s in with_r] == [4.0, 3.0]
    # Legacy 4-tuple detections (no radius) round-trip to radius_px=None.
    without_r = bt.postprocess([(0, 100.0, 100.0, 0.9), (1, 100.0, 100.0, 0.9)], homography=None)
    assert all(s.radius_px is None for s in without_r)


def test_motion_peak_localizes_a_moving_blob():
    # A bright ball at (60,50) in cur that wasn't there in prev -> the frame-diff peak near the
    # predicted centre localizes it; a flat (no-motion) pair yields None.
    prev = np.zeros((96, 128), dtype=np.uint8)
    cur = np.zeros((96, 128), dtype=np.uint8)
    cur[48:53, 58:63] = 255  # a small bright blob ~ (60, 50)
    peak = _motion_peak(prev, cur, center=(60, 50), half=40, min_score=18.0)
    assert peak is not None
    x, y, score = peak
    assert abs(x - 60) <= 3 and abs(y - 50) <= 3 and score >= 18.0
    # No motion -> no peak.
    assert _motion_peak(prev, prev, center=(60, 50), half=40, min_score=18.0) is None
    # Motion outside the search window is not picked up.
    assert _motion_peak(prev, cur, center=(10, 10), half=8, min_score=18.0) is None


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
