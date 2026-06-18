import numpy as np

from pbengine.ball.kalman import clean_track_2d, fill_gaps_2d, gate_jumps, smooth


def test_gate_rejects_teleport():
    frames = np.array([0, 1, 2, 3, 4])
    xy = np.array([[0, 0], [10, 0], [1000, 0], [30, 0], [40, 0]], dtype=float)
    keep = gate_jumps(frames, xy, max_px_per_frame=100.0)
    # The teleport at index 2 is dropped; the rest survive.
    assert keep.tolist() == [True, True, False, True, True]


def test_smooth_tracks_linear_motion():
    frames = np.arange(20)
    truth = np.stack([frames * 5.0, frames * 2.0], axis=1)
    rng = np.random.default_rng(1)
    noisy = truth + rng.normal(0, 3, truth.shape)
    out = smooth(frames, noisy)
    # Smoothed track should be closer to truth than the raw noisy input on average.
    assert np.mean(np.abs(out - truth)) < np.mean(np.abs(noisy - truth))


def test_smooth_handles_gaps():
    frames = np.array([0, 1, 2, 10, 11])
    xy = np.array([[0, 0], [5, 2], [10, 4], [50, 20], [55, 22]], dtype=float)
    out = smooth(frames, xy)
    assert out.shape == (5, 2)
    assert np.all(np.isfinite(out))


def test_ransac_poly_rejects_scattered_fp():
    """clean_track_2d keeps a smooth pixel arc and drops scattered false positives the gate misses."""
    frames = np.arange(18)
    t = frames.astype(float)
    # A gentle pixel parabola (a projected arc), plus tiny noise.
    xy = np.stack([400 + 12 * t, 600 + 20 * t - 1.2 * t * t], axis=1)
    rng = np.random.default_rng(0)
    xy += rng.normal(0, 1.0, xy.shape)
    # Three false positives that are slow enough to survive gate_jumps but off the arc.
    for f, off in ((4, (60, -55)), (9, (-50, 65)), (13, (58, 50))):
        xy[f] += np.array(off)

    keep = clean_track_2d(frames, xy, max_gap=6)
    assert not keep[4] and not keep[9] and not keep[13]  # outliers rejected
    assert keep.sum() >= 13  # the true arc is retained


def test_fill_gaps_2d_fills_short_and_skips_long():
    frames = np.array([0, 2, 10])  # a 2-frame gap (fillable) then an 8-frame gap (too wide)
    xy = np.array([[0, 0], [2, 2], [10, 10]], dtype=float)
    out_f, out_xy, mask = fill_gaps_2d(frames, xy, max_fill_gap=4)
    assert out_f.tolist() == [0, 1, 2, 10]
    assert mask.tolist() == [False, True, False, False]  # only frame 1 is interpolated
    assert np.allclose(out_xy[1], [1.0, 1.0])  # linear midpoint
