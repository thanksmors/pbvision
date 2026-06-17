import numpy as np

from pbengine.ball.kalman import gate_jumps, smooth


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
