"""Physics gating + constant-velocity Kalman smoothing for the ball trajectory.

Heatmap ball detectors emit at most one candidate per frame and produce false positives on
bright shoes, line paint and reflections. Two cheap, dependency-light defenses:

* :func:`gate_jumps` drops detections that imply an impossible frame-to-frame speed
  (scene cuts, far-away false positives).
* :func:`smooth` runs a constant-velocity Kalman filter over the surviving detections to
  denoise positions and bridge tiny gaps.

Pure NumPy so it is unit-testable without the ``ml`` extra. (``filterpy`` is available in the
``ml`` extra if a richer model is wanted later.)
"""

from __future__ import annotations

import numpy as np


def gate_jumps(
    frames: np.ndarray, xy: np.ndarray, max_px_per_frame: float = 150.0
) -> np.ndarray:
    """Return a boolean mask keeping detections whose speed from the last kept point is sane.

    ``frames`` (N,) and ``xy`` (N, 2) must be sorted by frame.
    """
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    keep = np.ones(len(frames), dtype=bool)
    last = 0
    for i in range(1, len(frames)):
        dt = max(1, int(frames[i] - frames[last]))
        dist = float(np.hypot(*(xy[i] - xy[last])))
        if dist / dt > max_px_per_frame:
            keep[i] = False
        else:
            last = i
    return keep


def smooth(frames: np.ndarray, xy: np.ndarray, process_var: float = 1.0,
           meas_var: float = 9.0) -> np.ndarray:
    """Constant-velocity Kalman smoothing of a 2-D point track.

    State is ``[x, y, vx, vy]``. ``frames`` (N,) may be non-contiguous; the time step is the
    frame gap. Returns the filtered positions, shape ``(N, 2)``.
    """
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    if len(frames) == 0:
        return xy.reshape(0, 2)

    state = np.array([xy[0, 0], xy[0, 1], 0.0, 0.0])
    cov = np.eye(4) * 1e3
    meas_h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    meas_r = np.eye(2) * meas_var
    out = np.empty_like(xy)
    out[0] = xy[0]

    for i in range(1, len(frames)):
        dt = float(max(1, frames[i] - frames[i - 1]))
        trans = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        proc_q = np.eye(4) * process_var * dt
        # Predict.
        state = trans @ state
        cov = trans @ cov @ trans.T + proc_q
        # Update.
        innovation = xy[i] - meas_h @ state
        s = meas_h @ cov @ meas_h.T + meas_r
        gain = cov @ meas_h.T @ np.linalg.inv(s)
        state = state + gain @ innovation
        cov = (np.eye(4) - gain @ meas_h) @ cov
        out[i] = state[:2]
    return out
