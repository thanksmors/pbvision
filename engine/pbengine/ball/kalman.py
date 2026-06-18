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


def _ransac_poly1d(t: np.ndarray, y: np.ndarray, *, inlier: float, iters: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Boolean inlier mask for a robust degree-2 fit of ``y(t)`` (3 points determine a parabola)."""
    n = len(t)
    if n <= 3:
        return np.ones(n, dtype=bool)
    idx = np.arange(n)
    best: tuple[int, np.ndarray] | None = None
    for _ in range(iters):
        pick = rng.choice(idx, size=3, replace=False)
        try:
            coef = np.polyfit(t[pick], y[pick], 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        mask = np.abs(np.polyval(coef, t) - y) < inlier
        cnt = int(mask.sum())
        if best is None or cnt > best[0]:
            best = (cnt, mask)
    return best[1] if best is not None else np.ones(n, dtype=bool)


def ransac_poly_segment(frames: np.ndarray, xy: np.ndarray, *, inlier_px: float = 12.0,
                        iters: int = 60, rng: np.random.Generator | None = None) -> np.ndarray:
    """Robustly fit ``x(t)``/``y(t)`` as degree-2 polynomials; return a per-point inlier mask.

    A short ballistic arc projects to a roughly parabolic pixel path, so a point that disagrees with
    a degree-2 fit of its segment is a scattered false positive. A point must be an inlier on *both*
    axes to be kept.
    """
    frames = np.asarray(frames, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if len(frames) <= 3:
        return np.ones(len(frames), dtype=bool)
    rng = rng or np.random.default_rng(0)
    t = frames - frames[0]
    mx = _ransac_poly1d(t, xy[:, 0], inlier=inlier_px, iters=iters, rng=rng)
    my = _ransac_poly1d(t, xy[:, 1], inlier=inlier_px, iters=iters, rng=rng)
    return mx & my


def clean_track_2d(frames: np.ndarray, xy: np.ndarray, *, max_gap: int = 6,
                   bounce_frames: tuple[int, ...] = (), inlier_px: float = 12.0) -> np.ndarray:
    """Per-point keep mask: split on gaps/bounces, RANSAC each segment, drop pixel-space outliers.

    The robust pixel-parabola defense the constant-velocity :func:`smooth` cannot provide on its own
    (it smears a physically-plausible false positive into the track instead of rejecting it).
    """
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    n = len(frames)
    if n == 0:
        return np.zeros(0, dtype=bool)
    bset = set(bounce_frames)
    # Segment boundaries: large gaps, or a bounce between consecutive detections.
    starts = [0]
    for k in range(1, n):
        if frames[k] - frames[k - 1] > max_gap or any(
            frames[k - 1] < bf <= frames[k] for bf in bset
        ):
            starts.append(k)
    starts.append(n)
    keep = np.ones(n, dtype=bool)
    rng = np.random.default_rng(0)
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a <= 3:
            continue  # too short to reject anything
        keep[a:b] = ransac_poly_segment(frames[a:b], xy[a:b], inlier_px=inlier_px, rng=rng)
    return keep


def fill_gaps_2d(
    frames: np.ndarray, xy: np.ndarray, max_fill_gap: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate pixel positions across *short* detection gaps.

    Constant-velocity between two detections is a straight line, so we fill a gap of width
    ``<= max_fill_gap`` with evenly-spaced points. Gaps wider than that are left empty rather than
    hallucinated. Returns ``(frames_out, xy_out, filled_mask)`` with one entry per emitted frame;
    ``filled_mask[i]`` is True for interpolated frames, False for original detections.

    ``frames`` (N,) must be sorted. Used as the 2D-only fallback when no camera/3D is available.
    """
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    if len(frames) < 2:
        return frames, xy, np.zeros(len(frames), dtype=bool)

    out_f: list[int] = []
    out_xy: list[np.ndarray] = []
    mask: list[bool] = []
    for i in range(len(frames)):
        out_f.append(int(frames[i]))
        out_xy.append(xy[i])
        mask.append(False)
        if i + 1 < len(frames):
            gap = int(frames[i + 1] - frames[i])
            if 1 < gap <= max_fill_gap:
                for k in range(1, gap):
                    a = k / gap
                    out_f.append(int(frames[i]) + k)
                    out_xy.append((1 - a) * xy[i] + a * xy[i + 1])
                    mask.append(True)
    return np.array(out_f), np.array(out_xy), np.array(mask, dtype=bool)
