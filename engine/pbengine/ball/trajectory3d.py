"""Lift the 2D ball track into 3D (feet) with a gravity prior — monocular reconstruction.

Given the per-frame 2D detections, the recovered camera (:mod:`pbengine.ball.camera`), and the
bounces (ground-plane ``Z = 0`` anchors), we reconstruct the ball's 3D position. Between contacts
the ball is a projectile, so within a short window its motion is

    p(t) = p0 + v0 * t + 0.5 * (0, 0, -g) * t^2     (g = 32.174 ft/s^2)

The trick that keeps this cheap and robust: a pixel observation ``s [u v 1]^T = P [X Y Z 1]^T``
becomes, after cross-multiplying out the unknown scale ``s``, two equations *linear* in
``(X, Y, Z)`` — and ``(X, Y, Z)`` are linear in the unknowns ``(p0, v0)``. So each frame's fit is
an ordinary weighted linear least-squares in 6 unknowns (no nonlinear solver, NumPy only).

We fit one parabola per frame from a window of nearby detections (centred so the target frame is
``t = 0``, making the solved ``p0`` that frame's position and ``v0`` its velocity). Windows never
cross a bounce or a large detection gap, so a single ballistic arc is never blended with the next.
Single-camera 3D is approximate; per-window reprojection residual gates out unreliable frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbengine.ball.camera import CameraModel
from pbengine.court.court_model import LENGTH_FT, WIDTH_FT
from pbengine.schema.models import BallSample, Bounce

G_FT = 32.174  # gravitational acceleration, ft/s^2
FT_PER_S_TO_MPH = 0.6818181818


@dataclass
class _Obs:
    t: float  # time relative to the target frame (s)
    u: float
    v: float
    w: float  # weight (sqrt confidence, or anchor weight)


def _fit_parabola(
    obs: list[_Obs], anchors: list[tuple[float, float, float]], P: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Solve for ``m = [p0(3), v0(3)]`` from pixel observations + ground anchors.

    ``anchors`` are ``(t, X, Y)`` ground contacts (``Z = 0``) added as hard linear constraints.
    Returns ``(m, rms_reprojection_px)`` or ``None`` if under-constrained / singular.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    p1, p2, p3 = P[0], P[1], P[2]

    def world_basis(t: float) -> tuple[np.ndarray, np.ndarray]:
        # Pw(t) = N @ m + c, with the gravity term folded into c.
        n = np.zeros((4, 6))
        n[0, 0] = n[1, 1] = n[2, 2] = 1.0
        n[0, 3] = n[1, 4] = n[2, 5] = t
        c = np.array([0.0, 0.0, -0.5 * G_FT * t * t, 1.0])
        return n, c

    for o in obs:
        n, c = world_basis(o.t)
        for prow, coord in ((p1, o.u), (p2, o.v)):
            a = prow - coord * p3  # (P_row - coord * P3) . Pw = 0
            rows.append(o.w * (a @ n))
            rhs.append(-o.w * float(a @ c))

    for (t, x, y) in anchors:
        n, c = world_basis(t)
        wa = 5.0  # anchors are reliable; weight them above a single pixel obs
        # X(t) = x, Y(t) = y, Z(t) = 0  (Z row: p0z + v0z t - 0.5 g t^2 = 0)
        rows.append(wa * n[0])
        rhs.append(wa * x)
        rows.append(wa * n[1])
        rhs.append(wa * y)
        rows.append(wa * n[2])
        rhs.append(wa * (0.5 * G_FT * t * t))

    a_mat = np.asarray(rows)
    if a_mat.shape[0] < 6 or np.linalg.matrix_rank(a_mat) < 6:
        return None
    m, *_ = np.linalg.lstsq(a_mat, np.asarray(rhs), rcond=None)

    # Reprojection RMS over the (unweighted) pixel observations.
    sq = 0.0
    for o in obs:
        n, c = world_basis(o.t)
        pw = n @ m + c
        proj = P @ pw
        px = proj[:2] / proj[2]
        sq += (px[0] - o.u) ** 2 + (px[1] - o.v) ** 2
    rms = float(np.sqrt(sq / len(obs)))
    return m, rms


def _bounce_world(bounces: list[Bounce]) -> dict[int, tuple[float, float]]:
    return {b.frame: (b.court_xy[0] * WIDTH_FT, b.court_xy[1] * LENGTH_FT) for b in bounces}


def _estimate_at(
    center: int,
    samples: list[BallSample],
    frames: np.ndarray,
    bounce_frames: list[int],
    bounce_world: dict[int, tuple[float, float]],
    P: np.ndarray,
    fps: float,
    window: int,
    min_points: int,
    max_gap: int,
    max_reproj_px: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit the local gravity parabola centred at frame ``center`` and return ``(pos, vel)``.

    ``center`` may be a detected frame (used by :func:`reconstruct_3d`) or a missing one (used by
    :func:`fill_gaps_3d`); the windowing is identical. Neighbours are gathered within ``window``
    frames, never crossing a bounce or a gap larger than ``max_gap``. Returns ``None`` if the fit
    is under-constrained or reprojects worse than ``max_reproj_px``.
    """

    def crosses_bounce(a: int, b: int) -> bool:
        lo, hi = (a, b) if a < b else (b, a)
        return any(lo < bf < hi for bf in bounce_frames)

    obs: list[_Obs] = []
    lo = int(np.searchsorted(frames, center, side="right"))  # first sample with frame > center
    prev = center
    for k in range(lo - 1, -1, -1):  # walk left (includes a detection at ``center`` itself)
        fj = int(frames[k])
        if center - fj > window or crosses_bounce(fj, center) or (prev - fj) > max_gap:
            break
        s = samples[k]
        obs.append(_Obs((fj - center) / fps, s.px[0], s.px[1], np.sqrt(max(s.conf, 1e-3))))
        prev = fj
    prev = center
    for k in range(lo, len(samples)):  # walk right
        fj = int(frames[k])
        if fj - center > window or crosses_bounce(center, fj) or (fj - prev) > max_gap:
            break
        s = samples[k]
        obs.append(_Obs((fj - center) / fps, s.px[0], s.px[1], np.sqrt(max(s.conf, 1e-3))))
        prev = fj

    if len(obs) < min_points:
        return None
    anchors: list[tuple[float, float, float]] = []
    for bf in bounce_frames:
        if abs(bf - center) <= window and not crosses_bounce(min(bf, center), max(bf, center)):
            bx, by = bounce_world[bf]
            anchors.append(((bf - center) / fps, bx, by))

    fit = _fit_parabola(obs, anchors, P)
    if fit is None:
        return None
    m, rms = fit
    if rms > max_reproj_px:
        return None
    return m[:3], np.array([m[3], m[4], m[5]])  # position, velocity at t = 0 (this frame)


def reconstruct_3d(
    samples: list[BallSample],
    bounces: list[Bounce],
    camera: CameraModel,
    fps: float,
    window: int = 4,
    min_points: int = 4,
    max_gap: int = 6,
    max_reproj_px: float = 20.0,
) -> list[BallSample]:
    """Return copies of ``samples`` with ``world_ft`` and ``speed_mph`` filled where recoverable.

    ``window`` is the half-width (in frames) of the local fit. Windows are clipped so they never
    span a bounce or a gap larger than ``max_gap`` frames — that keeps two ballistic arcs from
    being blended. Frames whose fit reprojects worse than ``max_reproj_px`` are left in 2D only.
    """
    if not samples:
        return []
    samples = sorted(samples, key=lambda s: s.frame)
    frames = np.array([s.frame for s in samples])
    bounce_frames = sorted(b.frame for b in bounces)
    bounce_world = _bounce_world(bounces)

    out: list[BallSample] = []
    for s in samples:
        new = s.model_copy()
        est = _estimate_at(s.frame, samples, frames, bounce_frames, bounce_world, camera.P,
                           fps, window, min_points, max_gap, max_reproj_px)
        if est is not None:
            pos, vel = est
            new.world_ft = (float(pos[0]), float(pos[1]), float(max(pos[2], 0.0)))
            new.speed_mph = float(np.linalg.norm(vel)) * FT_PER_S_TO_MPH
        out.append(new)
    return out


def fill_gaps_3d(
    samples: list[BallSample],
    bounces: list[Bounce],
    camera: CameraModel,
    fps: float,
    max_fill_gap: int = 6,
    window: int = 4,
    min_points: int = 4,
    max_reproj_px: float = 20.0,
) -> list[BallSample]:
    """Add physics-interpolated samples at frames *between* detections so the track is continuous.

    For each missing frame inside a gap of at most ``max_fill_gap`` frames that does not cross a
    bounce, fit the local gravity parabola (same machinery as :func:`reconstruct_3d`), evaluate the
    3D position, and reproject it to pixels via ``camera``. Filled samples are flagged
    ``interpolated=True`` with ``conf=0`` so analytics never treat a guess as a measurement.

    Expects ``samples`` already enriched by :func:`reconstruct_3d`; returns measured + filled,
    sorted by frame.
    """
    if not samples:
        return []
    samples = sorted(samples, key=lambda s: s.frame)
    frames = np.array([s.frame for s in samples])
    bounce_frames = sorted(b.frame for b in bounces)
    bounce_world = _bounce_world(bounces)
    bset = set(bounce_frames)

    filled: list[BallSample] = []
    for a, b in zip(frames[:-1], frames[1:]):
        gap = int(b - a)
        if gap <= 1 or gap > max_fill_gap:
            continue  # adjacent, or too wide to interpolate safely
        if any(a < bf < b for bf in bounce_frames):
            continue  # a contact happened in this gap — don't blend two arcs
        for f in range(int(a) + 1, int(b)):
            if f in bset:
                continue
            est = _estimate_at(f, samples, frames, bounce_frames, bounce_world, camera.P,
                               fps, window, min_points, max_fill_gap, max_reproj_px)
            if est is None:
                continue
            pos, vel = est
            x, y, z = float(pos[0]), float(pos[1]), float(max(pos[2], 0.0))
            px = camera.project(np.array([[x, y, z]]))[0]
            filled.append(BallSample(
                frame=f,
                px=(float(px[0]), float(px[1])),
                court_xy=(x / WIDTH_FT, y / LENGTH_FT),
                conf=0.0,
                world_ft=(x, y, z),
                speed_mph=float(np.linalg.norm(vel)) * FT_PER_S_TO_MPH,
                interpolated=True,
            ))
    merged = samples + filled
    merged.sort(key=lambda s: s.frame)
    return merged


def fill_gaps_2d_samples(samples: list[BallSample], max_fill_gap: int = 4) -> list[BallSample]:
    """2D-only gap fill for the no-camera case: linearly interpolate pixels across short gaps.

    Continuity for the overlay when 3D is unavailable. Bounded by ``max_fill_gap`` to avoid
    hallucinating long stretches. Filled samples are flagged ``interpolated=True`` (no 3D/court).
    """
    from pbengine.ball.kalman import fill_gaps_2d

    if len(samples) < 2:
        return list(samples)
    samples = sorted(samples, key=lambda s: s.frame)
    frames = np.array([s.frame for s in samples])
    xy = np.array([[s.px[0], s.px[1]] for s in samples], dtype=float)
    out_f, out_xy, mask = fill_gaps_2d(frames, xy, max_fill_gap=max_fill_gap)

    by_frame = {s.frame: s for s in samples}
    out: list[BallSample] = []
    for fr, p, is_fill in zip(out_f, out_xy, mask):
        if is_fill:
            out.append(BallSample(frame=int(fr), px=(float(p[0]), float(p[1])),
                                  conf=0.0, interpolated=True))
        else:
            out.append(by_frame[int(fr)])
    return out
