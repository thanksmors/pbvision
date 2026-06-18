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
    bounce_world = {
        b.frame: (b.court_xy[0] * WIDTH_FT, b.court_xy[1] * LENGTH_FT) for b in bounces
    }

    def crosses_bounce(f_lo: int, f_hi: int) -> bool:
        return any(f_lo < bf < f_hi for bf in bounce_frames)

    out: list[BallSample] = []
    for i, s in enumerate(samples):
        f0 = s.frame
        obs: list[_Obs] = []
        prev = f0
        # Walk outward from the target collecting neighbours, stopping at a bounce or a big gap.
        for j in range(i, -1, -1):  # backwards incl. self
            fj = int(frames[j])
            if f0 - fj > window or crosses_bounce(fj, f0) or (prev - fj) > max_gap:
                break
            obs.append(_Obs((fj - f0) / fps, samples[j].px[0], samples[j].px[1],
                            np.sqrt(max(samples[j].conf, 1e-3))))
            prev = fj
        prev = f0
        for j in range(i + 1, len(samples)):
            fj = int(frames[j])
            if fj - f0 > window or crosses_bounce(f0, fj) or (fj - prev) > max_gap:
                break
            obs.append(_Obs((fj - f0) / fps, samples[j].px[0], samples[j].px[1],
                            np.sqrt(max(samples[j].conf, 1e-3))))
            prev = fj

        anchors: list[tuple[float, float, float]] = []
        for bf in bounce_frames:
            if abs(bf - f0) <= window and not crosses_bounce(min(bf, f0), max(bf, f0)):
                bx, by = bounce_world[bf]
                anchors.append(((bf - f0) / fps, bx, by))

        new = s.model_copy()
        if len(obs) >= min_points:
            fit = _fit_parabola(obs, anchors, camera.P)
            if fit is not None:
                m, rms = fit
                if rms <= max_reproj_px:
                    pos = m[:3]
                    vel = np.array([m[3], m[4], m[5]])  # velocity at t = 0 (this frame)
                    speed = float(np.linalg.norm(vel)) * FT_PER_S_TO_MPH
                    new.world_ft = (float(pos[0]), float(pos[1]), float(max(pos[2], 0.0)))
                    new.speed_mph = speed
        out.append(new)
    return out
