"""Lift the 2D ball track into 3D (feet) with a gravity prior — monocular reconstruction.

Given the per-frame 2D detections, the recovered camera (:mod:`pbengine.ball.camera`), and the
bounces (ground-plane ``Z = 0`` anchors), we reconstruct the ball's 3D position. Between contacts
the ball is a projectile, so over a whole ballistic segment its motion is the single parabola

    p(t) = p0 + v0 * t + 0.5 * (0, 0, -g) * t^2     (g = 32.174 ft/s^2)

The trick that keeps this cheap: a pixel observation ``s [u v 1]^T = P [X Y Z 1]^T`` becomes, after
cross-multiplying out the unknown scale ``s``, two equations *linear* in ``(X, Y, Z)`` — and
``(X, Y, Z)`` are linear in the unknowns ``(p0, v0)``. So fitting a parabola is an ordinary linear
least-squares in 6 unknowns (no nonlinear solver, NumPy only).

We fit **one parabola per ballistic segment**, robustly (RANSAC → refit on inliers), then sample it
analytically at every frame. Because all frames in a segment share one fit, the recovered arc is
smooth by construction, and detections that don't lie on the parabola are flagged as outliers (false
positives) instead of corrupting the track. Segments are bounded by bounces, large detection gaps,
and — via a reprojection-kink test — paddle hits that the bounce heuristic can't see. Single-camera
3D is approximate; a per-segment reprojection residual gates out unreliable segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbengine.ball.camera import CameraModel
from pbengine.court.court_model import LENGTH_FT, NET_CLEARANCE_FT, NET_Y, WIDTH_FT
from pbengine.schema.models import BallSample, Bounce

G_FT = 32.174  # gravitational acceleration, ft/s^2
FT_PER_S_TO_MPH = 0.6818181818

# Robust-fit tunables (see _fit_segment_robust / _segment_bounds).
_INLIER_PX = 8.0          # a detection within this reprojection error joins the segment's parabola
_SPLIT_REPROJ_PX = 12.0   # a run whose single-parabola RMS exceeds this is tested for a kink (hit)
_RANSAC_ITERS = 80
_MIN_SPLIT_POINTS = 5     # min detections on each side to consider a soft (paddle-hit) split
_GRAV = np.array([0.0, 0.0, -G_FT])


@dataclass
class _Obs:
    t: float  # time relative to the segment origin (s)
    u: float
    v: float
    w: float  # weight (sqrt confidence, or anchor weight)


def _world_basis(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Pw(t) = N @ m + c, with the gravity term folded into c. ``m = [p0(3), v0(3)]``."""
    n = np.zeros((4, 6))
    n[0, 0] = n[1, 1] = n[2, 2] = 1.0
    n[0, 3] = n[1, 4] = n[2, 5] = t
    c = np.array([0.0, 0.0, -0.5 * G_FT * t * t, 1.0])
    return n, c


def _sample_parabola(m: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (position, velocity) of the fitted parabola ``m`` at time ``t`` (s)."""
    pos = m[:3] + m[3:] * t + 0.5 * _GRAV * t * t
    vel = m[3:] + _GRAV * t
    return pos, vel


def _reproject_residuals(m: np.ndarray, obs: list[_Obs], P: np.ndarray) -> list[float]:
    """Per-observation reprojection error (px) of parabola ``m`` against ``obs``."""
    res: list[float] = []
    for o in obs:
        n, c = _world_basis(o.t)
        proj = P @ (n @ m + c)
        px = proj[:2] / proj[2]
        res.append(float(np.hypot(px[0] - o.u, px[1] - o.v)))
    return res


def _fit_parabola(
    obs: list[_Obs],
    anchors: list[tuple[float, float, float]],
    P: np.ndarray,
    *,
    return_residuals: bool = False,
):
    """Solve for ``m = [p0(3), v0(3)]`` from pixel observations + ground anchors.

    ``anchors`` are ``(t, X, Y)`` ground contacts (``Z = 0``) added as hard linear constraints.
    Returns ``(m, rms_reprojection_px)`` — or ``(m, rms, per_obs_residuals)`` when
    ``return_residuals`` — or ``None`` if under-constrained / singular.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    p1, p2, p3 = P[0], P[1], P[2]

    for o in obs:
        n, c = _world_basis(o.t)
        for prow, coord in ((p1, o.u), (p2, o.v)):
            a = prow - coord * p3  # (P_row - coord * P3) . Pw = 0
            rows.append(o.w * (a @ n))
            rhs.append(-o.w * float(a @ c))

    for (t, x, y) in anchors:
        n, c = _world_basis(t)
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

    res = _reproject_residuals(m, obs, P)
    rms = float(np.sqrt(np.mean(np.square(res)))) if res else 0.0
    if return_residuals:
        return m, rms, res
    return m, rms


def _bounce_world(bounces: list[Bounce]) -> dict[int, tuple[float, float]]:
    return {b.frame: (b.court_xy[0] * WIDTH_FT, b.court_xy[1] * LENGTH_FT) for b in bounces}


def _seg_obs(samples: list[BallSample], lo: int, hi: int, t0: int, fps: float) -> list[_Obs]:
    return [
        _Obs((samples[k].frame - t0) / fps, samples[k].px[0], samples[k].px[1],
             np.sqrt(max(samples[k].conf, 1e-3)))
        for k in range(lo, hi)
    ]


def _split_run(
    samples: list[BallSample], frames: np.ndarray, lo: int, hi: int, P: np.ndarray, fps: float
) -> list[tuple[int, int]]:
    """Recursively split a gap/bounce-free run at velocity discontinuities (paddle hits).

    A single ballistic arc fits one parabola with small RMS. If the whole run reprojects poorly but
    a left/right partition both fit cleanly, there's a kink (a contact the bounce heuristic missed);
    split at the partition that best separates the two arcs. Pure reprojection test — no 3D needed.
    """
    if hi - lo < 2 * _MIN_SPLIT_POINTS:
        return [(lo, hi)]
    t0 = int(frames[lo])
    obs_all = _seg_obs(samples, lo, hi, t0, fps)
    whole = _fit_parabola(obs_all, [], P)
    if whole is None or whole[1] <= _SPLIT_REPROJ_PX:
        return [(lo, hi)]

    best: tuple[float, int] | None = None
    for j in range(lo + _MIN_SPLIT_POINTS, hi - _MIN_SPLIT_POINTS + 1):
        left = _fit_parabola(obs_all[: j - lo], [], P)
        right = _fit_parabola(obs_all[j - lo:], [], P)
        if left is None or right is None:
            continue
        rl, rr = left[1], right[1]
        if rl < _SPLIT_REPROJ_PX and rr < _SPLIT_REPROJ_PX:
            score = whole[1] - max(rl, rr)
            if best is None or score > best[0]:
                best = (score, j)
    if best is None:
        return [(lo, hi)]
    j = best[1]
    return _split_run(samples, frames, lo, j, P, fps) + _split_run(samples, frames, j, hi, P, fps)


def _segment_bounds(
    frames: np.ndarray,
    samples: list[BallSample],
    bounce_frames: list[int],
    P: np.ndarray,
    fps: float,
    max_gap: int,
    *,
    split_on_kinks: bool = True,
) -> list[tuple[int, int]]:
    """Index ranges ``[lo, hi)`` into ``samples``, each a single ballistic arc.

    Hard cuts (certain): a detection gap > ``max_gap`` frames, and any bounce frame (a bounce
    reverses Z velocity, so it starts a new arc). Soft cuts (optional): paddle hits via
    :func:`_split_run`.
    """
    n = len(frames)
    if n == 0:
        return []
    cuts: set[int] = set()
    for k in range(n - 1):
        if frames[k + 1] - frames[k] > max_gap:
            cuts.add(k + 1)
        if any(frames[k] < bf <= frames[k + 1] for bf in bounce_frames):
            cuts.add(k + 1)
    # A detection landing exactly on a bounce frame begins its (new) arc.
    bset = set(bounce_frames)
    for k in range(1, n):
        if int(frames[k]) in bset:
            cuts.add(k)

    runs: list[tuple[int, int]] = []
    start = 0
    for k in range(1, n):
        if k in cuts:
            runs.append((start, k))
            start = k
    runs.append((start, n))

    if not split_on_kinks:
        return runs
    out: list[tuple[int, int]] = []
    for lo, hi in runs:
        out.extend(_split_run(samples, frames, lo, hi, P, fps))
    return out


def _fit_segment_robust(
    seg_obs: list[_Obs],
    anchors: list[tuple[float, float, float]],
    P: np.ndarray,
    *,
    inlier_px: float = _INLIER_PX,
    min_inliers: int = 4,
    iters: int = _RANSAC_ITERS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """RANSAC a single gravity parabola over a segment's observations.

    Returns ``(m, inlier_mask, rms_inliers)`` or ``None`` if too few observations agree. A minimal
    3-observation subset (6 equations) determines ``m``; the model with the most inliers
    (reprojection < ``inlier_px``) wins and is refit on its inliers.
    """
    nobs = len(seg_obs)
    if nobs < min_inliers:
        return None
    if nobs <= 3:
        fit = _fit_parabola(seg_obs, anchors, P)
        if fit is None:
            return None
        return fit[0], np.ones(nobs, dtype=bool), fit[1]

    rng = rng or np.random.default_rng(0)
    idx = np.arange(nobs)
    best: tuple[tuple[int, float], np.ndarray] | None = None
    for _ in range(iters):
        pick = rng.choice(idx, size=3, replace=False)
        fit = _fit_parabola([seg_obs[i] for i in pick], anchors, P)
        if fit is None:
            continue
        res = np.array(_reproject_residuals(fit[0], seg_obs, P))
        mask = res < inlier_px
        cnt = int(mask.sum())
        if cnt < 3:
            continue
        rms_in = float(np.sqrt(np.mean(np.square(res[mask]))))
        key = (cnt, -rms_in)
        if best is None or key > best[0]:
            best = (key, mask)
    if best is None or int(best[1].sum()) < min_inliers:
        return None

    inliers = [seg_obs[i] for i in idx if best[1][i]]
    fit = _fit_parabola(inliers, anchors, P)
    if fit is None:
        return None
    m = fit[0]
    res = np.array(_reproject_residuals(m, seg_obs, P))
    mask = res < inlier_px
    if int(mask.sum()) < min_inliers:
        mask = best[1]
    rms = float(np.sqrt(np.mean(np.square(res[mask]))))
    return m, mask, rms


def _segment_fits(
    samples: list[BallSample],
    frames: np.ndarray,
    bounces: list[Bounce],
    P: np.ndarray,
    fps: float,
    min_points: int,
    max_gap: int,
    max_reproj_px: float,
    split_on_kinks: bool,
    rng: np.random.Generator,
) -> list[dict]:
    """Robustly fit one parabola per ballistic segment. Returns fit records with inlier masks."""
    bounce_frames = sorted(b.frame for b in bounces)
    bounce_world = _bounce_world(bounces)
    segments = _segment_bounds(frames, samples, bounce_frames, P, fps, max_gap,
                               split_on_kinks=split_on_kinks)
    fits: list[dict] = []
    for lo, hi in segments:
        if hi - lo < min_points:
            continue
        t0 = int(frames[lo])
        last = int(frames[hi - 1])
        seg_obs = _seg_obs(samples, lo, hi, t0, fps)
        anchors = [((bf - t0) / fps, *bounce_world[bf]) for bf in bounce_frames if t0 <= bf <= last]
        fit = _fit_segment_robust(seg_obs, anchors, P, min_inliers=min_points, rng=rng)
        if fit is None:
            continue
        m, mask, rms = fit
        if rms > max_reproj_px:
            continue
        fits.append({"lo": lo, "hi": hi, "t0": t0, "last": last, "m": m, "mask": mask})
    return fits


def reconstruct_3d_segments(
    samples: list[BallSample],
    bounces: list[Bounce],
    camera: CameraModel,
    fps: float,
    window: int = 4,
    min_points: int = 4,
    max_gap: int = 6,
    max_reproj_px: float = 20.0,
    *,
    split_on_kinks: bool = True,
) -> tuple[list[BallSample], set[int]]:
    """Lift the track to 3D per ballistic segment; also return the rejected (outlier) frames.

    Each segment gets one robust parabola (see :func:`_fit_segment_robust`) sampled analytically at
    every inlier frame — so the arc is smooth by construction. Detections the segment fit rejects
    (false positives) are returned in ``outlier_frames`` so callers can cull them from the overlay.
    ``window`` is retained for signature compatibility (the fit is now per-segment, not per-window).
    """
    if not samples:
        return [], set()
    samples = sorted(samples, key=lambda s: s.frame)
    frames = np.array([s.frame for s in samples])
    rng = np.random.default_rng(0)
    fits = _segment_fits(samples, frames, bounces, camera.P, fps, min_points, max_gap,
                         max_reproj_px, split_on_kinks, rng)

    out = [s.model_copy() for s in samples]
    outlier_frames: set[int] = set()
    for fit in fits:
        lo, t0, m, mask = fit["lo"], fit["t0"], fit["m"], fit["mask"]
        for j in range(len(mask)):
            s = samples[lo + j]
            if not mask[j]:
                outlier_frames.add(s.frame)
                continue
            pos, vel = _sample_parabola(m, (s.frame - t0) / fps)
            o = out[lo + j]
            o.world_ft = (float(pos[0]), float(pos[1]), float(max(pos[2], 0.0)))
            o.speed_mph = float(np.linalg.norm(vel)) * FT_PER_S_TO_MPH
    return out, outlier_frames


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

    Thin wrapper over :func:`reconstruct_3d_segments` (drops the outlier set) for the callers/tests
    that only need the enriched samples.
    """
    out, _ = reconstruct_3d_segments(samples, bounces, camera, fps, window, min_points, max_gap,
                                     max_reproj_px)
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
    *,
    split_on_kinks: bool = True,
) -> list[BallSample]:
    """Add physics-interpolated samples at frames *between* detections so the track is continuous.

    Refits the same per-segment parabolas as :func:`reconstruct_3d_segments` and samples them at the
    missing frames inside each segment, so fills come from the *same* clean arc (no per-frame
    jitter). Filled samples are flagged ``interpolated=True`` with ``conf=0``. Segments are built
    with ``max_gap = max_fill_gap``, so a gap wider than ``max_fill_gap`` (or one containing a
    bounce) is a segment boundary and is never bridged.

    Expects ``samples`` already enriched by :func:`reconstruct_3d`; returns measured + filled.
    """
    if not samples:
        return []
    samples = sorted(samples, key=lambda s: s.frame)
    frames = np.array([s.frame for s in samples])
    present = {int(f) for f in frames}
    bset = {b.frame for b in bounces}
    rng = np.random.default_rng(0)
    fits = _segment_fits(samples, frames, bounces, camera.P, fps, min_points, max_fill_gap,
                         max_reproj_px, split_on_kinks, rng)

    filled: list[BallSample] = []
    for fit in fits:
        t0, last, m = fit["t0"], fit["last"], fit["m"]
        for f in range(t0 + 1, last):
            if f in present or f in bset:
                continue
            pos, vel = _sample_parabola(m, (f - t0) / fps)
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


def clean_2d_samples(samples: list[BallSample], max_fill_gap: int = 4, max_gap: int = 6) -> list[BallSample]:
    """No-camera path: robustly reject false-positive detections, then gap-fill in pixels.

    Splits the track on detection gaps, RANSAC-fits a degree-2 pixel parabola per segment to drop
    scattered false positives (which the constant-velocity Kalman would otherwise smear in), then
    linearly fills short gaps for a continuous overlay. Filled frames are flagged ``interpolated``.
    """
    from pbengine.ball.kalman import clean_track_2d

    if len(samples) < 2:
        return list(samples)
    samples = sorted(samples, key=lambda s: s.frame)
    frames = np.array([s.frame for s in samples])
    xy = np.array([[s.px[0], s.px[1]] for s in samples], dtype=float)
    keep = clean_track_2d(frames, xy, max_gap=max_gap)
    kept = [s for s, k in zip(samples, keep) if k]
    return fill_gaps_2d_samples(kept, max_fill_gap=max_fill_gap)


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


_ARC_APEX_CAP_FT = 18.0   # cap a single arc's peak height so a missing bounce can't shoot it skyward
_COURT_CLAMP_FT = 4.0     # keep horizontal anchors within the court footprint + this slack


def _ground_ft(court_xy: tuple[float, float] | None) -> tuple[float, float] | None:
    """Normalized court point -> (X, Y) feet, clamped to the court footprint + slack."""
    if court_xy is None:
        return None
    return _clamp_xy(court_xy[0] * WIDTH_FT, court_xy[1] * LENGTH_FT)


def _clamp_xy(x: float, y: float) -> tuple[float, float]:
    """Clamp a horizontal point to the court footprint + slack (backstop against a wild ray solve)."""
    return (min(max(x, -_COURT_CLAMP_FT), WIDTH_FT + _COURT_CLAMP_FT),
            min(max(y, -_COURT_CLAMP_FT), LENGTH_FT + _COURT_CLAMP_FT))


def _camera_center(camera: CameraModel) -> tuple[np.ndarray, np.ndarray] | None:
    """``(O, Minv)``: camera centre (world ft) and ``M⁻¹`` (``M = P[:, :3]``) for pixel-ray back-proj.

    A point ``O + s·(Minv @ [u, v, 1])`` reprojects through ``P`` **exactly** to ``(u, v)`` — for any
    ``s`` — regardless of focal error. So fixing the depth via a known height keeps the 3D ball on its
    detection's ray (it overlays the 2D track) and can't fly off the way a free-depth solve can.
    """
    M = camera.P[:, :3]
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    return -Minv @ camera.P[:, 3], Minv


_CONTACT_Z_RANGE = (0.5, 10.0)  # plausible ball height at a paddle contact (dink .. overhead), ft


def _net_crossing_frames(out: list[BallSample]) -> list[int]:
    """Frames where the ball crosses the net line (``court_xy.y`` passes ``NET_Y``)."""
    pts = [(s.frame, s.court_xy[1]) for s in out if s.court_xy is not None]
    crossings: list[int] = []
    for (f0, y0), (f1, y1) in zip(pts, pts[1:]):
        if (y0 - NET_Y) * (y1 - NET_Y) < 0:  # straddles the net between these samples
            crossings.append(f0 if abs(y0 - NET_Y) <= abs(y1 - NET_Y) else f1)
    return crossings


def _height_knots(
    out: list[BallSample],
    bounces: list[Bounce],
    contacts: list[tuple[int, float]] | None,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> list[tuple[int, float]]:
    """Typed height anchors ``(frame, Z_ft)`` the modeled arc must pass through, by precedence.

    Floor contacts pin ``Z=0`` (bounces, rally edges); paddle contacts pin the hitter's contact
    height. Net-crossings are a **lower bound**, not a hard pin: the arc is lifted to net clearance
    only where the bounce/contact arc would otherwise dip *below* the tape — so a lob or high drive
    keeps its natural (higher) arc instead of being flattened to net height.
    """
    knots: dict[int, float] = {}
    knots.setdefault(out[0].frame, 0.0)
    knots.setdefault(out[-1].frame, 0.0)
    for b in bounces:
        if start_frame <= b.frame <= end_frame:
            knots[b.frame] = 0.0
    for f, z in contacts or []:  # highest precedence
        if start_frame <= f <= end_frame:
            knots[f] = min(max(z, _CONTACT_Z_RANGE[0]), _CONTACT_Z_RANGE[1])
    # Net clearance as a floor: only add a knot where the bounce/contact arc sinks below the tape.
    base = sorted(knots.items())
    for f in _net_crossing_frames(out):
        if f not in knots and _height_at(f, base, fps) < NET_CLEARANCE_FT:
            knots[f] = NET_CLEARANCE_FT
    return sorted(knots.items())


def _height_at(frame: int, knots: list[tuple[int, float]], fps: float) -> float:
    """Modeled ball height (ft) at ``frame``: a capped gravity arc between bracketing height knots."""
    if not knots:
        return 0.0
    if frame <= knots[0][0]:
        return knots[0][1]
    if frame >= knots[-1][0]:
        return knots[-1][1]
    lo = max((k for k in knots if k[0] <= frame), key=lambda k: k[0])
    hi = min((k for k in knots if k[0] > frame), key=lambda k: k[0])
    T = (hi[0] - lo[0]) / fps
    t = (frame - lo[0]) / fps
    baseline = lo[1] + (hi[1] - lo[1]) * (t / T)  # linear ramp between the two knot heights
    bump = 0.5 * G_FT * t * (T - t)               # gravity parabola on top (0 at both knots)
    apex_raw = G_FT * T * T / 8.0
    if apex_raw > _ARC_APEX_CAP_FT:
        bump *= _ARC_APEX_CAP_FT / apex_raw  # cap the peak (e.g. a missed bounce leaving a huge gap)
    return max(baseline + bump, 0.0)


def ball_world_ft(
    traj: list[BallSample],
    bounces: list[Bounce],
    camera: CameraModel | None,
    start_frame: int,
    end_frame: int,
    fps: float,
    contacts: list[tuple[int, float]] | None = None,
) -> list[BallSample]:
    """Set an on-court ``world_ft`` at every sample by placing each ball pixel on its camera ray.

    Single-camera height/depth can't be *measured*, so we fix the height from a model — a gravity arc
    through typed height knots (bounces & rally edges at Z=0, net-crossings at net clearance, and
    paddle ``contacts`` at the hitter's contact height) — and read the horizontal off the **per-frame**
    pixel ray at that height (``s = (Z − O_z) / d_z``). Because the point stays on the detection's ray
    it reprojects exactly to the 2D track (the 3D ball overlays what you see), and at Z=0 it equals the
    exact ground projection; for airborne frames it walks back along the ray, undoing the
    ground-projection's outward bias. With no camera (or a degenerate ray) it falls back to the raw
    ground projection ``court_xy`` at the modeled height. ``contacts`` is a list of
    ``(frame, height_ft)``. ``speed_mph`` is finite-differenced from the result; the ``interpolated``
    flag is left untouched.
    """
    if not traj:
        return traj
    out = [s.model_copy() for s in sorted(traj, key=lambda s: s.frame)]
    knots = _height_knots(out, bounces, contacts, start_frame, end_frame, fps)
    cc = _camera_center(camera) if camera is not None else None

    for s in out:
        z = _height_at(s.frame, knots, fps)
        xy: tuple[float, float] | None = None
        if cc is not None:
            cam_o, Minv = cc
            d = Minv @ np.array([float(s.px[0]), float(s.px[1]), 1.0])
            if abs(d[2]) > 1e-9:  # ray meets the horizontal Z=z plane at a unique point
                w = cam_o + (z - cam_o[2]) / d[2] * d  # ray sign/scale is arbitrary; intersection isn't
                xy = _clamp_xy(float(w[0]), float(w[1]))
        if xy is None:  # no camera / degenerate ray -> raw ground projection at the modeled height
            xy = _ground_ft(s.court_xy)
        if xy is None:
            continue
        s.world_ft = (xy[0], xy[1], z)

    # Finite-difference speed from the now-continuous 3D path (central where possible).
    n = len(out)
    for k in range(n):
        if out[k].world_ft is None:
            continue
        a = out[k - 1] if k > 0 and out[k - 1].world_ft is not None else out[k]
        b = out[k + 1] if k + 1 < n and out[k + 1].world_ft is not None else out[k]
        df = b.frame - a.frame
        if df <= 0:
            continue
        disp = np.asarray(b.world_ft, dtype=float) - np.asarray(a.world_ft, dtype=float)
        out[k].speed_mph = float(np.linalg.norm(disp)) / (df / fps) * FT_PER_S_TO_MPH
    return out


def densify_rally(
    traj: list[BallSample],
    start_frame: int,
    end_frame: int,
    bounces: list[Bounce],
    camera: CameraModel | None,
    fps: float,
) -> list[BallSample]:
    """Guarantee a ball sample at **every** frame in ``[start_frame, end_frame]``.

    The physics/2D fills (:func:`fill_gaps_3d` / :func:`clean_2d_samples`) only bridge short
    within-arc gaps; any wider miss is left as a hole, so the stored track is incomplete. Downstream
    data analysis expects a value per frame, so this linearly interpolates every *still-missing*
    frame in the rally span between its two bounding samples — ``px`` always, ``court_xy`` /
    ``world_ft`` / ``speed_mph`` when both ends carry them. Fills are flagged ``interpolated=True``
    with ``conf=0`` (measured-only stats already ignore them). A rally is run separately, so this
    never bridges across a rally split.

    When a bounce falls inside a gap, an anchor at the bounce (``z=0``) is inserted so a long fill
    dips to the floor instead of cutting a straight line through the court.
    """
    if not traj:
        return traj
    knots = sorted(traj, key=lambda s: s.frame)
    present = {s.frame for s in knots}

    # Ground anchors at bounce frames that landed inside a gap (needs the camera to get the pixel).
    if camera is not None:
        for b in bounces:
            if b.frame in present or not (start_frame < b.frame < end_frame):
                continue
            X, Y = b.court_xy[0] * WIDTH_FT, b.court_xy[1] * LENGTH_FT
            px = camera.project(np.array([[X, Y, 0.0]]))[0]
            knots.append(BallSample(
                frame=int(b.frame), px=(float(px[0]), float(px[1])),
                court_xy=(float(b.court_xy[0]), float(b.court_xy[1])), conf=0.0,
                world_ft=(float(X), float(Y), 0.0), interpolated=True,
            ))
            present.add(b.frame)
        knots.sort(key=lambda s: s.frame)

    fills: list[BallSample] = []
    for a, c in zip(knots, knots[1:]):
        span = c.frame - a.frame
        for f in range(a.frame + 1, c.frame):
            if f in present:
                continue
            t = (f - a.frame) / span
            px = (a.px[0] + t * (c.px[0] - a.px[0]), a.px[1] + t * (c.px[1] - a.px[1]))
            court = world = speed = None
            if a.court_xy is not None and c.court_xy is not None:
                court = (a.court_xy[0] + t * (c.court_xy[0] - a.court_xy[0]),
                         a.court_xy[1] + t * (c.court_xy[1] - a.court_xy[1]))
            if a.world_ft is not None and c.world_ft is not None:
                world = tuple(a.world_ft[k] + t * (c.world_ft[k] - a.world_ft[k]) for k in range(3))
            if a.speed_mph is not None and c.speed_mph is not None:
                speed = a.speed_mph + t * (c.speed_mph - a.speed_mph)
            fills.append(BallSample(
                frame=f, px=(float(px[0]), float(px[1])), court_xy=court, conf=0.0,
                world_ft=world, speed_mph=speed, interpolated=True,
            ))

    out = knots + fills
    out.sort(key=lambda s: s.frame)
    return out
