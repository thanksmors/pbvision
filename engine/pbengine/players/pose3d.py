"""Lift 2D pose keypoints into a real 3D skeleton and derive a wrist-anchored paddle (monocular).

A single camera can't *directly* measure per-keypoint depth, but the depth is recoverable here from
three constraints (the geometry behind :func:`lift_pose_3d`):

1. **Assumed height** (``PLAYER_HEIGHT_FT`` ≈ 5'9") → metric **bone lengths** (``_BONE_FRAC``).
2. **Feet on the ground** → an exact anchor: an ankle's pixel ray ∩ ``Z=0`` gives its true court
   position (the homography is exact on the ground plane).
3. **Players face each other** across the net → the body forward direction is known, which breaks the
   front/back depth-sign ambiguity that pure monocular can't.

The key fact: a pixel ``(u,v)`` back-projects to the world ray ``X(s)=C+s·d`` where, using the *same*
projection matrix ``P`` the forward projection uses, ``M=P[:, :3]``, ``C=-M⁻¹P[:, 3]``,
``d=M⁻¹[u,v,1]``. Every point on that ray reprojects **exactly** to ``(u,v)``. So each joint is
placed on its own ray (the 3D skeleton's silhouette always matches the observed 2D pose) and we only
solve *where along the ray* (depth) using the bone length to its parent. Of the two depth roots we
pick using the body facing (+ temporal consistency). This is the Taylor (2000) articulated
reconstruction, anchored at the feet. From the broadcast-camera angle the result is identical to the
2D pose; rotated, it shows real depth.

:func:`billboard_lift` is kept as a degenerate-geometry fallback (flat cutout). The paddle is a
wrist-anchored estimate (no paddle model): a short segment past the stronger wrist along elbow→wrist.

World frame matches :mod:`pbengine.ball.camera`: X across the 20 ft width, Y along the 44 ft
length, Z up; ground plane Z = 0.
"""

from __future__ import annotations

import math

import numpy as np

from pbengine.ball.camera import CameraModel
from pbengine.detect.pose import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, N_KEYPOINTS, NOSE,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)
from pbengine.court.court_model import LENGTH_FT, WIDTH_FT

_CONF_FLOOR = 0.3  # keypoints below this are unreliable; skipped in the lift / paddle
PLAYER_HEIGHT_FT = 5.75  # assumed standing height (~5'9", average player) for the bone-length scale

# Bone lengths as fractions of stature (Drillis & Contini anthropometry, approximate).
_BONE_FRAC = {"shank": 0.246, "thigh": 0.245, "torso": 0.30, "uarm": 0.186, "farm": 0.146,
              "head": 0.16}

# Trunk breadths as fractions of stature — biacromial (shoulder) and bi-iliac (hip) widths. These
# drive the body-yaw refinement (_yaw_pair): the metric width a turned trunk must span to reproject
# the observed L/R pixel separation tells us how the player is rotated about the vertical.
SHOULDER_W_FRAC = 0.259
HIP_W_FRAC = 0.191

# Kinematic chains solved foot-up: (child, parent, bone, forward_tendency). forward_tendency is the
# sign of the child's typical depth offset from its parent along the body-forward axis (+1 forward,
# -1 back, 0 axial/upright) — used to pick the depth root on the seed frame. Legs + trunk are solved
# first (each joint's depth is well-conditioned from its bone to a parent at a different image
# location); the trunk pairs then get a yaw refinement, and the arms are solved last against the
# resulting dynamic body-forward.
_LEG_TRUNK_CHAIN = (
    (L_KNEE, L_ANKLE, "shank", +1), (R_KNEE, R_ANKLE, "shank", +1),
    (L_HIP, L_KNEE, "thigh", -1), (R_HIP, R_KNEE, "thigh", -1),
    (L_SHOULDER, L_HIP, "torso", 0), (R_SHOULDER, R_HIP, "torso", 0),
)
_ARM_CHAIN = (
    (L_ELBOW, L_SHOULDER, "uarm"), (R_ELBOW, R_SHOULDER, "uarm"),
    (L_WRIST, L_ELBOW, "farm"), (R_WRIST, R_ELBOW, "farm"),
)


def _back_project(camera: CameraModel):
    """Return ``(C, Minv)``: camera centre (world ft) and ``M⁻¹`` for ray back-projection.

    Uses the same ``P`` as :meth:`CameraModel.project` (``M = P[:, :3]``), so a pixel's ray
    ``C + s·(Minv @ [u,v,1])`` reprojects exactly and its ∩ ``Z=0`` equals the homography ground
    point — sidestepping the focal/orthonormalization inconsistency that collapsed the billboard.
    Returns ``None`` if ``M`` is singular (degenerate camera).
    """
    M = camera.P[:, :3]
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    C = -Minv @ camera.P[:, 3]
    return C, Minv


def _clamp_z(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], p[1], max(p[2], 0.0)])


def _solve_joint(C, d, parent, L, ftend, sigma, prev):
    """Place a child joint on its ray ``C+s·d`` at distance ``L`` from ``parent``; resolve the sign.

    Two depth roots (near/far); pick by previous-frame proximity if available, else the body-facing
    tendency ``ftend`` and camera-front sign ``sigma`` (axial joints pick the more upright root). If
    the bone is more foreshortened than ``L`` allows (no real root) use the closest point on the ray.
    """
    a = float(d @ d)
    diff = C - parent
    b = 2.0 * float(d @ diff)
    c = float(diff @ diff) - L * L
    disc = b * b - 4.0 * a * c
    if a < 1e-12:
        return _clamp_z(parent)
    if disc <= 0.0:
        return _clamp_z(C + (-b / (2.0 * a)) * d)  # closest point on ray (max foreshortening)
    sq = math.sqrt(disc)
    cands = [C + s * d for s in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)) if s > 0]
    if not cands:
        return _clamp_z(C + (-b / (2.0 * a)) * d)
    if len(cands) == 1:
        return _clamp_z(cands[0])
    near, far = cands  # ordered by s (near = smaller depth)
    if prev is not None:
        prev = np.asarray(prev, dtype=float)
        chosen = near if np.linalg.norm(near - prev) <= np.linalg.norm(far - prev) else far
    elif ftend != 0:
        chosen = near if (ftend * sigma) > 0 else far  # forward parts are nearer when camera in front
    else:
        chosen = near if near[2] >= far[2] else far    # axial: pick the more upright root
    return _clamp_z(chosen)


def _yaw_pair(P3, iL, iR, width, camera, keypoints_px, conf):
    """Give a trunk pair (hips or shoulders) real thickness/orientation via a body-yaw refinement.

    Keeps the pair's well-conditioned midpoint depth (from the bone-length chain) but rotates the two
    joints about the vertical through that midpoint to the yaw whose ``width``-apart endpoints best
    reproject the observed L/R pixels. Front-on -> the bar stays lateral (no depth); turned -> the near
    joint comes forward, the far one back. Solving *orientation* (a normalized, clamped quantity) this
    way avoids the depth blow-up of inferring a trunk's absolute depth from its width. Endpoints land
    on a horizontal bar through the midpoint; a no-op when either keypoint is low-confidence.
    """
    if conf[iL] < _CONF_FLOOR or conf[iR] < _CONF_FLOOR or P3[iL] is None or P3[iR] is None:
        return
    M = (P3[iL] + P3[iR]) / 2.0
    obs = np.array([keypoints_px[iR], keypoints_px[iL]], dtype=float)  # [R, L]
    half = width / 2.0

    def err(theta):
        lat = np.array([math.cos(theta), math.sin(theta), 0.0])
        pts = np.array([M + half * lat, M - half * lat])  # [+lat -> R, -lat -> L]
        proj = camera.project(pts)
        return float(np.hypot(*(proj - obs).T).sum()), pts

    # Coarse sweep over all yaws, then a local refine (theta and theta+pi swap L/R, so the full
    # circle covers both assignments).
    best_t, best = 0.0, None
    for i in range(72):
        t = 2.0 * math.pi * i / 72
        e, _ = err(t)
        if best is None or e < best:
            best, best_t = e, t
    for i in range(-25, 26):
        t = best_t + (math.pi / 72) * (i / 25.0)
        e, _ = err(t)
        if e < best:
            best, best_t = e, t
    _, pts = err(best_t)
    P3[iR], P3[iL] = _clamp_z(pts[0]), _clamp_z(pts[1])


def lift_pose_3d(
    keypoints_px: list[tuple[float, float]],
    keypoint_conf: list[float] | None,
    court_xy: tuple[float, float],
    camera: CameraModel,
    forward_dir: tuple[float, float, float],
    prev_pose: list[tuple[float, float, float]] | None = None,
    player_height_ft: float = PLAYER_HEIGHT_FT,
) -> list[tuple[float, float, float]] | None:
    """Reconstruct a real 3D skeleton (COCO-17, court feet) from the 2D pose. See module docstring.

    ``court_xy`` is the player's normalized foot position (for the feet fallback anchor);
    ``forward_dir`` is the unit body-forward vector (``+Y`` side A, ``−Y`` side B); ``prev_pose`` is
    the previous frame's solved pose for temporal sign consistency. Returns 17 ``(X,Y,Z)`` or
    ``None`` on degenerate geometry (caller falls back to :func:`billboard_lift`).
    """
    bp = _back_project(camera)
    if bp is None:
        return None
    C, Minv = bp
    conf = keypoint_conf if keypoint_conf is not None else [1.0] * N_KEYPOINTS
    F = np.asarray(forward_dir, dtype=float)
    bones = {k: f * player_height_ft for k, f in _BONE_FRAC.items()}
    rays = [Minv @ np.array([float(u), float(v), 1.0]) for u, v in keypoints_px]
    foot_ground = np.array([*ground_xy_ft(court_xy), 0.0])

    P3: list[np.ndarray | None] = [None] * N_KEYPOINTS

    # Anchor the ankles on the ground (exact); low-conf ankle -> the court_xy foot point.
    for ankle in (L_ANKLE, R_ANKLE):
        d = rays[ankle]
        g = None
        if conf[ankle] >= _CONF_FLOOR and abs(d[2]) > 1e-9:
            s = -C[2] / d[2]
            if s > 0:
                g = C + s * d
        P3[ankle] = _clamp_z(g) if g is not None else foot_ground.copy()

    foot = (P3[L_ANKLE] + P3[R_ANKLE]) / 2.0
    sigma0 = 1.0 if float((C - foot) @ F) >= 0 else -1.0

    # Legs + trunk on their bone-length rays (well-conditioned depth from each parent).
    for child, parent, bone, ftend in _LEG_TRUNK_CHAIN:
        prev = prev_pose[child] if prev_pose is not None else None
        P3[child] = _solve_joint(C, rays[child], P3[parent], bones[bone], ftend, sigma0, prev)

    # Real torso volume: rotate each trunk pair about the vertical to the body yaw that reprojects the
    # observed L/R pixels at the anthropometric width (keeps the chain's midpoint depth). This is what
    # turns the flat "upright" trunk into one with genuine front/back thickness and facing.
    _yaw_pair(P3, L_HIP, R_HIP, HIP_W_FRAC * player_height_ft, camera, keypoints_px, conf)
    _yaw_pair(P3, L_SHOULDER, R_SHOULDER, SHOULDER_W_FRAC * player_height_ft, camera, keypoints_px, conf)

    # Dynamic body-forward from the yawed shoulder line (perpendicular, horizontal), oriented by the
    # coarse facing prior; re-solve the arms against it so a reaching arm gets real in/out depth
    # relative to how the player is actually turned, not just the court axis.
    F_dyn = np.cross(np.array([0.0, 0.0, 1.0]), P3[R_SHOULDER] - P3[L_SHOULDER])
    nf = float(np.linalg.norm(F_dyn))
    F_dyn = (F_dyn / nf if float(F_dyn @ F) >= 0 else -F_dyn / nf) if nf > 1e-6 else F
    sigma = 1.0 if float((C - foot) @ F_dyn) >= 0 else -1.0
    for child, parent, bone in _ARM_CHAIN:
        prev = prev_pose[child] if prev_pose is not None else None
        P3[child] = _solve_joint(C, rays[child], P3[parent], bones[bone], +1, sigma, prev)

    # Head: nose from the shoulder centre; eyes/ears placed at the nose's depth on their own rays.
    mid_sh = (P3[L_SHOULDER] + P3[R_SHOULDER]) / 2.0
    prev_nose = prev_pose[NOSE] if prev_pose is not None else None
    P3[NOSE] = _solve_joint(C, rays[NOSE], mid_sh, bones["head"], 0, sigma, prev_nose)
    s_nose = float((P3[NOSE] - C) @ rays[NOSE]) / float(rays[NOSE] @ rays[NOSE])
    for k in (1, 2, 3, 4):  # eyes, ears (not drawn; kept for a full 17-length array)
        P3[k] = _clamp_z(C + s_nose * rays[k])

    out = [(float(p[0]), float(p[1]), float(p[2])) for p in P3]
    if not all(np.isfinite(v) for p in out for v in p):
        return None  # numeric blow-up -> caller falls back to the billboard
    return out


def _camera_right_horizontal(camera: CameraModel) -> np.ndarray | None:
    """Camera's right axis (world dir of image +u), projected horizontal and normalized.

    ``camera.R`` maps world -> camera, so ``R.T @ [1,0,0]`` is the camera x (right) axis in world
    feet. Flattening Z gives the horizontal in-plane axis of a vertical billboard that faces the
    camera. Returns None if it degenerates (camera looking straight down -> no horizontal facing).
    """
    right = camera.R.T @ np.array([1.0, 0.0, 0.0])
    right[2] = 0.0
    n = float(np.linalg.norm(right))
    if n < 1e-6:
        return None
    return right / n


def billboard_lift(
    keypoints_px: list[tuple[float, float]],
    keypoint_conf: list[float] | None,
    ground_xy_ft: tuple[float, float],
    camera: CameraModel,
    bbox_px: tuple[float, float, float, float] | None = None,
    player_height_ft: float = PLAYER_HEIGHT_FT,
) -> list[tuple[float, float, float]] | None:
    """Lift COCO keypoints to 3D court feet via a vertical billboard at ``ground_xy_ft``.

    ``ground_xy_ft`` is the player's foot position on the court (Z=0), already projected from the
    homography (exact on the ground plane). The skeleton is placed in a vertical plane through that
    point, facing the camera; each keypoint's *pixel offset* from the foot is mapped into the plane
    at a scale of ``player_height_ft / bbox_pixel_height``.

    Crucially the vertical scale comes from the bounding-box pixel height + an assumed standing
    height, **not** from the recovered camera's focal length — that focal is only approximate
    (recovered from a homography) and, when off, used to squish every keypoint to Z≈0 ("stick"
    collapse). Anchoring to a known height makes the standing skeleton robust. The camera is used
    only for the (well-conditioned) facing direction. Limbs reaching toward/away from the camera
    still flatten onto the plane — the inherent monocular approximation.

    Returns one (X, Y, Z) per keypoint (Z clamped >= 0), or ``None`` if the geometry is degenerate
    (camera ~overhead) or no usable pixel height is available.
    """
    side = _camera_right_horizontal(camera)
    if side is None:
        return None  # camera ~overhead -> no horizontal facing direction
    up = np.array([0.0, 0.0, 1.0])
    ground = np.array([float(ground_xy_ft[0]), float(ground_xy_ft[1]), 0.0])

    # Foot anchor pixel + vertical scale (feet per pixel) from the bbox; fall back to the confident
    # keypoints' own vertical pixel span if no bbox is available.
    if bbox_px is not None:
        x1, y1, x2, y2 = (float(c) for c in bbox_px)
        foot_x, foot_y = (x1 + x2) / 2.0, max(y1, y2)
        h_px = abs(y2 - y1)
    else:
        vs = [v for (u, v), c in zip(keypoints_px, keypoint_conf or [1.0] * len(keypoints_px))
              if c >= _CONF_FLOOR]
        us = [u for (u, v), c in zip(keypoints_px, keypoint_conf or [1.0] * len(keypoints_px))
              if c >= _CONF_FLOOR]
        if len(vs) < 2:
            return None
        foot_x, foot_y, h_px = (min(us) + max(us)) / 2.0, max(vs), abs(max(vs) - min(vs))
    if h_px < 1.0:
        return None
    scale = player_height_ft / h_px

    out: list[tuple[float, float, float]] = []
    for u, v in keypoints_px:
        du, dv = u - foot_x, foot_y - v  # dv > 0 for keypoints above the foot
        p = ground + side * (du * scale) + up * (dv * scale)
        out.append((float(p[0]), float(p[1]), float(max(p[2], 0.0))))
    return out


def _strong_wrist(keypoint_conf: list[float] | None) -> int | None:
    """Index of the higher-confidence wrist (right vs left), or None if both are weak/unknown."""
    if keypoint_conf is None:
        return R_WRIST  # no confidences (e.g. fixture) -> default to right
    lc, rc = keypoint_conf[L_WRIST], keypoint_conf[R_WRIST]
    if max(lc, rc) < _CONF_FLOOR:
        return None
    return R_WRIST if rc >= lc else L_WRIST


def paddle_segment_px(
    keypoints_px: list[tuple[float, float]], keypoint_conf: list[float] | None
) -> tuple[float, float, float, float] | None:
    """Wrist-anchored paddle as (base_x, base_y, tip_x, tip_y) in pixels, or None.

    Base = the stronger wrist; tip = wrist + (wrist - elbow), i.e. one forearm-length past the hand
    along the forearm direction (where a held paddle sits).
    """
    wrist = _strong_wrist(keypoint_conf)
    if wrist is None:
        return None
    elbow = L_ELBOW if wrist == L_WRIST else R_ELBOW
    wx, wy = keypoints_px[wrist]
    ex, ey = keypoints_px[elbow]
    tx, ty = wx + (wx - ex), wy + (wy - ey)
    return (float(wx), float(wy), float(tx), float(ty))


def paddle_tip_world(
    pose_world_ft: list[tuple[float, float, float]] | None, keypoint_conf: list[float] | None
) -> tuple[float, float, float] | None:
    """3D paddle tip from the lifted keypoints (same elbow->wrist extension), or None."""
    if pose_world_ft is None:
        return None
    wrist = _strong_wrist(keypoint_conf)
    if wrist is None:
        return None
    elbow = L_ELBOW if wrist == L_WRIST else R_ELBOW
    w = np.array(pose_world_ft[wrist])
    e = np.array(pose_world_ft[elbow])
    tip = w + (w - e)
    return (float(tip[0]), float(tip[1]), float(max(tip[2], 0.0)))


def ground_xy_ft(court_xy: tuple[float, float]) -> tuple[float, float]:
    """Normalized court coords -> court feet (the billboard ground anchor)."""
    return (float(court_xy[0]) * WIDTH_FT, float(court_xy[1]) * LENGTH_FT)
