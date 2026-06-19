"""Lift 2D pose keypoints into 3D and derive a wrist-anchored paddle — monocular, camera-based.

A single camera can't recover per-keypoint depth, so we use a **billboard** ("cardboard cutout")
approximation: the player stands on the court at the ground point under their feet, in a vertical
plane that faces the camera. Each keypoint's pixel offset from the foot is mapped into that plane at
a scale fixed by the bounding-box pixel height and an assumed standing height (``PLAYER_HEIGHT_FT``)
— so the skeleton keeps its apparent shape and stands at a realistic height regardless of how
accurately the camera's focal length was recovered. (Back-projecting keypoint rays through the
recovered camera instead made every height collapse to ~0 whenever that focal estimate was off.)
Limbs reaching toward or away from the camera flatten onto the plane (the known approximation). When
the lift is degenerate (camera ~overhead) the caller falls back to a 2D-only skeleton / a stick.

The paddle is a wrist-anchored estimate (no paddle model): a short segment extending past the
stronger-confidence wrist along the elbow->wrist direction.

World frame matches :mod:`pbengine.ball.camera`: X across the 20 ft width, Y along the 44 ft
length, Z up; ground plane Z = 0.
"""

from __future__ import annotations

import numpy as np

from pbengine.ball.camera import CameraModel
from pbengine.detect.pose import L_ELBOW, L_WRIST, R_ELBOW, R_WRIST
from pbengine.court.court_model import LENGTH_FT, WIDTH_FT

_CONF_FLOOR = 0.3  # keypoints below this are unreliable; skipped in the lift / paddle
PLAYER_HEIGHT_FT = 5.75  # assumed standing height (~5'9", average player) for the billboard scale


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
