"""Lift 2D pose keypoints into 3D and derive a wrist-anchored paddle — monocular, camera-based.

A single camera can't recover per-keypoint depth, so we use a **billboard** ("cardboard cutout")
approximation: the player stands on the court at the ground point under their feet, in a vertical
plane that faces the camera. Each keypoint pixel is back-projected onto that plane. This keeps the
skeleton's apparent shape and puts it at the right court position and height; limbs reaching toward
or away from the camera flatten onto the plane (the known approximation). When the lift is
degenerate (bad geometry) the caller falls back to a 2D-only skeleton / a vertical stick in 3D.

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


def _camera_center(camera: CameraModel) -> np.ndarray:
    """World position (ft) of the camera centre: C = -R^T t."""
    return -camera.R.T @ camera.t


def _pixel_ray(camera: CameraModel, u: float, v: float) -> np.ndarray:
    """World-space direction of the ray through pixel (u, v): d = R^T K^-1 [u, v, 1]."""
    return camera.R.T @ (np.linalg.inv(camera.K) @ np.array([u, v, 1.0]))


def billboard_lift(
    keypoints_px: list[tuple[float, float]],
    keypoint_conf: list[float] | None,
    ground_xy_ft: tuple[float, float],
    camera: CameraModel,
) -> list[tuple[float, float, float]] | None:
    """Lift COCO keypoints to 3D court feet via the billboard plane at ``ground_xy_ft``.

    ``ground_xy_ft`` is the player's foot position on the court (Z=0), already projected from the
    homography. Returns one (X, Y, Z) per keypoint (Z clamped >= 0), or ``None`` if the geometry is
    degenerate (camera roughly in the billboard plane). Low-confidence keypoints are still lifted
    (callers gate drawing on ``pose_conf``); only NaN/degenerate solves are dropped to a fallback.
    """
    cam_c = _camera_center(camera)
    gx, gy = float(ground_xy_ft[0]), float(ground_xy_ft[1])
    ground = np.array([gx, gy, 0.0])

    # Vertical billboard plane through the ground point, facing the camera (horizontal normal).
    normal = cam_c - ground
    normal[2] = 0.0
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-6:
        return None  # camera directly above the player -> no facing direction
    normal /= n_norm

    out: list[tuple[float, float, float]] = []
    for u, v in keypoints_px:
        d = _pixel_ray(camera, u, v)
        denom = float(normal @ d)
        if abs(denom) < 1e-9:
            return None  # ray parallel to the plane -> degenerate
        s = float(normal @ (ground - cam_c)) / denom
        if s <= 0:
            return None  # intersection behind the camera
        p = cam_c + s * d
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
