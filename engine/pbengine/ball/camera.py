"""Recover a metric camera model from the court homography (single-camera, no calibration rig).

The court homography maps image pixels to the *flat* court plane, which is enough for top-down
positions but throws away height — a ball 8 ft in the air projects to a ground point 15 ft away.
To get true 3D we need the camera's pose, which we recover from the same homography plus the
known court size, using the classic single-plane calibration (Zhang): the homography columns are
``K [r1 r2 t]``, and the orthonormality of ``r1, r2`` pins down the focal length.

World frame is **feet**: ``X`` across the 20 ft width, ``Y`` along the 44 ft length, ``Z`` up
from the court (ground plane ``Z = 0``). The result is a full projection matrix ``P = K [R|t]``
mapping world feet -> source pixels, which :mod:`pbengine.ball.trajectory3d` inverts (with a
gravity prior) to lift the 2D ball track into 3D.

Single-camera 3D is inherently approximate: accuracy rides on the homography quality and the
focal-length estimate. ``reprojection_error_px`` is exposed so callers can gate on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbengine.court.court_model import LENGTH_FT, WIDTH_FT


@dataclass
class CameraModel:
    """A metric pinhole camera recovered from the court homography.

    ``P`` is the 3x4 projection matrix (world feet -> pixels). ``reprojection_error_px`` is the
    RMS error reprojecting the four court corners; large values mean the recovery is unreliable
    (bad homography or a near-degenerate viewing angle) and 3D output should not be trusted.
    """

    P: np.ndarray  # 3x4 projection matrix, world(ft) -> pixel
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 rotation, world -> camera
    t: np.ndarray  # 3, translation, world -> camera
    focal_px: float
    reprojection_error_px: float

    def project(self, world_ft: np.ndarray) -> np.ndarray:
        """Project Nx3 world points (feet) to Nx2 source pixels."""
        pts = np.atleast_2d(np.asarray(world_ft, dtype=float))
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
        proj = (self.P @ homog.T).T
        return proj[:, :2] / proj[:, 2:3]

    def depth(self, world_ft: np.ndarray) -> np.ndarray:
        """Camera-frame depth (Z) of Nx3 world points; positive means in front of the camera."""
        pts = np.atleast_2d(np.asarray(world_ft, dtype=float))
        return (self.R @ pts.T).T[:, 2] + self.t[2]


def _plane_to_pixel_homography(pixel_to_norm: np.ndarray) -> np.ndarray:
    """Build G mapping court points in **feet** (X, Y, 1) -> pixels.

    ``pixel_to_norm`` is the engine homography (pixel -> normalized [0,1] court). Its inverse
    maps normalized court -> pixel; pre-scaling by 1/width, 1/length turns feet into normalized
    first, so ``G @ [X, Y, 1] = pixel``.
    """
    norm_to_pixel = np.linalg.inv(pixel_to_norm)
    feet_to_norm = np.diag([1.0 / WIDTH_FT, 1.0 / LENGTH_FT, 1.0])
    return norm_to_pixel @ feet_to_norm


def recover_camera(
    pixel_to_norm: np.ndarray, width: int, height: int
) -> CameraModel:
    """Recover ``P = K [R|t]`` (world feet -> pixels) from the court homography.

    Assumes a pinhole camera with the principal point at the image center, zero skew, and square
    pixels (``fx = fy``) — the usual single-image assumptions. The focal length is solved from the
    orthonormality of the homography's rotation columns; if that is degenerate it falls back to
    ``f = width`` (~a moderate field of view).
    """
    g = _plane_to_pixel_homography(np.asarray(pixel_to_norm, dtype=float))
    cx, cy = width / 2.0, height / 2.0

    # Move the principal point to the origin so K reduces to diag(f, f, 1).
    t_pp = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    gp = t_pp @ g
    g1, g2 = gp[:, 0], gp[:, 1]

    # Solve for s = 1/f^2 from the two single-plane constraints (K0^-1 = diag(1/f, 1/f, 1)):
    #   r1 . r2 = 0      -> (g1x g2x + g1y g2y) s + g1z g2z       = 0
    #   |r1| = |r2|      -> (g1x^2+g1y^2 - g2x^2-g2y^2) s + (g1z^2 - g2z^2) = 0
    # Either can be degenerate for a given camera (e.g. an axis-aligned column), so combine both
    # in a least-squares — the well-conditioned one dominates.
    a = np.array([
        g1[0] * g2[0] + g1[1] * g2[1],
        g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2,
    ])
    rhs = np.array([-g1[2] * g2[2], g2[2] ** 2 - g1[2] ** 2])
    denom = float(a @ a)
    s = float(a @ rhs) / denom if denom > 1e-20 else -1.0
    focal = float(1.0 / np.sqrt(s)) if s > 1e-12 else float(width)

    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    k_inv = np.linalg.inv(k)

    # [r1 r2 t] = K^-1 G / lambda, with lambda fixed by ||r1|| = ||r2|| = 1.
    m = k_inv @ g
    lam = 2.0 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
    r1, r2, t = m[:, 0] * lam, m[:, 1] * lam, m[:, 2] * lam

    # The court must sit in front of the camera; flip the scale sign if it came out behind.
    center_depth = (np.array([r1, r2]).T @ np.array([WIDTH_FT / 2, LENGTH_FT / 2]))[2] + t[2]
    if center_depth < 0:
        r1, r2, t = -r1, -r2, -t

    r3 = np.cross(r1, r2)
    # Closest orthonormal rotation to [r1 r2 r3] via SVD (the columns are only approximately so).
    r_approx = np.column_stack([r1, r2, r3])
    u, _, vh = np.linalg.svd(r_approx)
    rot = u @ vh
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vh

    proj = k @ np.column_stack([rot, t])

    cam = CameraModel(
        P=proj, K=k, R=rot, t=t, focal_px=focal, reprojection_error_px=0.0
    )
    # Reprojection error on the four court corners: a health check on the whole recovery.
    corners_ft = np.array(
        [[0.0, 0.0, 0.0], [WIDTH_FT, 0.0, 0.0], [WIDTH_FT, LENGTH_FT, 0.0], [0.0, LENGTH_FT, 0.0]]
    )
    corners_norm = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    expected_px = (np.linalg.inv(pixel_to_norm) @ np.hstack(
        [corners_norm, np.ones((4, 1))]
    ).T).T
    expected_px = expected_px[:, :2] / expected_px[:, 2:3]
    got_px = cam.project(corners_ft)
    cam.reprojection_error_px = float(np.sqrt(((got_px - expected_px) ** 2).sum(axis=1).mean()))
    return cam
