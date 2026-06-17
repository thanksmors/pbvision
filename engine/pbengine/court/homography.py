"""Homography estimation and point projection (pure NumPy, no OpenCV dependency).

The court detector yields named keypoints in pixel space. We match them to the normalized
reference template in :mod:`pbengine.court.court_model` and solve a 3x3 homography with the
normalized Direct Linear Transform (DLT). With a static camera this is solved once per shot
and reused for every frame.
"""

from __future__ import annotations

import numpy as np

from pbengine.court.court_model import REFERENCE_POINTS


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley normalization: translate to centroid, scale so mean distance is sqrt(2)."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted**2).sum(axis=1)).mean()
    scale = np.sqrt(2) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )


def compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve ``H`` mapping ``src`` (Nx2) to ``dst`` (Nx2) with the normalized DLT.

    Requires at least 4 non-collinear correspondences. Returns a 3x3 matrix normalized so
    ``H[2, 2] == 1``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need >= 4 matched point pairs of equal shape")

    t_src = _normalization_matrix(src)
    t_dst = _normalization_matrix(dst)
    src_n = _apply(t_src, src)
    dst_n = _apply(t_dst, dst)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows))
    h_norm = vh[-1].reshape(3, 3)

    # Denormalize: H = inv(T_dst) @ H_norm @ T_src
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    return h / h[2, 2]


def project(homography: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to ``pts`` (Nx2 or 2,) -> (Nx2) in the destination plane."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (homography @ homog.T).T
    return out[:, :2] / out[:, 2:3]


def _apply(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return project(matrix, pts)


def homography_from_named_points(named_px: dict[str, tuple[float, float]]) -> np.ndarray:
    """Solve the court homography from named pixel keypoints.

    ``named_px`` maps reference-point names (see ``court_model.REFERENCE_POINTS``) to their
    detected pixel locations. Only names present in both the input and the reference template
    are used, so a detector that finds a subset of points still works (>= 4 required).
    """
    names = [n for n in named_px if n in REFERENCE_POINTS]
    if len(names) < 4:
        raise ValueError(f"need >= 4 known court points, got {len(names)}: {names}")
    src = np.array([named_px[n] for n in names], dtype=float)
    dst = np.array([REFERENCE_POINTS[n] for n in names], dtype=float)
    return compute_homography(src, dst)
