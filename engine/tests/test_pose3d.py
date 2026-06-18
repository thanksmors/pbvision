"""Monocular pose lift: project a known 3D stick figure to pixels, lift it back, check recovery.

The billboard lift is exact when the true figure lies in the vertical plane through the foot point
that faces the camera (the model's assumption), so we build the ground-truth figure in that plane
and assert recovery to within a tight tolerance — plus that the ankles land on the ground (Z≈0).
We hand-build a camera whose ``P`` is exactly consistent with ``(K, R, t)`` so the lift inverts the
projection precisely (``recover_camera`` orthonormalizes R, which would add small inconsistency).
"""

from __future__ import annotations

import numpy as np

from pbengine.ball.camera import CameraModel
from pbengine.detect.pose import L_ELBOW, L_WRIST, N_KEYPOINTS, R_ELBOW, R_WRIST
from pbengine.players.pose3d import (
    billboard_lift,
    ground_xy_ft,
    paddle_segment_px,
    paddle_tip_world,
)

W, H, F_PX = 1920, 1080, 1500.0


def _camera():
    """A consistent pinhole camera (P == K[R|t]) behind the baseline, looking at court centre."""
    cam_c = np.array([10.0, -15.0, 8.0])
    fwd = np.array([10.0, 22.0, 0.0]) - cam_c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.vstack([right, down, fwd])  # world -> camera
    t = -rot @ cam_c
    k = np.array([[F_PX, 0, W / 2], [0, F_PX, H / 2], [0, 0, 1]])
    p = k @ np.column_stack([rot, t])
    return CameraModel(P=p, K=k, R=rot, t=t, focal_px=F_PX, reprojection_error_px=0.0)


def _figure_in_billboard_plane(cam, gx, gy):
    """Build a COCO-17 figure standing at ground (gx, gy) in the camera-facing vertical plane."""
    ground = np.array([gx, gy, 0.0])
    cam_c = -cam.R.T @ cam.t
    n = cam_c - ground
    n[2] = 0.0
    n /= np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(up, n)
    side /= np.linalg.norm(side)
    # (lateral ft, height ft) per COCO index: feet at Z=0, head ~5.8 ft, arms spread.
    lh = [
        (0.0, 5.6),                       # nose
        (-0.2, 5.7), (0.2, 5.7),          # eyes
        (-0.4, 5.6), (0.4, 5.6),          # ears
        (-0.7, 4.8), (0.7, 4.8),          # shoulders
        (-1.2, 4.0), (1.2, 4.0),          # elbows
        (-1.6, 3.2), (1.6, 3.2),          # wrists
        (-0.5, 3.0), (0.5, 3.0),          # hips
        (-0.5, 1.5), (0.5, 1.5),          # knees
        (-0.5, 0.0), (0.5, 0.0),          # ankles
    ]
    return [ground + lat * side + h * up for lat, h in lh]


def test_billboard_lift_recovers_in_plane_figure():
    cam = _camera()
    gx, gy = 8.0, 18.0
    truth = _figure_in_billboard_plane(cam, gx, gy)
    px = [tuple(cam.project(np.array([p]))[0]) for p in truth]
    conf = [0.9] * N_KEYPOINTS

    lifted = billboard_lift(px, conf, ground_xy_ft((gx / 20.0, gy / 44.0)), cam)
    assert lifted is not None
    for got, want in zip(lifted, truth):
        assert np.linalg.norm(np.array(got) - want) < 0.05  # within ~half an inch
    # Ankles sit on the court.
    assert lifted[15][2] < 0.05 and lifted[16][2] < 0.05


def test_paddle_segment_extends_past_wrist():
    kp = [(0.0, 0.0)] * N_KEYPOINTS
    kp[R_ELBOW] = (100.0, 100.0)
    kp[R_WRIST] = (120.0, 90.0)
    conf = [0.9] * N_KEYPOINTS
    seg = paddle_segment_px(kp, conf)
    # base = wrist, tip = wrist + (wrist - elbow)
    assert seg == (120.0, 90.0, 140.0, 80.0)


def test_paddle_picks_stronger_wrist_and_handles_low_conf():
    kp = [(0.0, 0.0)] * N_KEYPOINTS
    kp[L_ELBOW] = (50.0, 50.0)
    kp[L_WRIST] = (60.0, 40.0)
    conf = [0.0] * N_KEYPOINTS
    conf[L_WRIST] = 0.8  # left wrist strong, right weak -> pick left
    seg = paddle_segment_px(kp, conf)
    assert seg == (60.0, 40.0, 70.0, 30.0)
    # Both wrists weak -> no paddle.
    assert paddle_segment_px(kp, [0.0] * N_KEYPOINTS) is None


def test_paddle_tip_world_extends_in_3d():
    pose_world = [(0.0, 0.0, 0.0)] * N_KEYPOINTS
    pose_world[R_ELBOW] = (5.0, 20.0, 3.0)
    pose_world[R_WRIST] = (5.5, 20.0, 3.4)
    tip = paddle_tip_world(pose_world, [0.9] * N_KEYPOINTS)
    assert tip is not None
    assert np.allclose(tip, (6.0, 20.0, 3.8))
    assert paddle_tip_world(None, [0.9] * N_KEYPOINTS) is None
