"""Monocular pose lift: project a known 3D stick figure to pixels, lift it back, check recovery.

The lift scales the 2D pose into a vertical, camera-facing plane at the foot point using the bbox
pixel height + an assumed standing height (so heights are robust to focal-length error). We build a
ground-truth figure standing in that plane, project it through a consistent camera, derive its bbox
from the projected pixels, lift it back, and assert it recovers a standing skeleton: ankles on the
ground (Z≈0), head near the assumed height, foot at the court point, left/right preserved.
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


# (lateral ft, height ft) per COCO index: feet at Z=0, head ~5.7 ft, arms spread. Left keypoints
# (odd indices) are at negative lateral, right (even) at positive, matching image left/right.
_FIGURE_LH = [
    (0.0, 5.6),                       # 0 nose
    (-0.2, 5.7), (0.2, 5.7),          # 1/2 eyes
    (-0.4, 5.6), (0.4, 5.6),          # 3/4 ears
    (-0.7, 4.8), (0.7, 4.8),          # 5/6 shoulders
    (-1.2, 4.0), (1.2, 4.0),          # 7/8 elbows
    (-1.6, 3.2), (1.6, 3.2),          # 9/10 wrists
    (-0.5, 3.0), (0.5, 3.0),          # 11/12 hips
    (-0.5, 1.5), (0.5, 1.5),          # 13/14 knees
    (-0.5, 0.0), (0.5, 0.0),          # 15/16 ankles
]
_FIGURE_HT = 5.7  # head height of the figure above the ground


def _figure_in_billboard_plane(cam, gx, gy):
    """Build a COCO-17 figure standing at ground (gx, gy) in the camera-facing vertical plane.

    Uses the same lateral axis the lift uses (camera right, horizontalized) so a consistent camera
    round-trips the figure back to (approximately) itself.
    """
    ground = np.array([gx, gy, 0.0])
    side = cam.R.T @ np.array([1.0, 0.0, 0.0])
    side[2] = 0.0
    side /= np.linalg.norm(side)
    up = np.array([0.0, 0.0, 1.0])
    return [ground + lat * side + h * up for lat, h in _FIGURE_LH]


def test_billboard_lift_recovers_standing_figure():
    cam = _camera()
    gx, gy = 8.0, 18.0
    truth = _figure_in_billboard_plane(cam, gx, gy)
    px = [tuple(cam.project(np.array([p]))[0]) for p in truth]
    conf = [0.9] * N_KEYPOINTS
    # Bbox from the projected pixels (what the detector would produce), height anchored to ~the
    # figure's true height so the recovered scale matches the truth.
    xs, ys = [p[0] for p in px], [p[1] for p in px]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    lifted = billboard_lift(px, conf, ground_xy_ft((gx / 20.0, gy / 44.0)), cam,
                            bbox_px=bbox, player_height_ft=_FIGURE_HT)
    assert lifted is not None
    # Ankles on the court; head near the figure height; standing (not collapsed).
    assert lifted[15][2] < 0.1 and lifted[16][2] < 0.1
    assert abs(lifted[0][2] - _FIGURE_HT) < 0.4
    assert max(p[2] for p in lifted) > 4.0
    # Foot (avg ankles) sits at the court ground point.
    foot = (np.array(lifted[15]) + np.array(lifted[16])) / 2
    assert abs(foot[0] - gx) < 0.3 and abs(foot[1] - gy) < 0.3
    # Heights monotonic head -> shoulder -> hip -> knee -> ankle, and full recovery near truth.
    assert lifted[0][2] > lifted[5][2] > lifted[11][2] > lifted[13][2] > lifted[15][2]
    for got, want in zip(lifted, truth):
        assert np.linalg.norm(np.array(got) - want) < 0.4
    # Left wrist (9, lateral -1.6) and right wrist (10, +1.6) on opposite sides along the plane's
    # lateral axis (camera right), i.e. orientation preserved.
    side = cam.R.T @ np.array([1.0, 0.0, 0.0]); side[2] = 0.0; side /= np.linalg.norm(side)
    assert np.dot(np.array(lifted[9]) - foot, side) < 0 < np.dot(np.array(lifted[10]) - foot, side)


def test_billboard_lift_none_when_camera_overhead():
    cam = _camera()
    # A straight-down camera (R = identity-ish with no horizontal right axis) -> degenerate facing.
    overhead = CameraModel(P=cam.P, K=cam.K, R=np.array([[0.0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
                           t=cam.t, focal_px=F_PX, reprojection_error_px=0.0)
    px = [(960.0, 540.0)] * N_KEYPOINTS
    assert billboard_lift(px, [0.9] * N_KEYPOINTS, (8.0, 18.0), overhead, bbox_px=(0, 0, 50, 120)) is None


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
