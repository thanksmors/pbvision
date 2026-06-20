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
from pbengine.detect.pose import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, N_KEYPOINTS, NOSE,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)
from pbengine.players.pose3d import (
    _BONE_FRAC,
    HIP_W_FRAC,
    PLAYER_HEIGHT_FT,
    SHOULDER_W_FRAC,
    billboard_lift,
    ground_xy_ft,
    lift_pose_3d,
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


# --- lift_pose_3d: real volumetric reconstruction --------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _build_3d_pose(gx, gy, yaw=0.0):
    """A known 3D COCO pose standing at (gx, gy), turned by ``yaw`` about the vertical (0 -> faces +Y).

    Built with anthropometric trunk widths (bi-iliac hips, biacromial shoulders) and exact bone
    lengths so the lift — which assumes those widths/lengths — can recover it. Genuine depth comes
    from the right arm reaching along the body-forward direction. ``lat`` is the L->R body axis and
    ``fwd`` the forward axis, both rotated by ``yaw``."""
    H = PLAYER_HEIGHT_FT
    bone = {k: f * H for k, f in _BONE_FRAC.items()}
    hip_half = HIP_W_FRAC * H / 2.0
    sh_half = SHOULDER_W_FRAC * H / 2.0
    up = np.array([0.0, 0.0, 1.0])
    lat = np.array([np.cos(yaw), np.sin(yaw), 0.0])   # L->R body axis
    fwd = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # body forward (yaw=0 -> +Y)
    splay = (sh_half - hip_half) / bone["torso"]      # outward lean to widen hips -> shoulders
    p = [None] * N_KEYPOINTS

    def place(parent, direction, length):
        return np.asarray(parent, dtype=float) + length * _unit(direction)

    # Legs bottom-up: ankles on the ground at bi-iliac stance, knees bent FORWARD, hips a touch back —
    # genuine, resolvable leg depth. The lateral (lat) stance is preserved up the leg (the bends are
    # along fwd), so hip width stays anthropometric.
    mid_ground = np.array([gx, gy, 0.0])
    p[L_ANKLE], p[R_ANKLE] = mid_ground - hip_half * lat, mid_ground + hip_half * lat
    p[L_KNEE] = place(p[L_ANKLE], 0.5 * fwd + up, bone["shank"])
    p[R_KNEE] = place(p[R_ANKLE], 0.5 * fwd + up, bone["shank"])
    p[L_HIP] = place(p[L_KNEE], -0.3 * fwd + up, bone["thigh"])
    p[R_HIP] = place(p[R_KNEE], -0.3 * fwd + up, bone["thigh"])
    # Shoulders a torso above, splayed outward along lat to biacromial width (exact bone length).
    p[L_SHOULDER] = place(p[L_HIP], -splay * lat + up, bone["torso"])
    p[R_SHOULDER] = place(p[R_HIP], splay * lat + up, bone["torso"])
    mid_sh = (p[L_SHOULDER] + p[R_SHOULDER]) / 2.0
    # Left arm hangs down; right arm reaches FORWARD (+fwd) and slightly down.
    p[L_ELBOW] = place(p[L_SHOULDER], -up, bone["uarm"])
    p[R_ELBOW] = place(p[R_SHOULDER], fwd - 0.5 * up, bone["uarm"])
    p[L_WRIST] = place(p[L_ELBOW], -up, bone["farm"])
    p[R_WRIST] = place(p[R_ELBOW], fwd - 0.3 * up, bone["farm"])
    p[NOSE] = place(mid_sh, up + 0.1 * fwd, bone["head"])
    for k in (1, 2, 3, 4):  # eyes/ears near the nose
        p[k] = p[NOSE] + np.array([0.1 * (k - 2.5), 0.0, 0.1])
    return p


def test_lift_pose_3d_recovers_depth_and_reprojects_exactly():
    cam = _camera()
    gx, gy = 9.0, 16.0  # side A (Y < 22) -> faces +Y
    truth = _build_3d_pose(gx, gy)
    px = [tuple(cam.project(np.array([t]))[0]) for t in truth]
    conf = [0.9] * N_KEYPOINTS
    court_xy = (gx / 20.0, gy / 44.0)

    lifted = lift_pose_3d(px, conf, court_xy, cam, forward_dir=(0.0, 1.0, 0.0))
    assert lifted is not None

    # Core property: limb joints lie exactly on their own ray -> reproject exactly to the input
    # pixel. The four trunk joints get a yaw refinement (orientation, not absolute depth) and so
    # reproject near-exactly (~1px) rather than bit-exact — the deliberate trade for stable depth.
    trunk = {L_SHOULDER, R_SHOULDER, L_HIP, R_HIP}
    reproj = cam.project(np.array(lifted))
    for k, ((ru, rv), (u, v)) in enumerate(zip(reproj, px)):
        tol = 2.0 if k in trunk else 1.0
        assert abs(ru - u) < tol and abs(rv - v) < tol

    # Feet on the ground; recovered pose close to truth (depth resolved, not flat).
    assert lifted[L_ANKLE][2] < 0.1 and lifted[R_ANKLE][2] < 0.1
    for k in (L_KNEE, R_KNEE, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER, R_ELBOW, R_WRIST, NOSE):
        assert np.linalg.norm(np.array(lifted[k]) - truth[k]) < 0.5

    # The right arm is recovered reaching FORWARD (+Y), not flattened/behind.
    fwd_reach = (np.array(lifted[R_WRIST]) - np.array(lifted[R_SHOULDER]))[1]
    assert fwd_reach > 0.3  # genuine forward depth

    # Not flat: joints span a real range of Y (depth), unlike a billboard (single Y plane).
    ys = [p[1] for p in lifted]
    assert max(ys) - min(ys) > 1.0


def test_lift_pose_3d_facing_resolves_front_back_ambiguity():
    cam = _camera()
    gx, gy = 9.0, 16.0
    truth = _build_3d_pose(gx, gy)
    px = [tuple(cam.project(np.array([t]))[0]) for t in truth]
    conf = [0.9] * N_KEYPOINTS
    court_xy = (gx / 20.0, gy / 44.0)

    # Correct facing (+Y) -> arm forward; flipped facing (-Y) -> the SAME pixels are resolved with
    # the arm going backward. This shows the facing assumption is what breaks the depth ambiguity.
    fwd = lift_pose_3d(px, conf, court_xy, cam, forward_dir=(0.0, 1.0, 0.0))
    bwd = lift_pose_3d(px, conf, court_xy, cam, forward_dir=(0.0, -1.0, 0.0))
    reach_fwd = (np.array(fwd[R_WRIST]) - np.array(fwd[R_SHOULDER]))[1]
    reach_bwd = (np.array(bwd[R_WRIST]) - np.array(bwd[R_SHOULDER]))[1]
    assert reach_fwd > 0.3 and reach_bwd < 0.0  # depth sign flips with facing


def test_lift_pose_3d_recovers_body_yaw_depth():
    """A player turned 40° about the vertical: the trunk must come out with real front/back depth
    (L/R shoulders and hips at different Y), not a flat coplanar cutout."""
    cam = _camera()
    gx, gy, yaw = 9.0, 16.0, np.radians(40.0)
    truth = _build_3d_pose(gx, gy, yaw=yaw)
    px = [tuple(cam.project(np.array([t]))[0]) for t in truth]
    conf = [0.9] * N_KEYPOINTS
    court_xy = (gx / 20.0, gy / 44.0)

    lifted = lift_pose_3d(px, conf, court_xy, cam, forward_dir=(0.0, 1.0, 0.0))
    assert lifted is not None

    # The shoulders (and hips) span real depth — turned body, not flat. Truth Y gap at 40° is
    # 2 * sh_half * sin(yaw); require the lift to recover a solid fraction of it.
    sh_half = SHOULDER_W_FRAC * PLAYER_HEIGHT_FT / 2.0
    want_gap = 2.0 * sh_half * np.sin(yaw)
    got_sh_gap = abs(lifted[R_SHOULDER][1] - lifted[L_SHOULDER][1])
    got_hip_gap = abs(lifted[R_HIP][1] - lifted[L_HIP][1])
    assert got_sh_gap > 0.5 * want_gap
    assert got_hip_gap > 0.3 * (2.0 * HIP_W_FRAC * PLAYER_HEIGHT_FT / 2.0 * np.sin(yaw))
    # Recovered trunk is close to the (turned) truth.
    for k in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        assert np.linalg.norm(np.array(lifted[k]) - truth[k]) < 0.6


def test_lift_pose_3d_none_on_singular_camera():
    cam = _camera()
    bad = CameraModel(P=np.zeros((3, 4)), K=cam.K, R=cam.R, t=cam.t, focal_px=F_PX,
                      reprojection_error_px=0.0)
    px = [(960.0, 540.0)] * N_KEYPOINTS
    assert lift_pose_3d(px, [0.9] * N_KEYPOINTS, (0.4, 0.4), bad, (0.0, 1.0, 0.0)) is None


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
