"""Monocular 3D reconstruction: recover a known camera + lift a known parabola back to 3D.

We build a ground-truth camera, project the four court corners through it to synthesize a
homography (exactly what the court stage produces), then project a known ballistic arc to pixels.
The test asserts ``camera.recover_camera`` inverts the geometry and ``reconstruct_3d`` recovers the
3D positions and speed of the arc to within tight tolerances on noise-free input.
"""

from __future__ import annotations

import numpy as np

from pbengine.ball.camera import recover_camera
from pbengine.ball.trajectory3d import G_FT, FT_PER_S_TO_MPH, reconstruct_3d
from pbengine.court.court_model import LENGTH_FT, WIDTH_FT
from pbengine.court.homography import compute_homography
from pbengine.schema.models import BallSample

W, H, F_PX = 1920, 1080, 1500.0


def _lookat_camera():
    """A camera behind the baseline, elevated, aimed at court centre -> P (feet -> px)."""
    cam_c = np.array([10.0, -15.0, 8.0])  # world position (ft)
    target = np.array([10.0, 22.0, 0.0])
    fwd = target - cam_c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.vstack([right, down, fwd])  # world -> camera
    t = -rot @ cam_c
    k = np.array([[F_PX, 0, W / 2], [0, F_PX, H / 2], [0, 0, 1]])
    return k @ np.column_stack([rot, t])


def _project(P, pts3d):
    homog = np.hstack([pts3d, np.ones((len(pts3d), 1))])
    proj = (P @ homog.T).T
    return proj[:, :2] / proj[:, 2:3]


def _homography_from_camera(P):
    corners_ft = np.array(
        [[0, 0, 0], [WIDTH_FT, 0, 0], [WIDTH_FT, LENGTH_FT, 0], [0, LENGTH_FT, 0]], dtype=float
    )
    corners_norm = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    px = _project(P, corners_ft)
    return compute_homography(px, corners_norm)  # pixel -> normalized court


def test_recover_camera_matches_ground_truth():
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    assert cam.reprojection_error_px < 1.0
    assert abs(cam.focal_px - F_PX) / F_PX < 0.02
    # A mid-air point must reproject to the same pixel under both cameras.
    pt = np.array([[12.0, 25.0, 6.0]])
    assert np.linalg.norm(_project(p_gt, pt) - cam.project(pt)) < 2.0


def test_ground_plane_reprojection_is_exact_even_off_center():
    """Corner (ground) reprojection must be ~0 for ANY valid homography, even when the real
    camera violates the principal-point-at-centre assumption — the regression behind the
    spurious "camera recovery unreliable (57px)" skip. The clicked homography is honoured exactly.
    """
    k = np.array([[1300.0, 0, W / 2 + 200], [0, 1300.0, H / 2 - 130], [0, 0, 1]])
    cam_c = np.array([3.0, -18.0, 6.0])
    fwd = np.array([12.0, 20.0, 0.0]) - cam_c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.vstack([right, down, fwd])
    p_gt = k @ np.column_stack([rot, -rot @ cam_c])

    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    assert cam.reprojection_error_px < 1.0  # ground plane is exact regardless of focal error


def test_reconstruct_recovers_parabola_position_and_speed():
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)

    fps = 30.0
    p0 = np.array([4.0, 8.0, 1.5])
    v0 = np.array([3.0, 18.0, 9.0])  # ft/s
    n = 20
    truth = []
    samples = []
    for i in range(n):
        t = i / fps
        pos = p0 + v0 * t + 0.5 * np.array([0, 0, -G_FT]) * t * t
        truth.append(pos)
        px = _project(p_gt, pos.reshape(1, 3))[0]
        samples.append(BallSample(frame=i, px=(float(px[0]), float(px[1])), conf=1.0))

    out = reconstruct_3d(samples, bounces=[], camera=cam, fps=fps)
    recovered = [(s.frame, s.world_ft, s.speed_mph) for s in out if s.world_ft is not None]
    assert len(recovered) >= n - 4  # interior frames should all reconstruct

    for frame, world, speed in recovered:
        t = frame / fps
        exp_pos = p0 + v0 * t + 0.5 * np.array([0, 0, -G_FT]) * t * t
        assert np.linalg.norm(np.array(world) - exp_pos) < 0.5  # within half a foot
        exp_speed = np.linalg.norm(v0 + np.array([0, 0, -G_FT]) * t) * FT_PER_S_TO_MPH
        assert abs(speed - exp_speed) < 0.10 * exp_speed + 0.5


def test_no_camera_or_short_track_is_safe():
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    # Too few points -> stays 2D, no crash, fields remain None.
    short = [BallSample(frame=i, px=(900.0 + i, 500.0), conf=1.0) for i in range(2)]
    out = reconstruct_3d(short, bounces=[], camera=cam, fps=30.0)
    assert all(s.world_ft is None and s.speed_mph is None for s in out)
    assert reconstruct_3d([], bounces=[], camera=cam, fps=30.0) == []
