"""Monocular 3D reconstruction: recover a known camera + lift a known parabola back to 3D.

We build a ground-truth camera, project the four court corners through it to synthesize a
homography (exactly what the court stage produces), then project a known ballistic arc to pixels.
The test asserts ``camera.recover_camera`` inverts the geometry and ``reconstruct_3d`` recovers the
3D positions and speed of the arc to within tight tolerances on noise-free input.
"""

from __future__ import annotations

import numpy as np

from pbengine.ball.camera import recover_camera
from pbengine.ball.trajectory3d import (
    G_FT,
    FT_PER_S_TO_MPH,
    _Obs,
    _fit_parabola,
    _segment_bounds,
    bridge_world_ft,
    densify_rally,
    fill_gaps_3d,
    reconstruct_3d,
    reconstruct_3d_segments,
)
from pbengine.court.court_model import LENGTH_FT, WIDTH_FT
from pbengine.court.homography import compute_homography
from pbengine.schema.models import BallSample, Bounce, Team

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


def _parabola_sample(p_gt, p0, v0, i, fps):
    t = i / fps
    pos = p0 + v0 * t + 0.5 * np.array([0, 0, -G_FT]) * t * t
    px = _project(p_gt, pos.reshape(1, 3))[0]
    return pos, BallSample(frame=i, px=(float(px[0]), float(px[1])), conf=1.0)


def test_fill_gaps_3d_recovers_deleted_frames():
    """Physics gap-fill: drop the odd frames, then interpolate them back to ~the true 3D point."""
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    fps = 30.0
    p0, v0 = np.array([4.0, 8.0, 1.5]), np.array([3.0, 18.0, 9.0])
    truth, kept = {}, []
    for i in range(20):
        pos, s = _parabola_sample(p_gt, p0, v0, i, fps)
        truth[i] = pos
        if i % 2 == 0:  # keep even frames; odd frames become gaps to fill
            kept.append(s)

    out = fill_gaps_3d(reconstruct_3d(kept, [], cam, fps), [], cam, fps, max_fill_gap=6)
    by = {s.frame: s for s in out}
    for i in range(3, 16, 2):  # interior odd frames now present, flagged, and accurate
        assert i in by and by[i].interpolated is True and by[i].world_ft is not None
        assert np.linalg.norm(np.array(by[i].world_ft) - truth[i]) < 0.7
    assert by[6].interpolated is False and by[6].conf > 0  # measured frames untouched


def test_fill_gaps_3d_respects_bounce_and_gap_limits():
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    fps = 30.0
    p0, v0 = np.array([4.0, 8.0, 1.5]), np.array([3.0, 18.0, 9.0])
    frames = [0, 2, 4, 6, 8, 20, 22]  # 8->20 is a 12-frame gap (wider than max_fill_gap=6)
    kept = [_parabola_sample(p_gt, p0, v0, i, fps)[1] for i in frames]
    bounces = [Bounce(frame=5, court_xy=(0.3, 0.3), side=Team.A, in_bounds=True)]

    out = fill_gaps_3d(reconstruct_3d(kept, bounces, cam, fps), bounces, cam, fps, max_fill_gap=6)
    fills = {s.frame for s in out if s.interpolated}
    assert all(f not in fills for f in range(9, 20))  # wide gap is never bridged
    assert 5 not in fills  # a gap containing a bounce (contact) is skipped


def test_segment_fit_rejects_outliers():
    """A robust per-segment fit must drop scattered false positives and still recover a clean arc."""
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    fps = 30.0
    p0, v0 = np.array([4.0, 8.0, 1.5]), np.array([3.0, 18.0, 9.0])

    truth, samples = {}, []
    for i in range(20):
        pos, s = _parabola_sample(p_gt, p0, v0, i, fps)
        truth[i] = pos
        samples.append(s)
    # Inject 3 gross false positives (off by ~300 px) at scattered frames.
    outlier_frames = {5, 11, 16}
    for f in outlier_frames:
        bad = samples[f]
        samples[f] = BallSample(frame=f, px=(bad.px[0] + 320.0, bad.px[1] - 280.0), conf=1.0)

    out, flagged = reconstruct_3d_segments(samples, bounces=[], camera=cam, fps=fps)
    assert outlier_frames <= flagged  # every injected outlier rejected
    by = {s.frame: s for s in out}
    for f in outlier_frames:
        assert by[f].world_ft is None  # outliers are not lifted to 3D
    # The surviving (true) frames still reconstruct the clean arc accurately.
    clean = [s for s in out if s.frame not in outlier_frames and s.world_ft is not None]
    assert len(clean) >= 14
    for s in clean:
        assert np.linalg.norm(np.array(s.world_ft) - truth[s.frame]) < 0.5


def test_segment_boundaries_land_at_bounces():
    """A provided bounce frame must split the track into two arcs (Z velocity reverses there)."""
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    fps = 30.0
    samples = [_parabola_sample(p_gt, np.array([4.0, 8.0, 1.5]), np.array([3.0, 18.0, 9.0]),
                                i, fps)[1] for i in range(16)]
    frames = np.array([s.frame for s in samples])
    segs = _segment_bounds(frames, samples, [8], cam.P, fps, max_gap=6, split_on_kinks=False)
    assert len(segs) == 2
    assert segs[0][1] == segs[1][0]  # contiguous split
    assert int(frames[segs[1][0]]) == 8  # the bounce frame starts the second arc


def test_paddle_hit_split():
    """A mid-air velocity discontinuity (paddle hit) with NO bounce must still split via the kink
    test — a single parabola cannot fit both arcs."""
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    fps = 30.0
    samples, all_obs = [], []
    p0a, v0a = np.array([4.0, 8.0, 2.0]), np.array([2.0, 16.0, 6.0])
    # Arc 2 continues from arc 1's position at the hit but reverses horizontal velocity.
    t_hit = 10 / fps
    p_hit = p0a + v0a * t_hit + 0.5 * np.array([0, 0, -G_FT]) * t_hit * t_hit
    v0b = np.array([2.0, -16.0, 7.0])
    for i in range(20):
        if i < 10:
            pos = p0a + v0a * (i / fps) + 0.5 * np.array([0, 0, -G_FT]) * (i / fps) ** 2
        else:
            tt = (i - 10) / fps
            pos = p_hit + v0b * tt + 0.5 * np.array([0, 0, -G_FT]) * tt * tt
        px = _project(p_gt, pos.reshape(1, 3))[0]
        samples.append(BallSample(frame=i, px=(float(px[0]), float(px[1])), conf=1.0))
        all_obs.append(_Obs(i / fps, float(px[0]), float(px[1]), 1.0))

    # A single whole-run parabola reprojects badly (the kink), but the split fixes it.
    whole = _fit_parabola(all_obs, [], cam.P)
    assert whole is not None and whole[1] > 12.0
    frames = np.array([s.frame for s in samples])
    segs = _segment_bounds(frames, samples, [], cam.P, fps, max_gap=6, split_on_kinks=True)
    assert len(segs) >= 2
    assert any(8 <= seg[1] <= 12 for seg in segs[:-1])  # a boundary lands at the hit


def test_build_points_culls_outlier_from_trajectory():
    """End-to-end through _build_points: a false-positive detection must not appear as a measured
    sample, and any backfill at that frame must be flagged interpolated."""
    from pbengine.pipeline import _build_points

    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    fps = 30.0
    p0, v0 = np.array([8.0, 10.0, 1.5]), np.array([1.0, 9.0, 7.0])
    ball = [_parabola_sample(p_gt, p0, v0, i, fps)[1] for i in range(40)]  # > min rally length
    bad = ball[20]
    ball[20] = BallSample(frame=20, px=(bad.px[0] + 320.0, bad.px[1] - 260.0), conf=1.0)

    points = _build_points(ball, fps, camera=cam)
    assert len(points) == 1
    by = {s.frame: s for s in points[0].ball_trajectory}
    assert 20 in by  # backfilled for continuity...
    assert by[20].interpolated is True  # ...but flagged a guess, never a measurement
    measured = [s for s in points[0].ball_trajectory if not s.interpolated]
    assert all(s.frame != 20 or s.interpolated for s in measured)


def test_no_camera_or_short_track_is_safe():
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    # Too few points -> stays 2D, no crash, fields remain None.
    short = [BallSample(frame=i, px=(900.0 + i, 500.0), conf=1.0) for i in range(2)]
    out = reconstruct_3d(short, bounces=[], camera=cam, fps=30.0)
    assert all(s.world_ft is None and s.speed_mph is None for s in out)
    assert reconstruct_3d([], bounces=[], camera=cam, fps=30.0) == []


def test_densify_rally_fills_every_frame_in_span():
    # A wide gap (frames 5..25 missing) that fill_gaps_3d would never bridge. densify must fill it.
    traj = [BallSample(frame=f, px=(100.0 + 10 * f, 200.0 + 5 * f),
                       court_xy=(0.1, 0.2), world_ft=(1.0 + f, 2.0, 3.0), speed_mph=20.0)
            for f in [0, 1, 2, 3, 4, 25, 26, 27, 28]]
    out = densify_rally(traj, start_frame=0, end_frame=28, bounces=[], camera=None, fps=30.0)
    frames = [s.frame for s in out]
    assert frames == list(range(29))  # no hole anywhere in the span
    fills = [s for s in out if s.interpolated]
    assert {s.frame for s in fills} == set(range(5, 25))
    # Linear interp carried px + world_ft + court_xy across the gap, flagged interpolated/conf 0.
    mid = next(s for s in out if s.frame == 15)
    assert mid.interpolated and mid.conf == 0.0
    assert mid.world_ft is not None and mid.court_xy is not None
    assert abs(mid.world_ft[0] - 16.0) < 1e-6  # linear between f4 (x=5) and f25 (x=26) at t=11/21
    # measured endpoints untouched
    assert not out[0].interpolated and not out[28].interpolated


def test_densify_rally_anchors_bounce_at_floor():
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    # Detections either side of a gap that contains a bounce at frame 10 (mid-court).
    fids = [4, 5, 6, 16, 17, 18]
    traj = []
    for f in fids:
        X, Y, Z = 8.0, 10.0 + 0.5 * f, 4.0
        px = cam.project(np.array([[X, Y, Z]]))[0]
        traj.append(BallSample(frame=f, px=(float(px[0]), float(px[1])),
                               court_xy=(X / WIDTH_FT, Y / LENGTH_FT), world_ft=(X, Y, Z)))
    bounces = [Bounce(frame=10, court_xy=(0.4, 0.4), side=Team.A, in_bounds=True)]
    out = densify_rally(traj, start_frame=4, end_frame=18, bounces=bounces, camera=cam, fps=30.0)
    assert [s.frame for s in out] == list(range(4, 19))  # complete
    bounce_s = next(s for s in out if s.frame == 10)
    assert bounce_s.world_ft is not None and abs(bounce_s.world_ft[2]) < 1e-6  # anchored to floor
    # The fill dips toward the floor around the bounce rather than straight-lining over it.
    z9 = next(s for s in out if s.frame == 9).world_ft[2]
    z11 = next(s for s in out if s.frame == 11).world_ft[2]
    assert z9 < 4.0 and z11 < 4.0


def test_bridge_world_ft_fills_2d_only_frames_on_their_rays():
    p_gt = _lookat_camera()
    cam = recover_camera(_homography_from_camera(p_gt), W, H)
    # A smooth 3D arc; every frame has a pixel detection, but a middle run (7..13) was left without a
    # 3D fit (world_ft None) and the last two frames (19, 20) are an edge run (3D only on the left).
    arc = [(8.0, 6.0 + 1.4 * f, 3.0 + 0.9 * f - 0.05 * f * f) for f in range(21)]
    traj = []
    for f, (X, Y, Z) in enumerate(arc):
        px = cam.project(np.array([[X, Y, max(Z, 0.0)]]))[0]
        has3d = not (7 <= f <= 13 or f >= 19)
        traj.append(BallSample(
            frame=f, px=(float(px[0]), float(px[1])), conf=1.0,
            world_ft=(X, Y, max(Z, 0.0)) if has3d else None))

    out = bridge_world_ft(traj, cam, fps=30.0)
    # Every frame now carries a 3D position; nothing blinks.
    assert all(s.world_ft is not None for s in out)
    # Each bridged point lies on its detection's ray -> reprojects exactly to the input pixel.
    for s_in, s_out in zip(traj, out):
        if s_in.world_ft is None:
            assert s_out.interpolated
            rp = cam.project(np.array([list(s_out.world_ft)]))[0]
            assert abs(rp[0] - s_in.px[0]) < 1e-3 and abs(rp[1] - s_in.px[1]) < 1e-3
    # Bracketed fills ride the chord between the two 3D knots (monotonic Y, between the endpoints).
    ys = [s.world_ft[1] for s in out if 7 <= s.frame <= 13]
    assert ys == sorted(ys) and out[6].world_ft[1] < ys[0] and ys[-1] < out[14].world_ft[1]
    # Measured 3D samples are untouched; bridged samples get a speed.
    assert not out[0].interpolated
    assert all(s.speed_mph is not None for s in out if s.interpolated)
