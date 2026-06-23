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
    _ARC_APEX_CAP_FT,
    _COURT_CLAMP_FT,
    _Obs,
    _fit_parabola,
    _height_at,
    _height_knots,
    _net_crossing_frames,
    _segment_bounds,
    ball_world_ft,
    densify_rally,
    fill_gaps_3d,
    reconstruct_3d,
    reconstruct_3d_segments,
)
from pbengine.court.court_model import LENGTH_FT, NET_CLEARANCE_FT, NET_HEIGHT_FT, WIDTH_FT
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


def test_ball_world_ft_no_camera_uses_ground_projection_and_arcs():
    fps = 30.0
    # Ball stays on side A (no net crossing) so the arc between the two bounces is a clean parabola.
    bounces = [
        Bounce(frame=30, court_xy=(0.5, 0.32), side=Team.A, in_bounds=True),
        Bounce(frame=60, court_xy=(0.5, 0.36), side=Team.A, in_bounds=True),
    ]
    traj = [BallSample(frame=f, px=(960.0, 540.0), court_xy=(0.5, 0.30 + 0.08 * (f - 10) / 70),
                       conf=1.0) for f in range(10, 81)]

    out = ball_world_ft(traj, bounces, None, start_frame=10, end_frame=80, fps=fps)

    # Every frame gets a sane, on-court 3D position (the old metric solve put Y at ~86 ft, Z at ~26).
    assert all(s.world_ft is not None for s in out)
    for s in out:
        x, y, z = s.world_ft
        assert -_COURT_CLAMP_FT <= x <= WIDTH_FT + _COURT_CLAMP_FT
        assert -_COURT_CLAMP_FT <= y <= LENGTH_FT + _COURT_CLAMP_FT
        assert 0.0 <= z <= _ARC_APEX_CAP_FT + 1e-6
    at = {s.frame: s.world_ft for s in out}
    # No camera -> horizontal is the raw ground projection of each sample's court_xy.
    assert abs(at[45][0] - 0.5 * WIDTH_FT) < 1e-6
    # Touches the floor at each bounce/anchor; arcs to the gravity apex midway between them.
    assert at[30][2] < 1e-6 and at[60][2] < 1e-6
    T = (60 - 30) / fps
    assert abs(at[45][2] - G_FT * T * T / 8.0) < 1e-6
    assert all(s.speed_mph is not None for s in out)


def test_height_model_clears_the_net_and_honors_contacts():
    fps = 30.0
    # Ball crosses the net (court_xy.y 0.3 -> 0.7) between two ground bounces.
    bounces = [Bounce(frame=10, court_xy=(0.5, 0.3), side=Team.A, in_bounds=True),
               Bounce(frame=70, court_xy=(0.5, 0.7), side=Team.B, in_bounds=True)]
    samples = [BallSample(frame=f, px=(0.0, 0.0), court_xy=(0.5, 0.30 + 0.40 * (f - 10) / 63),
                          conf=1.0) for f in range(10, 71)]
    knots = _height_knots(samples, bounces, None, 10, 70, fps)
    cross = _net_crossing_frames(samples)[0]
    # The natural bounce-to-bounce arc already clears the 36" net, so no hard pin is needed.
    assert _height_at(cross, knots, fps) >= NET_HEIGHT_FT
    # A contact knot pins the hitter's contact height.
    k2 = dict(_height_knots(samples, bounces, [(40, 4.0)], 10, 70, fps))
    assert abs(k2[40] - 4.0) < 1e-9
    assert abs(_height_at(40, sorted(k2.items()), fps) - 4.0) < 1e-9


def test_lob_over_net_is_not_flattened_to_net_height():
    fps = 30.0
    # A high contact (9 ft) arcing down to a bounce: the ball crosses the net well above the tape.
    bounces = [Bounce(frame=60, court_xy=(0.5, 0.7), side=Team.B, in_bounds=True)]
    samples = [BallSample(frame=f, px=(0.0, 0.0), court_xy=(0.5, 0.30 + 0.37 * (f - 20) / 40),
                          conf=1.0) for f in range(20, 61)]
    knots = _height_knots(samples, bounces, [(20, 9.0)], 20, 60, fps)
    cross = _net_crossing_frames(samples)[0]
    # The old code pinned every crossing to ~3.2 ft; the lob must keep its natural (much higher) arc.
    assert _height_at(cross, knots, fps) > NET_CLEARANCE_FT + 2.0
    assert all(abs(z - NET_CLEARANCE_FT) > 1e-9 for f, z in knots if f == cross)  # no net pin added


def test_low_ball_is_lifted_to_clear_the_net():
    fps = 30.0
    # Two Z=0 rally edges a few frames apart straddling the net: the natural arc barely rises, so the
    # net-crossing must be lifted to clearance so the ball doesn't pass through the tape.
    samples = [BallSample(frame=f, px=(0.0, 0.0), court_xy=(0.5, 0.30 + 0.37 * f / 4), conf=1.0)
               for f in range(0, 5)]
    knots = dict(_height_knots(samples, [], None, 0, 4, fps))
    cross = _net_crossing_frames(samples)[0]
    assert abs(knots[cross] - NET_CLEARANCE_FT) < 1e-9               # pinned up to clearance
    assert _height_at(cross, sorted(knots.items()), fps) >= NET_HEIGHT_FT


def test_ball_world_ft_with_camera_reprojects_to_the_2d_track():
    fps = 30.0
    cam = recover_camera(_homography_from_camera(_lookat_camera()), W, H)
    bounces = [Bounce(frame=10, court_xy=(0.5, 0.3), side=Team.A, in_bounds=True),
               Bounce(frame=40, court_xy=(0.5, 0.7), side=Team.B, in_bounds=True)]
    # Build samples first, derive the modeled-height knots, then synthesize pixels from an on-court arc
    # AT that modeled height; ball_world_ft must recover points that reproject to exactly those pixels.
    samples = [BallSample(frame=f, px=(0.0, 0.0), court_xy=(0.5, 0.3 + 0.4 * f / 50), conf=1.0)
               for f in range(0, 51)]
    knots = _height_knots(samples, bounces, None, 0, 50, fps)
    for s in samples:
        gt = np.array([8.0, 12.0 + 0.3 * s.frame, _height_at(s.frame, knots, fps)])
        px = cam.project(np.array([gt]))[0]
        s.px = (float(px[0]), float(px[1]))

    out = ball_world_ft(samples, bounces, cam, 0, 50, fps=fps)
    for s in out:
        assert s.world_ft is not None
        rp = cam.project(np.array([list(s.world_ft)]))[0]  # reprojects to its detection pixel
        assert abs(rp[0] - s.px[0]) < 1e-3 and abs(rp[1] - s.px[1]) < 1e-3
        assert abs(s.world_ft[2] - _height_at(s.frame, knots, fps)) < 1e-6  # height from the model
        assert -_COURT_CLAMP_FT <= s.world_ft[0] <= WIDTH_FT + _COURT_CLAMP_FT
        assert -_COURT_CLAMP_FT <= s.world_ft[1] <= LENGTH_FT + _COURT_CLAMP_FT


def test_ball_world_ft_caps_apex_on_a_long_gap():
    # A 3 s flight (anchors 3 s apart) would parabola to g*T^2/8 ~= 36 ft; the cap holds it down.
    bounces = [Bounce(frame=0, court_xy=(0.5, 0.2), side=Team.A, in_bounds=True),
               Bounce(frame=90, court_xy=(0.5, 0.8), side=Team.B, in_bounds=True)]
    traj = [BallSample(frame=f, px=(0.0, 0.0), court_xy=(0.5, 0.5), conf=0.0) for f in range(0, 91)]
    out = ball_world_ft(traj, bounces, None, 0, 90, fps=30.0)
    assert max(s.world_ft[2] for s in out) <= _ARC_APEX_CAP_FT + 1e-6
