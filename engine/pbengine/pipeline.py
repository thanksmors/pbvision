"""End-to-end match analysis: video in -> ``MatchResult`` JSON out.

Orchestrates the stages in the order from the build plan:

    probe -> court homography -> player tracking -> ball tracking
          -> rally segmentation -> per-rally serve / bounce / winner

Each stage's heavy model is loaded lazily inside its module, so importing this file is cheap
and the pure-logic stages stay testable without the ``ml`` extra. Run it standalone on the
rented GPU box::

    python -m pbengine.pipeline match.mp4 -o result.json
"""

from __future__ import annotations

import argparse
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from pbengine.ball.tracker import BallTracker
from pbengine.bounce.heuristic import detect_bounces
from pbengine.court.court_model import LENGTH_FT, WIDTH_FT, side_of
from pbengine.court.detector import CourtDetector
from pbengine.detect.players import PlayerDetector
from pbengine.errors import CourtNotFound, ModelUnavailable
from pbengine.rally.segmentation import segment_rallies
from pbengine.rally.serve import detect_serve
from pbengine.rally.winner import determine_winner
from pbengine.schema.models import (
    BallSample,
    CourtModel,
    EngineInfo,
    JobStatus,
    MatchResult,
    Player,
    PlayerPosition,
    Point,
    Team,
)
from pbengine.io.video import iter_frames, probe

StatusCb = Callable[[str, float], None]

ENGINE_VERSION = "0.1.0"
_MODELS = {"detector": "yolo11m-pose", "ball": "wasb-tennis", "court": "tenniscourtdetector"}


def analyze_match(
    video_path: str | Path,
    out_path: str | Path | None = None,
    status_cb: StatusCb | None = None,
    stride: int = 1,
    *,
    court_detector: CourtDetector | None = None,
    player_detector: PlayerDetector | None = None,
    ball_tracker: BallTracker | None = None,
) -> MatchResult:
    """Run the full pipeline and (optionally) write ``result.json``.

    The three model-backed stages are injectable so tests and the fixture demo can supply
    scripted stand-ins (see :mod:`pbengine.fixtures`) without the ``ml`` extra. They default
    to the real, lazily-loaded detectors.
    """
    video_path = Path(video_path)
    report = status_cb or (lambda stage, pct: None)
    court_detector = court_detector or CourtDetector()
    player_detector = player_detector or PlayerDetector()
    ball_tracker = ball_tracker or BallTracker()
    warnings: list[str] = []

    report("probe", 0.02)
    meta = probe(video_path)
    print(f"video: {meta.width}x{meta.height} @ {meta.fps:.0f}fps · {meta.frames} frames "
          f"(~{meta.frames / meta.fps / 60:.1f} min)", flush=True)

    # Each model-backed stage degrades gracefully: if its weights/deps are missing the stage is
    # skipped (empty output + a warning) so the engine can run one model at a time.
    report("court", 0.1)
    court_model, homography = _skip_if_unavailable(
        "court", warnings, (None, None), _solve_court, video_path, court_detector,
        exc_types=(ModelUnavailable, CourtNotFound),
    )

    # Recover a metric camera from the court homography (needs only homography + dims). Recovered
    # before players so pose keypoints can be lifted to 3D; also used to lift the ball track.
    # Optional: if it fails or the corner reprojection is poor, output keeps its 2D coords.
    camera = None
    if homography is not None:
        camera = _skip_if_unavailable(
            "camera_3d", warnings, None, _recover_camera, homography, meta.width, meta.height,
            exc_types=(Exception,),
        )

    report("players", 0.3)
    players = _skip_if_unavailable(
        "players", warnings, [], _track_players, video_path, homography, player_detector, camera,
        meta.fps, report, meta.frames,
    )

    report("ball", 0.6)
    # Throttle the ball tracker's per-frame callback into the ball span (0.60–0.85) + a logged ETA.
    _ball = {"t0": time.monotonic(), "last": 0.0}

    def ball_prog(phase: str, done: int, total: int) -> None:
        now = time.monotonic()
        if now - _ball["last"] < 2.0:
            return
        _ball["last"] = now
        el = now - _ball["t0"]
        rate = done / el if el > 0 else 0.0
        if total:
            eta = (total - done) / rate if rate > 0 else 0.0
            report(f"ball {done}/{total} ({100 * done // total}%) {rate:.1f}fps ETA "
                   f"{int(eta) // 60}:{int(eta) % 60:02d}", 0.60 + 0.25 * (done / total))
        else:
            report(f"ball {done} frames {rate:.1f}fps", 0.62)

    ball = _skip_if_unavailable(
        "ball", warnings, [], ball_tracker.track, video_path, homography=homography, stride=stride,
        progress=ball_prog,
    )
    # Ball diagnostics: coverage, inter-detection speed, and gap structure -> run.log, so a real run
    # shows whether fast balls are missed by the CNN (low coverage + big gaps) vs killed downstream.
    try:
        from pbengine.ball.diag import coverage_report

        gate = ball_tracker._gate_px(meta.width, meta.height)
        # When a court was solved, a None court_xy means the ground projection was discarded as an
        # off-court outlier (airborne/vanishing-line); report how many so the effect is visible.
        outliers = sum(1 for s in ball if s.court_xy is None) if homography is not None else None
        focal = getattr(camera, "focal_px", None)
        coverage_report(ball, meta.frames, meta.fps, gate_px=gate, court_outliers=outliers,
                        focal_px=focal)
    except Exception as exc:  # diagnostics must never break a run
        print(f"ball diag: skipped ({exc})", flush=True)

    report("rallies", 0.85)
    points = _build_points(ball, meta.fps, camera, players)

    result = MatchResult(
        match_id=str(uuid.uuid4()),
        video=meta,
        court=court_model,
        points=points,
        players=players,
        engine=EngineInfo(version=ENGINE_VERSION, models=_MODELS),
        warnings=warnings,
    )

    if out_path is not None:
        Path(out_path).write_text(result.model_dump_json(indent=2))
    report("done", 1.0)
    return result


def _skip_if_unavailable(stage, warnings, fallback, fn, *args, exc_types=(ModelUnavailable,), **kwargs):
    """Run ``fn``; if it reports a skippable condition, record a warning and return ``fallback``."""
    try:
        return fn(*args, **kwargs)
    except exc_types as exc:
        warnings.append(f"{stage}: {exc}")
        return fallback


def _solve_court(
    video_path: Path, detector: CourtDetector
) -> tuple[CourtModel | None, np.ndarray | None]:
    """Detect court corners on the first frame and solve a single homography for the shot."""
    from pbengine.court.homography import homography_from_named_points, project

    for _idx, frame in iter_frames(video_path):
        named = detector.detect(frame)
        if len(named) < 4:
            raise CourtNotFound(f"only {len(named)}/4 court corners localized")
        homography = homography_from_named_points(named)
        # Project the four canonical court corners back to pixels so the viewer can draw the
        # full court — including corners that are off-screen (extrapolated from the homography).
        inv = np.linalg.inv(homography)
        quad = [tuple(map(float, project(inv, c)[0])) for c in ((0, 0), (1, 0), (1, 1), (0, 1))]
        return (
            CourtModel(
                homography=homography.tolist(),
                keypoints_px=list(named.values()),
                court_quad_px=quad,
            ),
            homography,
        )
    return None, None


def _log_raw_tracks(raw_by_id, side_votes, homography) -> None:
    """Log the raw, pre-stitch ByteTrack roster so a real run shows what detection actually saw.

    This is the missing top of the player funnel: the roster log (players/select.py) only fires after
    stitching + capping, so it can't tell "the 4th player was never detected" from "stitch merged two
    real players" from "the per-side cap dropped one". This prints, per raw track id, its presence,
    span, court-position median, in-bounds fraction, and side-vote tally.
    """
    from statistics import median

    from pbengine.court.court_model import is_in_bounds

    if homography is None:
        print(f"DETECT: {len(raw_by_id)} raw tracks · homography MISSING "
              "(no court sides; on-court selection is skipped, every track is kept)", flush=True)
        return
    print(f"DETECT: {len(raw_by_id)} raw tracks · homography present", flush=True)
    for tid in sorted(raw_by_id):
        recs = raw_by_id[tid]
        frames = [r["frame"] for r in recs]
        xs = [r["court_xy"][0] for r in recs]
        ys = [r["court_xy"][1] for r in recs]
        inb = sum(1 for x, y in zip(xs, ys) if is_in_bounds((x, y), 0.35))
        votes = side_votes.get(tid, [])
        va = votes.count("A")
        vb = votes.count("B")
        print(f"  raw #{tid} frames={len(recs)} span=[{min(frames)}-{max(frames)}] "
              f"median=({median(xs):.2f},{median(ys):.2f}) inbounds={inb}/{len(recs)} "
              f"votes(A={va} B={vb})", flush=True)


def _track_players(
    video_path: Path, homography: np.ndarray | None, detector: PlayerDetector, camera=None,
    fps: float = 30.0, status_cb: StatusCb | None = None, total_frames: int | None = None,
) -> list[Player]:
    """Track players, project feet to court coords, lift pose to real 3D, assign teams by side."""
    from pbengine.court.homography import project
    from pbengine.players.interpolate import interpolate_positions
    from pbengine.players.pose3d import (
        billboard_lift,
        ground_xy_ft,
        lift_pose_3d,
        paddle_segment_px,
        paddle_tip_world,
    )

    # First pass: gather raw per-frame detections per track + side votes (the 3D solve needs each
    # track's stable forward direction and runs in frame order with temporal state, so it can't be
    # done in this interleaved loop).
    # Advance the players span (0.30–0.60) as frames stream so the API bar moves and status shows
    # the live frame count instead of parking at 30%.
    prog = None
    if status_cb is not None:
        prog = lambda done, total: status_cb(  # noqa: E731
            f"players {done}/{total}", 0.30 + 0.30 * (done / total if total else 0))
    tracks = detector.track(video_path, total_frames=total_frames, progress_cb=prog)
    raw_by_id: dict[int, list[dict]] = defaultdict(list)
    side_votes: dict[int, list[str]] = defaultdict(list)
    for t in tracks:
        court_xy = (0.0, 0.0)
        if homography is not None:
            court_xy = tuple(project(homography, t.foot_px)[0])
        raw_by_id[t.track_id].append(
            {"frame": t.frame, "court_xy": court_xy, "bbox_px": t.bbox_px,
             "kpx": t.keypoints_px, "kconf": t.keypoint_conf}
        )
        if homography is not None:
            side_votes[t.track_id].append(side_of(court_xy))

    # DETECT diagnostics: the raw, pre-stitch roster. Logged so a real run shows whether ByteTrack
    # ever saw a 4th distinct person (vs. it being lost later to stitching or the per-side cap).
    # Diagnostics must never be able to abort a run, so failures are swallowed (they'd otherwise
    # propagate past _skip_if_unavailable, which only tolerates ModelUnavailable).
    try:
        _log_raw_tracks(raw_by_id, side_votes, homography)
    except Exception as exc:
        print(f"DETECT diag: skipped ({exc})", flush=True)

    # Merge ByteTrack fragments of the same physical player (occlusion -> new id) so a player stays
    # one continuous person; the wider bridge cap below then glides the pose across the occlusion.
    from pbengine.players.stitch import max_stitch_frames, stitch_tracks

    groups = stitch_tracks(raw_by_id, side_votes, fps)

    players: list[Player] = []
    for group in groups:
        recs = sorted((r for tid in group for r in raw_by_id[tid]), key=lambda r: r["frame"])
        votes = [v for tid in group for v in side_votes.get(tid, [])]
        team = Team(max(set(votes), key=votes.count)) if votes else None
        # Players face their opponents across the net: side A (baseline Y=0) faces +Y, side B faces
        # -Y. This known body-forward breaks the monocular front/back depth ambiguity in lift_pose_3d.
        forward = (0.0, -1.0, 0.0) if team == Team.B else (0.0, 1.0, 0.0)

        prev_pose = None  # previous frame's solved 3D pose -> temporal depth-sign consistency
        positions: list[PlayerPosition] = []
        for r in recs:
            kpx, kconf, court_xy = r["kpx"], r["kconf"], r["court_xy"]
            pose_world = paddle_px = paddle_world = None
            if kpx is not None:
                paddle_px = paddle_segment_px(kpx, kconf)
                if camera is not None and homography is not None:
                    pose_world = lift_pose_3d(kpx, kconf, court_xy, camera, forward,
                                              prev_pose=prev_pose)
                    if pose_world is not None:
                        prev_pose = pose_world  # only seed from real solves, not the flat fallback
                    else:
                        pose_world = billboard_lift(kpx, kconf, ground_xy_ft(court_xy), camera,
                                                    bbox_px=r["bbox_px"])
                    paddle_world = paddle_tip_world(pose_world, kconf)
            positions.append(
                PlayerPosition(
                    frame=r["frame"], court_xy=court_xy, bbox_px=r["bbox_px"],
                    pose_px=kpx, pose_conf=kconf, pose_world_ft=pose_world,
                    paddle_px=paddle_px, paddle_world_ft=paddle_world,
                )
            )
        # Bridge dropouts (incl. the stitched occlusion) so the skeleton glides through rather than
        # blinking; the cap matches the stitch window so a genuine long absence is still left a hole.
        positions = interpolate_positions(positions, fps, max_gap_frames=max_stitch_frames(fps))
        players.append(Player(track_id=min(group), team=team, positions=positions))
    # Drop background people (refs, spectators, adjacent court) and cap to the doubles roster, so the
    # overlay shows the 4 players — not every person ByteTrack saw. Needs a homography for court sides.
    if homography is not None:
        from pbengine.players.select import select_on_court_players

        players = select_on_court_players(players)
    return players


def _recover_camera(homography: np.ndarray, width: int, height: int, max_reproj_px: float = 25.0):
    """Recover the metric camera, returning ``None`` if the corner reprojection is too poor."""
    from pbengine.ball.camera import recover_camera

    cam = recover_camera(homography, width, height)
    if cam.reprojection_error_px > max_reproj_px:
        raise CourtNotFound(
            f"camera recovery unreliable ({cam.reprojection_error_px:.0f}px corner error); "
            "3D ball trajectory skipped"
        )
    return cam


_OFFC = 4.0  # off-court slack (ft) for the diagnostic count; matches trajectory3d._COURT_CLAMP_FT


def _log_rally_diag(i, span, measured, bounces, traj, camera, shots=()) -> None:
    """Per-rally diagnostics to run.log so the 3D-ball approach can be judged by the numbers."""
    def _rng(vals):
        return f"[{min(vals):.1f},{max(vals):.1f}]" if vals else "[]"

    cxs = [s.court_xy[0] for s in measured if s.court_xy]
    cys = [s.court_xy[1] for s in measured if s.court_xy]
    wf = [s.world_ft for s in traj if s.world_ft is not None]
    off = sum(1 for w in wf if not (-_OFFC <= w[0] <= WIDTH_FT + _OFFC
                                    and -_OFFC <= w[1] <= LENGTH_FT + _OFFC))
    cam = f"reproj={camera.reprojection_error_px:.1f}px" if camera is not None else "none"
    bl = ", ".join(f"f{b.frame}:{b.side.value}@({b.court_xy[0]:.2f},{b.court_xy[1]:.2f})"
                   f"{'' if b.in_bounds else '!OUT'}" for b in bounces)
    print(
        f"rally {i}: f{span.start_frame}-{span.end_frame} · {len(measured)} measured · "
        f"{len(bounces)} bounces [{bl}] · camera={cam} · "
        f"court_xy X{_rng(cxs)} Y{_rng(cys)} · "
        f"world_ft X{_rng([w[0] for w in wf])} Y{_rng([w[1] for w in wf])} "
        f"Z{_rng([w[2] for w in wf])} · off-court {off}/{len(wf)}", flush=True)
    # One-shot bias estimate: at the highest modeled point, how far the ray+height result sits from the
    # raw ground projection (the airborne outward-bias the camera ray corrects).
    if camera is not None and wf:
        hi = max((s for s in traj if s.world_ft is not None), key=lambda s: s.world_ft[2])
        if hi.world_ft[2] > 0.5 and hi.court_xy is not None:
            gx, gy = hi.court_xy[0] * WIDTH_FT, hi.court_xy[1] * LENGTH_FT
            d = ((gx - hi.world_ft[0]) ** 2 + (gy - hi.world_ft[1]) ** 2) ** 0.5
            print(f"  bias@f{hi.frame} (Z={hi.world_ft[2]:.1f}ft): ground-proj vs ray "
                  f"differ {d:.1f}ft", flush=True)
    if shots:
        sl = " -> ".join(
            f"{s.shot_index}:{s.shot_type.value}@f{s.frame}"
            f"({'?' if s.team is None else s.team.value}"
            f"{'' if s.speed_mph is None else f',{s.speed_mph:.0f}mph'})" for s in shots)
        print(f"  shots ({len(shots)}): {sl}", flush=True)


def _build_points(ball: list[BallSample], fps: float, camera=None, players=None) -> list[Point]:
    """Segment the ball trajectory into points and run serve/bounce/winner per rally.

    When ``camera`` is available, each rally's ball track is lifted to 3D (feet) with per-frame
    height and speed via :mod:`pbengine.ball.trajectory3d`.
    """
    present = [s.frame for s in ball]
    spans = segment_rallies(present, fps)

    points: list[Point] = []
    for i, span in enumerate(spans):
        measured = [s for s in ball if span.start_frame <= s.frame <= span.end_frame]
        # Bounce/serve/winner run on *measured* detections only — never on interpolated frames.
        bounces = detect_bounces(measured)
        serve = detect_serve(measured)
        winner, reason, conf = determine_winner(bounces, measured)
        # The stored trajectory is enriched (3D) and gap-filled for a continuous overlay.
        from pbengine.ball.trajectory3d import densify_rally

        if camera is not None:
            from pbengine.ball.trajectory3d import fill_gaps_3d, reconstruct_3d_segments

            lifted, outlier_frames = reconstruct_3d_segments(measured, bounces, camera, fps)
            # Cull false-positive detections (rejected by the per-segment parabola) from the stored
            # track; fill_gaps_3d then backfills those frames from the same clean arc, flagged
            # interpolated — so the overlay stays continuous and measured-only stats never see them.
            clean = [s for s in lifted if s.frame not in outlier_frames]
            traj = fill_gaps_3d(clean, bounces, camera, fps)
        else:
            from pbengine.ball.trajectory3d import clean_2d_samples

            traj = clean_2d_samples(measured)
        # Backstop: the physics/2D fills only bridge short within-arc gaps. Linearly fill every frame
        # still missing in the rally span so the stored track is complete for downstream analysis.
        traj = densify_rally(traj, span.start_frame, span.end_frame, bounces, camera, fps)
        # Detect shots (player–ball contacts) from court_xy reversals; their contact heights anchor the
        # ball height model, then we enrich them (speed/type/outcome) from the lifted 3D track.
        from pbengine.ball.trajectory3d import ball_world_ft
        from pbengine.rally.shots import detect_shots, enrich_shots, shot_contacts

        shots = detect_shots(measured, bounces, players or [], serve, fps)
        # Lift to 3D: place each ball pixel on its camera ray at a height anchored to bounces, the net,
        # and contacts, so the 3D ball overlays the 2D track (reprojects to the detection) and clears
        # the net instead of floating off.
        traj = ball_world_ft(traj, bounces, camera, span.start_frame, span.end_frame, fps,
                             contacts=shot_contacts(shots))
        shots = enrich_shots(shots, traj, bounces, winner)
        _log_rally_diag(i, span, measured, bounces, traj, camera, shots)
        points.append(
            Point(
                point_index=i,
                start_frame=span.start_frame,
                end_frame=span.end_frame,
                serve=serve,
                rally_length_shots=len(shots),
                shots=shots,
                bounces=bounces,
                ball_trajectory=traj,
                winner_team=winner,
                win_reason=reason,
                confidence=conf,
                needs_review=conf < 0.6,
            )
        )
    return points


def _write_status(status_path: Path, job_id: str, state: str, stage: str, pct: float) -> None:
    status_path.write_text(
        JobStatus(job_id=job_id, state=state, stage=stage, progress=pct).model_dump_json()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a pickleball match video.")
    parser.add_argument("video", help="path to the match video")
    parser.add_argument("-o", "--out", default="result.json", help="output JSON path")
    parser.add_argument("--stride", type=int, default=1, help="ball-detection frame stride")
    parser.add_argument("--status", help="optional JobStatus file to write progress to")
    parser.add_argument("--job-id", default="cli", help="job id for the status file")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="run with scripted synthetic detectors (no ML); generates the video if missing",
    )
    from pbengine.detect.presets import DEFAULT_PRESET, PRESETS

    parser.add_argument(
        "--players-preset",
        choices=sorted(PRESETS),
        default=DEFAULT_PRESET,
        help="player-detection sensitivity bundle (model/imgsz/conf/stride/tracker): "
             "fast (quick, misses far players) → balanced → max → gpu. Individual flags below override.",
    )
    # These override the preset when given; left unset (None) they keep the preset's value.
    parser.add_argument(
        "--players-weights", default=None,
        help="override the preset's Ultralytics weights (e.g. yolo11n-pose.pt on CPU; "
             "a plain detector like yolo26m.pt skips skeletons)",
    )
    parser.add_argument(
        "--vid-stride", type=int, default=None,
        help="override: process every Nth video frame for player tracking (1 = densest/best motion)",
    )
    parser.add_argument(
        "--players-imgsz", type=int, default=None,
        help="override: inference resolution (e.g. 1280) — higher resolves far players, slower",
    )
    parser.add_argument(
        "--players-conf", type=float, default=None,
        help="override: detection confidence floor (lower catches faint far players, more false +ve)",
    )
    parser.add_argument(
        "--players-augment", action=argparse.BooleanOptionalAction, default=None,
        help="override: test-time augmentation (multi-scale) — helps small players, ~2-3x slower",
    )
    parser.add_argument(
        "--court-corners",
        help="JSON file of 4 manually-clicked court corners; bypasses auto court detection",
    )
    args = parser.parse_args(argv)

    status_path = Path(args.status) if args.status else None

    # Fixture mode: inject scripted stand-ins and synthesize the video if it doesn't exist,
    # so the whole pipeline + viewer can run with zero ML installed.
    injected: dict[str, object] = {}
    if args.fixture:
        from pbengine import fixtures

        if not Path(args.video).exists():
            fixtures.write_synthetic_video(args.video)
        court, players, ball = fixtures.fixture_detectors()
        injected = {
            "court_detector": court,
            "player_detector": players,
            "ball_tracker": ball,
        }
    else:
        # Real run: build the player detector from the chosen sensitivity preset, with any explicit
        # per-knob flags overriding it. Court and ball default to their real (lazy) detectors.
        from pbengine.detect.presets import build_player_detector

        pd = build_player_detector(
            args.players_preset,
            weights=args.players_weights,
            vid_stride=args.vid_stride,
            imgsz=args.players_imgsz,
            conf=args.players_conf,
            augment=args.players_augment,
        )
        injected = {"player_detector": pd}
        # Log the resolved player config + device up front so a slow run's ETA is interpretable.
        try:
            import torch  # noqa: PLC0415

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "unknown"
        print(f"players: preset={args.players_preset} weights={pd.weights} imgsz={pd.imgsz} "
              f"conf={pd.conf} vid_stride={pd.vid_stride} tracker={Path(pd.tracker).name} "
              f"augment={pd.augment} device={device}", flush=True)
        # Manual court calibration overrides automatic detection when corners were provided.
        if args.court_corners:
            from pbengine.court.detector import ManualCourtDetector, load_corners

            injected["court_detector"] = ManualCourtDetector(load_corners(args.court_corners))

    t_start = time.monotonic()

    def cb(stage: str, pct: float) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{pct:5.0%}] +{int(time.monotonic() - t_start)}s {stage}", flush=True)
        if status_path is not None:
            state = "done" if stage == "done" else "running"
            _write_status(status_path, args.job_id, state, stage, pct)

    try:
        result = analyze_match(
            args.video, out_path=args.out, status_cb=cb, stride=args.stride, **injected
        )
    except Exception as exc:  # surface failures to the status file for the API
        if status_path is not None:
            status_path.write_text(
                JobStatus(
                    job_id=args.job_id, state="error", message=str(exc)
                ).model_dump_json()
            )
        raise
    for w in result.warnings:
        print(f"  [skipped] {w}", flush=True)
    print(
        f"wrote {args.out} — {len(result.points)} points, {len(result.players)} player tracks"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
