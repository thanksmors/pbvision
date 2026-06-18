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
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from pbengine.ball.tracker import BallTracker
from pbengine.bounce.heuristic import detect_bounces
from pbengine.court.court_model import side_of
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
_MODELS = {"detector": "yolo26m", "ball": "wasb-tennis", "court": "tenniscourtdetector"}


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

    # Each model-backed stage degrades gracefully: if its weights/deps are missing the stage is
    # skipped (empty output + a warning) so the engine can run one model at a time.
    report("court", 0.1)
    court_model, homography = _skip_if_unavailable(
        "court", warnings, (None, None), _solve_court, video_path, court_detector,
        exc_types=(ModelUnavailable, CourtNotFound),
    )

    report("players", 0.3)
    players = _skip_if_unavailable(
        "players", warnings, [], _track_players, video_path, homography, player_detector
    )

    report("ball", 0.6)
    ball = _skip_if_unavailable(
        "ball", warnings, [], ball_tracker.track, video_path, homography=homography, stride=stride
    )

    # Recover a metric camera from the court homography so the ball track can be lifted to 3D.
    # Optional: if it fails or the corner reprojection is poor, points keep their 2D trajectory.
    camera = None
    if homography is not None:
        camera = _skip_if_unavailable(
            "camera_3d", warnings, None, _recover_camera, homography, meta.width, meta.height,
            exc_types=(Exception,),
        )

    report("rallies", 0.85)
    points = _build_points(ball, meta.fps, camera)

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


def _track_players(
    video_path: Path, homography: np.ndarray | None, detector: PlayerDetector
) -> list[Player]:
    """Track players and project foot positions into court coords; assign teams by side."""
    from pbengine.court.homography import project

    tracks = detector.track(video_path)
    by_id: dict[int, list[PlayerPosition]] = defaultdict(list)
    side_votes: dict[int, list[str]] = defaultdict(list)
    for t in tracks:
        court_xy = (0.0, 0.0)
        if homography is not None:
            court_xy = tuple(project(homography, t.foot_px)[0])
        by_id[t.track_id].append(
            PlayerPosition(frame=t.frame, court_xy=court_xy, bbox_px=t.bbox_px)
        )
        if homography is not None:
            side_votes[t.track_id].append(side_of(court_xy))

    players: list[Player] = []
    for tid, positions in by_id.items():
        votes = side_votes.get(tid, [])
        team = Team(max(set(votes), key=votes.count)) if votes else None
        players.append(Player(track_id=tid, team=team, positions=positions))
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


def _build_points(ball: list[BallSample], fps: float, camera=None) -> list[Point]:
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
        points.append(
            Point(
                point_index=i,
                start_frame=span.start_frame,
                end_frame=span.end_frame,
                serve=serve,
                rally_length_shots=max(0, len(bounces)),
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
    parser.add_argument(
        "--players-weights",
        default="yolo26m.pt",
        help="Ultralytics weights for player tracking (use yolo26n.pt on CPU)",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="process every Nth video frame for player tracking (speeds up CPU runs)",
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
        # Real run: configure the player detector for the chosen weights / stride. Court and
        # ball default to their real (lazy) detectors and skip gracefully if not installed.
        injected = {
            "player_detector": PlayerDetector(
                weights=args.players_weights, vid_stride=args.vid_stride
            )
        }
        # Manual court calibration overrides automatic detection when corners were provided.
        if args.court_corners:
            from pbengine.court.detector import ManualCourtDetector, load_corners

            injected["court_detector"] = ManualCourtDetector(load_corners(args.court_corners))

    def cb(stage: str, pct: float) -> None:
        print(f"[{pct:5.0%}] {stage}", flush=True)
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
