"""Synthetic fixtures: run the whole pipeline + viewer with **zero ML installed**.

The three model-backed stages (court, players, ball) are dependency-injected into
:func:`pbengine.pipeline.analyze_match`, so here we provide fake versions that return scripted
data instead of running YOLO26 / WASB / TennisCourtDetector. This lets us validate the
*logic* of the pipeline (rally segmentation, serve/bounce/winner, JSON contract) and the
viewer end-to-end before the real ML glue exists.

We also render a small real ``.mp4`` (court lines + a moving ball) so the upload/probe flow
and the viewer's video + trajectory overlay have something to show.

Coordinate flow: a fixed homography ``H`` maps source pixels -> normalized court coords
(``court_model``). To place scripted players/ball we go the other way with ``H^-1`` (court ->
pixels), so the pixel data the viewer draws is consistent with the court geometry the engine
reasons about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pbengine.court.homography import homography_from_named_points, project
from pbengine.schema.models import BallSample, VideoMeta

# --- Camera / court layout for a 1280x720 synthetic frame -------------------------------
WIDTH, HEIGHT, FPS = 1280, 720, 30.0

# Named court keypoints in pixel space, forming a perspective trapezoid (side A near top,
# narrower; side B near bottom, wider). These map to the reference template by name.
COURT_KEYPOINTS_PX: dict[str, tuple[float, float]] = {
    "corner_a_left": (440.0, 150.0),
    "corner_a_right": (840.0, 150.0),
    "corner_b_left": (240.0, 660.0),
    "corner_b_right": (1040.0, 660.0),
    "net_left": (340.0, 405.0),
    "net_right": (940.0, 405.0),
}

# Outline (in pixels) used both for drawing the demo video and the viewer overlay.
COURT_OUTLINE = ["corner_a_left", "corner_a_right", "corner_b_right", "corner_b_left"]


def fixture_homography() -> np.ndarray:
    """The fixed pixel -> court_xy homography for the synthetic camera."""
    return homography_from_named_points(COURT_KEYPOINTS_PX)


def _court_to_px(h: np.ndarray, court_xy: tuple[float, float]) -> tuple[float, float]:
    """Inverse-project a normalized court point back into pixel space."""
    px = project(np.linalg.inv(h), np.array(court_xy, dtype=float))[0]
    return (float(px[0]), float(px[1]))


def _stick_figure(
    fx: float, fy: float, frame: int
) -> tuple[list[tuple[float, float]], list[float]]:
    """A synthetic COCO-17 stick figure standing with feet at (fx, fy), with a little arm sway.

    Lets the zero-ML fixture demo exercise the skeleton overlay + 3D lift end to end. Coordinates
    are pixels; y grows downward, so 'up the body' is negative dy from the feet.
    """
    a = 0.10 * np.sin(frame / 15.0)  # arm swing
    # (dx, dy) offsets from the feet anchor (fy is bottom), in the COCO-17 order.
    off = [
        (0, -120),                       # 0 nose
        (-5, -124), (5, -124),           # 1,2 eyes
        (-10, -120), (10, -120),         # 3,4 ears
        (-16, -104), (16, -104),         # 5,6 shoulders
        (-26 + 18 * a, -84), (26 - 18 * a, -84),   # 7,8 elbows
        (-30 + 30 * a, -64), (30 - 30 * a, -64),   # 9,10 wrists
        (-12, -64), (12, -64),           # 11,12 hips
        (-13, -32), (13, -32),           # 13,14 knees
        (-14, 0), (14, 0),               # 15,16 ankles
    ]
    kp = [(fx + dx, fy + dy) for dx, dy in off]
    cf = [0.9] * len(kp)
    return kp, cf


# --- Fake model wrappers (match the real classes' interfaces) ----------------------------
@dataclass
class FixtureCourtDetector:
    """Stand-in for :class:`pbengine.court.detector.CourtDetector`."""

    def detect(self, frame) -> dict[str, tuple[float, float]]:  # noqa: ARG002 - frame ignored
        return dict(COURT_KEYPOINTS_PX)

    def solve(self, frame) -> np.ndarray:  # noqa: ARG002 - frame ignored
        return fixture_homography()


@dataclass
class FixturePlayerDetector:
    """Stand-in for :class:`pbengine.detect.players.PlayerDetector`.

    Four players (two per side) standing roughly mid-court, sampled every few frames with a
    little idle sway so the position track isn't a single point.
    """

    total_frames: int = 200
    step: int = 10

    def track(self, video_path=None, **_kw):  # noqa: ARG002 - video ignored; accept progress kwargs
        from pbengine.detect.players import PlayerTrack

        h = fixture_homography()
        # (track_id, base court_xy) — A near y=0.30, B near y=0.70.
        bases = {1: (0.35, 0.30), 2: (0.65, 0.30), 3: (0.35, 0.70), 4: (0.65, 0.70)}
        tracks: list[PlayerTrack] = []
        for frame in range(0, self.total_frames, self.step):
            sway = 0.02 * np.sin(frame / 20.0)
            for tid, (cx, cy) in bases.items():
                fx, fy = _court_to_px(h, (cx + sway, cy))
                kp, cf = _stick_figure(fx, fy, frame)
                tracks.append(
                    PlayerTrack(
                        track_id=tid,
                        frame=frame,
                        bbox_px=(fx - 28, fy - 120, fx + 28, fy),
                        keypoints_px=kp,
                        keypoint_conf=cf,
                    )
                )
        return tracks


@dataclass
class FixtureBallTracker:
    """Stand-in for :class:`pbengine.ball.tracker.BallTracker` — returns scripted rallies."""

    samples: list[BallSample] = field(default_factory=list)

    def track(self, video_path=None, homography=None, stride: int = 1, **_kw):  # noqa: ARG002
        return list(self.samples)  # accept progress kwarg from the pipeline; nothing to report


# --- Scripted ball trajectories ----------------------------------------------------------
def _rally(
    bounces_court: list[tuple[float, float]],
    start_frame: int,
    h: np.ndarray,
    *,
    shot_len: int = 16,
    amplitude: float = 200.0,
    conf: float = 0.85,
    tail: list[tuple[float, float]] | None = None,
) -> tuple[list[BallSample], int]:
    """Build one rally as a sequence of arcs between successive bounce landings.

    Ball height is modelled **explicitly**: the pixel position is the ground projection of the
    ball's court point minus a per-shot height parabola (0 at the bounces, ``amplitude`` at
    mid-flight). Because height dominates the small frame-to-frame change in ground position,
    every bounce — near *or* far side — is a clean local pixel-y maximum that
    :func:`pbengine.bounce.heuristic.detect_bounces` picks up. ``tail`` adds samples after the
    final bounce (ball rising off the ground, or dying at the net) so that last bounce has
    right-hand neighbours and is detectable too.
    """
    samples: list[BallSample] = []
    frame = start_frame

    def ground(court: tuple[float, float]) -> tuple[float, float]:
        return _court_to_px(h, court)

    for i in range(len(bounces_court) - 1):
        (cx0, cy0), (cx1, cy1) = bounces_court[i], bounces_court[i + 1]
        for s in range(shot_len):
            frac = s / shot_len  # frac == 0 is the bounce at index i
            court = (cx0 + (cx1 - cx0) * frac, cy0 + (cy1 - cy0) * frac)
            gx, gy = ground(court)
            height = amplitude * 4 * frac * (1 - frac)
            samples.append(BallSample(frame=frame, px=(gx, gy - height), court_xy=court, conf=conf))
            frame += 1

    # Final bounce sample (height 0).
    gx, gy = ground(bounces_court[-1])
    samples.append(BallSample(frame=frame, px=(gx, gy), court_xy=bounces_court[-1], conf=conf))
    frame += 1

    # Tail: ball rising off the final bounce (or arcing up to die at the net).
    if tail:
        n = len(tail)
        for j, court in enumerate(tail, start=1):
            gx, gy = ground(court)
            height = amplitude * 0.6 * (j / n)
            samples.append(BallSample(frame=frame, px=(gx, gy - height), court_xy=court, conf=conf))
            frame += 1
    return samples, frame


def scripted_ball() -> list[BallSample]:
    """Three rallies exercising each win-reason branch: ball_out, double_bounce, net."""
    h = fixture_homography()
    samples: list[BallSample] = []

    # Rally 1 — last bounce lands out past B's baseline (ball_out; winner = B).
    r1, end = _rally(
        [(0.5, 0.30), (0.5, 0.70), (0.5, 0.30), (0.5, 1.06)],
        start_frame=30,
        h=h,
        tail=[(0.5, 1.06), (0.5, 1.06), (0.5, 1.06)],
    )
    samples += r1

    # Rally 2 — two consecutive in-bounds bounces on side B (double_bounce; winner = A).
    r2, end = _rally(
        [(0.5, 0.30), (0.5, 0.70), (0.5, 0.78)],
        start_frame=end + 25,
        h=h,
        tail=[(0.5, 0.80), (0.5, 0.80), (0.5, 0.80)],
    )
    samples += r2

    # Rally 3 — ball dies at the net after A's shot (net; winner = B).
    r3, _ = _rally(
        [(0.5, 0.30), (0.5, 0.70), (0.5, 0.30)],
        start_frame=end + 25,
        h=h,
        tail=[(0.5, 0.40), (0.5, 0.45), (0.5, 0.50)],
    )
    samples += r3
    return samples


def fixture_detectors() -> tuple[FixtureCourtDetector, FixturePlayerDetector, FixtureBallTracker]:
    """The trio of fake detectors to inject into ``analyze_match``."""
    ball = scripted_ball()
    total = max(s.frame for s in ball) + 15
    return (
        FixtureCourtDetector(),
        FixturePlayerDetector(total_frames=total),
        FixtureBallTracker(samples=ball),
    )


def synthetic_meta(frames: int) -> VideoMeta:
    return VideoMeta(fps=FPS, frames=frames, width=WIDTH, height=HEIGHT)


# --- Demo video --------------------------------------------------------------------------
def write_synthetic_video(path: str | Path, ball: list[BallSample] | None = None) -> VideoMeta:
    """Render a small ``.mp4`` of the court + moving ball so the upload/probe/viewer flow works.

    Returns the :class:`VideoMeta` of what was written. Uses OpenCV's VideoWriter (core dep).
    """
    import cv2

    ball = ball if ball is not None else scripted_ball()
    by_frame = {s.frame: s.px for s in ball}
    total = (max(by_frame) if by_frame else 0) + 15

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outline = np.array([COURT_KEYPOINTS_PX[n] for n in COURT_OUTLINE], dtype=np.int32)
    net = np.array(
        [COURT_KEYPOINTS_PX["net_left"], COURT_KEYPOINTS_PX["net_right"]], dtype=np.int32
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():  # pragma: no cover - codec availability is environment-specific
        raise RuntimeError(f"OpenCV could not open a VideoWriter for {path}")
    try:
        for f in range(total):
            frame = np.full((HEIGHT, WIDTH, 3), (40, 90, 40), dtype=np.uint8)  # court green
            cv2.polylines(frame, [outline], isClosed=True, color=(255, 255, 255), thickness=2)
            cv2.polylines(frame, [net], isClosed=False, color=(220, 220, 220), thickness=2)
            if f in by_frame:
                x, y = by_frame[f]
                cv2.circle(frame, (int(x), int(y)), 7, (0, 230, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return synthetic_meta(total)
