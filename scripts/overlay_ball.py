"""Render the WASB ball trajectory onto a clip so you can *see* what it's tracking.

Coverage % tells you how often the ball was found, not whether it locked onto the ball
(vs. a shoe, a line, a bright spot). This draws each detection plus a short fading trail,
so a few seconds of eyeballing settles it.

    # defaults (badminton / score_threshold=0.2 / step=1)
    python scripts/overlay_ball.py your_clip.mp4 [out.mp4]

    # render a specific config on just the first N frames (quick visual check of a sweep winner)
    python scripts/overlay_ball.py your_clip.mp4 --weights badminton --score-threshold 0.2 \
        --step 1 --max-frames 300

Frames are streamed (read -> draw -> write), so memory stays flat on long/high-res clips,
same as the tracker itself.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts (debug_ball helpers)

from debug_ball import _progress_printer, _resolve_weights  # noqa: E402
from pbengine.ball.tracker import BallTracker  # noqa: E402

TRAIL = 12  # how many past points to fade behind the ball


def main(
    video: str,
    out_path: str,
    weights: str = "badminton",
    score_threshold: float = 0.2,
    step: int = 1,
    max_disp: float = 300.0,
    max_frames: int | None = None,
) -> int:
    import cv2

    bt = BallTracker(
        weights=_resolve_weights(weights),
        score_threshold=score_threshold,
        step=step,
        max_disp=max_disp,
    )
    if not Path(bt.weights).exists():
        print(f"weights not found: {bt.weights} (run scripts/download_weights.sh)")
        return 2
    bt._ensure_model()
    print(f"weights: {bt.weights} | score_threshold={score_threshold} step={step}"
          + (f" | max_frames={max_frames}" if max_frames else ""))
    print(f"device: {bt._model.device}"
          + ("  (CUDA not available — expect slow inference)"
             if bt._model.device == "cpu" else ""))
    samples = bt.track(video, progress=_progress_printer(), max_frames=max_frames)
    by_frame = {s.frame: s for s in samples}
    print(f"detections: {len(samples)}")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"cannot open {video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # cv2.VideoWriter fails *silently* if the codec isn't in this OpenCV build (it returns
    # an object that just drops every frame -> a 0-byte/absent file). Try mp4v, then fall
    # back to MJPG/.avi, which ships with essentially every OpenCV. Verify it opened.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        avi_path = str(Path(out_path).with_suffix(".avi"))
        print(f"mp4v codec unavailable; falling back to MJPG -> {avi_path}")
        writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        out_path = avi_path
    if not writer.isOpened():
        cap.release()
        print("could not open any video writer (no mp4v or MJPG codec). "
              "Install ffmpeg/opencv with codec support: pip install opencv-python")
        return 1

    trail: deque[tuple[int, int]] = deque(maxlen=TRAIL)
    idx = 0
    ok, frame = cap.read()
    while ok:
        if max_frames and idx >= max_frames:
            break  # only render the frames the detector actually ran on
        s = by_frame.get(idx)
        if s is not None:
            trail.append((int(round(s.px[0])), int(round(s.px[1]))))
        elif trail:
            trail.append(trail[-1])  # hold last point so the trail fades instead of snapping
        for age, pt in enumerate(trail):
            shade = int(80 + 175 * (age + 1) / len(trail))  # older = dimmer
            cv2.circle(frame, pt, 3, (0, shade, shade), -1)
        if s is not None:
            cv2.circle(frame, (int(round(s.px[0])), int(round(s.px[1]))), 9, (0, 0, 255), 2)
        writer.write(frame)
        idx += 1
        ok, frame = cap.read()

    cap.release()
    writer.release()

    size = Path(out_path).stat().st_size if Path(out_path).exists() else 0
    if size == 0:
        print(f"ERROR: {out_path} is empty/missing — the writer dropped every frame. "
              "Your OpenCV likely lacks video-encoding support.")
        return 1
    print(f"wrote {out_path} ({idx} frames, {size // 1024} KB). "
          "Open it and confirm the red ring rides the ball.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the WASB ball trajectory onto a clip.")
    ap.add_argument("video")
    ap.add_argument("out", nargs="?", help="output path (default: <video>_ball.mp4)")
    ap.add_argument("--weights", default="badminton", help="'tennis', 'badminton', or a path")
    ap.add_argument("--score-threshold", type=float, default=0.2)
    ap.add_argument("--step", type=int, default=1, help="window stride; 1=overlapping (3x compute)")
    ap.add_argument("--max-disp", type=float, default=300.0)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="only process/render the first N frames (quick visual check)")
    args = ap.parse_args()

    dst = args.out or (str(Path(args.video).with_suffix("")) + "_ball.mp4")
    raise SystemExit(main(args.video, dst, args.weights, args.score_threshold, args.step,
                          args.max_disp, args.max_frames))
