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

from debug_ball import _load_dets, _progress_printer, _resolve_weights  # noqa: E402
from pbengine.ball.tracker import BallTracker  # noqa: E402

TRAIL = 12  # how many past points to fade behind the ball


def _speed_color(speed: float, fast: float = 400.0):
    """BGR ring colour: green (slow) -> red (fast). ``fast`` px/frame maps to full red."""
    t = max(0.0, min(1.0, speed / fast))
    return (0, int(255 * (1 - t)), int(255 * t))  # (B, G, R)


def main(
    video: str,
    out_path: str,
    weights: str = "badminton",
    score_threshold: float = 0.2,
    step: int = 1,
    max_disp: float = 300.0,
    max_frames: int | None = None,
    dets: str | None = None,
) -> int:
    import cv2

    if dets:  # reuse a cached run from debug_ball.py --save-dets (no torch, no inference)
        raw, dw, dh, _fps = _load_dets(dets)
        bt = BallTracker()  # no model needed; postprocess only
        samples = bt.postprocess(raw, homography=None, max_px_per_frame=bt._gate_px(dw, dh))
        print(f"loaded {len(raw)} cached detections from {dets} ({dw}x{dh}); no inference")
    else:
        bt = BallTracker(weights=_resolve_weights(weights), score_threshold=score_threshold,
                         step=step, max_disp=max_disp)
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
    prev: tuple[int, float, float] | None = None  # (frame_idx, x, y) of the last detection
    gap = 0  # consecutive frames with no detection
    idx = 0
    ok, frame = cap.read()
    while ok:
        if max_frames and idx >= max_frames:
            break  # only render the frames the detector actually ran on
        s = by_frame.get(idx)
        if s is not None:
            speed = 0.0
            if prev is not None and idx > prev[0]:
                speed = ((s.px[0] - prev[1]) ** 2 + (s.px[1] - prev[2]) ** 2) ** 0.5 / (idx - prev[0])
            trail.append((int(round(s.px[0])), int(round(s.px[1]))))
            prev, gap = (idx, s.px[0], s.px[1]), 0
        else:
            gap += 1
            if trail:
                trail.append(trail[-1])  # hold last point so the trail fades instead of snapping
        for age, pt in enumerate(trail):
            shade = int(80 + 175 * (age + 1) / len(trail))  # older = dimmer
            cv2.circle(frame, pt, 3, (0, shade, shade), -1)
        if s is not None:
            c = (int(round(s.px[0])), int(round(s.px[1])))
            cv2.circle(frame, c, 9, _speed_color(speed), 2)
            cv2.putText(frame, f"{speed:.0f} px/f", (c[0] + 12, c[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _speed_color(speed), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"NO BALL - gap {gap}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
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
    ap.add_argument("--dets", default=None, metavar="PATH",
                    help="reuse a debug_ball.py --save-dets cache instead of re-running inference")
    args = ap.parse_args()

    dst = args.out or (str(Path(args.video).with_suffix("")) + "_ball.mp4")
    raise SystemExit(main(args.video, dst, args.weights, args.score_threshold, args.step,
                          args.max_disp, args.max_frames, args.dets))
