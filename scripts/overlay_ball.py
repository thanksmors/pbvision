"""Render the WASB ball trajectory onto a clip so you can *see* what it's tracking.

Coverage % tells you how often the ball was found, not whether it locked onto the ball
(vs. a shoe, a line, a bright spot). This draws each detection plus a short fading trail,
so a few seconds of eyeballing settles it.

    python scripts/overlay_ball.py your_clip.mp4 [out.mp4]

Frames are streamed (read -> draw -> write), so memory stays flat on long/high-res clips,
same as the tracker itself.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from pbengine.ball.tracker import BallTracker  # noqa: E402

TRAIL = 12  # how many past points to fade behind the ball


def main(video: str, out_path: str) -> int:
    import cv2

    bt = BallTracker()
    print(f"weights: {bt.weights}")
    print("running detector (slow on CPU)...")
    samples = bt.track(video)  # pixel coords; no homography needed
    by_frame = {s.frame: s for s in samples}
    print(f"detections: {len(samples)}")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"cannot open {video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trail: deque[tuple[int, int]] = deque(maxlen=TRAIL)
    idx = 0
    ok, frame = cap.read()
    while ok:
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
    print(f"wrote {out_path} ({idx} frames). Open it and confirm the red ring rides the ball.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("usage: python scripts/overlay_ball.py video.mp4 [out.mp4]")
        raise SystemExit(2)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) == 3 else str(Path(src).with_suffix("")) + "_ball.mp4"
    raise SystemExit(main(src, dst))
