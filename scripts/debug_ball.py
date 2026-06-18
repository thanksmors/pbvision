"""Run the WASB ball tracker on a clip and report detection coverage.

Tells you whether the ball model is actually finding the ball on your footage (the open
transfer question for a tennis/badminton-trained net) before you trust the points/winners.

    python scripts/debug_ball.py path/to/clip.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from pbengine.ball.tracker import BallTracker  # noqa: E402


def main(video: str) -> int:
    bt = BallTracker()
    print(f"weights: {bt.weights}")
    t0 = time.time()
    samples = bt.track(video)  # no homography -> pixel coords only
    dt = time.time() - t0

    import cv2

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"frames: {n} | ball detected on: {len(samples)} | "
          f"coverage: {(len(samples) / n * 100) if n else 0:.0f}% | {dt:.1f}s")
    if samples:
        xs = [s.px[0] for s in samples]
        ys = [s.px[1] for s in samples]
        print(f"x range: {min(xs):.0f}..{max(xs):.0f}  y range: {min(ys):.0f}..{max(ys):.0f}")
        print("first few:", [(s.frame, round(s.px[0]), round(s.px[1])) for s in samples[:5]])
    cov = (len(samples) / n) if n else 0
    if cov < 0.2:
        print("=> Low coverage: the tennis/badminton net may not transfer to your ball. "
              "Try the badminton weights, or this stage needs pickleball fine-tuning.")
    else:
        print("=> Reasonable coverage. Inspect the trajectory overlay in the app to confirm.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/debug_ball.py path/to/clip.mp4")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
