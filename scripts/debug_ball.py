"""Run the WASB ball tracker on a clip and report detection coverage — and tune it.

Tells you whether the ball model is actually finding the ball on your footage (the open
transfer question for a tennis/badminton-trained net), and lets you calibrate the detector knobs
empirically:

    # single run with the defaults (recall-tuned: step=1 overlapping windows, score_threshold=0.3)
    python scripts/debug_ball.py clip.mp4

    # try specific settings
    python scripts/debug_ball.py clip.mp4 --weights badminton --score-threshold 0.2 --step 1

    # sweep a small grid (weights x thresholds x step) and print a coverage table to pick the best
    python scripts/debug_ball.py clip.mp4 --sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from pbengine.ball.tracker import BallTracker  # noqa: E402

_MODELS = Path(__file__).resolve().parents[1] / "engine" / "pbengine" / "models"
_WEIGHTS = {
    "tennis": _MODELS / "wasb_tennis_best.pth.tar",
    "badminton": _MODELS / "wasb_badminton_best.pth.tar",
}


def _resolve_weights(name_or_path: str) -> str:
    return str(_WEIGHTS.get(name_or_path, Path(name_or_path)))


def _frame_count(video: str) -> int:
    import cv2

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _progress_printer():
    """Build a throttled progress callback that draws a live one-line status to stderr."""
    state = {"t0": time.time(), "last": 0.0, "phase": None}

    def cb(phase: str, done: int, total: int) -> None:
        now = time.time()
        # Reset the timer when a new phase starts so ETA is per-phase.
        if phase != state["phase"]:
            state["phase"] = phase
            state["t0"] = now
            state["last"] = 0.0
        if now - state["last"] < 0.5 and done < total:
            return  # throttle to ~2 updates/sec (always emit the final tick)
        state["last"] = now
        elapsed = now - state["t0"]
        pct = (done / total * 100) if total else 0
        eta = (elapsed / done * (total - done)) if done else 0
        line = (f"  {phase:6} {done}/{total or '?'} ({pct:3.0f}%)  "
                f"{elapsed:4.0f}s elapsed  ~{eta:3.0f}s left")
        sys.stderr.write("\r" + line.ljust(64))
        sys.stderr.flush()
        if total and done >= total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    return cb


def _run(
    video: str,
    weights: str,
    score_threshold: float,
    step: int,
    max_disp: float,
    progress=None,
    max_frames: int | None = None,
):
    bt = BallTracker(
        weights=_resolve_weights(weights),
        score_threshold=score_threshold,
        step=step,
        max_disp=max_disp,
    )
    bt._ensure_model()
    print(f"device: {bt._model.device}"
          + ("  (CUDA not available — expect slow inference)"
             if bt._model.device == "cpu" else ""))
    t0 = time.time()
    samples = bt.track(video, progress=progress, max_frames=max_frames)  # pixel coords only
    return samples, time.time() - t0


def single(video: str, weights: str, score_threshold: float, step: int, max_disp: float,
           max_frames: int | None = None) -> int:
    n = _frame_count(video)
    if max_frames:
        n = min(n, max_frames) if n else max_frames
    bt_weights = _resolve_weights(weights)
    if not Path(bt_weights).exists():
        print(f"weights not found: {bt_weights} (run scripts/download_weights.sh)")
        return 2
    print(f"weights: {bt_weights} | score_threshold={score_threshold} step={step} "
          f"max_disp={max_disp}" + (f" | max_frames={max_frames}" if max_frames else ""))
    samples, dt = _run(video, weights, score_threshold, step, max_disp,
                       progress=_progress_printer(), max_frames=max_frames)
    cov = (len(samples) / n) if n else 0
    print(f"frames: {n} | ball detected on: {len(samples)} | coverage: {cov * 100:.0f}% | {dt:.1f}s")
    if samples:
        xs = [s.px[0] for s in samples]
        ys = [s.px[1] for s in samples]
        print(f"x range: {min(xs):.0f}..{max(xs):.0f}  y range: {min(ys):.0f}..{max(ys):.0f}")
        print("first few:", [(s.frame, round(s.px[0]), round(s.px[1])) for s in samples[:5]])
    if cov < 0.2:
        print("=> Low coverage. Try --sweep to compare weights/thresholds, or this stage needs "
              "pickleball fine-tuning.")
    else:
        print("=> Reasonable coverage. Inspect the overlay (scripts/overlay_ball.py) to confirm.")
    return 0


def sweep(video: str, max_disp: float, max_frames: int | None = None) -> int:
    """Grid over weights x score_threshold x step; print a coverage table to pick the best config."""
    n = _frame_count(video)
    if max_frames:
        n = min(n, max_frames) if n else max_frames
    weights = [w for w in ("tennis", "badminton") if _WEIGHTS[w].exists()]
    if not weights:
        print("no weights found; run scripts/download_weights.sh")
        return 2
    thresholds = [0.2, 0.3, 0.5]
    steps = [1, 3]  # overlapping vs non-overlapping windows
    print(f"frames: {n} | sweeping weights={weights} thresholds={thresholds} steps={steps}"
          + (f" | max_frames={max_frames}" if max_frames else ""))
    print(f"{'weights':10} {'thresh':>6} {'step':>4} {'coverage':>9} {'detected':>9} {'secs':>6}")
    best = None
    total_cfgs = len(weights) * len(thresholds) * len(steps)
    i = 0
    for w in weights:
        for th in thresholds:
            for st in steps:
                i += 1
                print(f"[{i}/{total_cfgs}] running {w} thresh={th:.2f} step={st} ...")
                samples, dt = _run(video, w, th, st, max_disp,
                                   progress=_progress_printer(), max_frames=max_frames)
                cov = (len(samples) / n) if n else 0
                print(f"{w:10} {th:6.2f} {st:4d} {cov * 100:8.0f}% {len(samples):9d} {dt:6.1f}")
                if best is None or cov > best[0]:
                    best = (cov, w, th, st)
    if best:
        print(f"=> best: weights={best[1]} score_threshold={best[2]} step={best[3]} "
              f"({best[0] * 100:.0f}% coverage). Re-run that config and check the overlay.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Report/tune WASB ball-detection coverage on a clip.")
    ap.add_argument("video")
    ap.add_argument("--weights", default="tennis", help="'tennis', 'badminton', or a path")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--step", type=int, default=1, help="window stride; 1=overlapping (3x compute)")
    ap.add_argument("--max-disp", type=float, default=300.0)
    ap.add_argument("--sweep", action="store_true", help="grid over weights/thresholds/steps")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="only process the first N frames (quick calibration on a subset)")
    args = ap.parse_args()

    if args.sweep:
        return sweep(args.video, args.max_disp, args.max_frames)
    return single(args.video, args.weights, args.score_threshold, args.step, args.max_disp,
                  args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
