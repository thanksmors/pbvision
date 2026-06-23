"""Run the WASB ball tracker on a clip and report detection coverage — and tune it.

Tells you whether the ball model is actually finding the ball on your footage (the open
transfer question for a tennis/badminton-trained net), and lets you calibrate the detector knobs
empirically:

    # single run with the defaults (sweep-tuned: badminton, step=1 overlapping, score_threshold=0.2)
    python scripts/debug_ball.py clip.mp4

    # try specific settings
    python scripts/debug_ball.py clip.mp4 --weights tennis --score-threshold 0.3 --step 1

    # sweep a small grid (weights x thresholds x step) and print a coverage table to pick the best
    python scripts/debug_ball.py clip.mp4 --sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

# Gap diagnostics live in the engine (pbengine.ball.diag) so the pipeline logs them too; re-exported
# here under their original names for back-compat with callers/tests of this script.
from pbengine.ball.diag import classify_gaps as _classify_gaps  # noqa: E402,F401
from pbengine.ball.diag import gap_report as _gap_report  # noqa: E402
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


def _gate_report(raw, bt, w: int, h: int) -> None:
    """Quantify the jump gate's effect on the raw detections: how many fast-ball detections the
    old fixed 150 px/frame would have discarded vs the resolution-aware auto gate."""
    import numpy as np

    if len(raw) < 2:
        print("jump-gate: too few detections to assess")
        return
    raw = sorted(raw, key=lambda r: r[0])
    f = np.array([r[0] for r in raw])
    xy = np.array([[r[1], r[2]] for r in raw], dtype=float)
    dt = np.maximum(1, np.diff(f))
    speed = np.hypot(*(np.diff(xy, axis=0).T)) / dt  # px/frame between consecutive detections
    auto = bt._gate_px(w, h)
    over_old = int((speed > 150).sum())
    over_auto = int((speed > auto).sum())
    kept_old = len(bt.postprocess(raw, None, max_px_per_frame=150.0))
    kept_auto = len(bt.postprocess(raw, None, max_px_per_frame=auto))
    print(f"jump-gate: frame {w}x{h} -> auto {auto:.0f} px/frame (old fixed: 150)")
    print(f"  inter-detection speed px/frame: median {np.median(speed):.0f}, "
          f"p90 {np.percentile(speed, 90):.0f}, max {speed.max():.0f}")
    print(f"  steps > 150 px/frame (fast balls the old gate would drop): {over_old}"
          f"  |  steps > auto gate (rejected now): {over_auto}")
    print(f"  detections kept: old-150 gate {kept_old}  ->  auto gate {kept_auto}  "
          f"(+{kept_auto - kept_old})")


def _save_dets(path: str, raw, w: int, h: int, fps: float) -> None:
    """Cache raw (pre-gate) detections so the overlay can reuse this run instead of re-inferring."""
    import json

    with open(path, "w") as fh:
        json.dump({"width": w, "height": h, "fps": fps,
                   "dets": [[int(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in raw]}, fh)
    print(f"saved {len(raw)} raw detections -> {path}")


def _load_dets(path: str):
    """Load a detections cache. Returns ``(raw, width, height, fps)``."""
    import json

    with open(path) as fh:
        d = json.load(fh)
    raw = [(int(f), float(x), float(y), float(c)) for f, x, y, c in d["dets"]]
    return raw, int(d["width"]), int(d["height"]), float(d.get("fps") or 30.0)


def _video_fps(video: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def single(video: str, weights: str, score_threshold: float, step: int, max_disp: float,
           max_frames: int | None = None, max_jump_frac: float = 0.5,
           save_dets: str | None = None) -> int:
    n = _frame_count(video)
    if max_frames:
        n = min(n, max_frames) if n else max_frames
    bt_weights = _resolve_weights(weights)
    if not Path(bt_weights).exists():
        print(f"weights not found: {bt_weights} (run scripts/download_weights.sh)")
        return 2
    print(f"weights: {bt_weights} | score_threshold={score_threshold} step={step} "
          f"max_disp={max_disp}" + (f" | max_frames={max_frames}" if max_frames else ""))
    bt = BallTracker(weights=bt_weights, score_threshold=score_threshold, step=step,
                     max_disp=max_disp, max_jump_frac=max_jump_frac)
    bt._ensure_model()
    print(f"device: {bt._model.device}"
          + ("  (CUDA not available — expect slow inference)" if bt._model.device == "cpu" else ""))
    w, h = bt._frame_size(video)
    fps = _video_fps(video)
    t0 = time.time()
    raw = bt._raw_detections(video, 1, progress=_progress_printer(), max_frames=max_frames)
    dt = time.time() - t0
    if save_dets:
        _save_dets(save_dets, raw, w, h, fps)
    _gate_report(raw, bt, w, h)
    samples = bt.postprocess(raw, homography=None, max_px_per_frame=bt._gate_px(w, h))
    cov = (len(samples) / n) if n else 0
    print(f"frames: {n} | ball detected on: {len(samples)} | coverage: {cov * 100:.0f}% | {dt:.1f}s")
    _gap_report([s.frame for s in samples], n, fps)
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
    ap.add_argument("--weights", default="badminton", help="'tennis', 'badminton', or a path")
    ap.add_argument("--score-threshold", type=float, default=0.2)
    ap.add_argument("--step", type=int, default=1, help="window stride; 1=overlapping (3x compute)")
    ap.add_argument("--max-disp", type=float, default=300.0)
    ap.add_argument("--sweep", action="store_true", help="grid over weights/thresholds/steps")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="only process the first N frames (quick calibration on a subset)")
    ap.add_argument("--max-jump-frac", type=float, default=0.5,
                    help="jump gate as a fraction of max(frame w,h) px/frame (default 0.5)")
    ap.add_argument("--save-dets", default=None, metavar="PATH",
                    help="dump raw detections to JSON so overlay_ball.py --dets can reuse this run")
    args = ap.parse_args()

    if args.sweep:
        return sweep(args.video, args.max_disp, args.max_frames)
    return single(args.video, args.weights, args.score_threshold, args.step, args.max_disp,
                  args.max_frames, args.max_jump_frac, args.save_dets)


if __name__ == "__main__":
    raise SystemExit(main())
