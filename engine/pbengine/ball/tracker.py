"""Ball detection + tracking — the highest-risk stage.

Detector is WASB-SBDT (``nttcom/WASB-SBDT``, vendored under ``third_party/``), wrapped without
its Hydra framework in :mod:`pbengine.ball.wasb`. Use its tennis or badminton weights as the
pickleball starting point (ball size/speed sits between the two) — pickleball transfer is the
known accuracy risk. Raw detections are gated for physically-impossible jumps and
Kalman-smoothed (see :mod:`pbengine.ball.kalman`).

The WASB import + model load are lazy, so the rest of the engine imports cleanly on a box
without the ``ml`` extra. v1 effort should go into validating *this* stage on real footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pbengine.ball.kalman import gate_jumps, smooth
from pbengine.court.court_model import LENGTH_FT, WIDTH_FT
from pbengine.court.homography import project
from pbengine.schema.models import BallSample

# A ball pixel high in the air (or near the image vanishing line) projected through the *ground*-plane
# homography lands far from the court — the projective division explodes to hundreds of feet. Such a
# position is meaningless and, left in, feeds false bounces/serve/winner calls. Beyond the court
# footprint + this slack (in feet) we keep the pixel but discard the bogus court position (set None);
# downstream already skips None court_xy. 8 ft keeps genuinely just-out bounces while killing the
# airborne/vanishing-line explosions.
_COURT_OUTLIER_SLACK_FT = 8.0


def _court_xy_or_none(cx: float, cy: float) -> tuple[float, float] | None:
    """Normalized court point, or None if its ground projection lands implausibly far off the court."""
    s = _COURT_OUTLIER_SLACK_FT
    if -s <= cx * WIDTH_FT <= WIDTH_FT + s and -s <= cy * LENGTH_FT <= LENGTH_FT + s:
        return (cx, cy)
    return None


# Fast-ball recall: the CNN drops the ball on short gaps during fast shots. A fast ball is the
# dominant inter-frame motion in a small window around its predicted path, so a frame-difference peak
# there recovers it — no GPU/retraining. Recovered points get a low confidence and pass through the
# same jump-gate + smoothing as real detections.
_RECALL_MAX_GAP = 18      # only fill gaps up to ~0.6 s (arc-breaks); longer is a real rally break
_RECALL_CROP_HALF = 40    # search a +/-40 px window around the predicted ball location
_RECALL_MIN_SCORE = 18.0  # min blurred abs-diff (0-255) to accept a motion peak as the ball
_RECALL_CONF = 0.3        # confidence stamped on a recovered detection (below a real CNN hit)


def _motion_peak(prev_gray, cur_gray, center, half: int, min_score: float):
    """Locate the strongest motion (a fast ball) near ``center`` via a blurred frame-difference peak.

    Returns ``(x, y, score)`` in full-frame pixels, or None if no motion clears ``min_score`` (or on
    any failure — this assists detection and must never raise into it).
    """
    try:
        import cv2

        height, width = cur_gray.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(width, cx + half + 1), min(height, cy + half + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        diff = cv2.GaussianBlur(
            np.abs(cur_gray[y0:y1, x0:x1].astype(np.float32)
                   - prev_gray[y0:y1, x0:x1].astype(np.float32)), (5, 5), 1.0)
        score = float(diff.max())
        if score < min_score:
            return None
        py, px = np.unravel_index(int(diff.argmax()), diff.shape)
        return (float(x0 + px), float(y0 + py), score)
    except Exception:
        return None

# Badminton weights + score_threshold=0.2 + step=1 won an empirical sweep on real pickleball
# footage (scripts/debug_ball.py --sweep): ~61% raw coverage, edging tennis, and the overlay
# confirmed the detections ride the ball. Swap weights/threshold per-clip if footage differs.
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "wasb_badminton_best.pth.tar"


@dataclass
class BallTracker:
    weights: str = str(_DEFAULT_WEIGHTS)
    # Jump gate: drop detections implying an impossible per-frame pixel speed. A fast pickleball
    # is genuinely fast — ~520 px/frame (40 mph) to ~650 px/frame (50 mph) at 1080p/30fps — so a
    # fixed 150 (tennis/badminton-era) silently discarded real fast balls. The bound is therefore
    # frame-relative: ``max_jump_frac * max(w, h)`` (~960 px at 1080p ⇒ ~74 mph headroom) while
    # still rejecting full-frame scene-cut teleports. ``max_px_per_frame`` overrides it when set.
    # Fine outlier rejection is now the per-segment RANSAC fit (pbengine.ball.trajectory3d), so the
    # coarse gate only needs to kill gross teleports — safe to loosen.
    max_px_per_frame: float | None = None
    max_jump_frac: float = 0.5
    device: str | None = None
    # WASB detector knobs (see pbengine.ball.wasb.WasbBall). Defaults favour recall: a low blob
    # threshold and overlapping windows (step=1) so each frame gets up to 3 detection attempts.
    # Note: the per-segment robust parabola fit (pbengine.ball.trajectory3d) now rejects
    # false-positive detections downstream, so raising this toward 0.3 admits fewer false positives
    # at some recall cost — re-validate with scripts/debug_ball.py --sweep before changing it.
    score_threshold: float = 0.2
    max_disp: float = 300.0
    step: int = 1
    _model: object = field(default=None, repr=False)

    def _gate_px(self, width: int | None = None, height: int | None = None) -> float:
        """Effective jump gate (px/frame): explicit override, else frame-relative, else a wide
        fallback when frame size is unknown (keeps direct ``postprocess`` calls safe)."""
        if self.max_px_per_frame is not None:
            return self.max_px_per_frame
        if width and height:
            return self.max_jump_frac * max(width, height)
        return 1e9  # unknown frame size -> effectively no coarse gate; RANSAC still rejects outliers

    @staticmethod
    def _frame_size(video_path: str | Path) -> tuple[int, int]:
        """(width, height) of the source video, read from container metadata (no frame decode)."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h

    def _ensure_model(self) -> None:
        if self._model is None:
            from pbengine.ball.wasb import WasbBall  # lazy: pulls torch + the submodule

            self._model = WasbBall(
                self.weights,
                device=self.device,
                score_threshold=self.score_threshold,
                max_disp=self.max_disp,
                step=self.step,
            )

    def _raw_detections(
        self,
        video_path: str | Path,
        stride: int,
        progress=None,
        max_frames: int | None = None,
    ) -> list[tuple[int, float, float, float]]:
        """Return ``(frame, x, y, conf)`` detections from WASB. WASB needs consecutive frames,
        so ``stride`` is ignored here (kept for interface compatibility)."""
        self._ensure_model()
        return self._model.infer_video(  # type: ignore[union-attr]
            str(video_path), progress=progress, max_frames=max_frames
        )

    def track(
        self,
        video_path: str | Path,
        homography: np.ndarray | None = None,
        stride: int = 1,
        progress=None,
        max_frames: int | None = None,
    ) -> list[BallSample]:
        """Detect, gate, smooth, and (if calibrated) project the ball trajectory.

        ``progress`` is an optional ``callable(phase, done, total)`` for run feedback and
        ``max_frames`` caps how many frames are processed; both default to inert.
        """
        raw = self._raw_detections(video_path, stride, progress=progress, max_frames=max_frames)
        w, h = self._frame_size(video_path)
        try:  # recover fast-ball frames the CNN missed; never let it break detection
            recovered = self._recover_gaps(video_path, raw)
            if recovered:
                raw = sorted(list(raw) + recovered, key=lambda r: r[0])
        except Exception as exc:
            print(f"fast-ball recall: skipped ({exc})", flush=True)
        return self.postprocess(raw, homography, max_px_per_frame=self._gate_px(w, h))

    def _recover_gaps(self, video_path: str | Path, raw: list[tuple]) -> list[tuple]:
        """Recover dropped fast-ball frames in short gaps via a motion peak near the predicted path.

        For each gap (``2 <= Δframe <= _RECALL_MAX_GAP``) between consecutive detections, the search
        centre per missing frame is a linear interpolation between the two bracketing (real)
        detections; the ball is then localized by the frame-difference peak in a small crop there.
        Returns low-confidence ``(frame, x, y, conf, None)`` tuples to merge into ``raw``.
        """
        import cv2

        if len(raw) < 2:
            return []
        rs = sorted(raw, key=lambda r: r[0])
        gaps = [(a, b) for a, b in zip(rs, rs[1:]) if 2 <= int(b[0]) - int(a[0]) <= _RECALL_MAX_GAP]
        if not gaps:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        recovered: list[tuple] = []
        n_missing = 0
        try:
            for (f0, x0, y0, *_), (f1, x1, y1, *_) in gaps:
                f0, f1 = int(f0), int(f1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
                okp, prev = cap.read()  # frame f0 (the last real detection)
                if not okp:
                    continue
                prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                for f in range(f0 + 1, f1):
                    okc, cur = cap.read()  # next sequential frame == f
                    if not okc:
                        break
                    n_missing += 1
                    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
                    a = (f - f0) / (f1 - f0)
                    center = (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
                    peak = _motion_peak(prev_g, cur_g, center, _RECALL_CROP_HALF, _RECALL_MIN_SCORE)
                    if peak is not None:
                        recovered.append((f, peak[0], peak[1], _RECALL_CONF, None))
                    prev_g = cur_g
        finally:
            cap.release()
        print(f"fast-ball recall: recovered {len(recovered)}/{n_missing} arc-break gap frames",
              flush=True)
        return recovered

    def postprocess(
        self,
        raw: list[tuple[int, float, float, float]],
        homography: np.ndarray | None = None,
        max_px_per_frame: float | None = None,
    ) -> list[BallSample]:
        """Gate impossible jumps, Kalman-smooth, and project to court coords.

        Split out from model inference so it is unit-testable with synthetic detections. The jump
        gate is ``max_px_per_frame`` if given, else the tracker's configured/auto bound (``_gate_px``).
        """
        if not raw:
            return []
        raw = sorted(raw, key=lambda r: r[0])
        frames = np.array([r[0] for r in raw])
        xy = np.array([[r[1], r[2]] for r in raw], dtype=float)
        conf = np.array([r[3] for r in raw], dtype=float)
        # Apparent ball radius (depth cue) is an optional 5th field; NaN where absent so it survives
        # the keep-mask alignment, then mapped back to None per sample.
        radii = np.array([r[4] if len(r) > 4 and r[4] is not None else np.nan for r in raw],
                         dtype=float)

        gate = max_px_per_frame if max_px_per_frame is not None else self._gate_px()
        keep = gate_jumps(frames, xy, gate)
        frames, xy, conf, radii = frames[keep], xy[keep], conf[keep], radii[keep]
        smoothed = smooth(frames, xy)

        court = project(homography, smoothed) if homography is not None else None
        samples: list[BallSample] = []
        for i, f in enumerate(frames):
            samples.append(
                BallSample(
                    frame=int(f),
                    px=(float(smoothed[i, 0]), float(smoothed[i, 1])),
                    court_xy=(_court_xy_or_none(float(court[i, 0]), float(court[i, 1]))
                          if court is not None else None),
                    conf=float(conf[i]),
                    radius_px=(float(radii[i]) if np.isfinite(radii[i]) else None),
                )
            )
        return samples
