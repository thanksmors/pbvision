"""Player detection + tracking via Ultralytics YOLO + ByteTrack, with pose keypoints.

People are COCO class 0, so no training is needed for v1. Defaulting to a ``-pose`` model gives 17
COCO keypoints (skeletons) **and** person boxes + track ids in the same pass — pose models are a
superset of the detector, so this costs ≈ the same as detect-only. The Ultralytics dependency is
AGPL-3.0 — fine for a localhost single-user app, but revisit (Enterprise license or a
permissively-licensed detector) before distributing a build or running it as a service for
others. Heavy imports are local so the rest of the engine works without the ``ml`` extra.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pbengine.errors import ModelUnavailable

_PERSON_CLASS = 0


@dataclass
class PlayerTrack:
    track_id: int
    frame: int
    bbox_px: tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints_px: list[tuple[float, float]] | None = None  # COCO-17 (x, y), if a pose model was used
    keypoint_conf: list[float] | None = None               # per-keypoint confidence [0, 1]

    @property
    def foot_px(self) -> tuple[float, float]:
        """Approximate ground-contact point: bottom-center of the bbox."""
        x1, _y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, y2)


@dataclass
class PlayerDetector:
    """Lazy wrapper around an Ultralytics model run in tracking mode.

    ``weights`` defaults to a medium **pose** model so each player carries a COCO-17 skeleton; pass
    a plain detector (e.g. ``yolo26m.pt``) to skip keypoints, or ``yolo11n-pose.pt`` + a
    ``vid_stride`` > 1 on a CPU dev box to keep runtimes sane (the ball/winner logic tolerates
    sparser player sampling). Ultralytics auto-downloads the weights on first use.
    """

    weights: str = "yolo11m-pose.pt"
    tracker: str = "bytetrack.yaml"
    conf: float = 0.3
    vid_stride: int = 1
    imgsz: int | None = None   # inference resolution; None = Ultralytics default (640). Higher (e.g.
                               # 1280) resolves small/far players at the cost of speed (~quadratic).
    augment: bool = False      # test-time augmentation (multi-scale); helps small players, ~2-3x slower
    _model: object = field(default=None, repr=False)

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from ultralytics import YOLO  # local heavy import
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailable(
                    "ultralytics not installed. Install the 'ml' extra: pip install -e '.[ml]'"
                ) from exc
            self._model = YOLO(self.weights)

    def track(
        self,
        video_path: str | Path,
        *,
        total_frames: int | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[PlayerTrack]:
        """Run detection+tracking over a video, returning per-frame player tracks.

        Frame indices are reported in *source-video* space (``processed_index * vid_stride``)
        so they line up with the court homography and ball trajectory regardless of striding.

        ``total_frames`` (source-video frame count) enables a throttled progress log with an ETA;
        ``progress_cb(done, total)`` is called on the same throttle so the API can advance its bar.
        Both are optional — omitted, behavior is unchanged.
        """
        self._ensure_model()
        tracks: list[PlayerTrack] = []
        # Only pass imgsz/augment when set so the defaults (and the fake-model tests) are unaffected.
        extra: dict[str, object] = {}
        if self.imgsz is not None:
            extra["imgsz"] = self.imgsz
        if self.augment:
            extra["augment"] = True
        results = self._model.track(  # type: ignore[union-attr]
            source=str(video_path),
            tracker=self.tracker,
            classes=[_PERSON_CLASS],
            conf=self.conf,
            vid_stride=self.vid_stride,
            stream=True,
            verbose=False,
            **extra,
        )
        # Processed-frame total (account for striding) for the percent/ETA readout.
        total_proc = max(1, math.ceil(total_frames / self.vid_stride)) if total_frames else 0
        t0 = last = time.monotonic()
        proc_idx = -1
        for proc_idx, res in enumerate(results):
            done = proc_idx + 1
            now = time.monotonic()
            if now - last >= 2.0:  # throttle so the log/ETA isn't per-frame spam
                last = now
                _report_progress("players", done, total_proc, now - t0)
                if progress_cb is not None:
                    progress_cb(done, total_proc)
            if res.boxes is None or res.boxes.id is None:
                continue
            frame = proc_idx * self.vid_stride
            # Pose models attach keypoints aligned by box index; detect-only models leave them None.
            kpts_xy = kpts_cf = None
            if getattr(res, "keypoints", None) is not None and res.keypoints.xy is not None:
                kpts_xy = res.keypoints.xy.tolist()
                kpts_cf = res.keypoints.conf.tolist() if res.keypoints.conf is not None else None
            for i, (box, tid) in enumerate(zip(res.boxes.xyxy.tolist(), res.boxes.id.tolist())):
                kp = [(float(x), float(y)) for x, y in kpts_xy[i]] if kpts_xy is not None else None
                cf = [float(c) for c in kpts_cf[i]] if kpts_cf is not None else None
                tracks.append(
                    PlayerTrack(track_id=int(tid), frame=frame, bbox_px=tuple(box),
                                keypoints_px=kp, keypoint_conf=cf)
                )
        done = proc_idx + 1
        if done:
            _report_progress("players", done, total_proc or done, time.monotonic() - t0)
            if progress_cb is not None:
                progress_cb(done, total_proc or done)
        return tracks


def _fmt_dur(secs: float) -> str:
    secs = int(max(secs, 0))
    return f"{secs // 60}:{secs % 60:02d}"


def _report_progress(stage: str, done: int, total: int, elapsed: float) -> None:
    """Print a throttled, flushed progress line with rate + ETA (captured to the per-job log)."""
    rate = done / elapsed if elapsed > 0 else 0.0
    if total > 0:
        pct = 100.0 * done / total
        eta = (total - done) / rate if rate > 0 else 0.0
        msg = f"{stage}: {done}/{total} ({pct:.0f}%) · {rate:.1f} fps · " \
              f"elapsed {_fmt_dur(elapsed)} · ETA {_fmt_dur(eta)}"
    else:
        msg = f"{stage}: {done} frames · {rate:.1f} fps · elapsed {_fmt_dur(elapsed)}"
    print(msg, flush=True)
