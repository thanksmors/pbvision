"""Player detection + tracking via Ultralytics YOLO26 + ByteTrack.

People are COCO class 0, so no training is needed for v1. The Ultralytics dependency is
AGPL-3.0 — fine for a localhost single-user app, but revisit (Enterprise license or a
permissively-licensed detector) before distributing a build or running it as a service for
others. Heavy imports are local so the rest of the engine works without the ``ml`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pbengine.errors import ModelUnavailable

_PERSON_CLASS = 0


@dataclass
class PlayerTrack:
    track_id: int
    frame: int
    bbox_px: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def foot_px(self) -> tuple[float, float]:
        """Approximate ground-contact point: bottom-center of the bbox."""
        x1, _y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, y2)


@dataclass
class PlayerDetector:
    """Lazy wrapper around an Ultralytics model run in tracking mode.

    ``weights`` defaults to the medium model; on a CPU dev box pass ``yolo26n.pt`` and a
    ``vid_stride`` > 1 to keep runtimes sane (the ball/winner logic tolerates sparser player
    sampling). Ultralytics auto-downloads the weights on first use.
    """

    weights: str = "yolo26m.pt"
    tracker: str = "bytetrack.yaml"
    conf: float = 0.3
    vid_stride: int = 1
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

    def track(self, video_path: str | Path) -> list[PlayerTrack]:
        """Run detection+tracking over a video, returning per-frame player tracks.

        Frame indices are reported in *source-video* space (``processed_index * vid_stride``)
        so they line up with the court homography and ball trajectory regardless of striding.
        """
        self._ensure_model()
        tracks: list[PlayerTrack] = []
        results = self._model.track(  # type: ignore[union-attr]
            source=str(video_path),
            tracker=self.tracker,
            classes=[_PERSON_CLASS],
            conf=self.conf,
            vid_stride=self.vid_stride,
            stream=True,
            verbose=False,
        )
        for proc_idx, res in enumerate(results):
            if res.boxes is None or res.boxes.id is None:
                continue
            frame = proc_idx * self.vid_stride
            for box, tid in zip(res.boxes.xyxy.tolist(), res.boxes.id.tolist()):
                tracks.append(
                    PlayerTrack(track_id=int(tid), frame=frame, bbox_px=tuple(box))
                )
        return tracks
