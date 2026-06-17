"""Ball detection + tracking — the highest-risk stage.

Primary detector is WASB-SBDT (``nttcom/WASB-SBDT``, vendored under ``third_party/``), using
its tennis/badminton weights as the pickleball starting point (ball size/speed sits between
the two). ``AndrewDettor/TrackNet-Pickleball`` is a fallback weights source. Raw heatmap
detections are gated for physically-impossible jumps and Kalman-smoothed (see
:mod:`pbengine.ball.kalman`).

The WASB import is lazy and lives behind :meth:`BallTracker.detect`, so the rest of the
engine imports cleanly on a CPU box without the model. v1 development should spend most of
its effort validating *this* module on real footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pbengine.ball.kalman import gate_jumps, smooth
from pbengine.court.homography import project
from pbengine.schema.models import BallSample


@dataclass
class BallTracker:
    weights: str = "wasb_tennis.pth.tar"
    max_px_per_frame: float = 150.0
    _model: object = field(default=None, repr=False)

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                # Vendored submodule; see scripts/download_weights.sh and third_party/.
                from wasb_sbdt import load_default_model  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "WASB-SBDT not available. Initialize the submodule under "
                    "engine/pbengine/third_party/WASB-SBDT and fetch weights via "
                    "scripts/download_weights.sh."
                ) from exc
            self._model = load_default_model(self.weights)

    def _raw_detections(self, video_path: str | Path, stride: int) -> list[tuple[int, float, float, float]]:
        """Return ``(frame, x, y, conf)`` heatmap detections from WASB. Lazy/model-backed."""
        self._ensure_model()
        return self._model.infer_video(str(video_path), stride=stride)  # type: ignore[union-attr]

    def track(
        self,
        video_path: str | Path,
        homography: np.ndarray | None = None,
        stride: int = 1,
    ) -> list[BallSample]:
        """Detect, gate, smooth, and (if calibrated) project the ball trajectory."""
        raw = self._raw_detections(video_path, stride)
        return self.postprocess(raw, homography)

    def postprocess(
        self,
        raw: list[tuple[int, float, float, float]],
        homography: np.ndarray | None = None,
    ) -> list[BallSample]:
        """Gate impossible jumps, Kalman-smooth, and project to court coords.

        Split out from model inference so it is unit-testable with synthetic detections.
        """
        if not raw:
            return []
        raw = sorted(raw, key=lambda r: r[0])
        frames = np.array([r[0] for r in raw])
        xy = np.array([[r[1], r[2]] for r in raw], dtype=float)
        conf = np.array([r[3] for r in raw], dtype=float)

        keep = gate_jumps(frames, xy, self.max_px_per_frame)
        frames, xy, conf = frames[keep], xy[keep], conf[keep]
        smoothed = smooth(frames, xy)

        court = project(homography, smoothed) if homography is not None else None
        samples: list[BallSample] = []
        for i, f in enumerate(frames):
            samples.append(
                BallSample(
                    frame=int(f),
                    px=(float(smoothed[i, 0]), float(smoothed[i, 1])),
                    court_xy=(float(court[i, 0]), float(court[i, 1])) if court is not None else None,
                    conf=float(conf[i]),
                )
            )
        return samples
