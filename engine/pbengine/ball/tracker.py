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
from pbengine.court.homography import project
from pbengine.schema.models import BallSample

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "wasb_tennis_best.pth.tar"


@dataclass
class BallTracker:
    weights: str = str(_DEFAULT_WEIGHTS)
    max_px_per_frame: float = 150.0
    device: str | None = None
    # WASB detector knobs (see pbengine.ball.wasb.WasbBall). Defaults favour recall: a lower blob
    # threshold and overlapping windows (step=1) so each frame gets up to 3 detection attempts.
    score_threshold: float = 0.3
    max_disp: float = 300.0
    step: int = 1
    _model: object = field(default=None, repr=False)

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
