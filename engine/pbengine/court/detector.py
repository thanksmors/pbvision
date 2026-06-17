"""Court keypoint detection via a TennisCourtDetector-style heatmap net.

Reuses ``yastrebksv/TennisCourtDetector`` (vendored under ``third_party/``), whose 14-point
output we map by name to the pickleball reference template in :mod:`court_model`. With a
static camera the keypoints are detected on a representative frame and the homography is
solved once for the whole shot.

The net is trained on tennis lines, so on pickleball the kitchen line is the most likely to
be mislocalized; fine-tuning on a few hundred labeled pickleball frames is a v1.1 task. Heavy
imports are lazy. When the model/weights are unavailable, callers can fall back to manually
supplied keypoints (see :func:`homography_from_named_points`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pbengine.court.homography import homography_from_named_points
from pbengine.errors import ModelUnavailable


@dataclass
class CourtDetector:
    weights: str = "court_detector.pth"
    _model: object = field(default=None, repr=False)

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from tenniscourtdetector import load_model  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailable(
                    "TennisCourtDetector not available. Initialize the submodule under "
                    "engine/pbengine/third_party/TennisCourtDetector and fetch weights."
                ) from exc
            self._model = load_model(self.weights)

    def detect(self, frame) -> dict[str, tuple[float, float]]:
        """Return named pixel keypoints for one BGR frame, mapped to reference-point names."""
        self._ensure_model()
        return self._model.predict_named(frame)  # type: ignore[union-attr]

    def solve(self, frame) -> np.ndarray:
        """Detect keypoints on a frame and return the pixel->court_xy homography."""
        return homography_from_named_points(self.detect(frame))
