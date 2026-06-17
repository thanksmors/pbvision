"""Court keypoint detection via the vendored ``yastrebksv/TennisCourtDetector`` net.

The net is a TrackNet-style heatmap model (``BallTrackerNet``, 15 output channels) that
predicts 14 tennis-court keypoints on a 640x360 input. Of those, the first four — indices
0..3 — are the **outer court corners** (top-left, top-right, bottom-left, bottom-right). Those
are the only points that carry over to pickleball; the rest mark tennis-specific inner lines
(singles sidelines, service lines) at distances that don't exist on a pickleball court.

So we take just the four outer corners and map them to the normalized pickleball reference
corners (``court_model.REFERENCE_POINTS``). A 4-corner homography to the unit square is exact
regardless of tennis-vs-pickleball proportions — the only question is whether a *tennis*-
trained net localizes a *pickleball* court's corners well, which we accept as the known risk
of the automatic approach. With a static camera this is solved once and reused for the shot.

Heavy imports (torch + the submodule) are lazy, so importing this module stays cheap and the
rest of the engine runs without the ``ml`` extra.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pbengine.court.homography import homography_from_named_points
from pbengine.errors import CourtNotFound, ModelUnavailable

# Vendored submodule and the default weights location (fetched by scripts/download_weights.sh).
_SUBMODULE = Path(__file__).resolve().parents[1] / "third_party" / "TennisCourtDetector"
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "court_detector.pt"

# Model I/O geometry. Input is 640x360; postprocess rescales heatmap peaks by 2, yielding
# coordinates in the dataset's native 1280x720 space, which we then scale to the real frame.
_MODEL_W, _MODEL_H = 640, 360
_REF_W, _REF_H = 1280, 720

# Outer-corner keypoint indices -> our normalized pickleball reference-point names.
_CORNER_NAMES = {
    0: "corner_a_left",   # baseline_top left
    1: "corner_a_right",  # baseline_top right
    2: "corner_b_left",   # baseline_bottom left
    3: "corner_b_right",  # baseline_bottom right
}


def corners_to_named(
    points: list[tuple[float | None, float | None]], frame_w: int, frame_h: int
) -> dict[str, tuple[float, float]]:
    """Map raw 14-keypoint predictions to named pickleball corners, scaled to the frame.

    ``points`` are ``(x, y)`` in 1280x720 reference space (``None`` where a point wasn't
    found). Pure function — unit-tested without torch.
    """
    named: dict[str, tuple[float, float]] = {}
    for idx, name in _CORNER_NAMES.items():
        x, y = points[idx]
        if x is None or y is None:
            continue
        named[name] = (x / _REF_W * frame_w, y / _REF_H * frame_h)
    return named


@dataclass
class CourtDetector:
    weights: str = str(_DEFAULT_WEIGHTS)
    low_thresh: int = 170
    max_radius: int = 25
    _model: object = field(default=None, repr=False)
    _postprocess: object = field(default=None, repr=False)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not (_SUBMODULE / "tracknet.py").exists():
            raise ModelUnavailable(
                "TennisCourtDetector submodule missing. Run: "
                "git submodule update --init engine/pbengine/third_party/TennisCourtDetector"
            )
        if not Path(self.weights).exists():
            raise ModelUnavailable(
                f"court weights not found at {self.weights}. Run scripts/download_weights.sh."
            )
        try:
            import torch  # local heavy import

            if str(_SUBMODULE) not in sys.path:
                sys.path.insert(0, str(_SUBMODULE))  # let the submodule's bare imports resolve
            from postprocess import postprocess  # type: ignore
            from tracknet import BallTrackerNet  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailable(
                "court detector deps unavailable. Install the 'ml' extra: pip install -e '.[ml]'"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BallTrackerNet(out_channels=15).to(device)
        model.load_state_dict(torch.load(self.weights, map_location=device))
        model.eval()
        self._model = (model, device, torch)
        self._postprocess = postprocess

    def _predict_points(self, frame) -> list[tuple[float | None, float | None]]:
        """Run the net on one BGR frame -> 14 ``(x, y)`` keypoints in 1280x720 space."""
        import cv2

        self._ensure_model()
        model, device, torch = self._model  # type: ignore[misc]
        img = cv2.resize(frame, (_MODEL_W, _MODEL_H)).astype(np.float32) / 255.0
        inp = torch.tensor(np.rollaxis(img, 2, 0)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(inp)[0]
            pred = torch.sigmoid(out).cpu().numpy()
        points: list[tuple[float | None, float | None]] = []
        for k in range(14):
            heatmap = (pred[k] * 255).astype(np.uint8)
            x, y = self._postprocess(  # type: ignore[misc]
                heatmap, low_thresh=self.low_thresh, max_radius=self.max_radius
            )
            points.append((x, y))
        return points

    def detect(self, frame) -> dict[str, tuple[float, float]]:
        """Return named outer-corner pixel keypoints for one BGR frame."""
        h, w = frame.shape[:2]
        return corners_to_named(self._predict_points(frame), w, h)

    def solve(self, frame) -> np.ndarray:
        """Detect corners on a frame and return the pixel->court_xy homography.

        Raises :class:`CourtNotFound` if fewer than four corners were localized.
        """
        named = self.detect(frame)
        if len(named) < 4:
            raise CourtNotFound(f"only {len(named)}/4 court corners localized")
        return homography_from_named_points(named)
