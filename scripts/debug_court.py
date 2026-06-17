"""Diagnose court detection on a clip's first frame.

Prints, per court keypoint channel, the model's peak activation and where it lands, plus
what the postprocess step returns. Use it to tell a genuine transfer failure (flat heatmaps)
from a too-strict postprocess threshold (strong peaks but no detection).

    python scripts/debug_court.py path/to/clip.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from pbengine.court.detector import _MODEL_H, _MODEL_W, _REF_H, _REF_W, CourtDetector  # noqa: E402


def main(video: str) -> int:
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"cannot read a frame from {video}")
        return 1
    h, w = frame.shape[:2]
    print(f"frame: {w}x{h}")

    det = CourtDetector()
    det._ensure_model()  # raises ModelUnavailable with a clear message if not set up
    model, device, torch = det._model

    img = cv2.resize(frame, (_MODEL_W, _MODEL_H)).astype(np.float32) / 255.0
    inp = torch.tensor(np.rollaxis(img, 2, 0)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(inp)[0]).cpu().numpy()

    corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    print("\nchannel        peak   peak-location(frame px)   postprocess")
    print("-" * 64)
    for k in range(14):
        hm = pred[k]
        peak = float(hm.max())
        yx = np.unravel_index(int(hm.argmax()), hm.shape)  # (y, x) in 360x640
        px = (yx[1] / _MODEL_W * w, yx[0] / _MODEL_H * h)
        heatmap_u8 = (hm * 255).astype(np.uint8)
        pp = det._postprocess(heatmap_u8, low_thresh=det.low_thresh, max_radius=det.max_radius)
        pp_frame = None
        if pp[0] is not None:
            pp_frame = (pp[0] / _REF_W * w, pp[1] / _REF_H * h)
        tag = f"  [{corner_names[k]}]" if k < 4 else ""
        print(
            f"{k:>2}  peak={peak:4.2f}  argmax=({px[0]:6.0f},{px[1]:6.0f})  "
            f"postproc={'None' if pp_frame is None else f'({pp_frame[0]:.0f},{pp_frame[1]:.0f})'}{tag}"
        )

    corner_peaks = [float(pred[k].max()) for k in range(4)]
    print(f"\nmean corner peak activation: {np.mean(corner_peaks):.3f}")
    if np.mean(corner_peaks) < 0.3:
        print("=> Heatmaps are weak: the tennis model isn't recognizing this court "
              "(transfer failure). Manual calibration is the reliable fix.")
    else:
        print("=> Strong peaks but postprocess missed them: try lowering --low-thresh / "
              "loosening HoughCircles. Tunable, not a transfer failure.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/debug_court.py path/to/clip.mp4")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
