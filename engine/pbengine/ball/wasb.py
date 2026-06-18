"""Thin, Hydra-free wrapper around the vendored WASB-SBDT ball detector.

WASB-SBDT (``nttcom/WASB-SBDT``, BMVC2023) ships as a Hydra/omegaconf research framework whose
detector hard-requires a GPU. We bypass all of that: build its HRNet model directly, reuse its
postprocessor + online tracker + affine utils, and drive them with a plain config. Because we
skip WASB's GPU-asserting detector wrapper, this runs on **CPU too** (slowly) and on a GPU
(fast) — ``device`` is auto-selected.

Pipeline (mirrors ``runners/eval.py::inference_video``):
    3 consecutive frames -> affine-warp to 512x288 -> ImageNet-normalize -> stack to 9ch
    -> HRNet -> 3 heatmaps -> postprocess (connected-component blobs) -> per-frame online
    tracker -> (frame, x, y, conf) in source-pixel coords.

Only trained-weight accuracy on pickleball is unproven (tennis/badminton transfer is the known
risk); the code path itself is validated on CPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from pbengine.errors import ModelUnavailable

_SUBMODULE = Path(__file__).resolve().parents[1] / "third_party" / "WASB-SBDT"
_SRC = _SUBMODULE / "src"
_MODEL_W, _MODEL_H = 512, 288
_FRAMES_IN = 3
_STEP = 3  # non-overlapping windows (WASB detector.step=3)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _DotDict(dict):
    """dict that also allows attribute access (HRNet reads cfg.MODEL.EXTRA and cfg['...'])."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _wrap(obj):
    if isinstance(obj, dict):
        return _DotDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(x) for x in obj]
    return obj


def _import_wasb():
    """Import WASB's leaf modules, bypassing package __init__ files that pull Hydra/pandas."""
    if not (_SRC / "models" / "hrnet.py").exists():
        raise ModelUnavailable(
            "WASB-SBDT submodule missing. Run: "
            "git submodule update --init engine/pbengine/third_party/WASB-SBDT"
        )
    try:
        import numpy as _np  # noqa: F401

        if not hasattr(np, "Inf"):
            np.Inf = np.inf  # WASB online.py uses np.Inf, removed in NumPy 2.0
        for p in (str(_SRC), str(_SRC / "models"), str(_SRC / "detectors"), str(_SRC / "trackers")):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Stub the heavy 'utils' package so its __init__ (pandas/matplotlib) doesn't run; the
        # leaf modules we need (utils.utils, utils.image) import cleanly under it.
        if "utils" not in sys.modules:
            pkg = types.ModuleType("utils")
            pkg.__path__ = [str(_SRC / "utils")]
            sys.modules["utils"] = pkg
        import hrnet  # type: ignore
        import online  # type: ignore
        import postprocessor  # type: ignore
        from utils.image import get_affine_transform  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModelUnavailable(
            "WASB deps unavailable. Install the 'ml' extra: pip install -e '.[ml]'"
        ) from exc
    return hrnet, postprocessor, online, get_affine_transform


def _build_cfg(yaml_path: Path) -> _DotDict:
    import yaml

    model_cfg = yaml.safe_load(yaml_path.read_text())
    return _wrap(
        {
            "model": model_cfg,
            "detector": {
                "postprocessor": {
                    "name": "tracknetv2",
                    "score_threshold": 0.5,
                    "scales": [0],
                    "blob_det_method": "concomp",
                    "use_hm_weight": True,
                }
            },
            "dataloader": {"heatmap": {"sigmas": [2.5]}},
            "tracker": {"max_disp": 300},
        }
    )


class WasbBall:
    """Loaded WASB model + postprocessor + online tracker, ready to infer a video."""

    def __init__(self, weights: str, device: str | None = None):
        import torch

        hrnet, postprocessor, online, get_affine_transform = _import_wasb()
        if not Path(weights).exists():
            raise ModelUnavailable(
                f"WASB weights not found at {weights}. Run scripts/download_weights.sh."
            )
        self._torch = torch
        self._get_affine = get_affine_transform
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        cfg = _build_cfg(_SRC / "configs" / "model" / "wasb.yaml")
        model = hrnet.HRNet(cfg["model"])
        ckpt = torch.load(weights, map_location=self._device)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        self._model = model.to(self._device).eval()
        self._pp = postprocessor.TracknetV2Postprocessor(cfg)
        self._tracker = online.OnlineTracker(cfg)

    def _preprocess(self, frames, trans_input):
        import cv2

        ts = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            warp = cv2.warpAffine(rgb, trans_input, (_MODEL_W, _MODEL_H), flags=cv2.INTER_LINEAR)
            t = self._torch.from_numpy(warp).float().permute(2, 0, 1) / 255.0
            t = (t - self._torch.tensor(_MEAN)[:, None, None]) / self._torch.tensor(_STD)[:, None, None]
            ts.append(t)
        return self._torch.cat(ts, dim=0).unsqueeze(0).to(self._device)  # (1, 9, H, W)

    def infer_video(self, video_path: str) -> list[tuple[int, float, float, float]]:
        """Return ``(frame, x, y, conf)`` ball detections in source-pixel coords.

        Frames are streamed in non-overlapping windows of ``_FRAMES_IN`` rather than decoded
        all at once: a full clip in RAM is tens of GB (a 1080p frame is ~6 MB, so a few
        minutes at 30 fps OOM-kills the process). Memory now stays flat regardless of length.
        """
        import cv2

        torch = self._torch
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")

        ok, first = cap.read()
        if not ok:
            cap.release()
            return []

        h, w = first.shape[:2]
        c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        s = max(h, w) * 1.0
        trans_in = self._get_affine(c, s, 0, [_MODEL_W, _MODEL_H], inv=0)
        trans_inv = self._get_affine(c, s, 0, [_MODEL_W, _MODEL_H], inv=1)
        affine_mats = {0: torch.tensor(trans_inv).unsqueeze(0)}

        self._tracker.refresh()
        raw: list[tuple[int, float, float, float]] = []
        buf = [first]
        start = 0
        while buf:
            while len(buf) < _FRAMES_IN:
                ok, fr = cap.read()
                if not ok:
                    break
                buf.append(fr)
            n_real = len(buf)  # real (non-padded) frames in this window
            window = [buf[min(k, n_real - 1)] for k in range(_FRAMES_IN)]
            imgs = self._preprocess(window, trans_in)
            with torch.no_grad():
                preds = self._model(imgs)
            results = self._pp.run({0: preds[0]}, affine_mats)  # results[bid][eid][scale]
            for eid in sorted(results[0].keys()):
                dets = [
                    {"xy": xy, "score": sc}
                    for xy, sc in zip(results[0][eid][0]["xys"], results[0][eid][0]["scores"])
                ]
                out = self._tracker.update(dets)
                if out["visi"] and eid < n_real:  # skip padded tail frames
                    raw.append((start + eid, float(out["x"]), float(out["y"]), float(out["score"])))
            if n_real < _FRAMES_IN:
                break  # reached end of stream
            start += _STEP
            ok, fr = cap.read()  # seed the next non-overlapping window
            if not ok:
                break
            buf = [fr]
        cap.release()

        # Blob scores are unbounded; normalize to [0, 1] so they satisfy the BallSample schema.
        if raw:
            max_score = max(r[3] for r in raw) or 1.0
            raw = [(f, x, y, min(1.0, s_ / max_score)) for (f, x, y, s_) in raw]
        return raw
