"""Named player-detection presets — one knob to trade speed for capture.

The default detector (nano weights, 640px inference, stride 3, conf 0.3) misses far/small players and
produces sparse poses, so skeletons glide instead of walking. These presets bundle the levers that
matter — model size, inference ``imgsz``, ``conf``, frame stride, and a sensitive ByteTrack config —
into a few choices, from ``fast`` (today's behavior) to ``gpu``. The pipeline CLI and the API both
build the detector through :func:`build_player_detector`, which applies a preset and then any explicit
per-knob overrides (so ``--players-imgsz 1536`` still works on top of a preset).

See :mod:`pbengine.detect.players` for the knobs and ``bytetrack_sensitive.yaml`` for the tracker.
"""

from __future__ import annotations

from pathlib import Path

from pbengine.detect.players import PlayerDetector

# Absolute path so Ultralytics loads our config regardless of the caller's cwd.
SENSITIVE_TRACKER = str(Path(__file__).with_name("bytetrack_sensitive.yaml"))

# Each preset is a full set of PlayerDetector kwargs. Heavier presets resolve small/far players and
# give dense, real poses (natural motion) at the cost of speed (imgsz is ~quadratic; stride 1 is ~3x).
PRESETS: dict[str, dict] = {
    # Today's CPU-friendly behavior — fast but misses far players / glides.
    "fast": {"weights": "yolo11n-pose.pt", "imgsz": 640, "conf": 0.30, "vid_stride": 3,
             "tracker": "bytetrack.yaml", "augment": False},
    # Good default: real far-player capture and articulated motion, tractable on CPU for short clips.
    "balanced": {"weights": "yolo11m-pose.pt", "imgsz": 960, "conf": 0.15, "vid_stride": 1,
                 "tracker": SENSITIVE_TRACKER, "augment": False},
    # Maximize capture; slow on CPU.
    "max": {"weights": "yolo11m-pose.pt", "imgsz": 1280, "conf": 0.10, "vid_stride": 1,
            "tracker": SENSITIVE_TRACKER, "augment": True},
    # For a GPU box: largest model, high resolution, every frame.
    "gpu": {"weights": "yolo11x-pose.pt", "imgsz": 1280, "conf": 0.10, "vid_stride": 1,
            "tracker": SENSITIVE_TRACKER, "augment": False},
}

DEFAULT_PRESET = "balanced"


def build_player_detector(preset: str = DEFAULT_PRESET, **overrides) -> PlayerDetector:
    """Build a :class:`PlayerDetector` from a named preset, applying any non-``None`` overrides.

    ``overrides`` are detector kwargs (``weights``, ``imgsz``, ``conf``, ``vid_stride``, ``augment``,
    ``tracker``); a ``None`` value means "not specified — keep the preset's". Raises ``KeyError`` (with
    the valid names) for an unknown preset.
    """
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    kwargs = dict(PRESETS[preset])
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return PlayerDetector(**kwargs)
