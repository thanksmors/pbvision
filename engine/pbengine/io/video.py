"""Video probing and frame iteration.

Uses OpenCV for portability on the CPU dev box. On the rented GPU box, swap the frame
iterator for TorchCodec (CUDA decode) without changing callers. Heavy imports are local so
``import pbengine`` works on a machine with only the core dependencies.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pbengine.schema.models import VideoMeta


def probe(path: str | Path) -> VideoMeta:
    """Read fps / frame count / resolution without decoding the whole file."""
    import cv2  # local import: only needed when actually processing a video

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoMeta(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path, stride: int = 1) -> Iterator[tuple[int, "object"]]:
    """Yield ``(frame_index, bgr_frame)``. ``stride > 1`` samples every Nth frame.

    Frame sampling for the ball net (e.g. ``stride=2`` + interpolation) roughly halves
    GPU cost on a full match for a minor accuracy hit.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
