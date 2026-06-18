"""WASB inference plumbing: streaming windows, overlap fusion, frame indexing.

Exercises ``WasbBall.infer_video`` with the heavy model/postprocessor/tracker stubbed out, so the
sliding-window streaming, overlap candidate-pooling, and per-frame tracking pass are tested without
weights or a GPU. (Detection *quality* is a separate, footage-dependent question.)
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from pbengine.ball import wasb
from pbengine.ball.wasb import WasbBall


def _make_video(path, n, w=64, h=48):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 7) % 255, np.uint8))
    vw.release()


def _stub(step, candidates_per_frame):
    """A WasbBall with stubbed model/pp/tracker; records the #candidates fed to the tracker."""
    b = object.__new__(WasbBall)
    b._torch = torch
    b._step = step
    b._get_affine = lambda c, s, r, sz, inv: np.eye(2, 3, dtype=np.float32)
    b._preprocess = lambda window, trans: torch.zeros(1)
    b._model = lambda imgs: [None]

    class PP:
        def run(self, preds, mats):  # one blob per eid
            return {0: {e: {0: {"xys": [(5.0, 5.0)], "scores": [1.0]}}
                        for e in range(wasb._FRAMES_IN)}}

    class TR:
        frame = -1

        def refresh(self):
            pass

        def update(self, dets):
            self.frame += 1
            candidates_per_frame[self.frame] = len(dets)
            if dets:
                d = dets[0]
                return {"visi": True, "x": d["xy"][0], "y": d["xy"][1], "score": d["score"]}
            return {"visi": False, "x": 0.0, "y": 0.0, "score": 0.0}

    b._pp, b._tracker = PP(), TR()
    return b


def test_every_frame_tracked_in_order(tmp_path):
    path = tmp_path / "v.mp4"
    _make_video(path, 12)
    raw = _stub(step=1, candidates_per_frame={}).infer_video(str(path))
    assert [r[0] for r in raw] == list(range(12))  # streamed, ordered, no full-buffer OOM


@pytest.mark.parametrize("step,interior", [(1, 3), (3, 1)])
def test_overlap_pools_more_candidates(tmp_path, step, interior):
    path = tmp_path / f"v{step}.mp4"
    n = 12
    _make_video(path, n)
    seen: dict[int, int] = {}
    _stub(step=step, candidates_per_frame=seen).infer_video(str(path))
    # An interior frame is scored in `interior` overlapping windows (3 for step=1, 1 for step=3).
    assert seen[n // 2] == interior


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _stub(step=1, candidates_per_frame={}).infer_video(str(tmp_path / "nope.mp4"))
