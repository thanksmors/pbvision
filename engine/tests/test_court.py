"""Court adapter: corner mapping + homography wiring, without torch or weights.

The neural net itself needs the ml extra + weights (smoke-tested separately), so here we
test the pure coordinate logic and the CourtNotFound path by stubbing the raw predictions.
"""

import numpy as np
import pytest

from pbengine.court.detector import CourtDetector, corners_to_named
from pbengine.court.homography import project
from pbengine.errors import CourtNotFound


def test_corners_to_named_scales_and_maps():
    # 14 points in 1280x720 ref space; only the 4 corners (indices 0-3) matter.
    pts = [(None, None)] * 14
    pts[0] = (0, 0)          # top-left
    pts[1] = (1280, 0)       # top-right
    pts[2] = (0, 720)        # bottom-left
    pts[3] = (1280, 720)     # bottom-right
    named = corners_to_named(pts, frame_w=640, frame_h=360)  # half-res frame
    assert named == {
        "corner_a_left": (0.0, 0.0),
        "corner_a_right": (640.0, 0.0),
        "corner_b_left": (0.0, 360.0),
        "corner_b_right": (640.0, 360.0),
    }


def test_corners_to_named_drops_missing():
    pts = [(None, None)] * 14
    pts[0] = (100, 100)
    pts[1] = (200, 100)
    named = corners_to_named(pts, 1280, 720)
    assert set(named) == {"corner_a_left", "corner_a_right"}  # only 2 found


def test_solve_raises_when_too_few_corners(monkeypatch):
    det = CourtDetector()
    pts = [(None, None)] * 14
    pts[0] = (10, 10)  # only one corner
    monkeypatch.setattr(det, "_predict_points", lambda frame: pts)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(CourtNotFound):
        det.solve(frame)


def test_solve_homography_maps_corner_to_origin(monkeypatch):
    det = CourtDetector()
    pts = [(None, None)] * 14
    pts[0], pts[1], pts[2], pts[3] = (0, 0), (1280, 0), (0, 720), (1280, 720)
    monkeypatch.setattr(det, "_predict_points", lambda frame: pts)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    h = det.solve(frame)
    assert h.shape == (3, 3)
    # The top-left corner pixel must land at normalized court origin (0, 0).
    xy = project(h, (0.0, 0.0))[0]
    assert np.allclose(xy, (0.0, 0.0), atol=1e-6)
    # The opposite corner maps to (1, 1).
    xy2 = project(h, (1280.0, 720.0))[0]
    assert np.allclose(xy2, (1.0, 1.0), atol=1e-6)
