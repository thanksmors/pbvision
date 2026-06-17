"""Manual court calibration: clicked corners drive the homography, no model needed."""

import numpy as np
import pytest

from pbengine.court.detector import (
    CORNER_ORDER,
    ManualCourtDetector,
    corners_from_clicks,
    load_corners,
)
from pbengine.court.homography import project
from pbengine.errors import CourtNotFound


def test_corners_from_clicks_maps_in_order():
    pts = [(10, 20), (300, 22), (5, 400), (310, 405)]
    named = corners_from_clicks(pts)
    assert list(named) == list(CORNER_ORDER)
    assert named["corner_a_left"] == (10.0, 20.0)
    assert named["corner_b_right"] == (310.0, 405.0)


def test_corners_from_clicks_requires_four():
    with pytest.raises(ValueError):
        corners_from_clicks([(0, 0), (1, 1)])


def test_manual_detector_solves_homography():
    # A frame-filling rectangle maps to the unit court.
    named = corners_from_clicks([(0, 0), (1280, 0), (0, 720), (1280, 720)])
    det = ManualCourtDetector(named)
    h = det.solve()
    assert np.allclose(project(h, (0.0, 0.0))[0], (0.0, 0.0), atol=1e-6)
    assert np.allclose(project(h, (1280.0, 720.0))[0], (1.0, 1.0), atol=1e-6)
    assert det.detect()["corner_a_right"] == (1280.0, 0.0)


def test_manual_detector_needs_four_corners():
    det = ManualCourtDetector({"corner_a_left": (0.0, 0.0)})
    with pytest.raises(CourtNotFound):
        det.solve()


def test_load_corners_roundtrip(tmp_path):
    import json

    p = tmp_path / "court.json"
    p.write_text(json.dumps({"corner_a_left": [1, 2], "corner_a_right": [3, 4]}))
    loaded = load_corners(p)
    assert loaded == {"corner_a_left": (1.0, 2.0), "corner_a_right": (3.0, 4.0)}
