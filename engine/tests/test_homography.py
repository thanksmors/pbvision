import numpy as np

from pbengine.court import homography_from_named_points, project
from pbengine.court.court_model import REFERENCE_POINTS
from pbengine.court.homography import compute_homography


def test_identity_recovers_points():
    # A simple affine-ish pixel layout that maps exactly onto the reference corners.
    named_px = {
        "corner_a_left": (100.0, 100.0),
        "corner_a_right": (900.0, 100.0),
        "corner_b_left": (100.0, 1000.0),
        "corner_b_right": (900.0, 1000.0),
    }
    h = homography_from_named_points(named_px)
    for name, px in named_px.items():
        got = project(h, px)[0]
        np.testing.assert_allclose(got, REFERENCE_POINTS[name], atol=1e-6)


def test_perspective_homography_roundtrip():
    # Build a random homography, push reference points through it to get "pixels",
    # then check we recover the same homography (up to scale) from those correspondences.
    src = np.array(list(REFERENCE_POINTS.values())[:6])
    h_true = np.array([[1.2, 0.1, 50.0], [0.05, 1.4, 30.0], [1e-4, 2e-4, 1.0]])
    px = project(h_true, src)
    h_est = compute_homography(px, src)
    back = project(h_est, px)
    np.testing.assert_allclose(back, src, atol=1e-6)


def test_requires_four_points():
    try:
        homography_from_named_points({"corner_a_left": (0.0, 0.0)})
    except ValueError as exc:
        assert ">= 4" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for < 4 points")
