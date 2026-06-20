"""Player-detection presets resolve to a configured PlayerDetector, with per-knob overrides."""

import pytest

from pbengine.detect.presets import (
    DEFAULT_PRESET,
    PRESETS,
    SENSITIVE_TRACKER,
    build_player_detector,
)


def test_each_preset_builds_a_detector():
    for name in PRESETS:
        det = build_player_detector(name)
        assert det.weights.endswith("-pose.pt")
        assert det.imgsz is not None and det.vid_stride >= 1
        assert 0.0 < det.conf <= 1.0


def test_balanced_is_more_sensitive_than_fast():
    fast, bal = build_player_detector("fast"), build_player_detector("balanced")
    assert bal.imgsz > fast.imgsz          # higher resolution -> far players
    assert bal.conf < fast.conf            # lower floor -> faint detections
    assert bal.vid_stride < fast.vid_stride  # every frame -> dense, articulated poses
    assert bal.tracker == SENSITIVE_TRACKER and fast.tracker == "bytetrack.yaml"


def test_default_preset_is_balanced():
    assert DEFAULT_PRESET == "balanced"
    assert build_player_detector().imgsz == PRESETS["balanced"]["imgsz"]


def test_overrides_apply_only_when_not_none():
    det = build_player_detector("balanced", imgsz=1536, conf=None, weights=None)
    assert det.imgsz == 1536                       # explicit override wins
    assert det.conf == PRESETS["balanced"]["conf"]  # None -> keep the preset value
    assert det.weights == PRESETS["balanced"]["weights"]


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        build_player_detector("turbo")
