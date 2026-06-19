"""Unit tests for the script-side gap diagnostics + detections cache (scripts/debug_ball.py).

These are pure helpers (no torch / no model), so they're cheap to test and guard the thresholds
that decide whether the heavier Phase 2 detection work is warranted.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from debug_ball import _classify_gaps, _load_dets, _save_dets  # noqa: E402


def test_classify_gaps_buckets_by_downstream_behaviour():
    # Detected frames at 30 fps: a 4-frame step (bridged), a 10-frame step (arc-break, 7..18),
    # and a 30-frame step (rally-split, >18). Consecutive frames produce no gap.
    detected = [0, 1, 2, 6, 16, 46]  # deltas: 1,1,4,10,30
    buckets, rally_delta = _classify_gaps(detected, fps=30.0)

    assert rally_delta == 18
    assert [d for _, d in buckets["bridged"]] == [4]
    assert [d for _, d in buckets["arc_break"]] == [10]
    assert [(a, d) for a, d in buckets["rally_split"]] == [(16, 30)]


def test_classify_gaps_ignores_leading_and_handles_empty():
    # A late first detection is not an interior gap; a single detection has no gaps at all.
    assert _classify_gaps([100, 101, 102], fps=30.0)[0] == {
        "bridged": [], "arc_break": [], "rally_split": []}
    assert _classify_gaps([], fps=30.0)[0]["bridged"] == []


def test_dets_cache_roundtrips(tmp_path):
    raw = [(0, 100.0, 200.0, 0.9), (3, 110.5, 205.25, 0.5)]
    path = str(tmp_path / "dets.json")
    _save_dets(path, raw, 1920, 1080, 30.0)
    loaded, w, h, fps = _load_dets(path)
    assert (w, h, fps) == (1920, 1080, 30.0)
    assert loaded == raw  # exact (frame,x,y,conf) round-trip
