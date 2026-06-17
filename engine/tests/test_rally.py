from pbengine.rally.segmentation import segment_rallies


def test_splits_on_long_gap():
    fps = 30.0
    # Rally 1: frames 0..60 (2s). Gap of ~1s (> 0.6s tol). Rally 2: 120..200.
    present = list(range(0, 61)) + list(range(120, 201))
    spans = segment_rallies(present, fps, max_gap_s=0.6, min_rally_s=1.0)
    assert len(spans) == 2
    assert (spans[0].start_frame, spans[0].end_frame) == (0, 60)
    assert (spans[1].start_frame, spans[1].end_frame) == (120, 200)


def test_bridges_short_gaps():
    fps = 30.0
    # A 10-frame occlusion (< 0.6s = 18 frames) should NOT split the rally.
    present = list(range(0, 30)) + list(range(40, 90))
    spans = segment_rallies(present, fps)
    assert len(spans) == 1
    assert spans[0].start_frame == 0 and spans[0].end_frame == 89


def test_drops_short_noise():
    fps = 30.0
    # A 5-frame blip is below the 1.0s minimum and must be discarded.
    present = [5, 6, 7, 8, 9] + list(range(100, 140))
    spans = segment_rallies(present, fps)
    assert len(spans) == 1
    assert spans[0].start_frame == 100


def test_empty_input():
    assert segment_rallies([], 30.0) == []
