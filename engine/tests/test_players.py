"""PlayerDetector.track() adapts the Ultralytics streaming API into PlayerTrack rows.

We mock the model so the parsing/striding logic is tested without downloading weights or
installing torch. Frame indices must be reported in source-video space (proc_index * stride).
"""

from types import SimpleNamespace

from pbengine.detect.players import PlayerDetector, PlayerTrack


class _Tensor(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    """Mimics the subset of the Ultralytics API that PlayerDetector.track() uses."""

    def __init__(self, frames):
        self._frames = frames  # list of (xyxy_list, id_list) or None for "no detections"
        self.calls = {}

    def track(self, **kwargs):
        self.calls = kwargs
        for f in self._frames:
            if f is None:
                yield SimpleNamespace(boxes=None)
            else:
                xyxy, ids = f
                yield SimpleNamespace(boxes=SimpleNamespace(id=_Tensor(ids), xyxy=_Tensor(xyxy)))


def test_track_parses_boxes_and_ids():
    model = _FakeModel(
        [
            ([[10, 20, 30, 120], [40, 50, 60, 160]], [1, 2]),
            None,  # a frame with no detections is skipped
            ([[12, 22, 32, 122]], [1]),
        ]
    )
    det = PlayerDetector(_model=model)
    tracks = det.track("ignored.mp4")

    assert all(isinstance(t, PlayerTrack) for t in tracks)
    assert [(t.track_id, t.frame) for t in tracks] == [(1, 0), (2, 0), (1, 2)]
    # foot point is bottom-center of the bbox.
    assert tracks[0].foot_px == (20.0, 120.0)
    # class filter + person class are passed through to Ultralytics.
    assert model.calls["classes"] == [0]


def test_vid_stride_maps_frames_to_source_space():
    model = _FakeModel([([[0, 0, 10, 10]], [7]), ([[0, 0, 10, 10]], [7])])
    det = PlayerDetector(_model=model, vid_stride=5)
    tracks = det.track("ignored.mp4")
    assert [t.frame for t in tracks] == [0, 5]
    assert model.calls["vid_stride"] == 5


def test_track_parses_pose_keypoints():
    """A pose model attaches res.keypoints (xy + conf), aligned by box index."""
    kxy = [[[float(i), float(i + 1)] for i in range(17)],
           [[float(i + 100), float(i)] for i in range(17)]]
    kcf = [[0.9] * 17, [0.7] * 17]

    class _PoseModel:
        calls: dict = {}

        def track(self, **kwargs):
            _PoseModel.calls = kwargs
            yield SimpleNamespace(
                boxes=SimpleNamespace(id=_Tensor([1, 2]),
                                      xyxy=_Tensor([[0, 0, 10, 10], [5, 5, 15, 15]])),
                keypoints=SimpleNamespace(xy=_Tensor(kxy), conf=_Tensor(kcf)),
            )

    tracks = PlayerDetector(_model=_PoseModel()).track("ignored.mp4")
    assert len(tracks) == 2
    assert tracks[0].keypoints_px == [(float(i), float(i + 1)) for i in range(17)]
    assert tracks[0].keypoint_conf == [0.9] * 17
    assert tracks[1].keypoints_px[0] == (100.0, 0.0)


def test_detect_only_model_has_no_keypoints():
    """A plain detector (no res.keypoints) leaves keypoint fields None — backward compatible."""
    tracks = PlayerDetector(_model=_FakeModel([([[0, 0, 10, 10]], [1])])).track("x.mp4")
    assert tracks[0].keypoints_px is None and tracks[0].keypoint_conf is None
