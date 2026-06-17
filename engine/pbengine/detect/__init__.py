"""Detection wrappers (players via YOLO26; ball lives in ``pbengine.ball``)."""

from pbengine.detect.players import PlayerDetector, PlayerTrack

__all__ = ["PlayerDetector", "PlayerTrack"]
