"""Rally segmentation from a ball-presence timeline.

The proven MVP backbone (cf. ``vinod-polinati/pickleball-rally-detection``): mark frames
where a valid ball was detected, bridge short gaps (the ball is briefly occluded behind the
net or a player), then split into rallies wherever the ball is absent for longer than the
gap tolerance. Very short "rallies" are dropped as noise.

Physics/false-positive gating (size caps, max px/frame jumps) happens upstream in ball
tracking; this module only consumes the resulting presence timeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RallySpan:
    start_frame: int
    end_frame: int

    @property
    def length_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def segment_rallies(
    present_frames: list[int],
    fps: float,
    max_gap_s: float = 0.6,
    min_rally_s: float = 1.0,
) -> list[RallySpan]:
    """Group frames with a detected ball into rallies.

    Parameters
    ----------
    present_frames:
        Frame indices (any order) that had a valid ball detection.
    fps:
        Video frame rate, used to convert the time thresholds to frames.
    max_gap_s:
        Maximum ball-absence (seconds) tolerated *within* a rally before it is split.
    min_rally_s:
        Rallies shorter than this (seconds) are discarded as noise.
    """
    if not present_frames or fps <= 0:
        return []

    max_gap = max(1, round(max_gap_s * fps))
    min_len = max(1, round(min_rally_s * fps))

    frames = sorted(set(present_frames))
    spans: list[RallySpan] = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f - prev > max_gap:  # gap too long -> rally boundary
            spans.append(RallySpan(start, prev))
            start = f
        prev = f
    spans.append(RallySpan(start, prev))

    return [s for s in spans if s.length_frames >= min_len]
