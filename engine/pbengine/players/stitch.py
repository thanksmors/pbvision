"""Stitch fragmented player tracks so one physical player stays one continuous person.

ByteTrack splits a single player into several ``track_id``s whenever it loses them briefly (an
occlusion behind a partner, a missed detection): the old id ends and a new one is born. Downstream
that reads as a player blinking out and reappearing as a stranger. This pass merges the fragments
that are clearly the same person — a new track whose first detection picks up, in space and time,
right where a recently-ended track left off (same side of the court, a short gap, near the
velocity-extrapolated last position). It is deliberately conservative: ambiguous matches are left
unmerged (a wrong merge would teleport a skeleton), and it never invents or caps a roster — only
fragments with strong continuity are joined.

Works on the raw per-track records gathered in the pipeline (normalized ``court_xy``); the caller
then builds one :class:`~pbengine.schema.models.Player` per returned group and interpolates across
the now-bridged gap so the pose glides through the occlusion.
"""

from __future__ import annotations

from pbengine.court.court_model import LENGTH_FT, WIDTH_FT

MAX_STITCH_SEC = 1.5  # longest occlusion bridged; a longer gap is treated as a genuine new player
MATCH_FT = 8.0        # a fragment must resume within this of the extrapolated last position
LAMBDA_FT_PER_SEC = 2.0  # time-gap penalty (ft per second) folded into the match cost
MARGIN_FT = 2.0       # the best candidate must beat the runner-up by this to merge (avoid ambiguity)


def max_stitch_frames(fps: float) -> int:
    """Longest gap (frames) bridged by stitching / the interpolation cap that glides across it."""
    return max(1, int(round(MAX_STITCH_SEC * fps)))


def _feet(court_xy: tuple[float, float]) -> tuple[float, float]:
    return (court_xy[0] * WIDTH_FT, court_xy[1] * LENGTH_FT)


def _summarize(recs: list[dict], votes: list[str]) -> dict:
    """First/last frame + foot position (feet), end velocity (ft/frame), and majority side."""
    rs = sorted(recs, key=lambda r: r["frame"])
    first, last = rs[0], rs[-1]
    fx0, fy0 = _feet(first["court_xy"])
    fx1, fy1 = _feet(last["court_xy"])
    vel = (0.0, 0.0)
    if len(rs) >= 2:
        prev = rs[-2]
        df = last["frame"] - prev["frame"]
        if df > 0:
            px, py = _feet(prev["court_xy"])
            vel = ((fx1 - px) / df, (fy1 - py) / df)
    side = max(set(votes), key=votes.count) if votes else None
    return {"first_frame": first["frame"], "last_frame": last["frame"],
            "first_xy": (fx0, fy0), "last_xy": (fx1, fy1), "vel": vel, "side": side}


def stitch_tracks(
    raw_by_id: dict[int, list[dict]], side_votes: dict[int, list[str]], fps: float
) -> list[list[int]]:
    """Group ``track_id``s that are the same physical player. Returns one id-list per merged player.

    Greedy by start time: each track is attached to the best still-open earlier chain whose end
    continues into it (same side; gap ``0 < Δf ≤`` :func:`max_stitch_frames`; resumes within
    ``MATCH_FT`` of the chain's velocity-extrapolated last position), provided that best match beats
    the runner-up by ``MARGIN_FT`` (else it's ambiguous and left to start its own chain).
    """
    max_gap = max_stitch_frames(fps)
    summ = {tid: _summarize(recs, side_votes.get(tid, [])) for tid, recs in raw_by_id.items()}
    order = sorted(raw_by_id, key=lambda tid: (summ[tid]["first_frame"], tid))

    chains: list[dict] = []  # {"ids": [...], end summary fields}
    for tid in order:
        s = summ[tid]
        costs: list[tuple[float, int]] = []
        for ci, ch in enumerate(chains):
            gap = s["first_frame"] - ch["last_frame"]
            if gap <= 0 or gap > max_gap or (ch["side"] is not None and ch["side"] != s["side"]):
                continue
            px = ch["last_xy"][0] + ch["vel"][0] * gap
            py = ch["last_xy"][1] + ch["vel"][1] * gap
            dist = ((px - s["first_xy"][0]) ** 2 + (py - s["first_xy"][1]) ** 2) ** 0.5
            if dist > MATCH_FT:
                continue
            costs.append((dist + LAMBDA_FT_PER_SEC * (gap / fps), ci))
        costs.sort()
        if costs and (len(costs) == 1 or costs[1][0] - costs[0][0] >= MARGIN_FT):
            ch = chains[costs[0][1]]
            ch["ids"].append(tid)
            ch.update(last_frame=s["last_frame"], last_xy=s["last_xy"], vel=s["vel"])
        else:
            chains.append({"ids": [tid], "last_frame": s["last_frame"], "last_xy": s["last_xy"],
                           "vel": s["vel"], "side": s["side"]})
    return [ch["ids"] for ch in chains]
