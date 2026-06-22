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

from statistics import median

from pbengine.court.court_model import LENGTH_FT, WIDTH_FT

MAX_STITCH_SEC = 1.5  # longest occlusion bridged; a longer gap is treated as a genuine new player
MATCH_FT = 8.0        # a fragment must resume within this of the extrapolated last position
LAMBDA_FT_PER_SEC = 2.0  # time-gap penalty (ft per second) folded into the match cost
MARGIN_FT = 2.0       # the best candidate must beat the runner-up by this to merge (avoid ambiguity)
# Two fragments only belong to the same player if they share the same lateral half of the court. In
# doubles the partners hold left/right, so a candidate whose *median* x is this far (ft) from the
# chain's median x is the other player — never merge, even when the endpoints happen to abut (the
# "chimera": a right-baseline track ending where a left-baseline track begins). y is ignored so one
# player moving up/back across a tracking gap still stitches.
MERGE_MAX_LATERAL_FT = 5.0


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
    med = (median(_feet(r["court_xy"])[0] for r in rs), median(_feet(r["court_xy"])[1] for r in rs))
    return {"first_frame": first["frame"], "last_frame": last["frame"],
            "first_xy": (fx0, fy0), "last_xy": (fx1, fy1), "vel": vel, "side": side,
            "med": med, "n": len(rs)}


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
            # Same player => same lateral half. A candidate whose median x is far from the chain's
            # median x is the other doubles partner; never fuse them, however close the endpoints sit
            # (the right-end-meets-left-start chimera). Endpoint distances can't see this — medians can.
            if abs(ch["cmed"][0] - s["med"][0]) >= MERGE_MAX_LATERAL_FT:
                continue
            # Raw last->first distance must be sane on its own. Velocity extrapolation (below) is only
            # a cost/tiebreaker — it must not be able to *expand* the match radius and claim a player
            # on the far side of the court (e.g. a right-baseline track ending, a left-baseline track
            # starting). Without this, a fast end-velocity over a long gap fuses two distinct players.
            raw_dist = ((ch["last_xy"][0] - s["first_xy"][0]) ** 2
                        + (ch["last_xy"][1] - s["first_xy"][1]) ** 2) ** 0.5
            if raw_dist > MATCH_FT:
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
            runner = f"{costs[1][0]:.1f}" if len(costs) > 1 else "none"
            gap = s["first_frame"] - ch["last_frame"]
            # A merge collapses two ByteTrack ids into one player; the lone-candidate branch (no
            # runner-up to beat) is exactly where two distinct people on the same side can fuse.
            print(f"stitch: track {tid} -> chain {ch['ids']} MERGED "
                  f"(gap={gap}f cost={costs[0][0]:.1f} runner-up={runner})", flush=True)
            ch["ids"].append(tid)
            # Presence-weighted running centroid of the chain's members (for the lateral gate above).
            cw = ch["cw"] + s["n"]
            ch["cmed"] = ((ch["cmed"][0] * ch["cw"] + s["med"][0] * s["n"]) / cw,
                          (ch["cmed"][1] * ch["cw"] + s["med"][1] * s["n"]) / cw)
            ch["cw"] = cw
            ch.update(last_frame=s["last_frame"], last_xy=s["last_xy"], vel=s["vel"])
        else:
            best = f"{costs[0][0]:.1f}" if costs else "none"
            runner = f"{costs[1][0]:.1f}" if len(costs) > 1 else "none"
            reason = "no candidate within gap/side/dist" if not costs else "ambiguous (margin<2ft)"
            print(f"stitch: track {tid} new chain ({reason}; best={best} runner-up={runner})",
                  flush=True)
            chains.append({"ids": [tid], "last_frame": s["last_frame"], "last_xy": s["last_xy"],
                           "vel": s["vel"], "side": s["side"], "cmed": s["med"], "cw": s["n"]})
    groups = [ch["ids"] for ch in chains]
    merged = [g for g in groups if len(g) > 1]
    print(f"stitch: {len(raw_by_id)} raw tracks -> {len(groups)} players"
          + (f" ({len(merged)} stitched: {merged})" if merged else " (no merges)"), flush=True)
    return groups
