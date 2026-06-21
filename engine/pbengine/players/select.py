"""Keep only the on-court roster, dropping background people (refs, spectators, next court).

ByteTrack returns *everyone* it sees, so a raw match carries far more than the players. We keep tracks
whose **median** court position sits on the court (with generous slack, so a player standing deep
behind the baseline or entering late still counts), drop the clearly off-court people (spectators
project far outside ``[0,1]^2``), then cap each side to the ``max_per_side`` most present (2 for
doubles, which naturally yields 1 per side for singles). A per-track diagnostic line is logged so it's
clear *why* a track was kept or dropped.
"""

from __future__ import annotations

from statistics import median

from pbengine.court.court_model import LENGTH_FT, WIDTH_FT, is_in_bounds
from pbengine.schema.models import Player

# Generous slack (~7 ft wide / ~15 ft long) so deep/late real players stay; spectators are far beyond.
_BOUND_MARGIN = 0.35
# Ignore blink tracks of just a few frames (a real player is present for hundreds).
_MIN_PRESENCE = 15
# A 2nd track on a side counts as a *distinct partner* only if it sits clearly apart from the one
# already kept — either side-by-side (a big lateral/x gap, the common doubles formation) or genuine
# up/back (a large depth/y gap). A track that's merely a few feet away (mostly drifting in depth) is a
# fragment of the same player; keeping it would draw two skeletons on one player and starve the real
# partner of a roster slot — the "only 3 players" bug (two near-identical left tracks were both kept).
_PARTNER_LATERAL_FT = 5.0  # side-by-side partners are at least this far apart in x
_PARTNER_DEPTH_FT = 12.0   # up/back partners are at least this far apart in y (a single player's
                           # fragment-to-fragment depth wander stays well under this)


def _median_xy(player: Player) -> tuple[float, float]:
    return (median(p.court_xy[0] for p in player.positions),
            median(p.court_xy[1] for p in player.positions))


def _median_ft(player: Player) -> tuple[float, float]:
    mx, my = _median_xy(player)
    return (mx * WIDTH_FT, my * LENGTH_FT)


def _distinct_partner(a_ft: tuple[float, float], b_ft: tuple[float, float]) -> bool:
    """True if two median positions (feet) are far enough apart to be two players, not one fragmented."""
    return (abs(a_ft[0] - b_ft[0]) >= _PARTNER_LATERAL_FT
            or abs(a_ft[1] - b_ft[1]) >= _PARTNER_DEPTH_FT)


def select_on_court_players(players: list[Player], max_per_side: int = 2) -> list[Player]:
    """Return the on-court roster: in-bounds players, at most ``max_per_side`` per side.

    Players without a ``team`` (no homography ⇒ sides are meaningless) are returned unchanged — there's
    nothing reliable to filter on.
    """
    if any(p.team is None for p in players):
        print(f"roster: no homography/teams — keeping all {len(players)} tracks", flush=True)
        return players

    # Decide keep/drop per track and log the reason.
    eligible: list[Player] = []
    rows: list[str] = []
    for p in sorted(players, key=lambda p: p.track_id):
        n = len(p.positions)
        mx, my = _median_xy(p) if n else (9.0, 9.0)
        on_court = n and is_in_bounds((mx, my), _BOUND_MARGIN)
        enough = n >= _MIN_PRESENCE
        keep = bool(on_court and enough)
        if keep:
            eligible.append(p)
        reason = "KEPT" if keep else ("DROPPED(off-court)" if not on_court else "DROPPED(too-brief)")
        frames = [pos.frame for pos in p.positions]
        span = f"{min(frames)}-{max(frames)}" if frames else "-"
        rows.append(f"  #{p.track_id} {p.team.value if p.team else '?'} frames={n} span=[{span}] "
                    f"median=({mx:.2f},{my:.2f}) -> {reason}")

    # Per side, keep the most-present tracks — but a 2nd (or beyond) slot only goes to a track that is
    # spatially *separated* from those already kept, so a side's two roster spots can't both be eaten
    # by near-duplicate fragments of the same player (which left the real partner unrepresented). A
    # close, less-present track is dropped as a duplicate; ``max_per_side`` still caps the count.
    roster: list[Player] = []
    capped: dict[int, str] = {}
    for team in {p.team for p in eligible}:
        side = sorted((p for p in eligible if p.team == team), key=lambda p: len(p.positions),
                      reverse=True)
        kept: list[Player] = []
        for p in side:
            if len(kept) >= max_per_side:
                capped[p.track_id] = "side full"
                continue
            pm = _median_ft(p)
            if all(_distinct_partner(pm, _median_ft(k)) for k in kept):
                kept.append(p)
            else:
                capped[p.track_id] = "near a kept track"
        roster.extend(kept)
    roster.sort(key=lambda p: p.track_id)

    for idx, p in enumerate(sorted(players, key=lambda p: p.track_id)):
        if p.track_id in capped:  # was eligible (KEPT) but lost its slot to the per-side roster rule
            rows[idx] = rows[idx].replace("-> KEPT", f"-> DROPPED(dup: {capped[p.track_id]})")
    a = sum(1 for p in roster if p.team and p.team.value == "A")
    b = len(roster) - a
    print(f"roster: {len(players)} tracks -> kept {len(roster)} (A:{a} B:{b})", flush=True)
    print("\n".join(rows), flush=True)
    return roster
