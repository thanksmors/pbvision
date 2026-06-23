"""Build the on-court roster: drop background people, then merge each player's fragments over time.

ByteTrack returns *everyone* it sees and splits a single player into many ``track_id``s (a fresh id
after every long occlusion). So two things are needed: drop the clearly off-court people (refs,
spectators, the next court — they project far outside ``[0,1]^2``), and **merge the fragments that
belong to the same physical player** so each of the four players is one continuous track across the
whole match (not just whichever single fragment happened to be longest). Without the merge the overlay
shows only the 1-3 players whose kept fragment spans the current frame.

Players hold left/right in doubles, so fragments are clustered per side by **lateral** position: a
track joins the nearest same-side cluster (within ``_PARTNER_LATERAL_FT``), else starts the other
partner's cluster — at most ``max_per_side`` per side. Each cluster is then merged (union of positions
across time). A per-track diagnostic line is logged so it's clear why a track was kept, merged, or
dropped.
"""

from __future__ import annotations

from statistics import median

from pbengine.court.court_model import WIDTH_FT, is_in_bounds
from pbengine.schema.models import Player

# Generous slack (~7 ft wide / ~15 ft long) so deep/late real players stay; spectators are far beyond.
_BOUND_MARGIN = 0.35
# Ignore blink tracks of just a few frames (a real player is present for hundreds).
_MIN_PRESENCE = 15
# Two same-side fragments belong to *different* players only if their median x is at least this far
# apart (doubles partners split the court laterally). Depth is deliberately NOT used: a single player
# roams their whole half (deep baseline to the net), so a y gap is the same player moving, not a
# partner — using it would split one player in two and drop the fragment that fills the timeline.
_PARTNER_LATERAL_FT = 5.0


def _median_xy(player: Player) -> tuple[float, float]:
    return (median(p.court_xy[0] for p in player.positions),
            median(p.court_xy[1] for p in player.positions))


def _median_x_ft(player: Player) -> float:
    return median(p.court_xy[0] for p in player.positions) * WIDTH_FT


def _merge_cluster(cluster: list[Player]) -> Player:
    """Merge fragments of one physical player into a single continuous Player.

    ``cluster`` is walked most-present first; positions are unioned across fragments and sorted by
    frame, and on the rare frame collision the more-present fragment's position wins. The merged player
    keeps the smallest ``track_id`` so a fragment that was previously the sole survivor keeps its id.
    """
    cluster = sorted(cluster, key=lambda p: len(p.positions), reverse=True)
    by_frame: dict[int, object] = {}
    for p in cluster:
        for pos in p.positions:
            by_frame.setdefault(pos.frame, pos)
    positions = [by_frame[f] for f in sorted(by_frame)]
    return Player(track_id=min(p.track_id for p in cluster), team=cluster[0].team,
                  positions=positions)


def select_on_court_players(players: list[Player], max_per_side: int = 2) -> list[Player]:
    """Return the on-court roster: in-bounds players, ``max_per_side`` per side, fragments merged.

    Players without a ``team`` (no homography ⇒ sides are meaningless) are returned unchanged — there's
    nothing reliable to filter on.
    """
    if any(p.team is None for p in players):
        print(f"roster: no homography/teams — keeping all {len(players)} tracks", flush=True)
        return players

    # Decide eligibility (on-court median + enough presence) per track and log the reason.
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

    # Per side, cluster the eligible fragments by lateral position into <= max_per_side players, then
    # merge each cluster across time so every player spans (almost) the whole match.
    roster: list[Player] = []
    merged_into: dict[int, int] = {}
    dropped3: set[int] = set()
    for team in {p.team for p in eligible}:
        side = sorted((p for p in eligible if p.team == team), key=lambda p: len(p.positions),
                      reverse=True)
        clusters: list[list[Player]] = []
        anchors_x: list[float] = []
        for p in side:
            px = _median_x_ft(p)
            match = next((i for i, ax in enumerate(anchors_x)
                          if abs(px - ax) < _PARTNER_LATERAL_FT), None)
            if match is not None:
                clusters[match].append(p)
            elif len(clusters) < max_per_side:
                clusters.append([p])
                anchors_x.append(px)
            else:  # distinct from both partners, no room — a 3rd person on this side
                dropped3.add(p.track_id)
        for c in clusters:
            m = _merge_cluster(c)
            roster.append(m)
            for p in c:
                if p.track_id != m.track_id:
                    merged_into[p.track_id] = m.track_id
    roster.sort(key=lambda p: p.track_id)

    for idx, p in enumerate(sorted(players, key=lambda p: p.track_id)):
        if p.track_id in merged_into:
            rows[idx] = rows[idx].replace("-> KEPT", f"-> MERGED into #{merged_into[p.track_id]}")
        elif p.track_id in dropped3:
            rows[idx] = rows[idx].replace("-> KEPT", "-> DROPPED(3rd lateral cluster)")
    a = sum(1 for p in roster if p.team and p.team.value == "A")
    b = len(roster) - a
    print(f"roster: {len(players)} tracks -> {len(roster)} players (A:{a} B:{b})", flush=True)
    for p in roster:
        fr = [pos.frame for pos in p.positions]
        print(f"  player #{p.track_id} {p.team.value if p.team else '?'} {len(fr)} frames "
              f"span=[{min(fr)}-{max(fr)}]", flush=True)
    print("\n".join(rows), flush=True)
    return roster
