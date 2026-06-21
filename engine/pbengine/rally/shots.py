"""Detect *shots* (player–ball contacts) — the unit of rally analysis.

A :class:`Shot` is a paddle hit, distinct from a :class:`Bounce` (ball↔ground). We find candidate
contacts where the ball reverses direction along the court (a player sent it back) — a signal available
from ``court_xy`` alone, no camera needed — exclude frames that coincide with bounces, attribute each to
the nearest player, and (after the 3D lift) classify it from speed/height/zone. Everything is
best-effort: with thin tracking or no players, shots may be unattributed or empty, never raising.
"""

from __future__ import annotations

from pbengine.court.court_model import KITCHEN_FT, LENGTH_FT, NET_HEIGHT_FT, NET_Y, WIDTH_FT
from pbengine.schema.models import (
    BallSample,
    Bounce,
    Player,
    Serve,
    Shot,
    ShotOutcome,
    ShotType,
    Team,
)

_KITCHEN_NORM = KITCHEN_FT / LENGTH_FT      # kitchen depth in normalized court units
_MIN_REVERSAL = 0.06                        # min |Δy| around a turning point to count as a contact
_MIN_SEP_S = 0.25                           # min time between contacts (debounce jitter)
_ATTRIB_FT = 6.0                            # max hitter→ball distance to attribute a contact
_DEFAULT_CONTACT_FT = 2.5                   # fallback contact height when no paddle estimate
_DRIVE_MPH = 25.0
_DINK_MPH = 15.0
_LOB_APEX_FT = 8.0


def _zone(y: float) -> str:
    """Court zone of a normalized ``y``: kitchen (near net), baseline (near a baseline), or between."""
    if abs(y - NET_Y) <= _KITCHEN_NORM:
        return "kitchen"
    if min(y, 1.0 - y) <= _KITCHEN_NORM:
        return "baseline"
    return "transition"


def _contact_frames(measured: list[BallSample], bounces: list[Bounce], fps: float) -> list[int]:
    """Frames where the ball reverses along-court direction (a hit), excluding bounce frames."""
    pts = [(s.frame, s.court_xy[1]) for s in measured if s.court_xy is not None]
    if len(pts) < 3:
        return []
    bset = {b.frame for b in bounces}
    min_sep = max(1, int(_MIN_SEP_S * fps))
    out: list[int] = []
    last_ext_y = pts[0][1]
    for (fp, yp), (f0, y0), (fn, yn) in zip(pts, pts[1:], pts[2:]):
        rising, falling = (y0 - yp), (yn - y0)
        if rising * falling < 0 and abs(y0 - last_ext_y) >= _MIN_REVERSAL:  # turning point w/ prominence
            if all(abs(f0 - bf) > 1 for bf in bset) and (not out or f0 - out[-1] >= min_sep):
                out.append(f0)
                last_ext_y = y0
    return out


def _nearest_player(players: list[Player], frame: int, ball_ft: tuple[float, float]) -> Player | None:
    """The player whose paddle (else body) is closest to the ball at ``frame``, within a gate."""
    best, best_d = None, _ATTRIB_FT
    for pl in players:
        pos = min((p for p in pl.positions), key=lambda p: abs(p.frame - frame), default=None)
        if pos is None or abs(pos.frame - frame) > 5:
            continue
        if pos.paddle_world_ft is not None:
            px, py = pos.paddle_world_ft[0], pos.paddle_world_ft[1]
        else:
            px, py = pos.court_xy[0] * WIDTH_FT, pos.court_xy[1] * LENGTH_FT
        d = ((px - ball_ft[0]) ** 2 + (py - ball_ft[1]) ** 2) ** 0.5
        if d < best_d:
            best, best_d = pl, d
    return best


def _ball_at(measured: list[BallSample], frame: int) -> BallSample | None:
    return min((s for s in measured if s.court_xy is not None),
               key=lambda s: abs(s.frame - frame), default=None)


def detect_shots(
    measured: list[BallSample], bounces: list[Bounce], players: list[Player],
    serve: Serve | None, fps: float,
) -> list[Shot]:
    """Preliminary shots: contact frame, hitter attribution, contact zone & height (pre-3D-lift).

    Returned heights feed the ball height model; :func:`enrich_shots` later fills speed/type/outcome
    from the lifted trajectory.
    """
    frames = _contact_frames(measured, bounces, fps)
    if serve is not None and (not frames or frames[0] > serve.frame):
        frames = [serve.frame, *frames]
    shots: list[Shot] = []
    for i, f in enumerate(sorted(dict.fromkeys(frames))):
        b = _ball_at(measured, f)
        if b is None:
            continue
        ball_ft = (b.court_xy[0] * WIDTH_FT, b.court_xy[1] * LENGTH_FT)
        hitter = _nearest_player(players, f, ball_ft)
        pos = (min(hitter.positions, key=lambda p: abs(p.frame - f)) if hitter else None)
        height = (pos.paddle_world_ft[2] if pos and pos.paddle_world_ft else _DEFAULT_CONTACT_FT)
        shots.append(Shot(
            shot_index=i, frame=f,
            player_track_id=hitter.track_id if hitter else None,
            team=hitter.team if hitter else None,
            contact_height_ft=height, contact_zone=_zone(b.court_xy[1]),
        ))
    return shots


def shot_contacts(shots: list[Shot]) -> list[tuple[int, float]]:
    """``(frame, height_ft)`` knots for :func:`pbengine.ball.trajectory3d.ball_world_ft`."""
    return [(s.frame, s.contact_height_ft or _DEFAULT_CONTACT_FT) for s in shots]


def enrich_shots(
    shots: list[Shot], traj: list[BallSample], bounces: list[Bounce],
    winner_team: Team | None,
) -> list[Shot]:
    """Fill speed/height/landing/type/outcome from the lifted trajectory (in place) and return it."""
    by_frame = {s.frame: s for s in traj}
    bframes = sorted(b.frame for b in bounces)
    for i, sh in enumerate(shots):
        nxt = shots[i + 1].frame if i + 1 < len(shots) else (traj[-1].frame if traj else sh.frame)
        window = [by_frame[f] for f in range(sh.frame, nxt + 1) if f in by_frame]
        speeds = [s.speed_mph for s in window if s.speed_mph is not None and not s.interpolated]
        sh.speed_mph = max(speeds) if speeds else sh.speed_mph
        here = by_frame.get(sh.frame)
        if here is not None and here.world_ft is not None:
            sh.contact_height_ft = here.world_ft[2]
        apex = max((s.world_ft[2] for s in window if s.world_ft is not None), default=0.0)
        nb = next((b for b in bounces if sh.frame < b.frame <= nxt), None)
        sh.landing_zone = _zone(nb.court_xy[1]) if nb else None
        had_bounce_before = any(shots[i - 1].frame < bf < sh.frame for bf in bframes) if i else False
        sh.shot_type = _classify(i, sh, apex, had_bounce_before)
    if shots and winner_team is not None:
        shots[-1].outcome = ShotOutcome.winner if shots[-1].team == winner_team else ShotOutcome.error
    return shots


def _classify(idx: int, sh: Shot, apex: float, had_bounce_before: bool) -> ShotType:
    spd, h = sh.speed_mph, sh.contact_height_ft or 0.0
    if idx == 0:
        return ShotType.serve
    if idx == 1:
        return ShotType.return_
    if not had_bounce_before and h > NET_HEIGHT_FT:
        return ShotType.volley                       # struck out of the air, above the net
    if apex > _LOB_APEX_FT:
        return ShotType.lob
    if spd is not None and spd >= _DRIVE_MPH:
        return ShotType.drive
    if spd is not None and spd <= _DINK_MPH and "kitchen" in (sh.contact_zone, sh.landing_zone):
        return ShotType.dink
    if sh.contact_zone == "baseline" and sh.landing_zone == "kitchen":
        return ShotType.drop                         # baseline → kitchen (third-shot drop shape)
    return ShotType.groundstroke
