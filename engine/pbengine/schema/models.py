"""Pydantic v2 models that define the engine <-> API contract.

A full run produces a :class:`MatchResult`, serialized to ``result.json``. Both the
analysis engine and the FastAPI layer import these models so the schema is enforced on
both sides of the wire.

Coordinate conventions
----------------------
* ``px``      : pixel coordinates in the source video, ``(x, y)``, origin top-left.
* ``court_xy``: normalized top-down court coordinates in ``[0, 1] x [0, 1]`` after the
  homography is applied. ``(0, 0)`` is one back corner, ``(1, 1)`` the diagonal corner.
  This makes side/out-of-bounds logic resolution-independent.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

Px = tuple[float, float]
CourtXY = tuple[float, float]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels


class Team(str, Enum):
    A = "A"
    B = "B"


class WinReason(str, Enum):
    ball_out = "ball_out"
    double_bounce = "double_bounce"
    net = "net"
    unknown = "unknown"


class ShotType(str, Enum):
    serve = "serve"
    return_ = "return"
    drive = "drive"
    dink = "dink"
    drop = "drop"
    volley = "volley"
    lob = "lob"
    groundstroke = "groundstroke"


class ShotOutcome(str, Enum):
    in_play = "in_play"   # rally continued
    winner = "winner"     # ended the rally in the hitter's favor
    error = "error"       # ended the rally against the hitter (out / net / set up a double bounce)


class VideoMeta(BaseModel):
    fps: float
    frames: int
    width: int
    height: int


class CourtModel(BaseModel):
    """The court calibration for a camera shot.

    ``homography`` maps source pixels -> normalized top-down court coords. With a static
    camera it is solved once and reused for the whole match.
    """

    homography: list[list[float]] = Field(..., description="3x3 pixel->court_xy matrix")
    keypoints_px: list[Px] = Field(default_factory=list, description="clicked/detected court points")
    court_quad_px: list[Px] = Field(
        default_factory=list,
        description="the 4 full-court corners projected to pixels (may fall outside the frame "
        "when extrapolated) — lets the viewer draw the whole court even if a corner is clipped",
    )
    model: str = "pickleball_doubles"


class Serve(BaseModel):
    frame: int
    server_team: Team
    server_side: str = Field(..., description='"left" or "right" service court')
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class Bounce(BaseModel):
    frame: int
    court_xy: CourtXY
    side: Team
    in_bounds: bool


class Shot(BaseModel):
    """A player–ball contact (a paddle hit), the unit of rally analysis.

    Distinct from :class:`Bounce` (ball↔ground). Contacts are detected at ballistic-segment breaks and
    attributed to the nearest player; features (speed, height, zones) drive shot categorization and the
    rally recap. All fields beyond ``frame`` are best-effort and may be absent when tracking is thin.
    """

    shot_index: int = Field(..., description="0-based order within the rally")
    frame: int
    player_track_id: int | None = Field(None, description="track id of the hitter (None if unattributed)")
    team: Team | None = None
    shot_type: ShotType = ShotType.groundstroke
    speed_mph: float | None = Field(None, description="ball speed just after contact (3D)")
    contact_height_ft: float | None = Field(None, description="ball height at contact")
    contact_zone: str | None = Field(None, description='"kitchen" | "transition" | "baseline"')
    landing_zone: str | None = Field(None, description="zone where the following arc first lands")
    outcome: "ShotOutcome" = Field(ShotOutcome.in_play, description="rally-ending result, if terminal")


class BallSample(BaseModel):
    frame: int
    px: Px
    court_xy: CourtXY | None = None
    conf: float = Field(0.0, ge=0.0, le=1.0)
    world_ft: tuple[float, float, float] | None = Field(
        None, description="3D ball position in feet (X across, Y along, Z up) from monocular "
        "reconstruction; None when the camera/trajectory could not be recovered"
    )
    speed_mph: float | None = Field(None, description="ball speed at this frame, mph (3D)")
    interpolated: bool = Field(
        False, description="True if this sample was physics-interpolated to fill a detection gap "
        "(conf=0); measured detections are False. Excluded from speed/height records."
    )


class PlayerPosition(BaseModel):
    frame: int
    court_xy: CourtXY
    bbox_px: BBox | None = Field(None, description="source-pixel bbox for the viewer overlay")
    pose_px: list[Px] | None = Field(
        None, description="COCO-17 keypoints in source pixels (None if no pose model)"
    )
    pose_conf: list[float] | None = Field(None, description="per-keypoint confidence [0, 1]")
    pose_world_ft: list[tuple[float, float, float]] | None = Field(
        None, description="keypoints lifted to 3D court feet (None if no camera / degenerate lift)"
    )
    paddle_px: BBox | None = Field(
        None, description="wrist-anchored paddle segment (base_x, base_y, tip_x, tip_y) in px"
    )
    paddle_world_ft: tuple[float, float, float] | None = Field(
        None, description="paddle tip in 3D court feet (None if no camera)"
    )
    interpolated: bool = Field(
        False, description="True if this position was linearly interpolated to bridge a short "
        "detection gap (so skeletons stay continuous); measured detections are False."
    )


class Player(BaseModel):
    track_id: int
    team: Team | None = None
    positions: list[PlayerPosition] = Field(default_factory=list)


class Point(BaseModel):
    point_index: int
    start_frame: int
    end_frame: int
    serve: Serve | None = None
    rally_length_shots: int = 0
    shots: list[Shot] = Field(default_factory=list)
    bounces: list[Bounce] = Field(default_factory=list)
    ball_trajectory: list[BallSample] = Field(default_factory=list)
    winner_team: Team | None = None
    win_reason: WinReason = WinReason.unknown
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    needs_review: bool = True
    clip_path: str | None = None

    @property
    def start_time_s(self) -> float | None:
        return None  # populated against fps by the pipeline if needed


class EngineInfo(BaseModel):
    version: str = "0.1.0"
    models: dict[str, str] = Field(default_factory=dict)


class MatchResult(BaseModel):
    match_id: str
    video: VideoMeta
    court: CourtModel | None = None
    points: list[Point] = Field(default_factory=list)
    players: list[Player] = Field(default_factory=list)
    engine: EngineInfo = Field(default_factory=EngineInfo)
    warnings: list[str] = Field(
        default_factory=list, description="stages skipped because their model was unavailable"
    )


class JobStatus(BaseModel):
    """Progress record the API polls. Written by the engine subprocess to a status file."""

    job_id: str
    state: str = Field("pending", description="pending|running|done|error")
    stage: str = ""
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = ""
    result_path: str | None = None
