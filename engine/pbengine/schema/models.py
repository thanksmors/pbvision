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
    keypoints_px: list[Px] = Field(default_factory=list, description="detected court points")
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


class BallSample(BaseModel):
    frame: int
    px: Px
    court_xy: CourtXY | None = None
    conf: float = Field(0.0, ge=0.0, le=1.0)


class PlayerPosition(BaseModel):
    frame: int
    court_xy: CourtXY
    bbox_px: BBox | None = Field(None, description="source-pixel bbox for the viewer overlay")


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
