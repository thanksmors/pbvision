"""FastAPI app: upload a match, poll progress, fetch the analysis JSON.

Runs natively (no Docker)::

    pip install -e '.[api]'
    uvicorn api.app.main:app --reload

The heavy analysis runs in a subprocess (see ``jobs.py``), so the API stays responsive and
needs only the core + ``api`` dependencies. Full runs additionally need the ``ml`` extra
installed wherever the subprocess executes (locally for short clips, or the rented GPU box).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api.app import jobs

app = FastAPI(title="pbvision", version="0.1.0")

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.post("/api/matches")
async def upload_match(file: UploadFile) -> dict[str, str]:
    """Accept a video upload, start analysis, return the job id."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    job_id = jobs.create_job(file.filename or "match.mp4", data)
    jobs.start_job(job_id)
    return {"job_id": job_id}


@app.post("/api/demo")
def start_demo() -> dict[str, str]:
    """Generate a synthetic match and analyze it in fixture mode (no upload, no ML)."""
    job_id = jobs.create_demo_job()
    jobs.start_job(job_id, fixture=True)
    return {"job_id": job_id}


@app.get("/api/matches/{job_id}/video")
def match_video(job_id: str) -> FileResponse:
    path = jobs.video_path(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/matches/{job_id}/frame")
def match_frame(job_id: str) -> Response:
    """First frame as JPEG, for the manual court-calibration UI."""
    jpeg = jobs.first_frame_jpeg(job_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="frame unavailable")
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/api/matches/{job_id}/court")
def calibrate_court(job_id: str, landmarks: dict[str, list[float]]) -> dict[str, str]:
    """Save >=4 named court landmarks and re-run analysis using the manual homography."""
    try:
        jobs.save_named_corners(job_id, landmarks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    jobs.start_job(job_id)
    return {"job_id": job_id}


@app.get("/api/matches/{job_id}/court.json")
def download_corners(job_id: str) -> FileResponse:
    """Return the saved calibration corners as a downloadable ``corners.json``.

    Lets the UI export the clicked court points for use with ``scripts/debug_3d.py``.
    """
    path = jobs.corners_path(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="no calibration saved for this match yet")
    return FileResponse(path, media_type="application/json", filename="corners.json")


@app.get("/api/court-landmarks")
def court_landmarks() -> JSONResponse:
    """The named pickleball reference landmarks (normalized court coords) for the UI diagram."""
    from pbengine.court.court_model import REFERENCE_POINTS

    return JSONResponse({name: list(xy) for name, xy in REFERENCE_POINTS.items()})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    return JSONResponse(jobs.read_status(job_id).model_dump())


@app.get("/api/matches/{job_id}")
def match_result(job_id: str) -> Response:
    result = jobs.read_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="result not ready")
    return Response(content=result, media_type="application/json")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (FRONTEND_DIR / "index.html").read_text()


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
