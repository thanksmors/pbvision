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
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
