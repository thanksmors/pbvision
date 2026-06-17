"""Background job management without a queue.

For a single user we don't need Redis/RQ. The API spawns the engine as a subprocess
(``python -m pbengine.pipeline``), which writes progress to a per-job ``status.json`` and the
final ``result.json``. The API just reads those files when the UI polls. Swapping to Redis+RQ
later only touches this module.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

from pbengine.schema.models import JobStatus

DATA_ROOT = Path("data")


def job_dir(job_id: str) -> Path:
    return DATA_ROOT / job_id


def create_job(upload_name: str, video_bytes: bytes) -> str:
    """Persist an uploaded video and return a new job id."""
    job_id = uuid.uuid4().hex[:12]
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload_name).suffix or ".mp4"
    (d / f"input{suffix}").write_bytes(video_bytes)
    _write_status(job_id, JobStatus(job_id=job_id, state="pending"))
    return job_id


def start_job(job_id: str) -> None:
    """Launch the engine subprocess. Detached: progress is tracked via the status file."""
    d = job_dir(job_id)
    video = next(d.glob("input.*"))
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pbengine.pipeline",
            str(video),
            "-o",
            str(d / "result.json"),
            "--status",
            str(d / "status.json"),
            "--job-id",
            job_id,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def read_status(job_id: str) -> JobStatus:
    path = job_dir(job_id) / "status.json"
    if not path.exists():
        return JobStatus(job_id=job_id, state="unknown")
    status = JobStatus.model_validate_json(path.read_text())
    if status.state == "done":
        status.result_path = str(job_dir(job_id) / "result.json")
    return status


def read_result(job_id: str) -> str | None:
    path = job_dir(job_id) / "result.json"
    return path.read_text() if path.exists() else None


def _write_status(job_id: str, status: JobStatus) -> None:
    (job_dir(job_id) / "status.json").write_text(status.model_dump_json())
