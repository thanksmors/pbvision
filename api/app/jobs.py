"""Background job management without a queue.

For a single user we don't need Redis/RQ. The API spawns the engine as a subprocess
(``python -m pbengine.pipeline``), which writes progress to a per-job ``status.json`` and the
final ``result.json``. The API just reads those files when the UI polls. Swapping to Redis+RQ
later only touches this module.
"""

from __future__ import annotations

import json
import os
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


def create_demo_job() -> str:
    """Create a job backed by a freshly-generated synthetic video (no upload, no ML)."""
    from pbengine import fixtures

    job_id = uuid.uuid4().hex[:12]
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    fixtures.write_synthetic_video(d / "input.mp4")
    _write_status(job_id, JobStatus(job_id=job_id, state="pending"))
    return job_id


def start_job(job_id: str, fixture: bool = False, preset: str | None = None) -> None:
    """Launch the engine subprocess. Detached: progress is tracked via the status file.

    ``fixture=True`` runs the pipeline with scripted synthetic detectors (no ML), used by the
    demo so the full flow works on a CPU-only box with nothing but the core deps installed.

    Real uploads pick a player-detection **preset** (fast / balanced / max / gpu — see
    :mod:`pbengine.detect.presets`), resolved from the ``preset`` arg → ``PBV_PLAYERS_PRESET`` →
    ``balanced``. The preset is persisted so a calibrate re-run reuses it. Individual knobs can still
    override via ``PBV_PLAYERS_WEIGHTS`` / ``PBV_VID_STRIDE`` / ``PBV_PLAYERS_IMGSZ`` /
    ``PBV_PLAYERS_CONF`` (e.g. a plain detector like ``yolo26m.pt`` to skip skeletons).
    """
    d = job_dir(job_id)
    video = next(d.glob("input.*"))
    # Reset status before (re-)launching so a stale "done" from a prior run can't be read as
    # this run's result (e.g. when re-analyzing after calibration).
    _write_status(job_id, JobStatus(job_id=job_id, state="pending"))
    cmd = [
        sys.executable,
        "-u",  # unbuffered, so progress lines land in run.log live (tail -f)
        "-m",
        "pbengine.pipeline",
        str(video),
        "-o",
        str(d / "result.json"),
        "--status",
        str(d / "status.json"),
        "--job-id",
        job_id,
    ]
    if fixture:
        cmd.append("--fixture")
    else:
        # Resolve + persist the preset (so a later calibrate re-run reuses the same one).
        preset = preset or _read_preset(d) or os.environ.get("PBV_PLAYERS_PRESET", "balanced")
        (d / "preset.txt").write_text(preset)
        cmd += ["--players-preset", preset]
        # Optional per-knob overrides from the environment.
        for env, flag in (("PBV_PLAYERS_WEIGHTS", "--players-weights"),
                          ("PBV_VID_STRIDE", "--vid-stride"),
                          ("PBV_PLAYERS_IMGSZ", "--players-imgsz"),
                          ("PBV_PLAYERS_CONF", "--players-conf")):
            val = os.environ.get(env)
            if val:
                cmd += [flag, val]
        corners = d / "court.json"
        if corners.exists():  # manual calibration overrides auto court detection
            cmd += ["--court-corners", str(corners)]
    # Capture stdout+stderr to a per-job log so progress (frame X/Y, ETA) and tracebacks are
    # visible — `tail -f data/<job>/run.log` or the UI log panel — instead of discarded.
    log = open(d / "run.log", "wb")  # noqa: SIM115 (handed to the detached child)
    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def read_log_tail(job_id: str, max_bytes: int = 8000) -> str:
    """Return the tail (~last ``max_bytes``) of a job's run.log, or '' if there is none yet."""
    f = job_dir(job_id) / "run.log"
    if not f.exists():
        return ""
    with f.open("rb") as fh:
        try:
            fh.seek(-max_bytes, 2)
        except OSError:
            fh.seek(0)
        return fh.read().decode("utf-8", "replace")


def _read_preset(d: Path) -> str | None:
    """The player-detection preset persisted for this job, if any (reused on calibrate re-run)."""
    f = d / "preset.txt"
    return f.read_text().strip() if f.exists() else None


def video_path(job_id: str) -> Path | None:
    """Path to the job's source video, if present (for the viewer's ``<video>`` element)."""
    return next(job_dir(job_id).glob("input.*"), None)


def corners_path(job_id: str) -> Path | None:
    """Path to the saved manual-calibration corners (``court.json``), if it exists.

    Same ``{name: [x, y]}`` format ``scripts/debug_3d.py --court-corners`` and the engine's
    ``ManualCourtDetector`` consume, so the UI can hand it back as a downloadable ``corners.json``.
    """
    path = job_dir(job_id) / "court.json"
    return path if path.exists() else None


def first_frame_jpeg(job_id: str) -> bytes | None:
    """Encode the job video's first frame as JPEG for the calibration UI."""
    import cv2

    path = video_path(job_id)
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def save_named_corners(job_id: str, named: dict[str, list[float]]) -> None:
    """Validate and persist named court landmarks (>=4) as the engine's court.json.

    Names must be known pickleball reference points (see ``court_model.REFERENCE_POINTS``);
    any 4+ visible, well-spread ones are enough to solve the homography — they need not be the
    outer corners, so a clipped corner doesn't block calibration.
    """
    from pbengine.court.court_model import REFERENCE_POINTS

    unknown = [n for n in named if n not in REFERENCE_POINTS]
    if unknown:
        raise ValueError(f"unknown court landmarks: {unknown}")
    if len(named) < 4:
        raise ValueError(f"need >=4 landmarks, got {len(named)}")
    clean = {n: [float(xy[0]), float(xy[1])] for n, xy in named.items()}
    (job_dir(job_id) / "court.json").write_text(json.dumps(clean))


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
