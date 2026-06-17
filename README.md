# pbvision

A from-scratch clone of [PBVision](https://pb.vision): upload a pickleball match video and
get **point-level analysis** — the match segmented into points, serves detected, the ball
tracked, and the **winner of each rally** called (with a confidence + "needs review" flag).

This is v0.1, focused on the point-level MVP. Rating, pose, and fine-tuned models come later.

## How it's built

A clean separation between a portable **engine** and a thin **app**:

- **`engine/pbengine/`** — the analysis engine. Importable and runnable on its own
  (`python -m pbengine.pipeline match.mp4 -o result.json`). Pure-logic stages (court geometry,
  rally/serve/winner heuristics, Kalman smoothing) depend only on the core requirements;
  model-backed stages load their heavy deps lazily.
- **`api/`** — a FastAPI app + a single-page viewer. Uploads a video, spawns the engine as a
  subprocess, polls a status file, and renders the per-point results.

**No Docker, no queue.** Portability ("localhost now → cloud or local-first later") comes from
the package structure, not containers. The engine is just a Python package you can run on a
rented GPU box (`git clone && pip install` on a RunPod/Vast PyTorch template) or bundle into a
desktop shell later.

### Pipeline

```
probe → court homography → player tracking → ball tracking
      → rally segmentation → per-rally serve / bounce / winner → result.json
```

| Stage | Tooling |
|---|---|
| Players | Ultralytics **YOLO26** + ByteTrack (`detect/players.py`) |
| Court | **TennisCourtDetector** → homography to a normalized pickleball court (`court/`) |
| Ball | **WASB-SBDT** + jump gating + Kalman smoothing (`ball/`) — the highest-risk stage |
| Rally / serve / winner | Heuristics on the rectified trajectory (`rally/`) |

## Quickstart

```bash
# Core engine + API (CPU dev box — runs the app and the pure-logic stages + tests)
pip install -e '.[api,dev]'

# Run the tests (pure-logic core, no model weights needed)
pytest

# Launch the local app
./scripts/run_local.sh           # http://localhost:8000

# Full model-backed runs (rented GPU box, or local for short clips)
pip install -e '.[ml]'
./scripts/download_weights.sh
python -m pbengine.pipeline match.mp4 -o result.json
```

## Status & caveats

- **Ball tracking** quality on real phone footage is the bottleneck (occlusion, motion blur,
  tiny/fast ball). Expect to iterate here most.
- **Point-winner** compounds ball + bounce + homography error, so low-confidence calls are
  flagged `needs_review` rather than asserted.
- **Ultralytics is AGPL-3.0** — fine for localhost single-user; revisit before distributing.

See `/root/.claude/plans/` or the project plan for the full design, risks, and roadmap.
