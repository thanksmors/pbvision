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

| Stage | Tooling | Status |
|---|---|---|
| Players | Ultralytics **YOLO26** + ByteTrack (`detect/players.py`) | ✅ wired + validated |
| Court | **TennisCourtDetector** (4 outer corners) → homography (`court/`) | ✅ wired (needs weights) |
| Ball | **WASB-SBDT** + jump gating + Kalman smoothing (`ball/`) — highest-risk stage | ⏳ stub |
| Rally / serve / winner | Heuristics on the rectified trajectory (`rally/`) | ✅ logic done |

Each model-backed stage **degrades gracefully**: if its weights/deps are missing it's skipped
with a warning (surfaced in the CLI and viewer) instead of crashing, so you can bring the
engine up one model at a time.

### Enabling the real models

```bash
pip install -e '.[ml]'              # torch + ultralytics + gdown
git submodule update --init         # vendored TennisCourtDetector
./scripts/download_weights.sh       # fetches the court weights via gdown

# Players use yolo26n + frame-striding by default on CPU; override on a GPU box:
pbengine clip.mp4 -o result.json --players-weights yolo26m.pt --vid-stride 1
```

> The court net is **tennis-trained**: it localizes the four outer court corners, which we map
> to the normalized pickleball court. Accuracy on pickleball footage is the open question of
> the automatic approach — a manual 4-corner override / pickleball fine-tune is the fallback.

## Quickstart

```bash
# Core engine + API (CPU dev box — runs the app, the demo, the pure-logic stages + tests)
pip install -e '.[api,dev]'

# Run the tests (pure-logic core + fixture pipeline, no model weights needed)
pytest

# Launch the local app
./scripts/run_local.sh           # http://localhost:8000

# Full model-backed runs (rented GPU box, or local for short clips)
pip install -e '.[ml]'
./scripts/download_weights.sh
python -m pbengine.pipeline match.mp4 -o result.json
```

## Try it now — synthetic demo (no ML, no video, CPU-only)

The three model-backed stages are **dependency-injected**, so a set of scripted stand-ins
(`engine/pbengine/fixtures.py`) drives the *entire* pipeline + viewer with nothing but the
core deps installed. This validates the logic (segmentation, serve/bounce/winner, the JSON
contract) and the viewer **before** the real ML glue exists.

```bash
# CLI: generates a synthetic court+ball video and analyzes it
python -m pbengine.pipeline demo.mp4 -o result.json --fixture

# App: click “Run demo” at http://localhost:8000 (POST /api/demo)
```

The demo produces three rallies exercising each win-reason branch — `ball_out`,
`double_bounce`, and `net` — with the rendered video, ball trajectory overlay, and bounce
markers shown in the viewer.

## Status & caveats

- **Ball tracking** quality on real phone footage is the bottleneck (occlusion, motion blur,
  tiny/fast ball). Expect to iterate here most.
- **Point-winner** compounds ball + bounce + homography error, so low-confidence calls are
  flagged `needs_review` rather than asserted.
- **Ultralytics is AGPL-3.0** — fine for localhost single-user; revisit before distributing.

See `/root/.claude/plans/` or the project plan for the full design, risks, and roadmap.
