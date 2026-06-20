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
| Court | Auto: **TennisCourtDetector**; reliable fallback: **manual 4-corner calibration** (`court/`) | ✅ manual works; auto weak on pickleball |
| Ball | **WASB-SBDT** (Hydra-free wrapper) + jump gating + Kalman (`ball/`) — highest-risk stage | ✅ wired (needs weights); pickleball transfer unproven |
| Rally / serve / winner | Heuristics on the rectified trajectory (`rally/`) | ✅ logic done |

Each model-backed stage **degrades gracefully**: if its weights/deps are missing it's skipped
with a warning (surfaced in the CLI and viewer) instead of crashing, so you can bring the
engine up one model at a time.

### Enabling the real models

```bash
pip install -e '.[ml]'              # torch + ultralytics + gdown
git submodule update --init         # vendored TennisCourtDetector
./scripts/download_weights.sh       # fetches the court weights via gdown

# Player detection ships as sensitivity presets (model + imgsz + conf + stride + tracker):
pbengine clip.mp4 -o result.json --players-preset max        # best capture (slow on CPU)
pbengine clip.mp4 -o result.json --players-preset fast       # quick; may miss far players
# Override any single knob on top of a preset:
pbengine clip.mp4 -o result.json --players-preset balanced --players-imgsz 1536 --players-conf 0.1
```

| preset | model | imgsz | conf | stride | tracker | use |
|---|---|---|---|---|---|---|
| `fast` | yolo11n-pose | 640 | 0.30 | 3 | default | quick CPU pass; misses far players, skeletons glide |
| `balanced` *(default)* | yolo11m-pose | 960 | 0.15 | 1 | sensitive | far players captured, natural motion; minutes/clip on CPU |
| `max` | yolo11m-pose | 1280 | 0.10 | 1 | sensitive + augment | best capture; slow on CPU |
| `gpu` | yolo11x-pose | 1280 | 0.10 | 1 | sensitive | for a GPU box |

Higher sensitivity catches far/small players and gives dense, *articulated* poses (vs. interpolated
gliding) at the cost of speed (imgsz is ~quadratic, stride 1 ~3×) and some extra false positives.
In the **web app**, pick the preset from the *Detection* dropdown before *Analyze*. Env overrides for
the API: `PBV_PLAYERS_PRESET`, `PBV_PLAYERS_WEIGHTS`, `PBV_VID_STRIDE`, `PBV_PLAYERS_IMGSZ`,
`PBV_PLAYERS_CONF` (e.g. a plain detector like `yolo26m.pt` to skip skeletons).

> The court net is **tennis-trained** and in practice localizes a pickleball court poorly
> (often 0/4 corners). The reliable path is **manual calibration**: in the viewer, click
> *📐 Calibrate court*, click the four court corners on the first frame, and Save & re-analyze.
> Since the camera is static, this is a one-time step and gives an exact homography.

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
