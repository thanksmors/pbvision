#!/usr/bin/env bash
# Fetch model weights and vendored third-party repos into the engine.
# Weights are gitignored (engine/pbengine/models/). Run once per machine after install.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/engine/pbengine/models"
THIRD_PARTY="$ROOT/engine/pbengine/third_party"
mkdir -p "$MODELS" "$THIRD_PARTY"

echo "==> YOLO26 detector weights (downloaded lazily by ultralytics on first use)"
echo "    yolo26m.pt will be fetched automatically when the engine first runs."

echo "==> Ball tracker: WASB-SBDT (vendored as a git submodule)"
git submodule update --init "$THIRD_PARTY/WASB-SBDT" 2>/dev/null || \
  { [ -d "$THIRD_PARTY/WASB-SBDT" ] || \
    git clone --depth 1 https://github.com/nttcom/WASB-SBDT "$THIRD_PARTY/WASB-SBDT"; }

BALL_WEIGHTS="$MODELS/wasb_tennis_best.pth.tar"
if [ ! -f "$BALL_WEIGHTS" ]; then
  echo "    Fetching WASB tennis weights via gdown..."
  python -m gdown 14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z -O "$BALL_WEIGHTS" || \
    echo "    !! gdown failed. Download WASB (Tennis) from MODEL_ZOO.md and save it to
       $BALL_WEIGHTS manually. (Badminton weights also work — pickleball sits between them.)"
else
  echo "    Ball weights already present at $BALL_WEIGHTS."
fi

# Badminton weights as an A/B alternative — pickleball's ball size/speed sits between tennis and
# badminton. Compare coverage on your own footage with: scripts/debug_ball.py --sweep
BADMINTON_WEIGHTS="$MODELS/wasb_badminton_best.pth.tar"
if [ ! -f "$BADMINTON_WEIGHTS" ]; then
  echo "    Fetching WASB badminton weights via gdown (for A/B)..."
  python -m gdown 17Ac0pO5oryh1JwgwTFQTjOKHY3umbDQu -O "$BADMINTON_WEIGHTS" || \
    echo "    !! gdown failed. Optional: download WASB (Badminton) from MODEL_ZOO.md to
       $BADMINTON_WEIGHTS to A/B against tennis."
else
  echo "    Badminton weights already present at $BADMINTON_WEIGHTS."
fi

echo "==> Court detector: TennisCourtDetector (vendored as a git submodule)"
git submodule update --init "$THIRD_PARTY/TennisCourtDetector" 2>/dev/null || \
  { [ -d "$THIRD_PARTY/TennisCourtDetector" ] || \
    git clone --depth 1 https://github.com/yastrebksv/TennisCourtDetector "$THIRD_PARTY/TennisCourtDetector"; }

COURT_WEIGHTS="$MODELS/court_detector.pt"
if [ ! -f "$COURT_WEIGHTS" ]; then
  echo "    Fetching pretrained court weights via gdown..."
  python -m gdown 1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG -O "$COURT_WEIGHTS" || \
    echo "    !! gdown failed (network/Drive quota). Download the model from the TennisCourtDetector
       README and save it to $COURT_WEIGHTS manually."
else
  echo "    Court weights already present at $COURT_WEIGHTS."
fi

echo "Done. Weights live in $MODELS (gitignored)."
