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

echo "==> Ball tracker: WASB-SBDT"
if [ ! -d "$THIRD_PARTY/WASB-SBDT" ]; then
  git clone --depth 1 https://github.com/nttcom/WASB-SBDT "$THIRD_PARTY/WASB-SBDT"
fi
echo "    Place tennis/badminton weights in $MODELS (see WASB-SBDT README)."

echo "==> Court detector: TennisCourtDetector"
if [ ! -d "$THIRD_PARTY/TennisCourtDetector" ]; then
  git clone --depth 1 https://github.com/yastrebksv/TennisCourtDetector "$THIRD_PARTY/TennisCourtDetector"
fi
echo "    Download the pretrained court model into $MODELS (see repo README)."

echo "Done. Weights live in $MODELS (gitignored)."
