#!/usr/bin/env bash
# Launch the local API + viewer natively (no Docker).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/engine:$ROOT"
exec uvicorn api.app.main:app --reload --port 8000
