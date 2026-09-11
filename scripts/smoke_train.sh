#!/usr/bin/env bash
set -euo pipefail

DINOFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINOFLOW_OUTPUT_DIR="${DINOFLOW_OUTPUT_DIR:-$DINOFLOW_ROOT/outputs/smoke_$(date +%Y%m%d_%H%M%S)}"

exec bash "$DINOFLOW_ROOT/scripts/train_phase1.sh" \
  --steps 1 \
  --batch-size 1 \
  --num-workers 0 \
  --prefetch-factor 2 \
  --log-freq 1 \
  --hidden-dim 128 \
  --num-layers 1 \
  --num-heads 4 \
  --num-integration-steps 2 \
  --val-freq 0 \
  --no-save-checkpoint \
  --no-wandb \
  --output-dir "$DINOFLOW_OUTPUT_DIR" \
  --job-name dinoflow-smoke
