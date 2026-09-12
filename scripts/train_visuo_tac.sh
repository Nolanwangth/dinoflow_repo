#!/usr/bin/env bash
set -euo pipefail

DINOFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINOFLOW_OUTPUT_DIR="${DINOFLOW_OUTPUT_DIR:-$DINOFLOW_ROOT/outputs/visuo_tac_phase1_force_tactile_history6_absolute_action_h256_lora_qv_r8_b32_lr1e-4_lora2e-5_30k_seed1000}"
DINOFLOW_JOB_NAME="${DINOFLOW_JOB_NAME:-$(basename "$DINOFLOW_OUTPUT_DIR")}"

exec bash "$DINOFLOW_ROOT/scripts/train_phase1.sh" \
  --steps "${DINOFLOW_STEPS:-30000}" \
  --batch-size "${DINOFLOW_BATCH_SIZE:-32}" \
  --num-workers "${DINOFLOW_NUM_WORKERS:-12}" \
  --prefetch-factor "${DINOFLOW_PREFETCH_FACTOR:-2}" \
  --log-freq "${DINOFLOW_LOG_FREQ:-50}" \
  --val-freq "${DINOFLOW_VAL_FREQ:-2500}" \
  --val-batch-size "${DINOFLOW_VAL_BATCH_SIZE:-8}" \
  --val-num-frames "${DINOFLOW_VAL_NUM_FRAMES:-128}" \
  --save-freq "${DINOFLOW_SAVE_FREQ:-5000}" \
  --vision-lora \
  --vision-gradient-checkpointing \
  --vision-lora-rank "${DINOFLOW_VISION_LORA_RANK:-8}" \
  --vision-lora-alpha "${DINOFLOW_VISION_LORA_ALPHA:-16}" \
  --vision-lora-lr "${DINOFLOW_VISION_LORA_LR:-2e-5}" \
  --optimizer-lr "${DINOFLOW_OPTIMIZER_LR:-1e-4}" \
  --hidden-dim "${DINOFLOW_HIDDEN_DIM:-256}" \
  --wandb \
  --save-checkpoint \
  --absolute-action \
  --output-dir "$DINOFLOW_OUTPUT_DIR" \
  --job-name "$DINOFLOW_JOB_NAME" \
  "$@"
