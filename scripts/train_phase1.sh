#!/usr/bin/env bash
set -euo pipefail

DINOFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINOFLOW_CONDA_ENV="${DINOFLOW_CONDA_ENV:-dinoflow_env}"
DINOFLOW_DATASET_ROOT="${DINOFLOW_DATASET_ROOT:-$DINOFLOW_ROOT/../openpi_repo/lerobot_datasets/splice_wires_phase1_split_300_21/train}"
DINOFLOW_VAL_DATASET_ROOT="${DINOFLOW_VAL_DATASET_ROOT:-$DINOFLOW_ROOT/../openpi_repo/lerobot_datasets/splice_wires_phase1_split_300_21/validation}"
DINOFLOW_VISION_ENCODER="${DINOFLOW_VISION_ENCODER:-/home/nolan/models/dinov3-vits16plus}"
DINOFLOW_OUTPUT_DIR="${DINOFLOW_OUTPUT_DIR:-$DINOFLOW_ROOT/outputs/dinoflow_phase1_$(date +%Y%m%d_%H%M%S)}"
DINOFLOW_JOB_NAME="${DINOFLOW_JOB_NAME:-$(basename "$DINOFLOW_OUTPUT_DIR")}"
DINOFLOW_DATASET_REPO_ID="${DINOFLOW_DATASET_REPO_ID:-local/splice_wires_phase1_train}"
DINOFLOW_VAL_DATASET_REPO_ID="${DINOFLOW_VAL_DATASET_REPO_ID:-local/splice_wires_phase1_validation}"
DINOFLOW_STEPS="${DINOFLOW_STEPS:-30000}"
DINOFLOW_BATCH_SIZE="${DINOFLOW_BATCH_SIZE:-32}"
DINOFLOW_NUM_WORKERS="${DINOFLOW_NUM_WORKERS:-12}"
DINOFLOW_PREFETCH_FACTOR="${DINOFLOW_PREFETCH_FACTOR:-2}"
DINOFLOW_LOG_FREQ="${DINOFLOW_LOG_FREQ:-50}"
DINOFLOW_VAL_FREQ="${DINOFLOW_VAL_FREQ:-2500}"
DINOFLOW_VAL_BATCH_SIZE="${DINOFLOW_VAL_BATCH_SIZE:-8}"
DINOFLOW_VAL_NUM_FRAMES="${DINOFLOW_VAL_NUM_FRAMES:-16}"
DINOFLOW_SAVE_FREQ="${DINOFLOW_SAVE_FREQ:-5000}"
DINOFLOW_HIDDEN_DIM="${DINOFLOW_HIDDEN_DIM:-256}"
DINOFLOW_NUM_LAYERS="${DINOFLOW_NUM_LAYERS:-6}"
DINOFLOW_NUM_HEADS="${DINOFLOW_NUM_HEADS:-8}"
DINOFLOW_INTEGRATION_STEPS="${DINOFLOW_INTEGRATION_STEPS:-8}"
DINOFLOW_INTEGRATION_METHOD="${DINOFLOW_INTEGRATION_METHOD:-euler}"
DINOFLOW_VISION_DIM="${DINOFLOW_VISION_DIM:-384}"
DINOFLOW_VISION_LORA_ENABLED="${DINOFLOW_VISION_LORA_ENABLED:-true}"
DINOFLOW_VISION_LORA_RANK="${DINOFLOW_VISION_LORA_RANK:-8}"
DINOFLOW_VISION_LORA_ALPHA="${DINOFLOW_VISION_LORA_ALPHA:-16}"
DINOFLOW_VISION_LORA_DROPOUT="${DINOFLOW_VISION_LORA_DROPOUT:-0.0}"
DINOFLOW_VISION_LORA_LR="${DINOFLOW_VISION_LORA_LR:-2e-5}"
DINOFLOW_VISION_GRADIENT_CHECKPOINTING="${DINOFLOW_VISION_GRADIENT_CHECKPOINTING:-true}"
DINOFLOW_OPTIMIZER_LR="${DINOFLOW_OPTIMIZER_LR:-1e-4}"
DINOFLOW_SCHEDULER_DECAY_LR="${DINOFLOW_SCHEDULER_DECAY_LR:-1e-5}"
DINOFLOW_WEIGHT_DECAY="${DINOFLOW_WEIGHT_DECAY:-1e-6}"
DINOFLOW_WANDB_ENABLE="${DINOFLOW_WANDB_ENABLE:-false}"
DINOFLOW_SAVE_CHECKPOINT="${DINOFLOW_SAVE_CHECKPOINT:-true}"
DINOFLOW_USE_DELTA_ACTION="${DINOFLOW_USE_DELTA_ACTION:-false}"
DINOFLOW_SEED="${DINOFLOW_SEED:-1000}"

usage() {
  cat <<'EOF'
用法:
  bash scripts/train_phase1.sh [选项]

数据/模型:
  --dataset-root PATH             train 数据集根目录
  --validation-dataset-root PATH validation 数据集根目录
  --vision-encoder PATH           DINOv3 本地目录或 Hugging Face 名称
  --output-dir PATH               checkpoint 和日志目录
  --job-name NAME                 训练任务名称

训练:
  --steps N                       训练步数，默认 30000
  --batch-size N                  batch size，默认 32
  --num-workers N                 DataLoader worker 数，默认 12
  --prefetch-factor N             每个 worker 预取数量，默认 2
  --log-freq N                    日志频率，默认 50
  --hidden-dim N                  action DiT 隐藏维度，默认 256
  --num-layers N                  action DiT 层数，默认 6
  --num-heads N                   attention heads，默认 8
  --num-integration-steps N       推理积分步数，默认 8
  --integration-method NAME       euler 或 heun，默认 euler
  --vision-lora / --no-vision-lora DINO Q/V LoRA，默认开启，rank=8
  --vision-lora-rank N             DINO LoRA rank，默认 8
  --vision-lora-alpha N            DINO LoRA alpha，默认 16
  --vision-lora-lr LR               DINO LoRA 学习率，默认 2e-5
  --vision-gradient-checkpointing / --no-vision-gradient-checkpointing
                                  LoRA 训练时对 DINO 重算激活，默认开启
  --optimizer-lr LR               学习率，默认 1e-4
  --scheduler-decay-lr LR         cosine 最低学习率，默认 1e-5
  --save-freq N                   checkpoint 保存频率，默认 5000
  --val-freq N                    validation 频率，默认 2500；0 表示关闭
  --val-batch-size N              validation batch size，默认 8
  --val-num-frames N              每次 validation 采样帧数，默认 16
  --seed N                        随机种子，默认 1000

开关:
  --wandb / --no-wandb            开启/关闭 W&B，默认关闭
  --save-checkpoint               保存 checkpoint（默认）
  --no-save-checkpoint            不保存 checkpoint
  --delta-action / --absolute-action
                                  使用 delta/absolute action，默认 absolute
  -h, --help                      显示帮助

也可以通过同名 DINOFLOW_* 环境变量覆盖默认值。
EOF
}

require_value() {
  if (($# < 2)) || [[ -z "${2:-}" ]]; then
    echo "$1 需要一个参数" >&2
    exit 2
  fi
}

while (($# > 0)); do
  case "$1" in
    --dataset-root) require_value "$1" "${2:-}"; DINOFLOW_DATASET_ROOT="$2"; shift 2 ;;
    --validation-dataset-root) require_value "$1" "${2:-}"; DINOFLOW_VAL_DATASET_ROOT="$2"; shift 2 ;;
    --vision-encoder) require_value "$1" "${2:-}"; DINOFLOW_VISION_ENCODER="$2"; shift 2 ;;
    --output-dir) require_value "$1" "${2:-}"; DINOFLOW_OUTPUT_DIR="$2"; DINOFLOW_JOB_NAME="$(basename "$2")"; shift 2 ;;
    --job-name) require_value "$1" "${2:-}"; DINOFLOW_JOB_NAME="$2"; shift 2 ;;
    --steps) require_value "$1" "${2:-}"; DINOFLOW_STEPS="$2"; shift 2 ;;
    --batch-size) require_value "$1" "${2:-}"; DINOFLOW_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) require_value "$1" "${2:-}"; DINOFLOW_NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) require_value "$1" "${2:-}"; DINOFLOW_PREFETCH_FACTOR="$2"; shift 2 ;;
    --log-freq) require_value "$1" "${2:-}"; DINOFLOW_LOG_FREQ="$2"; shift 2 ;;
    --hidden-dim) require_value "$1" "${2:-}"; DINOFLOW_HIDDEN_DIM="$2"; shift 2 ;;
    --num-layers) require_value "$1" "${2:-}"; DINOFLOW_NUM_LAYERS="$2"; shift 2 ;;
    --num-heads) require_value "$1" "${2:-}"; DINOFLOW_NUM_HEADS="$2"; shift 2 ;;
    --num-integration-steps) require_value "$1" "${2:-}"; DINOFLOW_INTEGRATION_STEPS="$2"; shift 2 ;;
    --integration-method) require_value "$1" "${2:-}"; DINOFLOW_INTEGRATION_METHOD="$2"; shift 2 ;;
    --vision-lora) DINOFLOW_VISION_LORA_ENABLED=true; shift ;;
    --no-vision-lora) DINOFLOW_VISION_LORA_ENABLED=false; shift ;;
    --vision-lora-rank) require_value "$1" "${2:-}"; DINOFLOW_VISION_LORA_RANK="$2"; shift 2 ;;
    --vision-lora-alpha) require_value "$1" "${2:-}"; DINOFLOW_VISION_LORA_ALPHA="$2"; shift 2 ;;
    --vision-lora-lr) require_value "$1" "${2:-}"; DINOFLOW_VISION_LORA_LR="$2"; shift 2 ;;
    --vision-gradient-checkpointing) DINOFLOW_VISION_GRADIENT_CHECKPOINTING=true; shift ;;
    --no-vision-gradient-checkpointing) DINOFLOW_VISION_GRADIENT_CHECKPOINTING=false; shift ;;
    --optimizer-lr) require_value "$1" "${2:-}"; DINOFLOW_OPTIMIZER_LR="$2"; shift 2 ;;
    --scheduler-decay-lr) require_value "$1" "${2:-}"; DINOFLOW_SCHEDULER_DECAY_LR="$2"; shift 2 ;;
    --save-freq) require_value "$1" "${2:-}"; DINOFLOW_SAVE_FREQ="$2"; shift 2 ;;
    --val-freq) require_value "$1" "${2:-}"; DINOFLOW_VAL_FREQ="$2"; shift 2 ;;
    --val-batch-size) require_value "$1" "${2:-}"; DINOFLOW_VAL_BATCH_SIZE="$2"; shift 2 ;;
    --val-num-frames) require_value "$1" "${2:-}"; DINOFLOW_VAL_NUM_FRAMES="$2"; shift 2 ;;
    --seed) require_value "$1" "${2:-}"; DINOFLOW_SEED="$2"; shift 2 ;;
    --wandb) DINOFLOW_WANDB_ENABLE=true; shift ;;
    --no-wandb) DINOFLOW_WANDB_ENABLE=false; shift ;;
    --save-checkpoint) DINOFLOW_SAVE_CHECKPOINT=true; shift ;;
    --no-save-checkpoint) DINOFLOW_SAVE_CHECKPOINT=false; shift ;;
    --delta-action) DINOFLOW_USE_DELTA_ACTION=true; shift ;;
    --absolute-action) DINOFLOW_USE_DELTA_ACTION=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知选项: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$DINOFLOW_DATASET_ROOT" ]]; then
  echo "train 数据集不存在: $DINOFLOW_DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DINOFLOW_VAL_DATASET_ROOT" ]]; then
  echo "validation 数据集不存在: $DINOFLOW_VAL_DATASET_ROOT" >&2
  exit 1
fi
if [[ "$DINOFLOW_VISION_ENCODER" = /* && ! -d "$DINOFLOW_VISION_ENCODER" ]]; then
  echo "本地 DINO checkpoint 不存在: $DINOFLOW_VISION_ENCODER" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$DINOFLOW_CONDA_ENV" ]]; then
  DINOFLOW_CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
  if [[ -z "$DINOFLOW_CONDA_EXE" ]]; then
    echo "找不到 conda；请先加载 Conda，或设置 CONDA_EXE。" >&2
    exit 1
  fi
  DINOFLOW_CONDA_BASE="$("$DINOFLOW_CONDA_EXE" info --base)"
  # shellcheck disable=SC1091
  source "$DINOFLOW_CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$DINOFLOW_CONDA_ENV"
fi

export PYTHONPATH="$DINOFLOW_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "DinoFlow phase-1 training"
echo "Repo:       $DINOFLOW_ROOT"
echo "Dataset:    $DINOFLOW_DATASET_ROOT"
echo "Validation: $DINOFLOW_VAL_DATASET_ROOT"
echo "Encoder:    $DINOFLOW_VISION_ENCODER"
echo "Steps:      $DINOFLOW_STEPS | Batch: $DINOFLOW_BATCH_SIZE | Workers: $DINOFLOW_NUM_WORKERS"
echo "Output:     $DINOFLOW_OUTPUT_DIR"
echo "=========================================="

exec python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id "$DINOFLOW_DATASET_REPO_ID" \
  --dataset.root "$DINOFLOW_DATASET_ROOT" \
  --dataset.return_uint8 true \
  --dataset.video_backend pyav \
  --dataset.use_imagenet_stats false \
  --validation_dataset.repo_id "$DINOFLOW_VAL_DATASET_REPO_ID" \
  --validation_dataset.root "$DINOFLOW_VAL_DATASET_ROOT" \
  --validation_dataset.return_uint8 true \
  --validation_dataset.video_backend pyav \
  --validation_dataset.use_imagenet_stats false \
  --policy.type dino_flow \
  --policy.vision_encoder_name "$DINOFLOW_VISION_ENCODER" \
  --policy.vision_encoder_dim "$DINOFLOW_VISION_DIM" \
  --policy.vision_lora_enabled "$DINOFLOW_VISION_LORA_ENABLED" \
  --policy.vision_lora_rank "$DINOFLOW_VISION_LORA_RANK" \
  --policy.vision_lora_alpha "$DINOFLOW_VISION_LORA_ALPHA" \
  --policy.vision_lora_dropout "$DINOFLOW_VISION_LORA_DROPOUT" \
  --policy.vision_lora_lr "$DINOFLOW_VISION_LORA_LR" \
  --policy.vision_gradient_checkpointing "$DINOFLOW_VISION_GRADIENT_CHECKPOINTING" \
  --policy.use_amp true \
  --policy.horizon 50 \
  --policy.n_action_steps 50 \
  --policy.do_mask_loss_for_padding false \
  --policy.hidden_dim "$DINOFLOW_HIDDEN_DIM" \
  --policy.num_layers "$DINOFLOW_NUM_LAYERS" \
  --policy.num_heads "$DINOFLOW_NUM_HEADS" \
  --policy.num_integration_steps "$DINOFLOW_INTEGRATION_STEPS" \
  --policy.integration_method "$DINOFLOW_INTEGRATION_METHOD" \
  --policy.use_delta_action "$DINOFLOW_USE_DELTA_ACTION" \
  --policy.scheduler_decay_lr "$DINOFLOW_SCHEDULER_DECAY_LR" \
  --policy.optimizer_lr "$DINOFLOW_OPTIMIZER_LR" \
  --policy.optimizer_weight_decay "$DINOFLOW_WEIGHT_DECAY" \
  --policy.push_to_hub false \
  --output_dir "$DINOFLOW_OUTPUT_DIR" \
  --job_name "$DINOFLOW_JOB_NAME" \
  --steps "$DINOFLOW_STEPS" \
  --batch_size "$DINOFLOW_BATCH_SIZE" \
  --num_workers "$DINOFLOW_NUM_WORKERS" \
  --prefetch_factor "$DINOFLOW_PREFETCH_FACTOR" \
  --log_freq "$DINOFLOW_LOG_FREQ" \
  --save_freq "$DINOFLOW_SAVE_FREQ" \
  --val_freq "$DINOFLOW_VAL_FREQ" \
  --val_batch_size "$DINOFLOW_VAL_BATCH_SIZE" \
  --val_num_frames "$DINOFLOW_VAL_NUM_FRAMES" \
  --save_checkpoint "$DINOFLOW_SAVE_CHECKPOINT" \
  --wandb.enable "$DINOFLOW_WANDB_ENABLE" \
  --wandb.project splice_wires_dinoflow \
  --wandb.disable_artifact true \
  --wandb.add_tags true \
  --seed "$DINOFLOW_SEED"
