#!/usr/bin/env bash
set -euo pipefail

DINOFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINOFLOW_CONDA_ENV="${DINOFLOW_CONDA_ENV:-dinoflow_env}"
DINOFLOW_INSTALL_DEPS=false

usage() {
  cat <<'EOF'
用法:
  bash scripts/setup_current_env.sh [--install-deps]

默认只检查并注册当前仓库，不改动已有依赖。
--install-deps  在检查前按 requirements.txt 补齐缺失依赖。
EOF
}

while (($# > 0)); do
  case "$1" in
    --install-deps)
      DINOFLOW_INSTALL_DEPS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知选项: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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

if [[ "$DINOFLOW_INSTALL_DEPS" == true ]]; then
  python -m pip install -r "$DINOFLOW_ROOT/requirements.txt"
fi

export PYTHONPATH="$DINOFLOW_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$DINOFLOW_ROOT"
python -m pip install --editable "$DINOFLOW_ROOT" --no-deps

python - <<'PY'
import importlib
import sys
from pathlib import Path

required = ["torch", "torchvision", "transformers", "accelerate", "datasets", "wandb", "ujson"]
for name in required:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'ok')}")

import torch
import lerobot
from lerobot.policies.factory import get_policy_class

repo_src = Path.cwd() / "src"
lerobot_path = Path(lerobot.__file__).resolve()
if repo_src not in lerobot_path.parents:
    raise RuntimeError(f"当前导入的 lerobot 不是新仓库源码: {lerobot_path}")
if get_policy_class("dino_flow").__name__ != "DinoFlowPolicy":
    raise RuntimeError("DinoFlow policy registration failed")

print(f"python: {sys.executable}")
print(f"lerobot: {lerobot_path}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
print("DinoFlow environment check: OK")
PY
