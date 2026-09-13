# DinoFlow

基于 DINOv3 视觉特征和 conditional flow matching 的机器人动作块策略。这个仓库是从 `openpi_repo` 当前已经跑通的 DinoFlow 实现中独立出来的，保留了 LeRobot 数据读取、训练循环、checkpoint 保存和推理 server；不会把原仓库的历史输出、数据集或模型权重复制进来。

## 当前实现

- 冻结 DINOv3 ViT-S/16+，输入三路相机图像。
- base 图像保持 `480×768`，两路 wrist 图像等比例缩放到高度 `480` 后中心裁到 `480×832`；当前 `480×848` wrist 输入只裁左右各 8 像素。base 为 `1440` 个 patch、每路 wrist 为 `1560` 个 patch，三路拼接为 `[B, 4560, 384]`，再通过共享 `Linear(384→256)` 交给 action DiT，并加入可学习的相机身份 embedding。
- DINOv3 的 12 个 attention block 只对 `q_proj`/`v_proj` 加 LoRA（`r=8`、`alpha=16`、LoRA lr=`2e-5`），原始 DINO 权重全部冻结；默认关闭 DINO gradient checkpointing 以减少重算、提高训练速度，需要节省显存时可显式开启。关闭 `vision_lora_enabled` 可使用冻结 DINO。
- 一次预测 50 步、26 维机器人绝对动作；256 是视觉和 Action DiT 的内部 latent 宽度，不是机器人命令维度。该基线不输入触觉、六维力或 contact token。
- 默认训练 batch 为 `16`，因为三路完整 patch token 在 32 GB GPU 上关闭 DINO checkpointing 时无法容纳 batch `32`；显存足够时可显式增大 batch。
- flow matching 使用 MSE velocity loss，推理支持 Euler/Heun 积分和可选 RTC chunk 对齐。
- 训练基于 LeRobot v3 数据集格式，支持 train/validation 两个本地数据集。

## 环境

本机已经配置好的环境是 `dinoflow_env`。当前已验证：Python 3.12、PyTorch 2.10.0 + CUDA 12.8、Transformers 5.5.4、Accelerate 1.13、Datasets 4.8、W&B 0.24。

先运行一次安装/检查：

```bash
cd /home/nolan/vla/dinoflow_repo
bash scripts/setup_current_env.sh
```

这个脚本会激活现有 `dinoflow_env`，检查 GPU 和关键依赖，并以 editable 方式注册当前仓库；不会复制数据或权重。

如果以后需要从头创建环境，可以使用：

```bash
conda env create -f environment.yml
conda activate dinoflow_env
```

## 快速 smoke test

使用现有数据和本地 DINOv3 权重跑 1 个训练 step：

```bash
cd /home/nolan/vla/dinoflow_repo
bash scripts/smoke_train.sh
```

默认路径是当前机器已有的：

```text
数据集: /home/nolan/vla/openpi_repo/lerobot_datasets/splice_wires_phase1_split_300_21
DINO:   /home/nolan/models/dinov3-vits16plus
```

也可以显式指定路径：

```bash
DINOFLOW_DATASET_ROOT=/path/to/train \
DINOFLOW_VAL_DATASET_ROOT=/path/to/validation \
DINOFLOW_VISION_ENCODER=/path/to/dinov3-vits16plus \
bash scripts/smoke_train.sh
```

## 正式训练

先确认 smoke test 成功，然后启动默认 phase-1 配置：

```bash
cd /home/nolan/vla/dinoflow_repo
bash scripts/train_phase1.sh
```

常用覆盖项：

```bash
bash scripts/train_phase1.sh \
  --steps 30000 \
  --batch-size 32 \
  --num-workers 12 \
  --output-dir /path/to/outputs/run_name \
  --wandb
```

脚本也支持 `--dataset-root`、`--validation-dataset-root`、`--vision-encoder`、`--hidden-dim`、`--num-layers`、`--save-freq`、`--val-freq` 等参数；完整列表运行：

```bash
bash scripts/train_phase1.sh --help
```

checkpoint 会保存为：

```text
<output-dir>/checkpoints/<step>/pretrained_model/
```

## 推理 server

训练得到 checkpoint 后可以启动 TCP server：

```bash
cd /home/nolan/vla/dinoflow_repo
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/nolan/anaconda3/envs/dinoflow_env/bin/python \
  deployment/server.py \
  --model-path /path/to/pretrained_model \
  --host 0.0.0.0 --port 9001
```

`deployment/client_mock.py` 和 `deployment/client.py` 是原有 AgiBot WBC 接口适配器的独立副本。它们需要机器人 SDK 提供 `wbc_gdk.WbcGdk`；训练和 server smoke test 不依赖机器人硬件。

## 数据格式

训练入口要求 LeRobot v3 数据集目录，至少包含 `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`data/` 和三路相机对应的 `videos/`。本实现从 `observation.state` 前 26 维读取状态和动作，从以下三个 image feature 读取相机：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
observation.images.right_wrist_0_rgb
```

如果数据字段不同，先在 `src/lerobot/policies/dino_flow/configuration_dino_flow.py` 中调整 `image_resize_shapes` 和 state/action 维度。

## 目录说明

```text
src/lerobot/                 精简保留的 LeRobot 训练/数据/策略运行时
src/lerobot/policies/dino_flow/  DinoFlow config、模型和 processor
scripts/train_phase1.sh      正式训练入口
scripts/smoke_train.sh       1-step 最小验证
deployment/server.py         checkpoint 推理 server
tests/                       不需要数据和 DINO 权重的单元测试
```

## 版权

LeRobot 相关代码遵循仓库中的 Apache-2.0 许可证；DinoFlow 新增代码与配置也按 Apache-2.0 发布。
