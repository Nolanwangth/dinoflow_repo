# DinoFlow baseline 工作日志

## 实验固定信息

- 分支：`visuo-basline`，基于 `main` commit `7eabb4e`
- 环境：`dinoflow_env`
- 参考代码：`/home/nolan/vla/openpi_repo/deploy/dino_flow`
- DINOv3：`/home/nolan/models/dinov3-vits16plus`
- 训练集：`splice_wires_phase1_split_300_21/train`，300 episodes / 137216 frames
- 验证集：`splice_wires_phase1_split_300_21/validation`，21 episodes / 9021 frames
- W&B project：`splice_wires_dinoflow`
- 权重只保存到 `outputs/`，不上传 W&B artifact

## 当前 baseline

- Head：`480×768`；每路 wrist：原图 `480×848` 左右各裁 `8px`，得到 `480×832`
- DINO patch tokens：Head `1440`，每路 wrist `1560`，三路拼接 `4560`
- 去掉 CLS/register tokens；DINO 原始权重冻结，12 个 block 的 attention Q/V 使用 LoRA
- LoRA：`r=8`、`alpha=16`、dropout `0`、lr `2e-5`
- 视觉投影：`Linear(384,512)`；无 Resampler、视觉位置编码、camera embedding、触觉和六维力
- Action DiT：6 层、hidden `512`、8 heads；state `26` 维；delta action
- Flow Matching：8 步 Euler
- 优化：Adam，DiT lr `1e-4`，weight decay `1e-6`，cosine scheduler 最低 lr `1e-5`，bf16
- batch `32`，workers `12`，训练 `30000` steps；每 `2500` steps 验证，每 `5000` steps 保存
- 验证配置：batch `8`，采样 `16` 帧
- wrist 当前无 padding；代码保留视觉有效性 mask，供 Action DiT cross-attention 使用

当前参数量：可训练参数 `35,762,202`，总参数 `64,455,066`。LoRA 新增 `147,456` 个参数；DINO 原始参数仍冻结。

## 代码与验证

- 主要代码：`configuration_dino_flow.py`、`modeling_dino_flow.py`、`scripts/train_phase1.sh`、`tests/test_dino_flow.py`
- 单测：2026-09-11，`6 passed`
- 真实 DINO 权重的 LoRA smoke：batch `1` 通过，Q/V adapter tensors `48` 个
- gradient checkpointing batch 探测：`32/64/128` 均通过；正式训练采用 batch `32`
- checkpoint 保存、严格重载测试：通过

## 当前训练

- 模型：DINOv3-S+/16 + Q/V LoRA + Full Patch Tokens + Action DiT
- 状态：运行中，约 step `2150/30000`
- W&B：[nhcgw57h](https://wandb.ai/nolanwangth-karlsruhe-institute-of-technology/splice_wires_dinoflow/runs/nhcgw57h)

输出目录：`outputs/visuo-basline_phase1_wrist832_lora_qv_r8_b32_lr1e-4_lora2e-5_30k_seed1000/`

当前约 `1.3 step/s`，预计总训练约 `6.3–6.5` 小时；验证在 step `2500`，checkpoint 在 step `5000`。

## 权重清理

2026-09-11 按要求删除了此前旧训练的 `005000/010000` checkpoint，以及所有临时 smoke checkpoint。旧日志和 W&B 记录保留；当前训练的正式权重只会写入上述 `outputs/` 目录。
