# DinoFlow baseline 工作日志

## 当前实验：纯视觉＋state absolute-action baseline（2026-09-13）

- 分支：`visuo-basline`
- 目标：建立不使用触觉、腕力和 contact token 的纯视觉＋state 对照基线
- 数据：`splice_wires_phase1_split_300_21`，train 300 episodes / 137216 frames，validation 21 episodes / 9021 frames
- 环境：`dinoflow_env`
- DINOv3：`/home/nolan/models/dinov3-vits16plus`
- 相机：base `480×768`，左右 wrist `480×832`
- 视觉：三路共 `4560` 个 patch token，DINO 原生 `384` 维通过共享 `Linear(384,256)` 投影；每路加入可学习相机身份 embedding
- 动作：物理命令为 26 维，内部 action token latent 为 `256` 维；直接学习归一化 absolute action，`use_delta_action=false`
- Action DiT：6 层、256 hidden、8 heads；Flow Matching 为 8 步 Euler
- DINO：Q/V LoRA 微调，rank `8`、alpha `16`、dropout `0`、lr `2e-5`；其余 DINO 权重冻结；gradient checkpointing 开启
- 优化：Adam，Action DiT lr `1e-4`，weight decay `1e-6`，cosine scheduler 最低 lr `1e-5`，warmup `500` 步，bf16
- 训练：batch `32`，workers `12`，目标 `30000` steps；每 `1000` steps 验证，每次采样 `128` 帧，每 `5000` steps 保存
- W&B project：`splice_wires_dinoflow`；job name：`visuo_baseline_phase1_absolute_h256_camid_fullpatch_loraqv_r8_b32_30k_seed1000`
- smoke test：真实数据 1 step 已通过，DINO Q/V LoRA 已成功加载，训练前向/反向完成
- 正式训练 W&B：[kou0rfsb](https://wandb.ai/nolanwangth-karlsruhe-institute-of-technology/splice_wires_dinoflow/runs/kou0rfsb)
- 正式输出：`outputs/visuo_baseline_phase1_absolute_h256_camid_fullpatch_loraqv_r8_b32_30k_seed1000_20260913/`
- 启动后速度约 `1.3–1.4 step/s`，预计约 6 小时；首次 validation 在 step `1000`，首次 checkpoint 在 step `5000`

该实验配置是当前文档中的有效 baseline。下面保留此前 512 hidden、delta action 的历史记录，便于追溯，不作为本次比较基线。

## 实验固定信息

- 分支：`visuo-basline`，基于 `main` commit `7eabb4e`
- 环境：`dinoflow_env`
- 参考代码：`/home/nolan/vla/openpi_repo/deploy/dino_flow`
- DINOv3：`/home/nolan/models/dinov3-vits16plus`
- 训练集：`splice_wires_phase1_split_300_21/train`，300 episodes / 137216 frames
- 验证集：`splice_wires_phase1_split_300_21/validation`，21 episodes / 9021 frames
- W&B project：`splice_wires_dinoflow`
- 权重只保存到 `outputs/`，不上传 W&B artifact

## 历史 baseline（已归档）

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

## 历史训练记录（2026-09-11）

- 模型：DINOv3-S+/16 + Q/V LoRA + Full Patch Tokens + Action DiT
- 状态：历史记录，约 step `2150/30000`
- W&B：[nhcgw57h](https://wandb.ai/nolanwangth-karlsruhe-institute-of-technology/splice_wires_dinoflow/runs/nhcgw57h)

输出目录：`outputs/visuo-basline_phase1_wrist832_lora_qv_r8_b32_lr1e-4_lora2e-5_30k_seed1000/`

当前约 `1.3 step/s`，预计总训练约 `6.3–6.5` 小时；验证在 step `2500`，checkpoint 在 step `5000`。

## 权重清理

2026-09-11 按要求删除了此前旧训练的 `005000/010000` checkpoint，以及所有临时 smoke checkpoint。旧日志和 W&B 记录保留；当前训练的正式权重只会写入上述 `outputs/` 目录。
