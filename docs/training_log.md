# DinoFlow 工作日志

## Visuo_Tac

- 分支：`Visuo_Tac`，从已提交的 `visuo-basline` 开出；环境：`dinoflow_env`
- 数据：train `300 episodes / 137216 frames`；validation `21 episodes / 9021 frames`
- DINO：`/home/nolan/models/dinov3-vits16plus`；原始权重冻结，12 个 block 的 Q/V 使用 LoRA `r=8, alpha=16, lr=2e-5`
- 视觉：Head `480×768`；两路 wrist 原图 `480×848` 各裁左右 `8px` 为 `480×832`；full patch tokens 共 `4560`，共享 `Linear(384→512)`
- 力触：`observation.state` 保留 `646` 维；关节 `0:26`、双腕力 `30:42`、触觉 `42:646`；状态历史 `[-5..0]` 六帧，图像只取当前帧
- 接触编码：共享十二区域 CNN，每帧 `420` 维；双腕力 MLP `32` 维；六帧历史 MLP 输出 `128` 维；投影到 Action DiT 的时间调制条件
- 动作：6 层 Action DiT，`50×26` delta action；单一 flow matching loss，8 步 Euler
- 优化：DiT/新接触模块 `1e-4`，DINO LoRA `2e-5`，cosine 最低 `1e-5`，bf16；默认 batch `32`
- RTC：预测 `H=50`，剩余 `30` 步时刷新，约执行 `20` 步；后台推理使用真实延迟对齐动作块
- 训练命令：`bash scripts/train_visuo_tac.sh`
- 输出：`outputs/visuo_tac_phase1_force_tactile_history6_lora_qv_r8_b32_lr1e-4_lora2e-5_30k_seed1000/`
- 保存：每 `5000` steps checkpoint；每 `2500` steps validation；W&B project `splice_wires_dinoflow`，不上传权重 artifact
- 正式运行：W&B [8hxtvkhp](https://wandb.ai/nolanwangth-karlsruhe-institute-of-technology/splice_wires_dinoflow/runs/8hxtvkhp)，2026-09-11 21:48 启动

## 验证

- 真实数据样本：`state=[6,646]`、`action=[50,32]`，预处理后 action 为 `[50,26]`
- 单测：8 passed
- 真实 DINO batch 32 单步训练：通过；可训练参数 `36,565,018`，总参数 `65,257,882`
- 带 validation/checkpoint 的 smoke：通过；验证和 checkpoint 保存均正常
- checkpoint 严格重载：通过；已修复保存配置含 `type` 时的加载顺序问题
