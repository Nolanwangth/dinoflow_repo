# Bbvla — PI05 with Two-Level Tactile/Force Injection

> Based on PI05 (PaliGemma 2B + Action Expert 300M), extending it with haptic sensing and Touch Dreaming.

## Architecture

```
Prefix (PaliGemma 2B, 2048-dim, bidirectional attention):
  [img_base_rgb] [img_left_wrist] [img_right_wrist] [text prompt]
      256             256               256            ~150 tokens

  Text prompt format:
  "Task: <desc>, Joints: <30 vals>, Force: <12 labeled>, Touch: <12 finger avg>;\nAction: "

Suffix (Action Expert 300M, 1024-dim, causal attention):
  [tactile_conv_emb (2 tokens)] [force_emb (1 token)] [noisy_action_0 ... _49]
         ↑ Conv2d per finger        ↑ MLP 12→1024

Touch Dreaming (on last suffix hidden):
  force_pred_head: Linear 1024→12      predict wrist force
  tactile_pred_head: Linear 1024→64    predict tactile latent (EMA target)
```

## Key Design Decisions

1. **Two-level haptic injection**: Coarse info (per-finger average, wrist force) enters the VLM via text tokens; fine info (full taxel grid, 6-axis force) enters the Action Expert via continuous tokens.
2. **Shared Conv2d per finger**: 6 Conv2d kernels (one per finger type), left and right hands share weights. Total: ~480 params.
3. **Touch Dreaming**: Self-supervised auxiliary loss encourages the suffix representation to encode haptic state, which can later be used for force/tactile prediction during deployment.

## Data Dimensions

| Modality | Dim | Path |
|----------|-----|------|
| Joints (arm+hand+head+waist) | 30 | Text tokens in prefix |
| Wrist force (6-axis × 2) | 12 | Text (prefix) + Continuous (suffix) |
| Tactile (6 fingers × 2 hands) | 604 | Finger averages as text (prefix) + Full Conv2d (suffix) |
| Actions | 32 | Suffix (flow matching target) |
| Images | 3 × 224² | SigLIP → 256 tokens each |

## State Layout (646-dim)

```
[0:14]    arm (14)         [14:26]  hands (12)
[26:28]   head (2)         [28:30]  waist (2)
[30:36]   force left (6)   [36:42]  force right (6)
[42:344]  tactile left (302: thumb 35, index 60, middle 60, ring 60, pinky 32, thumb_bend 55)
[344:646] tactile right (302)
```

## Differences from PI05

| | PI05 | Bbvla |
|---|---|---|
| max_state_dim | 32 | 646 |
| max_action_dim | 32 | 32 |
| State in model | Text only | Text + Continuous (force+tactile) |
| Modalities | Joint position | Joints + Force + Tactile |
| Auxiliary loss | None | Touch Dreaming (force + tactile latent) |

## Training

```bash
conda activate bbvla_env
cd lerobot
uv run lerobot-train \
  --policy.type=bbvla \
  --policy.pretrained_path=pi05_base \
  --dataset.repo_id=local/agibot \
  --dataset.root=lerobot_datasets/agibot_tasks \
  --batch_size=8 \
  --steps=50000 \
  --output_dir=outputs/bbvla_run \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true
```

## Future Phases

- **Phase 2**: Use consecutive-frame data for true next-step touch prediction (currently self-reconstruction)
- **Phase 3**: Inference-time tactile latent loop (predict → observe → update)
- **Phase 4**: Online adaptation / LoRA posterior update
