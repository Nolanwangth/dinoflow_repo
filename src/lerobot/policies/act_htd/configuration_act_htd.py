#!/usr/bin/env python

"""Configuration for ACT-HTD: Action Chunking Transformer with Touch Dreaming.

Implements the HTD (Humanoid Transformer with Touch Dreaming) architecture
from "Learning Versatile Humanoid Manipulation with Touch Dreaming" (arXiv 2604.13015)
as a lerobot policy.
"""

from dataclasses import dataclass, field

import torch

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, FUTURE_OBS_STATE, NEXT_OBS_STATE, OBS_STATE

# ═══════════════════════════════════════════════════════════════════════════════
# Finger layout constants (copied from bbvla for self-contained policy)
# ═══════════════════════════════════════════════════════════════════════════════

FINGER_LAYOUTS: list[dict] = [
    {"name": "thumb", "rows": 5, "cols": 7, "taxels": 35},
    {"name": "index", "rows": 5, "cols": 12, "taxels": 60},
    {"name": "middle", "rows": 5, "cols": 12, "taxels": 60},
    {"name": "ring", "rows": 5, "cols": 12, "taxels": 60},
    {"name": "pinky", "rows": 4, "cols": 8, "taxels": 32},
    {"name": "thumb_bend", "rows": 5, "cols": 11, "taxels": 55},
]
NUM_FINGERS = len(FINGER_LAYOUTS)  # 6
TAXELS_PER_HAND = sum(f["taxels"] for f in FINGER_LAYOUTS)  # 302
TAXELS_BOTH_HANDS = TAXELS_PER_HAND * 2  # 604

# ═══════════════════════════════════════════════════════════════════════════════
# State slice indices (646-dim vector)
# ═══════════════════════════════════════════════════════════════════════════════

STATE_JOINTS_START, STATE_JOINTS_END = 0, 26       # arm(14) + hands(12), 不含head/waist
STATE_FORCE_START, STATE_FORCE_END = 30, 42         # wrist force left(6) + right(6)
STATE_TACTILE_LEFT_START, STATE_TACTILE_LEFT_END = 42, 42 + TAXELS_PER_HAND     # 302
STATE_TACTILE_RIGHT_START, STATE_TACTILE_RIGHT_END = 42 + TAXELS_PER_HAND, 42 + TAXELS_BOTH_HANDS  # 302

# ═══════════════════════════════════════════════════════════════════════════════
# Action group definitions (26-dim action → 3 groups)
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_GROUPS: dict[str, tuple[int, int]] = {
    "arm":        (0, 14),   # 14
    "left_hand":  (14, 20),  # 6
    "right_hand": (20, 26),  # 6
}


@PreTrainedConfig.register_subclass("act_htd")
@dataclass
class ACTHTDConfig(PreTrainedConfig):
    """Configuration for ACT-HTD policy.

    Architecture (Fig.4 of HTD paper):
      [3× Camera] → shared ResNet18 + cross-attn pooling → N_img tokens each
      [Proprio 26D] → Linear → 1 token
      [Force 12D] → L/R Linear → 2 tokens
      [Tactile 604D] → Per-finger Conv2d + cross-attn → 2 tokens (L/R latent)
         ↓ all tokens concatenated + modality/position embeddings
      Transformer Encoder → Transformer Decoder
         ↓ shared decoder hidden
      Modular Action Experts (3 groups) → (B, chunk_size, 26)
    """

    # ── Input / output structure ──
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── Vision backbone ──
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    image_downscale_ratio: float = 1.0  # 1.0=原始, 2.0=半分辨率

    # ── Transformer architecture ──
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    dropout: float = 0.1

    # ── No VAE (deterministic BC as per HTD paper) ──
    # use_vae is intentionally absent — HTD uses deterministic BC.

    # ── Modality tokenizer params ──
    num_image_tokens_per_camera: int = 16       # cross-attention query tokens per camera view
    num_image_cross_attn_heads: int = 4

    # ── Tactile encoder params ──
    tactile_encoder_channels: int = 8
    tactile_latent_dim: int = 64
    num_tactile_tokens: int = 2  # left + right hand latent

    # ── Inference ──
    temporal_ensemble_coeff: float | None = 0.01

    # ── Training ──
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # ── Touch dreaming ──
    touch_dreaming_enabled: bool = True
    force_pred_weight: float = 0.1
    tactile_pred_weight: float = 0.1
    variance_weight: float = 0.1
    ema_decay: float = 0.999
    tactile_latent_loss_beta: float = 0.1
    touch_dreaming_use_cosine_loss: bool = True
    touch_dreaming_pooling: str = "mean"  # mean | last | mid

    # Multi-step touch dreaming: predict future force/tactile at K horizon offsets
    touch_dreaming_horizon: int = 10
    touch_dreaming_min_offset: int = 1
    touch_dreaming_max_offset: int | None = None  # None → chunk_size - 1
    touch_dreaming_offsets: list[int] | None = None  # explicit offsets override linspace

    # Finger layouts (not serialized, set at init)
    finger_layouts: list[dict] = field(default_factory=lambda: FINGER_LAYOUTS)

    # ── State dimension constants (for model-internal slicing) ──
    max_state_dim: int = 646
    max_action_dim: int = 26

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got n_obs_steps={self.n_obs_steps}"
            )
        if self.touch_dreaming_pooling not in {"mean", "last", "mid"}:
            raise ValueError(
                f"touch_dreaming_pooling must be 'mean', 'last', or 'mid'. Got '{self.touch_dreaming_pooling}'."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("ACT-HTD requires at least one camera input.")
        if OBS_STATE not in self.input_features:
            raise ValueError("ACT-HTD requires observation.state (646-dim).")

        # Ensure state dim accommodates all sub-modalities
        expected_total = STATE_TACTILE_RIGHT_END  # 646
        if self.max_state_dim < expected_total:
            raise ValueError(
                f"max_state_dim ({self.max_state_dim}) must be >= {expected_total}"
            )
        if OBS_STATE in self.input_features:
            self.input_features[OBS_STATE].shape = (self.max_state_dim,)

        # Register next.observation.state for tau=1 touch dreaming target (only when enabled)
        if self.touch_dreaming_enabled and NEXT_OBS_STATE not in self.input_features:
            self.input_features[NEXT_OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.max_state_dim,)
            )

        # Register future.observation.state for multi-step touch dreaming (only when enabled)
        if self.touch_dreaming_enabled and FUTURE_OBS_STATE not in self.input_features:
            k = len(self.get_touch_dreaming_offsets())
            self.input_features[FUTURE_OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(k, self.max_state_dim,)
            )

        if ACTION in self.output_features:
            self.output_features[ACTION].shape = (self.max_action_dim,)

    # ── Delta indices ──

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def future_observation_delta_indices(self) -> list[int] | None:
        """Delta indices for fetching future observation.state from the dataset."""
        if not self.touch_dreaming_enabled:
            return None
        return self.get_touch_dreaming_offsets()

    # ── Touch dreaming offset helpers ──

    def get_touch_dreaming_offsets(self) -> list[int]:
        """Return the list of frame offsets for touch dreaming predictions."""
        max_off = self.chunk_size - 1
        if self.touch_dreaming_max_offset is not None:
            max_off = min(self.touch_dreaming_max_offset, max_off)

        if self.touch_dreaming_offsets is not None:
            raw = self.touch_dreaming_offsets
        else:
            raw_float = torch.linspace(
                float(self.touch_dreaming_min_offset),
                float(max_off),
                self.touch_dreaming_horizon,
            )
            raw = [round(float(o)) for o in raw_float]

        offsets: list[int] = sorted(set(max(1, min(int(o), max_off)) for o in raw))
        if not offsets:
            raise ValueError(
                f"touch_dreaming_offsets is empty after clamping to [1, {max_off}]."
            )
        return offsets
