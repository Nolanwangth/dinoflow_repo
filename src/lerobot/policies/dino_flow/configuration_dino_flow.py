#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig


@PreTrainedConfig.register_subclass("dino_flow")
@dataclass
class DinoFlowConfig(PreTrainedConfig):
    """DINOv3-conditioned action DiT trained with conditional flow matching."""

    n_obs_steps: int = 1
    # Predict and execute a 50-step action chunk. At an episode boundary the
    # dataset repeats the final valid action until this fixed horizon is full.
    horizon: int = 50
    n_action_steps: int = 50

    state_dim: int = 26
    action_dim: int = 26

    vision_encoder_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    vision_encoder_dim: int = 384
    vision_patch_size: int = 16
    # Freeze all original DINO weights and train only Q/V LoRA.
    vision_lora_enabled: bool = True
    vision_lora_rank: int = 8
    vision_lora_alpha: int = 16
    vision_lora_dropout: float = 0.0
    vision_lora_lr: float = 2e-5
    vision_gradient_checkpointing: bool = True
    # Shared policy latent width after projecting DINO's native 384-d features.
    # The physical robot action remains ``action_dim`` (26 joints); 256 is the
    # internal visual/action-token width, not the command dimensionality.
    hidden_dim: int = 256

    # Keep camera identity after concatenating the three camera token streams.
    use_camera_embedding: bool = True

    # Keep the head at 480x768 and preserve the wrist horizontal field of view
    # with a 480x832 target.  The preprocessing resizes to target height and
    # center-crops only excess width, so the current 480x848 wrist frames lose
    # 8 pixels on each side instead of being letterboxed down to 434x768.
    image_resize_shapes: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "observation.images.base_0_rgb": (480, 768),
            "observation.images.left_wrist_0_rgb": (480, 832),
            "observation.images.right_wrist_0_rgb": (480, 832),
        }
    )

    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    timestep_embed_dim: int = 256
    num_integration_steps: int = 8
    integration_method: str = "euler"
    sigma_min: float = 0.0
    # Keep sampled normalized actions inside the range expected by the
    # MIN_MAX action postprocessor. This only affects inference.
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Regress directly on normalized absolute actions. This branch is the
    # absolute-action comparison baseline; delta remains an opt-in mode for
    # compatibility with old runs.
    use_delta_action: bool = False

    # Non-zero floor for the cosine decay so the run doesn't starve LR to ~0
    # (the diffusers "cosine" preset decays all the way to zero).
    scheduler_decay_lr: float = 1e-5
    # None => cosine decay spans the whole run (resolved from cfg.steps in build()).
    scheduler_num_decay_steps: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    # Padding is copied from the final valid action and is intentionally used
    # as supervision for the fixed-length 50-step chunk.
    do_mask_loss_for_padding: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1
        self.validate_features()

    def validate_features(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.n_action_steps <= 0 or self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps must satisfy 0 < n_action_steps <= horizon")
        if self.vision_encoder_dim <= 0 or self.vision_patch_size <= 0:
            raise ValueError("vision_encoder_dim and vision_patch_size must be positive")
        if self.vision_lora_rank <= 0 or self.vision_lora_alpha <= 0:
            raise ValueError("vision_lora_rank and vision_lora_alpha must be positive")
        if not 0 <= self.vision_lora_dropout < 1:
            raise ValueError("vision_lora_dropout must satisfy 0 <= vision_lora_dropout < 1")
        if not 0 < self.vision_lora_lr <= self.optimizer_lr:
            raise ValueError("vision_lora_lr must satisfy 0 < vision_lora_lr <= optimizer_lr")
        if self.hidden_dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_dim, num_layers, and num_heads must be positive")
        if self.timestep_embed_dim <= 0 or self.timestep_embed_dim % 2 != 0:
            raise ValueError("timestep_embed_dim must be a positive even number")
        if self.num_integration_steps <= 0:
            raise ValueError("num_integration_steps must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.use_delta_action and self.state_dim != self.action_dim:
            raise ValueError("use_delta_action=True requires state_dim == action_dim (delta = action - state)")
        if not 0 < self.scheduler_decay_lr <= self.optimizer_lr:
            raise ValueError("scheduler_decay_lr must satisfy 0 < scheduler_decay_lr <= optimizer_lr")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not self.image_resize_shapes:
            raise ValueError("image_resize_shapes must contain at least one camera")
        for key, (height, width) in self.image_resize_shapes.items():
            if height <= 0 or width <= 0:
                raise ValueError(f"Image size for {key} must be positive, got {(height, width)}")
            if height % self.vision_patch_size != 0 or width % self.vision_patch_size != 0:
                raise ValueError(
                    f"Image size for {key} must be divisible by vision_patch_size={self.vision_patch_size}, "
                    f"got {(height, width)}"
                )

        # Feature shapes are filled by ``make_policy`` after the config is
        # constructed.  An empty feature mapping is therefore valid here.
        if not self.input_features or not self.output_features:
            return
        state = self.robot_state_feature
        action = self.action_feature
        if state is None or state.shape[0] < self.state_dim:
            raise ValueError(f"DinoFlow requires at least {self.state_dim} state values, got {state}")
        if action is None or action.shape[0] < self.action_dim:
            raise ValueError(f"DinoFlow requires at least {self.action_dim} action values, got {action}")
        missing = [key for key in self.image_resize_shapes if key not in self.image_features]
        if missing:
            raise ValueError(f"DinoFlow missing image features: {missing}")
        if self.integration_method not in {"euler", "heun"}:
            raise ValueError("integration_method must be 'euler' or 'heun'")
        if self.clip_sample_range <= 0:
            raise ValueError("clip_sample_range must be positive")

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DinoFlowCosineDecayWithWarmupSchedulerConfig:
        return DinoFlowCosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_num_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )


@LRSchedulerConfig.register_subclass("dino_flow_cosine_decay_with_warmup")
@dataclass
class DinoFlowCosineDecayWithWarmupSchedulerConfig(CosineDecayWithWarmupSchedulerConfig):
    """DinoFlow-local cosine scheduler with optional decay-step auto-match.

    Leaving num_decay_steps unset means "decay across this run's training steps";
    build() is the first point where num_training_steps is known. The decay floors
    at decay_lr (non-zero) instead of the diffusers "cosine" schedule which goes to 0.
    """

    num_decay_steps: int | None

    def build(self, optimizer, num_training_steps: int):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.peak_lr,
            decay_lr=self.decay_lr,
            num_warmup_steps=self.num_warmup_steps,
            num_decay_steps=num_training_steps if self.num_decay_steps is None else self.num_decay_steps,
        ).build(optimizer, num_training_steps=num_training_steps)
