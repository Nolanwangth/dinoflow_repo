#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

from dataclasses import dataclass, field

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, FUTURE_OBS_STATE, NEXT_OBS_STATE, OBS_STATE

from ..pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config

# Per-finger tactile taxel layout: finger name → (rows, cols, taxel_count)
# Dataset stores taxels in a 1D flat array ordered by these indices.
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

# Text annotation names for prompt formatting
# Hand joint order matches hardware: thumb_rotate, index, middle, ring, pinky, thumb_bend
HAND_JOINT_NAMES_TEXT: list[str] = ["thumb_rotate", "index", "middle", "ring", "pinky", "thumb_bend"]
# Tactile finger order matches FINGER_LAYOUTS
TACTILE_FINGER_NAMES_TEXT: list[str] = ["thumb", "index", "middle", "ring", "pinky", "thumb_bend"]

# State index ranges within the 646-dim vector
STATE_JOINTS_START, STATE_JOINTS_END = 0, 30       # arm(14) + hands(12) + head(2) + waist(2)
STATE_FORCE_START, STATE_FORCE_END = 30, 42         # wrist force left(6) + right(6)
STATE_TACTILE_LEFT_START, STATE_TACTILE_LEFT_END = 42, 42 + TAXELS_PER_HAND     # 302
STATE_TACTILE_RIGHT_START, STATE_TACTILE_RIGHT_END = (42 + TAXELS_PER_HAND, 42 + TAXELS_BOTH_HANDS)  # 302


@PreTrainedConfig.register_subclass("bbvla")
@dataclass
class BbvlaConfig(PI05Config):
    """Configuration for Bbvla policy — PI05 with per-hand belief latent + force token suffix."""

    # Override PI05 defaults for the 646-dim dataset
    max_state_dim: int = 646
    max_action_dim: int = 32

    # Force modality
    max_force_dim: int = 12

    # Tactile modality
    max_tactile_dim: int = TAXELS_BOTH_HANDS  # 604
    tactile_encoder_channels: int = 8          # Conv2d output channels per finger
    num_tactile_suffix_tokens: int = 2         # left/right belief tokens in suffix
    num_force_suffix_tokens: int = 2           # left + right wrist tokens in suffix

    # Touch dreaming
    touch_dreaming_enabled: bool = True
    force_pred_weight: float = 0.1
    tactile_pred_weight: float = 0.1
    variance_weight: float = 0.1
    tactile_latent_dim: int = 64
    ema_decay: float = 0.999
    normalized_value_clip: float = 10.0

    # Tactile latent loss config
    tactile_latent_loss_beta: float = 0.1
    touch_dreaming_use_cosine_loss: bool = True
    touch_dreaming_pooling: str = "mean"  # options: mean, last, mid

    # Multi-step touch dreaming: predict future force/tactile at K horizon offsets
    touch_dreaming_horizon: int = 10
    touch_dreaming_min_offset: int = 1
    touch_dreaming_max_offset: int | None = None  # None → chunk_size - 1
    touch_dreaming_offsets: list[int] | None = None  # explicit offsets override linspace

    # Finger layouts for tactile reshaping (not serialized, set at init)
    finger_layouts: list[dict] = field(default_factory=lambda: FINGER_LAYOUTS)

    # Increase token budget to accommodate extra force/touch text in prompt
    tokenizer_max_length: int = 400

    def get_touch_dreaming_offsets(self) -> list[int]:
        """Return the list of frame offsets for touch dreaming predictions.

        If ``touch_dreaming_offsets`` is set explicitly, use it (sorted).
        Otherwise, evenly space ``touch_dreaming_horizon`` offsets between
        ``touch_dreaming_min_offset`` and ``touch_dreaming_max_offset``
        (defaulting to ``chunk_size - 1``).

        All offsets are clamped to [1, chunk_size - 1], deduplicated, and sorted.
        Raises ValueError if the resulting list is empty.
        """
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

        # Clamp, deduplicate, sort
        offsets: list[int] = sorted(set(max(1, min(int(o), max_off)) for o in raw))
        if not offsets:
            raise ValueError(
                f"touch_dreaming_offsets is empty after clamping to [1, {max_off}]. "
                f"Check touch_dreaming_min_offset / max_offset / horizon."
            )
        return offsets

    @property
    def future_observation_delta_indices(self) -> list[int] | None:
        """Delta indices for fetching future observation.state from the dataset.

        Used by ``resolve_delta_timestamps`` to build ``delta_timestamps``
        so the dataset dynamically fetches observation.state at future frame
        offsets (without pre-computed columns).
        """
        if not self.touch_dreaming_enabled:
            return None
        return self.get_touch_dreaming_offsets()

    def __post_init__(self):
        # Let PI05Config.__post_init__ run first
        super().__post_init__()

    def validate_features(self) -> None:
        super().validate_features()
        # Ensure max_state_dim accommodates all sub-modalities
        expected_total = STATE_TACTILE_RIGHT_END  # 646
        if self.max_state_dim < expected_total:
            raise ValueError(
                f"max_state_dim ({self.max_state_dim}) must be >= {expected_total} "
                f"to hold joints(30) + force(12) + tactile(604)"
            )
        if OBS_STATE in self.input_features:
            self.input_features[OBS_STATE].shape = (self.max_state_dim,)
        # Register next.observation.state so the normalizer processes it too
        # (same shape/stats as observation.state — just shifted by 1 frame)
        if NEXT_OBS_STATE not in self.input_features:
            self.input_features[NEXT_OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.max_state_dim,)
            )
        # Register future.observation.state for multi-step touch dreaming.
        # Shape is (K, max_state_dim) where K = number of touch dreaming offsets.
        # The dataset dynamically fetches K future frames at those offsets.
        if FUTURE_OBS_STATE not in self.input_features:
            k = len(self.get_touch_dreaming_offsets())
            self.input_features[FUTURE_OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(k, self.max_state_dim,)
            )
        if ACTION in self.output_features:
            self.output_features[ACTION].shape = (self.max_action_dim,)
