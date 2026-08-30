#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

from __future__ import annotations

import builtins
import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    FUTURE_OBS_STATE,
    NEXT_OBS_STATE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PaliGemmaWithExpertModel,
    compute_layer_complete,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    get_safe_dtype,
    make_att_2d_masks,
    resize_with_pad_torch,
    sample_beta,
)
from ..pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE
from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_bbvla import (
    BbvlaConfig,
    FINGER_LAYOUTS,
    TAXELS_PER_HAND,
)


# ── Tactile Conv2d Encoder ──────────────────────────────────────────────────

class TactileConvEncoder(nn.Module):
    """Per-finger shared Conv2d encoder with cross-attention aggregation.

    Inspired by HTD (Touch Dreaming): lightweight CNNs per finger, then
    cross-attention to aggregate finger features into per-hand belief tokens.

    The per-hand belief latents are used as EMA targets for tactile prediction —
    the online encoder receives gradient through the action-flow suffix, and its
    EMA copy provides stable targets (JEPA-style, identical to HTD).

    Left and right hands share the same 6 Conv2d weights.
    Per finger: Conv2d → GlobalAvgPool → 8D → project to d_model.
    Cross-attention: learnable query tokens attend to 12 finger features (6×2 hands).
    Per-hand latent: shared latent_proj applied to each hand token → (B, 2, latent_dim).

    Each finger encodes two complementary modalities:
    - per-finger normalized spatial pattern (via Conv2d)
    - per-finger pressure magnitude statistics: mean / std / max (via MLP)
    The two feature vectors are summed elementwise before projection.
    """

    def __init__(self, finger_layouts, channels=8, d_model=1024, num_tokens=2,
                 num_heads=4, latent_dim=64):
        super().__init__()
        self.finger_layouts = finger_layouts
        self.num_tokens = num_tokens
        self.d_model = d_model

        self.finger_convs = nn.ModuleDict()
        for fl in finger_layouts:
            self.finger_convs[fl["name"]] = nn.Conv2d(1, channels, kernel_size=3, padding=1)

        # Per-finger pressure magnitude MLP: mean/std/max → channels
        self.finger_mag_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

        # Per-finger feature projection: 8D → d_model
        self.finger_proj = nn.Linear(channels, d_model)

        # Separate query tokens for left/right hands; shared cross-attn weights
        self.left_query  = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.right_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Per-hand latent projection: shared across left/right (EMA target for touch dreaming)
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, latent_dim),
        )

    def forward(self, tactile):
        """tactile: (B, 604) → (B, 2, latent_dim) per-hand belief latent."""
        tokens = self._encode(tactile)                             # (B, 2, d_model)
        latent = self.latent_proj(tokens)                          # (B, 2, latent_dim)
        return latent

    def encode_latent(self, tactile):
        """Only produce the latent (EMA target, no gradient path needed)."""
        with torch.no_grad():
            latent = self.forward(tactile)
        return latent

    def _encode(self, tactile):
        """tactile: (B, 604) → (B, num_tokens, d_model).

        Per-finger normalization before Conv2d: each finger's taxels are
        normalized to zero mean / unit std per sample (InstanceNorm-style).
        This is critical because ~37% of tactile taxels are near-constant
        (always ~0), so per-dimension QUANTILES normalization would otherwise
        produce extreme values of 10^13+ that corrupt the Conv2d features.
        Normalizing per finger instead of per taxel is robust to sparse
        tactile data — a finger with no activation produces ≈ 0, while a
        finger with partial activation preserves its relative spatial pattern.
        """
        left = tactile[:, :TAXELS_PER_HAND]
        right = tactile[:, TAXELS_PER_HAND:]

        # Collect all finger features: 12 × (B, channels)
        all_finger_feats = []
        for hand_tactile in (left, right):
            offset = 0
            for fl in self.finger_layouts:
                n = fl["taxels"]
                rows, cols = fl["rows"], fl["cols"]

                # Raw finger tactile
                x = hand_tactile[:, offset : offset + n]                    # (B, n)

                # Per-finger pressure magnitude statistics (before z-score)
                mag = torch.cat([
                    x.mean(dim=-1, keepdim=True),                           # (B, 1)
                    x.std(dim=-1, keepdim=True),                            # (B, 1)
                    x.amax(dim=-1, keepdim=True),                           # (B, 1)
                ], dim=-1)                                                  # (B, 3)
                mag_feat = self.finger_mag_mlp(mag)                         # (B, channels)

                # Per-finger normalization (InstanceNorm-style over taxels)
                finger_mean = x.mean(dim=-1, keepdim=True)                  # (B, 1)
                finger_std = x.std(dim=-1, keepdim=True) + 1e-5             # (B, 1)
                x = (x - finger_mean) / finger_std                          # (B, n)

                x = x.reshape(-1, 1, rows, cols)
                x = self.finger_convs[fl["name"]](x)
                x = x.mean(dim=[-2, -1])                                    # (B, channels)
                x = x + mag_feat                                            # pattern + magnitude
                all_finger_feats.append(x)
                offset += n

        # Stack: (B, 12, channels) → project → (B, 12, d_model)
        finger_stack = torch.stack(all_finger_feats, dim=1)      # (B, 12, channels)
        finger_emb = self.finger_proj(finger_stack)               # (B, 12, d_model)

        # Left/right cross-attention: each hand's query attends to its own 6 fingers
        b = finger_emb.shape[0]
        left_emb  = finger_emb[:, :6, :]    # (B, 6, d_model)
        right_emb = finger_emb[:, 6:, :]    # (B, 6, d_model)

        left_q  = self.left_query.unsqueeze(0).expand(b, -1, -1)   # (B, 1, d_model)
        right_q = self.right_query.unsqueeze(0).expand(b, -1, -1)  # (B, 1, d_model)

        left_out,  _ = self.cross_attn(left_q,  left_emb,  left_emb)
        right_out, _ = self.cross_attn(right_q, right_emb, right_emb)

        left_out  = self.attn_norm(left_out  + left_q)   # (B, 1, d_model)
        right_out = self.attn_norm(right_out + right_q)  # (B, 1, d_model)
        return torch.cat([left_out, right_out], dim=1)   # (B, 2, d_model)


# ── Core Model ───────────────────────────────────────────────────────────────

class BbvlaPytorch(nn.Module):
    """Core Bbvla model — PI05 with per-hand belief latent + force suffix injection."""

    def __init__(self, config: BbvlaConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        # Architecture invariants: TactileConvEncoder produces 2 per-hand belief tokens
        # (left_query + right_query → 2 hand latents) and embed_suffix projects them as (B,2,D).
        # Force also emits exactly 2 tokens (left + right wrist).
        assert config.num_tactile_suffix_tokens == 2, (
            "TactileConvEncoder produces exactly 2 tokens (left+right); num_tactile_suffix_tokens must be 2."
        )
        assert config.num_force_suffix_tokens == 2, (
            "embed_suffix emits exactly 2 force tokens (left+right); num_force_suffix_tokens must be 2."
        )

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError("PaliGemma expects square image resolution")

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        expert_width = action_expert_config.width  # 1024 for gemma_300m

        # PI05-original projections
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)

        # Bbvla: force projection — left and right wrist as separate suffix tokens
        self.left_force_proj  = nn.Linear(config.max_force_dim // 2, expert_width)  # 6 → D
        self.right_force_proj = nn.Linear(config.max_force_dim // 2, expert_width)  # 6 → D

        # Bbvla: tactile Conv2d encoder → per-hand belief latents (B,2,64);
        # latent_proj receives action-flow gradients via suffix belief tokens; EMA tracks it for touch dreaming
        self.tactile_encoder = TactileConvEncoder(
            finger_layouts=config.finger_layouts,
            channels=config.tactile_encoder_channels,
            d_model=expert_width,
            num_tokens=config.num_tactile_suffix_tokens,
            latent_dim=config.tactile_latent_dim,
        )

        # Touch dreaming heads (HTD-style: predict force + tactile latent from action chunk hidden mean)
        self.force_pred_head = nn.Linear(expert_width, config.max_force_dim)
        self.tactile_pred_head = nn.Linear(expert_width, 2 * config.tactile_latent_dim)

        # Tactile latent → suffix token so latent_proj receives action-flow gradients;
        # EMA teacher then tracks a trained latent projection.
        self.tactile_latent_token_proj = nn.Linear(config.tactile_latent_dim, expert_width)

        # Dream action projection: fuse x0_hat (clean action estimate) into dream hidden
        self.dream_action_proj = nn.Linear(config.max_action_dim, expert_width)

        # EMA target encoder = deepcopy of TactileConvEncoder (HTD JEPA-style).
        # tactile_encoder gets gradient via flow-matching (its tokens enter the suffix).
        # touch_ema_encoder is its slow EMA copy, providing stable latent targets.
        self.touch_ema_encoder = copy.deepcopy(self.tactile_encoder)
        for p in self.touch_ema_encoder.parameters():
            p.requires_grad = False

        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    # ── gradient checkpointing ──

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    # ── helpers ──

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)
        return func(*args)

    @staticmethod
    def sample_noise(shape, device):
        return torch.randn(shape, device=device)

    @staticmethod
    def sample_time(bsize, device):
        time = sample_beta(1.5, 1.0, bsize, device)
        return time * 0.999 + 0.001

    # ── EMA update ──

    @torch.no_grad()
    def ema_update_touch_encoder(self):
        """EMA: tactile_encoder (online, gradient via flow-matching) → touch_ema_encoder (target)."""
        for p_ema, p in zip(self.touch_ema_encoder.parameters(), self.tactile_encoder.parameters()):
            p_ema.data = self.config.ema_decay * p_ema.data + (1 - self.config.ema_decay) * p.data

    # ── attention mask helper ──

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Same as PI05: (B, N, N) → (B, 1, N, N) float mask."""
        from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
        att_2d_masks_4d = att_2d_masks[:, None, :, :]  # add head dim
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    # ── embedding ──

    def embed_prefix(self, images, img_masks, tokens, masks):
        """Identical to PI05: embed images + language tokens for the PaliGemma prefix."""
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, tactile, force, noisy_actions, timestep):
        """Embed tactile, force, noisy_actions, timestep for the Action Expert suffix.

        Suffix structure: [L_tactile_belief(1)] [R_tactile_belief(1)] [left_force(1)] [right_force(1)] [actions(C)]
        All suffix tokens share bidirectional attention among themselves.
        Returns (embs, pad_masks, att_masks, adarms_cond, tactile_latent).
        """
        embs, pad_masks, att_masks = [], [], []

        # Tactile belief tokens: per-hand 64D latent → (B, 2, D) suffix tokens
        tactile_latent = self.tactile_encoder(tactile)                 # (B, 2, latent_dim)
        bsize = tactile_latent.shape[0]
        device = tactile_latent.device
        tactile_belief_emb = self.tactile_latent_token_proj(tactile_latent)  # (B, 2, D)
        embs.append(tactile_belief_emb)
        pad_masks.append(torch.ones(bsize, 2, dtype=torch.bool, device=device))
        att_masks += [1, 0]  # first belief token as boundary

        # Force: left and right wrist as separate suffix tokens
        left_force_emb  = self.left_force_proj(force[:, :6])[:, None, :]   # (B, 1, D)
        right_force_emb = self.right_force_proj(force[:, 6:])[:, None, :]  # (B, 1, D)
        embs.append(torch.cat([left_force_emb, right_force_emb], dim=1))   # (B, 2, D)
        pad_masks.append(torch.ones(bsize, 2, dtype=torch.bool, device=device))
        att_masks += [0, 0]  # continuous sensor tokens

        # Timestep embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=self.config.min_period, max_period=self.config.max_period,
            device=device,
        ).type(dtype=timestep.dtype)

        # Action embedding
        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)
        time_emb = self._apply_checkpoint(
            lambda te: F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(te)))),
            time_emb,
        )
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device))
        # First action token as causal boundary, rest as causal
        att_masks += [1] + [0] * (self.config.chunk_size - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond, tactile_latent

    # ── forward / denoise ──

    def forward(self, images, img_masks, tokens, masks, actions,
                tactile, force, noise, time,
                next_tactile=None, next_force=None,
                future_tactile=None, future_force=None):
        """Training forward pass: flow matching + optional touch dreaming loss.

        Parameters
        ----------
        tactile, force : Tensor
            Current observationʼs tactile (B,604) and force (B,12) — used as
            suffix tokens so the action expert can condition on current sensor state.
        next_tactile, next_force : Tensor | None
            **Next** observationʼs tactile and force (tau=1). Used as fallback
            when multi-step future data is not available.
        future_tactile, future_force : Tensor | None
            Multi-step future tactile (B,K,604) and force (B,K,12). When provided,
            the model predicts force/tactile at each of the K future offsets.
        """
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, tactile_latent = self.embed_suffix(
            tactile, force, x_t, time)

        if (self.paligemma_with_expert.paligemma.model.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        flow_losses = F.mse_loss(u_t, v_t, reduction="none")

        # Touch dreaming loss (HTD-style: predict future force/tactile from action hidden)
        # `force` / `tactile` are current observation → suffix embedding (conditioning).
        # `future_force` / `future_tactile` are multi-step targets (K frames ahead).
        # `next_force` / `next_tactile` are tau=1 fallback targets.
        touch_losses = {}
        if self.config.touch_dreaming_enabled:
            # VICReg variance loss: prevent tactile representation collapse
            # unbiased=False avoids NaN when batch_size=1 (default unbiased divides by N-1=0)
            if tactile_latent.shape[0] > 1:
                variance_loss = F.relu(1.0 - tactile_latent.reshape(-1, tactile_latent.shape[-1]).std(dim=0, unbiased=False)).mean()
            else:
                variance_loss = tactile_latent.new_zeros(())
            touch_losses["variance_loss"] = variance_loss

            action_hidden = suffix_out[:, -self.config.chunk_size:, :]  # (B, chunk_size, D)

            # Compute clean action estimate x0_hat and fuse into dream hidden
            x0_hat = x_t - time[:, None, None] * v_t
            dream_hidden = action_hidden + self.dream_action_proj(x0_hat)

            # ── Decide mode: multi-step > tau=1 > fallback to current ──
            use_multi_step = future_tactile is not None and future_force is not None

            if use_multi_step:
                # ── Multi-step: predict at K specific action positions ──
                offsets = self.config.get_touch_dreaming_offsets()
                K = len(offsets)

                # Strict shape checks: config offsets must match dataset horizon
                if future_tactile.shape[1] != K:
                    raise ValueError(
                        f"future_tactile horizon ({future_tactile.shape[1]}) != "
                        f"touch_dreaming_offsets count ({K}). "
                        f"Offsets: {offsets}. Check that touch_dreaming_horizon / offsets "
                        f"match the delta_indices used to fetch future.observation.state."
                    )
                if future_force.shape[1] != K:
                    raise ValueError(
                        f"future_force horizon ({future_force.shape[1]}) != "
                        f"touch_dreaming_offsets count ({K}). "
                        f"Offsets: {offsets}."
                    )

                # offset=1 corresponds to dream_hidden[0], the action leading to the next observation.
                action_indices = [off - 1 for off in offsets]  # zero-based
                selected_hidden = dream_hidden[:, action_indices, :]  # (B, K, D)

                # Force prediction — per-step, matching Eq.7 (τ future moments)
                force_pred = self.force_pred_head(selected_hidden)                # (B, K, 12)
                force_target = future_force                                        # (B, K, 12)
                force_loss = F.smooth_l1_loss(force_pred, force_target, reduction="mean")
                touch_losses["force_pred_loss"] = force_loss

                # Encode all K future tactile frames through EMA encoder
                B = future_tactile.shape[0]
                flat_tactile = future_tactile.reshape(B * K, -1)          # (B*K, 604)
                flat_latent = self.touch_ema_encoder.encode_latent(flat_tactile)  # (B*K, 2, latent_dim)
                tactile_target = flat_latent.reshape(B, K, 2, -1)         # (B, K, 2, latent_dim)
                tactile_pred = self.tactile_pred_head(selected_hidden).reshape(B, K, 2, -1)  # (B, K, 2, latent_dim)

                if self.config.touch_dreaming_use_cosine_loss:
                    tp = tactile_pred.reshape(B * K * 2, -1)              # (B*K*2, latent_dim)
                    tt = tactile_target.reshape(B * K * 2, -1)            # (B*K*2, latent_dim)
                    direction_loss = (1 - F.cosine_similarity(tp, tt, dim=-1)).mean()
                    magnitude_loss = F.smooth_l1_loss(
                        tp.norm(dim=-1), tt.norm(dim=-1), reduction="mean",
                    )
                    tactile_loss = direction_loss + self.config.tactile_latent_loss_beta * magnitude_loss
                else:
                    tactile_loss = F.mse_loss(tactile_pred, tactile_target, reduction="mean")
            else:
                # ── Tau=1: single-step with configurable pooling ──
                pooling = self.config.touch_dreaming_pooling
                if pooling == "mean":
                    pooled_hidden = dream_hidden.mean(dim=1)
                elif pooling == "last":
                    pooled_hidden = dream_hidden[:, -1, :]
                elif pooling == "mid":
                    pooled_hidden = dream_hidden[:, dream_hidden.shape[1] // 2, :]
                else:
                    raise ValueError(
                        f"Unknown touch_dreaming_pooling='{pooling}'. "
                        f"Expected one of: mean, last, mid."
                    )

                if next_force is not None:
                    force_pred = self.force_pred_head(pooled_hidden)       # (B, 12)
                    force_target = next_force                                     # predict absolute
                    touch_losses["force_pred_loss"] = F.smooth_l1_loss(force_pred, force_target, reduction="mean")

                tactile_pred = self.tactile_pred_head(pooled_hidden).reshape(-1, 2, self.config.tactile_latent_dim)  # (B, 2, latent_dim)
                td_tactile = next_tactile if next_tactile is not None else tactile
                tactile_target = self.touch_ema_encoder.encode_latent(td_tactile)  # (B, 2, latent_dim)
                Bs = pooled_hidden.shape[0]

                if self.config.touch_dreaming_use_cosine_loss:
                    tp = tactile_pred.reshape(Bs * 2, -1)                # (B*2, latent_dim)
                    tt = tactile_target.reshape(Bs * 2, -1)              # (B*2, latent_dim)
                    direction_loss = (1 - F.cosine_similarity(tp, tt, dim=-1)).mean()
                    magnitude_loss = F.smooth_l1_loss(
                        tp.norm(dim=-1), tt.norm(dim=-1), reduction="mean",
                    )
                    tactile_loss = direction_loss + self.config.tactile_latent_loss_beta * magnitude_loss
                else:
                    tactile_loss = F.mse_loss(tactile_pred, tactile_target, reduction="mean")

            touch_losses["tactile_pred_loss"] = tactile_loss

        return flow_losses, touch_losses

    # ── inference ──

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, tactile, force,
                       noise=None, num_steps=None,
                       **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        lm_cfg = self.paligemma_with_expert.paligemma.model.language_model.config
        _saved_attn = lm_cfg._attn_implementation
        lm_cfg._attn_implementation = "eager"
        try:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            dt = -1.0 / num_steps
            x_t = noise
            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
                v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, tactile, force, time_tensor)
                x_t = x_t + dt * v_t
            return x_t
        finally:
            lm_cfg._attn_implementation = _saved_attn

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, tactile, force, timestep):
        """Single denoising step. Runs suffix (with tactile/force) through expert."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, _ = self.embed_suffix(
            tactile, force, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        suffix_out = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )[0][1]

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


# ── Policy ───────────────────────────────────────────────────────────────────

class BbvlaPolicy(PreTrainedPolicy):
    """Bbvla Policy — PI05 with per-hand belief latent + force token suffix + Touch Dreaming."""

    config_class = BbvlaConfig
    name = "bbvla"

    def __init__(self, config: BbvlaConfig, **kwargs):
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = BbvlaPytorch(config, rtc_processor=None)  # RTC set in init_rtc_processor
        self.init_rtc_processor()

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,  # default False: VLM keys load, bbvla-new keys skipped
        **kwargs,
    ) -> T:
        """Load pretrained VLM from PI05 checkpoint; bbvla-specific modules are random init."""
        print(
            "Bbvla: Loading PI05 VLM backbone, tactile/force encoders initialized randomly.\n"
            f"  Pretrained path: {pretrained_name_or_path}"
        )

        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)

        model = cls(config, **kwargs)

        from transformers.utils import cached_file
        resolved = cached_file(pretrained_name_or_path, "model.safetensors",
                               cache_dir=kwargs.get("cache_dir"),
                               force_download=kwargs.get("force_download", False),
                               resume_download=kwargs.get("resume_download"),
                               proxies=kwargs.get("proxies"),
                               token=kwargs.get("token"),
                               revision=kwargs.get("revision"),
                               local_files_only=kwargs.get("local_files_only", False))
        from safetensors.torch import load_file
        state_dict = load_file(resolved)

        # Fix key differences from OpenPI format
        fixed = model._fix_pytorch_state_dict_keys(state_dict, model.config)

        # Remap: add "model." prefix (PI05 stores with this prefix)
        remapped = {}
        for k, v in fixed.items():
            if not k.startswith("model."):
                remapped[f"model.{k}"] = v
            else:
                remapped[k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=strict)
        if missing:
            print(f"  Missing keys (bbvla new modules, expected): {len(missing)}")
            for k in sorted(missing)[:8]:
                print(f"    - {k}")
            if len(missing) > 8:
                print(f"    ... and {len(missing) - 8} more")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        if not missing and not unexpected:
            print("  All keys loaded (perfect match)")

        return model

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        """Fix state dict keys to match model architecture. Same logic as PI05."""
        import re

        fixed = {}
        for key, value in state_dict.items():
            new_key = key

            # Skip state_proj (PI0 artifact, not used in bbvla)
            if key.startswith("state_proj."):
                continue

            # Handle layer norm structure changes for adaRMS
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    continue

            # Remap PI0 MLP names to pi05 style
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")

            # lm_head.weight → embed_tokens.weight (tied embeddings)
            if key == "model.paligemma_with_expert.paligemma.lm_head.weight" or \
               key == "paligemma_with_expert.paligemma.lm_head.weight":
                fixed["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = value.clone()

            fixed[new_key] = value

        return fixed

    # ── RTC ──

    def init_rtc_processor(self):
        """Initialize RTC processor if configured."""
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            if hasattr(self, "model") and self.model is not None:
                self.model.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # ── standard policy interface ──

    def reset(self):
        self._action_queue = []

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model. Same logic as PI05."""
        images = []
        img_masks = []
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. "
                f"(batch keys: {list(batch.keys())}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]
            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)  # BCHW → BHWC

            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            img = img * 2.0 - 1.0  # [0,1] → [-1,1]

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # BHWC → BCHW

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Fill missing cameras with zero images
        for _ in missing_img_keys:
            empty_img = torch.ones_like(img) * -1  # padded with -1 for SigLIP
            empty_mask = torch.zeros_like(mask)
            images.append(empty_img)
            img_masks.append(empty_mask)

        return images, img_masks

    def prepare_action(self, batch):
        actions = batch[ACTION]
        if actions.ndim == 4 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        if actions.shape[-1] < self.config.max_action_dim:
            pad = torch.zeros(
                *actions.shape[:-1], self.config.max_action_dim - actions.shape[-1],
                device=actions.device, dtype=actions.dtype,
            )
            actions = torch.cat([actions, pad], dim=-1)
        return actions

    def _extract_tactile_force(self, batch, state_key=OBS_STATE):
        """Extract tactile and force from a state tensor in batch.

        Parameters
        ----------
        batch : dict
            Dataset batch dict.
        state_key : str
            Key to look up in batch (e.g. OBS_STATE for current observation,
            NEXT_OBS_STATE for next observation, FUTURE_OBS_STATE for multi-step).

        Returns
        -------
        tactile : Tensor, shape (B, 604) or (B, K, 604) for multi-step
        force : Tensor, shape (B, 12) or (B, K, 12) for multi-step
        """
        state = batch.get(state_key)
        if state is None:
            raise ValueError(f"Bbvla requires {state_key} in batch")

        # Handle extra batch dimension from AddBatchDimensionProcessorStep.
        # ndim=3 for single-frame (1, B, 646), ndim=4 for multi-frame (1, B, K, 646).
        if state.ndim >= 3 and state.shape[0] == 1:
            state = state.squeeze(0)

        # Indices from configuration
        from .configuration_bbvla import (
            STATE_FORCE_START, STATE_FORCE_END,
            STATE_TACTILE_LEFT_START, STATE_TACTILE_LEFT_END,
            STATE_TACTILE_RIGHT_START, STATE_TACTILE_RIGHT_END,
        )
        force = state[..., STATE_FORCE_START:STATE_FORCE_END]
        tac_left = state[..., STATE_TACTILE_LEFT_START:STATE_TACTILE_LEFT_END]
        tac_right = state[..., STATE_TACTILE_RIGHT_START:STATE_TACTILE_RIGHT_END]
        tactile = torch.cat([tac_left, tac_right], dim=-1)  # (B, 604) or (B, K, 604)
        return tactile, force

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.prepare_action(batch)
        tactile, force = self._extract_tactile_force(batch, OBS_STATE)

        # Extract future-state force/tactile for multi-step touch dreaming.
        # future.observation.state is dynamically fetched via delta_indices,
        # shaped (B, K, 646) where K = touch_dreaming_horizon.
        future_tactile, future_force = None, None
        if FUTURE_OBS_STATE in batch:
            future_tactile, future_force = self._extract_tactile_force(batch, FUTURE_OBS_STATE)

        # Extract NEXT-state force/tactile as tau=1 fallback targets.
        next_tactile, next_force = None, None
        if NEXT_OBS_STATE in batch:
            next_tactile, next_force = self._extract_tactile_force(batch, NEXT_OBS_STATE)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        flow_losses, touch_losses = self.model.forward(
            images, img_masks, tokens, masks, actions, tactile, force, noise, time,
            next_tactile=next_tactile, next_force=next_force,
            future_tactile=future_tactile, future_force=future_force)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :original_action_dim]
        loss = flow_losses.mean()

        loss_dict = {"loss": None, "flow_loss": loss.item()}

        if self.config.touch_dreaming_enabled and touch_losses:
            force_loss = touch_losses.get("force_pred_loss")
            tactile_loss = touch_losses.get("tactile_pred_loss")
            variance_loss = touch_losses.get("variance_loss")
            if force_loss is not None:
                loss = loss + self.config.force_pred_weight * force_loss
                loss_dict["force_pred_loss"] = force_loss.item()
            if tactile_loss is not None:
                loss = loss + self.config.tactile_pred_weight * tactile_loss
                loss_dict["tactile_pred_loss"] = tactile_loss.item()
            if variance_loss is not None:
                loss = loss + self.config.variance_weight * variance_loss
                loss_dict["variance_loss"] = variance_loss.item()
        loss_dict["loss"] = loss.item()

        if reduction == "none":
            per_sample = flow_losses.mean(dim=(1, 2))
            return per_sample, loss_dict

        return loss, loss_dict

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        tactile, force = self._extract_tactile_force(batch)

        actions = self.model.sample_actions(
            images, img_masks, tokens, masks, tactile, force, **kwargs)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            for i in range(self.config.n_action_steps):
                self._action_queue.append(chunk[:, i, :])
        return self._action_queue.pop(0)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def update(self):
        """Called by the training loop after optimizer.step().

        Updates the EMA touch encoder so it tracks the post-optimizer online encoder
        parameters (HTD JEPA-style), rather than the pre-optimizer snapshot.
        """
        if self.config.touch_dreaming_enabled and self.training:
            self.model.ema_update_touch_encoder()

    def _get_default_peft_targets(self) -> dict[str, any]:
        """LoRA on SigLIP + PaliGemma 2B, full training on Action Expert + bbvla modules."""
        return {
            "target_modules": (
                r".*vision_tower.*self_attn\.(q_proj|k_proj|v_proj|out_proj)"
                r"|.*language_model.*self_attn\.(q_proj|v_proj|out_proj)"
            ),
            "modules_to_save": [
                "multi_modal_projector",
                "gemma_expert",
                "left_force_proj",
                "right_force_proj",
                "force_pred_head",
                "tactile_encoder",
                "touch_ema_encoder",  # EMA target — not gradient-trained but must be checkpointed for resume
                "tactile_pred_head",
                "tactile_latent_token_proj",
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
                "dream_action_proj",
            ],
        }
