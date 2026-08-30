#!/usr/bin/env python

"""ACT-HTD: Action Chunking Transformer with Touch Dreaming.

Implements the HTD architecture (arXiv 2604.13015) as a lerobot policy.
Deterministic BC with modular modality tokenizers, per-group action experts,
and touch-dreaming auxiliary objectives (force prediction + tactile latent prediction
with EMA teacher).

Architecture (Fig.4):
    [3× Camera] → shared ResNet18 + cross-attn pooling → N_img tokens each
    [Proprio 26D] → Linear → 1 token
    [Force 12D] → L/R Linear → 2 tokens
    [Tactile 604D] → Per-finger Conv2d + cross-attn → 2 tokens (L/R latent)
       ↓ all tokens concatenated + modality/position embeddings
    Transformer Encoder → Transformer Decoder
       ↓ shared decoder hidden
    Modular Action Experts (3 groups) → (B, chunk_size, 26)
"""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import (
    ACTION,
    FUTURE_OBS_STATE,
    NEXT_OBS_STATE,
    OBS_IMAGES,
    OBS_STATE,
)

from ..pretrained import PreTrainedPolicy
from .configuration_act_htd import (
    ACTHTDConfig,
    ACTION_GROUPS,
    FINGER_LAYOUTS,
    STATE_FORCE_END,
    STATE_FORCE_START,
    STATE_JOINTS_END,
    STATE_JOINTS_START,
    STATE_TACTILE_LEFT_END,
    STATE_TACTILE_LEFT_START,
    STATE_TACTILE_RIGHT_END,
    STATE_TACTILE_RIGHT_START,
    TAXELS_PER_HAND,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions (from modeling_act.py)
# ═══════════════════════════════════════════════════════════════════════════════


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings for image feature maps."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tactile Encoder (from bbvla TactileConvEncoder, self-contained copy)
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTDTactileEncoder(nn.Module):
    """Per-finger shared Conv2d encoder with cross-attention aggregation.

    Left and right hands share the same 6 Conv2d weights.
    Per finger: Conv2d → GlobalAvgPool → project to d_model.
    Cross-attention: learnable L/R query tokens attend to 12 finger features.
    Per-hand latent: shared latent_proj applied to each hand token → (B, 2, latent_dim).
    """

    def __init__(self, finger_layouts, channels=8, d_model=512, num_tokens=2,
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

        self.finger_proj = nn.Linear(channels, d_model)

        self.left_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.right_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, latent_dim),
        )

    def forward(self, tactile):
        """tactile: (B, 604) → (B, 2, latent_dim) per-hand belief latent."""
        tokens = self._encode(tactile)
        latent = self.latent_proj(tokens)
        return latent

    def encode_latent(self, tactile):
        """Produce latent without gradients (EMA target)."""
        with torch.no_grad():
            latent = self.forward(tactile)
        return latent

    def _encode(self, tactile):
        """tactile: (B, 604) → (B, num_tokens, d_model).

        Per-finger normalization: each finger's taxels normalized to zero mean/unit std
        per sample, then Conv2d + magnitude stats, then cross-attention aggregation.
        """
        left = tactile[:, :TAXELS_PER_HAND]
        right = tactile[:, TAXELS_PER_HAND:]

        all_finger_feats = []
        for hand_tactile in (left, right):
            offset = 0
            for fl in self.finger_layouts:
                n = fl["taxels"]
                rows, cols = fl["rows"], fl["cols"]

                x = hand_tactile[:, offset: offset + n]

                # Per-finger pressure magnitude statistics
                mag = torch.cat([
                    x.mean(dim=-1, keepdim=True),
                    x.std(dim=-1, keepdim=True),
                    x.amax(dim=-1, keepdim=True),
                ], dim=-1)
                mag_feat = self.finger_mag_mlp(mag)

                # Per-finger normalization
                finger_mean = x.mean(dim=-1, keepdim=True)
                finger_std = x.std(dim=-1, keepdim=True) + 1e-5
                x = (x - finger_mean) / finger_std

                x = x.reshape(-1, 1, rows, cols)
                x = self.finger_convs[fl["name"]](x)
                x = x.mean(dim=[-2, -1])
                x = x + mag_feat
                all_finger_feats.append(x)
                offset += n

        # Stack: (B, 12, channels) → project → (B, 12, d_model)
        finger_stack = torch.stack(all_finger_feats, dim=1)
        finger_emb = self.finger_proj(finger_stack)

        # Left/right cross-attention: each hand's query attends to its own 6 fingers
        b = finger_emb.shape[0]
        left_emb = finger_emb[:, :6, :]
        right_emb = finger_emb[:, 6:, :]

        left_q = self.left_query.unsqueeze(0).expand(b, -1, -1)
        right_q = self.right_query.unsqueeze(0).expand(b, -1, -1)

        left_out, _ = self.cross_attn(left_q, left_emb, left_emb)
        right_out, _ = self.cross_attn(right_q, right_emb, right_emb)

        left_out = self.attn_norm(left_out + left_q)
        right_out = self.attn_norm(right_out + right_q)
        return torch.cat([left_out, right_out], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Modality Tokenizers
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTDImageTokenizer(nn.Module):
    """Shared ResNet18 backbone + per-camera cross-attention pooling → fixed N tokens.

    Each camera view gets its own set of learnable query tokens that cross-attend
    to the spatial feature map, producing num_image_tokens_per_camera tokens.
    """

    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.config = config
        self.num_tokens = config.num_image_tokens_per_camera
        dim_model = config.dim_model

        # Shared ResNet18 backbone
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # 1x1 conv to project backbone features to dim_model
        self.feat_proj = nn.Conv2d(backbone_model.fc.in_features, dim_model, kernel_size=1)

        # 2D position embedding for image feature maps
        self.pos_embed = ACTSinusoidalPositionEmbedding2d(dim_model // 2)

        # Per-camera learnable query tokens + cross-attention
        # Derive number of cameras from config.image_features (matches runtime batch[OBS_IMAGES] length)
        self.num_cameras = len(config.image_features) if config.image_features else 3
        self.cam_queries = nn.ParameterList([
            nn.Parameter(torch.randn(config.num_image_tokens_per_camera, dim_model) * 0.02)
            for _ in range(self.num_cameras)
        ])
        self.cam_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim_model,
                num_heads=config.num_image_cross_attn_heads,
                batch_first=True,
            )
            for _ in range(self.num_cameras)
        ])
        self.cam_attn_norm = nn.ModuleList([
            nn.LayerNorm(dim_model) for _ in range(self.num_cameras)
        ])

    def forward(self, images: list[Tensor]) -> list[Tensor]:
        """Process camera images into tokens.

        Args:
            images: list of (B, C, H, W) tensors, one per camera.

        Returns:
            list of (B, num_tokens, dim_model) tensors.
        """
        results = []
        for cam_idx, img in enumerate(images):
            # Optional downscale (e.g. 2.0 = half resolution)
            if self.config.image_downscale_ratio > 1.0:
                _, _, h, w = img.shape
                img = F.interpolate(
                    img, scale_factor=1.0 / self.config.image_downscale_ratio,
                    mode="bilinear", align_corners=False,
                )
            # ResNet18 backbone
            feat_map = self.backbone(img)["feature_map"]  # (B, C_backbone, h, w)

            # Project to dim_model
            feat_map = self.feat_proj(feat_map)  # (B, dim_model, h, w)

            # 2D position embedding
            pos = self.pos_embed(feat_map).to(dtype=feat_map.dtype)  # (1, dim_model//2, h, w)
            # Note: pos_embed outputs dim_model//2 channels. The code from ACT
            # concatenates y and x position embeddings to get dim_model channels,
            # but ACTSinusoidalPositionEmbedding2d already outputs dim_model channels
            # (it stacks y and x halves). So shape is actually (1, dim_model, h, w).

            # Flatten spatial dims
            b, d, h, w = feat_map.shape
            feat_flat = einops.rearrange(feat_map, "b d h w -> b (h w) d")
            pos_flat = einops.rearrange(pos, "b d h w -> b (h w) d").expand(b, -1, -1)

            # Cross-attention pooling: queries attend to spatial positions
            query = self.cam_queries[cam_idx].unsqueeze(0).expand(b, -1, -1)  # (B, N, D)
            key_val = feat_flat + pos_flat  # add positional info to key/value
            tokens, _ = self.cam_cross_attn[cam_idx](query, key_val, feat_flat)
            tokens = self.cam_attn_norm[cam_idx](tokens + query)  # residual + norm

            results.append(tokens)

        return results


class ACTHTDProprioTokenizer(nn.Module):
    """Linear projection of 30-dim joints → 1 token."""

    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.proj = nn.Linear(26, config.dim_model)

    def forward(self, joints: Tensor) -> Tensor:
        """joints: (B, 26) → (B, 1, dim_model)"""
        return self.proj(joints).unsqueeze(1)


class ACTHTDForceTokenizer(nn.Module):
    """Separate left/right wrist force projections → 2 tokens."""

    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.left_proj = nn.Linear(6, config.dim_model)
        self.right_proj = nn.Linear(6, config.dim_model)

    def forward(self, force: Tensor) -> Tensor:
        """force: (B, 12) → (B, 2, dim_model)"""
        left = self.left_proj(force[:, :6]).unsqueeze(1)
        right = self.right_proj(force[:, 6:]).unsqueeze(1)
        return torch.cat([left, right], dim=1)


class ACTHTDTactileTokenizer(nn.Module):
    """TactileConvEncoder → latent + token projection → 2 tokens."""

    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.encoder = ACTHTDTactileEncoder(
            finger_layouts=config.finger_layouts,
            channels=config.tactile_encoder_channels,
            d_model=config.dim_model,
            num_tokens=config.num_tactile_tokens,
            latent_dim=config.tactile_latent_dim,
        )
        # Project latent to dim_model for the transformer
        self.latent_token_proj = nn.Linear(config.tactile_latent_dim, config.dim_model)

    def forward(self, tactile: Tensor) -> tuple[Tensor, Tensor]:
        """tactile: (B, 604) → (tokens (B,2,D), latent (B,2,latent_dim))"""
        latent = self.encoder(tactile)  # (B, 2, latent_dim)
        tokens = self.latent_token_proj(latent)  # (B, 2, dim_model)
        return tokens, latent


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer Encoder / Decoder (ACT-style, no VAE)
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTDEncoderLayer(nn.Module):
    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTHTDEncoder(nn.Module):
    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTHTDEncoderLayer(config) for _ in range(config.n_encoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        x = self.norm(x)
        return x


class ACTHTDDecoderLayer(nn.Module):
    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    @staticmethod
    def maybe_add_pos_embed(tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class ACTHTDDecoder(nn.Module):
    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTHTDDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out,
                      decoder_pos_embed=decoder_pos_embed,
                      encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Modular Action Expert
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTDActionExpert(nn.Module):
    """One action expert per action group.

    Cross-attention: query = learnable group embedding (expanded to chunk_size),
    key/value = shared decoder hidden.
    Output: (B, chunk_size, group_dim).
    """

    def __init__(self, config: ACTHTDConfig, group_name: str, group_start: int, group_end: int):
        super().__init__()
        self.group_name = group_name
        self.group_start = group_start
        self.group_end = group_end
        self.group_dim = group_end - group_start

        # Learnable query: (1, chunk_size, dim_model) — one query per time step
        self.query_embed = nn.Parameter(torch.randn(1, config.chunk_size, config.dim_model) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.dim_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.dim_model)
        self.output_proj = nn.Linear(config.dim_model, self.group_dim)

        # Position embedding to add temporal info to queries
        self.register_buffer(
            "temporal_pos_embed",
            create_sinusoidal_pos_embedding(config.chunk_size, config.dim_model).unsqueeze(0),
        )

    def forward(self, decoder_hidden: Tensor) -> Tensor:
        """decoder_hidden: (B, chunk_size, dim_model) → (B, chunk_size, group_dim)"""
        b = decoder_hidden.shape[0]
        query = self.query_embed.expand(b, -1, -1)  # (B, chunk_size, D)
        query = query + self.temporal_pos_embed.to(dtype=query.dtype, device=query.device)

        # Cross-attend to decoder hidden
        out, _ = self.cross_attn(query, decoder_hidden, decoder_hidden)
        out = self.norm(out + query)
        return self.output_proj(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Dream Expert (Touch Dreaming auxiliary heads)
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTDDreamExpert(nn.Module):
    """Touch dreaming prediction heads: force and tactile latent.

    Structure mirrors bbvla's force_pred_head / tactile_pred_head.
    During training only: predicts future force and EMA tactile latent
    from pooled decoder hidden states.
    """

    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.config = config

        # Force prediction head: dim_model → 12 (absolute force)
        self.force_pred_head = nn.Linear(config.dim_model, 12)

        # Tactile latent prediction head: dim_model → 2 * latent_dim (L/R hands)
        self.tactile_pred_head = nn.Linear(config.dim_model, 2 * config.tactile_latent_dim)

        # Project action estimate into dream hidden (bbvla's dream_action_proj)
        self.dream_action_proj = nn.Linear(config.max_action_dim, config.dim_model)

    def compute_losses(
        self,
        decoder_hidden: Tensor,
        action_chunk: Tensor,
        tactile_online_latent: Tensor,
        touch_ema_encoder: ACTHTDTactileEncoder,
        future_force: Tensor | None = None,
        future_tactile: Tensor | None = None,
        future_pad_mask: Tensor | None = None,
        next_force: Tensor | None = None,
        next_tactile: Tensor | None = None,
        current_tactile: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute touch dreaming losses.

        Uses decoder_hidden (B, chunk_size, D) to predict future force/tactile.
        Multi-step mode: uses future_force/future_tactile (B, K, D) as targets.
        Tau=1 fallback: uses next_force/next_tactile as targets.
        """
        losses = {}
        config = self.config

        # Fuse action estimate into dream hidden (bbvla pattern)
        dream_hidden = decoder_hidden + self.dream_action_proj(action_chunk)  # (B, C, D)

        use_multi_step = future_tactile is not None and future_force is not None

        if use_multi_step:
            # Multi-step: predict at K specific action positions
            offsets = config.get_touch_dreaming_offsets()
            K = len(offsets)

            action_indices = [off - 1 for off in offsets]  # zero-based
            selected_hidden = dream_hidden[:, action_indices, :]  # (B, K, D)

            # Force prediction — per-step, matching Eq.7 (τ future moments)
            force_pred = self.force_pred_head(selected_hidden)  # (B, K, 12)
            force_target = future_force  # (B, K, 12)
            force_loss = F.smooth_l1_loss(force_pred, force_target, reduction="none").mean(dim=-1)

            # Tactile latent prediction via EMA teacher
            B = future_tactile.shape[0]
            flat_tactile = future_tactile.reshape(B * K, -1)  # (B*K, 604)
            flat_latent = touch_ema_encoder.encode_latent(flat_tactile)  # (B*K, 2, latent_dim)
            tactile_target = flat_latent.reshape(B, K, 2, -1)  # (B, K, 2, latent_dim)
            tactile_pred = self.tactile_pred_head(selected_hidden).reshape(B, K, 2, -1)

            if config.touch_dreaming_use_cosine_loss:
                tp = tactile_pred.reshape(B * K * 2, -1)
                tt = tactile_target.reshape(B * K * 2, -1)
                direction_loss = (1 - F.cosine_similarity(tp, tt, dim=-1)).reshape(B, K, 2).mean(dim=-1)
                magnitude_loss = F.smooth_l1_loss(
                    tp.norm(dim=-1), tt.norm(dim=-1), reduction="none",
                ).reshape(B, K, 2).mean(dim=-1)
                tactile_loss = direction_loss + config.tactile_latent_loss_beta * magnitude_loss
            else:
                tactile_loss = F.mse_loss(tactile_pred, tactile_target, reduction="none").mean(dim=(-1, -2))

            if future_pad_mask is not None:
                valid = ~future_pad_mask.to(dtype=torch.bool, device=force_loss.device)
                valid_count = valid.sum()
                if valid_count > 0:
                    losses["force_pred_loss"] = force_loss.masked_select(valid).mean()
                    losses["tactile_pred_loss"] = tactile_loss.masked_select(valid).mean()
                else:
                    losses["force_pred_loss"] = force_loss.new_zeros(())
                    losses["tactile_pred_loss"] = tactile_loss.new_zeros(())
            else:
                losses["force_pred_loss"] = force_loss.mean()
                losses["tactile_pred_loss"] = tactile_loss.mean()

        else:
            # Tau=1 single-step mode with configurable pooling
            pooling = config.touch_dreaming_pooling
            if pooling == "mean":
                pooled_hidden = dream_hidden.mean(dim=1)
            elif pooling == "last":
                pooled_hidden = dream_hidden[:, -1, :]
            elif pooling == "mid":
                pooled_hidden = dream_hidden[:, dream_hidden.shape[1] // 2, :]
            else:
                raise ValueError(f"Unknown touch_dreaming_pooling='{pooling}'.")

            if next_force is not None:
                force_pred = self.force_pred_head(pooled_hidden)
                losses["force_pred_loss"] = F.smooth_l1_loss(force_pred, next_force, reduction="mean")

            tactile_pred = self.tactile_pred_head(pooled_hidden).reshape(-1, 2, config.tactile_latent_dim)
            td_tactile = next_tactile if next_tactile is not None else current_tactile
            tactile_target = touch_ema_encoder.encode_latent(td_tactile)
            Bs = pooled_hidden.shape[0]

            if config.touch_dreaming_use_cosine_loss:
                tp = tactile_pred.reshape(Bs * 2, -1)
                tt = tactile_target.reshape(Bs * 2, -1)
                direction_loss = (1 - F.cosine_similarity(tp, tt, dim=-1)).mean()
                magnitude_loss = F.smooth_l1_loss(
                    tp.norm(dim=-1), tt.norm(dim=-1), reduction="mean",
                )
                losses["tactile_pred_loss"] = direction_loss + config.tactile_latent_loss_beta * magnitude_loss
            else:
                losses["tactile_pred_loss"] = F.mse_loss(tactile_pred, tactile_target, reduction="mean")

        # VICReg variance loss to prevent tactile representation collapse
        # Computed in both multi-step and single-step modes
        if tactile_online_latent.shape[0] > 1:
            variance_loss = F.relu(
                1.0 - tactile_online_latent.reshape(-1, tactile_online_latent.shape[-1]).std(
                    dim=0, unbiased=False
                )
            ).mean()
        else:
            variance_loss = tactile_online_latent.new_zeros(())
        losses["variance_loss"] = variance_loss

        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# Core Model
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTD(nn.Module):
    """ACT-HTD: Action Chunking Transformer with Touch Dreaming.

    Full architecture:
    1. Modality tokenization (images, proprio, force, tactile)
    2. Transformer encoder → decoder
    3. Modular action experts → action chunk
    4. Dream experts (training only) → touch dreaming losses
    """

    def __init__(self, config: ACTHTDConfig):
        super().__init__()
        self.config = config

        # ── Modality tokenizers ──
        self.image_tokenizer = ACTHTDImageTokenizer(config)
        self.proprio_tokenizer = ACTHTDProprioTokenizer(config)
        self.force_tokenizer = ACTHTDForceTokenizer(config)
        self.tactile_tokenizer = ACTHTDTactileTokenizer(config)

        # ── Encoder token position & modality embeddings ──
        # Token layout: [cam0(N)][cam1(N)][cam2(N)][proprio(1)][force(2)][tactile(2)]
        N = config.num_image_tokens_per_camera
        self._num_img_tokens = 3 * N
        self._num_nonimg_tokens = 1 + 2 + 2  # proprio + force + tactile = 5
        total_tokens = self._num_img_tokens + self._num_nonimg_tokens

        # Modality-type embedding (4 types: image, proprio, force, tactile)
        self.modality_embed = nn.Embedding(4, config.dim_model)

        # Learned position embedding for all encoder tokens
        self.encoder_pos_embed = nn.Embedding(total_tokens, config.dim_model)

        # ── Transformer ──
        self.encoder = ACTHTDEncoder(config)
        self.decoder = ACTHTDDecoder(config)

        # Decoder query positions
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # ── Action experts (one per group) ──
        self.action_experts = nn.ModuleDict()
        for name, (start, end) in ACTION_GROUPS.items():
            self.action_experts[name] = ACTHTDActionExpert(config, name, start, end)

        # ── Dream expert (force + tactile prediction heads) ──
        self.dream_expert = ACTHTDDreamExpert(config)

        # ── EMA teacher encoder (for touch dreaming targets) ──
        self.tactile_ema_encoder = copy.deepcopy(self.tactile_tokenizer.encoder)
        for p in self.tactile_ema_encoder.parameters():
            p.requires_grad = False

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization for transformer parameters."""
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def ema_update_tactile_encoder(self):
        """EMA: online tactile encoder → EMA teacher (JEPA-style)."""
        for p_ema, p in zip(
            self.tactile_ema_encoder.parameters(),
            self.tactile_tokenizer.encoder.parameters(),
        ):
            p_ema.data = self.config.ema_decay * p_ema.data + (1 - self.config.ema_decay) * p.data

    # ── State slicing helpers ──

    @staticmethod
    def _slice_joints(state: Tensor) -> Tensor:
        return state[:, STATE_JOINTS_START:STATE_JOINTS_END]  # (B, 26)

    @staticmethod
    def _slice_force(state: Tensor) -> Tensor:
        return state[:, STATE_FORCE_START:STATE_FORCE_END]  # (B, 12)

    @staticmethod
    def _slice_tactile(state: Tensor) -> Tensor:
        left = state[:, STATE_TACTILE_LEFT_START:STATE_TACTILE_LEFT_END]    # (B, 302)
        right = state[:, STATE_TACTILE_RIGHT_START:STATE_TACTILE_RIGHT_END]  # (B, 302)
        return torch.cat([left, right], dim=-1)  # (B, 604)

    # ── Modality-type indices for encoder tokens ──
    # 0=image, 1=proprio, 2=force, 3=tactile

    def _build_modality_type_indices(self, device: torch.device) -> Tensor:
        """Build modality type index tensor for encoder tokens."""
        N = self.config.num_image_tokens_per_camera
        nc = self.image_tokenizer.num_cameras
        indices = []
        # nc cameras × N tokens each → modality 0 (image)
        indices.extend([0] * (nc * N))
        # 1 proprio token → modality 1
        indices.append(1)
        # 2 force tokens → modality 2
        indices.extend([2, 2])
        # 2 tactile tokens → modality 3
        indices.extend([3, 3])
        return torch.tensor(indices, device=device, dtype=torch.long)

    def forward(
        self,
        batch: dict[str, Tensor],
        return_dream_predictions: bool = False,
        dream_offset: int = 30,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward pass. Returns (action_chunk, dream_losses_dict).

        In training mode with touch_dreaming_enabled, dream_losses_dict contains
        force_pred_loss, tactile_pred_loss, variance_loss.
        In eval mode, dream_losses_dict is empty. When
        ``return_dream_predictions`` is true, a third return value contains the
        current tactile latent and the selected future force/tactile predictions.
        """
        images = batch[OBS_IMAGES]  # list of (B, C, H, W)
        state = batch[OBS_STATE]  # (B, 646)

        joints = self._slice_joints(state)
        force = self._slice_force(state)
        tactile = self._slice_tactile(state)

        # ── 1. Tokenize each modality ──
        img_tokens_list = self.image_tokenizer(images)  # list of (B, N, D) × 3
        proprio_tokens = self.proprio_tokenizer(joints)  # (B, 1, D)
        force_tokens = self.force_tokenizer(force)  # (B, 2, D)
        tactile_tokens, tactile_latent = self.tactile_tokenizer(tactile)  # (B, 2, D), (B, 2, latent_dim)

        # ── 2. Concatenate all encoder tokens ──
        all_tokens = torch.cat(
            [*img_tokens_list, proprio_tokens, force_tokens, tactile_tokens], dim=1
        )  # (B, total_tokens, D)
        b, total_tokens, d = all_tokens.shape

        # Add modality type embeddings
        modality_ids = self._build_modality_type_indices(all_tokens.device)  # (total_tokens,)
        all_tokens = all_tokens + self.modality_embed(modality_ids).unsqueeze(0)  # (B, total_tokens, D)

        # Add learned position embeddings
        pos_ids = torch.arange(total_tokens, device=all_tokens.device)  # (total_tokens,)
        all_tokens = all_tokens + self.encoder_pos_embed(pos_ids).unsqueeze(0)  # (B, total_tokens, D)

        # ── 3. Transformer encoder ──
        # Convert to (seq, batch, dim) for transformer
        encoder_in = all_tokens.permute(1, 0, 2)  # (total_tokens, B, D)
        encoder_out = self.encoder(encoder_in)  # (total_tokens, B, D)

        # ── 4. Transformer decoder ──
        decoder_in = torch.zeros(
            (self.config.chunk_size, b, d),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )  # (chunk_size, B, D)

        # (B, chunk_size, D)
        decoder_out = decoder_out.permute(1, 0, 2)

        # ── 5. Action experts ──
        action_parts = []
        for name in ACTION_GROUPS:
            expert = self.action_experts[name]
            part = expert(decoder_out)  # (B, chunk_size, group_dim)
            action_parts.append(part)
        action_chunk = torch.cat(action_parts, dim=-1)  # (B, chunk_size, 26)

        dream_predictions = None
        if return_dream_predictions and self.config.touch_dreaming_enabled:
            # The auxiliary head is trained from decoder states at configured
            # future offsets. Deployment explicitly asks for the 30th frame
            # (one second at 30 Hz), so clamp only to the available chunk.
            offset = max(1, min(int(dream_offset), self.config.chunk_size))
            selected_hidden = decoder_out[:, offset - 1, :]
            dream_hidden = selected_hidden + self.dream_expert.dream_action_proj(
                action_chunk[:, offset - 1, :]
            )
            force_pred = self.dream_expert.force_pred_head(dream_hidden)
            tactile_pred = self.dream_expert.tactile_pred_head(dream_hidden).reshape(
                -1, 2, self.config.tactile_latent_dim
            )
            dream_predictions = {
                "offset": offset,
                "current_tactile_latent": tactile_latent,
                "force_pred": force_pred,
                "tactile_pred": tactile_pred,
            }

        # ── 6. Dream losses (training only) ──
        dream_losses = {}
        if self.training and self.config.touch_dreaming_enabled:
            # Extract future targets from batch
            future_force, future_tactile = None, None
            if FUTURE_OBS_STATE in batch:
                future_state = batch[FUTURE_OBS_STATE]
                # Handle batch dim: (1, B, K, 646) → (B, K, 646)
                if future_state.ndim >= 4 and future_state.shape[0] == 1:
                    future_state = future_state.squeeze(0)
                future_force = future_state[..., STATE_FORCE_START:STATE_FORCE_END]
                fl = future_state[..., STATE_TACTILE_LEFT_START:STATE_TACTILE_LEFT_END]
                fr = future_state[..., STATE_TACTILE_RIGHT_START:STATE_TACTILE_RIGHT_END]
                future_tactile = torch.cat([fl, fr], dim=-1)

            future_pad_mask = batch.get(f"{FUTURE_OBS_STATE}_is_pad")
            if future_pad_mask is not None and future_pad_mask.ndim >= 3 and future_pad_mask.shape[0] == 1:
                future_pad_mask = future_pad_mask.squeeze(0)

            next_force, next_tactile = None, None
            if NEXT_OBS_STATE in batch:
                next_state = batch[NEXT_OBS_STATE]
                if next_state.ndim >= 3 and next_state.shape[0] == 1:
                    next_state = next_state.squeeze(0)
                next_force = next_state[..., STATE_FORCE_START:STATE_FORCE_END]
                nl = next_state[..., STATE_TACTILE_LEFT_START:STATE_TACTILE_LEFT_END]
                nr = next_state[..., STATE_TACTILE_RIGHT_START:STATE_TACTILE_RIGHT_END]
                next_tactile = torch.cat([nl, nr], dim=-1)

            dream_losses = self.dream_expert.compute_losses(
                decoder_hidden=decoder_out,
                action_chunk=action_chunk,
                tactile_online_latent=tactile_latent,
                touch_ema_encoder=self.tactile_ema_encoder,
                future_force=future_force,
                future_tactile=future_tactile,
                future_pad_mask=future_pad_mask,
                next_force=next_force,
                next_tactile=next_tactile,
                current_tactile=tactile,
            )

        if return_dream_predictions:
            return action_chunk, dream_losses, dream_predictions
        return action_chunk, dream_losses


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal Ensembler (from modeling_act.py)
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTDTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


# ═══════════════════════════════════════════════════════════════════════════════
# Policy
# ═══════════════════════════════════════════════════════════════════════════════


class ACTHTDPolicy(PreTrainedPolicy):
    """ACT-HTD Policy: Action Chunking Transformer with Touch Dreaming.

    Standard lerobot policy interface:
    - forward(batch) → (loss, loss_dict)
    - predict_action_chunk(batch) → (B, chunk_size, action_dim)
    - select_action(batch) → (B, action_dim)
    - reset(), update()
    """

    config_class = ACTHTDConfig
    name = "act_htd"

    def __init__(self, config: ACTHTDConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTHTD(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTHTDTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("model.image_tokenizer.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.image_tokenizer.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """Reset state between episodes."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action for environment execution."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions, _ = self.model(batch)
        return actions

    @torch.no_grad()
    def predict_action_and_dream(
        self, batch: dict[str, Tensor], dream_offset: int = 30
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Inference output for control plus the touch-dreaming diagnostics.

        The action output is exactly the same action chunk used by normal
        inference. The extra output is normalized model-space force prediction,
        predicted left/right tactile latents, and the current tactile latent.
        """
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions, _, dream = self.model(
            batch, return_dream_predictions=True, dream_offset=dream_offset
        )
        return actions, dream

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward: compute action L1 loss + touch dreaming auxiliary losses."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, dream_losses = self.model(batch)

        # Action L1 loss (slice to 26-dim, dataset stores 32-dim with head/waist/base)
        target_action = batch[ACTION][..., :self.config.max_action_dim]
        abs_err = F.l1_loss(target_action, actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        # Add touch dreaming losses
        if self.config.touch_dreaming_enabled and dream_losses:
            force_loss = dream_losses.get("force_pred_loss")
            tactile_loss = dream_losses.get("tactile_pred_loss")
            variance_loss = dream_losses.get("variance_loss")
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
        return loss, loss_dict

    def update(self):
        """Called by the training loop after optimizer.step().

        Updates the EMA tactile encoder (HTD JEPA-style).
        """
        if self.config.touch_dreaming_enabled and self.training:
            self.model.ema_update_tactile_encoder()
