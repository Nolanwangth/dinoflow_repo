from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from .configuration_dino_flow import DinoFlowConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel
else:
    AutoModel = None


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        emb = t[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DinoVisionEncoder(nn.Module):
    def __init__(self, config: DinoFlowConfig):
        super().__init__()
        require_package("transformers", extra="dino_flow")
        self.model = AutoModel.from_pretrained(config.vision_encoder_name)
        model_hidden_size = getattr(self.model.config, "hidden_size", None)
        if model_hidden_size != config.vision_encoder_dim:
            raise ValueError(
                f"DINO hidden size mismatch: checkpoint has {model_hidden_size}, "
                f"but vision_encoder_dim={config.vision_encoder_dim}"
            )
        model_patch_size = getattr(self.model.config, "patch_size", None)
        if model_patch_size is not None and model_patch_size != config.vision_patch_size:
            raise ValueError(
                f"DINO patch size mismatch: checkpoint has {model_patch_size}, "
                f"but vision_patch_size={config.vision_patch_size}"
            )
        self.lora_enabled = config.vision_lora_enabled
        self.gradient_checkpointing_enabled = False
        if self.lora_enabled:
            require_package("peft", extra="dino_flow")
            from peft import LoraConfig, get_peft_model

            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=config.vision_lora_rank,
                    lora_alpha=config.vision_lora_alpha,
                    lora_dropout=config.vision_lora_dropout,
                    target_modules=["q_proj", "v_proj"],
                    bias="none",
                ),
            )
            if config.vision_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                self.gradient_checkpointing_enabled = True
            lora_parameter_names = [
                name
                for name, param in self.model.named_parameters()
                if param.requires_grad and ("lora_A" in name or "lora_B" in name)
            ]
            expected_lora_parameters = 2 * 2 * getattr(self.model.config, "num_hidden_layers", 12)
            if len(lora_parameter_names) != expected_lora_parameters:
                raise RuntimeError(
                    "DINO Q/V LoRA target check failed: "
                    f"found {len(lora_parameter_names)} trainable adapter tensors, "
                    f"expected {expected_lora_parameters} for Q/V in all Transformer blocks."
                )
            logging.info(
                "DINOv3 Q/V LoRA enabled: rank=%s alpha=%s dropout=%s trainable_tensors=%s",
                config.vision_lora_rank,
                config.vision_lora_alpha,
                config.vision_lora_dropout,
                len(lora_parameter_names),
            )
            if config.vision_gradient_checkpointing:
                logging.info("DINO gradient checkpointing enabled for LoRA training")
        else:
            for param in self.model.parameters():
                param.requires_grad_(False)
        self.model.eval()
        self.image_sizes = config.image_resize_shapes
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        # DINO's checkpointing wrapper only activates while each Transformer
        # layer is in train mode. Keep the frozen model's stochastic layers in
        # eval mode, while marking the checkpointed layers as training layers.
        # DINOv3-S+ has no active dropout/drop-path here (drop_path_rate=0).
        if mode and self.gradient_checkpointing_enabled:
            base_model = self.model.get_base_model()
            for layer in base_model.model.layer:
                layer.train(True)
        return self

    def _preprocess(self, image: Tensor, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Resize one camera and return its image plus valid patch mask.

        The image is resized to the target height while preserving aspect
        ratio. If it is wider than the target, excess width is center-cropped;
        if it is narrower, it is centered and padded with black pixels. A
        patch is valid when any part of its receptive field overlaps the
        resized image; fully padded patches are masked out in the Action DiT
        cross-attention.
        """
        if image.ndim != 4:
            raise ValueError(f"Expected camera images in BCHW format, got shape {tuple(image.shape)}")

        target_height, target_width = size
        _, _, source_height, source_width = image.shape
        resize_ratio = source_height / target_height
        resized_height = target_height
        resized_width = max(1, int(round(source_width / resize_ratio)))

        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        image = F.interpolate(
            image, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        crop_width = max(0, resized_width - target_width)
        crop_left = crop_width // 2
        crop_right = crop_width - crop_left
        if crop_width:
            image = image[..., crop_left : resized_width - crop_right]
            resized_width = target_width

        pad_width = target_width - resized_width
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        image = F.pad(image, (pad_left, pad_right, 0, 0), mode="constant", value=0.0)

        valid_pixels = torch.zeros(
            (image.shape[0], 1, target_height, target_width), device=image.device, dtype=torch.bool
        )
        valid_pixels[:, :, :, pad_left : pad_left + resized_width] = True
        grid_height = target_height // self.model.config.patch_size
        grid_width = target_width // self.model.config.patch_size
        patch_size = self.model.config.patch_size
        valid_patches = valid_pixels.reshape(
            image.shape[0], 1, grid_height, patch_size, grid_width, patch_size
        ).any(dim=(1, 3, 5)).reshape(image.shape[0], grid_height * grid_width)

        mean = self.mean.to(image.device, image.dtype)
        std = self.std.to(image.device, image.dtype)
        return (image - mean) / std, valid_patches

    def forward(self, images: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
        outputs = {}
        for key, image in images.items():
            if image.ndim == 5:
                image = image[:, -1]
            image, valid_patches = self._preprocess(image, self.image_sizes[key])
            context = torch.enable_grad() if self.lora_enabled and torch.is_grad_enabled() else torch.no_grad()
            with context:
                model_out = self.model(pixel_values=image)
            # DINOv3 exposes CLS + four register tokens.  Keep this dynamic so
            # an accessible DINOv2 checkpoint can be used for a local smoke test.
            num_prefix_tokens = getattr(self.model.config, "num_register_tokens", 0) + 1
            patch_tokens = model_out.last_hidden_state[:, num_prefix_tokens:]
            if patch_tokens.shape[1] != valid_patches.shape[1]:
                raise RuntimeError(
                    f"DINO patch count mismatch for {key}: model returned {patch_tokens.shape[1]} patches, "
                    f"but the {self.image_sizes[key]} input requires {valid_patches.shape[1]}."
                )
            outputs[key] = (patch_tokens, valid_patches)
        return outputs


class TactileRegionEncoder(nn.Module):
    """Shared encoder for the twelve tactile regions on the two hands."""

    # l/r thumb, index, middle, ring, pinky, and palm.  The state vector keeps
    # this order, with one complete hand followed by the other.
    REGION_SHAPES = ((5, 7), (5, 12), (5, 12), (5, 12), (4, 8), (5, 11)) * 2

    def __init__(self, config: DinoFlowConfig):
        super().__init__()
        region_size = sum(height * width for height, width in self.REGION_SHAPES)
        if region_size != config.tactile_dim:
            raise ValueError(
                f"Configured tactile_dim={config.tactile_dim} does not match the 12-region layout ({region_size})"
            )
        self.region_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 32),
        )
        self.active_threshold = config.tactile_active_threshold
        self.history_steps = config.tactile_history_steps

    def forward_regions(self, tactile: Tensor) -> Tensor:
        """Encode ``[B, history, 604]`` as ``[B, history, 12, 35]``."""
        if tactile.ndim != 3:
            raise ValueError(f"Expected tactile history [B,T,D], got {tuple(tactile.shape)}")
        batch, history, width = tactile.shape
        if width != sum(height * width for height, width in self.REGION_SHAPES):
            raise ValueError(f"Expected {sum(h * w for h, w in self.REGION_SHAPES)} tactile values, got {width}")

        features = []
        offset = 0
        for height, region_width in self.REGION_SHAPES:
            size = height * region_width
            region = tactile[..., offset : offset + size].reshape(
                batch * history, 1, height, region_width
            )
            encoded = self.region_encoder(region).reshape(batch, history, 32)
            # The first statistic is the mean value, the second captures a
            # local peak, and the ratio tracks how much of the pad is active.
            raw_region = region.reshape(batch, history, size)
            stats = torch.stack(
                [
                    raw_region.mean(dim=-1),
                    raw_region.amax(dim=-1),
                    (raw_region > self.active_threshold).to(raw_region.dtype).mean(dim=-1),
                ],
                dim=-1,
            )
            features.append(torch.cat([encoded, stats], dim=-1))
            offset += size

        return torch.stack(features, dim=2)

    def forward(self, tactile: Tensor) -> Tensor:
        """Encode ``[B, history, 604]`` without per-frame standardization."""
        region_features = self.forward_regions(tactile)
        return region_features.flatten(2)


class ContactHistoryEncoder(nn.Module):
    """Fuse six frames of tactile regions and two wrist wrenches."""

    def __init__(self, config: DinoFlowConfig):
        super().__init__()
        self.force_in = nn.Sequential(nn.Linear(config.wrist_force_dim, 32), nn.SiLU())
        self.tactile_in = TactileRegionEncoder(config)
        frame_dim = 32 + 12 * (32 + 3)
        self.history_in = nn.Sequential(
            nn.Linear(config.tactile_history_steps * frame_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        token_dim = config.contact_token_dim
        wrist_dim = config.wrist_force_dim // 2
        self.left_force_token_in = nn.Sequential(
            nn.Linear(wrist_dim, 32), nn.SiLU(), nn.Linear(32, token_dim)
        )
        self.right_force_token_in = nn.Sequential(
            nn.Linear(wrist_dim, 32), nn.SiLU(), nn.Linear(32, token_dim)
        )
        self.tactile_token_in = nn.Sequential(
            nn.Linear(32 + 3, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim)
        )
        # Token order is twelve tactile regions, left wrist, right wrist for
        # each frame. Learned identities make that order explicit to attention.
        self.contact_identity = nn.Parameter(torch.randn(14, token_dim) * 0.02)
        self.contact_time = nn.Parameter(torch.randn(config.tactile_history_steps, token_dim) * 0.02)
        self.contact_token_norm = nn.LayerNorm(token_dim)
        self.force_offset = config.tactile_state_offset - config.wrist_force_dim
        self.tactile_offset = config.tactile_state_offset
        self.force_end = config.tactile_state_offset
        self.tactile_dim = config.tactile_dim
        self.history_steps = config.tactile_history_steps

    def _encoded_components(self, state_history: Tensor) -> tuple[Tensor, Tensor]:
        if state_history.ndim != 3:
            raise ValueError(f"Expected state history [B,T,D], got {tuple(state_history.shape)}")
        if state_history.shape[1] != self.history_steps:
            raise ValueError(
                f"Expected {self.history_steps} contact history frames, got {state_history.shape[1]}"
            )
        force = state_history[..., self.force_offset : self.force_end]
        tactile = state_history[..., self.tactile_offset : self.tactile_offset + self.tactile_dim]
        tactile_features = self.tactile_in.forward_regions(tactile)
        return force, tactile_features

    def _tokens_from_components(self, force: Tensor, tactile_features: Tensor) -> Tensor:
        batch, history = force.shape[:2]
        tactile_tokens = self.tactile_token_in(tactile_features)
        wrist_dim = force.shape[-1] // 2
        left_token = self.left_force_token_in(force[..., :wrist_dim]).unsqueeze(2)
        right_token = self.right_force_token_in(force[..., wrist_dim:]).unsqueeze(2)
        frame_tokens = torch.cat([tactile_tokens, left_token, right_token], dim=2)
        frame_tokens = frame_tokens + self.contact_identity[None, None]
        frame_tokens = frame_tokens + self.contact_time[None, :, None]
        return self.contact_token_norm(frame_tokens.reshape(batch, history * 14, -1))

    def forward(self, state_history: Tensor) -> Tensor:
        force, tactile_features = self._encoded_components(state_history)
        force_features = self.force_in(force)
        frame_features = torch.cat([force_features, tactile_features.flatten(2)], dim=-1)
        return self.history_in(frame_features.reshape(state_history.shape[0], -1))

    def forward_tokens(self, state_history: Tensor) -> Tensor:
        """Return contact history as identity- and time-aware queryable tokens."""
        force, tactile_features = self._encoded_components(state_history)
        return self._tokens_from_components(force, tactile_features)

    def forward_with_tokens(self, state_history: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the global summary and local tokens with one tactile CNN pass."""
        force, tactile_features = self._encoded_components(state_history)
        force_features = self.force_in(force)
        frame_features = torch.cat([force_features, tactile_features.flatten(2)], dim=-1)
        global_features = self.history_in(frame_features.reshape(state_history.shape[0], -1))
        return global_features, self._tokens_from_components(force, tactile_features)


class ActionDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        contact_token_dim: int = 128,
        contact_token_heads: int = 4,
        use_contact_attention: bool = False,
    ):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.contact_cross_attn = None
        if use_contact_attention:
            self.norm_contact = nn.LayerNorm(dim)
            self.contact_query = nn.Linear(dim, contact_token_dim)
            self.contact_kv_norm = nn.LayerNorm(contact_token_dim)
            self.contact_cross_attn = nn.MultiheadAttention(
                contact_token_dim,
                contact_token_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.contact_out = nn.Linear(contact_token_dim, dim)
            # Preserve the old network's output at initialization. The new
            # branch learns its residual after contact_out starts moving.
            nn.init.zeros_(self.contact_out.weight)
            nn.init.zeros_(self.contact_out.bias)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        condition_key_padding_mask: Tensor | None,
        time_emb: Tensor,
        contact_tokens: Tensor | None = None,
    ) -> Tensor:
        shift_s, scale_s, gate_s, shift_c, scale_c, gate_c = self.ada(time_emb).chunk(6, dim=-1)
        h = self.norm_self(x) * (1 + scale_s[:, None]) + shift_s[:, None]
        self_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_s[:, None] * self_out
        h = self.norm_cross(x) * (1 + scale_c[:, None]) + shift_c[:, None]
        cross, _ = self.cross_attn(
            h,
            condition,
            condition,
            key_padding_mask=condition_key_padding_mask,
            need_weights=False,
        )
        x = x + gate_c[:, None] * cross
        if self.contact_cross_attn is not None and contact_tokens is not None:
            contact_query = self.contact_query(self.norm_contact(x))
            contact_kv = self.contact_kv_norm(contact_tokens)
            contact_update, _ = self.contact_cross_attn(
                contact_query, contact_kv, contact_kv, need_weights=False
            )
            x = x + self.contact_out(contact_update)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ActionDiT(nn.Module):
    def __init__(self, config: DinoFlowConfig):
        super().__init__()
        # Concat the current state onto every noisy-action token so the model has
        # direct per-timestep access to the observed joints (not just a single
        # state token among hundreds of visual tokens).
        self.action_in = nn.Linear(config.action_dim + config.state_dim, config.hidden_dim)
        self.state_in = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.time_in = nn.Sequential(
            SinusoidalEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.contact_in = nn.Linear(128, config.hidden_dim)
        nn.init.zeros_(self.contact_in.weight)
        nn.init.zeros_(self.contact_in.bias)
        self.action_pos = nn.Parameter(torch.randn(1, config.horizon, config.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                ActionDiTBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.dropout,
                    contact_token_dim=config.contact_token_dim,
                    contact_token_heads=config.contact_token_heads,
                    use_contact_attention=(
                        config.use_contact_tokens
                        and block_idx >= max(0, config.num_layers - config.contact_attention_layers)
                    ),
                )
                for block_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.action_out = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(
        self,
        noisy_action: Tensor,
        state: Tensor,
        visual_tokens: Tensor,
        visual_valid_mask: Tensor,
        t: Tensor,
        contact_features: Tensor | None = None,
        contact_tokens: Tensor | None = None,
    ) -> Tensor:
        state_expanded = state[:, None, :].expand(-1, noisy_action.shape[1], -1)
        action_input = torch.cat([noisy_action, state_expanded], dim=-1)
        action_tokens = self.action_in(action_input) + self.action_pos[:, : noisy_action.shape[1]]
        condition = torch.cat([visual_tokens, self.state_in(state)[:, None]], dim=1)
        condition_key_padding_mask = torch.cat(
            [
                ~visual_valid_mask.to(torch.bool),
                torch.zeros((state.shape[0], 1), device=state.device, dtype=torch.bool),
            ],
            dim=1,
        )
        time_emb = self.time_in(t)
        if contact_features is not None:
            time_emb = time_emb + self.contact_in(contact_features)
        for block in self.blocks:
            action_tokens = block(
                action_tokens,
                condition,
                condition_key_padding_mask,
                time_emb,
                contact_tokens=contact_tokens,
            )
        return self.action_out(self.norm(action_tokens))


class DinoFlowPolicy(PreTrainedPolicy):
    config_class = DinoFlowConfig
    name = "dino_flow"

    def __init__(self, config: DinoFlowConfig, **kwargs):
        super().__init__(config)
        self.config = config
        # Features are attached after config construction in the training
        # factory, so validate again on the normal policy creation path.
        self.config.validate_features()
        self.vision = DinoVisionEncoder(config)
        self.contact_encoder = ContactHistoryEncoder(config)
        self.camera_keys = list(config.image_resize_shapes)
        self.visual_projection = nn.Linear(config.vision_encoder_dim, config.hidden_dim)
        self.camera_embeddings = nn.Parameter(
            torch.zeros(len(self.camera_keys), config.hidden_dim)
        )
        self.action_model = ActionDiT(config)
        # A fixed initial noise makes deployment deterministic.  The policy
        # still changes with image/state conditioning, but repeated requests
        # no longer introduce an unrelated chunk-to-chunk random jump.
        self._inference_noise: dict[tuple, Tensor] = {}
        self._last_sample_clip_fraction = 0.0
        self._action_queue = deque(maxlen=config.n_action_steps)
        self.reset()

    def get_optim_params(self) -> list:
        other_params = []
        lora_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.config.vision_lora_lr})
        return param_groups

    def reset(self):
        self._action_queue.clear()

    def _state_history(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim == 2:
            state = state[:, None]
        if state.ndim != 3:
            raise ValueError(f"Expected observation.state [B,T,D] or [B,D], got {tuple(state.shape)}")
        if state.shape[1] < self.config.tactile_history_steps:
            pad = state[:, :1].expand(-1, self.config.tactile_history_steps - state.shape[1], -1)
            state = torch.cat([pad, state], dim=1)
        return state[:, -self.config.tactile_history_steps :]

    def _current_state(self, batch: dict[str, Tensor]) -> Tensor:
        return self._state_history(batch)[:, -1, : self.config.state_dim]

    def _current_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        encoded = self.vision({key: batch[key] for key in self.camera_keys})
        projected_tokens = []
        valid_masks = []
        for camera_idx, key in enumerate(self.camera_keys):
            tokens, valid_mask = encoded[key]
            tokens = self.visual_projection(tokens)
            if self.config.use_camera_embedding:
                tokens = tokens + self.camera_embeddings[camera_idx].view(1, 1, -1)
            projected_tokens.append(tokens)
            valid_masks.append(valid_mask)
        return torch.cat(projected_tokens, dim=1), torch.cat(valid_masks, dim=1)

    def _contact_conditions(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        state_history = self._state_history(batch)
        required = self.config.tactile_state_offset + self.config.tactile_dim
        if state_history.shape[-1] < required:
            # Keep small synthetic/unit-test configurations usable; real
            # DinoFlow training fails loudly at dataset validation instead.
            global_features = torch.zeros(
                (state_history.shape[0], 128), device=state_history.device, dtype=state_history.dtype
            )
            contact_tokens = None
            if self.config.use_contact_tokens:
                contact_tokens = torch.zeros(
                    (
                        state_history.shape[0],
                        self.config.tactile_history_steps * 14,
                        self.config.contact_token_dim,
                    ),
                    device=state_history.device,
                    dtype=state_history.dtype,
                )
            return global_features, contact_tokens
        if self.config.use_contact_tokens:
            global_features, contact_tokens = self.contact_encoder.forward_with_tokens(state_history)
        else:
            global_features = self.contact_encoder(state_history)
            contact_tokens = None
        return global_features, contact_tokens

    def _condition(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        state = self._current_state(batch)
        visual_tokens, visual_valid_mask = self._current_images(batch)
        contact_features, contact_tokens = self._contact_conditions(batch)
        return state, visual_tokens, visual_valid_mask, contact_features, contact_tokens

    def _predict_velocity(
        self,
        x: Tensor,
        t: Tensor,
        state: Tensor,
        visual_tokens: Tensor,
        visual_valid_mask: Tensor,
        contact_features: Tensor,
        contact_tokens: Tensor | None,
    ) -> Tensor:
        return self.action_model(
            x,
            state,
            visual_tokens,
            visual_valid_mask,
            t,
            contact_features,
            contact_tokens,
        )

    def _flow_loss(
        self,
        batch: dict[str, Tensor],
        state: Tensor,
        visual_tokens: Tensor,
        visual_valid_mask: Tensor,
        contact_features: Tensor,
        contact_tokens: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        target = batch[ACTION][..., : self.config.action_dim]
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if target.shape[1] != self.config.horizon:
            target = target[:, : self.config.horizon]
        if self.config.use_delta_action:
            # Regress on delta = action_norm - state_norm. The first chunk step is
            # then ~0 (action == current pose), which is far better conditioned than
            # predicting absolute joints and anchors step 0 to the observed state.
            target = target - state[:, None, :]
        noise = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        t = torch.rand(
            target.shape[0],
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        t_view = t[:, None, None]
        x_t = (1 - t_view) * noise + t_view * target
        target_velocity = target - noise
        pred = self._predict_velocity(
            x_t,
            t,
            state,
            visual_tokens,
            visual_valid_mask,
            contact_features,
            contact_tokens,
        )
        loss = F.mse_loss(pred, target_velocity, reduction="none")
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            action_is_pad = batch["action_is_pad"]
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            mask = (~action_is_pad[:, : self.config.horizon]).unsqueeze(-1).to(loss.dtype)
            return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)
        return loss.mean()

    @staticmethod
    def _rtc_prefix_weights(
        total: int,
        start: int,
        end: int,
        device,
        dtype,
        available_length: int | None = None,
    ) -> Tensor:
        """Weights for the RTC-style prefix constraint.

        The prefix is strongest for the actions that are about to be executed,
        then fades to zero at ``end``.  This is the same idea as LeRobot RTC,
        but expressed directly for this policy's 0 -> 1 flow parameterization.
        """
        start = max(0, min(int(start), total))
        end = max(start, min(int(end), total))
        if available_length is not None:
            end = min(end, max(0, int(available_length)))
            if end <= start:
                return torch.zeros(total, device=device, dtype=dtype)
        weights = torch.zeros(total, device=device, dtype=dtype)
        if start:
            weights[:start] = 1.0
        if end > start:
            weights[start:end] = torch.linspace(1.0, 0.0, end - start, device=device, dtype=dtype)
        return weights

    def _rtc_velocity(
        self,
        x: Tensor,
        velocity: Tensor,
        t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        execution_horizon: int,
    ) -> Tensor:
        """Apply an endpoint-based RTC correction for DinoFlow.

        DinoFlow integrates noise -> action (t increasing), while the generic
        RTC implementation in this repository is written for the opposite
        convention used by PI05.  We therefore use the equivalent endpoint
        constraint without importing PI05's reverse-time assumptions.
        """
        if prev_chunk_left_over is None:
            return velocity
        if prev_chunk_left_over.ndim == 2:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
        if prev_chunk_left_over.shape[0] != x.shape[0]:
            prev_chunk_left_over = prev_chunk_left_over.expand(x.shape[0], -1, -1)
        prefix = torch.zeros_like(x)
        copy_t = min(prefix.shape[1], prev_chunk_left_over.shape[1])
        copy_d = min(prefix.shape[2], prev_chunk_left_over.shape[2])
        prefix[:, :copy_t, :copy_d] = prev_chunk_left_over[:, :copy_t, :copy_d].to(x.dtype)

        # x + (1-t)v is the model's current estimate of the final action.
        remaining = (1.0 - t).clamp_min(1.0 / max(self.config.num_integration_steps, 1))
        endpoint = x + remaining[:, None, None] * velocity
        weights = self._rtc_prefix_weights(
            x.shape[1],
            int(inference_delay),
            int(inference_delay) + int(execution_horizon),
            x.device,
            x.dtype,
            available_length=prev_chunk_left_over.shape[1],
        )[None, :, None]
        active = (weights > 0).to(x.dtype)
        # Convert endpoint error back to a velocity correction.  Keep the
        # coefficient modest: the next observation will re-anchor the policy.
        correction = (prefix - endpoint) / remaining[:, None, None]
        correction = correction * weights * active
        gain = 0.35 * (1.0 - t).clamp(0.0, 1.0)[:, None, None]
        return velocity + gain * correction

    @torch.no_grad()
    def _sample(
        self,
        batch: dict[str, Tensor],
        prev_chunk_left_over: Tensor | None = None,
        inference_delay: int = 0,
        execution_horizon: int = 20,
    ) -> Tensor:
        state, visual_tokens, visual_valid_mask, contact_features, contact_tokens = self._condition(batch)
        dtype = next(self.action_model.parameters()).dtype
        noise_shape = (state.shape[0], self.config.horizon, self.config.action_dim)
        noise_key = (noise_shape, str(state.device), dtype)
        noise = self._inference_noise.get(noise_key)
        if noise is None or noise.device != state.device:
            generator = torch.Generator(device=state.device)
            generator.manual_seed(0)
            noise = torch.randn(noise_shape, device=state.device, dtype=dtype, generator=generator)
            self._inference_noise[noise_key] = noise
        x = noise.clone()
        if self.config.use_delta_action and prev_chunk_left_over is not None:
            # prev_chunk_left_over arrives in normalized ACTION space. Convert to
            # delta space once so _rtc_velocity (which corrects in the integration
            # space of x) stays consistent.
            if prev_chunk_left_over.ndim == 2:
                prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
            if prev_chunk_left_over.shape[0] != state.shape[0]:
                prev_chunk_left_over = prev_chunk_left_over.expand(state.shape[0], -1, -1)
            prev_chunk_left_over = prev_chunk_left_over - state[:, None, :]
        steps = self.config.num_integration_steps
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((state.shape[0],), i / steps, device=state.device, dtype=dtype)
            velocity = self._predict_velocity(
                x, t, state, visual_tokens, visual_valid_mask, contact_features, contact_tokens
            )
            velocity = self._rtc_velocity(
                x, velocity, t, prev_chunk_left_over, inference_delay, execution_horizon
            )
            if self.config.integration_method == "heun" and i < steps - 1:
                x_euler = x + dt * velocity
                t_next = torch.full((state.shape[0],), (i + 1) / steps, device=state.device, dtype=dtype)
                velocity_next = self._predict_velocity(
                    x_euler,
                    t_next,
                    state,
                    visual_tokens,
                    visual_valid_mask,
                    contact_features,
                    contact_tokens,
                )
                velocity_next = self._rtc_velocity(
                    x_euler, velocity_next, t_next, prev_chunk_left_over,
                    inference_delay, execution_horizon,
                )
                x = x + dt * 0.5 * (velocity + velocity_next)
            else:
                x = x + dt * velocity
        if self.config.use_delta_action:
            # Integrate in delta space, then convert back to absolute action space.
            x = x + state[:, None, :]
        if self.config.clip_sample:
            self._last_sample_clip_fraction = float(
                (x.abs() > self.config.clip_sample_range).to(torch.float32).mean().item()
            )
            x = x.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
        else:
            self._last_sample_clip_fraction = 0.0
        return x

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions[:, : self.config.n_action_steps].transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Generate one full action horizon in normalized action coordinates."""
        execution_horizon = kwargs.get("execution_horizon")
        if execution_horizon is None:
            execution_horizon = 20
        return self._sample(
            batch,
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            inference_delay=int(kwargs.get("inference_delay", 0) or 0),
            execution_horizon=int(execution_horizon),
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        deterministic: bool = False,
    ) -> tuple[Tensor, dict | None]:
        state, visual_tokens, visual_valid_mask, contact_features, contact_tokens = self._condition(batch)
        generator = None
        if deterministic:
            generator = torch.Generator(device=state.device)
            generator.manual_seed(0)
        return (
            self._flow_loss(
                batch,
                state,
                visual_tokens,
                visual_valid_mask,
                contact_features,
                contact_tokens,
                generator=generator,
            ),
            None,
        )
