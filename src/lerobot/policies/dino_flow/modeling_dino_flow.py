from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        emb = t[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AttentionResampler(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_tokens: int, num_heads: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        keys = self.input_proj(patch_tokens)
        queries = self.queries.expand(patch_tokens.shape[0], -1, -1)
        attended, _ = self.attn(queries, keys, keys, need_weights=False)
        out = self.norm1(queries + attended)
        return self.norm2(out + self.ffn(out))


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
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.image_sizes = config.image_resize_shapes
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _preprocess(self, image: Tensor, size: tuple[int, int]) -> Tensor:
        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        image = torchvision.transforms.functional.resize(image, size, antialias=True)
        mean = self.mean.to(image.device, image.dtype)
        std = self.std.to(image.device, image.dtype)
        return (image - mean) / std

    def forward(self, images: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = {}
        for key, image in images.items():
            if image.ndim == 5:
                image = image[:, -1]
            image = self._preprocess(image, self.image_sizes[key])
            with torch.no_grad():
                model_out = self.model(pixel_values=image)
            # DINOv3 exposes CLS + four register tokens.  Keep this dynamic so
            # an accessible DINOv2 checkpoint can be used for a local smoke test.
            num_prefix_tokens = getattr(self.model.config, "num_register_tokens", 0) + 1
            patch_tokens = model_out.last_hidden_state[:, num_prefix_tokens:]
            outputs[key] = patch_tokens
        return outputs


class ActionDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: Tensor, condition: Tensor, time_emb: Tensor) -> Tensor:
        shift_s, scale_s, gate_s, shift_c, scale_c, gate_c = self.ada(time_emb).chunk(6, dim=-1)
        h = self.norm_self(x) * (1 + scale_s[:, None]) + shift_s[:, None]
        self_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_s[:, None] * self_out
        h = self.norm_cross(x) * (1 + scale_c[:, None]) + shift_c[:, None]
        cross, _ = self.cross_attn(h, condition, condition, need_weights=False)
        x = x + gate_c[:, None] * cross
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ActionDiT(nn.Module):
    def __init__(self, config: DinoFlowConfig):
        super().__init__()
        # Concat the current state onto every noisy-action token so the model has
        # direct per-timestep access to the observed joints (not just a single
        # state token among hundreds of visual tokens).
        self.action_in = nn.Linear(config.action_dim + config.state_dim, config.hidden_dim)
        self.state_in = nn.Sequential(nn.Linear(config.state_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.time_in = nn.Sequential(SinusoidalEmbedding(config.timestep_embed_dim), nn.Linear(config.timestep_embed_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.action_pos = nn.Parameter(torch.randn(1, config.horizon, config.hidden_dim) * 0.02)
        self.camera_embeddings = nn.Parameter(
            torch.randn(len(config.image_resize_shapes), 1, config.hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList([ActionDiTBlock(config.hidden_dim, config.num_heads, config.dropout) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.action_out = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, noisy_action: Tensor, state: Tensor, visual_tokens: list[Tensor], t: Tensor) -> Tensor:
        state_expanded = state[:, None, :].expand(-1, noisy_action.shape[1], -1)
        action_input = torch.cat([noisy_action, state_expanded], dim=-1)
        action_tokens = self.action_in(action_input) + self.action_pos[:, : noisy_action.shape[1]]
        condition_parts = []
        for idx, tokens in enumerate(visual_tokens):
            condition_parts.append(tokens + self.camera_embeddings[idx])
        condition_parts.append(self.state_in(state)[:, None])
        condition = torch.cat(condition_parts, dim=1)
        time_emb = self.time_in(t)
        for block in self.blocks:
            action_tokens = block(action_tokens, condition, time_emb)
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
        self.camera_keys = list(config.image_resize_shapes)
        self.resamplers = nn.ModuleList([
            AttentionResampler(
                config.vision_encoder_dim,
                config.hidden_dim,
                config.resampler_tokens,
                config.resampler_heads,
            )
            for _ in self.camera_keys
        ])
        self.action_model = ActionDiT(config)
        # A fixed initial noise makes deployment deterministic.  The policy
        # still changes with image/state conditioning, but repeated requests
        # no longer introduce an unrelated chunk-to-chunk random jump.
        self._inference_noise: dict[tuple, Tensor] = {}
        self._action_queue = deque(maxlen=config.n_action_steps)
        self.reset()

    def get_optim_params(self) -> list:
        return [param for param in self.parameters() if param.requires_grad]

    def reset(self):
        self._action_queue.clear()

    def _current_state(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state[:, -1]
        return state[..., : self.config.state_dim]

    def _current_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        patch_tokens = self.vision({key: batch[key] for key in self.camera_keys})
        return [resampler(patch_tokens[key]) for key, resampler in zip(self.camera_keys, self.resamplers)]

    def _condition(self, batch: dict[str, Tensor]) -> tuple[Tensor, list[Tensor]]:
        state = self._current_state(batch)
        visual_tokens = self._current_images(batch)
        return state, visual_tokens

    def _predict_velocity(self, x: Tensor, t: Tensor, state: Tensor, visual_tokens: list[Tensor]) -> Tensor:
        return self.action_model(x, state, visual_tokens, t)

    def _flow_loss(self, batch: dict[str, Tensor], state: Tensor, visual_tokens: list[Tensor]) -> Tensor:
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
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t_view = t[:, None, None]
        x_t = (1 - t_view) * noise + t_view * target
        target_velocity = target - noise
        pred = self._predict_velocity(x_t, t, state, visual_tokens)
        loss = F.mse_loss(pred, target_velocity, reduction="none")
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            action_is_pad = batch["action_is_pad"]
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            mask = (~action_is_pad[:, : self.config.horizon]).unsqueeze(-1).to(loss.dtype)
            return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)
        return loss.mean()

    @staticmethod
    def _rtc_prefix_weights(total: int, start: int, end: int, device, dtype) -> Tensor:
        """Weights for the RTC-style prefix constraint.

        The prefix is strongest for the actions that are about to be executed,
        then fades to zero at ``end``.  This is the same idea as LeRobot RTC,
        but expressed directly for this policy's 0 -> 1 flow parameterization.
        """
        start = max(0, min(int(start), total))
        end = max(start, min(int(end), total))
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
            x.shape[1], int(inference_delay), int(inference_delay) + int(execution_horizon),
            x.device, x.dtype,
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
        execution_horizon: int = 10,
    ) -> Tensor:
        state, visual_tokens = self._condition(batch)
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
            velocity = self._predict_velocity(x, t, state, visual_tokens)
            velocity = self._rtc_velocity(
                x, velocity, t, prev_chunk_left_over, inference_delay, execution_horizon
            )
            if self.config.integration_method == "heun" and i < steps - 1:
                x_euler = x + dt * velocity
                t_next = torch.full((state.shape[0],), (i + 1) / steps, device=state.device, dtype=dtype)
                velocity_next = self._predict_velocity(x_euler, t_next, state, visual_tokens)
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
            x = x.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
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
        return self._sample(
            batch,
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            inference_delay=int(kwargs.get("inference_delay", 0) or 0),
            execution_horizon=int(kwargs.get("execution_horizon", 10) or 10),
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        state, visual_tokens = self._condition(batch)
        return self._flow_loss(batch, state, visual_tokens), None
