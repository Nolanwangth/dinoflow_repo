#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    FUTURE_OBS_STATE,
    NEXT_OBS_STATE,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_bbvla import (
    STATE_JOINTS_END,
    STATE_JOINTS_START,
    BbvlaConfig,
)


@ProcessorStepRegistry.register(name="bbvla_prepare_state_tokenizer_processor_step")
@dataclass
class BbvlaPrepareStateTokenizerProcessorStep(ProcessorStep):
    """Processor step to split 646-dim state into joints/force/tactile and format
    the PaliGemma language prompt (joints only; force/tactile go through suffix tokens).

    State layout (646-dim):
      [0:30]    joints (arm 14 + hands 12 + head 2 + waist 2)
      [30:42]   wrist force (left 6 + right 6) — suffix only
      [42:344]  tactile left  (302: 6 fingers) — suffix only
      [344:646] tactile right (302: 6 fingers) — suffix only
    """

    max_state_dim: int = 646
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for Bbvla")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # State should already be normalized to [-1, 1] by NormalizerProcessorStep
        state_np = state.cpu().numpy().astype(np.float64)

        # --- Split state (force/tactile unused here — they go through suffix tokens) ---
        joints = state_np[..., STATE_JOINTS_START:STATE_JOINTS_END]        # 30 dims

        # --- Discretize (256 bins, same as PI05) ---
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        joints_disc = np.digitize(joints, bins=bins) - 1

        # --- Format prompt: simple PI05-style bare values ---
        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, joints_disc[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="bbvla_clamp_normalized_processor_step")
@dataclass
class BbvlaClampNormalizedProcessorStep(ProcessorStep):
    """Clamp q01/q99-normalized sensor/action tensors before BBVLA loss computation."""

    max_abs_value: float = 10.0

    def _clamp_tensor(self, value: Any) -> Any:
        if not isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
            return value
        return torch.as_tensor(value).clamp(-self.max_abs_value, self.max_abs_value)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            observation = observation.copy()
            for key in (OBS_STATE, NEXT_OBS_STATE, FUTURE_OBS_STATE):
                if key in observation:
                    observation[key] = self._clamp_tensor(observation[key])
            transition[TransitionKey.OBSERVATION] = observation

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = self._clamp_tensor(action)

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_bbvla_pre_post_processors(
    config: BbvlaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre/post processor pipelines for Bbvla policy.

    Pre-processing pipeline:
      1. Rename observations
      2. Add batch dimension
      3. Relative actions (if enabled)
      4. Normalize (QUANTILES for state/action, IDENTITY for visual)
      5. Bbvla state tokenizer (split + coarse text injection)
      6. PaliGemma tokenizer
      7. Move to device

    Post-processing pipeline:
      1. Unnormalize
      2. Absolute actions (if relative)
      3. Move to CPU
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    # Load PaliGemma tokenizer from local pi05_base checkpoint
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nolan/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c",
        local_files_only=True,
    )

    # Share normalization stats: future.observation.state = observation.state
    if dataset_stats is not None and OBS_STATE in dataset_stats:
        if FUTURE_OBS_STATE not in dataset_stats:
            dataset_stats[FUTURE_OBS_STATE] = dataset_stats[OBS_STATE]

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        BbvlaClampNormalizedProcessorStep(max_abs_value=config.normalized_value_clip),
        BbvlaPrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer=tokenizer,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
