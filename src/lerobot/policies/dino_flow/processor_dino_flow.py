from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_dino_flow import DinoFlowConfig


@dataclass
@ProcessorStepRegistry.register(name="dino_flow_slice_features")
class SliceDinoFlowFeaturesStep(ProcessorStep):
    state_dim: int = 26
    observation_state_dim: int = 646
    action_dim: int = 26

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION.value)
        if observation is not None and OBS_STATE in observation:
            observation = dict(observation)
            observation[OBS_STATE] = observation[OBS_STATE][..., : self.observation_state_dim]
            new_transition[TransitionKey.OBSERVATION.value] = observation
        action = new_transition.get(TransitionKey.ACTION.value)
        if action is not None:
            new_transition[TransitionKey.ACTION.value] = action[..., : self.action_dim]
        return new_transition

    def transform_features(self, features):
        transformed = deepcopy(features)
        if OBS_STATE in transformed.get(PipelineFeatureType.OBSERVATION, {}):
            ft = transformed[PipelineFeatureType.OBSERVATION][OBS_STATE]
            transformed[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                ft.type, (self.observation_state_dim,)
            )
        if ACTION in transformed.get(PipelineFeatureType.ACTION, {}):
            ft = transformed[PipelineFeatureType.ACTION][ACTION]
            transformed[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(ft.type, (self.action_dim,))
        return transformed


def _active_features(config: DinoFlowConfig) -> dict[str, PolicyFeature]:
    features = dict(config.input_features)
    features[OBS_STATE] = PolicyFeature(FeatureType.STATE, (config.observation_state_dim,))
    return features


def _active_output_features(config: DinoFlowConfig) -> dict[str, PolicyFeature]:
    return {ACTION: PolicyFeature(FeatureType.ACTION, (config.action_dim,))}


def _slice_stats(dataset_stats: dict[str, dict[str, Any]] | None, config: DinoFlowConfig):
    if dataset_stats is None:
        return None
    stats = deepcopy(dataset_stats)
    # Keep the complete observation.state statistics: the policy extracts the
    # first 26 joint values and the force/tactile suffix separately.
    for key, dim in ((ACTION, config.action_dim),):
        if key in stats:
            for stat_name, value in list(stats[key].items()):
                if isinstance(value, torch.Tensor):
                    stats[key][stat_name] = value[:dim]
                elif isinstance(value, (list, np.ndarray)):
                    stats[key][stat_name] = value[:dim]
    return stats


def make_dino_flow_pre_post_processors(
    config: DinoFlowConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    active_features = _active_features(config)
    active_outputs = _active_output_features(config)
    stats = _slice_stats(dataset_stats, config)
    pre_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        SliceDinoFlowFeaturesStep(
            state_dim=config.state_dim,
            observation_state_dim=config.observation_state_dim,
            action_dim=config.action_dim,
        ),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**active_features, **active_outputs},
            norm_map=config.normalization_mapping,
            stats=stats,
            device=config.device,
        ),
    ]
    post_steps = [
        UnnormalizerProcessorStep(
            features=active_outputs,
            norm_map=config.normalization_mapping,
            stats=stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(steps=pre_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(
            steps=post_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
