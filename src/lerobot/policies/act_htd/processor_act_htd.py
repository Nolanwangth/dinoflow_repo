#!/usr/bin/env python

"""Pre/post processor pipeline for ACT-HTD policy.

Follows the same pattern as ACT: Rename→AddBatchDim→Device→Normalize
for pre-processing, and Unnormalize→Device for post-processing.
The model internally slices the 646-dim state into joints/force/tactile,
so the processor normalizes the full 646-dim vector as one unit.
"""

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    FUTURE_OBS_STATE,
    NEXT_OBS_STATE,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_act_htd import ACTHTDConfig


def make_act_htd_pre_post_processors(
    config: ACTHTDConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT-HTD policy.

    Args:
        config: The ACT-HTD policy configuration.
        dataset_stats: Dataset statistics for normalization.

    Returns:
        tuple of (pre-processor, post-processor) pipelines.
    """
    # Share normalization stats: future/next.observation.state → observation.state
    if dataset_stats is not None and OBS_STATE in dataset_stats:
        for key in (NEXT_OBS_STATE, FUTURE_OBS_STATE):
            if key not in dataset_stats:
                dataset_stats[key] = dataset_stats[OBS_STATE]

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
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
