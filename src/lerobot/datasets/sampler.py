#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from collections.abc import Iterator

import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class FixedEpisodeSampler:
    """Deterministic, temporally spread samples for validation.

    Samples are distributed across the selected episodes and uniformly across
    each episode's timeline. This avoids a validation pass that changes every
    time and gives short validation runs early, middle, and late examples.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        max_frames: int,
        episode_indices_to_use: list[int] | None = None,
    ):
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        if len(dataset_from_indices) != len(dataset_to_indices):
            raise ValueError("Episode start/end index lists must have the same length")

        selected = set(episode_indices_to_use) if episode_indices_to_use is not None else None
        episodes = [
            (episode_idx, int(start), int(end))
            for episode_idx, (start, end) in enumerate(zip(dataset_from_indices, dataset_to_indices, strict=True))
            if (selected is None or episode_idx in selected) and end > start
        ]
        if not episodes:
            raise ValueError("No non-empty episodes remain for validation")

        total_frames = sum(end - start for _, start, end in episodes)
        target_frames = min(int(max_frames), total_frames)
        if target_frames < len(episodes):
            episode_positions = torch.linspace(0, len(episodes) - 1, target_frames).round().long().tolist()
            counts = [0] * len(episodes)
            for position in episode_positions:
                counts[position] += 1
        else:
            base, remainder = divmod(target_frames, len(episodes))
            counts = [base + (index < remainder) for index in range(len(episodes))]

        indices: list[int] = []
        for (_, start, end), count in zip(episodes, counts, strict=True):
            if count == 0:
                continue
            if count == 1:
                positions = [0.5]
            else:
                positions = torch.linspace(0.0, 1.0, count).tolist()
            indices.extend(
                start + min(end - start - 1, int(round(position * (end - start - 1))))
                for position in positions
            )
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)
