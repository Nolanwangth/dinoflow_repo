#!/usr/bin/env python3
"""Evaluate one complete validation episode with raw DinoFlow actions.

This is an offline diagnostic.  It does not use RTC, EMA, a robot, or the
deployment TCP server.  The dataset is loaded with the same delta timestamps
as training: six 30 Hz contact-history states and a 50-step action horizon.

Example:
    PYTHONPATH=src python scripts/evaluate_validation_episode.py
    PYTHONPATH=src python scripts/evaluate_validation_episode.py --episode 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


DEFAULT_MODEL = (
    "/home/nolan/vla/dinoflow_repo/outputs/"
    "contact_tokens01_phase1_absolute_action_h256_camid_contactglobal128_"
    "contactlocal128_84tok_last2_loraqv_r8_b32_30k_seed1000_20260913/"
    "checkpoints/015000/pretrained_model"
)
DEFAULT_DATASET = "/home/nolan/vla/openpi_repo/lerobot_datasets/splice_wires_phase1_split_300_21/validation"


def _load_policy(model_path: Path, device: str):
    # Reuse the deployment loader so model loading and processor loading stay
    # identical to the actual DinoFlow server.  Solver settings are taken from
    # the checkpoint config below, so this evaluation matches training.
    sys.path.insert(0, str(model_path.parents[4]))
    from deployment.server import load_policy

    config = json.loads((model_path / "config.json").read_text())
    return load_policy(
        model_path=model_path,
        device=device,
        n_action_steps=int(config.get("n_action_steps", 50)),
        num_integration_steps=int(config.get("num_integration_steps", 8)),
        integration_method=str(config.get("integration_method", "euler")),
    )


def _make_dataset(dataset_root: Path, policy):
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    meta = LeRobotDatasetMetadata("local/validation_episode", root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, meta)
    dataset = LeRobotDataset(
        "local/validation_episode",
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    return dataset


def _rmse(values: list[np.ndarray]) -> float:
    if not values:
        return float("nan")
    flat = np.concatenate([value.reshape(-1) for value in values])
    return float(np.sqrt(np.mean(flat**2)))


def _mae(values: list[np.ndarray]) -> float:
    if not values:
        return float("nan")
    flat = np.concatenate([value.reshape(-1) for value in values])
    return float(np.mean(np.abs(flat)))


def _group_metrics(errors: list[np.ndarray]) -> dict[str, float]:
    return {
        "all_rmse_rad": _rmse(errors),
        "all_mae_rad": _mae(errors),
        "arm_rmse_rad": _rmse([value[..., :14] for value in errors]),
        "arm_mae_rad": _mae([value[..., :14] for value in errors]),
        "hand_rmse_rad": _rmse([value[..., 14:26] for value in errors]),
        "hand_mae_rad": _mae([value[..., 14:26] for value in errors]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--episode", type=int, default=None, help="Validation episode index; default is a seeded random episode.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation_episode_eval"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {args.dataset_root}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    print(f"[eval] model={args.model_path}", flush=True)
    print(f"[eval] dataset={args.dataset_root}", flush=True)
    print("[eval] mode=absolute_action, rtc=off, ema=off", flush=True)

    policy, preprocessor, _postprocessor = _load_policy(args.model_path, args.device)
    policy.eval()
    dataset = _make_dataset(args.dataset_root, policy)
    episodes = dataset.meta.episodes
    episode_count = len(episodes)
    if args.episode is None:
        episode = int(np.random.default_rng(args.seed).integers(episode_count))
    else:
        episode = int(args.episode)
    if episode < 0 or episode >= episode_count:
        raise ValueError(f"episode must be in [0, {episode_count}), got {episode}")

    start = int(episodes["dataset_from_index"][episode])
    end = int(episodes["dataset_to_index"][episode])
    length = end - start
    indices = list(range(start, end))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    normalizer = next(
        step for step in preprocessor.steps if step.__class__.__name__ == "NormalizerProcessorStep"
    )

    action_dim = int(policy.config.action_dim)
    horizon = int(policy.config.horizon)
    camera_keys = tuple(dataset.meta.camera_keys)
    frame_indices: list[int] = []
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    pred0_rows: list[np.ndarray] = []
    gt0_rows: list[np.ndarray] = []
    clip_saturated = 0
    clip_total = 0
    first_errors: list[np.ndarray] = []
    execution_errors: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    first_jumps: list[np.ndarray] = []
    chunk_diff_values: list[np.ndarray] = []
    gt_diff_values: list[np.ndarray] = []
    first_continuity_values: list[np.ndarray] = []
    previous_pred0: np.ndarray | None = None
    horizon_error_sq = np.zeros(horizon, dtype=np.float64)
    horizon_error_count = np.zeros(horizon, dtype=np.int64)
    horizon_arm_sq = np.zeros(horizon, dtype=np.float64)
    horizon_hand_sq = np.zeros(horizon, dtype=np.float64)

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            # Keep the dataset-local frame index before the policy
            # preprocessor removes bookkeeping fields.
            batch_episode_frames = batch["frame_index"].cpu().numpy().reshape(-1)
            for key in camera_keys:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].to(torch.float32) / 255.0
            batch = preprocessor(batch)

            # No RTC kwargs and no post-inference smoothing: this is the raw
            # absolute-action prediction produced by the checkpoint.
            pred_norm = policy.predict_action_chunk(batch)
            gt_norm = batch["action"][..., :action_dim].float()
            pred_raw = normalizer._normalize_action(pred_norm.float(), inverse=True).cpu().numpy()
            gt_raw = normalizer._normalize_action(gt_norm, inverse=True).cpu().numpy()
            # The stored observation state has 646 features, while DinoFlow
            # uses only its first 26 joint features as the current state.
            # Invert normalization at the full feature width first; the
            # normalizer stores one statistic per observation feature.
            state_norm_full = batch["observation.state"][:, -1, :]
            state_raw_full = normalizer._normalize_observation(
                {"observation.state": state_norm_full}, inverse=True
            )["observation.state"]
            state_raw = state_raw_full[:, :action_dim].cpu().numpy()
            pred_norm_np = pred_norm.float().cpu().numpy()

            for row, local_frame in enumerate(batch_episode_frames):
                global_index = start + int(local_frame)
                valid = min(horizon, end - global_index)
                if valid <= 0:
                    continue
                p = pred_raw[row, :valid]
                g = gt_raw[row, :valid]
                e = p - g
                frame_indices.append(int(local_frame))
                pred_chunks.append(pred_raw[row])
                gt_chunks.append(gt_raw[row])
                state_rows.append(state_raw[row])
                pred0_rows.append(p[0])
                gt0_rows.append(g[0])
                first_errors.append(e[:1])
                execution_errors.append(e[: min(20, valid)])
                all_errors.append(e)
                first_jumps.append((p[0] - state_raw[row])[None])
                if valid > 1:
                    chunk_diff_values.append(np.diff(p, axis=0))
                    gt_diff_values.append(np.diff(g, axis=0))
                if previous_pred0 is not None:
                    first_continuity_values.append((p[0] - previous_pred0)[None])
                previous_pred0 = p[0].copy()

                clip_saturated += int(np.count_nonzero(np.abs(pred_norm_np[row, :valid]) >= 0.999))
                clip_total += int(valid * action_dim)
                horizon_error_sq[:valid] += np.sum(e**2, axis=1)
                horizon_error_count[:valid] += action_dim
                horizon_arm_sq[:valid] += np.sum(e[:, :14] ** 2, axis=1)
                horizon_hand_sq[:valid] += np.sum(e[:, 14:26] ** 2, axis=1)

            if batch_number == 1 or batch_number % 10 == 0:
                print(f"[eval] processed {min(batch_number * args.batch_size, length)}/{length} frames", flush=True)

    if not pred_chunks:
        raise RuntimeError("no valid frames were evaluated")

    first_error_metrics = _group_metrics(first_errors)
    execution_error_metrics = _group_metrics(execution_errors)
    all_error_metrics = _group_metrics(all_errors)
    jump_metrics = _group_metrics(first_jumps)
    pred_diff_metrics = _group_metrics(chunk_diff_values)
    gt_diff_metrics = _group_metrics(gt_diff_values)
    continuity_metrics = _group_metrics(first_continuity_values)
    horizon_rmse = np.sqrt(horizon_error_sq / np.maximum(horizon_error_count, 1)).tolist()
    horizon_arm_rmse = np.sqrt(horizon_arm_sq / np.maximum(horizon_error_count / action_dim * 14, 1)).tolist()
    horizon_hand_rmse = np.sqrt(horizon_hand_sq / np.maximum(horizon_error_count / action_dim * 12, 1)).tolist()

    pred0 = np.stack(pred0_rows)
    gt0 = np.stack(gt0_rows)
    state = np.stack(state_rows)
    summary = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "episode": episode,
        "episode_length_frames": length,
        "episode_duration_s_at_30hz": length / 30.0,
        "mode": {"rtc": False, "ema": False, "action": "absolute", "solver": "checkpoint_config"},
        "first_step_error": first_error_metrics,
        "first_20_execution_error": execution_error_metrics,
        "all_valid_horizon_error": all_error_metrics,
        "first_step_minus_current_state": jump_metrics,
        "predicted_chunk_temporal_difference": pred_diff_metrics,
        "ground_truth_temporal_difference": gt_diff_metrics,
        "predicted_first_step_continuity_across_observations": continuity_metrics,
        "normalized_output_saturation_fraction": clip_saturated / max(clip_total, 1),
        "horizon_rmse_rad": horizon_rmse,
        "horizon_arm_rmse_rad": horizon_arm_rmse,
        "horizon_hand_rmse_rad": horizon_hand_rmse,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"episode_{episode:03d}.npz",
        frame_index=np.asarray(frame_indices, dtype=np.int64),
        pred_chunk=np.stack(pred_chunks),
        gt_chunk=np.stack(gt_chunks),
        state=state,
    )
    (args.output_dir / f"episode_{episode:03d}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.asarray(frame_indices) / 30.0
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        arm_to_plot = [0, 7, 11, 12, 13]
        hand_to_plot = [14, 19, 20, 25]
        for dim in arm_to_plot:
            axes[0].plot(t, pred0[:, dim], label=f"pred arm_{dim}", linewidth=1.0)
            axes[0].plot(t, gt0[:, dim], "--", label=f"gt arm_{dim}", linewidth=0.8)
        for dim in hand_to_plot:
            axes[1].plot(t, pred0[:, dim], label=f"pred action_{dim}", linewidth=1.0)
            axes[1].plot(t, gt0[:, dim], "--", label=f"gt action_{dim}", linewidth=0.8)
        axes[2].plot(np.arange(horizon) / 30.0, horizon_rmse, label="all")
        axes[2].plot(np.arange(horizon) / 30.0, horizon_arm_rmse, label="arm")
        axes[2].plot(np.arange(horizon) / 30.0, horizon_hand_rmse, label="hand")
        axes[0].set_title(f"Validation episode {episode}: raw first-step arm action")
        axes[1].set_title("Raw first-step hand action")
        axes[2].set_title("Prediction RMSE by horizon offset")
        axes[0].set_ylabel("rad")
        axes[1].set_ylabel("rad")
        axes[2].set_ylabel("RMSE (rad)")
        axes[2].set_xlabel("horizon offset (s)")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"episode_{episode:03d}.png", dpi=140)
        plt.close(fig)
    except ImportError:
        print("[eval] matplotlib unavailable; skipped plot", flush=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[eval] artifacts written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
