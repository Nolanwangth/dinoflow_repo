#!/usr/bin/env python3
"""DinoFlow TCP inference server.

The policy predicts a 50-step flow-matching chunk and consumes 30 actions
from each chunk before requesting a new observation-conditioned chunk.
"""
from __future__ import annotations

import argparse
import io
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import ujson
from PIL import Image

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.dino_flow.configuration_dino_flow import DinoFlowConfig
from lerobot.policies.dino_flow.modeling_dino_flow import DinoFlowPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    while n:
        got = sock.recv_into(view, n)
        if got == 0:
            raise ConnectionError("connection closed")
        view = view[got:]
        n -= got
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[dict, dict[str, bytes]]:
    total = struct.unpack(">I", recv_exactly(sock, 4))[0]
    data = recv_exactly(sock, total)
    json_len = struct.unpack(">I", data[:4])[0]
    meta = ujson.loads(data[4 : 4 + json_len])
    offset = 4 + json_len
    count = data[offset]
    offset += 1
    images = {}
    for _ in range(count):
        key_len = data[offset]
        offset += 1
        key = data[offset : offset + key_len].decode("ascii")
        offset += key_len
        image_len = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        images[key] = data[offset : offset + image_len]
        offset += image_len
    return meta, images


def send_response(sock: socket.socket, payload: dict) -> None:
    raw = ujson.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _features(raw: dict) -> dict[str, PolicyFeature]:
    return {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in raw.items()
    }


def load_policy(
    model_path: Path,
    device: str,
    n_action_steps: int,
    num_integration_steps: int,
    integration_method: str,
):
    config_data = ujson.loads((model_path / "config.json").read_bytes())
    config_data.pop("type", None)
    config_data["input_features"] = _features(config_data["input_features"])
    config_data["output_features"] = _features(config_data["output_features"])
    config_data["n_action_steps"] = n_action_steps
    config_data["num_integration_steps"] = num_integration_steps
    config_data["integration_method"] = integration_method
    config_data["device"] = device
    cfg = DinoFlowConfig(**config_data)
    policy = DinoFlowPolicy.from_pretrained(
        str(model_path), config=cfg, local_files_only=True, strict=False
    )
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post


@dataclass
class Observation:
    state: np.ndarray
    head: bytes
    left_wrist: bytes
    right_wrist: bytes


class DinoFlowSession:
    def __init__(self, policy, preprocessor, postprocessor):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state_dim = int(policy.config.state_dim)
        self.observation_state_dim = int(
            getattr(policy.config, "observation_state_dim", self.state_dim)
        )
        self.action_dim = int(policy.config.action_dim)
        self.action_normalizer = next(
            (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)), None
        )

    @staticmethod
    def _decode(data: bytes) -> torch.Tensor:
        array = np.ascontiguousarray(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).copy())
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)

    def _batch(self, obs: Observation) -> dict:
        values = np.asarray(obs.state, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError(f"state must have shape [D] or [T,D], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("state contains NaN or Inf")
        state_history = np.zeros(
            (values.shape[0], self.observation_state_dim), dtype=np.float32
        )
        state_history[:, : min(values.shape[1], self.observation_state_dim)] = values[
            :, : self.observation_state_dim
        ]
        return {
            # The generic batch processor only adds a batch dimension to
            # 1-D states. Keep the history explicitly batched as [1,T,D].
            "observation.state": torch.from_numpy(state_history[None, ...]),
            "observation.images.base_0_rgb": self._decode(obs.head),
            "observation.images.left_wrist_0_rgb": self._decode(obs.left_wrist),
            "observation.images.right_wrist_0_rgb": self._decode(obs.right_wrist),
        }

    @torch.no_grad()
    def _normalize_previous_actions(self, actions) -> torch.Tensor | None:
        if actions is None or len(actions) == 0:
            return None
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"prev_chunk_left_over must be 2D, got {values.shape}")
        values = values[:, : self.action_dim]
        tensor = torch.from_numpy(values)
        if self.action_normalizer is None:
            return tensor.unsqueeze(0).to(self.policy.config.device)
        return self.action_normalizer._normalize_action(  # noqa: SLF001
            tensor.to(self.policy.config.device), inverse=False
        ).unsqueeze(0)

    def infer(
        self,
        obs: Observation,
        prev_chunk_left_over=None,
        inference_delay: int = 0,
        execution_horizon: int = 10,
    ) -> tuple[np.ndarray, float]:
        tic = time.perf_counter()
        batch = self.preprocessor(self._batch(obs))
        # Always generate the complete 50-step chunk. The client decides
        # how many actions to execute.
        previous = self._normalize_previous_actions(prev_chunk_left_over)
        action_chunk = self.policy.predict_action_chunk(
            batch,
            prev_chunk_left_over=previous,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        action_chunk = self.postprocessor(action_chunk)[0].float().cpu().numpy()
        return action_chunk[:, :26], (time.perf_counter() - tic) * 1000.0


def handle(conn: socket.socket, addr, session: DinoFlowSession) -> None:
    print(f"[tcp] client connected: {addr}", flush=True)
    count = 0
    try:
        while True:
            meta, images = recv_frame(conn)
            if meta.get("type") != "step_request":
                continue
            required = ("head", "left_wrist", "right_wrist")
            if any(key not in images for key in required):
                raise ValueError(f"missing images; expected {required}")
            obs = Observation(
                state=np.asarray(
                    meta.get("state", meta.get("joints", [])), dtype=np.float32
                ),
                head=images["head"],
                left_wrist=images["left_wrist"],
                right_wrist=images["right_wrist"],
            )
            action_chunk, infer_ms = session.infer(
                obs,
                prev_chunk_left_over=meta.get("prev_chunk_left_over"),
                inference_delay=int(meta.get("inference_delay", 0)),
                execution_horizon=int(meta.get("execution_horizon", 10)),
            )
            count += 1
            send_response(conn, {
                "type": "step_action",
                "actions": action_chunk.tolist(),
                "server_ms": infer_ms,
                "horizon": 50,
                "n_action_steps": 30,
                "rtc_enabled": meta.get("prev_chunk_left_over") is not None,
                "timestamp": time.time(),
            })
            if count == 1 or count % 30 == 0:
                print(f"[infer] request={count} server={infer_ms:.1f}ms chunk=50", flush=True)
    except (ConnectionError, OSError, struct.error, ValueError) as exc:
        print(f"[tcp] client disconnected/error {addr}: {exc}", flush=True)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-integration-steps", type=int, default=16)
    parser.add_argument("--integration-method", choices=("euler", "heun"), default="heun")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_path).resolve()
    print(f"[init] env=dinoflow_env model={model_path} device={device}", flush=True)
    policy, pre, post = load_policy(
        model_path,
        device,
        n_action_steps=30,
        num_integration_steps=args.num_integration_steps,
        integration_method=args.integration_method,
    )
    session = DinoFlowSession(policy, pre, post)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(
        f"[serve] listening on {args.host}:{args.port} horizon=50 client_steps=30 "
        f"solver={args.integration_method}/{args.num_integration_steps}",
        flush=True,
    )
    try:
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            handle(conn, addr, session)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        print("[done] server stopped", flush=True)


if __name__ == "__main__":
    main()
