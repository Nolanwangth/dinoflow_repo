#!/usr/bin/env python3
"""Minimal DinoFlow mock client.

Runs in gdk_env after sourcing agibot/a2d_sdk/env.sh. It reads the real
robot state and cameras, sends requests for 50-action chunks, and prints
actions without commanding the robot.
"""
from __future__ import annotations

import argparse
import io
import os
import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import ujson
from PIL import Image

CHUNK_REFRESH_STEPS = 20
RTC_EXECUTION_HORIZON = 20
CHUNK_BLEND_STEPS = 0
INFERENCE_DELAY_STEPS = 3
TACTILE_REGION_LENGTHS = (35, 60, 60, 60, 32, 55)
ROBOT_ROOT = os.environ.get("DINOFLOW_ROBOT_ROOT")
if ROBOT_ROOT:
    ROBOT_ROOT_PATH = Path(ROBOT_ROOT).expanduser().resolve()
    sys.path.insert(0, str(ROBOT_ROOT_PATH / "agibot"))
    sys.path.insert(0, str(ROBOT_ROOT_PATH / "agibot_gdk"))


def send_frame(sock, meta, images):
    meta_raw = ujson.dumps(meta).encode()
    parts = [struct.pack("B", len(images))]
    for key, raw in images.items():
        key_raw = key.encode("ascii")
        parts += [struct.pack("B", len(key_raw)), key_raw, struct.pack(">I", len(raw)), raw]
    body = meta_raw + b"".join(parts)
    sock.sendall(struct.pack(">I", len(body) + 4) + struct.pack(">I", len(meta_raw)) + body)


def recv_exactly(sock, n):
    data = bytearray(n)
    view = memoryview(data)
    while n:
        got = sock.recv_into(view, n)
        if not got:
            raise ConnectionError("server closed connection")
        view = view[got:]
        n -= got
    return bytes(data)


def recv_response(sock):
    n = struct.unpack(">I", recv_exactly(sock, 4))[0]
    return ujson.loads(recv_exactly(sock, n))


def jpeg(image):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


class MockClient:
    def __init__(
        self,
        host,
        port,
        hz,
        max_arm_step_rad=0.02618,
        arm_ema_alpha=0.2,
        rtc_enabled=True,
        chunk_refresh_steps=CHUNK_REFRESH_STEPS,
        rtc_execution_horizon=RTC_EXECUTION_HORIZON,
        chunk_blend_steps=CHUNK_BLEND_STEPS,
        inference_delay_steps=INFERENCE_DELAY_STEPS,
        strict_sensors=False,
    ):
        self.host, self.port, self.dt = host, port, 1.0 / hz
        self.last_images = {}
        self.max_arm_step_rad = max_arm_step_rad
        if not 0.0 < arm_ema_alpha <= 1.0:
            raise ValueError("arm_ema_alpha must be in (0, 1]")
        self.arm_ema_alpha = arm_ema_alpha
        self.rtc_enabled = rtc_enabled
        self.chunk_refresh_steps = max(1, int(chunk_refresh_steps))
        self.rtc_execution_horizon = max(0, int(rtc_execution_horizon))
        self.chunk_blend_steps = max(0, int(chunk_blend_steps))
        self.inference_delay_steps = max(0, int(inference_delay_steps))
        self.strict_sensors = bool(strict_sensors)
        self.previous_arm_target = None
        self.current_arm = None
        self.sock = None
        self.lock = threading.Lock()
        self.current_chunk = None
        self.current_idx = 0
        self.request_inflight = False
        self.exec_counter = 0
        self.last_observation_time = None
        # Keep a little more than six raw samples so an irregular host loop
        # can still construct the six states at nominal 30 Hz offsets.
        self.state_samples = deque(maxlen=32)

    def _flatten_tactile(self, value, sensor_name: str) -> np.ndarray:
        """Flatten six GDK finger regions into the 302-value hand layout."""
        if value is None:
            if self.strict_sensors:
                raise RuntimeError(f"missing {sensor_name} tactile data")
            return np.zeros(302, dtype=np.float32)
        parts = [np.asarray(region, dtype=np.float32).reshape(-1) for region in value]
        if self.strict_sensors:
            if len(parts) != len(TACTILE_REGION_LENGTHS):
                raise RuntimeError(
                    f"{sensor_name} tactile region count={len(parts)}, "
                    f"expected {len(TACTILE_REGION_LENGTHS)}"
                )
            for index, (part, expected) in enumerate(zip(parts, TACTILE_REGION_LENGTHS, strict=True)):
                if part.size != expected:
                    raise RuntimeError(
                        f"{sensor_name} tactile region {index} length={part.size}, expected {expected}"
                    )
                if not np.isfinite(part).all():
                    raise RuntimeError(f"{sensor_name} tactile region {index} contains NaN or Inf")
        flattened = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        if self.strict_sensors and flattened.size != 302:
            raise RuntimeError(f"{sensor_name} tactile length={flattened.size}, expected 302")
        output = np.zeros(302, dtype=np.float32)
        output[: min(output.size, flattened.size)] = flattened[: output.size]
        return output

    def _encode_images(self) -> dict[str, bytes]:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        missing = [key for key in ("head", "left_wrist", "right_wrist") if key not in self.last_images]
        if self.strict_sensors and missing:
            raise RuntimeError(f"missing camera frames: {missing}")
        return {key: jpeg(self.last_images.get(key, blank)) for key in ("head", "left_wrist", "right_wrist")}

    def _resampled_state_history(self, now: float) -> tuple[np.ndarray, list[float]]:
        samples = tuple(self.state_samples)
        if not samples:
            raise RuntimeError("state history is empty")
        target_times = [now - (5 - index) * self.dt for index in range(6)]
        history = []
        for target in target_times:
            selected = samples[0][1]
            for sample_time, sample in reversed(samples):
                if sample_time <= target:
                    selected = sample
                    break
            history.append(selected)
        return np.stack(history, axis=0), target_times

    def observation(self, read_images: bool = True, encode_images: bool = True):
        from wbc_gdk import WbcGdk

        if not hasattr(self, "wbc"):
            self.wbc = WbcGdk(cameras=["head", "hand_left", "hand_right"])
            time.sleep(1.0)
        state = self.wbc.read_state(include_images=read_images)
        arm_feedback = np.asarray(state["arm_joints"], dtype=np.float64).reshape(-1)
        if self.strict_sensors and arm_feedback.size != 14:
            raise RuntimeError(f"arm_joints length={arm_feedback.size}, expected 14")
        if not np.isfinite(arm_feedback).all():
            raise RuntimeError("arm_joints contains NaN or Inf")
        self.current_arm = np.zeros(14, dtype=np.float64)
        self.current_arm[: min(14, arm_feedback.size)] = arm_feedback[:14]
        hand_feedback = np.asarray(state["hand_joints"], dtype=np.float32).reshape(-1)
        if self.strict_sensors and hand_feedback.size != 12:
            raise RuntimeError(f"hand_joints length={hand_feedback.size}, expected 12")
        if not np.isfinite(hand_feedback).all():
            raise RuntimeError("hand_joints contains NaN or Inf")
        hand = np.radians(hand_feedback)
        joints = np.concatenate([self.current_arm.astype(np.float32), hand])
        full_state = np.zeros(646, dtype=np.float32)
        full_state[: min(26, joints.size)] = joints[:26]
        if self.strict_sensors and ("head_joints" not in state or "waist_joints" not in state):
            raise RuntimeError("missing head_joints or waist_joints")
        auxiliary = np.asarray(
            (*state.get("head_joints", (0, 0)), *state.get("waist_joints", (0, 0))),
            dtype=np.float32,
        ).reshape(-1)
        if self.strict_sensors and auxiliary.size != 4:
            raise RuntimeError(f"head/waist joints length={auxiliary.size}, expected 4")
        if not np.isfinite(auxiliary).all():
            raise RuntimeError("head/waist joints contains NaN or Inf")
        full_state[26:30] = auxiliary[:4]
        force = np.asarray(state.get("hand_force", []), dtype=np.float32).reshape(-1)
        if self.strict_sensors and force.size != 12:
            raise RuntimeError(f"hand_force length={force.size}, expected 12")
        if not np.isfinite(force).all():
            raise RuntimeError("hand_force contains NaN or Inf")
        if force.size == 12:
            full_state[30:42] = force
        tactile = np.concatenate(
            (
                self._flatten_tactile(state.get("tactile_left"), "left"),
                self._flatten_tactile(state.get("tactile_right"), "right"),
            )
        )
        full_state[42:646] = tactile[:604]

        now = time.monotonic()
        observation_interval = (
            None if self.last_observation_time is None else now - self.last_observation_time
        )
        self.last_observation_time = now
        self.state_samples.append((now, full_state.copy()))
        history, history_timestamps = self._resampled_state_history(now)

        if read_images:
            images = state.get("images", {})
            required = {"head": "head", "left_wrist": "hand_left", "right_wrist": "hand_right"}
            missing = [sdk_key for sdk_key in required.values() if images.get(sdk_key) is None]
            if self.strict_sensors and missing:
                raise RuntimeError(f"missing camera frames: {missing}")
            for out_key, sdk_key in required.items():
                if images.get(sdk_key) is not None:
                    self.last_images[out_key] = images[sdk_key]
        encoded = self._encode_images() if encode_images else {}
        return {
            "type": "step_request",
            "timestamp": time.time(),
            "observation_monotonic": now,
            "observation_interval": observation_interval,
            "state_timestamps": history_timestamps,
            "state": history.tolist(),
        }, encoded

    def _request_is_due(self) -> bool:
        with self.lock:
            return (
                (self.current_chunk is None or self.current_idx >= self.chunk_refresh_steps)
                and not self.request_inflight
            )

    def _wait_for_tick(self, next_tick: float) -> float:
        next_tick += self.dt
        now = time.monotonic()
        remaining = next_tick - now
        if remaining > 0:
            time.sleep(remaining)
        elif now - next_tick > self.dt:
            # A slow read/inference cycle missed more than one tick. Restart
            # from the current time instead of spinning to catch up.
            next_tick = now
        return next_tick

    def run(self):
        from wbc_gdk import WbcGdk  # noqa: F401
        self.sock = socket.create_connection((self.host, self.port), timeout=10)
        print(f"[mock] connected to {self.host}:{self.port}", flush=True)
        next_tick = time.monotonic()
        try:
            while True:
                request_due = self._request_is_due()
                meta, images = self.observation(
                    read_images=request_due,
                    encode_images=request_due,
                )
                with self.lock:
                    need = self.current_chunk is None or self.current_idx >= self.chunk_refresh_steps
                    previous = None
                    if self.rtc_enabled and self.current_chunk is not None and self.current_idx < len(self.current_chunk):
                        previous = self.current_chunk[self.current_idx:].tolist()
                if need and not self.request_inflight:
                    if not images:
                        images = self._encode_images()
                    self._request_async(meta, images, previous)
                action = self._next_action()
                if action is not None and self.exec_counter % 30 == 0:
                    print(f"[mock] action[{self.exec_counter}] arm_norm={np.linalg.norm(action[:14]):.3f} hand_norm={np.linalg.norm(action[14:]):.3f}", flush=True)
                next_tick = self._wait_for_tick(next_tick)
        finally:
            self.sock.close()
            if hasattr(self, "wbc"):
                self.wbc.shutdown()

    def _request_async(self, meta, images, previous):
        self.request_inflight = True
        request_start = self.exec_counter
        request = dict(meta)
        # At 30 Hz the normal warmed-up server latency is about 2--3 control
        # ticks.  RTC uses this to align the old prefix with the time at which
        # the new chunk becomes available.
        request["inference_delay"] = self.inference_delay_steps
        request["execution_horizon"] = self.rtc_execution_horizon
        request["prev_chunk_left_over"] = previous

        def worker():
            try:
                send_frame(self.sock, request, images)
                response = recv_response(self.sock)
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape != (50, 26) or not np.isfinite(actions).all():
                    raise RuntimeError(f"invalid action chunk {actions.shape}")
                delay = max(0, self.exec_counter - request_start)
                with self.lock:
                    # Use the measured delay as the prediction for the next
                    # request. The first request uses the CLI warm-up value;
                    # subsequent RTC calls follow the actual 30 Hz pipeline.
                    self.inference_delay_steps = min(delay, 49)
                    new_chunk = actions[min(delay, 49):]
                    old_chunk = None
                    if self.current_chunk is not None and self.current_idx < len(self.current_chunk):
                        old_chunk = self.current_chunk[self.current_idx:].copy()
                    if old_chunk is not None and len(old_chunk) > 0:
                        overlap = min(self.chunk_blend_steps, len(old_chunk), len(new_chunk))
                        # Keep the currently executing trajectory initially,
                        # then hand control to the new observation-conditioned
                        # chunk over 10 ticks.
                        w = np.linspace(0.15, 1.0, overlap, dtype=np.float32)[:, None]
                        new_chunk[:overlap] = old_chunk[:overlap] * (1.0 - w) + new_chunk[:overlap] * w
                    self.current_chunk = new_chunk
                    self.current_idx = 0
                print(f"[mock] chunk=50 server={response.get('server_ms', -1):.1f}ms delay={delay} rtc={response.get('rtc_enabled')}", flush=True)
            except Exception as exc:
                print(f"[mock] request failed: {exc}", flush=True)
            finally:
                self.request_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def _next_action(self):
        with self.lock:
            if self.current_chunk is None or self.current_idx >= len(self.current_chunk):
                return None
            action = self.current_chunk[self.current_idx].copy()
            self.current_idx += 1
            self.exec_counter += 1
            return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-arm-step-rad", type=float, default=0.02618)
    parser.add_argument("--arm-ema-alpha", type=float, default=0.2)
    parser.add_argument("--rtc", dest="rtc_enabled", action="store_true", default=True)
    parser.add_argument("--no-rtc", dest="rtc_enabled", action="store_false")
    parser.add_argument("--chunk-refresh-steps", type=int, default=20)
    parser.add_argument("--rtc-execution-horizon", type=int, default=20)
    parser.add_argument("--chunk-blend-steps", type=int, default=0)
    parser.add_argument("--inference-delay-steps", type=int, default=3)
    parser.add_argument("--strict-sensors", action="store_true")
    args = parser.parse_args()
    MockClient(
        args.host,
        args.port,
        args.hz,
        args.max_arm_step_rad,
        args.arm_ema_alpha,
        args.rtc_enabled,
        args.chunk_refresh_steps,
        args.rtc_execution_horizon,
        args.chunk_blend_steps,
        args.inference_delay_steps,
        args.strict_sensors,
    ).run()


if __name__ == "__main__":
    main()
