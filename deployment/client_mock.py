#!/usr/bin/env python3
"""Minimal DinoFlow mock client.

Runs in gdk_env after sourcing agibot/a2d_sdk/env.sh. It reads the real
robot state and cameras, sends one request, receives 50 actions, and prints
the first 30 actions without commanding the robot.
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
from pathlib import Path

import numpy as np
import ujson
from PIL import Image

CHUNK_REFRESH_STEPS = 10
RTC_EXECUTION_HORIZON = 10
CHUNK_BLEND_STEPS = 10
INFERENCE_DELAY_STEPS = 3
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
        chunk_refresh_steps=10,
        rtc_execution_horizon=10,
        chunk_blend_steps=10,
        inference_delay_steps=3,
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
        self.previous_arm_target = None
        self.current_arm = None
        self.sock = None
        self.lock = threading.Lock()
        self.current_chunk = None
        self.current_idx = 0
        self.request_inflight = False
        self.exec_counter = 0

    def observation(self):
        from wbc_gdk import WbcGdk

        if not hasattr(self, "wbc"):
            self.wbc = WbcGdk(cameras=["head", "hand_left", "hand_right"])
            time.sleep(1.0)
        state = self.wbc.read_state(include_images=True)
        arm_feedback = np.asarray(state["arm_joints"], dtype=np.float64).reshape(-1)
        self.current_arm = np.zeros(14, dtype=np.float64)
        self.current_arm[: min(14, arm_feedback.size)] = arm_feedback[:14]
        hand = np.radians(np.asarray(state["hand_joints"], dtype=np.float32))
        joints = np.concatenate([self.current_arm.astype(np.float32), hand])
        full_state = np.zeros(646, dtype=np.float32)
        full_state[: min(26, joints.size)] = joints[:26]
        full_state[26:30] = np.asarray((*state.get("head_joints", (0, 0)), *state.get("waist_joints", (0, 0))), dtype=np.float32)[:4]
        force = np.asarray(state.get("hand_force", []), dtype=np.float32).reshape(-1)
        if force.size == 12:
            full_state[30:42] = force

        images = state.get("images", {})
        for out_key, sdk_key in (("head", "head"), ("left_wrist", "hand_left"), ("right_wrist", "hand_right")):
            if images.get(sdk_key) is not None:
                self.last_images[out_key] = images[sdk_key]
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        encoded = {key: jpeg(self.last_images.get(key, blank)) for key in ("head", "left_wrist", "right_wrist")}
        return {"type": "step_request", "timestamp": time.time(), "state": full_state.tolist()}, encoded

    def run(self):
        from wbc_gdk import WbcGdk  # noqa: F401
        self.sock = socket.create_connection((self.host, self.port), timeout=10)
        print(f"[mock] connected to {self.host}:{self.port}", flush=True)
        try:
            while True:
                meta, images = self.observation()
                with self.lock:
                    need = self.current_chunk is None or self.current_idx >= self.chunk_refresh_steps
                    previous = None
                    if self.rtc_enabled and self.current_chunk is not None and self.current_idx < len(self.current_chunk):
                        previous = self.current_chunk[self.current_idx:].tolist()
                if need and not self.request_inflight:
                    self._request_async(meta, images, previous)
                action = self._next_action()
                if action is not None and self.exec_counter % 30 == 0:
                    print(f"[mock] action[{self.exec_counter}] arm_norm={np.linalg.norm(action[:14]):.3f} hand_norm={np.linalg.norm(action[14:]):.3f}", flush=True)
                time.sleep(self.dt)
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
    parser.add_argument("--chunk-refresh-steps", type=int, default=10)
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--chunk-blend-steps", type=int, default=10)
    parser.add_argument("--inference-delay-steps", type=int, default=3)
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
    ).run()


if __name__ == "__main__":
    main()
