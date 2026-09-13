#!/usr/bin/env python3
"""Minimal DinoFlow robot client.

This is the same 50-predicted/30-executed loop as the mock, but sends the
first 30 actions to the robot.
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

ROBOT_ROOT = os.environ.get("DINOFLOW_ROBOT_ROOT")
if ROBOT_ROOT:
    ROBOT_ROOT_PATH = Path(ROBOT_ROOT).expanduser().resolve()
    sys.path.insert(0, str(ROBOT_ROOT_PATH / "agibot"))
    sys.path.insert(0, str(ROBOT_ROOT_PATH / "agibot_gdk"))

# Reuse the minimal protocol and observation implementation from mock. The
# actual client only changes the action execution step.
from client_mock import (
    FORCE_CALIBRATION_FRAMES,
    MockClient,
    recv_response,
    send_frame,
)  # noqa: E402


class RobotClient(MockClient):
    def __init__(self, host, port, hz, **kwargs):
        # The robot client must fail loudly on a missing or malformed sensor;
        # the mock client keeps its permissive mode for SDK smoke tests.
        kwargs.setdefault("strict_sensors", True)
        super().__init__(host, port, hz, **kwargs)
        self.previous_arm_target = None

    def run(self):
        from wbc_gdk import WbcGdk

        self.wbc = WbcGdk(cameras=["head", "hand_left", "hand_right"])
        time.sleep(1.0)
        sock = socket.create_connection((self.host, self.port), timeout=10)
        self.sock = sock
        print(f"[client] connected to {self.host}:{self.port}", flush=True)
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
                if action is None:
                    next_tick = self._wait_for_tick(next_tick)
                    continue
                # Reuse the arm feedback from this observation. An additional
                # SDK read here would add jitter to the 30 Hz control period.
                arm = np.asarray(self.current_arm, dtype=np.float64).copy()
                if arm.shape != (14,) or not np.isfinite(arm).all():
                    raise RuntimeError(f"invalid arm feedback: shape={arm.shape}")
                raw_arm_target = np.asarray(action[:14], dtype=np.float64)
                if self.previous_arm_target is None:
                    self.previous_arm_target = arm.copy()
                filtered_arm_target = (
                    self.arm_ema_alpha * raw_arm_target
                    + (1.0 - self.arm_ema_alpha) * self.previous_arm_target
                )
                arm_target = np.clip(
                    filtered_arm_target,
                    arm - self.max_arm_step_rad,
                    arm + self.max_arm_step_rad,
                )
                self.previous_arm_target = arm_target.copy()
                self.wbc.move_arm(arm_target.tolist())
                self.wbc.move_hand(action[14:26].tolist())
                next_tick = self._wait_for_tick(next_tick)
        finally:
            sock.close()
            self.wbc.shutdown()

    def _request_async(self, meta, images, previous):
        with self.lock:
            if self.request_inflight:
                return
            self.request_inflight = True
            request_start_exec = self.exec_counter
            predicted_delay = self.inference_delay_steps
        request = dict(meta)
        # This is only the predicted consumed-action delay. The response path
        # measures the actual wall-clock age and consumed action count.
        request["inference_delay"] = predicted_delay
        request["execution_horizon"] = self.rtc_execution_horizon
        request["prev_chunk_left_over"] = previous

        def worker():
            try:
                request_send_time = time.monotonic()
                send_frame(self.sock, request, images)
                response = recv_response(self.sock)
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape != (50, 26) or not np.isfinite(actions).all():
                    raise RuntimeError(f"invalid action chunk {actions.shape}")
                observation_time = float(meta.get("observation_monotonic", request_send_time))
                consumed, age, stale = self._install_action_chunk(
                    actions, request_start_exec, observation_time
                )
                if stale:
                    print(
                        f"[client] dropped stale chunk age={age * 1000.0:.1f}ms "
                        f"consumed={consumed} server={response.get('server_ms', -1):.1f}ms",
                        flush=True,
                    )
                else:
                    print(
                        f"[client] chunk=50 server={response.get('server_ms', -1):.1f}ms "
                        f"age={age * 1000.0:.1f}ms consumed={consumed} "
                        f"rtc={response.get('rtc_enabled')}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[client] request failed: {exc}", flush=True)
            finally:
                with self.lock:
                    self.request_inflight = False

        threading.Thread(target=worker, daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--max-arm-step-rad",
        type=float,
        default=0.02618,
        help="Maximum arm target change per cycle; default 0.02618 rad (~1.5 deg/cycle).",
    )
    parser.add_argument(
        "--arm-ema-alpha",
        type=float,
        default=0.2,
        help="Arm target EMA alpha; 0.2 is smoother, 1.0 disables filtering.",
    )
    parser.add_argument("--rtc", dest="rtc_enabled", action="store_true", default=True)
    parser.add_argument("--no-rtc", dest="rtc_enabled", action="store_false")
    parser.add_argument("--chunk-refresh-steps", type=int, default=20)
    parser.add_argument("--rtc-execution-horizon", type=int, default=20)
    parser.add_argument("--chunk-blend-steps", type=int, default=0)
    parser.add_argument("--inference-delay-steps", type=int, default=3)
    parser.add_argument(
        "--force-calibration-frames",
        type=int,
        default=FORCE_CALIBRATION_FRAMES,
        help="Initial stationary samples for right-wrist force baseline; 0 disables calibration.",
    )
    args = parser.parse_args()
    RobotClient(
        args.host,
        args.port,
        args.hz,
        max_arm_step_rad=args.max_arm_step_rad,
        arm_ema_alpha=args.arm_ema_alpha,
        rtc_enabled=args.rtc_enabled,
        chunk_refresh_steps=args.chunk_refresh_steps,
        rtc_execution_horizon=args.rtc_execution_horizon,
        chunk_blend_steps=args.chunk_blend_steps,
        inference_delay_steps=args.inference_delay_steps,
        force_calibration_frames=args.force_calibration_frames,
    ).run()


if __name__ == "__main__":
    main()
