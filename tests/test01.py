from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artimes_pt.dynamixel.control_adapter import (
    PITCH_RANGE_RAD,
    PitchYawControlAdapter,
    PitchYawFeedback,
)
from artimes_pt.dynamixel.control_loop import LatestValueControlLoop
from artimes_pt.dynamixel.contronller import DynamixelConfig


class ObservedPitchYawControlAdapter:
    """Wrap the real adapter and cache the latest command/response for observation."""

    def __init__(self, config: DynamixelConfig) -> None:
        self._adapter = PitchYawControlAdapter(config)
        self._lock = threading.Lock()
        self._last_command: np.ndarray | None = None
        self._last_present_radians: np.ndarray | None = None
        self._last_telemetry: np.ndarray | None = None
        self._write_count = 0

    def open(self) -> None:
        self._adapter.open()

    def close(self) -> None:
        self._adapter.close()

    def write_radians(self, target_radians: np.ndarray) -> None:
        command = np.asarray(target_radians, dtype=np.float64).copy()
        self._adapter.write_radians(command)

        with self._lock:
            self._last_command = command
            self._write_count += 1

    def read_feedback(self) -> PitchYawFeedback:
        feedback = self._adapter.read_feedback()

        with self._lock:
            self._last_present_radians = feedback.state_radians.copy()
            self._last_telemetry = feedback.telemetry.copy()

        return PitchYawFeedback(
            state_radians=feedback.state_radians.copy(),
            telemetry=feedback.telemetry.copy(),
        )

    def get_observation(self) -> dict[str, object]:
        with self._lock:
            return {
                "last_command": None if self._last_command is None else self._last_command.copy(),
                "last_present_radians": (
                    None
                    if self._last_present_radians is None
                    else self._last_present_radians.copy()
                ),
                "last_telemetry": None if self._last_telemetry is None else self._last_telemetry.copy(),
                "write_count": self._write_count,
            }


def format_array(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return np.array2string(value, precision=4, suppress_small=True)


def validate_command(command: np.ndarray) -> np.ndarray:
    target = np.asarray(command, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"expected command shape (2,), got {target.shape}")

    pitch = float(target[0])
    yaw = float(target[1])
    if not 0.0 <= pitch <= PITCH_RANGE_RAD:
        raise ValueError(f"pitch must be within [0, pi/2], got {pitch}")
    if not -np.pi <= yaw < np.pi:
        raise ValueError(f"yaw must be within [-pi, pi), got {yaw}")

    return target.copy()


def build_demo_commands() -> list[np.ndarray]:
    commands = [
        np.array([0.10, -2.40], dtype=np.float64),
        np.array([0.25, -0.80], dtype=np.float64),
        np.array([0.50, 0.00], dtype=np.float64),
        np.array([0.90, 1.20], dtype=np.float64),
        np.array([1.20, 2.80], dtype=np.float64),
    ]
    return [validate_command(command) for command in commands]


def producer_worker(
    loop: LatestValueControlLoop,
    commands: list[np.ndarray],
    interval_sec: float,
) -> None:
    for index, command in enumerate(commands, start=1):
        validated_command = validate_command(command)
        loop.submit_target(validated_command)
        print(f"[producer {index}] submit command={format_array(validated_command)}")
        if index < len(commands):
            time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one target per second to the real Dynamixel control loop."
    )
    parser.add_argument("--device", default="COM9")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    config = DynamixelConfig(device_name=args.device, baudrate=args.baudrate, dxl_ids=(1, 2))
    adapter = ObservedPitchYawControlAdapter(config)
    loop = LatestValueControlLoop(adapter=adapter)
    commands = build_demo_commands()

    producer = threading.Thread(
        target=producer_worker,
        args=(loop, commands, args.interval),
        name="external-command-producer",
        daemon=False,
    )

    loop.start()
    producer.start()

    try:
        for index in range(1, len(commands) + 1):
            time.sleep(args.interval)

            state = loop.get_state()
            observation = adapter.get_observation()
            print(f"[observe {index}]")
            print(f"latest_target : {format_array(state.latest_target)}")
            print(f"last_target   : {format_array(state.last_target)}")
            print(f"last_command  : {format_array(observation['last_command'])}")
            print(f"present_rad   : {format_array(observation['last_present_radians'])}")
            print(f"telemetry     : {format_array(observation['last_telemetry'])}")
            print(f"last_write_ok : {state.last_write_ok}")
            print(f"last_telem_ok : {state.last_telemetry_ok}")
            print(f"last_error    : {state.last_error}")
            print(f"faulted       : {state.is_faulted}")
            print(f"error_count   : {state.consecutive_error_count}")
            print(f"tick_count    : {state.tick_count}")
            print(f"last_tick     : {state.last_tick_time}")
            print(f"write_count   : {observation['write_count']}")
            print(f"worker_error  : {state.worker_error}")
            print("-" * 60)
    finally:
        producer.join()
        loop.stop()

        state = loop.get_state()
        observation = adapter.get_observation()
        print("[final]")
        print(f"is_running    : {state.is_running}")
        print(f"latest_target : {format_array(state.latest_target)}")
        print(f"last_target   : {format_array(state.last_target)}")
        print(f"present_rad   : {format_array(observation['last_present_radians'])}")
        print(f"telemetry     : {format_array(observation['last_telemetry'])}")
        print(f"last_write_ok : {state.last_write_ok}")
        print(f"last_telem_ok : {state.last_telemetry_ok}")
        print(f"last_error    : {state.last_error}")
        print(f"faulted       : {state.is_faulted}")
        print(f"error_count   : {state.consecutive_error_count}")
        print(f"tick_count    : {state.tick_count}")
        print(f"write_count   : {observation['write_count']}")
        print(f"worker_error  : {state.worker_error}")


if __name__ == "__main__":
    main()
