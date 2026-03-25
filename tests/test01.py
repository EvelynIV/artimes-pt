from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artimes_pt.dynamixel.control_adapter import PITCH_RANGE_RAD
from artimes_pt.dynamixel.control_loop import LatestValueControlLoop
from artimes_pt.dynamixel.contronller import DynamixelConfig


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
        np.array([0.25, 0.00], dtype=np.float64),
        np.array([0.25, -0.80], dtype=np.float64),
        np.array([0.50, 0.00], dtype=np.float64),
        np.array([0.90, 1.20], dtype=np.float64),
        np.array([1.20, 2.80], dtype=np.float64),
        np.array([1.00, 3.00], dtype=np.float64),
        np.array([0.00, 0.00], dtype=np.float64),
    ]
    return [validate_command(command) for command in commands]


def producer_worker(
    loop: LatestValueControlLoop,
    commands: list[np.ndarray],
    interval_sec: float,
) -> None:
    for index, command in enumerate(commands, start=1):
        loop.submit_target(command)
        print(f"[producer {index}] submit command={format_array(command)}")
        if index < len(commands):
            time.sleep(interval_sec)


def print_state(tag: str, loop: LatestValueControlLoop) -> None:
    state = loop.get_state()
    print(tag)
    print(f"latest_target : {format_array(state.latest_target)}")
    print(f"last_target   : {format_array(state.last_target)}")
    print(f"latest_state  : {format_array(state.latest_state)}")
    print(f"telemetry     : {format_array(state.latest_telemetry)}")
    print(f"pitch_oor     : {state.latest_pitch_out_of_range}")
    print(f"last_write_ok : {state.last_write_ok}")
    print(f"last_telem_ok : {state.last_telemetry_ok}")
    print(f"last_error    : {state.last_error}")
    print(f"faulted       : {state.is_faulted}")
    print(f"error_count   : {state.consecutive_error_count}")
    print(f"tick_count    : {state.tick_count}")
    print(f"last_tick     : {state.last_tick_time}")
    print(f"worker_error  : {state.worker_error}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one target per second to the real Dynamixel control loop."
    )
    parser.add_argument("--device", default="COM9")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--max-errors", type=int, default=3)
    args = parser.parse_args()

    config = DynamixelConfig(device_name=args.device, baudrate=args.baudrate, dxl_ids=(1, 2))
    loop = LatestValueControlLoop(
        config=config,
        period_sec=1.0 / args.frequency,
        max_consecutive_errors=args.max_errors,
    )
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
            print_state(f"[observe {index}]", loop)
    finally:
        producer.join()
        try:
            loop.stop()
        finally:
            print_state("[final]", loop)


if __name__ == "__main__":
    main()
