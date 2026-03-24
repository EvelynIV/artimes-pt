import math
import time
from collections.abc import Iterable, Iterator

import numpy as np

from contronller import DualDynamixelController, DynamixelConfig

COUNTS_PER_REV = 4096
ZERO_PULSE = 1024
DEFAULT_START_PULSES = np.array([1024, 1024], dtype=np.int64)

RAD_PER_PULSE = 2.0 * math.pi / COUNTS_PER_REV
PULSE_PER_RAD = COUNTS_PER_REV / (2.0 * math.pi)

def pulse_to_rad(pulse: int) -> float:
    return (pulse - ZERO_PULSE) * RAD_PER_PULSE

def rad_to_pulse(rad: float) -> int:
    return round(ZERO_PULSE + rad * PULSE_PER_RAD)


def dual_rad_to_pulse(target_rads: np.ndarray) -> np.ndarray:
    rads = np.asarray(target_rads, dtype=np.float64)
    if rads.shape != (2,):
        raise ValueError(f"expected target_rads shape (2,), got {rads.shape}")

    return np.rint(ZERO_PULSE + rads * PULSE_PER_RAD).astype(np.int64)


def dual_pulse_to_rad(target_pulses: np.ndarray) -> np.ndarray:
    pulses = np.asarray(target_pulses, dtype=np.int64)
    if pulses.shape != (2,):
        raise ValueError(f"expected target_pulses shape (2,), got {pulses.shape}")

    return (pulses - ZERO_PULSE).astype(np.float64) * RAD_PER_PULSE


def rad_stream_to_pulse_stream(rad_stream: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
    for target_rads in rad_stream:
        yield dual_rad_to_pulse(target_rads)


def fake_inference_stream() -> Iterator[np.ndarray]:
    center_rads = dual_pulse_to_rad(DEFAULT_START_PULSES)
    amplitude_rads = np.array([0.45, 0.35], dtype=np.float64)
    phase = 0.0

    while True:
        offset_rads = np.array(
            [
                amplitude_rads[0] * math.sin(phase),
                amplitude_rads[1] * math.cos(phase),
            ],
            dtype=np.float64,
        )
        yield center_rads + offset_rads
        phase += 0.08


def timed_rad_stream(duration_sec: float, interval_sec: float = 0.03) -> Iterator[np.ndarray]:
    deadline = math.inf if duration_sec <= 0 else duration_sec
    start_time = None

    for target_rads in fake_inference_stream():
        if start_time is None:
            start_time = time.monotonic()

        if time.monotonic() - start_time >= deadline:
            break

        yield target_rads
        time.sleep(interval_sec)


def run_fake_dual_motor_demo(
    duration_sec: float = 10.0,
    device_name: str = "COM9",
    baudrate: int = 57600,
    dxl_ids: tuple[int, int] = (1, 2),
) -> None:
    config = DynamixelConfig(device_name=device_name, baudrate=baudrate, dxl_ids=dxl_ids)

    with DualDynamixelController(config) as controller:
        controller.write_positions(DEFAULT_START_PULSES)
        print(f"startup_pulse={DEFAULT_START_PULSES.tolist()}")

        for target_rads in timed_rad_stream(duration_sec):
            target_pulses = dual_rad_to_pulse(target_rads)
            telemetry = controller.write_and_read(target_pulses)
            present_pulses = telemetry[:, 0].astype(np.int64)
            present_rads = dual_pulse_to_rad(present_pulses)
            print(
                f"target_rad={np.round(target_rads, 4).tolist()} "
                f"target_pulse={target_pulses.tolist()} "
                f"present_pulse={present_pulses.tolist()} "
                f"present_rad={np.round(present_rads, 4).tolist()} "
                f"telemetry={telemetry.tolist()}"
            )


if __name__ == "__main__":
    run_fake_dual_motor_demo()
