from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .contronller import DualDynamixelController, DynamixelConfig

COUNTS_PER_REV = 4096
HALF_TURN_RAD = math.pi
PITCH_RANGE_RAD = math.pi / 2.0
YAW_ZERO_POSITION = 1024
PITCH_ZERO_POSITION = 1024
PITCH_INDEX = 0
YAW_INDEX = 1
PULSE_PER_RAD = COUNTS_PER_REV / (2.0 * math.pi)
RAD_PER_PULSE = (2.0 * math.pi) / COUNTS_PER_REV


def _normalize_yaw_rad(yaw_rad: float) -> float:
    return ((yaw_rad + math.pi) % (2.0 * math.pi)) - math.pi


def _validate_target_radians(target_radians: np.ndarray) -> np.ndarray:
    radians = np.asarray(target_radians, dtype=np.float64)
    if radians.shape != (2,):
        raise ValueError(f"expected target_radians shape (2,), got {radians.shape}")
    return radians


def pitch_rad_to_position(pitch_rad: float) -> int:
    if not 0.0 <= pitch_rad <= PITCH_RANGE_RAD:
        raise ValueError(f"pitch must be within [0, pi/2], got {pitch_rad}")
    return int(round(PITCH_ZERO_POSITION + pitch_rad * PULSE_PER_RAD))


def yaw_rad_to_position(yaw_rad: float) -> int:
    normalized_yaw = _normalize_yaw_rad(yaw_rad)
    return int(round(YAW_ZERO_POSITION + normalized_yaw * PULSE_PER_RAD)) % COUNTS_PER_REV


def position_to_pitch_rad(position: int) -> float:
    pitch_rad = (int(position) - PITCH_ZERO_POSITION) * RAD_PER_PULSE
    if pitch_rad < 0.0 or pitch_rad > PITCH_RANGE_RAD:
        raise ValueError(
            f"pitch position out of calibrated range [0, pi/2]: position={position}"
        )
    return pitch_rad


def position_to_yaw_rad(position: int) -> float:
    wrapped_position = int(position) % COUNTS_PER_REV
    yaw_rad = (wrapped_position - YAW_ZERO_POSITION) * RAD_PER_PULSE
    return _normalize_yaw_rad(yaw_rad)


def pitch_yaw_rad_to_positions(target_radians: np.ndarray) -> np.ndarray:
    radians = _validate_target_radians(target_radians)
    return np.array(
        [
            pitch_rad_to_position(float(radians[PITCH_INDEX])),
            yaw_rad_to_position(float(radians[YAW_INDEX])),
        ],
        dtype=np.int64,
    )


def positions_to_pitch_yaw_rad(positions: np.ndarray) -> np.ndarray:
    motor_positions = np.asarray(positions, dtype=np.int64)
    if motor_positions.shape != (2,):
        raise ValueError(f"expected positions shape (2,), got {motor_positions.shape}")

    return np.array(
        [
            position_to_pitch_rad(int(motor_positions[PITCH_INDEX])),
            position_to_yaw_rad(int(motor_positions[YAW_INDEX])),
        ],
        dtype=np.float64,
    )


def rad_command_stream_to_position_stream(
    rad_command_stream: Iterable[np.ndarray],
) -> Iterator[np.ndarray]:
    for target_radians in rad_command_stream:
        yield pitch_yaw_rad_to_positions(target_radians)


@dataclass(frozen=True)
class PitchYawCommand:
    pitch: float
    yaw: float

    def as_array(self) -> np.ndarray:
        return np.array([self.pitch, self.yaw], dtype=np.float64)


class PitchYawControlAdapter:
    """Accept pitch/yaw radians and forward motor position commands to the controller."""

    def __init__(self, config: DynamixelConfig | None = None):
        self.controller = DualDynamixelController(config)

    def open(self) -> None:
        self.controller.open()

    def close(self) -> None:
        self.controller.close()

    def __enter__(self) -> "PitchYawControlAdapter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def write_radians(self, target_radians: np.ndarray | PitchYawCommand) -> None:
        command = (
            target_radians.as_array()
            if isinstance(target_radians, PitchYawCommand)
            else target_radians
        )
        self.controller.write_positions(pitch_yaw_rad_to_positions(command))

    def write_and_read_radians(
        self, target_radians: np.ndarray | PitchYawCommand
    ) -> tuple[np.ndarray, np.ndarray]:
        command = (
            target_radians.as_array()
            if isinstance(target_radians, PitchYawCommand)
            else target_radians
        )
        telemetry = self.controller.write_and_read(pitch_yaw_rad_to_positions(command))
        present_radians = positions_to_pitch_yaw_rad(telemetry[:, 0].astype(np.int64))
        return present_radians, telemetry

    def stream_radians(
        self, rad_command_stream: Iterable[np.ndarray | PitchYawCommand]
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for target_radians in rad_command_stream:
            yield self.write_and_read_radians(target_radians)


__all__ = [
    "COUNTS_PER_REV",
    "PITCH_ZERO_POSITION",
    "YAW_ZERO_POSITION",
    "PitchYawCommand",
    "PitchYawControlAdapter",
    "pitch_rad_to_position",
    "yaw_rad_to_position",
    "position_to_pitch_rad",
    "position_to_yaw_rad",
    "pitch_yaw_rad_to_positions",
    "positions_to_pitch_yaw_rad",
    "rad_command_stream_to_position_stream",
]
