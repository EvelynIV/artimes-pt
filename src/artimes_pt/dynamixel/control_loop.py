from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .control_adapter import HOME_TARGET_RADIANS, PitchYawControlAdapter, PitchYawFeedback
from .contronller import DynamixelConfig

CONTROL_FREQUENCY_HZ = 100.0
CONTROL_PERIOD_SEC = 1.0 / CONTROL_FREQUENCY_HZ
DEFAULT_MAX_CONSECUTIVE_ERRORS = 3
DEFAULT_HOMING_FREQUENCY_HZ = 5.0
DEFAULT_HOMING_PERIOD_SEC = 1.0 / DEFAULT_HOMING_FREQUENCY_HZ
DEFAULT_HOMING_TOLERANCE_RAD = 0.05
DEFAULT_HOMING_STABLE_CYCLES = 3
DEFAULT_HOMING_MAX_DURATION_SEC = 10.0
DEFAULT_HOMING_MAX_CONSECUTIVE_ERRORS = 8


def _validate_target(target_rad: np.ndarray) -> np.ndarray:
    target = np.asarray(target_rad, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"expected target shape (2,), got {target.shape}")
    return target.copy()


class ControlAdapterProtocol(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def write_radians(self, target_radians: np.ndarray) -> None: ...

    def read_feedback(self) -> PitchYawFeedback: ...


@dataclass(frozen=True)
class ControlLoopState:
    is_running: bool
    is_faulted: bool
    latest_target: Optional[np.ndarray]
    last_target: Optional[np.ndarray]
    latest_state: Optional[np.ndarray]
    latest_telemetry: Optional[np.ndarray]
    latest_pitch_out_of_range: Optional[bool]
    last_write_ok: Optional[bool]
    last_telemetry_ok: Optional[bool]
    last_error: Optional[str]
    consecutive_error_count: int
    tick_count: int
    last_tick_time: Optional[float]
    worker_error: Optional[str]


class LatestValueControlLoop:
    """Background control loop with latest-value overwrite semantics."""

    def __init__(
        self,
        adapter: ControlAdapterProtocol | None = None,
        config: DynamixelConfig | None = None,
        period_sec: float = CONTROL_PERIOD_SEC,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        homing_period_sec: float = DEFAULT_HOMING_PERIOD_SEC,
        homing_tolerance_rad: float = DEFAULT_HOMING_TOLERANCE_RAD,
        homing_stable_cycles: int = DEFAULT_HOMING_STABLE_CYCLES,
        homing_max_duration_sec: float = DEFAULT_HOMING_MAX_DURATION_SEC,
        homing_max_consecutive_errors: int = DEFAULT_HOMING_MAX_CONSECUTIVE_ERRORS,
    ) -> None:
        if period_sec <= 0.0:
            raise ValueError(f"period_sec must be > 0, got {period_sec}")
        if max_consecutive_errors <= 0:
            raise ValueError(
                f"max_consecutive_errors must be > 0, got {max_consecutive_errors}"
            )
        if homing_period_sec <= 0.0:
            raise ValueError(f"homing_period_sec must be > 0, got {homing_period_sec}")
        if homing_tolerance_rad < 0.0:
            raise ValueError(f"homing_tolerance_rad must be >= 0, got {homing_tolerance_rad}")
        if homing_stable_cycles <= 0:
            raise ValueError(f"homing_stable_cycles must be > 0, got {homing_stable_cycles}")
        if homing_max_duration_sec <= 0.0:
            raise ValueError(
                f"homing_max_duration_sec must be > 0, got {homing_max_duration_sec}"
            )
        if homing_max_consecutive_errors <= 0:
            raise ValueError(
                "homing_max_consecutive_errors must be > 0, "
                f"got {homing_max_consecutive_errors}"
            )
        if adapter is not None and config is not None:
            raise ValueError("adapter and config cannot be provided at the same time")

        self._adapter = adapter if adapter is not None else PitchYawControlAdapter(config)
        self._period_sec = float(period_sec)
        self._max_consecutive_errors = int(max_consecutive_errors)
        self._homing_period_sec = float(homing_period_sec)
        self._homing_tolerance_rad = float(homing_tolerance_rad)
        self._homing_stable_cycles = int(homing_stable_cycles)
        self._homing_max_duration_sec = float(homing_max_duration_sec)
        self._homing_max_consecutive_errors = int(homing_max_consecutive_errors)
        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_target: Optional[np.ndarray] = None
        self._last_target: Optional[np.ndarray] = None
        self._latest_state: Optional[np.ndarray] = None
        self._latest_telemetry: Optional[np.ndarray] = None
        self._latest_pitch_out_of_range: Optional[bool] = None
        self._homing_active = True
        self._homing_completed = False
        self._homing_started_at: float | None = None
        self._homing_stable_counter = 0
        self._last_write_ok: Optional[bool] = None
        self._last_telemetry_ok: Optional[bool] = None
        self._last_error: Optional[str] = None
        self._consecutive_error_count = 0
        self._is_faulted = False
        self._worker_error: BaseException | None = None
        self._tick_count = 0
        self._last_tick_time: float | None = None

    @property
    def latest_target(self) -> Optional[np.ndarray]:
        with self._state_lock:
            if self._latest_target is None:
                return None
            return self._latest_target.copy()

    @property
    def last_target(self) -> Optional[np.ndarray]:
        with self._state_lock:
            if self._last_target is None:
                return None
            return self._last_target.copy()

    @property
    def latest_state(self) -> Optional[np.ndarray]:
        with self._state_lock:
            if self._latest_state is None:
                return None
            return self._latest_state.copy()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._worker_error = None
            with self._state_lock:
                self._is_faulted = False
                self._homing_active = True
                self._homing_completed = False
                self._homing_started_at = None
                self._homing_stable_counter = 0
            self._thread = threading.Thread(
                target=self._run,
                name="dynamixel-control-loop",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._thread

        if thread is not None:
            thread.join()

        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None

            if self._worker_error is not None:
                error = self._worker_error
                self._worker_error = None
                raise RuntimeError("control loop stopped due to worker error") from error

    def submit_target(self, target_rad: np.ndarray) -> None:
        target = _validate_target(target_rad)
        with self._state_lock:
            self._latest_target = target
            self._homing_active = False

    def get_latest_target(self) -> Optional[np.ndarray]:
        return self.latest_target

    def get_latest_state(self) -> Optional[np.ndarray]:
        return self.latest_state

    def get_latest_telemetry(self) -> Optional[np.ndarray]:
        with self._state_lock:
            if self._latest_telemetry is None:
                return None
            return self._latest_telemetry.copy()

    def get_state(self) -> ControlLoopState:
        with self._state_lock:
            latest_target = None if self._latest_target is None else self._latest_target.copy()
            last_target = None if self._last_target is None else self._last_target.copy()
            latest_state = None if self._latest_state is None else self._latest_state.copy()
            latest_telemetry = (
                None if self._latest_telemetry is None else self._latest_telemetry.copy()
            )
            latest_pitch_out_of_range = self._latest_pitch_out_of_range
            last_write_ok = self._last_write_ok
            last_telemetry_ok = self._last_telemetry_ok
            last_error = self._last_error
            consecutive_error_count = self._consecutive_error_count
            tick_count = self._tick_count
            last_tick_time = self._last_tick_time
            is_faulted = self._is_faulted

        with self._lifecycle_lock:
            is_running = self._thread is not None and self._thread.is_alive()
            worker_error = None if self._worker_error is None else repr(self._worker_error)

        return ControlLoopState(
            is_running=is_running,
            is_faulted=is_faulted,
            latest_target=latest_target,
            last_target=last_target,
            latest_state=latest_state,
            latest_telemetry=latest_telemetry,
            latest_pitch_out_of_range=latest_pitch_out_of_range,
            last_write_ok=last_write_ok,
            last_telemetry_ok=last_telemetry_ok,
            last_error=last_error,
            consecutive_error_count=consecutive_error_count,
            tick_count=tick_count,
            last_tick_time=last_tick_time,
            worker_error=worker_error,
        )

    def _run(self) -> None:
        next_tick = time.monotonic()

        try:
            self._adapter.open()
            while not self._stop_event.is_set():
                command, period_sec, error_threshold, is_homing_step = self._select_operation_for_tick()
                if command is not None:
                    feedback = self._execute_control_step(
                        command,
                        max_consecutive_errors=error_threshold,
                    )
                    if is_homing_step:
                        self._update_homing_progress(feedback)
                self._record_tick()

                next_tick += period_sec
                remaining = next_tick - time.monotonic()
                if remaining > 0.0:
                    self._stop_event.wait(remaining)
                else:
                    next_tick = time.monotonic()
        except BaseException as exc:
            with self._lifecycle_lock:
                self._worker_error = exc
            self._stop_event.set()
        finally:
            self._adapter.close()

    def _select_operation_for_tick(self) -> tuple[Optional[np.ndarray], float, int, bool]:
        with self._state_lock:
            if self._latest_target is not None:
                command = self._latest_target.copy()
                self._last_target = command.copy()
                return command, self._period_sec, self._max_consecutive_errors, False

            if self._last_target is not None:
                return self._last_target.copy(), self._period_sec, self._max_consecutive_errors, False

            if self._homing_active and not self._homing_completed:
                if self._homing_started_at is None:
                    self._homing_started_at = time.monotonic()
                return (
                    HOME_TARGET_RADIANS.copy(),
                    self._homing_period_sec,
                    self._homing_max_consecutive_errors,
                    True,
                )

            return None, self._period_sec, self._max_consecutive_errors, False

    def _update_feedback(
        self,
        present_radians: np.ndarray,
        telemetry: np.ndarray,
        pitch_out_of_range: bool,
    ) -> None:
        with self._state_lock:
            self._latest_state = np.asarray(present_radians, dtype=np.float64).copy()
            self._latest_telemetry = np.asarray(telemetry, dtype=np.float64).copy()
            self._latest_pitch_out_of_range = bool(pitch_out_of_range)

    def _execute_control_step(
        self,
        command: np.ndarray,
        *,
        max_consecutive_errors: int | None = None,
    ) -> Optional[PitchYawFeedback]:
        error_threshold = (
            self._max_consecutive_errors
            if max_consecutive_errors is None
            else int(max_consecutive_errors)
        )
        try:
            self._adapter.write_radians(command)
        except Exception as exc:
            self._record_step_error(
                write_ok=False,
                telemetry_ok=False,
                error=exc,
                phase="write",
                max_consecutive_errors=error_threshold,
            )
            return None

        self._record_write_success()

        try:
            feedback = self._adapter.read_feedback()
        except Exception as exc:
            self._record_step_error(
                write_ok=True,
                telemetry_ok=False,
                error=exc,
                phase="telemetry",
                max_consecutive_errors=error_threshold,
            )
            return None

        self._update_feedback(
            feedback.state_radians,
            feedback.telemetry,
            feedback.pitch_out_of_range,
        )
        self._record_telemetry_success()
        return feedback

    def _record_write_success(self) -> None:
        with self._state_lock:
            self._last_write_ok = True
            self._last_telemetry_ok = None
            self._last_error = None

    def _record_telemetry_success(self) -> None:
        with self._state_lock:
            self._last_write_ok = True
            self._last_telemetry_ok = True
            self._last_error = None
            self._consecutive_error_count = 0

    def _record_step_error(
        self,
        *,
        write_ok: bool,
        telemetry_ok: bool,
        error: Exception,
        phase: str,
        max_consecutive_errors: int,
    ) -> None:
        should_fault = False
        with self._state_lock:
            self._last_write_ok = write_ok
            self._last_telemetry_ok = telemetry_ok
            self._last_error = f"{phase} error: {error!r}"
            self._consecutive_error_count += 1
            should_fault = self._consecutive_error_count >= max_consecutive_errors
            if should_fault:
                self._is_faulted = True

        if should_fault:
            fault_error = RuntimeError(
                f"control loop faulted after {self._consecutive_error_count} consecutive errors; "
                f"last_error={self._last_error}"
            )
            with self._lifecycle_lock:
                self._worker_error = fault_error
            self._stop_event.set()

    def _is_homed(self, feedback: PitchYawFeedback) -> bool:
        if feedback.pitch_out_of_range:
            return False

        error = np.abs(feedback.state_radians - HOME_TARGET_RADIANS)
        return bool(np.all(error <= self._homing_tolerance_rad))

    def _update_homing_progress(self, feedback: Optional[PitchYawFeedback]) -> None:
        with self._state_lock:
            if not self._homing_active or self._homing_completed:
                return

            if feedback is not None and self._is_homed(feedback):
                self._homing_stable_counter += 1
                if self._homing_stable_counter >= self._homing_stable_cycles:
                    self._homing_completed = True
                    self._homing_active = False
                    self._homing_started_at = None
                return

            self._homing_stable_counter = 0
            if (
                self._homing_started_at is not None
                and time.monotonic() - self._homing_started_at >= self._homing_max_duration_sec
            ):
                self._homing_active = False

    def _record_tick(self) -> None:
        with self._state_lock:
            self._tick_count += 1
            self._last_tick_time = time.monotonic()


__all__ = [
    "CONTROL_FREQUENCY_HZ",
    "CONTROL_PERIOD_SEC",
    "DEFAULT_MAX_CONSECUTIVE_ERRORS",
    "DEFAULT_HOMING_FREQUENCY_HZ",
    "DEFAULT_HOMING_MAX_CONSECUTIVE_ERRORS",
    "DEFAULT_HOMING_MAX_DURATION_SEC",
    "DEFAULT_HOMING_PERIOD_SEC",
    "DEFAULT_HOMING_STABLE_CYCLES",
    "DEFAULT_HOMING_TOLERANCE_RAD",
    "ControlLoopState",
    "LatestValueControlLoop",
]
