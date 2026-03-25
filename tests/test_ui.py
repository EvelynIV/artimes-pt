from __future__ import annotations

import argparse
import atexit
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required for tests/test_ui.py. Install it with: pip install opencv-python"
    ) from exc

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "Gradio is required for tests/test_ui.py. Install it with: pip install gradio"
    ) from exc

from artimes_pt.dynamixel.control_adapter import PITCH_RANGE_RAD
from artimes_pt.dynamixel.control_loop import LatestValueControlLoop
from artimes_pt.dynamixel.contronller import DynamixelConfig

DEFAULT_REFRESH_SEC = 0.10
DEFAULT_WAIT_TIMEOUT_SEC = 2.0
DEFAULT_BAUDRATE = 57600
DEFAULT_PITCH_DEG = 0.0
DEFAULT_YAW_DEG = 0.0
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 560
TARGET_COLOR = (70, 90, 240)
CURRENT_COLOR = (70, 180, 80)
AXIS_COLOR = (180, 180, 180)
TEXT_COLOR = (40, 40, 40)
BACKGROUND_COLOR = (248, 248, 244)
PANEL_COLOR = (255, 255, 255)


def normalize_yaw_rad(yaw_rad: float) -> float:
    return ((yaw_rad + math.pi) % (2.0 * math.pi)) - math.pi


def wait_for_initial_state(
    loop: LatestValueControlLoop,
    timeout_sec: float,
    poll_interval_sec: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        state = loop.get_state()
        if state.latest_state is not None:
            return True
        if state.worker_error is not None:
            return False
        time.sleep(poll_interval_sec)
    return False


def format_array(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return np.array2string(value, precision=4, suppress_small=True)


def build_target_rad(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    pitch_rad = np.deg2rad(float(pitch_deg))
    yaw_rad = normalize_yaw_rad(np.deg2rad(float(yaw_deg)))
    if not 0.0 <= pitch_rad <= PITCH_RANGE_RAD:
        raise ValueError(
            f"pitch must be within [0, {np.rad2deg(PITCH_RANGE_RAD):.2f}] deg, got {pitch_deg}"
        )
    return np.array([pitch_rad, yaw_rad], dtype=np.float64)


def pitch_yaw_to_direction(target_rad: np.ndarray) -> np.ndarray:
    pitch_rad = float(target_rad[0])
    yaw_rad = float(target_rad[1])
    cos_pitch = math.cos(pitch_rad)
    return np.array(
        [
            cos_pitch * math.cos(yaw_rad),
            cos_pitch * math.sin(yaw_rad),
            math.sin(pitch_rad),
        ],
        dtype=np.float64,
    )


def _draw_arrow(
    canvas: np.ndarray,
    origin: tuple[int, int],
    vector_xy: np.ndarray | None,
    color: tuple[int, int, int],
    *,
    axis_scale: int,
    label: str,
    label_offset_y: int,
) -> None:
    if vector_xy is None:
        cv2.putText(
            canvas,
            f"{label}: unavailable",
            (origin[0] - 110, origin[1] + label_offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        return

    end_x = int(round(origin[0] + float(vector_xy[0]) * axis_scale))
    end_y = int(round(origin[1] - float(vector_xy[1]) * axis_scale))
    cv2.arrowedLine(
        canvas,
        origin,
        (end_x, end_y),
        color,
        3,
        cv2.LINE_AA,
        tipLength=0.10,
    )
    cv2.putText(
        canvas,
        label,
        (origin[0] - 40, origin[1] + label_offset_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_projection_panel(
    canvas: np.ndarray,
    *,
    rect: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    target_xy: np.ndarray | None,
    current_xy: np.ndarray | None,
) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_COLOR, -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (220, 220, 220), 1)

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2 + 15
    axis_scale = min(right - left, bottom - top) // 3

    cv2.putText(
        canvas,
        title,
        (left + 20, top + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        canvas,
        (left + 30, center_y),
        (right - 30, center_y),
        AXIS_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (center_x, top + 60),
        (center_x, bottom - 30),
        AXIS_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, (center_x, center_y), axis_scale, AXIS_COLOR, 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        x_label,
        (right - 64, center_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        y_label,
        (center_x + 8, top + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    _draw_arrow(
        canvas,
        (center_x, center_y),
        target_xy,
        TARGET_COLOR,
        axis_scale=axis_scale,
        label="target",
        label_offset_y=36,
    )
    _draw_arrow(
        canvas,
        (center_x, center_y),
        current_xy,
        CURRENT_COLOR,
        axis_scale=axis_scale,
        label="current",
        label_offset_y=58,
    )


def render_direction_image(
    target_rad: np.ndarray,
    current_rad: np.ndarray | None,
) -> np.ndarray:
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
    target_vec = pitch_yaw_to_direction(target_rad)
    current_vec = None if current_rad is None else pitch_yaw_to_direction(current_rad)

    _draw_projection_panel(
        canvas,
        rect=(30, 80, 520, 520),
        title="Top View (X-Y)",
        x_label="+X",
        y_label="+Y",
        target_xy=target_vec[[0, 1]],
        current_xy=None if current_vec is None else current_vec[[0, 1]],
    )
    _draw_projection_panel(
        canvas,
        rect=(560, 80, 1050, 520),
        title="Side View (X-Z)",
        x_label="+X",
        y_label="+Z",
        target_xy=target_vec[[0, 2]],
        current_xy=None if current_vec is None else current_vec[[0, 2]],
    )

    cv2.putText(
        canvas,
        "Dynamixel Pitch/Yaw Direction Debug UI",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    target_pitch_deg = math.degrees(float(target_rad[0]))
    target_yaw_deg = math.degrees(float(target_rad[1]))
    cv2.putText(
        canvas,
        f"Target pitch={target_pitch_deg:6.2f} deg  yaw={target_yaw_deg:7.2f} deg",
        (30, 555),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        TARGET_COLOR,
        2,
        cv2.LINE_AA,
    )

    if current_rad is None:
        current_text = "Current pitch/yaw unavailable"
    else:
        current_text = (
            f"Current pitch={math.degrees(float(current_rad[0])):6.2f} deg  "
            f"yaw={math.degrees(float(current_rad[1])):7.2f} deg"
        )

    cv2.putText(
        canvas,
        current_text,
        (420, 555),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        CURRENT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def build_status_text(loop: LatestValueControlLoop, target_rad: np.ndarray) -> str:
    state = loop.get_state()
    target_deg = np.rad2deg(target_rad)
    current_deg = None if state.latest_state is None else np.rad2deg(state.latest_state)
    current_vec = None if state.latest_state is None else pitch_yaw_to_direction(state.latest_state)
    target_vec = pitch_yaw_to_direction(target_rad)

    lines = [
        f"target(deg): pitch={target_deg[0]:.2f}, yaw={target_deg[1]:.2f}",
        f"target(vec): {format_array(target_vec)}",
    ]
    if current_deg is None:
        lines.append("current(deg): unavailable")
        lines.append("current(vec): unavailable")
    else:
        lines.append(f"current(deg): pitch={current_deg[0]:.2f}, yaw={current_deg[1]:.2f}")
        lines.append(f"current(vec): {format_array(current_vec)}")

    lines.extend(
        [
            "",
            f"is_running: {state.is_running}",
            f"is_faulted: {state.is_faulted}",
            f"pitch_out_of_range: {state.latest_pitch_out_of_range}",
            f"last_write_ok: {state.last_write_ok}",
            f"last_telemetry_ok: {state.last_telemetry_ok}",
            f"consecutive_error_count: {state.consecutive_error_count}",
            f"tick_count: {state.tick_count}",
            f"last_error: {state.last_error}",
            f"worker_error: {state.worker_error}",
            "",
            f"latest_target(rad): {format_array(state.latest_target)}",
            f"last_target(rad): {format_array(state.last_target)}",
            f"latest_state(rad): {format_array(state.latest_state)}",
            f"latest_telemetry: {format_array(state.latest_telemetry)}",
        ]
    )
    return "\n".join(lines)


class MotorUiApp:
    def __init__(
        self,
        *,
        config: DynamixelConfig,
        frequency_hz: float,
        max_errors: int,
        wait_timeout_sec: float,
    ) -> None:
        self.loop = LatestValueControlLoop(
            config=config,
            period_sec=1.0 / frequency_hz,
            max_consecutive_errors=max_errors,
        )
        self.wait_timeout_sec = float(wait_timeout_sec)
        self.target_rad = build_target_rad(DEFAULT_PITCH_DEG, DEFAULT_YAW_DEG)
        self._closed = False

    def start(self) -> None:
        self.loop.start()
        if wait_for_initial_state(self.loop, timeout_sec=self.wait_timeout_sec):
            state = self.loop.get_state()
            if state.latest_state is not None:
                self.target_rad = state.latest_state.copy()
            return

        state = self.loop.get_state()
        if state.worker_error is not None:
            raise RuntimeError(
                f"control loop failed during startup: {state.worker_error}"
            )

        raise RuntimeError("control loop startup timed out before any initial feedback arrived")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.loop.stop()
        except Exception as exc:
            print(f"[shutdown] failed to stop control loop cleanly: {exc!r}", file=sys.stderr)

    def submit_target(self, pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, str]:
        self.target_rad = build_target_rad(pitch_deg, yaw_deg)
        self.loop.submit_target(self.target_rad)
        return self.refresh()

    def refresh(self) -> tuple[np.ndarray, str]:
        current_rad = self.loop.get_latest_state()
        image = render_direction_image(self.target_rad, current_rad)
        status = build_status_text(self.loop, self.target_rad)
        return image, status

    def initial_slider_values(self) -> tuple[float, float]:
        return (
            math.degrees(float(self.target_rad[0])),
            math.degrees(float(self.target_rad[1])),
        )


def build_demo(app: MotorUiApp, refresh_sec: float):
    initial_pitch_deg, initial_yaw_deg = app.initial_slider_values()

    with gr.Blocks(title="Dynamixel UI Test") as demo:
        gr.Markdown(
            """
            # Dynamixel Pitch/Yaw UI Test
            Red arrows show the target direction. Green arrows show the current motor feedback.
            The left panel is the `X-Y` top view. The right panel is the `X-Z` side view.
            """
        )

        with gr.Row():
            pitch_slider = gr.Slider(
                minimum=0.0,
                maximum=float(np.rad2deg(PITCH_RANGE_RAD)),
                value=initial_pitch_deg,
                step=0.1,
                label="Pitch (deg)",
            )
            yaw_slider = gr.Slider(
                minimum=-180.0,
                maximum=180.0,
                value=initial_yaw_deg,
                step=0.1,
                label="Yaw (deg)",
            )

        image = gr.Image(label="Direction Vectors", type="numpy")
        status = gr.Textbox(label="Loop Status", lines=16, interactive=False)

        for slider in (pitch_slider, yaw_slider):
            if hasattr(slider, "input"):
                slider.input(
                    fn=app.submit_target,
                    inputs=[pitch_slider, yaw_slider],
                    outputs=[image, status],
                )
            else:
                slider.change(
                    fn=app.submit_target,
                    inputs=[pitch_slider, yaw_slider],
                    outputs=[image, status],
                )

        demo.load(fn=app.refresh, outputs=[image, status])

        if hasattr(gr, "Timer"):
            timer = gr.Timer(value=refresh_sec)
            timer.tick(fn=app.refresh, outputs=[image, status])
        else:
            demo.load(fn=app.refresh, outputs=[image, status], every=refresh_sec)

        if hasattr(demo, "unload"):
            demo.unload(fn=app.close)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio UI for manually dragging pitch/yaw sliders and observing motor feedback."
    )
    parser.add_argument("--device", default="COM9")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--wait-timeout", type=float, default=DEFAULT_WAIT_TIMEOUT_SEC)
    parser.add_argument("--refresh-sec", type=float, default=DEFAULT_REFRESH_SEC)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    config = DynamixelConfig(
        device_name=args.device,
        baudrate=args.baudrate,
        dxl_ids=(1, 2),
    )
    app = MotorUiApp(
        config=config,
        frequency_hz=args.frequency,
        max_errors=args.max_errors,
        wait_timeout_sec=args.wait_timeout,
    )
    app.start()
    atexit.register(app.close)

    demo = build_demo(app, refresh_sec=args.refresh_sec)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
