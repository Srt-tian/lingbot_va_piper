"""Agilex/Piper 直连推理客户端入口。

本文件只负责组装对象和管理生命周期，核心数据流在 ``InferenceRuntime`` 中：

    RobotIO 采集观测 -> Policy Client 请求 GPU 服务 -> Action Buffer -> RobotIO 执行动作

真实硬件和 LeRobot mock 数据共用同一个 Runtime，只在 ``_make_robot_io`` 处选择不同
的 RobotIO 实现。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
CLIENT_TOOLS_DIR = REPO_ROOT / "client" / "tools"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))
if str(CLIENT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_TOOLS_DIR))


def _control_key(value: object, default: str) -> str:
    text = str(default if value is None else value).strip().lower()
    if text in {"space", "spacebar"}:
        return " "
    if len(text) != 1:
        raise ValueError(f"keyboard control key must be one character or 'space', got {value!r}")
    return text


class KeyboardEpisodeController:
    """Read single-key episode commands while Runtime owns the main thread."""

    def __init__(self, cfg: dict):
        self.start_key = _control_key(cfg.get("start_key"), "s")
        self.stop_key = _control_key(cfg.get("stop_key"), "space")
        self.quit_key = _control_key(cfg.get("quit_key"), "q")
        self.start_paused = bool(cfg.get("start_paused", True))
        self._start_requested = threading.Event()
        self._quit_requested = threading.Event()
        self._reader_stop = threading.Event()
        self._runtime_lock = threading.Lock()
        self._runtime = None
        self._fd: int | None = None
        self._terminal_attrs = None
        self._reader_thread: threading.Thread | None = None

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested.is_set()

    def set_runtime(self, runtime) -> None:
        with self._runtime_lock:
            self._runtime = runtime

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard episode control requires an interactive TTY")
        self._fd = sys.stdin.fileno()
        self._terminal_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd, termios.TCSANOW)
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="episode-keyboard",
            daemon=True,
        )
        self._reader_thread.start()
        if not self.start_paused:
            self._start_requested.set()

    def close(self) -> None:
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._fd is not None and self._terminal_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._terminal_attrs)
            except termios.error:
                pass
        self._fd = None
        self._terminal_attrs = None

    def wait_for_start(self) -> bool:
        while not self._quit_requested.is_set():
            if not self._start_requested.wait(timeout=0.1):
                continue
            self._start_requested.clear()
            return not self._quit_requested.is_set()
        return False

    def _runtime_snapshot(self):
        with self._runtime_lock:
            return self._runtime

    def _read_loop(self) -> None:
        assert self._fd is not None
        while not self._reader_stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                raw = os.read(self._fd, 1)
            except OSError:
                break
            if not raw:
                continue
            key = raw.decode(errors="ignore")
            runtime = self._runtime_snapshot()
            if key.lower() == self.start_key:
                if runtime is None:
                    logging.info("[episode control] start requested")
                    self._start_requested.set()
                else:
                    logging.info("[episode control] episode is already running")
                continue
            if key == self.stop_key:
                if runtime is None:
                    logging.info("[episode control] already paused; press %s to start", self.start_key)
                else:
                    logging.info("[episode control] stop requested; finalizing episode")
                    runtime.request_episode_stop()
                continue
            if key.lower() == self.quit_key:
                logging.info("[episode control] quit requested")
                self._quit_requested.set()
                self._start_requested.set()
                if runtime is not None:
                    runtime.request_episode_stop()
                return


def _build_runtime_config(cfg) -> dict:
    from config import section

    profile = cfg.runtime_options()
    server = section(cfg, "server")
    profile.setdefault("host", server.get("host", "localhost"))
    profile.setdefault("port", server.get("port", 8000))
    profile.setdefault("transport", server.get("transport", "websocket"))
    profile.setdefault("shared_memory_socket_path", server.get("shared_memory_socket_path", "/tmp/openpi_policy.sock"))
    if server.get("connect_timeout_s") is not None:
        profile.setdefault("connect_timeout_s", server.get("connect_timeout_s"))
    for key in (
        "endpoints",
        "servers",
        "connections_per_endpoint",
        "max_in_flight",
        "result_timeout_s",
        "first_result_timeout_s",
        "connect_retry_s",
    ):
        if server.get(key) is not None:
            profile.setdefault(key, server.get(key))

    profile.setdefault("state_dim", 14)

    execution_mode = profile.get("execution_mode", "sync")
    if execution_mode == "async":
        async_mode = profile.get("async_mode", "")
        if async_mode == "legato" and profile.get("delay_clip_max") is None:
            profile["delay_clip_max"] = int(profile.get("chunk_size", 50)) - 1
        if "delay_clip_max" in profile and int(profile.get("delay_clip_max", -1)) < 0:
            profile["delay_clip_max"] = int(profile.get("chunk_size", 50)) - 1
    return profile


def _recording_config(cfg, *, run_subdir: str | None = None) -> dict:
    from config import section

    recording = dict(section(cfg, "recording"))
    unique_run_dir = bool(recording.pop("unique_run_dir", False))
    for key in ("root_dir", "record_dir"):
        if not recording.get(key):
            continue
        path = Path(str(recording[key])).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if unique_run_dir:
            if run_subdir is None:
                raise ValueError("recording.unique_run_dir requires one run_subdir per client process")
            path /= run_subdir
        recording[key] = str(path)
    return recording


def _make_robot_io(cfg, telemetry):
    from config import section
    from robot_io import AgilexRobotIO
    from robot_io import MockLeRobotRobotIO

    robot_io_cfg = dict(section(cfg, "robot_io"))
    io_type = str(robot_io_cfg.get("type", robot_io_cfg.get("kind", "agilex"))).replace("-", "_").lower()
    if io_type in {"mock", "mock_lerobot", "lerobot"}:
        return MockLeRobotRobotIO.from_config(robot_io_cfg)
    if io_type != "agilex":
        raise ValueError(f"Unsupported robot_io.type: {io_type}")
    return AgilexRobotIO(
        camera_cfg=section(cfg, "camera"),
        arm_cfg=section(cfg, "arm"),
        can_cfg=section(cfg, "can"),
        telemetry=telemetry,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config_agilex.yaml", help="inference yaml config")
    parser.add_argument("--check-hardware", action="store_true", help="Read one state/image observation and exit")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    from config import load_config
    from config import section
    from multi_runtime import MultiServerInferenceRuntime
    from robot_io import check_hardware
    from runtime import InferenceRuntime
    from realtime_plot import RealtimePlotConfig
    from realtime_plot import TelemetryPublisher

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    # YAML 同时描述设备、远程 Policy Server、推理模式和记录选项。
    cfg = load_config(args.config)
    recording_section = section(cfg, "recording")
    recording_run_subdir = None
    if bool(recording_section.get("unique_run_dir", False)):
        recording_run_subdir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    recording_cfg = _recording_config(cfg, run_subdir=recording_run_subdir)
    if recording_run_subdir is not None:
        recording_root = recording_cfg.get("root_dir", recording_cfg.get("record_dir"))
        logging.info("recordings for this client session: %s", recording_root)
    telemetry = TelemetryPublisher(RealtimePlotConfig.from_config(section(cfg, "telemetry_plot")))
    telemetry.start_viewer()
    io = _make_robot_io(cfg, telemetry)
    runtime = None
    keyboard_controller = None
    full_shutdown_requested = False
    try:
        # 设备必须先启动，Runtime 随后才会创建观测/推理/控制循环。
        io.start()
        if args.check_hardware:
            check_hardware(io)
            return
        io.move_to_init(section(cfg, "arm"))
        runtime_cfg = _build_runtime_config(cfg)
        # multi_websocket 使用独立的并发 Runtime；其余传输复用标准 Runtime。
        runtime_cls = (
            MultiServerInferenceRuntime
            if str(runtime_cfg.get("transport", "websocket")).replace("-", "_").lower()
            in {"multi_websocket", "websocket_multi"}
            else InferenceRuntime
        )
        keyboard_cfg = dict(section(cfg, "keyboard_control"))
        keyboard_enabled = bool(keyboard_cfg.get("enabled", False))
        if keyboard_enabled and not sys.stdin.isatty():
            logging.warning("keyboard episode control requested without a TTY; running one episode immediately")
            keyboard_enabled = False

        if not keyboard_enabled:
            runtime = runtime_cls(
                io=io,
                cfg=runtime_cfg,
                recording_cfg=dict(recording_cfg),
                telemetry=telemetry,
            )
            runtime.run()
            full_shutdown_requested = runtime.signal_shutdown_received
            return

        keyboard_controller = KeyboardEpisodeController(keyboard_cfg)
        keyboard_controller.start()
        logging.info("[episode control] READY: press 's' to start, SPACE to stop and return to init, 'q' to quit")

        while keyboard_controller.wait_for_start():
            runtime = runtime_cls(
                io=io,
                cfg=runtime_cfg,
                recording_cfg=dict(recording_cfg),
                telemetry=telemetry,
            )
            episode_idx = runtime.output_manager.episode_idx
            keyboard_controller.set_runtime(runtime)
            logging.info("[episode control] episode_%d STARTED", episode_idx)
            completed_runtime = runtime
            episode_start = time.monotonic()
            episode_duration = 0.0
            try:
                completed_runtime.run()
            finally:
                episode_duration = time.monotonic() - episode_start
                keyboard_controller.set_runtime(None)
                completed_runtime.close()
                runtime = None

            signal_shutdown = completed_runtime.signal_shutdown_received
            logging.info(
                "[episode control] episode_%d SAVED (duration %.1fs)", episode_idx, episode_duration
            )
            if signal_shutdown or keyboard_controller.quit_requested:
                full_shutdown_requested = True
                break

            logging.info("[episode control] returning arms to init position")
            io.move_to_init(section(cfg, "arm"))
            logging.info("[episode control] READY: press 's' for a new episode, or 'q' to quit")

        if keyboard_controller.quit_requested:
            full_shutdown_requested = True
    except KeyboardInterrupt:
        logging.info("keyboard interrupt received; shutting down")
        full_shutdown_requested = True
    except Exception:
        full_shutdown_requested = True
        raise
    finally:
        # 无论正常结束还是异常退出，都先停止 Runtime，再释放机器人和相机。
        if keyboard_controller is not None:
            keyboard_controller.close()
        if runtime is not None:
            runtime.close()
            full_shutdown_requested = full_shutdown_requested or runtime.signal_shutdown_received
        if full_shutdown_requested:
            try:
                io.move_to_shutdown(section(cfg, "arm"))
            except Exception as exc:
                logging.warning("shutdown zero-position move failed: %s", exc)
        io.close()
        telemetry.close()


if __name__ == "__main__":
    main()
