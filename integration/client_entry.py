#!/usr/bin/env python3
"""Use the existing workstation client with native-resolution LingBot images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


class FiniteSyncBuffer:
    """End a synchronous chunk instead of repeating its final action indefinitely."""
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def pop_next_action(self):
        if self.inner.pending_count() <= 0:
            return None
        return self.inner.pop_next_action()


def validate_execution(cfg):
    if cfg.get("policy_protocol") == "official_kv":
        if cfg.get("execution_mode") != "sync" or int(cfg.get("chunk_size", 0)) != 48:
            raise ValueError("Official KV flow requires sync execution and chunk_size=48")
        if int(cfg.get("execute_prefix_steps", 48)) != 48 or float(cfg.get("publish_rate", 30)) != 30:
            raise ValueError("Official KV flow executes complete chunks at training 30 Hz")
        return
    if not 1 <= int(cfg.get("execute_prefix_steps", 36)) <= 36:
        raise ValueError("execute_prefix_steps must be between 1 and 36")
    mode = cfg.get("execution_mode")
    if mode not in {"sync", "async"} or int(cfg.get("chunk_size", 0)) != 36:
        raise ValueError("LingBot requires sync/async execution and chunk_size=36")
    if mode == "async" and cfg.get("async_mode") not in {"naive", "temporal_smoothing"}:
        raise ValueError("LingBot supports naive/temporal_smoothing async; RTC/Legato are not supported")


def install_adapter(reference_root):
    reference_root = Path(reference_root)
    manifest = json.loads((HERE / "reference_manifest.json").read_text())
    changed = [p for p, h in manifest["files"].items()
               if not (reference_root / p).is_file() or hashlib.sha256((reference_root / p).read_bytes()).hexdigest() != h]
    if changed:
        raise RuntimeError(f"Reference inference client changed; review adapter compatibility first: {changed}")
    for part in ["client/inference", "client/tools", "packages/openpi-client/src"]:
        sys.path.insert(0, str(reference_root / part))
    import cv2
    import numpy as np
    import runtime
    import threading
    from async_runtime_adapter import AtomicActionBuffer, PrefixActionMode, control_loop
    import robot_io
    from collections import deque

    class LingbotRealSenseCamera(robot_io.RealSenseCamera):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Retaining thousands of SDK frames exhausts the camera frame pool while paused.
            self._frames = deque(maxlen=2)

    robot_io.RealSenseCamera = LingbotRealSenseCamera

    class LingbotRuntime(runtime.InferenceRuntime):
        def __init__(self, *args, **kwargs):
            self._publish_lock = threading.RLock()
            super().__init__(*args, **kwargs)
            validate_execution(self.cfg)
            if self.cfg.get("execution_mode") == "sync":
                self.stream_buffer = FiniteSyncBuffer(self.stream_buffer)
            else:
                self.stream_buffer = AtomicActionBuffer(self.stream_buffer)
                self.mode_handler = PrefixActionMode(self.mode_handler, self.cfg.get("execute_prefix_steps", 36))

        def _warmup_inference(self):
            if self.cfg.get("policy_protocol") == "official_kv":
                return  # first real request warms up; never create discarded history
            return super()._warmup_inference()

        def _sync_loop(self):
            if self.cfg.get("policy_protocol") == "official_kv":
                from official_runtime import run_official_loop
                return run_official_loop(self)
            return super()._sync_loop()

        def _control_loop(self):
            control_loop(self)

        def request_episode_stop(self):
            with self._publish_lock:
                super().request_episode_stop()

        def _log_action_step(self, action, step):
            super()._log_action_step(action, step)
            if self.cfg.get("execution_mode") == "async":
                self.log_event({"event": "action_buffer_step", "chunk_id": int(step["chunk_id"]),
                                "chunk_step_index": int(step["chunk_step_index"]),
                                "is_hold": bool(step.get("is_hold", False))})

        def close(self):
            # An in-flight async RPC takes ~2.6 s; finish it before closing recorders/socket.
            with self._publish_lock:
                self.shutdown.set()
                if isinstance(self.stream_buffer, AtomicActionBuffer):
                    self.stream_buffer.close()
                self._hold_robot_position()
            for thread in self.threads:
                thread.join(timeout=10.0)
            super().close()

        def _base_payload(self, obs):
            raw = {k: cv2.cvtColor(obs["images"][k], cv2.COLOR_BGR2RGB)
                   for k in ("top_head", "hand_right", "hand_left")}
            payload = {"images": {k: np.ascontiguousarray(v.transpose(2, 0, 1)) for k, v in raw.items()},
                       "prompt": str(self.cfg.get("prompt", "flat the cloth with both robot arms"))}
            steps = runtime._configured_num_denoising_steps(self.cfg)
            if steps is not None:
                payload["num_steps"] = steps
            return payload, raw, np.asarray(obs["qpos"], dtype=float)

        def _make_policy_client(self):
            policy = super()._make_policy_client()
            metadata = policy.get_server_metadata()
            if self.cfg.get("policy_protocol") == "official_kv":
                if metadata.get("policy_protocol") != "official_kv" or metadata.get("action_horizon") != 48:
                    policy.close()
                    raise RuntimeError("Expected the official KV protocol server")
                return policy
            if metadata.get("policy_family") != "lingbot_va" or metadata.get("action_horizon") != 36 or metadata.get("action_dim") != 14:
                policy.close()
                raise RuntimeError(f"Expected LingBot-VA 36x14 server, got {metadata}")
            return policy

    runtime.InferenceRuntime = LingbotRuntime
    return LingbotRuntime


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-root", default=os.environ.get("INFERENCE_REPO_ROOT", str(Path(__file__).resolve().parents[1] / "runtime")))
    parser.add_argument("--dry-run", action="store_true")
    known, remaining = parser.parse_known_args()
    install_adapter(known.reference_root)
    from config import load_config
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(HERE / "client_lingbot.yaml"))
    cfg_args, _ = config_parser.parse_known_args(remaining)
    cfg = load_config(cfg_args.config)
    validate_execution(cfg.runtime_options())
    if known.dry_run:
        print(json.dumps({"status": "dry_run_ok", "reference_root": known.reference_root,
                          "config": cfg_args.config, "image_resize": "server-side direct 256x256",
                          "chunk_size": cfg.runtime_options().get("chunk_size"),
                          "policy_protocol": cfg.runtime_options().get("policy_protocol", "stateless"), "hardware_started": False}, indent=2))
        return
    # Fail before constructing RobotIO: the reference client auto-runs without a TTY.
    if not sys.stdin.isatty():
        raise RuntimeError("Real robot client requires an interactive TTY; use ssh -tt. Hardware was not started.")
    # Match the workstation client: root-created recordings must remain writable by the host user.
    os.umask(0)
    import agilex_inference_openpi
    sys.argv = [sys.argv[0], *remaining]
    if "--config" not in remaining:
        sys.argv += ["--config", cfg_args.config]
    agilex_inference_openpi.main()


if __name__ == "__main__":
    main()
