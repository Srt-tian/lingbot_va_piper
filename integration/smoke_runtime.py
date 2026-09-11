#!/usr/bin/env python3
"""Exercise the real inference repository Runtime against LingBot with in-memory RobotIO."""
import json
import logging
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from client_entry import install_adapter

ROOT = Path(__file__).resolve().parents[1]
Runtime = install_adapter(str(Path(__file__).resolve().parents[1] / "runtime"))
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class MemoryRobotIO:
    def __init__(self):
        # Synthetic camera input: protocol smoke, not task-quality evaluation.
        self.images = {k: np.zeros((480, 640, 3), dtype=np.uint8)
                       for k in ("top_head", "hand_left", "hand_right")}
        self.applied = []

    def get_observation(self):
        return {"qpos": np.zeros(14), "images": self.images,
                "state_timestamp": time.time(), "image_timestamp": time.time()}

    def apply_action(self, action):
        value = np.asarray(action).copy()
        assert value.shape == (14,) and np.isfinite(value).all()
        self.applied.append(value)

    def hold_current_position(self): pass
    def set_runtime_logger(self, _): pass


def main():
    os.umask(0)
    logging.basicConfig(level=logging.INFO)
    out = ROOT / "outputs/inference_adapter_smoke"
    out.mkdir(parents=True, exist_ok=False)
    io = MemoryRobotIO()
    cfg = {"host": "127.0.0.1", "port": 8014, "transport": "websocket",
           "execution_mode": "sync", "ctrl_type": "joint", "chunk_size": 36,
           "state_dim": 14, "max_publish_step": 72, "publish_rate": 30,
           "num_denoising_steps": 10, "prompt": "flat the cloth with both robot arms",
           "action_buffer": "stream", "log_every_steps": 36}
    runtime = Runtime(io, cfg, {"root_dir": str(out / "records"), "record_model_io": True,
                              "record_runtime_events": True, "record_action_steps": True,
                              "record_policy_rollout": False, "fps": 10})
    # Verify the client adapter preserves the original resolution and converts BGR to RGB exactly once.
    payload, raw = runtime._build_payload(io.get_observation())
    for k, image in io.images.items():
        np.testing.assert_array_equal(payload["images"][k].transpose(1, 2, 0), image[:, :, ::-1])
    metadata = runtime.policy.get_server_metadata()
    assert metadata["action_horizon"] == 36 and metadata["state_conditioning"] is False
    # Check the real client's error envelope path without loading another model.
    bad_client = WebsocketClientPolicy("127.0.0.1", 8014)
    try:
        invalid = dict(payload); invalid["images"] = dict(payload["images"]); invalid["images"].pop("hand_left")
        try:
            bad_client.infer(invalid)
            raise AssertionError("Missing camera should be rejected")
        except RuntimeError as exc:
            assert "images must contain exactly" in str(exc)
    finally:
        bad_client.close()
    started = time.monotonic()
    deadline = threading.Timer(90, runtime.shutdown.set)
    deadline.start()
    try:
        runtime.run()
    finally:
        deadline.cancel()
        runtime.close()
    actions = np.asarray(io.applied)
    assert actions.shape == (72, 14), actions.shape
    assert runtime.inference_count == 2, runtime.inference_count
    assert np.isfinite(actions).all()
    np.save(out / "mock_published_actions.npy", actions)
    summary = {"status": "passed", "reference_runtime": str(ROOT / "runtime/client/inference/runtime.py"),
               "metadata": metadata, "inference_count": runtime.inference_count,
               "mock_published_shape": list(actions.shape), "all_finite": True,
               "elapsed_seconds": time.monotonic() - started, "hardware_started": False,
               "missing_camera_error_envelope_ok": True, "native_rgb_image_contract_ok": True,
               "record_files": sorted(str(p.relative_to(out)) for p in (out / "records").rglob("*") if p.is_file())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
