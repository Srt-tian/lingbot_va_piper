#!/usr/bin/env python3
"""OpenPI-envelope WebSocket service, with fresh-observation LingBot chunks."""
import argparse
import asyncio
import copy
import http
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wan_va"))
from configs.va_fold_clothes_cfg import va_fold_clothes_cfg
from distributed.util import init_distributed
from wan_va_server import VA_Server
from contract import CAMERAS, CHANNELS, future_actions, validate_request
import msgpack_numpy
from training_preprocessing import resize_observation


class LingbotPolicy:
    def __init__(self, record_dir, seed):
        torch.set_num_threads(min(16, len(os.sched_getaffinity(0))))
        cfg = copy.deepcopy(va_fold_clothes_cfg)
        cfg.wan22_pretrained_model_name_or_path = os.environ.get("LINGBOT_BASE_MODEL", str(ROOT / "models/base"))
        cfg.resume_from = os.environ.get("LINGBOT_CHECKPOINT", str(ROOT / "models/step14000"))
        cfg.norm_stat = json.loads(Path(os.environ.get("LINGBOT_NORM_PATH", str(ROOT / "meta/lingbot_action_norm.json"))).read_text())
        if cfg.norm_stat.get("used_action_channel_ids") != CHANNELS:
            raise ValueError("Expected fixed-2846 action normalization")
        q01, q99 = (np.asarray(cfg.norm_stat[k]) for k in ("q01", "q99"))
        if q01.shape != (30,) or q99.shape != (30,) or not np.isfinite(q01).all() or not np.isfinite(q99).all() or not (q99[CHANNELS] > q01[CHANNELS]).all():
            raise ValueError("Invalid normalization quantiles")
        cfg.obs_cam_keys = [f"observation.images.{k}" for k in CAMERAS]
        cfg.used_action_channel_ids = CHANNELS
        cfg.inverse_used_action_channel_ids = [14] * 30
        for i, c in enumerate(CHANNELS):
            cfg.inverse_used_action_channel_ids[c] = i
        cfg.action_per_frame = 12
        cfg.rank = cfg.local_rank = 0
        cfg.world_size = 1
        cfg.enable_offload = True
        cfg.save_root = str(record_dir)
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.model = VA_Server(cfg)
        self.seed = seed
        self.prompt = None
        self.embeddings = None
        self.counter = 0
        self.metadata = {
            "policy_family": "lingbot_va", "checkpoint_step": int(os.environ.get("LINGBOT_CHECKPOINT_STEP", "14000")),
            "action_dim": 14, "action_horizon": 36, "action_fps": 30,
            "camera_keys": list(CAMERAS), "image_format": "RGB uint8 CHW",
            "image_preprocessing": "RGB uint8 OpenCV INTER_AREA to 256x256, matching training",
            "negative_text_condition": "zeros_like_positive_embedding",
            "single_session_required": False, "max_concurrent_inferences": 1,
            "supported_modes": ["sync", "base"], "state_conditioning": False,
            "reset_semantics": "fresh observation, empty KV cache on every request; no cross-request autoregression",
            "action_semantics": "absolute left6 joints,left gripper,right6 joints,right gripper; first conditioning block removed",
        }
        # Cache the deployment prompt before listening; it is reused across requests and connections.
        with torch.no_grad():
            self._set_prompt("flat the cloth with both robot arms")

    def _set_prompt(self, prompt):
        if prompt == self.prompt:
            return True
        self.model.transformer.clear_cache(self.model.cache_name)
        self.model.vae.to("cpu")
        torch.cuda.empty_cache()
        use_gpu = torch.cuda.mem_get_info()[0] > 14 * 1024**3
        self.model.text_encoder.to(self.model.device if use_gpu else "cpu")
        positive, _ = self.model.encode_prompt(
            prompt=prompt, negative_prompt=None, do_classifier_free_guidance=False,
            num_videos_per_prompt=1, max_sequence_length=512,
            device=self.model.device, dtype=self.model.dtype,
        )
        self.embeddings = (positive, torch.zeros_like(positive))
        logging.info("TRAINING_PREPROCESS_READY resize=INTER_AREA negative_nonzero=%d",
                     torch.count_nonzero(self.embeddings[1]).item())
        self.model.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        self.model.vae.to(self.model.device)
        self.prompt = prompt
        return False

    @torch.no_grad()
    def infer(self, payload):
        observation, state, prompt, steps = validate_request(payload)
        observation = resize_observation(observation, self.model.job_config.height, self.model.job_config.width)
        torch.cuda.set_device(0)
        started = time.monotonic()
        self.counter += 1
        request_dir = self.record_dir / f"request_{time.time_ns()}_{self.counter}"
        request_dir.mkdir()
        prompt_hit = self._set_prompt(prompt)
        self.model.save_root = str(request_dir)
        self.model.job_config.num_inference_steps = steps
        self.model.job_config.action_num_inference_steps = steps
        torch.manual_seed(self.seed + self.counter - 1)
        torch.cuda.manual_seed_all(self.seed + self.counter - 1)
        self.model._reset(None)
        self.model.prompt_embeds, self.model.negative_prompt_embeds = self.embeddings
        raw, latents = self.model._infer(observation, frame_st_id=0)
        if not torch.isfinite(latents).all():
            raise ValueError("Non-finite predicted video latents")
        actions = future_actions(raw)
        torch.cuda.synchronize()
        elapsed_ms = (time.monotonic() - started) * 1000
        result = {
            "actions": actions, "policy_timing": {"infer_ms": elapsed_ms},
            "lingbot": {"state_used": False, "prompt_cache_hit": prompt_hit,
                        "checkpoint_step": int(os.environ.get("LINGBOT_CHECKPOINT_STEP", "14000")), "action_fps": 30,
                        "reset_per_request": True, "conditioning_block_removed": True},
        }
        np.save(request_dir / "actions.npy", actions)
        (request_dir / "summary.json").write_text(json.dumps({
            "prompt": prompt, "num_steps": steps, "seed": self.seed + self.counter - 1,
            "state": state.tolist(), "state_used": False, "actions_shape": list(actions.shape),
            "camera_shapes": {k: list(v.shape) for k, v in observation["obs"][0].items()},
            "infer_ms": elapsed_ms, "all_finite": True,
        }, indent=2) + "\n")
        logging.info("request=%d actions=%s infer_ms=%.1f prompt_cache_hit=%s", self.counter, actions.shape, elapsed_ms, prompt_hit)
        return result


async def run_server(policy, host, port):
    lock = asyncio.Lock()

    def health(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def handler(ws):
        await ws.send(msgpack_numpy.packb(policy.metadata))
        server_request_id = 0
        try:
            async for data in ws:
                request_id = None
                try:
                    if not isinstance(data, bytes):
                        raise ValueError("Expected binary MsgPack request")
                    envelope = msgpack_numpy.unpackb(data)
                    if not isinstance(envelope, dict) or envelope.get("type") != "infer" or "payload" not in envelope:
                        raise ValueError("Expected type=infer envelope with payload")
                    request_id = envelope.get("request_id", envelope.get("request_index"))
                    validate_request(envelope["payload"])
                    server_request_id += 1
                    start = time.monotonic()
                    async with lock:
                        acquired = time.monotonic()
                        result = await asyncio.to_thread(policy.infer, envelope["payload"])
                    result["server_timing"] = {
                        "request_id": request_id, "server_request_id": server_request_id,
                        "infer_ms": result["policy_timing"]["infer_ms"],
                        "queue_wait_ms": (acquired - start) * 1000,
                    }
                    await ws.send(msgpack_numpy.packb({"type": "result", "request_id": request_id,
                        "server_request_id": server_request_id, "payload": result}))
                except Exception as exc:
                    logging.exception("Inference request failed")
                    await ws.send(msgpack_numpy.packb({"type": "error", "request_id": request_id,
                        "server_request_id": server_request_id, "traceback": f"{type(exc).__name__}: {exc}"}))
                    await ws.close(code=1011, reason="Inference request failed")
                    return
        except ConnectionClosed:
            pass

    async with serve(handler, host, port, compression=None, max_size=32 * 1024**2, process_request=health):
        logging.info("LINGBOT_READY host=%s port=%d metadata=%s", host, port, policy.metadata)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--record-dir", type=Path, default=ROOT / "outputs/server_requests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29642")
    init_distributed(world_size=1, local_rank=0, rank=0)
    try:
        asyncio.run(run_server(LingbotPolicy(args.record_dir, args.seed), args.host, args.port))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
