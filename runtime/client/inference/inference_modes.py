"""异步推理模式适配层。

Runtime 负责通用 RPC 和 Action Buffer；本模块只处理不同算法的请求/响应差异：

* Base：发送当前机器人状态；
* VLASH：用旧 chunk 中的未来动作近似网络延迟后的状态；
* RTC：把上一 chunk、执行窗口和估计延迟交给 RTC 模型；
* Legato：把模型空间中的上一 chunk 交给去噪过程做连续性引导。
* Structured EAPN：发送上一轮之后实际执行的步数，驱动跨 chunk 噪声状态。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import threading


logger = logging.getLogger(__name__)


def bounded_int(name: str, value, *, min_value: int, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be convertible to int, got {value!r}") from exc
    if out < min_value:
        out = min_value
    if max_value is not None and out > max_value:
        out = max_value
    return out


def validate_actions_model(actions_model, expected_horizon: int) -> np.ndarray | None:
    if actions_model is None:
        return None
    arr = np.asarray(actions_model, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"actions_model must have shape [H, D], got {arr.shape}")
    if arr.shape[0] != int(expected_horizon):
        raise ValueError(f"actions_model must have horizon {expected_horizon}, got {arr.shape[0]}")
    return arr


class BaseInferenceMode:
    """标准模式：以当前 proprio 为条件，直接使用服务端返回的动作。"""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    @property
    def cfg(self) -> dict[str, Any]:
        return self.runtime.cfg

    @property
    def buffer(self):
        return self.runtime.stream_buffer

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        base_payload["state"] = proprio
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        actions = out.get("actions", None) if isinstance(out, dict) else None
        if actions is None or len(actions) == 0:
            return None
        return np.asarray(actions, dtype=float)

    def reset(self) -> None:
        """Reset per-mode transport state after warmup or at an episode boundary."""



class VlashMode(BaseInferenceMode):
    """用预计请求返回时刻的未来状态代替当前状态，减小观测延迟。"""

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        delay_steps = self.runtime.get_delay_steps()
        # Buffer 中的未来动作可视作短期状态预测；没有可用预测时安全回退到实测状态。
        future = self.buffer.peek_future_action(delay_steps)
        use_future = bool(self.cfg.get("enable_future_state_injection", True)) and future is not None
        base_payload["state"] = future if use_future else proprio
        self.runtime.log_mode_payload(
            {
                "used_future_state": bool(use_future),
                "pred_delay_steps": int(delay_steps),
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        self.runtime.update_delay_steps(rtt_sec)
        return super().handle_result(out, rtt_sec)


class RtcMode(BaseInferenceMode):
    """为 RTC 模型维护上一 action chunk，并传递实时执行进度。"""

    def __init__(self, runtime: Any):
        super().__init__(runtime)
        self.prev_action_chunk: np.ndarray | None = None

        self.lock = threading.Lock()

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        chunk_size = int(self.cfg.get("chunk_size", 50))
        execute_horizon = self.cfg.get("execute_horizon", chunk_size)
        execute_horizon = max(1, min(int(execute_horizon), chunk_size))
        base_payload.update(
            {
                "state": proprio,
                "execute_horizon": execute_horizon,
                "enable_rtc": True,
                "mask_prefix_delay": bool(self.cfg.get("mask_prefix_delay", False)),
                "max_guidance_weight": float(self.cfg.get("max_guidance_weight", 0.5)),
                "inference_delay": int(max(0, self.runtime.get_delay_steps())),
            }
        )
        with self.lock:
            # 推理线程可能同时读取/更新上一 chunk，因此复制后再放入请求。
            prev = None if self.prev_action_chunk is None else self.prev_action_chunk.copy()
        if prev is not None:
            base_payload["prev_action_chunk"] = prev.tolist()
        self.runtime.log_mode_payload(
            {
                "execute_horizon": int(execute_horizon),
                "inference_delay": int(self.runtime.get_delay_steps()),
                "has_prev_action_chunk": prev is not None,
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        self.runtime.update_delay_steps(rtt_sec)
        actions = super().handle_result(out, rtt_sec)
        if actions is not None:
            with self.lock:
                self.prev_action_chunk = np.asarray(actions, dtype=float).copy()
        return actions


class LegatoMode(BaseInferenceMode):
    """让新 chunk 在模型去噪阶段参考上一 chunk，而非仅在客户端后处理。"""

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        chunk_size = int(self.cfg.get("chunk_size", 50))
        progress = self.buffer.get_chunk_progress()
        inference_delay = bounded_int(
            "inference_delay",
            self.runtime.get_delay_steps(),
            min_value=0,
            max_value=chunk_size,
        )
        execute_horizon = bounded_int(
            "execute_horizon",
            # 请求发出前已执行的步数，加上请求往返期间预计继续执行的步数。
            int(progress["executed_steps"]) + inference_delay,
            min_value=0,
            max_value=chunk_size,
        )
        ramp_down = bounded_int(
            "ramp_down",
            self.cfg.get("ramp_down_steps", 22),
            min_value=0,
            max_value=chunk_size,
        )
        base_payload.update(
            {
                "state": proprio,
                "inference_delay": inference_delay,
                "execute_horizon": execute_horizon,
                "ramp_down": ramp_down,
            }
        )
        prev_action_chunk_model = self.buffer.get_prev_action_chunk_model()
        if prev_action_chunk_model is not None and len(prev_action_chunk_model) > 0:
            # 必须使用归一化后的模型空间动作，服务端会直接把它送入去噪引导。
            base_payload["prev_action_chunk_model"] = prev_action_chunk_model
        self.runtime.log_mode_payload(
            {
                "inference_delay": int(inference_delay),
                "execute_horizon": int(execute_horizon),
                "ramp_down": int(ramp_down),
                "executed_steps_at_trigger": int(progress["executed_steps"]),
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        self.runtime.update_delay_steps(rtt_sec)
        actions = super().handle_result(out, rtt_sec)
        if actions is None:
            return None
        actions_model = out.get("actions_model", None) if isinstance(out, dict) else None
        try:
            self.runtime.pending_actions_model = validate_actions_model(actions_model, len(actions))
        except Exception as exc:
            self.runtime.pending_actions_model = None
            logger.warning("[Legato] actions_model ignored: %s", exc)
        return actions


class StructuredEapnMode(BaseInferenceMode):
    """维护 Structured EAPN 的 reset/executed_steps 传输协议。"""

    def __init__(self, runtime: Any):
        super().__init__(runtime)
        self._reset_pending = True

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        progress = self.buffer.get_chunk_progress()
        executed_steps = bounded_int(
            "executed_steps",
            progress.get("executed_steps", 0),
            min_value=0,
            max_value=int(self.cfg.get("chunk_size", 50)),
        )
        base_payload.update(
            {
                "state": proprio,
                "executed_steps": 0 if self._reset_pending else executed_steps,
            }
        )
        if self._reset_pending:
            base_payload["reset_progressive_noise"] = True
            # Mark the reset as sent when the request is built so a warmup or a
            # fast follow-up request cannot enqueue duplicate reset markers.
            self._reset_pending = False
        self.runtime.log_mode_payload(
            {
                "structured_eapn": True,
                "executed_steps": int(base_payload["executed_steps"]),
                "reset_progressive_noise": bool(self._reset_pending),
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        actions = super().handle_result(out, rtt_sec)
        return actions

    def reset(self) -> None:
        self._reset_pending = True


def _norm(value: Any, default: str = "") -> str:
    text = str(default if value is None else value).strip()
    return text.replace("-", "_").lower()


def stream_smooth_method(cfg: dict[str, Any]) -> str:
    smooth_method = _norm(cfg.get("smooth_method")) if cfg.get("smooth_method") is not None else None
    execution_mode = cfg.get("execution_mode")

    if execution_mode == "sync":
         return "raw"

    async_mode = _norm(cfg.get("async_mode"))
    if async_mode == "temporal_smoothing":
        return "temporal_smoothing"
    if async_mode == "temporal_ensembling":
        return "temporal_ensembling"

    if smooth_method is not None:
        if smooth_method in {"raw", "temporal_smoothing", "temporal_ensembling"}:
            return smooth_method
        raise ValueError(
            "smooth_method must be one of 'raw', 'temporal_smoothing', or 'temporal_ensembling', "
            f"got {smooth_method!r}"
        )
    return "raw"


def build_inference_mode(runtime: Any) -> BaseInferenceMode:
    """根据 execution_mode/async_mode 创建请求协议适配器。"""
    exec_mode = execution_mode(runtime.cfg)
    async_mode = _norm(runtime.cfg.get("async_mode")) if exec_mode == "async" else None
    runtime.cfg["execution_mode"] = exec_mode
    if async_mode == "vlash":
        return VlashMode(runtime)
    if async_mode == "rtc":
        return RtcMode(runtime)
    if async_mode == "legato":
        return LegatoMode(runtime)
    if async_mode in {"eapn", "structured_eapn"}:
        return StructuredEapnMode(runtime)
    return BaseInferenceMode(runtime)


def execution_mode(cfg: dict[str, Any]) -> str:
    value = cfg.get("execution_mode")
    if value is None:
        return "sync"
    if value not in {"sync", "async"}:
        raise ValueError(f"execution_mode must be 'sync' or 'async', got {value!r}")
    return value
