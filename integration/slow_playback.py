"""Bounded stateless prefetch with complete chunks played at a configurable rate.

This is an experimental scheduling adapter, not the paper's FDM/KV algorithm.
Only one next chunk may be pending (including an in-flight RPC).
"""
import math
import queue
import threading
import time

import numpy as np


def enabled(cfg):
    return cfg.get('playback_mode') == 'slow_prefetch'


def validate_config(cfg):
    if cfg.get('execution_mode') != 'async' or cfg.get('policy_protocol', 'stateless') != 'stateless':
        raise ValueError('slow_prefetch requires stateless async; official_kv is unsupported')
    if cfg.get('async_mode') != 'naive' or cfg.get('ctrl_type', 'joint') != 'joint':
        raise ValueError('slow_prefetch requires naive joint mode')
    if cfg.get('transport', 'websocket') != 'websocket':
        raise ValueError('slow_prefetch requires WebSocket transport for cancellation')
    if int(cfg.get('chunk_size', 0)) != 36 or int(cfg.get('execute_prefix_steps', 36)) != 36:
        raise ValueError('slow_prefetch plays complete 36-step chunks')
    for name, default in [('publish_rate', 12), ('max_prediction_wait_s', 10),
                          ('max_result_age_s', 10)]:
        value = cfg.get(name, default)
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if float(cfg.get('publish_rate', 12)) > 30:
        raise ValueError('slow_prefetch publish_rate must be <= training rate 30 Hz')
    if int(cfg.get('min_smooth_steps', 0)) != 0 or int(cfg.get('latency_k', 0)) != 0:
        raise ValueError('slow_prefetch does not blend, truncate or skip predicted steps')
    maximum = cfg.get('max_publish_step')
    if maximum is not None and (isinstance(maximum, bool) or int(maximum) != maximum or maximum < 1):
        raise ValueError('max_publish_step must be a positive integer or null')


class OneChunkPrefetch:
    def __init__(self):
        self.ready = queue.Queue(maxsize=1)
        self.requested = threading.Event()
        self.lock = threading.Lock()
        self.pending = False
        self.closed = False
        self.error = None

    def request(self):
        with self.lock:
            if self.closed:
                return
            if self.pending:
                raise RuntimeError('Only one next chunk may be pending')
            self.pending = True
            self.requested.set()

    def deliver(self, result):
        with self.lock:
            if not self.closed:
                self.ready.put_nowait(result)

    def take(self):
        with self.lock:
            result = self.ready.get_nowait()
            self.pending = False
            return result

    def fail(self, error):
        with self.lock:
            self.error = error

    def raise_if_failed(self):
        with self.lock:
            error = self.error
        if error is not None:
            raise RuntimeError(f'slow_prefetch inference failed: {error}') from error

    def close(self):
        with self.lock:
            self.closed = True
            while not self.ready.empty():
                self.ready.get_nowait()
            self.requested.set()


def inference_loop(rt):
    state = rt.slow_prefetch
    try:
        while not rt.shutdown.is_set():
            if not state.requested.wait(.02):
                continue
            state.requested.clear()
            if rt.shutdown.is_set():
                return
            obs = rt._get_latest_observation()
            while obs is None and not rt.shutdown.wait(.005):
                obs = rt._get_latest_observation()
            if rt.shutdown.is_set():
                return
            requested_at = time.monotonic()
            payload, raw_images, proprio = rt._base_payload(obs)
            payload['state'] = proprio
            result = rt.policy.infer(payload)
            finished = time.monotonic()
            if rt.shutdown.is_set():
                return  # never record or publish a late result after stop
            actions = np.asarray(result.get('actions'), dtype=np.float32)
            if actions.shape != (36, 14) or not np.isfinite(actions).all():
                raise ValueError('Expected finite stateless actions [36,14]')
            rt.inference_count += 1
            chunk_id = rt.inference_count
            rt.log_event({'event': 'slow_prefetch_ready', 'chunk_id': chunk_id,
                          'rtt_ms': (finished-requested_at)*1000,
                          'playback_rate_hz': float(rt.cfg['publish_rate']),
                          'chunk_duration_s': 36/float(rt.cfg['publish_rate'])})
            if rt.recorder is not None:
                rt.recorder.record_model_io(payload=payload, model_output_actions=actions,
                    timestamp_sec=time.time(), raw_images=raw_images)
            state.deliver({'actions': actions.copy(), 'chunk_id': chunk_id,
                           'requested_at': requested_at, 'ready_at': finished})
    except Exception as exc:
        if not rt.shutdown.is_set():
            state.fail(exc)
            rt.request_episode_stop()


def control_loop(rt):
    state = rt.slow_prefetch
    period = 1/float(rt.cfg['publish_rate'])
    maximum = rt.cfg.get('max_publish_step')
    count = 0
    last_publish = None
    state.request()  # cold start; no discarded warmup call
    try:
        while not rt.shutdown.is_set():
            wait_start = time.monotonic()
            holding = False
            while not rt.shutdown.is_set():
                state.raise_if_failed()
                try:
                    chunk = state.take()
                    break
                except queue.Empty:
                    if not holding:
                        with rt._publish_lock:
                            if not rt.shutdown.is_set():
                                rt._hold_robot_position()
                        rt.log_event({'event': 'slow_prefetch_wait', 'published_steps': count})
                        holding = True
                    if time.monotonic()-wait_start > float(rt.cfg.get('max_prediction_wait_s', 10)):
                        raise TimeoutError('Timed out waiting for the next action chunk')
                    rt.shutdown.wait(.005)
            if rt.shutdown.is_set():
                break
            age = time.monotonic()-chunk['requested_at']
            if age > float(rt.cfg.get('max_result_age_s', 10)):
                raise TimeoutError('Predicted action chunk is too old to begin playback')
            # Once taken for execution, request exactly one successor. A fast
            # producer cannot overwrite this chunk or run multiple chunks ahead.
            if maximum is None or count+len(chunk['actions']) < maximum:
                state.request()
            rt.log_event({'event': 'slow_prefetch_chunk_start', 'chunk_id': chunk['chunk_id'],
                          'result_age_s': age, 'wait_s': time.monotonic()-wait_start,
                          'boundary_gap_s': 0 if last_publish is None else
                          max(0, time.monotonic()-last_publish-period)})
            for index, action in enumerate(chunk['actions']):
                state.raise_if_failed()
                with rt._publish_lock:
                    if rt.shutdown.is_set():
                        break
                    step = {'chunk_id': chunk['chunk_id'], 'chunk_step_index': index,
                            'raw_action_index': index+12, 'is_hold': False,
                            'action': action.copy()}
                    if rt.first_action_publish_monotonic is None:
                        rt.first_action_publish_monotonic = time.monotonic()
                    rt.io.apply_action(action, action_step=step)
                    last_publish = time.monotonic()
                    rt._record_policy_rollout(action, step)
                    rt._log_action_step(action, step)
                    rt._maybe_log_action_publish_rate()
                    rt.action_pop_count += 1
                    count += 1
                # No catch-up burst after a stalled RPC, SDK call or logger.
                if rt.shutdown.wait(period):
                    break
                if maximum is not None and count >= maximum:
                    rt.shutdown.set()
                    break
        state.raise_if_failed()
    finally:
        rt.request_episode_stop()
        state.close()
