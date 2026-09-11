"""Fix scheduling/metadata around the reference buffers; reuse their blending math."""
import logging
import threading
import time
import numpy as np


class AtomicActionBuffer:
    """Serialize whole switches with pops and distinguish prediction from hold ticks."""
    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.RLock()
        self.last = None
        self.closed = False
        self.held_steps = 0

    def __getattr__(self, name):
        value = getattr(self.inner, name)
        if not callable(value):
            return value
        def locked(*args, **kwargs):
            with self.lock:
                return value(*args, **kwargs)
        return locked

    def integrate_new_chunk(self, *args, **kwargs):
        with self.lock:
            if self.closed:
                return None
            # Reference NaiveAsyncBuffer reads its clock and replaces the chunk in
            # separate critical sections. The outer lock spans both operations.
            return self.inner.integrate_new_chunk(*args, **kwargs)

    def pop_next_action(self):
        with self.lock:
            if self.closed:
                return None
            if self.inner.pending_count() <= 0:
                if self.last is None:
                    return None
                self.held_steps += 1
                return {**self.last, 'action': self.last['action'].copy(), 'is_hold': True}
            step = self.inner.pop_next_action()
            if step is None:
                return None
            self.last = {**step, 'action': np.asarray(step['action']).copy(), 'is_hold': False}
            return {**self.last, 'action': self.last['action'].copy()}

    def close(self):
        with self.lock:
            self.closed = True


def control_loop(rt):
    """Pace from each real publication; never catch up after empty-buffer waits."""
    period = 1.0 / float(rt.cfg.get('publish_rate', 30))
    configured_maximum = rt.cfg.get('max_publish_step', 10000)
    maximum = None if configured_maximum is None else int(configured_maximum)
    next_publish = time.monotonic()
    count = 0
    while (maximum is None or count < maximum) and not rt.shutdown.is_set():
        if rt.shutdown.wait(max(0.0, next_publish - time.monotonic())):
            break
        with rt._publish_lock:
            if rt.shutdown.is_set():
                break
            step = rt.stream_buffer.pop_next_action()
            if step is not None:
                if str(rt.cfg.get('ctrl_type', 'joint')) != 'joint':
                    raise ValueError('SDK runtime supports joint actions only')
                action = np.asarray(step['action'], dtype=float)
                now = time.monotonic()
                if rt.first_action_publish_monotonic is None:
                    rt.first_action_publish_monotonic = now
                if rt.telemetry is not None:
                    rt.telemetry.publish('cmd_vla_30hz', action)
                rt.io.apply_action(action, action_step=step)
                rt._record_policy_rollout(action, step)
                rt._log_action_step(action, step)
                rt._maybe_log_action_publish_rate()
                count += 1
                if count % max(1, int(rt.cfg.get("log_every_steps", 36))) == 0:
                    logging.getLogger(__name__).info(
                        "async published=%d held=%d pending=%d", count,
                        rt.stream_buffer.held_steps, rt.stream_buffer.pending_count())
                # No shortened interval after slow logging or a stall.
                next_publish = time.monotonic() + period
        if step is None:
            rt.shutdown.wait(.001)
    rt.shutdown.set()


class PrefixActionMode:
    """Keep the reference request protocol; admit only the action prefix to execution."""
    def __init__(self, inner, steps):
        self.inner = inner
        self.steps = int(steps)
        if not 1 <= self.steps <= 36:
            raise ValueError('execute_prefix_steps must be between 1 and 36')

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def handle_result(self, out, rtt_sec):
        actions = self.inner.handle_result(out, rtt_sec)
        if actions is None:
            return None
        return np.asarray(actions)[:self.steps].copy()
