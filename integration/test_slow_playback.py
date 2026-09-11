import queue
import threading
import time
import unittest

import numpy as np
from smoke_runtime import MemoryRobotIO, Runtime
from slow_playback import OneChunkPrefetch, validate_config


def config(**changes):
    cfg = dict(execution_mode='async', policy_protocol='stateless', playback_mode='slow_prefetch',
               async_mode='naive', ctrl_type='joint', chunk_size=36, execute_prefix_steps=36,
               publish_rate=30, observation_rate=30, max_publish_step=72,
               max_prediction_wait_s=2, max_result_age_s=5, min_smooth_steps=0, latency_k=0,
               action_buffer='stream', prompt='cloth')
    cfg.update(changes)
    return cfg


class Policy:
    def __init__(self, second_delay=.02, error=False, invalid=False, first_delay=.01):
        self.calls = []
        self.second_delay, self.first_delay = second_delay, first_delay
        self.error, self.invalid = error, invalid
        self.closed = threading.Event()
        self.inflight = 0
        self.max_inflight = 0

    def get_server_metadata(self):
        return dict(policy_family='lingbot_va', action_horizon=36, action_dim=14)

    def infer(self, payload):
        self.calls.append(time.monotonic())
        index = len(self.calls)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            delay = self.first_delay if index == 1 else self.second_delay
            self.closed.wait(delay)
            if self.closed.is_set():
                raise RuntimeError('closed during RPC')
            if self.error and index == 2:
                raise RuntimeError('injected inference failure')
            actions = np.zeros((35 if self.invalid else 36, 14), dtype=np.float32)
            actions[:, 0] = np.arange(len(actions))+(index-1)*36
            return {'actions': actions}
        finally:
            self.inflight -= 1

    def close(self):
        self.closed.set()


class IO(MemoryRobotIO):
    def __init__(self):
        super().__init__()
        self.trace = []
        self.holds = 0
        self.eight_steps = threading.Event()

    def apply_action(self, action, action_step=None):
        super().apply_action(action)
        self.trace.append((time.monotonic(), dict(action_step)))
        if len(self.applied) == 8:
            self.eight_steps.set()

    def hold_current_position(self):
        self.holds += 1


class ConfigAndBufferTests(unittest.TestCase):
    def test_only_explicit_stateless_complete_chunks(self):
        validate_config(config(publish_rate=12))
        for changes in [dict(policy_protocol='official_kv'), dict(execute_prefix_steps=16),
                        dict(publish_rate=float('nan')), dict(publish_rate=31),
                        dict(publish_rate=0), dict(async_mode='temporal_smoothing'),
                        dict(min_smooth_steps=4), dict(max_result_age_s=float('inf')),
                        dict(max_publish_step=0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(config(**changes))

    def test_exactly_one_pending_and_stop_discards_late_result(self):
        state = OneChunkPrefetch()
        state.request()
        with self.assertRaises(RuntimeError): state.request()
        state.deliver('first')
        with self.assertRaises(RuntimeError): state.request()
        self.assertEqual(state.take(), 'first')
        state.request()
        state.close()
        state.deliver('late')
        with self.assertRaises(queue.Empty): state.take()


class RuntimeTests(unittest.TestCase):
    def make(self, policy=None, **changes):
        io = IO()
        rt = Runtime(io, config(**changes), {'record_model_io': False, 'record_runtime_events': False,
                     'record_action_steps': False, 'record_policy_rollout': False})
        rt._policy = policy or Policy()
        events = []
        rt.log_event = events.append
        return rt, io, events

    def run_clean(self, rt):
        try:
            rt.run()
        finally:
            rt.close()
        self.assertTrue(all(not t.is_alive() for t in rt.threads))

    def test_prefetch_overlaps_playback_without_replacing_tail(self):
        policy = Policy()
        rt, io, events = self.make(policy)
        self.run_clean(rt)
        np.testing.assert_array_equal(np.asarray(io.applied)[:, 0], np.arange(72))
        self.assertEqual(len(policy.calls), 2)
        self.assertEqual(policy.max_inflight, 1)
        self.assertLess(policy.calls[1], io.trace[35][0])
        self.assertEqual([x[1]['chunk_step_index'] for x in io.trace], list(range(36))*2)
        self.assertEqual(rt.action_pop_count, 72)
        starts = [e for e in events if e['event'] == 'slow_prefetch_chunk_start']
        self.assertLess(starts[1]['boundary_gap_s'], .1)

    def test_twelve_hz_changes_action_spacing(self):
        rt, io, _ = self.make(publish_rate=12, max_publish_step=3)
        self.run_clean(rt)
        self.assertEqual(len(io.applied), 3)
        self.assertTrue(all(b[0]-a[0] >= 1/12 for a, b in zip(io.trace, io.trace[1:])))

    def test_late_next_chunk_holds_without_replaying_steps(self):
        rt, io, events = self.make(Policy(second_delay=1.4))
        self.run_clean(rt)
        np.testing.assert_array_equal(np.asarray(io.applied)[:, 0], np.arange(72))
        waits = [e for e in events if e['event'] == 'slow_prefetch_wait']
        self.assertTrue(any(e['published_steps'] == 36 for e in waits))
        self.assertGreater(io.trace[36][0]-io.trace[35][0], .1)
        self.assertGreaterEqual(io.holds, 2)

    def test_stop_unblocks_inflight_rpc_and_discards_result(self):
        policy = Policy(second_delay=30)
        rt, io, _ = self.make(policy)
        def stop():
            if io.eight_steps.wait(3): rt.request_episode_stop()
        stopper = threading.Thread(target=stop)
        stopper.start()
        self.run_clean(rt)
        stopper.join(4)
        self.assertFalse(stopper.is_alive())
        self.assertEqual(len(io.applied), 8)
        self.assertEqual(policy.inflight, 0)
        time.sleep(.05)
        self.assertEqual(len(io.applied), 8)

    def test_prediction_failure_stops_current_playback(self):
        rt, io, _ = self.make(Policy(error=True))
        with self.assertRaisesRegex(RuntimeError, 'injected inference failure'):
            self.run_clean(rt)
        self.assertLess(len(io.applied), 36)
        self.assertTrue(all(not t.is_alive() for t in rt.threads))

    def test_wait_timeout_closes_stuck_rpc(self):
        policy = Policy(second_delay=30)
        rt, io, _ = self.make(policy, max_prediction_wait_s=.1)
        with self.assertRaisesRegex(TimeoutError, 'waiting'):
            self.run_clean(rt)
        self.assertEqual(len(io.applied), 36)
        self.assertEqual(policy.inflight, 0)

    def test_invalid_chunk_is_never_published(self):
        rt, io, _ = self.make(Policy(invalid=True))
        with self.assertRaisesRegex(RuntimeError, 'finite stateless actions'):
            self.run_clean(rt)
        self.assertEqual(len(io.applied), 0)

    def test_stale_prediction_is_never_started(self):
        rt, io, _ = self.make(Policy(first_delay=.05), max_result_age_s=.01)
        with self.assertRaisesRegex(TimeoutError, 'too old'):
            self.run_clean(rt)
        self.assertEqual(len(io.applied), 0)


if __name__ == '__main__': unittest.main()
