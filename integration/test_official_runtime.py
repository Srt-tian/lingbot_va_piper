"""Stop/failure tests through the existing Runtime, using memory IO and a fake policy."""
import threading
import time
import unittest
import numpy as np
from smoke_runtime import MemoryRobotIO, Runtime, ROOT
from config import load_config
from official_flow import ChunkFeedback


class Policy:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.predicting = threading.Event()

    def get_server_metadata(self):
        return {'policy_protocol': 'official_kv', 'action_horizon': 48}

    def infer(self, payload):
        op = payload['official_op']
        self.calls.append(op)
        if op == 'reset':
            return {}
        if op == 'feedback':
            raise RuntimeError('injected feedback failure')
        self.predicting.set()
        if self.mode == 'during_rpc':
            time.sleep(.15)
        raw = np.zeros((14, 4, 12), dtype=np.float32)
        return {'raw_actions': raw, 'actions': ChunkFeedback(raw, True).actions,
                'chunk_id': 1, 'frame_start': 0}

    def close(self):
        pass


class IO(MemoryRobotIO):
    def __init__(self):
        super().__init__()
        self.prefix_done = threading.Event()

    def apply_action(self, action, action_step=None):
        super().apply_action(action)
        if len(self.applied) == 16:
            self.prefix_done.set()


class RuntimeStopTests(unittest.TestCase):
    def check_case(self, mode, expected):
        cfg = load_config(ROOT/'integration/client_lingbot_history.yaml').runtime_options()
        io, policy = IO(), Policy(mode)
        rt = Runtime(io, cfg, {'record_model_io': False, 'record_runtime_events': False,
                              'record_action_steps': False, 'record_policy_rollout': False})
        rt._policy = policy
        worker = None
        if mode != 'feedback_failure':
            event = policy.predicting if mode == 'during_rpc' else io.prefix_done
            def stop():
                if event.wait(5):
                    rt.request_episode_stop()
            worker = threading.Thread(target=stop)
            worker.start()
        try:
            if mode == 'feedback_failure':
                with self.assertRaisesRegex(RuntimeError, 'injected feedback failure'):
                    rt.run()
            else:
                rt.run()
        finally:
            rt.close()
            if worker:
                worker.join(6)
        self.assertEqual(len(io.applied), expected)
        self.assertEqual(rt.action_pop_count, expected)
        self.assertEqual(policy.calls.count('predict'), 1)
        self.assertEqual(policy.calls.count('feedback'), int(mode == 'feedback_failure'))
        self.assertTrue(all(not thread.is_alive() for thread in rt.threads))
        time.sleep(.05)
        self.assertEqual(len(io.applied), expected)
        self.assertEqual(rt.action_pop_count, expected)

    def test_stop_during_rpc_discards_result(self):
        self.check_case('during_rpc', 0)

    def test_stop_after_prefix_does_not_commit_tail(self):
        self.check_case('partial', 16)

    def test_feedback_failure_does_not_predict_again(self):
        self.check_case('feedback_failure', 36)


if __name__ == '__main__':
    unittest.main()
