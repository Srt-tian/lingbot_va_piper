"""Real stateless model + memory-only RobotIO; never opens CAN or cameras."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from smoke_runtime import MemoryRobotIO, Runtime, ROOT
from config import load_config


class IO(MemoryRobotIO):
    def __init__(self):
        super().__init__()
        self.trace = []

    def apply_action(self, action, action_step=None):
        super().apply_action(action)
        self.trace.append((time.monotonic(), dict(action_step)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8014)
    parser.add_argument('--rates', type=float, nargs='+', default=[30, 15, 12])
    args = parser.parse_args()
    os.umask(0)
    output = ROOT/'outputs'/('slow_playback_smoke_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True)
    reports = []
    for rate in args.rates:
        cfg = load_config(ROOT/'integration/client_lingbot_slow_async.yaml').runtime_options()
        cfg.update(host=args.host, port=args.port, transport='websocket',
                   publish_rate=rate, max_publish_step=72, max_prediction_wait_s=30,
                   max_result_age_s=30)
        io = IO()
        rt = Runtime(io, cfg, dict(root_dir=str(output/f'rate_{rate:g}'),
                     record_model_io=True, record_runtime_events=True,
                     record_action_steps=True, record_policy_rollout=False))
        events = []
        log = rt.log_event
        def record(event):
            events.append(event)
            log(event)
        rt.log_event = record
        timer = threading.Timer(90, rt.request_episode_stop)
        timer.start()
        try:
            rt.run()
        finally:
            timer.cancel()
            rt.close()
        assert len(io.applied) == 72, len(io.applied)
        assert np.isfinite(io.applied).all()
        assert rt.inference_count == 2, rt.inference_count
        assert [s['chunk_step_index'] for _, s in io.trace] == list(range(36))*2
        assert all(not t.is_alive() for t in rt.threads)
        dt = [b[0]-a[0] for a, b in zip(io.trace, io.trace[1:])
              if a[1]['chunk_id'] == b[1]['chunk_id']]
        assert min(dt) >= 1/rate
        starts = [e for e in events if e['event'] == 'slow_prefetch_chunk_start']
        ready = [e for e in events if e['event'] == 'slow_prefetch_ready']
        reports.append(dict(playback_rate_hz=rate, actions=72, chunks=2,
            median_action_interval_ms=float(np.median(dt)*1000),
            min_action_interval_ms=min(dt)*1000,
            boundary_gap_ms=starts[1]['boundary_gap_s']*1000,
            cold_start_rtt_ms=ready[0]['rtt_ms'], next_chunk_rtt_ms=ready[1]['rtt_ms'],
            result_age_at_second_chunk_s=starts[1]['result_age_s'],
            all_threads_stopped=True, hardware_commands=0))
        print(json.dumps(reports[-1]), flush=True)
    (output/'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    print('summary:', output/'summary.json')


if __name__ == '__main__':
    main()
