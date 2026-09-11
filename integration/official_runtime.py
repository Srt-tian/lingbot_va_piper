"""Plug official chunk feedback into the existing inference Runtime and RobotIO."""
import time
import threading
import uuid
import numpy as np
import cv2
from official_flow import ChunkFeedback


def run_official_loop(rt):
    # Reuse the existing acquisition thread so camera reads cannot slow the action cadence.
    observer = threading.Thread(target=rt._observation_thread, name='official-observation', daemon=True)
    rt.threads.append(observer)
    observer.start()
    def observation():
        while not rt.shutdown.is_set():
            value = rt._get_latest_observation()
            if value is not None:
                return value
            rt.shutdown.wait(.005)
        return None
    session_id = uuid.uuid4().hex
    prompt = str(rt.cfg.get('prompt', 'flat the cloth with both robot arms'))
    rt.policy.infer({'official_op': 'reset', 'session_id': session_id, 'prompt': prompt,
                     'num_steps': int(rt.cfg.get('num_denoising_steps', 10))})
    first = True
    published = 0
    maximum = rt.cfg.get('max_publish_step')
    period = 1 / float(rt.cfg.get('publish_rate', 30))
    while not rt.shutdown.is_set():
        obs = observation()
        if obs is None:
            return
        payload, raw_images, state = rt._base_payload(obs)
        started = time.monotonic()
        result = rt.policy.infer({'official_op': 'predict', 'session_id': session_id,
                                  'images': payload['images']})
        if rt.shutdown.is_set():
            return
        chunk = ChunkFeedback(result['raw_actions'], first)
        if not np.array_equal(chunk.actions, np.asarray(result['actions'], dtype=np.float32)):
            raise ValueError('Server flattened action order differs from official chunk order')
        rt.inference_count += 1
        rt.log_event({'event': 'official_prediction', 'chunk_id': result['chunk_id'],
                      'frame_start': result['frame_start'], 'execution_steps': len(chunk.actions),
                      'rtt_ms': (time.monotonic()-started)*1000})
        # Fixed 48-row storage includes the first conditioning block, unlike execution CSV.
        if rt.recorder is not None:
            rt.recorder.record_model_io(payload={'state':state,'prompt':prompt},
                model_output_actions=chunk.raw.transpose(1,2,0).reshape(48,14),
                timestamp_sec=time.time(),raw_images=raw_images)
        for i, action in enumerate(chunk.actions):
            with rt._publish_lock:
                if rt.shutdown.is_set():
                    return
                step = {'chunk_id': result['chunk_id'], 'chunk_step_index': i,
                        'raw_action_index': i + (12 if first else 0), 'is_hold': False,
                        'action': action.copy()}
                if rt.first_action_publish_monotonic is None:
                    rt.first_action_publish_monotonic = time.monotonic()
                rt.io.apply_action(action, action_step=step)
                rt._record_policy_rollout(action, step)
                rt._log_action_step(action, step)
                rt._maybe_log_action_publish_rate()
                published += 1
                rt.action_pop_count += 1
                take_observation = chunk.action_executed()
            # Let this action's control interval elapse before observing its effect.
            if rt.shutdown.wait(period):
                return
            if take_observation:
                observed = observation()
                if observed is None:
                    return
                frame_payload, _, _ = rt._base_payload(observed)
                # Resize with the exact training kernel before bundling 16 x 3 frames.
                # Native 640 x 480 would exceed the WebSocket message limit.
                history_images = {k: np.ascontiguousarray(cv2.resize(v.transpose(1,2,0),
                    (256,256), interpolation=cv2.INTER_AREA).transpose(2,0,1))
                    for k,v in frame_payload['images'].items()}
                chunk.observe(history_images)
                rt.log_event({'event': 'official_history_observation', 'chunk_id': result['chunk_id'],
                              'after_action': i+1, 'image_timestamp': observed.get('image_timestamp'),
                              'state_timestamp': observed.get('state_timestamp')})
            if maximum is not None and published >= int(maximum):
                rt.shutdown.set()
                return
        if rt.shutdown.is_set():
            return
        feedback = chunk.feedback()
        acknowledged = rt.policy.infer({'official_op': 'feedback', 'session_id': session_id,
            'chunk_id': result['chunk_id'], 'observations': feedback['obs'], 'actions': feedback['state']})
        rt.log_event({'event': 'official_history_committed', 'chunk_id': result['chunk_id'],
                      'observation_count': len(feedback['obs']), **acknowledged})
        first = False
