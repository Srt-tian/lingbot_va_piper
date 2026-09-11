"""Official LIBERO/Robotwin chunk-feedback state machine, independent of hardware."""
import re
import numpy as np


class ChunkFeedback:
    def __init__(self, raw, first):
        self.raw = np.asarray(raw, dtype=np.float32).copy()
        if self.raw.shape != (14, 4, 12) or not np.isfinite(self.raw).all():
            raise ValueError('Expected finite raw actions [14,4,12]')
        self.start = 1 if first else 0
        self.actions = self.raw[:, self.start:].transpose(1, 2, 0).reshape(-1, 14).copy()
        self.executed = 0
        self.observations = []

    def action_executed(self):
        if self.executed >= len(self.actions):
            raise ValueError('Chunk already complete')
        self.executed += 1
        return self.executed % 3 == 0

    def observe(self, frame):
        if self.executed % 3 or len(self.observations) != self.executed // 3 - 1:
            raise ValueError('Observation must follow every third executed action')
        self.observations.append(frame)

    def feedback(self):
        if self.executed != len(self.actions) or len(self.observations) != self.executed // 3:
            raise ValueError('Partial execution cannot be committed as complete history')
        return {'obs': self.observations, 'state': self.raw.copy()}


class OfficialSession:
    """Enforce reset -> predict -> complete-feedback -> predict; never cache prediction tails."""
    def __init__(self, model, reset_model):
        self.model, self.reset_model = model, reset_model
        self.session_id = None
        self.phase = 'reset_required'
        self.chunk_id = 0
        self.raw = None
        self.first = True

    def reset(self, session_id, prompt):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError('Invalid episode session ID')
        self.phase = 'reset_required'
        self.reset_model(prompt)
        self.session_id, self.phase = session_id, 'ready'
        self.chunk_id, self.raw, self.first = 0, None, True

    def check(self, session_id, phase):
        if session_id != self.session_id or self.phase != phase:
            raise ValueError(f'History session/phase mismatch: expected {self.phase}')

    def predict(self, session_id, observation):
        self.check(session_id, 'ready')
        frame_start = int(self.model.frame_st_id)
        try:
            result = self.model.infer({'obs': observation})
            feedback = ChunkFeedback(result['action'], self.first)
        except Exception:
            self.phase = 'reset_required'
            raise
        self.raw = feedback.raw.copy()
        self.chunk_id += 1
        self.phase = 'feedback_required'
        return {'actions': feedback.actions, 'raw_actions': self.raw.copy(),
                'chunk_id': self.chunk_id, 'frame_start': frame_start,
                'first_chunk': self.first, 'expected_observations': 12 if self.first else 16}

    def commit(self, session_id, chunk_id, observations, actions):
        self.check(session_id, 'feedback_required')
        expected = 12 if self.first else 16
        actions = np.asarray(actions, dtype=np.float32)
        if chunk_id != self.chunk_id or len(observations) != expected:
            raise ValueError('Feedback chunk ID or observation count mismatch')
        if actions.shape != (14,4,12) or not np.isfinite(actions).all() or not np.array_equal(actions, self.raw):
            raise ValueError('Feedback must contain the full command chunk that was executed')
        before = int(self.model.frame_st_id)
        try:
            self.model.infer({'compute_kv_cache': True, 'imagine': False,
                              'obs': observations, 'state': actions})
            if int(self.model.frame_st_id) != before + 4:
                raise RuntimeError('Official KV update must advance exactly four latent frames')
        except Exception:
            self.phase = 'reset_required'
            raise
        self.phase, self.first = 'ready', False
        self.raw = None
        return {'history_committed': True, 'frame_start': int(self.model.frame_st_id)}
