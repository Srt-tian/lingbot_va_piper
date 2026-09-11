import ast,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from official_flow import ChunkFeedback,OfficialSession

class FakeModel:
    def __init__(self):self.frame_st_id=0;self.calls=[]
    def infer(self,payload):
        self.calls.append(payload)
        if payload.get('compute_kv_cache'):
            self.frame_st_id+=4;return {}
        return {'action':(np.arange(14*4*12,dtype=np.float32).reshape(14,4,12)+self.frame_st_id)}

class FlowTests(unittest.TestCase):
    def session(self):
        model=FakeModel();resets=[]
        def reset(prompt):model.frame_st_id=0;resets.append(prompt)
        session=OfficialSession(model,reset);session.reset('episode_a','cloth')
        return model,session,resets
    def test_three_chunks_follow_official_order_and_history(self):
        model,session,resets=self.session()
        for i,steps in enumerate((36,48,48)):
            result=session.predict('episode_a',{'frame':'initial'})
            self.assertEqual(result['frame_start'],4*i)
            chunk=ChunkFeedback(result['raw_actions'],i==0)
            self.assertEqual(len(chunk.actions),steps)
            self.assertEqual(chunk.actions[0,0],12 if i==0 else 4*i)
            for j in range(steps):
                if chunk.action_executed():chunk.observe({'after_action':j+1})
            feedback=chunk.feedback()
            session.commit('episode_a',result['chunk_id'],feedback['obs'],feedback['state'])
            self.assertEqual(len(model.calls[-1]['obs']),steps//3)
            self.assertEqual(model.frame_st_id,4*(i+1))
        self.assertEqual(len(resets),1)
    def test_cannot_reinfer_or_commit_prediction_tail(self):
        _,s,_=self.session();r=s.predict('episode_a',{})
        with self.assertRaises(ValueError):s.predict('episode_a',{})
        with self.assertRaises(ValueError):s.commit('episode_a',1,[{}]*4,r['raw_actions'])
        bad=r['raw_actions'].copy();bad[0,1,0]+=1
        with self.assertRaises(ValueError):s.commit('episode_a',1,[{}]*12,bad)
        self.assertEqual(s.phase,'feedback_required')
    def test_partial_stop_never_commits_complete_feedback(self):
        chunk=ChunkFeedback(np.zeros((14,4,12)),True)
        for _ in range(16):
            if chunk.action_executed():chunk.observe({})
        with self.assertRaises(ValueError):chunk.feedback()
    def test_reset_and_cross_episode_rejection(self):
        m,s,resets=self.session();r=s.predict('episode_a',{})
        with self.assertRaises(ValueError):s.commit('episode_b',1,[{}]*12,r['raw_actions'])
        s.reset('episode_b','new');self.assertEqual(s.phase,'ready')
        self.assertTrue(s.predict('episode_b',{})['first_chunk'])
        self.assertEqual(resets,['cloth','new'])

class TrainingSpanTests(unittest.TestCase):
    def test_exact_segments_and_explicit_unique_fallback(self):
        source=Path(__file__).resolve().parents[1]/'wan_va/dataset/lerobot_latent_dataset.py'
        node=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_find_latent_span')
        ns={'Path':Path};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        with tempfile.TemporaryDirectory() as tmp:
            keys=['top','left','right']
            for k in keys:
                d=Path(tmp)/'chunk-000'/k;d.mkdir(parents=True)
                for a,b in [(0,48),(48,96)]: (d/f'episode_000000_{a}_{b}.pth').touch()
            obj=SimpleNamespace(meta=SimpleNamespace(get_episode_chunk=lambda _:0),latent_path=tmp,used_video_keys=keys)
            fn=ns['_find_latent_span']
            self.assertEqual(fn(obj,0,48,96),(48,96))
            with self.assertRaises(ValueError):fn(obj,0,0,96,allow_unique_fallback=True)
            for k in keys:(Path(tmp)/'chunk-000'/k/'episode_000000_48_96.pth').unlink()
            self.assertEqual(fn(obj,0,0,96,allow_unique_fallback=True),(0,48))
            with self.assertRaises(ValueError):fn(obj,0,0,96)

if __name__=='__main__':unittest.main()
