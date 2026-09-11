"""End-to-end official history protocol, actual Runtime/model and memory-only RobotIO."""
import json,time,threading,os
from pathlib import Path
import numpy as np
from smoke_runtime import MemoryRobotIO,Runtime,ROOT
from config import load_config

class IO(MemoryRobotIO):
    def __init__(self):super().__init__();self.trace=[]
    def apply_action(self,action,action_step=None):
        super().apply_action(action);self.trace.append((time.monotonic(),dict(action_step)))

def main():
    os.umask(0)
    cfg=load_config(ROOT/'integration/client_lingbot_history.yaml').runtime_options()
    cfg.update(host='127.0.0.1', port=8015, transport='websocket', max_publish_step=132)
    out=ROOT/'outputs'/('official_smoke_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    io=IO();rt=Runtime(io,cfg,{'root_dir':str(out/'records'),'record_model_io':True,
        'record_runtime_events':True,'record_action_steps':True,'record_policy_rollout':False})
    timer=threading.Timer(90,rt.request_episode_stop);timer.start()
    try:rt.run()
    finally:timer.cancel();rt.close()
    assert len(io.applied)==132,len(io.applied)
    assert np.isfinite(np.asarray(io.applied)).all()
    rows=[json.loads(line) for file in (out/'records').rglob('runtime_events.jsonl') for line in file.read_text().splitlines()]
    predictions=[r for r in rows if r.get('event')=='official_prediction']
    commits=[r for r in rows if r.get('event')=='official_history_committed']
    assert [r['execution_steps'] for r in predictions]==[36,48,48],predictions
    assert [r['frame_start'] for r in predictions]==[0,4,8],predictions
    assert [r['frame_start'] for r in commits]==[4,8],commits
    assert [r['observation_count'] for r in commits]==[12,16],commits
    assert all(not t.is_alive() for t in rt.threads)
    before=len(io.applied);time.sleep(.1);assert len(io.applied)==before
    dt=[b[0]-a[0] for a,b in zip(io.trace,io.trace[1:]) if a[1]['chunk_id']==b[1]['chunk_id']]
    result={'passed':True,'hardware_commands':0,'actions':132,'horizons':[36,48,48],
        'prediction_frame_starts':[0,4,8],'committed_frame_starts':[4,8],
        'all_threads_stopped':True,'min_action_interval_ms':float(min(dt)*1000),
        'median_action_interval_ms':float(np.median(dt)*1000)}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
