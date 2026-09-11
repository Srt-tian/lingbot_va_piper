"""Real async Runtime/server, memory-only actions; checks chunk refresh and stop."""
import json, time, threading, os
from pathlib import Path
import numpy as np
from smoke_runtime import MemoryRobotIO, Runtime, ROOT
from config import load_config
class IO(MemoryRobotIO):
    def __init__(self):
        super().__init__(); self.trace=[]
    def apply_action(self, action, action_step=None):
        super().apply_action(action)
        self.trace.append((time.monotonic(),action_step['chunk_id'],action_step['chunk_step_index'],action_step.get('is_hold',False)))
def main():
    os.umask(0)
    cfg=load_config(ROOT/'integration/client_lingbot_async.yaml').runtime_options()
    cfg.update(host='127.0.0.1',port=8014,max_publish_step=180)
    out=ROOT/'outputs'/('async_smoke_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir()
    io=IO();rt=Runtime(io,cfg,{'root_dir':str(out/'records'),'record_model_io':False,'record_runtime_events':True,'record_action_steps':True})
    timer=threading.Timer(35,rt.request_episode_stop);timer.start()
    try:rt.run()
    finally:timer.cancel();rt.close()
    a=np.asarray(io.applied); count=len(a);time.sleep(.2)
    assert a.shape==(180,14) and np.isfinite(a).all(),a.shape
    assert len(io.applied)==count and all(not t.is_alive() for t in rt.threads)
    ids=set(x[1] for x in io.trace);assert len(ids)>=2,ids
    held=sum(x[3] for x in io.trace);assert held>0
    assert all(x[2] == int(cfg.get("execute_prefix_steps", 36))-1 for x in io.trace if x[3])
    assert all(0 <= x[2] < int(cfg.get("execute_prefix_steps", 36)) for x in io.trace)
    assert min(np.diff([x[0] for x in io.trace])) >= 1/30
    event_rows=[json.loads(line) for file in (out/'records').rglob('runtime_events.jsonl') for line in file.read_text().splitlines()]
    buffer_rows=[r for r in event_rows if r.get('event')=='action_buffer_step']
    assert len(buffer_rows)==180 and sum(r['is_hold'] for r in buffer_rows)==held
    np.save(out/'memory_actions.npy',a)
    report={'status':'passed','shape':list(a.shape),'chunks_published':len(ids),'held_steps':held,'all_threads_stopped':True,'min_publish_interval_ms':float(min(np.diff([x[0] for x in io.trace]))*1000),'effective_publish_hz':179/(io.trace[-1][0]-io.trace[0][0]),'hardware_commands':0}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
