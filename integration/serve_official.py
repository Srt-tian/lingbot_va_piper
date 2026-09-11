"""Exclusive-session OpenPI-envelope server using the official KV feedback calls."""
import argparse, asyncio, http, json, logging, os, time
from pathlib import Path
import numpy as np
import torch
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from serve_lingbot import LingbotPolicy, ROOT, init_distributed, dist
from contract import CAMERAS, validate_request
from training_preprocessing import resize_observation
from official_flow import OfficialSession
import msgpack_numpy


class OfficialPolicy(LingbotPolicy):
    def __init__(self, record_dir, seed):
        super().__init__(record_dir, seed)
        self.session = OfficialSession(self.model, self.reset_episode)
        self.metadata.update(policy_protocol='official_kv', action_horizon=48,
            initial_action_horizon=36, supported_modes=['sync'], single_session_required=True,
            history_action_conditioning=True, observation_every_actions=3,
            reset_semantics='once per episode; official compute_kv_cache feedback after each full chunk')

    def reset_episode(self, prompt):
        self._set_prompt(prompt)
        self.model._reset(None)
        self.model.prompt_embeds, self.model.negative_prompt_embeds = self.embeddings
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def observation(self, images):
        # Reuse the trained Piper image contract without adding state conditioning.
        obs,_,_,_ = validate_request({'images':images,'state':np.zeros(14,dtype=np.float32)})
        return resize_observation(obs)['obs'][0]

    @torch.no_grad()
    def infer(self, payload):
        if not isinstance(payload,dict):
            raise ValueError('Expected mapping')
        op=payload.get('official_op'); sid=payload.get('session_id')
        started=time.monotonic()
        if op=='reset':
            if set(payload)-{'official_op','session_id','prompt','num_steps'}:
                raise ValueError('Unsupported reset fields')
            steps=payload.get('num_steps',10);prompt=payload.get('prompt')
            if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=50:
                raise ValueError('Invalid sampling steps')
            if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>2048:
                raise ValueError('Invalid prompt')
            self.model.job_config.num_inference_steps=steps
            self.model.job_config.action_num_inference_steps=steps
            self.session.reset(sid,prompt)
            result={'reset':True,'frame_start':0}
        elif op=='predict':
            if set(payload)!={'official_op','session_id','images'}:
                raise ValueError('Invalid predict fields')
            result=self.session.predict(sid,self.observation(payload['images']))
            if not np.isfinite(result['actions']).all():
                raise ValueError('Nonfinite action')
        elif op=='feedback':
            if set(payload)!={'official_op','session_id','chunk_id','observations','actions'}:
                raise ValueError('Invalid feedback fields')
            frames=payload['observations']
            if not isinstance(frames,list) or len(frames) not in (12,16):
                raise ValueError('Expected 12 initial or 16 later observations')
            result=self.session.commit(sid,payload['chunk_id'],[self.observation(f) for f in frames],payload['actions'])
        else:
            raise ValueError('Expected reset, predict, or feedback')
        elapsed=(time.monotonic()-started)*1000
        result['policy_timing']={'infer_ms':elapsed}
        with (self.record_dir/'official_events.jsonl').open('a') as f:
            f.write(json.dumps({'event':op,'session_id':sid,'chunk_id':self.session.chunk_id,
                'frame_start':int(self.model.frame_st_id),'elapsed_ms':elapsed,'timestamp':time.time()})+'\n')
        if op=='predict':
            np.save(self.record_dir/f'{sid}_chunk{self.session.chunk_id}_raw.npy',result['raw_actions'])
        return result


async def run(policy,host,port):
    occupied=False
    def health(connection,request):
        if request.path=='/healthz':return connection.respond(http.HTTPStatus.OK,'OK\n')
    async def handler(ws):
        nonlocal occupied
        if occupied:
            await ws.close(code=1013,reason='One history session at a time');return
        occupied=True
        try:
            await ws.send(msgpack_numpy.packb(policy.metadata))
            async for data in ws:
                request_id=None
                try:
                    if not isinstance(data,bytes):raise ValueError('Binary request required')
                    env=msgpack_numpy.unpackb(data)
                    if not isinstance(env,dict) or env.get('type')!='infer':raise ValueError('Infer envelope required')
                    request_id=env.get('request_id',env.get('request_index'))
                    result=await asyncio.to_thread(policy.infer,env['payload'])
                    await ws.send(msgpack_numpy.packb({'type':'result','request_id':request_id,'payload':result}))
                except Exception as exc:
                    logging.exception('Official request failed')
                    await ws.send(msgpack_numpy.packb({'type':'error','request_id':request_id,'traceback':str(exc)}))
                    await ws.close(code=1011,reason='History session requires reset');return
        except ConnectionClosed:pass
        finally:
            policy.session.phase='reset_required'
            occupied=False
    async with serve(handler,host,port,compression=None,max_size=32*1024**2,process_request=health):
        logging.info('OFFICIAL_KV_READY port=%d',port)
        await asyncio.Future()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8015)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    os.environ.setdefault('MASTER_ADDR','127.0.0.1');os.environ.setdefault('MASTER_PORT','29644')
    init_distributed(world_size=1,local_rank=0,rank=0)
    try:asyncio.run(run(OfficialPolicy(ROOT/'outputs/official_server',7),'0.0.0.0',args.port))
    finally:
        if dist.is_initialized():dist.destroy_process_group()
