import sys, threading, time, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from client_entry import install_adapter
install_adapter(Path(__file__).resolve().parents[1] / 'runtime')
from action_buffers import NaiveAsyncBuffer, StreamActionBuffer
from async_runtime_adapter import AtomicActionBuffer, control_loop

class BufferTests(unittest.TestCase):
    def test_prefix_excludes_tail_before_buffer(self):
        from async_runtime_adapter import PrefixActionMode
        inner=SimpleNamespace(handle_result=lambda out,dt:out['actions'])
        mode=PrefixActionMode(inner,8)
        prediction=np.arange(36)[:,None]
        prefix=mode.handle_result({'actions':prediction},.1)
        self.assertEqual(prefix.shape,(8,1))
        b=AtomicActionBuffer(StreamActionBuffer(smooth_method='temporal_smoothing'))
        b.integrate_new_chunk(prefix,max_k=0,min_m=4)
        for i in range(80):
            step=b.pop_next_action()
            self.assertEqual(step['action'][0],min(i,7))
            self.assertEqual(step['chunk_step_index'],min(i,7))
            self.assertEqual(step['is_hold'],i>=8)

    def test_empty_hold_and_resume(self):
        for inner in (NaiveAsyncBuffer(36, 1), StreamActionBuffer(smooth_method='temporal_smoothing')):
            b=AtomicActionBuffer(inner)
            for _ in range(100): self.assertIsNone(b.pop_next_action())
            self.assertEqual(b.get_chunk_progress()['executed_steps'],0)
            b.integrate_new_chunk(np.arange(36)[:,None],max_k=0,chunk_id=1)
            for i in range(36):
                s=b.pop_next_action(); self.assertEqual(s['chunk_step_index'],i);self.assertFalse(s['is_hold'])
            for _ in range(80):
                s=b.pop_next_action();self.assertTrue(s['is_hold']);self.assertEqual(s['chunk_step_index'],35);self.assertEqual(s['action'][0],35)
            self.assertEqual(b.get_chunk_progress()['executed_steps'],36)
            b.integrate_new_chunk((100+np.arange(36))[:,None],max_k=0,min_m=12,chunk_id=2)
            s=b.pop_next_action();self.assertEqual(s['chunk_step_index'],0);self.assertFalse(s['is_hold'])
            b.close();self.assertIsNone(b.integrate_new_chunk(np.ones((36,1)),max_k=0));self.assertIsNone(b.pop_next_action())

    def test_switch_pop_interleaving(self):
        b=AtomicActionBuffer(NaiveAsyncBuffer(36,1));b.integrate_new_chunk(np.arange(36)[:,None],max_k=0)
        entered=threading.Event();release=threading.Event();consumer_started=threading.Event();consumed=threading.Event()
        original=b.inner.get_current_timestep
        def paused_clock():
            t=original();entered.set();release.wait(2);return t
        b.inner.get_current_timestep=paused_clock
        switches=[];steps=[]
        producer=threading.Thread(target=lambda:switches.append(b.integrate_new_chunk((100+np.arange(36))[:,None],max_k=0)))
        def consume():
            consumer_started.set();steps.append(b.pop_next_action());consumed.set()
        consumer=threading.Thread(target=consume)
        producer.start();self.assertTrue(entered.wait(1));consumer.start();self.assertTrue(consumer_started.wait(1))
        try:self.assertFalse(consumed.wait(.03))
        finally:release.set();producer.join(2);consumer.join(2)
        self.assertEqual(switches[0]['dropped_new_chunk_steps'],0);self.assertEqual(steps[0]['action'][0],100)

    def test_reference_smoothing_boundary_and_tail(self):
        b=AtomicActionBuffer(StreamActionBuffer(smooth_method='temporal_smoothing'))
        b.integrate_new_chunk(np.zeros((36,1)),max_k=0)
        for _ in range(80):b.pop_next_action()
        b.integrate_new_chunk(np.ones((36,1)),max_k=0,min_m=12)
        a=np.array([b.pop_next_action()['action'][0] for _ in range(36)])
        self.assertEqual(a[0],0);np.testing.assert_allclose(a[:12],np.linspace(0,1,12));np.testing.assert_array_equal(a[12:],1)

class SchedulerTests(unittest.TestCase):
    def make_rt(self, maximum=4):
        b=AtomicActionBuffer(NaiveAsyncBuffer(36,1));times=[]
        rt=SimpleNamespace(cfg={'publish_rate':30,'max_publish_step':maximum},shutdown=threading.Event(),
            _publish_lock=threading.RLock(),stream_buffer=b,first_action_publish_monotonic=None,telemetry=None,
            io=SimpleNamespace(apply_action=lambda *a,**kw:times.append(time.monotonic())),
            _record_policy_rollout=lambda *a:None,_log_action_step=lambda *a:None,_maybe_log_action_publish_rate=lambda:None)
        return rt,times
    def test_unlimited_run_exits_on_stop(self):
        rt,times=self.make_rt(maximum=None)
        rt.stream_buffer.integrate_new_chunk(np.ones((36,1)),max_k=0)
        timer=threading.Timer(.15,rt.shutdown.set);timer.start()
        control_loop(rt);timer.join()
        self.assertGreater(len(times),0);count=len(times);time.sleep(.05);self.assertEqual(len(times),count)

    def test_first_steps_after_empty_buffer_are_paced(self):
        rt,times=self.make_rt();timer=threading.Timer(.1,lambda:rt.stream_buffer.integrate_new_chunk(np.ones((36,1)),max_k=0));timer.start()
        control_loop(rt);timer.join();self.assertEqual(len(times),4);self.assertTrue(all(dt>=1/30 for dt in np.diff(times)),times)
    def test_stop_during_empty_wait_does_not_publish_late_result(self):
        rt,times=self.make_rt();timer=threading.Timer(.03,rt.shutdown.set);timer.start();control_loop(rt);timer.join()
        rt.stream_buffer.integrate_new_chunk(np.ones((36,1)),max_k=0);self.assertEqual(times,[])

if __name__=='__main__':unittest.main()
