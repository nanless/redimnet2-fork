"""真实spawn只处理小合成侧车；Linux生命周期mock只验竞态逻辑，不冒充内核实证。"""
import ctypes
import fcntl
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HERE),str(HERE/'tests')]
import union_coverage as U
import benchmark_retained_reads as B
from test_union_coverage import UnionFixture


class RetainedReadProcessTests(unittest.TestCase):
    def setUp(self):self.f=UnionFixture();self.addCleanup(self.f.close)
    def configure(self,workers=2):
        self.f.spec.update(retained_read_backend='process',retained_read_workers=workers);self.f.write_spec()
    def test_actual_spawn_union_matches_serial_and_preserves_old_files_and_environment(self):
        before=self.f.old_snapshot();env={k:os.environ.get(k)for k in U._CPU_ENV}
        expected=U.validate_union(self.f.spec_path,self.f.spec_sha);self.configure()
        actual=U.validate_union(self.f.spec_path,self.f.spec_sha)
        for k in ('source_count','retained_source_count','remaining_source_count','decision_counts','exact_union_and_disjoint'):
            self.assertEqual(actual[k],expected[k])
        self.assertEqual(actual['retained_read_executor'],'process_pool');self.assertEqual(actual['retained_read_process_start_method'],'spawn')
        self.assertEqual(actual['retained_completed_source_reads'],2);self.assertEqual(actual['retained_read_max_submitted_batch'],2)
        self.assertEqual({k:os.environ.get(k)for k in U._CPU_ENV},env);self.assertEqual(before,self.f.old_snapshot())
        if sys.platform!='linux':self.assertEqual(actual['retained_read_process_lifecycle_platform'],'unsupported_nonlinux_cpu_fixture')
        self.assertFalse(multiprocessing.active_children())
    def test_actual_worker_failure_retains_partial_db_without_final_and_restores_environment(self):
        self.configure();source=self.f.base.sources[0];p=self.f.base.tree/source['sidecar_relpath'];p.write_bytes(p.read_bytes()+b' ')
        env={k:os.environ.get(k)for k in U._CPU_ENV};output=self.f.root/'failed-consumer.sqlite'
        with self.assertRaisesRegex(ValueError,'pinned JSON SHA differs'):U.validate_union(self.f.spec_path,self.f.spec_sha,output_index=output)
        self.assertFalse(output.exists());self.assertTrue(list(self.f.root.glob('.consumer-building-*.sqlite')))
        self.assertEqual({k:os.environ.get(k)for k in U._CPU_ENV},env);self.assertFalse(multiprocessing.active_children())
    def test_normal_and_parent_progress_error_drain_actual_process_pool_under_old_lock(self):
        self.configure();lock=self.f.base.queue/'.parent.lock';events=[]
        class Pool(ProcessPoolExecutor):
            def shutdown(pool,wait=True,*,cancel_futures=False):
                with lock.open('rb')as probe:
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:events.append('shutdown_with_old_lock')
                    else:raise AssertionError('process pool shutdown escaped old lock')
                return super().shutdown(wait=wait,cancel_futures=cancel_futures)
        with patch.object(U,'ProcessPoolExecutor',Pool):U.validate_union(self.f.spec_path,self.f.spec_sha)
        self.assertEqual(events,['shutdown_with_old_lock']);events.clear()
        def progress(event):
            if event['stage']=='original_full_and_retained'and event.get('retained_sources_checked',0)>0:
                raise ValueError('parent progress failed after complete batch')
        # 原progress限频；测试只把计时改成每次间隔31s，确保真实父异常发生在读后。
        clock=iter(range(0,100000,31))
        with patch.object(U,'ProcessPoolExecutor',Pool),patch.object(U.time,'monotonic',side_effect=lambda:next(clock)):
            with self.assertRaisesRegex(ValueError,'parent progress failed'):
                U.validate_union(self.f.spec_path,self.f.spec_sha,progress=progress)
        self.assertEqual(events,['shutdown_with_old_lock']);self.assertFalse(multiprocessing.active_children())
    def test_backend_rejects_unknown_values_before_db_and_nonlinux_production_process(self):
        for value in (None,True,2,'fork','',{},[]):
            self.f.spec['retained_read_backend']=value;self.f.write_spec()
            with patch.object(U.sqlite3,'connect')as connect:
                with self.assertRaisesRegex(ValueError,'retained_read_backend'):U.validate_union(self.f.spec_path,self.f.spec_sha)
                connect.assert_not_called()
        with patch.object(U.sys,'platform','darwin'):
            with self.assertRaisesRegex(ValueError,'require Linux'):U.retained_read_backend({'retained_read_backend':'process','retained_read_workers':2,'purpose':'production'})
        self.assertEqual(U.retained_read_backend({'retained_read_backend':'process','retained_read_workers':1,'purpose':'production'}),'process')
    def test_loaded_heavy_module_refuses_before_pool_and_environment_changes(self):
        env={k:os.environ.get(k)for k in U._CPU_ENV}
        with patch.dict(sys.modules,{'torch.fake':object()}),patch.object(U,'ProcessPoolExecutor')as pool:
            with self.assertRaisesRegex(ValueError,'heavy inference'):
                with U.retained_read_pool(2,'process',producer_value={},contract={},purpose='cpu_fixture'):pass
            pool.assert_not_called()
        self.assertEqual({k:os.environ.get(k)for k in U._CPU_ENV},env)
    def test_kernel_guard_sets_and_reads_unignorable_kill_and_checks_parent_both_sides(self):
        calls=[];parents=[]
        def prctl(op,arg,*rest):
            calls.append((op.value,arg.value if hasattr(arg,'value')else None))
            if op.value==2:ctypes.cast(arg,ctypes.POINTER(ctypes.c_int)).contents.value=signal.SIGKILL
            return 0
        def parent():parents.append(123);return 123
        U._linux_arm_parent_death(123,'456',getppid=parent,start_reader=lambda pid:'456',prctl=prctl)
        self.assertEqual(calls,[(1,signal.SIGKILL),(2,None)]);self.assertEqual(len(parents),2)
        with self.assertRaisesRegex(ValueError,'PID changed'):
            U._linux_arm_parent_death(123,'456',getppid=iter([123,1]).__next__,start_reader=lambda pid:'456',prctl=prctl)
        with self.assertRaisesRegex(ValueError,'start identity'):
            U._linux_arm_parent_death(123,'456',getppid=lambda:123,start_reader=lambda pid:'999',prctl=prctl)
        starts=iter(['456','999'])
        with self.assertRaisesRegex(ValueError,'start identity'):
            U._linux_arm_parent_death(123,'456',getppid=lambda:123,start_reader=lambda pid:next(starts),prctl=prctl)
    def test_initializer_source_mismatch_fails_pool_and_restores_environment(self):
        self.configure();env={k:os.environ.get(k)for k in U._CPU_ENV}
        # spawn不会继承mock；父端伪造pins会被子进程重新核实际源码拒绝。
        with patch.object(U,'_cpu_source_pins',return_value={'union_coverage.py':'0'*64}):
            with self.assertRaises(Exception)as caught:U.validate_union(self.f.spec_path,self.f.spec_sha)
        self.assertEqual(type(caught.exception).__name__,'BrokenProcessPool')
        self.assertEqual({k:os.environ.get(k)for k in U._CPU_ENV},env)
        self.assertFalse(multiprocessing.active_children())
    def test_context_exception_and_constructor_failure_restore_hidden_environment(self):
        env={k:os.environ.get(k)for k in U._CPU_ENV}
        with self.assertRaisesRegex(ValueError,'parent-side failure'):
            with U.retained_read_pool(2,'process',producer_value={},contract={},purpose='cpu_fixture'):
                self.assertTrue(all(os.environ.get(k)==v for k,v in U._CPU_ENV.items()))
                raise ValueError('parent-side failure')
        self.assertEqual({k:os.environ.get(k)for k in U._CPU_ENV},env)
        with patch.object(U,'ProcessPoolExecutor',side_effect=ValueError('constructor failure')):
            with self.assertRaisesRegex(ValueError,'constructor failure'):
                with U.retained_read_pool(2,'process',producer_value={},contract={},purpose='cpu_fixture'):pass
        self.assertEqual({k:os.environ.get(k)for k in U._CPU_ENV},env)
    def test_guard_failure_and_readback_mismatch_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'PR_SET'):
            U._linux_arm_parent_death(123,'456',getppid=lambda:123,start_reader=lambda pid:'456',prctl=lambda *a:-1)
        with self.assertRaisesRegex(ValueError,'PR_GET'):
            U._linux_arm_parent_death(123,'456',getppid=lambda:123,start_reader=lambda pid:'456',prctl=lambda *a:0)
        with self.assertRaisesRegex(ValueError,'exact Linux'):
            U._linux_arm_parent_death(123,None)
    def test_benchmark_subprocess_has_unique_stdout_json_worker_receipts_and_identical_proofs(self):
        cmd=[sys.executable,str(HERE/'benchmark_retained_reads.py'),'--plan',str(self.f.plan_path),'--plan-sha',U.sha_file(self.f.plan_path),
             '--positions','0','--modes','serial1,thread4,process2,process4,serial1','--max-sources','2','--purpose','cpu_fixture']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=40,check=True);report=json.loads(result.stdout)
        self.assertTrue(report['all_ordered_proofs_identical']);self.assertEqual(len(report['modes']),5)
        self.assertTrue(all(r['source_count']==2 for r in report['modes']));self.assertEqual(len({r['ordered_proofs_sha256']for r in report['modes']}),1)
        workers=[json.loads(line)for line in result.stderr.splitlines()if line.strip()]
        self.assertTrue(workers);self.assertTrue(all(r['schema']=='annotation-retained-cpu-worker/v1'for r in workers))
        self.assertTrue(all(r['pid']!=r['parent_pid']and r['actual_parent_pid']==r['parent_pid']for r in workers))
        self.assertTrue(all(r['validator_source_files']['union_coverage.py']==U.sha_file(HERE/'union_coverage.py')for r in workers))
        if sys.platform!='linux':self.assertTrue(all(not r['linux_parent_death_signal_verified']for r in workers))
        self.assertIsNone(report['validated_GPU_speedup']);self.assertFalse(report['formal_runtime_admitted'])
    def test_benchmark_bounds_and_broker_pin_do_not_create_old_queues(self):
        for positions,modes,limit in (([],['serial1'],2),([0,0],['serial1'],2),([99],['serial1'],2),([0],['serial1'],1),([0],['invalid'],2),([0],['serial1'],True)):
            with self.subTest(positions=positions,modes=modes,limit=limit):
                with self.assertRaises(ValueError):B.benchmark(self.f.plan_path,U.sha_file(self.f.plan_path),positions,modes,purpose='cpu_fixture',max_sources=limit)
        broker=self.f.base.queue/'broker.json';broker.write_bytes(broker.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'broker differs'):B.benchmark(self.f.plan_path,U.sha_file(self.f.plan_path),[0],['serial1'],purpose='cpu_fixture',max_sources=2)


if __name__=='__main__':unittest.main()
