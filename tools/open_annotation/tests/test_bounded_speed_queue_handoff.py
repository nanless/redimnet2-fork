"""有界新八卡等待的真实本机文件CPU反例；无GPU、无平台操作。"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import bounded_speed_queue_handoff as M
import bounded_speed_handoff as H
import queue_handoff as F
import test_bounded_speed_runtime as RuntimeFixtures


class Clock:
    def __init__(self):self.wall=float(int(time.time()));self.mono=100.
    def now(self):return self.wall
    def monotonic(self):return self.mono
    def sleep(self,n):self.wall+=n;self.mono+=n


class QueueFixture(unittest.TestCase):
    def setUp(self):
        self.parent=RuntimeFixtures.RuntimeFixture();self.parent.setUp();self.addCleanup(self.parent.doCleanups);self.f=self.parent.f;self.clock=Clock()
        self.put=self.parent.put;self.identity=dict(hostname='t-20261028150000-new8-worker-0',task_id='t-20261028150000-new8',pid=999,start_ticks=123,boot_id='11111111-2222-3333-4444-555555555555')
        run=Path(self.parent.runtime['new_run_root']);run.rmdir()
        self.base={k:v for k,v in self.parent.runtime.items()if k not in {'index','current_platform_observation'}}
        code=Path(self.base['new_code_root']);members={}
        for name in M.REQUIRED_SOURCE:
            target=code/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((HERE/name).read_bytes());members[name]=hashlib.sha256(target.read_bytes()).hexdigest()
        self.base['new_bundle']=self.put(str((code/'bundle_files.json').relative_to(self.f.root)),members)
        python=self.f.root/'python-fixture';python.write_text('CPU fixture; never execute')
        self.value=dict(schema=M.SCHEMA,purpose='cpu_fixture',task_name=self.base['task_name'],runtime_base=self.base,handoff_spec=dict(path=str(self.parent.fixture.spec),sha256=self.parent.fixture.pin),final_handoff_root=str(self.f.root/'late-final'),handoff_root=str(self.f.root/'late-wait'),python_binary=F.file_pin(python),wait_seconds=12,required_prior_attempt_ids=[],gpu_runtime_seconds=60,memory_budget_mib=60000)
        outer=M.build_controller(code,self.base['new_bundle'],self.f.root/'outer-control')
        self.value['outer_controller']=outer
        self.pin=self.put('late-blueprint.json',self.value);self.calls=[]

    def save(self):self.pin=self.put('late-blueprint.json',self.value)
    def resources(self):
        return dict(devices=[dict(uuid='GPU-'+str(i),name='NVIDIA A800',total_mib=81920,used_mib=0,free_mib=81920)for i in range(8)],compute_processes=[])
    def request(self):
        c=M.check(self.pin);directory=F.mkdir_new(F.mkdir_new(c['handoff_root'])/'request-v1')
        resource=F.write_new(directory/'resource-snapshot.json',dict(identity=self.identity,snapshot=self.resources(),GPU_context_created=False))
        v=dict(schema=M.REQUEST,blueprint=self.pin,identity=self.identity,nonce='a'*64,created_utc=F._utc(self.clock.now()),wait_seconds=self.value['wait_seconds'],reply_path=str(directory/'reply.json'),resource_snapshot=resource)
        return F.write_new(directory/'request.json',v),v
    def observed(self,request):
        self.f.write_terminal(active_workers=0)
        # 合成时钟允许等待前进；终态使用真实当前时钟，H仍实际执行原一小时时效门。
        observed=json.loads(self.f.terminal.read_text())['observed_utc'];self.clock.wall=max(self.clock.wall,datetime.datetime.fromisoformat(observed).timestamp())
        v=dict(schema='annotation-bounded-speed-platform-observation/v1',old_task_id=self.f.task,old_queue_root=str(self.f.queue),observed_utc=observed,competing_successor_tasks=[],remote_proc_enumeration_performed=False,evidence_basis='platform_terminal_and_current_shared_locks',terminal_observation=dict(path=str(self.f.terminal),sha256=self.f.tsha),prior_attempts=[],current_successor_task=dict(task_id=request['identity']['task_id'],task_name=self.base['task_name'],state='Running',active_workers=1,workers=[dict(name=request['identity']['hostname'],state='Running')]))
        return self.put('late-observation.json',v)
    def publish(self,request_pin,request):return M.publish(self.pin,request_pin,self.observed(request),clock=self.clock.now)
    def run_wait(self,sleeper=None,identify=None):
        return M.wait(self.pin,execute=True,runner=lambda argv,env:self.calls.append((argv,env))or {'synthetic_CPU_exec':True},clock=self.clock.now,monotonic=self.clock.monotonic,sleeper=sleeper or self.clock.sleep,identity_provider=identify or (lambda:self.identity),resource_provider=self.resources)
    def on_sleep(self,mutate=None):
        called=False
        def step(seconds):
            nonlocal called
            self.clock.sleep(seconds)
            if not called:
                called=True;path=Path(self.value['handoff_root'])/'request-v1/request.json';pin=F.file_pin(path,private=True);request=F.ref(pin,private=True)
                self.publish(pin,request)
                if mutate:mutate(request)
        return step

    def test_check_active_old_writer_only_reads_control_without_outputs(self):
        with (self.f.queue/'.parent.lock').open('rb')as fd:
            fcntl.flock(fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch('sqlite3.connect',side_effect=AssertionError('no DB')),patch.object(H.P,'read_shard',side_effect=AssertionError('no source reads before terminal')):
                v=M.wait(self.pin)
        self.assertTrue(v['passed']);self.assertFalse(Path(self.value['handoff_root']).exists());self.assertFalse(Path(self.value['final_handoff_root']).exists())

    def test_wait_then_actual_bounded_publish_routes_fixed_auto8(self):
        before=self.f.snapshot();calls=[];real=H.P.read_shard
        with patch('sqlite3.connect',side_effect=AssertionError('no DB')),patch.object(H.P,'read_shard',side_effect=lambda root,shard:calls.append(shard['id'])or real(root,shard)):
            v=self.run_wait(self.on_sleep())
        self.assertTrue(v['synthetic_CPU_exec']);self.assertEqual(len(self.calls),1);argv,env=self.calls[0]
        self.assertEqual(argv[argv.index('--target-uuids')+1],'auto8');self.assertIn('--bounded-runtime-spec',argv);self.assertNotIn('--full-verify',argv)
        self.assertTrue(calls);self.assertTrue(set(calls)<={'s1','partial-s1'})
        self.assertEqual(self.f.snapshot(),before);self.assertNotIn('RANK',env)

    def test_timeout_preserves_request_and_has_no_gpu_run(self):
        with self.assertRaisesRegex(ValueError,'timed out'):self.run_wait()
        d=Path(self.value['handoff_root'])/'request-v1';self.assertTrue((d/'request.json').is_file());self.assertTrue((d/'waiter-failure.json').is_file());self.assertFalse(Path(self.base['new_run_root']).exists());self.assertEqual(self.calls,[])

    def test_old_parent_still_held_refuses_final_preparation_and_preserves_failure(self):
        p,r=self.request();obs=self.observed(r)
        with (self.f.queue/'.parent.lock').open('rb')as fd:
            fcntl.flock(fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'still owned'):M.publish(self.pin,p,obs,clock=self.clock.now)
        self.assertTrue((Path(p['path']).parent/'publication-v1/publisher-failure.json').is_file());self.assertFalse(Path(r['reply_path']).exists())

    def test_old_task_running_cannot_be_published(self):
        p,r=self.request();obs=self.observed(r);v=F.ref(obs);term=F.ref(v['terminal_observation']);term['state']='Running';v['terminal_observation']=self.put('running-terminal.json',term);obs=self.put('running-observation.json',v)
        with self.assertRaisesRegex(ValueError,'must be terminal'):M.publish(self.pin,p,obs,clock=self.clock.now)
        self.assertFalse(Path(self.value['final_handoff_root']).exists())

    def test_observation_of_another_new_task_refuses(self):
        p,r=self.request();v=F.ref(self.observed(r));v['current_successor_task']['task_id']='t-20261028150001-wrong'
        with self.assertRaisesRegex(ValueError,'actual new allocated'):M.publish(self.pin,p,self.put('wrong-new.json',v),clock=self.clock.now)

    def test_observation_must_follow_actual_request(self):
        p,r=self.request();v=F.ref(self.observed(r));v['observed_utc']=F._utc(self.clock.now()-60)
        with self.assertRaisesRegex(ValueError,'follow actual'):M.publish(self.pin,p,self.put('old-observation.json',v),clock=self.clock.now)

    def test_reply_nonce_spoof_refused(self):
        def tamper(r):
            p=Path(r['reply_path']);v=F.ref(F.file_pin(p,private=True),private=True);v['request_nonce']='b'*64;p.write_bytes(F.canonical(v))
        with self.assertRaisesRegex(ValueError,'nonce/blueprint'):self.run_wait(self.on_sleep(tamper))
        self.assertEqual(self.calls,[])

    def test_runtime_cannot_replace_new_producer_at_reply(self):
        def tamper(r):
            reply=F.ref(F.file_pin(r['reply_path'],private=True),private=True);pin=reply['runtime_spec'];v=F.ref(pin,private=True);v['new_run_root']=str(self.f.root/'another-run');Path(pin['path']).write_bytes(F.canonical(v));reply['runtime_spec']=F.file_pin(pin['path'],private=True);Path(r['reply_path']).write_bytes(F.canonical(reply))
        with self.assertRaisesRegex(ValueError,'changed beyond'):self.run_wait(self.on_sleep(tamper))
        self.assertEqual(self.calls,[])

    def test_wait_process_identity_drift_refuses_before_gpu(self):
        count=0
        def identify():
            nonlocal count
            count+=1;return dict(self.identity,pid=1000)if count>1 else self.identity
        with self.assertRaisesRegex(ValueError,'process identity changed'):self.run_wait(identify=identify)
        self.assertEqual(self.calls,[])

    def test_duplicate_publish_never_overwrites_existing_reply(self):
        p,r=self.request();self.publish(p,r);before=Path(r['reply_path']).read_bytes()
        with self.assertRaisesRegex(ValueError,'reply exists'):self.publish(p,r)
        self.assertEqual(Path(r['reply_path']).read_bytes(),before)

    def test_signal_during_wait_preserves_evidence_and_restores_handlers(self):
        old=signal.getsignal(signal.SIGTERM)
        def interrupt(n):signal.raise_signal(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError,'interrupted before GPU'):self.run_wait(interrupt)
        self.assertEqual(signal.getsignal(signal.SIGTERM),old);self.assertEqual(self.calls,[])
        self.assertTrue((Path(self.value['handoff_root'])/'request-v1/waiter-failure.json').exists())

    def test_handoff_cannot_overlap_running_ptn_or_old_annotation(self):
        self.value['handoff_root']=str(self.f.queue/'wait');self.save()
        with self.assertRaisesRegex(ValueError,'overlaps old'):M.wait(self.pin)

    def test_wait_limit_is_not_extended_for_eight_gpu_queue(self):
        self.value['wait_seconds']=1801;self.save()
        with self.assertRaisesRegex(ValueError,'1800 seconds'):M.wait(self.pin)

    def test_no_model_api_or_label_release_may_be_enabled(self):
        self.base['model_API_calls_allowed']=1;self.save()
        with self.assertRaisesRegex(ValueError,'offline'):M.wait(self.pin)

    def test_new_bundle_changed_during_wait_refuses_before_exec(self):
        def tamper(r):(Path(self.base['new_code_root'])/'worker.py').write_text('changed source')
        with self.assertRaisesRegex(ValueError,'frozen bundle member changed'):self.run_wait(self.on_sleep(tamper))
        self.assertEqual(self.calls,[])

    def test_production_rejects_synthetic_runtime_hooks(self):
        c=M.check(self.pin);c['blueprint']['purpose']='production'
        with patch.object(M,'check',return_value=c),self.assertRaisesRegex(ValueError,'cannot inject'):self.run_wait()

    def test_waiter_refuses_unknown_compute_pid_without_any_cuda(self):
        bad=self.resources();bad['compute_processes']=[dict(pid=54321,gpu_uuid='GPU-0',used_mib=200)]
        with patch.object(self,'resources',return_value=bad),self.assertRaisesRegex(ValueError,'unknown/existing compute'):self.run_wait()
        self.assertEqual(self.calls,[]);self.assertFalse((Path(self.value['handoff_root'])/'request-v1/request.json').exists())

    def test_waiter_refuses_wrong_physical_card_count(self):
        bad=self.resources();bad['devices']=bad['devices'][:2]
        with patch.object(self,'resources',return_value=bad),self.assertRaisesRegex(ValueError,'eight idle A800'):self.run_wait()
        self.assertEqual(self.calls,[])


    def test_ddp_rank_environment_cannot_be_erased_to_masquerade_single_parent(self):
        for env in ({'WORLD_SIZE':'8','RANK':'0'},{'WORLD_SIZE':'1','RANK':'1'},{'LOCAL_RANK':'0'}):
            with self.subTest(env=env),patch.dict(os.environ,env),self.assertRaisesRegex(ValueError,'outside platform/DDP'):self.run_wait()
        self.assertFalse(Path(self.value['handoff_root']).exists());self.assertEqual(self.calls,[])


    def test_task_and_worker_may_have_different_real_terminal_states(self):
        p,r=self.request();obs=self.observed(r);value=F.ref(obs);terminal=F.ref(value['terminal_observation']);terminal['state']='Stopped';terminal['workers'][0]['state']='Failed'
        value['terminal_observation']=self.put('mixed-terminal.json',terminal);obs=self.put('mixed-observation.json',value)
        result=M.publish(self.pin,p,obs,clock=self.clock.now)
        self.assertFalse(result['GPU_started']);self.assertEqual(F.ref(value['terminal_observation'])['workers'][0]['state'],'Failed')

    def test_outer_helper_change_is_not_accepted_as_gpu_source(self):
        outer=Path(self.value['outer_controller']['root']);(outer/'contracts.py').write_text('# unvalidated helper')
        members=F.ref(self.value['outer_controller']['manifest']);members['contracts.py']=hashlib.sha256((outer/'contracts.py').read_bytes()).hexdigest()
        self.value['outer_controller']['manifest']=self.put(str((outer/'controller-files.json').relative_to(self.f.root)),members);self.save()
        with self.assertRaisesRegex(ValueError,'differs from frozen GPU'):M.wait(self.pin)

    def test_outer_undeclared_import_directory_refuses(self):
        outer=Path(self.value['outer_controller']['root']);(outer/'undeclared_package').mkdir()
        with self.assertRaisesRegex(ValueError,'undeclared import'):M.wait(self.pin)

    def test_frozen_gpu_package_is_not_modified_when_building_outer(self):
        code=Path(self.base['new_code_root']);before={str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()}
        result=M.build_controller(code,self.base['new_bundle'],self.f.root/'second-outer')
        self.assertNotEqual(result['root'],str(code));self.assertEqual({str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()},before)


    def test_unvalidated_native_semantic_bytes_cannot_enter_dedicated_cf7_production(self):
        M.verify_dedicated_gpu_source(dict(M.CF7_REQUIRED_SHA))
        altered=dict(M.CF7_REQUIRED_SHA);altered['contracts.py']='f'*64
        with self.assertRaisesRegex(ValueError,'dedicated cf7e859'):M.verify_dedicated_gpu_source(altered)


if __name__=='__main__':unittest.main()
