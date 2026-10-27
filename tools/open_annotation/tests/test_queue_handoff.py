"""排队交接的真实本机文件/信号CPU反例，不操作平台或GPU。"""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import queue_handoff as H


class Clock:
    def __init__(self):self.wall=datetime.datetime(2026,10,27,13,tzinfo=datetime.timezone.utc).timestamp();self.mono=100.
    def now(self):return self.wall
    def monotonic(self):return self.mono
    def sleep(self,seconds):self.wall+=seconds;self.mono+=seconds


class HandoffFixture:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='annotation-queue-handoff-CPU-');self.root=Path(self.tmp.name).resolve()
        self.base=self.directory('task');self.inputs=self.directory('task/inputs');self.oldrun=self.directory('task/old-run')
        self.oldqueue=self.directory('task/old-run/queue');self.oldcode=self.directory('task/old-code')
        self.claims=self.directory('task/claims');self.locks=self.directory('task/device-locks')
        self.shared=self.directory('shared');self.outer=self.directory('outer');self.inner=self.directory('task/runtime-v6')
        self.frozen=self.directory('task/frozen');self.code=self.directory('task/frozen/code');self.uniondir=self.directory('task/union')
        self.oldid='t-20261025054739-4kzhp';self.prior='t-20261027155513-hbql2';self.own='t-20261027210000-zhand'
        self.identity={'hostname':self.own+'-worker-0','task_id':self.own,'pid':999,'start_ticks':123,
            'boot_id':'11111111-2222-3333-4444-555555555555'}
        self.clock=Clock()
        for name in ('queue_handoff.py','launch_queue_handoff.sh'):shutil.copyfile(HERE/name,self.outer/name)
        for name in ('launch_portable_successor.sh','run_portable_successor.py','verify_portable_successor_mounts.py'):
            (self.inner/name).write_text('CPU fixture only, never execute\n')
        self.outerref=self.controller(self.outer,'queue-handoff-files.sha256')
        self.innerref=self.controller(self.inner,'controller-files.sha256')
        python=self.root/'python-fixture';shutil.copyfile(Path(sys.executable).resolve(),python)
        empty=self.save(self.inputs/'fixed.json',{})
        old={'task_root':str(self.base),'run_root':str(self.oldrun),'queue_root':str(self.oldqueue),
            'code_root':str(self.oldcode),'old_task_id':self.oldid,'semantic_lock':str(self.claims/'old.lock'),
            'device_lock_root':str(self.locks),'legacy_source_alias':{'target_path':str(self.shared)},
            'full_index':empty,'contract':empty,'admission':empty,'original_launcher':empty,'original_claim':empty}
        oldref=self.save(self.inputs/'old-control.json',old)
        freeze=self.save(self.inputs/'freeze.json',{'code_root':str(self.code)})
        union=self.save(self.inputs/'union.json',{'continuation_plan':empty,'portable_source_spec_ref':empty,
            'portable_source_ledger_ref':empty,'portable_sidecar_inventory_ref':empty,'new_producer':{'index':empty}})
        database=self.uniondir/'plan.sqlite';database.write_bytes(b'CPU outer fixture; actual SQLite checked by frozen inner controller only')
        data=self.save(self.uniondir/'plan-report.json',{'schema':'annotation-union-validation/v1','mode':'plan',
            'purpose':'cpu_fixture','spec_sha256':union['sha256'],'exact_union_and_disjoint':True,
            'formal_runtime_admitted':False,'GPU_executed':False,'consumer_index':H.file_pin(database)})
        runtime={'schema':'annotation-portable-successor-run/v1','purpose':'cpu_fixture','task_name':'zhounan_queue_CPU_fixture',
            'model_API_calls_allowed':0,'release_training_labels':False,'old_control':oldref,'freeze_receipt':freeze,
            'union_spec':union,'union_attestation':data,'old_broker':empty,'old_completion_map':empty,'old_queue_identity':empty,
            'new_run_root':str(self.base/'new-run'),'evidence_root':str(self.base/'runtime-evidence'),
            'lineage_claim':str(self.claims/'new.claim.json')}
        self.value={'schema':H.BLUEPRINT_SCHEMA,'purpose':'cpu_fixture','task_name':runtime['task_name'],'runtime_base':runtime,
            'handoff_root':str(self.base/'handoff'),'outer_controller':self.outerref,'runtime_controller':self.innerref,
            'python_binary':H.file_pin(python),'wait_seconds':6,'required_prior_attempt_ids':[self.prior]}
        self.path=self.inputs/'blueprint.json';self.refresh();self.calls=[]
    def directory(self,name):
        p=self.root/name;p.mkdir(parents=True,mode=0o700);p.chmod(0o700);return p
    def controller(self,root,name):
        raw=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'for p in sorted(root.iterdir())if p.name!=name)
        p=root/name;p.write_text(raw);return {'root':str(root),'manifest':H.file_pin(p)}
    def save(self,path,value):
        path.write_bytes(H.canonical(value));path.chmod(0o600);return H.file_pin(path)
    def refresh(self):self.pin=self.save(self.path,self.value)
    def checked(self):
        c=H.check(self.path,self.pin['sha256']);c['blueprint_pin']=self.pin;return c
    def request(self):
        root=H.mkdir_new(self.value['handoff_root']);directory=H.mkdir_new(root/'request-v1')
        request={'schema':H.REQUEST_SCHEMA,'blueprint':self.pin,'identity':self.identity,'nonce':'a'*64,
            'created_utc':H._utc(self.clock.now()),'wait_seconds':self.value['wait_seconds'],'reply_path':str(directory/'reply.json')}
        return H.write_new(directory/'request.json',request),request
    def observation(self,request):
        value={'schema':'annotation-portable-platform-observation/v1','old_task_id':self.oldid,
            'observed_utc':H._utc(self.clock.now()),'competing_successor_tasks':[],
            'remote_proc_enumeration_performed':False,'evidence_basis':'platform_terminal_and_current_shared_locks',
            'original_terminal':{'schema':'annotation-writer-terminal-observation/v1','task_id':self.oldid,
                'queue_root':str(self.oldqueue),'state':'Failed','active_workers':0,'observed_utc':H._utc(self.clock.now()),
                'workers':[{'name':self.oldid+'-worker-0','state':'Failed'}]},
            'prior_attempts':[{'task_id':self.prior,'state':'Failed','active_workers':0}],
            'current_successor_task':{'task_id':request['identity']['task_id'],'task_name':self.value['task_name'],
                'state':'Running','active_workers':1,'workers':[{'name':request['identity']['hostname'],'state':'Running'}]}}
        return value
    def publish(self,request_pin,request,change=None):
        obs=self.observation(request)
        if change:change(obs)
        obspin=self.save(self.inputs/'current-platform.json',obs)
        return H.publish(self.path,self.pin['sha256'],request_pin,obspin,clock=self.clock.now)
    def deliver(self,seconds):
        rp=H.file_pin(Path(self.value['handoff_root'])/'request-v1/request.json',private=True)
        self.publish(rp,H.ref(rp,private=True));self.clock.sleep(.1)
    def runner(self,argv,env):self.calls.append((argv,env));return {'synthetic_exec':True}
    def wait(self,**kwargs):
        options={'execute':True,'runner':self.runner,'clock':self.clock.now,'monotonic':self.clock.monotonic,
            'sleeper':self.deliver,'identity_provider':lambda:copy.deepcopy(self.identity)};options.update(kwargs)
        return H.wait(self.path,self.pin['sha256'],**options)
    def close(self):self.tmp.cleanup()


def use_prepared_remaining(f):
    """构造已完成准备收据；不伪造联合审计成功，也不需要联合数据库。"""
    runtime=f.value['runtime_base'];layout=H.ref(runtime['union_spec'])
    plan={'remaining_source_count':3,'portable_source_ledger_ref':{'count':0}}
    planref=f.save(f.inputs/'prepared-plan.json',plan)
    layout['continuation_plan']=planref
    runtime['union_spec']=f.save(f.inputs/'prepared-layout.json',layout)
    receipt={'schema':'annotation-remaining-role-finalize/v1','new_plan':planref,
        'new_remaining_index':layout['new_producer']['index'],
        'parent_full_index':H.ref(runtime['old_control'])['full_index'],
        'inventory_ref':layout['portable_sidecar_inventory_ref'],
        'remaining_sources_checked':3,'portable_proofs_reopened':0,
        'old_plan':planref,'old_remaining_index':layout['new_producer']['index'],
        **{key:False for key in ('formal_runtime_admitted','GPU_executed',
            'source_shard_bytes_changed','v1_bytes_rewritten','source_media_read','models_executed')}}
    runtime['schema']='annotation-portable-successor-run/v2'
    runtime.pop('union_attestation')
    runtime['preparation_attestation']=f.save(f.inputs/'prepared-receipt.json',receipt)
    f.refresh();return receipt


class PreparedQueueHandoffTests(unittest.TestCase):
    def setUp(self):
        self.f=HandoffFixture();self.receipt=use_prepared_remaining(self.f)
    def tearDown(self):self.f.close()
    def update_receipt(self,change):
        change(self.receipt)
        self.f.value['runtime_base']['preparation_attestation']=self.f.save(self.f.inputs/'prepared-receipt.json',self.receipt)
        self.f.refresh()
    def test_completed_preparation_does_not_require_union_database(self):
        (self.f.uniondir/'plan.sqlite').unlink()
        result=self.f.checked()
        self.assertFalse(result['controller_inputs_check_executed'])
        self.assertFalse(Path(self.f.value['handoff_root']).exists())
    def test_prepared_handoff_preserves_v2_and_only_adds_fresh_observation(self):
        (self.f.uniondir/'plan.sqlite').unlink()
        result=self.f.wait();self.assertTrue(result['synthetic_exec'])
        argv=self.f.calls[0][0];runtime=H.ref({'path':argv[2],'sha256':argv[3]},private=True)
        runtime.pop('current_platform_observation')
        self.assertEqual(runtime,self.f.value['runtime_base'])
        self.assertNotIn('union_attestation',runtime)
    def test_prepared_receipt_required_and_mismatches_rejected(self):
        for field,value in (('schema','annotation-union-validation/v1'),('remaining_sources_checked',2),
                            ('remaining_sources_checked',True),('portable_proofs_reopened',1),
                            ('portable_proofs_reopened',False),
                            ('source_shard_bytes_changed',True),('formal_runtime_admitted',True)):
            with self.subTest(field=field,value=value):
                old=self.receipt[field];self.update_receipt(lambda v:v.update({field:value}))
                with self.assertRaises(ValueError):self.f.checked()
                self.update_receipt(lambda v:v.update({field:old}))
    def test_prepared_receipt_cannot_be_used_as_v1_union_completion(self):
        base=self.f.value['runtime_base'];base['schema']='annotation-portable-successor-run/v1'
        base['union_attestation']=base.pop('preparation_attestation');self.f.refresh()
        with self.assertRaisesRegex(ValueError,'finished union'):self.f.checked()
    def test_prepared_plan_proof_count_refuses_boolean_and_negative(self):
        for count,receipt_count in ((False,0),(-1,-1)):
            with self.subTest(count=count):
                base=self.f.value['runtime_base'];layout=H.ref(base['union_spec'])
                plan=H.ref(layout['continuation_plan']);plan['portable_source_ledger_ref']['count']=count
                planref=self.f.save(self.f.inputs/'prepared-plan.json',plan)
                layout['continuation_plan']=planref
                base['union_spec']=self.f.save(self.f.inputs/'prepared-layout.json',layout)
                self.update_receipt(lambda v:v.update(new_plan=planref,portable_proofs_reopened=receipt_count))
                with self.assertRaisesRegex(ValueError,'completed remaining preparation'):self.f.checked()
    def test_v2_rejects_extra_union_attestation_field(self):
        base=self.f.value['runtime_base'];base['union_attestation']=base['preparation_attestation'];self.f.refresh()
        with self.assertRaisesRegex(ValueError,'runtime_base'):self.f.checked()


class QueueHandoffTests(unittest.TestCase):
    def setUp(self):self.f=HandoffFixture()
    def tearDown(self):self.f.close()
    def test_default_check_is_readonly_and_grants_no_runtime(self):
        r=H.wait(self.f.path,self.f.pin['sha256'])
        self.assertTrue(r['passed']);self.assertFalse(r['request_created']);self.assertFalse(r['runtime_admitted'])
        self.assertFalse(r['controller_inputs_check_executed']);self.assertFalse(r['consumer_database_content_SHA_verified_by_outer'])
        self.assertFalse(Path(self.f.value['handoff_root']).exists());self.assertEqual(self.f.calls,[])
    def test_actual_file_handoff_only_adds_observation_then_executes_fixed_inner_shell(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-keep-original','RANK':'0','WORLD_SIZE':'1',
            'PYTHONPATH':'evil','PYTHONHOME':'evil','LD_PRELOAD':'evil','LD_AUDIT':'evil','BASH_ENV':'evil','ENV':'evil'}):
            result=self.f.wait()
        self.assertTrue(result['synthetic_exec']);self.assertEqual(len(self.f.calls),1)
        argv,env=self.f.calls[0];self.assertEqual(argv[:2],['/bin/bash',str(self.f.inner/'launch_portable_successor.sh')])
        runtime=H.ref({'path':argv[2],'sha256':argv[3]},private=True);observation=runtime.pop('current_platform_observation')
        self.assertEqual(H.canonical(runtime),H.canonical(self.f.value['runtime_base']))
        self.assertEqual(argv[4],self.f.innerref['manifest']['sha256']);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-keep-original')
        self.assertEqual((env['RANK'],env['WORLD_SIZE']),('0','1'))
        self.assertFalse(set(('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','BASH_ENV','ENV'))&set(env))
        self.assertFalse(Path(runtime['new_run_root']).exists());self.assertFalse(Path(runtime['lineage_claim']).exists())
        directory=Path(self.f.value['handoff_root'])/'request-v1'
        self.assertEqual(directory.stat().st_mode&0o777,0o700)
        for p in directory.rglob('*.json'):self.assertEqual(p.stat().st_mode&0o777,0o600);self.assertEqual(p.stat().st_nlink,1)
        self.assertEqual(H.ref(observation)['current_successor_task']['task_id'],self.f.own)
    def test_publish_does_not_enumerate_development_host_processes(self):
        rp,request=self.f.request()
        with patch.object(H,'actual_identity',side_effect=AssertionError('publisher must not inspect local cloud PID')):
            result=self.f.publish(rp,request)
        self.assertFalse(result['platform_authentication_performed']);self.assertFalse(result['GPU_admission_granted'])
        self.assertFalse(result['controller_inputs_check_executed']);self.assertEqual(self.f.calls,[])
        self.assertFalse(result['consumer_database_content_SHA_verified_by_outer'])
    def test_real_waiter_and_publisher_processes_complete_one_cpu_only_handoff(self):
        self.f.value['wait_seconds']=15;self.f.refresh();children=[]
        env=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',PYTHONDONTWRITEBYTECODE='1')
        runner_path=self.f.root/'owned-cpu-runner.json'
        code='''import json,os,pathlib,sys
sys.path.insert(0,sys.argv[1])
import queue_handoff as H
own=sys.argv[4]
linux=sys.platform.startswith("linux")
start=int(pathlib.Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]) if linux else 1
boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip() if linux else "11111111-2222-3333-4444-555555555555"
identity={"hostname":own+"-worker-0","task_id":own,"pid":os.getpid(),"start_ticks":start,"boot_id":boot}
calls=[]
def runner(argv,env):
    calls.append(argv)
    assert len(calls)==1 and env.get("CUDA_VISIBLE_DEVICES")==""
    assert not {"torch","vllm","qwen_asr","numpy"}&set(sys.modules)
    H.write_new(pathlib.Path(sys.argv[5]),{"argv":argv,"waiter_pid":os.getpid(),"calls":len(calls),
        "identity_scope":"real_pid_start_boot_synthetic_cloud_hostname" if linux else "real_pid_synthetic_start_boot_cloud_hostname",
        "GPU_executed":False,"synthetic_fixture":True})
    return {"synthetic_exec":True}
print(json.dumps(H.wait(pathlib.Path(sys.argv[2]),sys.argv[3],execute=True,runner=runner,identity_provider=lambda:dict(identity))))
'''
        try:
            waiter=subprocess.Popen([sys.executable,'-B','-c',code,str(HERE),str(self.f.path),self.f.pin['sha256'],
                self.f.own,str(runner_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
            children.append(waiter);request_path=Path(self.f.value['handoff_root'])/'request-v1/request.json'
            deadline=time.monotonic()+10
            while (not request_path.exists()or H._reply_pending(request_path))and waiter.poll()is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue(request_path.exists(),'owned CPU waiter did not publish its request')
            self.assertFalse(H._reply_pending(request_path),'owned CPU request publication did not finish before test deadline')
            rp=H.file_pin(request_path,private=True);request=H.ref(rp,private=True)
            self.assertEqual(request['identity']['pid'],waiter.pid)
            self.f.clock.wall=time.time();obspin=self.f.save(self.f.inputs/'actual-process-observation.json',self.f.observation(request))
            command=[sys.executable,'-B',str(HERE/'queue_handoff.py'),'publish','--blueprint',str(self.f.path),
                '--blueprint-sha',self.f.pin['sha256'],'--request',str(request_path),'--request-sha',rp['sha256'],
                '--observation',obspin['path'],'--observation-sha',obspin['sha256']]
            publisher=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
            children.append(publisher);published_raw,publisher_error=publisher.communicate(timeout=10)
            self.assertEqual(publisher.returncode,0,publisher_error);published=json.loads(published_raw)
            stdout,stderr=waiter.communicate(timeout=10);self.assertEqual(waiter.returncode,0,stderr)
            self.assertNotEqual(waiter.pid,publisher.pid);self.assertNotEqual(waiter.pid,os.getpid())
            receipt=H.ref(H.file_pin(runner_path,private=True),private=True);self.assertEqual(receipt['calls'],1)
            reply=H.ref(published['reply'],private=True);self.assertEqual(reply['request'],rp)
            self.assertEqual(reply['observation'],obspin);self.assertEqual(receipt['argv'][2],published['runtime_spec']['path'])
            self.assertEqual(receipt['argv'][3],published['runtime_spec']['sha256'])
            self.assertEqual(receipt['argv'][4],self.f.innerref['manifest']['sha256'])
            runtime=H.ref(published['runtime_spec'],private=True)
            self.assertEqual(H.canonical(runtime),H.canonical({**self.f.value['runtime_base'],'current_platform_observation':obspin}))
            self.assertFalse(Path(runtime['lineage_claim']).exists());self.assertIn('synthetic_exec',stdout)
            print(json.dumps({'schema':'annotation-real-two-process-cpu-handoff-test/v1','waiter_pid':waiter.pid,
                'publisher_pid':publisher.pid,'request_sha256':rp['sha256'],'runtime_sha256':published['runtime_spec']['sha256'],
                'reply_sha256':published['reply']['sha256'],'identity_scope':receipt['identity_scope'],
                'fake_runner_calls':1,'GPU_executed':False,'children_reaped':all(c.poll()is not None for c in children)}))
        finally:
            # 只回收本测试直接Popen的子进程；失败/超时也不能遗留等待者。
            for child in children:
                if child.poll()is None:
                    child.terminate()
                    try:child.communicate(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.communicate(timeout=5)
    def test_timeout_preserves_request_and_failure_without_runtime_claim(self):
        with self.assertRaisesRegex(ValueError,'timed out'):self.f.wait(sleeper=self.f.clock.sleep)
        d=Path(self.f.value['handoff_root'])/'request-v1'
        self.assertTrue((d/'request.json').exists());self.assertTrue((d/'waiter-failure.json').exists());self.assertEqual(self.f.calls,[])
        self.assertFalse(Path(self.f.value['runtime_base']['lineage_claim']).exists())
    def test_signal_before_gpu_is_recorded_and_original_handler_restored(self):
        before=signal.getsignal(signal.SIGTERM)
        def stop(seconds):signal.raise_signal(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError,'interrupted before GPU'):self.f.wait(sleeper=stop)
        self.assertEqual(signal.getsignal(signal.SIGTERM),before);self.assertEqual(self.f.calls,[])
        self.assertTrue((Path(self.f.value['handoff_root'])/'request-v1/waiter-failure.json').exists())
    def test_duplicate_wait_and_publish_do_not_overwrite_existing_evidence(self):
        rp,request=self.f.request();before=Path(rp['path']).read_bytes()
        with self.assertRaises(FileExistsError):self.f.wait()
        self.assertEqual(Path(rp['path']).read_bytes(),before)
        self.f.publish(rp,request);reply=Path(request['reply_path']).read_bytes()
        with self.assertRaisesRegex(ValueError,'reply already exists'):self.f.publish(rp,request)
        self.assertEqual(Path(request['reply_path']).read_bytes(),reply)
    def test_blueprint_requires_exact_complete_runtime_base_without_observation(self):
        original=copy.deepcopy(self.f.value)
        for mode in ('missing','extra','observation','bool_api'):
            with self.subTest(mode=mode):
                self.f.value=copy.deepcopy(original)
                if mode=='missing':self.f.value['runtime_base'].pop('lineage_claim')
                if mode=='extra':self.f.value['runtime_base']['execute']=True
                if mode=='observation':self.f.value['runtime_base']['current_platform_observation']={}
                if mode=='bool_api':self.f.value['runtime_base']['model_API_calls_allowed']=False
                self.f.refresh()
                with self.assertRaises(ValueError):self.f.checked()
        self.assertFalse(Path(original['handoff_root']).exists())
    def test_handoff_namespace_cannot_overlap_existing_or_future_inputs_outputs(self):
        original=self.f.value['handoff_root']
        choices=[self.f.inputs,self.f.oldrun,self.f.oldcode,self.f.frozen,self.f.claims,self.f.inner,
            Path(self.f.value['runtime_base']['new_run_root']),Path(self.f.value['runtime_base']['evidence_root']),self.f.uniondir]
        for directory in choices:
            with self.subTest(directory=str(directory)):
                self.f.value['handoff_root']=str(directory);self.f.refresh()
                with self.assertRaises(ValueError):self.f.checked()
        self.f.value['handoff_root']=original;self.f.refresh();self.f.checked()
    def test_symlink_and_hardlink_blueprint_or_request_are_refused(self):
        linked=self.f.inputs/'linked-blueprint.json';linked.symlink_to(self.f.path)
        with self.assertRaisesRegex(ValueError,'physical'):H.check(linked,self.f.pin['sha256'])
        linked.unlink();os.link(self.f.path,linked)
        with self.assertRaisesRegex(ValueError,'one link'):self.f.checked()
        linked.unlink();rp,request=self.f.request();os.link(rp['path'],self.f.inputs/'linked-request.json')
        with self.assertRaisesRegex(ValueError,'one link'):self.f.publish(rp,request)
    def test_existing_runtime_claim_is_not_deleted_or_retried(self):
        p=Path(self.f.value['runtime_base']['lineage_claim']);p.write_text('preserve existing claim')
        with self.assertRaisesRegex(ValueError,'claim already exists'):self.f.checked()
        self.assertEqual(p.read_text(),'preserve existing claim');self.assertEqual(self.f.calls,[])
    def test_outer_inner_manifest_and_python_changes_are_refused(self):
        for target in (self.f.outer/'queue_handoff.py',self.f.inner/'run_portable_successor.py',Path(self.f.value['python_binary']['path'])):
            with self.subTest(target=str(target)):
                before=target.read_bytes();target.write_bytes(before+b'\nchanged')
                with self.assertRaisesRegex(ValueError,'SHA differs'):self.f.checked()
                target.write_bytes(before)
    def test_manifest_filenames_and_extra_import_members_are_fixed(self):
        old=copy.deepcopy(self.f.value)
        for key in ('outer_controller','runtime_controller'):
            with self.subTest(controller=key):
                self.f.value=copy.deepcopy(old);item=self.f.value[key]
                original=Path(item['manifest']['path']);wrong=original.with_name('wrong-files.sha256');wrong.write_bytes(original.read_bytes())
                item['manifest']=H.file_pin(wrong);self.f.refresh()
                with self.assertRaisesRegex(ValueError,'manifest filename'):self.f.checked()
                wrong.unlink()
        self.f.value=copy.deepcopy(old);self.f.refresh();extra=self.f.inner/'json.py';extra.write_text('unlisted import')
        with self.assertRaisesRegex(ValueError,'extra or linked import'):self.f.checked()
    def test_production_requires_the_complete_frozen_21_member_closure(self):
        self.f.value['purpose']=self.f.value['runtime_base']['purpose']='production';self.f.refresh()
        with self.assertRaisesRegex(ValueError,'member set differs'):self.f.checked()
        for name in H.INNER_SOURCE_NAMES:
            p=self.f.inner/name
            if not p.exists():p.write_text('CPU metadata only, never imported')
        self.f.value['runtime_controller']=self.f.controller(self.f.inner,'controller-files.sha256');self.f.refresh()
        # 完整成员后才会走到生产真实源码位置门；本机合成副本仍不授生产资格。
        with self.assertRaisesRegex(ValueError,'production outer source root'):self.f.checked()
        p=Path(self.f.value['runtime_controller']['manifest']['path'])
        p.write_text(''.join(line for line in p.read_text().splitlines(keepends=True)if not line.endswith('  resume_existing_annotation.py\n')))
        self.f.value['runtime_controller']['manifest']=H.file_pin(p);self.f.refresh()
        with self.assertRaisesRegex(ValueError,'member set differs'):self.f.checked()
    def test_partial_union_database_cannot_be_presented_as_final(self):
        p=Path(self.f.value['runtime_base']['union_attestation']['path']);value=H.ref(self.f.value['runtime_base']['union_attestation'])
        partial=self.f.uniondir/'.consumer-building-unfinished.sqlite';partial.write_bytes(b'partial')
        value['consumer_index']=H.file_pin(partial);self.f.value['runtime_base']['union_attestation']=self.f.save(p,value);self.f.refresh()
        with self.assertRaisesRegex(ValueError,'final union plan database'):self.f.checked()
    def test_running_old_missing_prior_competing_or_wrong_own_task_refused(self):
        rp,request=self.f.request()
        changes=[lambda v:v['original_terminal'].update(state='Running'),lambda v:v.update(prior_attempts=[]),
            lambda v:v.update(competing_successor_tasks=['t-other']),lambda v:v['current_successor_task'].update(task_id=self.f.prior),
            lambda v:v['current_successor_task'].update(task_name='zhounan_wrong'),
            lambda v:v['current_successor_task'].update(workers=[]),lambda v:v['current_successor_task'].update(active_workers=True),
            lambda v:v.update(remote_proc_enumeration_performed=True)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):self.f.publish(rp,request,change)
        self.assertFalse((Path(rp['path']).parent/'publication-v1').exists())
    def test_stale_future_and_pre_request_observations_refused(self):
        rp,request=self.f.request()
        for offset in (-3601,1,-1):
            def change(v):
                t=H._utc(self.f.clock.now()+offset);v['observed_utc']=v['original_terminal']['observed_utc']=t
            with self.subTest(offset=offset),self.assertRaises(ValueError):self.f.publish(rp,request,change)
    def test_reply_cannot_change_other_runtime_fields_even_bool_equals_zero(self):
        rp,request=self.f.request();published=self.f.publish(rp,request);replypath=Path(request['reply_path'])
        original=H.ref(published['runtime_spec'],private=True);reply=H.ref(published['reply'],private=True)
        for key,value in (('model_API_calls_allowed',False),('new_run_root',str(self.f.base/'other-run'))):
            runtime=copy.deepcopy(original);runtime[key]=value
            reply['runtime_spec']=self.f.save(Path(published['runtime_spec']['path']),runtime);self.f.save(replypath,reply)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'only current_platform_observation'):
                H._reply(self.f.checked(),rp,request,self.f.identity,self.f.clock.now())
    def test_reply_nonce_or_actual_cloud_identity_cannot_be_rebound(self):
        rp,request=self.f.request();published=self.f.publish(rp,request);reply=H.ref(published['reply'],private=True)
        reply['request_nonce']='b'*64;self.f.save(Path(request['reply_path']),reply)
        with self.assertRaisesRegex(ValueError,'binding differs'):H._reply(self.f.checked(),rp,request,self.f.identity,self.f.clock.now())
        with self.assertRaisesRegex(ValueError,'identity changed'):
            H._reply(self.f.checked(),rp,request,{**self.f.identity,'pid':1000},self.f.clock.now())
    def test_request_replacement_during_wait_is_not_ignored(self):
        def replace(seconds):
            p=Path(self.f.value['handoff_root'])/'request-v1/request.json';value=json.loads(p.read_text());value['nonce']='b'*64
            p.write_bytes(H.canonical(value));self.f.clock.sleep(.1)
        with self.assertRaisesRegex(ValueError,'pinned JSON differs'):self.f.wait(sleeper=replace)
        self.assertEqual(self.f.calls,[])
    def test_source_mutation_after_reply_prevents_exec(self):
        def deliver(seconds):self.f.deliver(seconds);(self.f.inner/'run_portable_successor.py').write_text('changed before exec')
        with self.assertRaisesRegex(ValueError,'member SHA differs'):self.f.wait(sleeper=deliver)
        self.assertEqual(self.f.calls,[])
    def test_exec_intent_write_cannot_hide_changes_to_any_fixed_input(self):
        for kind in ('blueprint','request','reply','runtime_spec','observation','outer_source','inner_source','python','identity'):
            f=HandoffFixture();original_write=H.write_new
            def mutate(path,value):
                result=original_write(path,value)
                if Path(path).name=='exec-intent.json':
                    directory=Path(path).parent
                    if kind=='identity':f.identity['pid']+=1
                    else:
                        targets={'blueprint':f.path,'request':directory/'request.json','reply':directory/'reply.json',
                            'runtime_spec':directory/'publication-v1/runtime-spec.json','observation':f.inputs/'current-platform.json',
                            'outer_source':f.outer/'queue_handoff.py','inner_source':f.inner/'run_portable_successor.py',
                            'python':Path(f.value['python_binary']['path'])}
                        p=targets[kind];p.write_bytes(p.read_bytes()+b'\nchanged during exec intent')
                return result
            try:
                with self.subTest(kind=kind),patch.object(H,'write_new',side_effect=mutate),self.assertRaises((ValueError,json.JSONDecodeError)):
                    f.wait()
                self.assertEqual(f.calls,[])
            finally:f.close()
    def test_sleep_never_exceeds_actual_deadline_remaining(self):
        self.f.value['wait_seconds']=5;self.f.refresh();delays=[]
        def sleep(seconds):delays.append(seconds);self.f.clock.sleep(seconds)
        with self.assertRaisesRegex(ValueError,'timed out'):self.f.wait(sleeper=sleep)
        self.assertEqual(delays,[2.,2.,1.]);self.assertEqual(self.f.calls,[])
    def test_exec_intent_storage_delay_cannot_start_gpu_after_wait_deadline(self):
        original=H.write_new
        def delay(path,value):
            result=original(path,value)
            if Path(path).name=='exec-intent.json':self.f.clock.sleep(self.f.value['wait_seconds'])
            return result
        with patch.object(H,'write_new',side_effect=delay),self.assertRaisesRegex(ValueError,'timed out'):
            self.f.wait()
        self.assertEqual(self.f.calls,[])
    def test_malformed_reply_is_a_stable_error_not_an_infinite_wait(self):
        def corrupt(seconds):
            p=Path(self.f.value['handoff_root'])/'request-v1/reply.json';p.write_bytes(b'{broken');p.chmod(0o600)
        with self.assertRaises(ValueError):self.f.wait(sleeper=corrupt)
        self.assertEqual(self.f.calls,[])
    def test_inflight_hardlink_reply_is_never_consumed(self):
        rp,request=self.f.request();p=Path(request['reply_path']);pending=p.with_name('.reply.json.pending')
        pending.write_text('{}');pending.chmod(0o600);os.link(pending,p)
        self.assertTrue(H._reply_pending(p))
        with self.assertRaisesRegex(ValueError,'one link'):H.file_pin(p,private=True)
        pending.unlink();self.assertFalse(H._reply_pending(p))
    def test_production_refuses_synthetic_runner_clock_or_identity_hooks(self):
        checked=self.f.checked();checked['blueprint']['purpose']='production'
        for kwargs in ({'runner':lambda *a:None},{'clock':self.f.clock.now},{'identity_provider':lambda:self.f.identity}):
            with patch.object(H,'check',return_value=checked),self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'production waiter'):
                H.wait(self.f.path,self.f.pin['sha256'],execute=True,**kwargs)
        self.assertFalse(Path(self.f.value['handoff_root']).exists())
    def test_fixture_cannot_invoke_real_exec(self):
        with self.assertRaisesRegex(ValueError,'CPU fixture cannot execute'):
            H.wait(self.f.path,self.f.pin['sha256'],execute=True)
        self.assertFalse(Path(self.f.value['handoff_root']).exists())
    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}',b'{"a":1e999}',b'{"a":-1e999}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):H.json_value(raw)
    def test_fifo_replacement_is_rejected_without_blocking_a_reader_process(self):
        fifo=self.f.root/'not-a-regular-file';os.mkfifo(fifo,0o600)
        code='''import pathlib,sys
sys.path.insert(0,sys.argv[1]);import queue_handoff as H
try:H.read_file(pathlib.Path(sys.argv[2]),private=True)
except ValueError as exc:
    assert "regular file" in str(exc)
else:raise AssertionError("FIFO accepted")
'''
        subprocess.run([sys.executable,'-B','-c',code,str(HERE),str(fifo)],capture_output=True,text=True,check=True,timeout=5)
    def test_fifo_swap_after_lstat_is_nonblocking_and_refused(self):
        p=self.f.root/'swap-to-fifo';p.write_text('regular before inspection');p.chmod(0o600)
        code='''import os,pathlib,sys
from unittest.mock import patch
sys.path.insert(0,sys.argv[1]);import queue_handoff as H
p=pathlib.Path(sys.argv[2]);real_open=H.os.open
def replaced(path,flags,*args,**kwargs):
    if pathlib.Path(path)==p:
        p.unlink();os.mkfifo(p,0o600)
    return real_open(path,flags,*args,**kwargs)
with patch.object(H.os,"open",side_effect=replaced):
    try:H.read_file(p,private=True)
    except ValueError as exc:assert "regular file" in str(exc)
    else:raise AssertionError("swapped FIFO accepted")
'''
        subprocess.run([sys.executable,'-B','-c',code,str(HERE),str(p)],capture_output=True,text=True,check=True,timeout=5)
    def test_import_and_help_do_not_load_heavy_inference_modules(self):
        code='import sys;sys.path.insert(0,%r);import queue_handoff;assert not {"torch","vllm","qwen_asr","numpy"}&set(sys.modules)'%str(HERE)
        subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,check=True)
        result=subprocess.run([sys.executable,'-B',str(HERE/'queue_handoff.py'),'--help'],capture_output=True,text=True,check=True)
        self.assertIn('--blueprint-sha',result.stdout)
        subprocess.run(['/bin/bash','-n',str(HERE/'launch_queue_handoff.sh')],capture_output=True,text=True,check=True)


if __name__=='__main__':unittest.main()
