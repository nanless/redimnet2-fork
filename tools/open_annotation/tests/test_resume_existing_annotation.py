"""同合同恢复的合成文件反例；不连接服务器，不执行GPU或真实检查命令。"""
import copy
import datetime
import fcntl
import json
from pathlib import Path
import shutil
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import resume_existing_annotation as R
import test_prepare_speed_continuation as F


def pin(path):return {'path':str(path),'sha256':R.file_sha(path)}


class ResumeFixture:
    def __init__(self):
        self.base=F.SpeedContinuationTests(methodName='runTest');self.base.setUp();b=self.base
        self.root=b.root;self.task=self.root/'task';self.task.mkdir();self.run=self.task/'formal';self.run.mkdir()
        self.code=self.task/'production-code';self.code.mkdir();self.control=self.task/'control';self.control.mkdir()
        shutil.move(b.queue,self.run/'queue');b.queue=self.run/'queue'
        (b.queue/'broker.lock').touch();(b.queue/'claims').mkdir();(b.queue/'claims/s1.lock').touch()
        shutil.move(b.tree.parent,self.run/'labels');b.tree=self.run/'labels/old-v1'
        self.locks=self.root/'shared-locks';self.locks.mkdir();self.python=self.root/'explicit-python';self.python.write_bytes(b'synthetic interpreter not executed')
        for name in ('gpu_launch.py','worker.py'):(self.code/name).write_text('# synthetic frozen source\n')
        bundle=self.code/'bundle_files.json';bundle.write_bytes(R.canonical({p.name:R.file_sha(p)for p in self.code.iterdir()}))
        scripts=[]
        for name in ('cloud_gate.py','verify_cloud_mounts.py'):
            p=self.control/name;p.write_text('# synthetic check not executed\n');scripts.append(pin(p))
        launcher=self.control/'launch-ultra-w8.sh';launcher.write_text('# synthetic original fresh-only launcher\n')
        admission={'schema':'ultra-eight-three-train-spec/v3','code_root':str(self.code),'silver_contract':pin(b.contract),
            'full_index':pin(b.index),'bundle':pin(bundle),'cloud':{'task_root':str(self.task),'run_root':str(self.run),
            'python':str(self.python),'device_lock_root':str(self.locks),'platform_workers':1,'gpus_per_worker':8,
            'priority':6,'max_runtime_seconds':2592000}}
        self.admission=self.control/'admission-spec.json';self.admission.write_bytes(R.canonical(admission))
        key=b.csha+'-'+b.isha;semantic=self.task/'.formal-parent-locks';semantic.mkdir()
        (semantic/(key+'.lock')).touch();claim=semantic/(key+'.claim.json')
        claim.write_bytes(R.canonical({'schema':'ultra-three-train-semantic-job-claim/v1','run_root':str(self.run),
            'spec_sha256':R.file_sha(self.admission),'contract_sha256':b.csha,'index_sha256':b.isha,'status':'old claim retained'}))
        self.terminal=self.root/'terminal-proof.json';self.absence=self.root/'absence-proof.json'
        now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.terminal.write_bytes(R.canonical({'schema':'annotation-writer-terminal-observation/v1','task_id':b.task,
            'queue_root':str(b.queue),'state':'Failed','observed_utc':now,'active_workers':0,
            'workers':[{'name':b.task+'-worker-0','state':'Failed'}]}))
        self.absence.write_bytes(R.canonical({'schema':'annotation-writer-absence-observation/v1','task_id':b.task,
            'run_root':str(self.run),'observed_utc':now,'active_parent_count':0,'active_worker_count':0,'other_successor_tasks':[]}))
        ap=pin(self.admission)
        self.spec={'schema':R.SCHEMA,'purpose':'cpu_fixture','protocol':'explicit_roots_v3','attempt_number':2,'model_API_calls_allowed':0,
            'release_training_labels':False,'old_task_id':b.task,'task_root':str(self.task),'run_root':str(self.run),
            'code_root':str(self.code),'queue_root':str(b.queue),'output_root':str(self.run/'labels'),
            'resource_output':str(self.run/'resources-attempt2'),'evidence_root':str(self.root/'new-resume-evidence'),
            'device_lock_root':str(self.locks),'python':str(self.python),'python_binary':pin(self.python),'contract':pin(b.contract),'full_index':pin(b.index),
            'admission':ap,'bundle':pin(bundle),'original_launcher':pin(launcher),'original_claim':pin(claim),
            'semantic_lock':str(semantic/(key+'.lock')),'resume_claim':str(semantic/(key+'.resume-attempt2.claim.json')),
            'terminal_observation':pin(self.terminal),'writer_absence':pin(self.absence),
            'checks':[{'script':scripts[0],'args':['--live','--spec',ap['path'],'--spec-sha',ap['sha256']]},
                      {'script':scripts[1],'args':['--spec',ap['path'],'--spec-sha',ap['sha256']]}]}
        self.path=self.root/'resume-spec.json';self.refresh();self.calls=[]

    def refresh(self):self.path.write_bytes(R.canonical(self.spec));self.sha=R.file_sha(self.path)
    def legacy(self):
        self.spec['protocol']='legacy_frozen_v2'
        target=self.root/'shared-mount';target.mkdir();alias=self.root/'shared-alias';alias.symlink_to(target,target_is_directory=True)
        self.spec['legacy_source_alias']={'alias_path':str(alias),'target_path':str(target)}
        source='from pathlib import Path\nROOT = Path(%r)\nGLOBAL_LOCKS = Path(%r)\ndef lock():\n    return G.DeviceLocks(GLOBAL_LOCKS, [])\n'%(str(self.task),str(self.locks))
        (self.code/'gpu_launch.py').write_text(source)
        bundle=self.code/'bundle_files.json';bundle.write_bytes(R.canonical({p.name:R.file_sha(p)for p in self.code.iterdir()if p!=bundle}))
        self.spec['bundle']=pin(bundle)
        a=json.loads(self.admission.read_bytes());a['schema']='ultra-eight-three-train-spec/v2';a['cloud'].pop('device_lock_root');a['bundle']=pin(bundle)
        self.admission.write_bytes(R.canonical(a));self.spec['admission']=pin(self.admission)
        cp=Path(self.spec['original_claim']['path']);claim=json.loads(cp.read_bytes());claim['spec_sha256']=self.spec['admission']['sha256'];cp.write_bytes(R.canonical(claim));self.spec['original_claim']=pin(cp)
        for item,name,prefix in zip(self.spec['checks'],('ultra_w8_gate_v2.py','verify_ultra_cloud_mounts_v2.py'),(['--live'],[])):
            p=self.control/name;p.write_text('# synthetic original v2 gate\n');item['script']=pin(p)
            item['args']=prefix+['--spec',str(self.admission),'--spec-sha',self.spec['admission']['sha256']]
        v=json.loads(self.absence.read_bytes());v.pop('active_parent_count');v.pop('active_worker_count')
        v.update(evidence_basis='platform_terminal_and_stable_locks',remote_proc_enumeration_performed=False,platform_instance_state='Failed')
        self.absence.write_bytes(R.canonical(v));self.spec['writer_absence']=pin(self.absence);self.refresh()
    def close(self):self.base.tearDown()
    def snapshot(self):
        homes=[self.base.inputs,self.base.queue,self.base.tree,self.code,self.control,Path(self.spec['original_claim']['path'])]
        paths=[]
        for home in homes:paths.extend(home.rglob('*')if home.is_dir()else[home])
        return {str(p):(p.read_bytes(),p.stat().st_mtime_ns)for p in paths if p.is_file()}
    def fake(self,argv,log,semantic_fd):
        self.calls.append(argv);log.write_text('synthetic command only\n')
        # 语义锁要由同一FD贯穿；独立打开无法再取锁。
        with self.assert_lock_held(semantic_fd):pass
        if '--execute'in argv:
            out=Path(self.spec['resource_output']);out.mkdir()
            (out/'parent_result.json').write_bytes(R.canonical({'schema':'open-annotation-independent-gpu-run/v1',
                'contract_sha256':self.spec['contract']['sha256'],'index_sha256':self.spec['full_index']['sha256'],
                'bundle_sha256':self.spec['bundle']['sha256'],'worker_count':8,'resource_returned':True,'verification':{'passed':True}}))
        return 0
    def assert_lock_held(self,fd):
        import contextlib,os
        @contextlib.contextmanager
        def check():
            own=os.fstat(fd)
            with Path(self.spec['semantic_lock']).open('rb')as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise AssertionError('semantic lock was released during command')
            yield own
        return check()


class ResumeExistingTests(unittest.TestCase):
    def setUp(self):self.f=ResumeFixture()
    def tearDown(self):self.f.close()
    def run_it(self,execute=False,runner=None):return R.run(self.f.path,self.f.sha,execute=execute,runner=runner)

    def test_default_is_readonly_and_keeps_original_full_denominator(self):
        before=self.f.snapshot();r=self.run_it();self.assertEqual(before,self.f.snapshot())
        self.assertEqual(r['progress']['source_count'],6);self.assertEqual(len(r['partial_sources']),1)
        self.assertFalse(Path(self.f.spec['evidence_root']).exists());self.assertFalse(Path(self.f.spec['resume_claim']).exists())
        self.assertFalse(r['GPU_started']);self.assertFalse(r['platform_authenticated_by_this_tool'])

    def test_real_legacy_shape_uses_frozen_constants_and_exact_old_CLI(self):
        self.f.legacy();before=self.f.snapshot();r=self.run_it()
        self.assertEqual(r['protocol'],'legacy_frozen_v2');self.assertNotIn('--run-root',r['gpu_command'])
        self.assertNotIn('--device-lock-root',r['gpu_command']);self.assertIn('ultra_w8_gate_v2.py',r['checks'][0][2])
        self.assertEqual(before,self.f.snapshot());done=self.run_it(True,self.f.fake);self.assertEqual(done['launch_count'],1)

    def test_legacy_constant_mismatch_cannot_be_hidden_by_spec_or_bundle_repin(self):
        self.f.legacy();p=self.f.code/'gpu_launch.py';p.write_text(p.read_text().replace(str(self.f.locks),str(self.f.root/'wrong-locks')))
        # 固定包摘要由描述绑定，不能以错误常量作为另一合法原协议。
        with self.assertRaisesRegex(ValueError,'bundle member'):self.run_it()
        with self.assertRaisesRegex(ValueError,'ROOT/GLOBAL_LOCKS'):
            R.frozen_roots(p,self.f.task,self.f.locks)

    def test_explicit_interpreter_symlink_keeps_original_argv_and_binary_pin(self):
        binary=self.f.python.with_name('explicit-python3.10');self.f.python.rename(binary);self.f.python.symlink_to(binary.name)
        self.f.spec['python_binary']=pin(binary);self.f.refresh();r=self.run_it()
        self.assertEqual(r['gpu_command'][0],str(self.f.python));self.assertEqual(r['python_resolved_path'],str(binary))
        binary.write_bytes(b'changed interpreter')
        with self.assertRaisesRegex(ValueError,'pinned file'):self.run_it()

    def test_legacy_alias_keeps_source_fields_and_full_stat_and_refuses_wrong_target(self):
        self.f.legacy();item=self.f.spec['legacy_source_alias'];alias=Path(item['alias_path']);target=Path(item['target_path'])
        media=target/'sample.wav';media.write_bytes(b'synthetic raw bytes')
        self.assertEqual(R.legacy_media_path(self.f.spec,str(alias/'sample.wav')),media)
        other=self.f.root/'another-shared';other.mkdir();alias.unlink();alias.symlink_to(other,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'alias target'):self.run_it()

    def test_partial_stat_device_difference_is_not_dropped_or_relabelled(self):
        source=self.f.base.sources[2];p=self.f.base.tree/source['sidecar_relpath'];record=json.loads(p.read_bytes())
        record['source_identity']['stat_before']['device']+=1;record['source_identity']['stat_after']['device']+=1
        p.write_bytes(R.canonical(record));cp=p.with_name(p.name+'.commit.json');commit=json.loads(cp.read_bytes())
        commit.update(sidecar_sha256=R.file_sha(p),bytes=p.stat().st_size);cp.write_bytes(R.canonical(commit))
        before=self.f.snapshot()
        with self.assertRaisesRegex(ValueError,'device/inode')as error:self.run_it()
        self.assertIn('"expected"',str(error.exception));self.assertIn('"actual"',str(error.exception))
        self.assertEqual(before,self.f.snapshot())
        self.assertFalse(Path(self.f.spec['resume_claim']).exists());self.assertFalse(Path(self.f.spec['evidence_root']).exists())

    def test_synthetic_execution_holds_semantic_lock_and_preserves_old_bytes(self):
        before=self.f.snapshot();r=self.run_it(True,self.f.fake)
        self.assertEqual(before,self.f.snapshot());self.assertEqual(len(self.f.calls),3)
        self.assertTrue(r['synthetic_fixture']);self.assertEqual(r['launch_count'],1)
        command=self.f.calls[-1]
        for name,key in (('--queue-root','queue_root'),('--output-root','output_root'),('--output','resource_output')):
            self.assertEqual(command[command.index(name)+1],self.f.spec[key])
        with self.assertRaisesRegex(ValueError,'fresh attempt2|evidence|claim'):self.run_it(True,self.f.fake)
        self.assertEqual(len(self.f.calls),3)

    def test_cpu_cli_path_cannot_run_real_commands(self):
        with self.assertRaisesRegex(ValueError,'cannot execute real'):self.run_it(True)
        self.assertFalse(Path(self.f.spec['evidence_root']).exists())

    def test_missing_old_metadata_never_reinitializes_queue(self):
        for name in ('identity.json','broker.json','completion_map.bin','.parent.lock','broker.lock'):
            with self.subTest(name=name):
                p=self.f.base.queue/name;raw=p.read_bytes();p.unlink()
                with self.assertRaisesRegex(ValueError,'metadata missing'):self.run_it()
                self.assertFalse(p.exists());p.write_bytes(raw)

    def test_active_parent_or_semantic_lock_refuses_without_writes(self):
        for p in (Path(self.f.spec['semantic_lock']),self.f.base.queue/'.parent.lock',self.f.base.queue/'claims/s1.lock'):
            with self.subTest(path=str(p)),p.open('rb')as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):self.run_it()
        self.assertFalse(Path(self.f.spec['evidence_root']).exists())

    def test_stale_nonterminal_or_active_worker_refused(self):
        original=json.loads(self.f.terminal.read_bytes())
        changes=[{'state':'Running'},{'active_workers':1},{'active_workers':False},
            {'observed_utc':(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(hours=2)).isoformat()}]
        for change in changes:
            with self.subTest(change=change):
                v={**original,**change};self.f.terminal.write_bytes(R.canonical(v))
                self.f.spec['terminal_observation']=pin(self.f.terminal);self.f.refresh()
                with self.assertRaises(ValueError):self.run_it()

    def test_competing_successor_or_unknown_parent_refused(self):
        v=json.loads(self.f.absence.read_bytes());v['other_successor_tasks']=['another-task']
        self.f.absence.write_bytes(R.canonical(v));self.f.spec['writer_absence']=pin(self.f.absence);self.f.refresh()
        with self.assertRaisesRegex(ValueError,'absence'):self.run_it()

    def test_wrong_claim_contract_or_frozen_source_refused(self):
        (self.f.code/'worker.py').write_text('# changed frozen source\n')
        with self.assertRaisesRegex(ValueError,'bundle member'):self.run_it()

    def test_orphan_partial_sidecar_preserved_and_refused(self):
        source=self.f.base.sources[2];p=self.f.base.tree/source['sidecar_relpath'];raw=p.read_bytes()
        p.with_name(p.name+'.commit.json').unlink()
        with self.assertRaisesRegex(ValueError,'orphan'):self.run_it()
        self.assertEqual(p.read_bytes(),raw);self.assertFalse(p.with_name(p.name+'.commit.json').exists())

    def test_gate_failure_keeps_claim_and_never_starts_gpu(self):
        def fail(argv,log,fd):self.f.calls.append(argv);log.write_text('synthetic gate failure');return 2
        with self.assertRaisesRegex(ValueError,'gate/mount'):self.run_it(True,fail)
        self.assertEqual(len(self.f.calls),1);self.assertTrue(Path(self.f.spec['resume_claim']).is_file())
        failure=json.loads((Path(self.f.spec['evidence_root'])/'failure.json').read_bytes())
        self.assertEqual(failure['launch_count'],0);self.assertTrue(failure['permanent_resume_claim_written'])

    def test_broker_change_during_gate_refuses_before_gpu_and_keeps_failure(self):
        def mutate(argv,log,fd):
            self.f.calls.append(argv);log.write_text('synthetic broker mutation')
            if len(self.f.calls)==2:
                p=self.f.base.queue/'broker.json';v=json.loads(p.read_bytes());v['leases']['s1']['nonce']='different';p.write_bytes(R.canonical(v))
            return 0
        with self.assertRaisesRegex(ValueError,'broker changed'):self.run_it(True,mutate)
        self.assertEqual(len(self.f.calls),2)
        f=json.loads((Path(self.f.spec['evidence_root'])/'failure.json').read_bytes());self.assertEqual(f['launch_count'],0)

    def test_gate_cannot_change_bundle_member_and_still_start_gpu(self):
        def mutate(argv,log,fd):
            self.f.calls.append(argv);log.write_text('synthetic source mutation during second gate')
            if len(self.f.calls)==2:(self.f.code/'worker.py').write_text('# changed after initial inspection\n')
            return 0
        with self.assertRaisesRegex(ValueError,'bundle member changed: worker.py'):self.run_it(True,mutate)
        self.assertEqual(len(self.f.calls),2)
        f=json.loads((Path(self.f.spec['evidence_root'])/'failure.json').read_bytes())
        self.assertEqual(f['launch_count'],0);self.assertEqual(f['launch_attempt_count'],0)

    def test_gate_cannot_replace_original_launcher_or_prior_check_script(self):
        for changed in ('original_launcher','first_check'):
            with self.subTest(changed=changed):
                fixture=ResumeFixture()
                try:
                    def mutate(argv,log,fd):
                        fixture.calls.append(argv);log.write_text('synthetic control mutation')
                        if len(fixture.calls)==2:
                            pin=fixture.spec['original_launcher']if changed=='original_launcher'else fixture.spec['checks'][0]['script']
                            Path(pin['path']).write_text('# tampered control after initial check\n')
                        return 0
                    with self.assertRaisesRegex(ValueError,'pinned file differs'):
                        R.run(fixture.path,fixture.sha,execute=True,runner=mutate)
                    self.assertEqual(len(fixture.calls),2)
                finally:fixture.close()

    def test_gpu_failure_retains_single_attempt_and_no_automatic_retry(self):
        def fail(argv,log,fd):
            self.f.calls.append(argv);log.write_text('synthetic parent failure');return 2 if '--execute'in argv else 0
        with self.assertRaisesRegex(ValueError,'GPU parent failed'):self.run_it(True,fail)
        failure=json.loads((Path(self.f.spec['evidence_root'])/'failure.json').read_bytes())
        self.assertEqual(failure['launch_count'],1);self.assertFalse(failure['automatic_retry_allowed'])
        self.assertEqual(len(self.f.calls),3)

    def test_gate_cannot_retarget_admission_or_use_unpinned_script(self):
        self.f.spec['checks'][0]['args'].remove('--live');self.f.refresh()
        with self.assertRaisesRegex(ValueError,'arguments'):self.run_it()

    def test_existing_output_or_evidence_and_path_alias_refused(self):
        out=Path(self.f.spec['resource_output']);out.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh attempt2'):self.run_it()
        out.rmdir();e=Path(self.f.spec['evidence_root']);e.symlink_to(self.f.run,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'physical'):self.run_it()


if __name__=='__main__':unittest.main()
