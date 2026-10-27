"""可移植旧计算方式接续的独立CPU反例；真实文件锁但绝不调用GPU。"""
import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import run_portable_successor as P
import resume_existing_annotation as R
from test_resume_existing_annotation import ResumeFixture,pin


class PortableSuccessorFixture:
    """元数据合成输入不冒充生产验收；沿用现有旧协议文件fixture。"""
    def __init__(self):
        self.f=ResumeFixture();self.f.legacy();self.root=self.f.root;old=self.f.spec;b=self.f.base
        self.code=self.f.task/'successor-code';shutil.copytree(self.f.code,self.code)
        oldc=json.loads(b.contract.read_bytes());newc=copy.deepcopy(oldc);newc['version']='new-v2'
        self.contract=self.root/'new-contract.json';self.contract.write_bytes(R.canonical(newc))
        self.freeze=self.root/'freeze.json';self.freeze_value={'schema':'annotation-legacy-successor-freeze/v1',
            'base_contract':old['contract'],'new_contract':pin(self.contract),'new_version':'new-v2','code_root':str(self.code),
            'bundle':pin(self.code/'bundle_files.json'),'semantic_engine_bytes_unchanged':True,
            'all_contract_fields_except_version_unchanged':True,'new_GPU_execution_verified':False}
        self.freeze.write_bytes(R.canonical(self.freeze_value))
        self.newrun=self.f.task/'portable-successor';self.evidence=self.root/'portable-evidence'
        remaining_rows=self.root/'remaining.jsonl';remaining_rows.write_bytes(b''.join(R.canonical(v)for v in b.sources[3:]))
        self.index=self.root/'remaining-index.json';self.index.write_bytes(R.canonical({'schema':'redimnet2-shard-index/v1',
            'source_count':3,'domains':{'fixture':3},'parent_full_index':old['full_index'],
            'shards':[{'id':'remaining-0','path':remaining_rows.name,'sha256':R.file_sha(remaining_rows),'count':3}]}))
        self.portable_spec=self.root/'portable-spec.json';self.portable_spec.write_bytes(R.canonical({'schema':'cpu-fixture-only'}))
        self.ledger=self.root/'portable-ledger.jsonl';self.ledger.write_bytes(b'{}\n')
        self.inventory=self.root/'inventory.sqlite';self.inventory.write_bytes(b'synthetic inventory fixed by outer union fixture')
        progress=P.inspect_progress(old['full_index']['path'],old['full_index']['sha256'],b.queue,old['contract']['sha256'])
        lock=(b.queue/'.parent.lock').stat()
        self.plan=self.root/'plan.json';self.plan_value={'schema':'annotation-speed-continuation-plan/v1',
            'source_identity_mode':'portable_encoded_sha256/v1','status':'final_prepared_not_started',
            'old_queue_mutated':False,'old_results_rewritten':False,'stage_cache_relabelled':False,
            'portable_source_spec_ref':pin(self.portable_spec),'portable_source_ledger_ref':pin(self.ledger),'portable_sidecar_inventory_ref':pin(self.inventory),
            'remaining_index':pin(self.index),'retained_source_count':3,'remaining_source_count':3,
            'old_progress_snapshot':progress,'old_parent_lock':{'path':str(b.queue/'.parent.lock'),
                'device':lock.st_dev+900,'inode':lock.st_ino,'existing_lock_opened_readonly':True,'lock_created':False},
            'terminal_observation':{**pin(self.f.terminal),'task_id':b.task,'platform_authentication_performed_by_this_tool':False}}
        self.plan.write_bytes(R.canonical(self.plan_value))
        self.union=self.root/'union.json';self.union_value={'schema':'annotation-union-coverage/v1','purpose':'cpu_fixture',
            'source_identity_mode':'portable_encoded_sha256/v1','expected_full_source_count':6,'full_index':old['full_index'],
            'continuation_plan':pin(self.plan),'old_producer':{'id':'old','contract':old['contract'],'queue_root':str(b.queue),'sidecar_tree':str(b.tree)},
            'new_producer':{'id':'new','contract':pin(self.contract),'index':pin(self.index),
                'queue_root':str(self.newrun/'queue'),'sidecar_tree':str(self.newrun/'labels/new-v2')},
            'portable_source_spec_ref':pin(self.portable_spec),'portable_source_ledger_ref':pin(self.ledger),'portable_sidecar_inventory_ref':pin(self.inventory)}
        self.union.write_bytes(R.canonical(self.union_value));self.att=self.root/'union-attestation.json';self.db=self.root/'plan.sqlite'
        self.validation={'schema':'annotation-union-validation/v1','spec_sha256':R.file_sha(self.union),'purpose':'cpu_fixture','mode':'plan',
            'exact_union_and_disjoint':True,'formal_runtime_admitted':False,'GPU_executed':False,
            'portable_source_ledger_verified':True,'old_retained_sidecars_reopened':True,
            'source_identity_mode':'portable_encoded_sha256/v1','source_count':6,'domains':{'fixture':6},
            'retained_source_count':3,'remaining_source_count':3}
        self.rebuild_attestation()
        semantic=Path(old['semantic_lock'])
        self.spec={'schema':P.SCHEMA,'purpose':'cpu_fixture','task_name':'zhounan_portable_cpu_fixture',
            'model_API_calls_allowed':0,'release_training_labels':False,'old_control':pin(self.f.path),
            'freeze_receipt':pin(self.freeze),'union_spec':pin(self.union),'union_attestation':pin(self.att),
            'new_run_root':str(self.newrun),'evidence_root':str(self.evidence),
            'lineage_claim':str(semantic.with_name(semantic.stem+'.portable-successor-v1.claim.json')),
            'old_broker':pin(b.queue/'broker.json'),'old_completion_map':pin(b.queue/'completion_map.bin'),
            'old_queue_identity':pin(b.queue/'identity.json')}
        self.platform=self.root/'current-platform.json';self.platform_value={'schema':'annotation-portable-platform-observation/v1',
            'old_task_id':old['old_task_id'],'competing_successor_tasks':[],'remote_proc_enumeration_performed':False,
            'evidence_basis':'platform_terminal_and_current_shared_locks','original_terminal':json.loads(self.f.terminal.read_bytes()),
            'observed_utc':json.loads(self.f.terminal.read_bytes())['observed_utc'],'prior_attempts':[]}
        self.platform.write_bytes(R.canonical(self.platform_value));self.spec['current_platform_observation']=pin(self.platform)
        self.path=self.root/'successor-spec.json';self.refresh();self.calls=[]
    def rebuild_attestation(self):
        if self.db.exists():self.db.unlink()
        db=sqlite3.connect(str(self.db));db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
        for key,value in [('spec',self.union_value),('spec_sha256',R.file_sha(self.union)),('validation',self.validation)]:
            db.execute('INSERT INTO metadata VALUES(?,?)',(key,json.dumps(value,ensure_ascii=False,sort_keys=True)))
        db.execute('CREATE TABLE sources(sid TEXT PRIMARY KEY,source_sha TEXT,shard INTEGER,row_number INTEGER,relative TEXT,selected TEXT,record_sha TEXT,commit_sha TEXT,decision TEXT)')
        for i,s in enumerate(self.f.base.sources):db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?)',
            (s['source_id'],'a'*64,i//2,i%2,s['sidecar_relpath'],'old'if i<3 else'new','b'*64 if i<3 else None,'c'*64 if i<3 else None,'candidate'if i<3 else None))
        db.commit();db.close();self.att.write_bytes(R.canonical({**self.validation,'consumer_index':pin(self.db)}))
    def refresh(self):self.path.write_bytes(R.canonical(self.spec));self.sha=R.file_sha(self.path)
    def close(self):self.f.close()
    def snapshot(self):return self.f.snapshot()
    def run(self,execute=False,runner=None):return P.run(self.path,self.sha,execute=execute,runner=runner)
    def assert_fds_held(self,fds):
        paths=[Path(self.f.spec['semantic_lock'])]
        if len(fds)==2:paths.append(self.f.base.queue/'.parent.lock')
        assert len(paths)==len(fds)
        for path,fd in zip(paths,fds):
            current=path.stat();held=os.fstat(fd);assert(current.st_dev,current.st_ino)==(held.st_dev,held.st_ino)
            with path.open('rb')as other:
                try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise AssertionError('old FD lock released during command')
    def fake(self,argv,log,fds):
        self.assert_fds_held(fds);self.calls.append(list(argv));log.write_text('CPU synthetic only\n')
        if '--execute'in argv:
            resources=self.newrun/'resources-attempt1';resources.mkdir()
            result={'schema':'open-annotation-independent-gpu-run/v1','worker_count':8,
                'bundle_sha256':self.freeze_value['bundle']['sha256'],'contract_sha256':self.freeze_value['new_contract']['sha256'],
                'index_sha256':self.union_value['new_producer']['index']['sha256'],'resource_returned':True,
                'model_API_calls':0,'verification':{'passed':True}}
            (resources/'parent_result.json').write_bytes(R.canonical(result))
        return 0


class PortableSuccessorTests(unittest.TestCase):
    def setUp(self):self.f=PortableSuccessorFixture()
    def tearDown(self):self.f.close()
    def test_default_readonly_accepts_fixed_plan_and_records_device_namespace_difference(self):
        before=self.f.snapshot();r=self.f.run();self.assertEqual(before,self.f.snapshot())
        self.assertTrue(r['passed']);self.assertFalse(r['GPU_started']);self.assertFalse(r['runtime_admitted'])
        self.assertNotEqual(r['current_old_parent_lock']['device'],r['historical_preparation_lock']['device'])
        self.assertFalse(self.f.evidence.exists());self.assertFalse(Path(self.f.spec['lineage_claim']).exists())
    def test_fixture_execution_holds_both_locks_and_preserves_old_producer(self):
        before=self.f.snapshot();r=self.f.run(True,self.f.fake);self.assertEqual(before,self.f.snapshot())
        self.assertTrue(r['completed']);self.assertEqual(len(self.f.calls),3)
        self.assertTrue(r['union_complete_validation_still_required']);self.assertTrue(Path(self.f.spec['lineage_claim']).exists())
        command=self.f.calls[-1]
        self.assertEqual(command[command.index('--shard-index')+1],str(self.f.index))
        self.assertNotIn('--run-root',command);self.assertNotIn('--device-lock-root',command)
        with self.assertRaisesRegex(ValueError,'permanent portable successor claim'):self.f.run(True,self.f.fake)
        self.assertEqual(len(self.f.calls),3)
    def test_CPU_fixture_cannot_invoke_real_commands(self):
        with self.assertRaisesRegex(ValueError,'CPU fixture'):self.f.run(True)
        self.assertFalse(self.f.evidence.exists())
    def test_gate_failure_starts_no_GPU_and_cannot_auto_retry_same_attempt(self):
        def fail(argv,log,fds):self.f.assert_fds_held(fds);self.f.calls.append(argv);log.write_text('failure');return 2
        with self.assertRaisesRegex(ValueError,'profile/mount gate'):self.f.run(True,fail)
        self.assertEqual(len(self.f.calls),1);self.assertFalse(Path(self.f.spec['lineage_claim']).exists())
        with self.assertRaisesRegex(ValueError,'evidence'):self.f.run(True,self.f.fake)
        self.assertEqual(len(self.f.calls),1)
    def test_GPU_failure_retains_permanent_claim_and_old_files(self):
        before=self.f.snapshot()
        def fail(argv,log,fds):
            self.f.assert_fds_held(fds);self.f.calls.append(argv);log.write_text('CPU failure');return 7 if '--execute'in argv else 0
        with self.assertRaisesRegex(ValueError,'GPU parent failed'):self.f.run(True,fail)
        self.assertEqual(before,self.f.snapshot());self.assertTrue(Path(self.f.spec['lineage_claim']).exists())
        failure=json.loads((self.f.evidence/'failure.json').read_bytes());self.assertFalse(failure['automatic_retry_allowed'])
        with self.assertRaisesRegex(ValueError,'permanent portable successor claim'):self.f.run(True,self.f.fake)
        self.assertEqual(len(self.f.calls),3)
    def test_changed_source_or_runtime_cannot_be_relabelled_as_same_profile(self):
        new=json.loads(self.f.contract.read_bytes());new['runtime']={'dtype':'bfloat16'}
        self.f.contract.write_bytes(R.canonical(new));self.f.freeze_value['new_contract']=pin(self.f.contract)
        self.f.freeze.write_bytes(R.canonical(self.f.freeze_value));self.f.spec['freeze_receipt']=pin(self.f.freeze);self.f.refresh()
        with self.assertRaisesRegex(ValueError,'only producer version'):self.f.run()
        self.assertFalse(self.f.evidence.exists())
    def test_plan_database_tamper_and_complete_label_confusion_refused(self):
        self.f.validation['mode']='complete';self.f.rebuild_attestation();self.f.spec['union_attestation']=pin(self.f.att);self.f.refresh()
        with self.assertRaisesRegex(ValueError,'plan attestation'):self.f.run()
    def test_changed_copied_source_cannot_start_GPU(self):
        (self.f.code/'worker.py').write_text('mutated copy')
        with self.assertRaisesRegex(ValueError,'bundle member'):self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls,[])
    def test_gate_mutation_of_next_script_prevents_GPU(self):
        def mutate(argv,log,fds):
            result=self.f.fake(argv,log,fds)
            if len(self.f.calls)==1:Path(self.f.f.spec['checks'][1]['script']['path']).write_text('mutated after first gate')
            return result
        with self.assertRaisesRegex(ValueError,'pinned file'):self.f.run(True,mutate)
        self.assertFalse(any('--execute'in v for v in self.f.calls))
    def test_queue_snapshot_change_after_gates_refused_before_claim(self):
        def mutate(argv,log,fds):
            result=self.f.fake(argv,log,fds)
            if len(self.f.calls)==2:(self.f.f.base.queue/'broker.json').write_text('{}')
            return result
        with self.assertRaisesRegex(ValueError,'pinned file'):self.f.run(True,mutate)
        self.assertFalse(any('--execute'in v for v in self.f.calls));self.assertFalse(Path(self.f.spec['lineage_claim']).exists())
    def test_claim_created_then_queue_mutation_is_refused_and_claim_preserved(self):
        original=P.write_new
        def mutate(path,value):
            original(path,value)
            if Path(path)==Path(self.f.spec['lineage_claim']):(self.f.f.base.queue/'completion_map.bin').write_bytes(b'changed')
        with patch.object(P,'write_new',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'pinned file'):self.f.run(True,self.f.fake)
        self.assertTrue(Path(self.f.spec['lineage_claim']).exists());self.assertFalse(any('--execute'in v for v in self.f.calls))
    def test_spec_file_mutation_after_gate_cannot_be_ignored(self):
        def mutate(argv,log,fds):
            result=self.f.fake(argv,log,fds)
            if len(self.f.calls)==1:self.f.path.write_bytes(R.canonical({**self.f.spec,'changed_after_gate':True}))
            return result
        with self.assertRaises((ValueError,RuntimeError)):self.f.run(True,mutate)
        self.assertFalse(any('--execute'in v for v in self.f.calls))
    def test_alias_retarget_refused_before_any_GPU_attempt(self):
        item=self.f.f.spec['legacy_source_alias'];alias=Path(item['alias_path']);other=self.f.root/'wrong-shared';other.mkdir()
        alias.unlink();alias.symlink_to(other,target_is_directory=True)
        with self.assertRaises((ValueError,RuntimeError)):self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls,[])
    def test_old_queue_lock_replaced_between_inputs_and_acquisition_refused(self):
        def mutate(argv,log,fds):
            result=self.f.fake(argv,log,fds)
            if len(self.f.calls)==2:
                old=self.f.f.base.queue/'.parent.lock';old.rename(old.with_name('.parent.lock.preserved'));old.touch()
            return result
        with self.assertRaises((ValueError,RuntimeError)):self.f.run(True,mutate)
        self.assertFalse(any('--execute'in v for v in self.f.calls))
    def test_held_old_semantic_or_queue_lock_refuses(self):
        for path in (Path(self.f.f.spec['semantic_lock']),self.f.f.base.queue/'.parent.lock'):
            with self.subTest(path=str(path)),path.open('rb')as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):self.f.run(True,self.f.fake)
                # Each failed attempt keeps evidence; use a distinct directory for next CPU case.
                self.f.spec['evidence_root']=str(self.f.root/('new-evidence-'+str(len(self.f.calls))));self.f.refresh()

    def test_current_platform_stale_running_competing_or_invented_process_claim_refused(self):
        original=copy.deepcopy(self.f.platform_value)
        for mode in ('stale','running','competing','invented_proc','boolean_workers'):
            with self.subTest(mode=mode):
                value=copy.deepcopy(original)
                if mode=='stale':
                    old=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(hours=2)).isoformat()
                    value['observed_utc']=value['original_terminal']['observed_utc']=old
                if mode=='running':value['original_terminal']['state']='Running'
                if mode=='competing':value['competing_successor_tasks']=['t-unobserved-successor']
                if mode=='invented_proc':value['remote_proc_enumeration_performed']=True
                if mode=='boolean_workers':value['original_terminal']['active_workers']=False
                self.f.platform.write_bytes(R.canonical(value));self.f.spec['current_platform_observation']=pin(self.f.platform);self.f.refresh()
                with self.assertRaises(ValueError):self.f.run(True,self.f.fake)
                self.assertEqual(self.f.calls,[])
    def test_current_prior_successor_running_cannot_be_hidden(self):
        value=copy.deepcopy(self.f.platform_value);value['prior_attempts']=[{'task_id':'t-prior','state':'Running','active_workers':1}]
        self.f.platform.write_bytes(R.canonical(value));self.f.spec['current_platform_observation']=pin(self.f.platform);self.f.refresh()
        with self.assertRaisesRegex(ValueError,'prior successor'):self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls,[])
    def test_current_platform_observation_mutation_after_gates_refused(self):
        def mutate(argv,log,fds):
            result=self.f.fake(argv,log,fds)
            if len(self.f.calls)==2:self.f.platform.write_bytes(b'{}')
            return result
        with self.assertRaisesRegex(ValueError,'JSON differs|pinned'):self.f.run(True,mutate)
        self.assertFalse(any('--execute'in v for v in self.f.calls))
    def test_preparation_terminal_wrong_task_cannot_be_hidden_in_plan(self):
        value=json.loads(self.f.f.terminal.read_bytes());value['task_id']='t-foreign'
        self.f.f.terminal.write_bytes(R.canonical(value))
        with self.assertRaisesRegex(ValueError,'pinned JSON differs'):self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls,[])
    def test_plan_SQLite_bytes_change_refused_before_commands(self):
        db=sqlite3.connect(str(self.f.db));db.execute("UPDATE metadata SET value='{}' WHERE key='validation'");db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'pinned file'):self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls,[])
    def test_exit_zero_without_original_resource_return_receipt_is_not_success(self):
        def omit(argv,log,fds):
            self.f.assert_fds_held(fds);self.f.calls.append(argv);log.write_text('synthetic exit zero but no resource receipt');return 0
        with self.assertRaises(FileNotFoundError):self.f.run(True,omit)
        self.assertTrue(Path(self.f.spec['lineage_claim']).exists());self.assertFalse((self.f.evidence/'completed.json').exists())
    def test_resource_return_false_is_not_completed(self):
        def wrong(argv,log,fds):
            result=self.f.fake(argv,log,fds)
            if '--execute'in argv:
                p=self.f.newrun/'resources-attempt1/parent_result.json';value=json.loads(p.read_bytes());value['resource_returned']=False;p.write_bytes(R.canonical(value))
            return result
        with self.assertRaisesRegex(ValueError,'resource-return receipt'):self.f.run(True,wrong)
        self.assertFalse((self.f.evidence/'completed.json').exists())
    def test_alias_changed_after_gates_is_refused(self):
        def mutate(argv,log,fds):
            result=self.f.fake(argv,log,fds)
            if len(self.f.calls)==2:
                alias=Path(self.f.f.spec['legacy_source_alias']['alias_path']);other=self.f.root/'new-wrong-target';other.mkdir();alias.unlink();alias.symlink_to(other,target_is_directory=True)
            return result
        with self.assertRaisesRegex(ValueError,'alias target'):self.f.run(True,mutate)
        self.assertFalse(any('--execute'in v for v in self.f.calls))
    def test_real_runner_inherits_actual_held_old_FDs_in_owned_CPU_child(self):
        paths=(Path(self.f.f.spec['semantic_lock']),self.f.f.base.queue/'.parent.lock')
        with R.existing_lock(paths[0])as a,R.existing_lock(paths[1])as b:
            code="import os,json;print(json.dumps({str(fd):[os.fstat(fd).st_dev,os.fstat(fd).st_ino] for fd in %r}))"%((a,b),)
            log=self.f.root/'owned-CPU-child.log';rc=P.real_runner([sys.executable,'-B','-c',code],log,(a,b))
            self.assertEqual(rc,0);values=json.loads(log.read_text())
            for path,fd in zip(paths,(a,b)):self.assertEqual(values[str(fd)],[path.stat().st_dev,path.stat().st_ino])
            self.f.assert_fds_held((a,b))
    def test_post_Popen_receipt_failure_terminates_and_waits_only_owned_CPU_child(self):
        real=P.subprocess.Popen;children=[]
        def record(*args,**kwargs):
            child=real(*args,**kwargs);children.append(child);return child
        with R.existing_lock(self.f.f.spec['semantic_lock'])as a,R.existing_lock(self.f.f.base.queue/'.parent.lock')as b:
            with patch.object(P.subprocess,'Popen',side_effect=record),patch.object(P,'write_new',side_effect=OSError('synthetic process receipt failure')):
                with self.assertRaisesRegex(OSError,'synthetic process receipt'):
                    P.real_runner([sys.executable,'-B','-c','import time;time.sleep(30)'],self.f.root/'owned-failure.log',(a,b))
            self.assertEqual(len(children),1);self.assertIsNotNone(children[0].poll());self.f.assert_fds_held((a,b))

    def _reject_rebound_database(self,sql,parameters=()):
        db=sqlite3.connect(str(self.f.db));db.execute(sql,parameters);db.commit();db.close()
        att=json.loads(self.f.att.read_bytes());att['consumer_index']=pin(self.f.db);self.f.att.write_bytes(R.canonical(att))
        self.f.spec['union_attestation']=pin(self.f.att);self.f.refresh()
        with self.assertRaises((ValueError,sqlite3.Error)):self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls,[])
    def test_plan_metadata_without_sources_table_is_refused_even_when_rebound(self):
        self._reject_rebound_database('DROP TABLE sources')
    def test_plan_empty_sources_is_refused_even_when_rebound(self):
        self._reject_rebound_database('DELETE FROM sources')
    def test_plan_extra_source_is_refused_even_when_rebound(self):
        self._reject_rebound_database('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?)',('src_'+'f'*64,'a'*64,3,0,'fixture/extra.annotation.json','new',None,None,None))
    def test_plan_wrong_producer_is_refused_even_when_rebound(self):
        self._reject_rebound_database("UPDATE sources SET selected='foreign' WHERE sid=(SELECT sid FROM sources LIMIT 1)")
    def test_plan_new_selection_cannot_claim_completed_results(self):
        self._reject_rebound_database("UPDATE sources SET record_sha=?,commit_sha=?,decision='candidate' WHERE selected='new'",('d'*64,'e'*64))
    def test_plan_retained_selection_requires_real_proof_shapes(self):
        self._reject_rebound_database("UPDATE sources SET record_sha='bad-proof' WHERE selected='old'")
    def test_profile_refuses_unsafe_or_missing_versions(self):
        old={'schema':'redimnet2-production-contract/v2','version':'old-safe','runtime':{}}
        for version in ('../escape','one/two','',None,True):
            with self.subTest(version=version),self.assertRaises(ValueError):P.profile(old,{**old,'version':version})

    def test_plan_BLOB_is_not_a_hexadecimal_text_proof(self):
        self._reject_rebound_database("UPDATE sources SET record_sha=? WHERE selected='old'",(sqlite3.Binary(bytes(64)),))

    def test_plan_composite_primary_key_does_not_prove_unique_source_ids(self):
        db=sqlite3.connect(str(self.f.db));rows=db.execute('SELECT * FROM sources').fetchall();db.execute('DROP TABLE sources')
        db.execute('CREATE TABLE sources(sid TEXT,source_sha TEXT,shard INTEGER,row_number INTEGER,relative TEXT,selected TEXT,record_sha TEXT,commit_sha TEXT,decision TEXT,PRIMARY KEY(sid,source_sha))')
        for row in rows:db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?)',row)
        db.commit();db.close()
        att=json.loads(self.f.att.read_bytes());att['consumer_index']=pin(self.f.db);self.f.att.write_bytes(R.canonical(att))
        self.f.spec['union_attestation']=pin(self.f.att);self.f.refresh()
        with self.assertRaisesRegex(ValueError,'unique-source table'):self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls,[])

    def test_execution_keeps_W2_but_replaces_old_fresh_mount_command_with_new_spec(self):
        oldmount=Path(self.f.f.spec['checks'][1]['script']['path']);before=oldmount.read_bytes()
        self.f.run(True,self.f.fake)
        self.assertEqual(self.f.calls[0],[self.f.f.spec['python'],'-B',self.f.f.spec['checks'][0]['script']['path'],
            *self.f.f.spec['checks'][0]['args']])
        mount=self.f.calls[1]
        self.assertEqual(Path(mount[2]).name,'verify_portable_successor_mounts.py')
        self.assertEqual(mount[mount.index('--spec')+1],str(self.f.path))
        self.assertEqual(mount[mount.index('--spec-sha')+1],self.f.sha)
        self.assertNotIn(str(oldmount),[x for command in self.f.calls for x in command])
        self.assertEqual(oldmount.read_bytes(),before)

    def _rebind_newrun(self,path):
        self.f.newrun=Path(path);self.f.spec['new_run_root']=str(path)
        self.f.union_value['new_producer']['queue_root']=str(Path(path)/'queue')
        self.f.union_value['new_producer']['sidecar_tree']=str(Path(path)/'labels/new-v2')
        self.f.union.write_bytes(R.canonical(self.f.union_value));self.f.validation['spec_sha256']=R.file_sha(self.f.union)
        self.f.rebuild_attestation();self.f.spec['union_spec']=pin(self.f.union);self.f.spec['union_attestation']=pin(self.f.att);self.f.refresh()
    def test_rebound_new_run_inside_original_frozen_code_is_refused(self):
        before=self.f.snapshot();self._rebind_newrun(self.f.f.code/'mistyped-new-run')
        with self.assertRaisesRegex(ValueError,'namespace'):self.f.run()
        self.assertEqual(before,self.f.snapshot());self.assertFalse(self.f.newrun.exists())
    def test_rebound_new_run_inside_original_control_or_claim_directory_is_refused(self):
        for directory in (self.f.f.control,Path(self.f.f.spec['semantic_lock']).parent):
            with self.subTest(directory=str(directory)):
                self._rebind_newrun(directory/'mistyped-new-run')
                with self.assertRaisesRegex(ValueError,'namespace'):self.f.run()
                self.assertFalse(self.f.newrun.exists())
    def test_new_run_cannot_be_sibling_of_code_inside_frozen_package(self):
        package=self.f.f.task/'frozen-package';package.mkdir();code=package/'code';shutil.move(self.f.code,code)
        self.f.code=code;self.f.freeze_value['code_root']=str(code);self.f.freeze_value['bundle']=pin(code/'bundle_files.json')
        self.f.freeze.write_bytes(R.canonical(self.f.freeze_value));self.f.spec['freeze_receipt']=pin(self.f.freeze)
        self._rebind_newrun(package/'wrong-run')
        with self.assertRaisesRegex(ValueError,'namespace'):self.f.run()
        self.assertFalse(self.f.newrun.exists())
    def test_evidence_cannot_write_old_inputs_controls_claims_or_controller_source(self):
        roots=(self.f.f.base.inputs,self.f.f.control,Path(self.f.f.spec['semantic_lock']).parent,HERE)
        for directory in roots:
            with self.subTest(directory=str(directory)):
                self.f.spec['evidence_root']=str(directory/('forbidden-'+self.f.root.name));self.f.refresh()
                with self.assertRaisesRegex(ValueError,'namespace'):self.f.run()
                self.assertFalse(Path(self.f.spec['evidence_root']).exists())
    def test_namespace_guard_protects_dedicated_remaining_input_directory(self):
        from verify_portable_successor_mounts import validate_output_namespaces
        union=copy.deepcopy(self.f.union_value);directory=self.f.f.task/'remaining-inputs';directory.mkdir()
        union['new_producer']['index']={'path':str(directory/'index.json'),'sha256':'a'*64}
        with self.assertRaisesRegex(ValueError,'namespace'):
            validate_output_namespaces(self.f.f.spec,self.f.freeze_value,union,directory/'new-run',self.f.evidence,controller_root=HERE)
        with self.assertRaisesRegex(ValueError,'namespace'):
            validate_output_namespaces(self.f.f.spec,self.f.freeze_value,union,self.f.newrun,directory/'evidence',controller_root=HERE)

    def test_evidence_cannot_write_into_original_shared_input_tree(self):
        self.f.spec['evidence_root']=str(Path(self.f.f.spec['legacy_source_alias']['target_path'])/'forbidden-evidence');self.f.refresh()
        with self.assertRaisesRegex(ValueError,'namespace'):self.f.run()
        self.assertFalse(Path(self.f.spec['evidence_root']).exists())
