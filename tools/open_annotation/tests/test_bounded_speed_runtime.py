"""新schema的控制链、继承FD、已发结果跳过和只读报告的CPU反例。"""
from __future__ import annotations
import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import bounded_speed_runtime as M
import bounded_speed_handoff as H
import bounded_speed_queue as Q
import test_bounded_speed_handoff as Fixtures
from test_maintenance_entries import external_contract


class RuntimeFixture(unittest.TestCase):
    def put(self,name,value):
        p=self.f.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(H.P.canonical(value))
        return dict(path=str(p),sha256=H.P.byte_sha(p.read_bytes()))

    def setUp(self):
        self.fixture=Fixtures.BoundedHandoffTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.f=self.fixture.f
        current=self.f.root/'currentrun';current.mkdir();self.f.queue.rename(current/'queue');self.f.queue=current/'queue';self.f.tree.parent.rename(current/'labels');self.f.tree=current/'labels'/'old-v1';self.f.write_terminal()
        run=self.f.root/'speedrun';run.mkdir();self.code=self.f.root/'newcode';self.code.mkdir()
        self.fixture.value.update(old_queue_root=str(self.f.queue),old_sidecar_tree=str(self.f.tree),new_queue_root=str(run/'queue'),new_sidecar_root=str(run/'labels'));self.fixture.save()
        handoff=self.fixture.run_plan();self.index=handoff['new_index'];self.thin=json.loads(Path(self.index['path']).read_text())
        c=external_contract();c['purpose']='cpu_fixture';c['version']='speed-v1';c['runtime']['asr_compute']={'dtype':'float32','attention':'sdpa'}
        self.c=c;self.cp=self.put('new-contract.json',c);candidate=copy.deepcopy(c);candidate['version']='pilot-v1';self.candidate=self.put('pilot-contract.json',candidate)
        baseline=copy.deepcopy(candidate);baseline['runtime']['asr_compute']={'dtype':'float32','attention':'eager'};self.baseline_contract=self.put('baseline-contract.json',baseline)
        self.selected,self.evidence=self.make_pilot()
        oldrun=self.f.root/'originalrun';oldrun.mkdir();(oldrun/'queue').mkdir();(oldrun/'queue'/'.parent.lock').touch()
        oldcode=self.f.root/'oldcode';oldcode.mkdir();(oldcode/'gpu_launch.py').write_text('synthetic; never execute')
        orig=copy.deepcopy(json.loads(self.f.contract.read_text()));orig['version']='original-v1';self.orig=self.put('original-contract.json',orig)
        bundle=self.put('oldcode/bundle_files.json',{'gpu_launch.py':H.P.file_sha(oldcode/'gpu_launch.py')})
        key=self.orig['sha256']+'-'+self.f.isha;semantic=self.f.root/'.formal-parent-locks'/(key+'.lock');semantic.parent.mkdir();semantic.touch()
        admission=self.put('admission.json',dict(schema='ultra-eight-three-train-spec/v2',silver_contract=self.orig,full_index=dict(path=str(self.f.index),sha256=self.f.isha),bundle=bundle,code_root=str(oldcode),cloud=dict(task_root=str(self.f.root),run_root=str(oldrun),python='/python',priority=6,platform_workers=1,gpus_per_worker=8)))
        claim=self.put(str(semantic.with_name(key+'.claim.json').relative_to(self.f.root)),dict(schema='ultra-three-train-semantic-job-claim/v1',run_root=str(oldrun),spec_sha256=admission['sha256'],contract_sha256=self.orig['sha256'],index_sha256=self.f.isha))
        checks=[]
        for name,prefix in [('ultra_w8_gate_v2.py',['--live']),('verify_ultra_cloud_mounts_v2.py',[])]:
            p=self.f.root/name;p.write_text('# CPU fixture only')
            checks.append(dict(script=dict(path=str(p),sha256=H.P.file_sha(p)),args=prefix+['--spec',admission['path'],'--spec-sha',admission['sha256']]))
        old=dict(protocol='legacy_frozen_v2',task_root=str(self.f.root),run_root=str(oldrun),queue_root=str(oldrun/'queue'),code_root=str(oldcode),contract=self.orig,full_index=dict(path=str(self.f.index),sha256=self.f.isha),admission=admission,bundle=bundle,original_claim=claim,semantic_lock=str(semantic),python='/python',checks=checks,device_lock_root=str(self.f.root/'device-locks'))
        oldpin=self.put('old-control.json',old)
        freeze=self.put('previous-freeze.json',dict(schema='annotation-legacy-successor-freeze/v1',base_contract=self.orig,new_contract=self.thin['old_contract'],semantic_engine_bytes_unchanged=True,all_contract_fields_except_version_unchanged=True))
        union=self.put('previous-union.json',dict(new_producer=dict(contract=self.thin['old_contract'],index=self.thin['parent_index'],queue_root=str(self.f.queue),sidecar_tree=str(self.f.tree))))
        lineage=semantic.with_name(key+'.portable-successor-v1.claim.json')
        previous=self.put('previous-runtime.json',dict(schema='annotation-portable-successor-run/v2',purpose='cpu_fixture',old_control=oldpin,freeze_receipt=freeze,union_spec=union,new_run_root=str(current),lineage_claim=str(lineage)))
        self.put(str(lineage.relative_to(self.f.root)),dict(schema='annotation-portable-successor-claim/v1',spec_sha256=previous['sha256'],new_contract=self.thin['old_contract'],new_index=self.thin['parent_index'],new_run_root=str(current)))
        observed=json.loads(self.f.terminal.read_text())['observed_utc'];terminal=dict(path=str(self.f.terminal),sha256=self.f.tsha)
        platform=self.put('platform.json',dict(schema=M.PLATFORM_SCHEMA,old_task_id=self.f.task,old_queue_root=str(self.f.queue),competing_successor_tasks=[],remote_proc_enumeration_performed=False,evidence_basis='platform_terminal_and_current_shared_locks',terminal_observation=terminal,observed_utc=observed))
        self.runtime=dict(schema=M.SCHEMA,purpose='cpu_fixture',task_name='zhounan_fixture',index=self.index,previous_runtime_spec=previous,new_contract=self.cp,selected_pilot_admission=self.selected,new_run_root=str(run),new_code_root=str(self.code),new_bundle=dict(path=str(self.code/'bundle_files.json'),sha256='b'*64),device_lock_root=old['device_lock_root'],current_platform_observation=platform,model_API_calls_allowed=0,release_training_labels=False)
        (self.code/'bundle_files.json').write_text('{}')
        self.runtime['new_bundle']['sha256']=H.P.file_sha(self.code/'bundle_files.json');self.save()

    def save(self):self.pin=self.put('runtime.json',self.runtime)

    def make_pilot(self):
        index={'path':'/finite-index.json','sha256':'e'*64}
        tracks={}
        for name,c in [('baseline',self.baseline_contract),('main',self.candidate),('recovery',self.candidate)]:
            tracks[name]=self.put(name+'-track.json',dict(schema='redimnet2-annotation-speed-track/v1',contract=c,index=index,resources_root=str(self.f.root/name/'resources')))
        resources={};steps=[]
        for name in tracks:
            track=json.loads(Path(tracks[name]['path']).read_text());root=Path(track['resources_root'])
            parent=dict(schema='open-annotation-independent-gpu-run/v1',contract_sha256=track['contract']['sha256'],index_sha256=index['sha256'],resource_returned=True,full_sidecar_readback=True,worker_count=2,bundle_sha256='b'*64,verification=dict(passed=True,database_created=False,source_disjointness_storage='bounded_pilot_memory_set'))
            refs={}
            for field,filename,value in [('parent_result','parent_result.json',parent),('resource_return','resource_return_double_probe.json',dict(interval_seconds=5,snapshots=[dict(compute_processes=[],devices=[dict(used_mib=0)])]*2)),('terminal_drain','terminal_drain_complete.json',dict(all_attested_contexts_absent=True))]:refs[field]=self.put(str((root/filename).relative_to(self.f.root)),value)
            resources[name]=refs
            if name!='baseline':steps.append(dict(stage=name if name=='main'else'recovery',profile='asr-fp32-sdpa',source_count=3,sources=2,new_ASR_calls=0,new_FA_calls=0,resource_returned=True,parent_result_sha256=refs['parent_result']['sha256']))
        matrix=self.put('pilot-matrix.json',dict(variants={'asr-fp32-sdpa':dict(track=tracks['main'],freeze=dict(bundle_sha256='b'*64)),'asr-fp32-eager':dict(track=tracks['baseline'],freeze=dict(bundle_sha256='b'*64))}))
        plan=self.put('pilot-plan.json',dict(main_prepared=matrix))
        result=self.put('pilot-result.json',dict(schema='annotation-bounded-speed-pilot-result/v1',purpose='cpu_fixture',all_finite_audits_completed=True,database_created=False,steps=steps,plan=plan))
        comparison=dict(schema='redimnet2-annotation-speed-comparison/v1',mode='speed',candidate_contract_sha256=self.candidate['sha256'],baseline_contract_sha256=self.baseline_contract['sha256'],cold_speed_ratio_valid=True,cold_end_to_end_speed_ratio=2,all_planned_sources_accounted=True,full_sidecar_commit_readback_verified=True,source_count=3,baseline_performance=dict(actual_compute_execution_verified=True),candidate_performance=dict(actual_compute_execution_verified=True),baseline_spec_sha256=tracks['baseline']['sha256'],candidate_spec_sha256=tracks['main']['sha256'])
        comp=self.put('comparison.json',comparison)
        recovery=dict(schema='redimnet2-annotation-speed-comparison/v1',mode='recovery',candidate_contract_sha256=self.candidate['sha256'],strict_recovery_native_results_equal=True,all_planned_sources_accounted=True,full_sidecar_commit_readback_verified=True,source_count=2,candidate_performance=dict(totals=dict(sidecars_committed=0,sidecar_cache_hits=2,asr_rows=0,aligner_rows=0)),baseline_spec_sha256=tracks['main']['sha256'],candidate_spec_sha256=tracks['recovery']['sha256'])
        evidence=self.put('quality-evidence.json',dict(synthetic_fixture=True))
        quality=self.put('quality.json',dict(schema=M.QUALITY_SCHEMA,purpose='cpu_fixture',comparison=comp,candidate_contract=self.candidate,review_complete=True,all_changed_rows_dispositioned=True,text_quality_accepted=True,complete_word_boundary_quality_accepted=True,candidate_labels_released=False,review_evidence=[evidence]))
        selected=dict(schema=M.ADMISSION_SCHEMA,purpose='cpu_fixture',candidate_contract=self.candidate,comparison=comp,recovery_comparison=self.put('recovery-comparison.json',recovery),pilot_result=result,quality_review=quality,selected_asr_compute=dict(dtype='float32',attention='sdpa'),selected_profile_name='asr-fp32-sdpa',main_track=tracks['main'],baseline_track=tracks['baseline'],recovery_track=tracks['recovery'],resource_evidence=resources)
        return self.put('selected-pilot.json',selected),selected

    def rewrite_selected(self,field,value):
        self.evidence[field]=value;self.selected=self.put('selected-pilot.json',self.evidence);self.runtime['selected_pilot_admission']=self.selected;self.save()

    def test_complete_small_control_chain_no_database(self):
        with patch('sqlite3.connect',side_effect=AssertionError('database forbidden')):c=M.compile_runtime(self.pin)
        self.assertEqual(c['index']['source_count'],3);self.assertEqual(len(c['ancestor_paths']),2)
        self.assertEqual(c['original_commands'][0][-5:],['--live','--spec',c['old']['admission']['path'],'--spec-sha',c['old']['admission']['sha256']])

    def test_false_compute_observation_refuses_even_when_speed_flag_true(self):
        comp=json.loads(Path(self.evidence['comparison']['path']).read_text());comp['candidate_performance']['actual_compute_execution_verified']=False
        self.rewrite_selected('comparison',self.put('comparison-false.json',comp))
        with self.assertRaisesRegex(ValueError,'actual cold'):M.compile_runtime(self.pin)

    def test_unrelated_result_plan_cannot_admit(self):
        result=json.loads(Path(self.evidence['pilot_result']['path']).read_text());result['plan']=self.put('unrelated.json',dict(main_prepared=self.put('wrong-matrix.json',dict(variants={}))))
        self.rewrite_selected('pilot_result',self.put('bad-result.json',result))
        with self.assertRaisesRegex(ValueError,'does not bind'):M.compile_runtime(self.pin)

    def test_cpu_qualification_cannot_be_production_evidence(self):
        self.runtime['purpose']='production';self.save()
        with self.assertRaisesRegex(ValueError,'purpose differs'):M.compile_runtime(self.pin)

    def test_quality_review_missing_refuses(self):
        q=json.loads(Path(self.evidence['quality_review']['path']).read_text());q['complete_word_boundary_quality_accepted']=False
        self.rewrite_selected('quality_review',self.put('quality-bad.json',q))
        with self.assertRaisesRegex(ValueError,'word-boundary'):M.compile_runtime(self.pin)

    def test_recovery_model_call_refuses(self):
        r=json.loads(Path(self.evidence['recovery_comparison']['path']).read_text());r['candidate_performance']['totals']['asr_rows']=1
        self.rewrite_selected('recovery_comparison',self.put('recovery-bad.json',r))
        with self.assertRaisesRegex(ValueError,'without model calls'):M.compile_runtime(self.pin)

    def test_current_producer_index_must_match_previous_chain(self):
        prior=json.loads(Path(self.runtime['previous_runtime_spec']['path']).read_text());union=json.loads(Path(prior['union_spec']['path']).read_text());union['new_producer']['index']['sha256']='0'*64
        prior['union_spec']=self.put('bad-union.json',union);self.runtime['previous_runtime_spec']=self.put('bad-prior.json',prior);self.save()
        with self.assertRaisesRegex(ValueError,'not bound'):M.compile_runtime(self.pin)

    def test_parent_and_child_hold_all_ancestor_and_current_locks(self):
        with M.RuntimeLease(self.pin)as parent:
            self.assertEqual(len(parent.pass_fds),4)
            child=M.RuntimeLease.adopt(self.pin,parent.ticket_evidence)
            parent.__exit__(None,None,None)
            try:
                child.assert_held()
                for path in child.compiled['ancestor_paths']+[str(self.f.queue/'.parent.lock')]:
                    with Path(path).open('rb')as f:
                        with self.assertRaises(BlockingIOError):fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:child.__exit__(None,None,None)

    def test_ancestor_lock_inode_spoof_refused(self):
        with M.RuntimeLease(self.pin)as parent:
            e=parent.ticket_evidence;e['ancestor_locks'][0]['inode']+=1
            with self.assertRaisesRegex(ValueError,'identity differs'):M.RuntimeLease.adopt(self.pin,e)

    def test_evidence_changed_after_admission_refuses_before_next_gpu(self):
        with M.RuntimeLease(self.pin)as parent:
            p=Path(self.evidence['resource_evidence']['main']['terminal_drain']['path']);p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'pinned file differs'):parent.verify()

    def test_runtime_routes_only_explicit_thin(self):
        a=SimpleNamespace(shard_index=self.index['path'],index_sha=self.index['sha256'],bounded_runtime_spec=None,bounded_runtime_sha=None)
        with self.assertRaisesRegex(ValueError,'requires explicit'):M.runtime_options(a)
        a.bounded_runtime_spec=self.pin['path'];a.bounded_runtime_sha=self.pin['sha256'];self.assertEqual(M.runtime_options(a),self.pin)
        a.shard_index=str(self.f.index);a.index_sha=self.f.isha
        with self.assertRaisesRegex(ValueError,'v1 refuses'):M.runtime_options(a)

    def test_no_freshness_refresh_after_runtime_lease_admitted(self):
        with M.RuntimeLease(self.pin)as parent:
            with patch.object(H.P,'terminal_observation',side_effect=AssertionError('do not refresh during legitimate long GPU run')):
                parent.assert_held();parent.verify();parent.pass_fds

    def test_unknown_compute_source_change_cannot_borrow_pilot(self):
        a=copy.deepcopy(self.c);b=copy.deepcopy(self.c)
        a['engine_source_files']={'qwen_backend.py':'a'*64,'worker.py':'b'*64}
        b['engine_source_files']={'qwen_backend.py':'c'*64,'worker.py':'d'*64}
        with self.assertRaisesRegex(ValueError,'semantic computation'):M.same_profile(a,b,allow_control_changes=True)

    def test_current_worker_retains_old_producer_without_source_or_model_execution(self):
        import worker
        self.f.publish(3)
        args=SimpleNamespace(shard_index=self.index['path'],index_sha=self.index['sha256'],contract=self.cp['path'],contract_sha=self.cp['sha256'],queue_root=self.thin['new_queue_root'],output_root=self.thin['new_sidecar_root'])
        before=self.f.snapshot()
        with M.RuntimeLease(self.pin)as lease:
            queue=lease.queue(args,self.c)
            with queue.claim_next('fixture')as claim:
                sources=claim.sources();store=SimpleNamespace(lookup=lambda source:None)
                counters=dict(assigned_source_rows=0,retained_previous_producer_hits=0,retained_previous_decision_counts=dict(candidate=0,unknown=0,failed=0),decision_counts=dict(candidate=0,unknown=0,failed=0),pipeline_seconds={key:0.0 for key in worker.PIPELINE_TIMERS})
                with patch.object(worker,'inspect_source',side_effect=AssertionError('must not decode retained input')),patch.object(worker,'read_window',side_effect=AssertionError('must not read PCM')):
                    backend=Mock();backend.transcribe.side_effect=AssertionError('must not rerun ASR');backend.align.side_effect=AssertionError('must not rerun FA')
                    receipts=worker.process_group(sources,claim,store,None,backend,self.c,args,{},counters)
                claim.finish(receipts)
                self.assertEqual(counters['retained_previous_producer_hits'],1)
                self.assertEqual(receipts[0]['producer_contract_sha256'],self.f.csha)
                self.assertEqual(receipts[0]['job_key'],H.P.old_job_key(sources[0],self.f.csha));backend.transcribe.assert_not_called();backend.align.assert_not_called()
        self.assertEqual(self.f.snapshot(),before)

    def test_dual_old_new_source_results_are_not_silently_preferred(self):
        import worker
        self.f.publish(3)
        args=SimpleNamespace(shard_index=self.index['path'],index_sha=self.index['sha256'],contract=self.cp['path'],contract_sha=self.cp['sha256'],queue_root=self.thin['new_queue_root'],output_root=self.thin['new_sidecar_root'])
        with M.RuntimeLease(self.pin)as lease:
            queue=lease.queue(args,self.c)
            with queue.claim_next('fixture')as claim:
                counters=dict(assigned_source_rows=0,pipeline_seconds={key:0.0 for key in worker.PIPELINE_TIMERS})
                with self.assertRaisesRegex(RuntimeError,'ambiguous'):
                    worker.process_group(claim.sources(),claim,SimpleNamespace(lookup=lambda source:{'same_source':True}),None,None,self.c,args,{},counters)
                self.assertFalse(queue.chunk_receipt_path(claim.shard['id']).exists())

    def test_report_uses_only_bounded_metadata_and_refuses_full(self):
        import report
        args=SimpleNamespace(shard_index=self.index['path'],index_sha=self.index['sha256'],contract=self.cp['path'],contract_sha=self.cp['sha256'],queue_root=self.thin['new_queue_root'],output_root=self.thin['new_sidecar_root'])
        for i in (3,4,5):self.f.publish(i)
        with M.RuntimeLease(self.pin)as lease:
            queue=lease.queue(args,self.c)
            # Sidecar tree identity belongs to the new producer; old records stay in old tree.
            tree=Path(args.output_root)/self.c['version'];tree.mkdir(parents=True)
            self.put(str((tree/'identity.json').relative_to(self.f.root)),dict(schema='redimnet2-sidecar-tree/v1',version=self.c['version'],contract_sha256=args.contract_sha))
            while not queue.is_complete():
                with queue.claim_next('fixture')as claim:claim.finish([queue.previous_receipt(claim,s)for s in claim.sources()])
            lock=Path(args.queue_root)/'.parent.lock';lock.touch()
            with lock.open('rb')as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('sqlite3.connect',side_effect=AssertionError('no database')):
                    value=report.verify_run(self.cp['path'],self.index['path'],args.queue_root,args.output_root,bounded_parent_lock_fd=stream.fileno())
                self.assertTrue(value['passed']);self.assertFalse(value['full_sidecar_readback_verified']);self.assertEqual(value['sidecars_individually_reopened'],0)
                self.assertFalse(value['GPU_resource_return_verified_by_this_tool'])
                with self.assertRaisesRegex(RuntimeError,'forbids full-library'):
                    report.verify_run(self.cp['path'],self.index['path'],args.queue_root,args.output_root,full=True,bounded_parent_lock_fd=stream.fileno())


    def test_formal_bounded_cannot_request_two_development_uuids(self):
        import gpu_launch
        args=SimpleNamespace(bundle_sha='b'*64,contract=self.cp['path'],contract_sha=self.cp['sha256'],shard_index=self.index['path'],index_sha=self.index['sha256'],bounded_runtime_spec=self.pin['path'],bounded_runtime_sha=self.pin['sha256'],run_root=self.runtime['new_run_root'],device_lock_root=self.runtime['device_lock_root'],target_uuids='GPU-a,GPU-b',full_verify=False)
        fake=Mock();fake.compiled={'spec':dict(self.runtime,purpose='production',new_code_root=str(gpu_launch.HERE),new_bundle=dict(path=str(gpu_launch.HERE/'bundle_files.json'),sha256='b'*64))}
        context=Mock();context.__enter__=Mock(return_value=fake);context.__exit__=Mock(return_value=False)
        with patch.object(gpu_launch,'verify_bundle',return_value={}),patch.object(gpu_launch,'read_contract',return_value=dict(purpose='silver_annotation')),patch.object(M,'RuntimeLease',return_value=context),patch.object(gpu_launch,'_run')as launch:
            with self.assertRaisesRegex(RuntimeError,'requires auto8'):gpu_launch.run(args)
            launch.assert_not_called()

    def test_readonly_cli_explicit_bounded_does_not_construct_queue(self):
        import readonly_progress
        args=SimpleNamespace(shard_index=self.index['path'],index_sha=self.index['sha256'],contract=self.cp['path'],contract_sha=self.cp['sha256'],queue_root=self.thin['new_queue_root'],output_root=self.thin['new_sidecar_root'])
        with M.RuntimeLease(self.pin)as lease:lease.queue(args,self.c)
        before={str(p):p.read_bytes()for p in Path(args.queue_root).rglob('*')if p.is_file()}
        out=self.f.root/'readonly.json'
        with patch.object(Q,'BoundedWorkQueue',side_effect=AssertionError('read-only must not construct queue')),patch('sqlite3.connect',side_effect=AssertionError('no database')):
            readonly_progress.main(['--index',self.index['path'],'--index-sha',self.index['sha256'],'--queue-root',args.queue_root,'--contract-sha',args.contract_sha,'--output',str(out),'--bounded'])
        value=json.loads(out.read_text());self.assertEqual(value['queue_writes'],0);self.assertFalse(value['database_created'])
        self.assertEqual({str(p):p.read_bytes()for p in Path(args.queue_root).rglob('*')if p.is_file()},before)


    def test_unique_producer_claim_is_permanent_and_same_run_can_resume(self):
        with M.RuntimeLease(self.pin)as first:
            claim=first.unique_claim_pin;before=Path(claim['path']).read_bytes()
        self.assertTrue(Path(claim['path']).exists())
        with M.RuntimeLease(self.pin)as resumed:self.assertEqual(resumed.unique_claim_pin,claim)
        self.assertEqual(Path(claim['path']).read_bytes(),before)

    def test_changed_new_producer_cannot_reuse_old_terminal_snapshot(self):
        with M.RuntimeLease(self.pin)as first:claim_path=Path(first.unique_claim_pin['path'])
        preserved=claim_path.read_bytes()
        self.runtime['selected_pilot_admission']=dict(self.runtime['selected_pilot_admission'])
        # 仅给完全相同的选择材料换一个声明文件，就已是另一永久业务身份；不能默默重跑。
        self.runtime['selected_pilot_admission']=self.put('other-selection.json',self.evidence);self.save()
        with self.assertRaisesRegex(ValueError,'permanently claimed'):M.RuntimeLease(self.pin).__enter__()
        self.assertEqual(claim_path.read_bytes(),preserved)



if __name__=='__main__':unittest.main()
