"""有限中途恢复控制、真实文件事务与拒绝反例；无模型/GPU/原音。"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import run_native_partial_recovery_probe as P
import native_transformers_batch as S
from contracts import canonical_bytes,digest
import test_audit_native_transformers_batch as AuditFixtures
contract=AuditFixtures.contract


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.c=contract(8);self.c['purpose']='pilot_validation';self.records={};self.rows={};self.waves={}
        for i in range(30):
            sid='synthetic-'+str(i).zfill(2);source=dict(source_id=sid,dataset=('voxceleb2','voxblink2','cnceleb2')[i%3],
                sidecar_relpath=f'voxceleb2/s/{i}.wav.annotation.json');self.rows[sid]=source;windows=[]
            for j in range(2 if i==0 else 1):
                wid='w'+str(j);pcm=np.full(160+i+j,np.float32((i+j)/100),dtype=np.float32);self.waves[sid+':'+wid]=pcm
                windows.append(dict(window_id=wid,preprocessing_status='completed',input_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                    model_input_frames=len(pcm),model_sample_rate=16000))
            self.records[sid]=dict(source=source,source_identity=dict(decode_status='decoded_complete'),windows=windows)
        self.v=dict(contract=self.c,track=dict(contract=dict(path=str(self.root/'original-contract.json'),sha256=digest(self.c)),
            index=dict(path=str(self.root/'original-inputs/index.json'),sha256='3'*64)),
            track_ref=dict(path=str(self.root/'original-track.json'),sha256='4'*64),
            fields={'--bundle-sha':'5'*64,'--device-lock-root':str(self.root/'global-locks')},
            members={p:P.file_identity(Path(P.__file__).parent/p)['sha256']for p in P.CONTROL_DEPENDENCIES})
        self.plan=dict(schema=P.SCHEMA,purpose='cpu_fixture',candidate_prepared=dict(path=str(self.root/'prepared/prepared.json'),sha256='6'*64),
            profile='native-fp32-sdpa-b8',anchor_source_id='synthetic-00',protected_roots=[str(self.root/'PTN'),str(self.root/'old-annotation')],wall_budget_seconds=1800,
            controller_sha256=P.file_identity(Path(P.__file__).resolve())['sha256'])
        self.ref=self.put(self.root/'plan.json',self.plan);self.output=self.root/'new-probe'
    def put(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical_bytes(value));return dict(path=str(path),sha256=P.file_identity(path)['sha256'])
    def compile(self):
        with patch.object(P,'load_matrix',return_value=dict(variants={self.plan['profile']:self.v},rows=self.rows)),\
             patch('audit_native_transformers_batch.load_track',return_value=dict(records=self.records)):
            return P.compile_plan(self.ref,self.output)
    def compiled(self):
        value=self.compile();value['variant'].update(argv=['fake-python','-B','fake-frozen-gpu_launch.py'],code=self.root/'frozen-code')
        return value
    def native_store(self,c,v):
        expected=P.expected_plan(c,c['requests'],fixture=True);identity=expected['store_identity'];group=P.group_path(c,v)
        store=S.NativeTransformersBatchStore(group,identity['producer_sha256'],synthetic_fixture=True,
            contract_sha256=identity['contract_sha256'],model_sha256=identity['model_sha256'],max_batch_rows=c['batch'])
        rows=[dict(r,audio=self.waves[r['row_id']])for r in c['requests']]
        callback=lambda rows:AuditFixtures.NativeAuditTests.infer(SimpleNamespace(c=self.c),rows)
        return store,rows,callback
    def partial(self,c,v):
        store,rows,callback=self.native_store(c,v)
        def stop(_):raise KeyboardInterrupt('synthetic controlled stop after durable packet')
        with self.assertRaises(KeyboardInterrupt):store.run_batch(rows,infer_batch=callback,after_durable_commit=stop)
        return store,rows,callback

    def test_default_check_reads_fixed30_only_and_creates_nothing(self):
        c=self.compile();self.assertEqual(len(c['sources']),30);self.assertEqual(len(c['requests']),9)
        self.assertEqual(c['sources'][0]['source_id'],'synthetic-00');self.assertFalse(self.output.exists())
        with patch.object(P,'compile_plan',return_value=c),patch.object(P,'SerialExecutor',side_effect=AssertionError('no GPU command')):
            result=P.run(self.ref,self.output)
        self.assertFalse(result['GPU_executed']);self.assertFalse(result['partial_recovery_verified']);self.assertFalse(self.output.exists())
    def test_protected_output_and_changed_anchor_or_budget_refused(self):
        self.plan['protected_roots']=[str(self.output)];self.ref=self.put(self.root/'plan.json',self.plan)
        with self.assertRaisesRegex(RuntimeError,'overlaps'):self.compile()
        self.plan['protected_roots']=[str(self.root/'PTN')];self.plan['anchor_source_id']='unknown';self.ref=self.put(self.root/'plan.json',self.plan)
        with self.assertRaisesRegex(RuntimeError,'anchor'):self.compile()
        self.plan['wall_budget_seconds']=1801;self.ref=self.put(self.root/'plan.json',self.plan)
        with self.assertRaisesRegex(RuntimeError,'1800'):self.compile()
    def test_controller_and_frozen_lifecycle_source_pins_cannot_drift(self):
        self.plan['controller_sha256']='0'*64;self.ref=self.put(self.root/'plan.json',self.plan)
        with self.assertRaisesRegex(RuntimeError,'controller source SHA'):self.compile()
        self.plan['controller_sha256']=P.file_identity(Path(P.__file__).resolve())['sha256'];self.ref=self.put(self.root/'plan.json',self.plan)
        self.v['members']['run_bounded_speed_pilot.py']='0'*64
        with self.assertRaisesRegex(RuntimeError,'helpers must match'):self.compile()
    def test_first_group_must_leave_more_than_one_cohort(self):
        self.records['synthetic-00']['windows']=self.records['synthetic-00']['windows'][:1]
        with self.assertRaisesRegex(RuntimeError,r'9\+windows'):self.compile()
    def test_production_cannot_use_fake_executor_or_fake_resource_release(self):
        c=self.compiled();c['plan']['purpose']='production'
        with patch.object(P,'compile_plan',return_value=c),self.assertRaisesRegex(RuntimeError,'fake executor'):
            P.run(self.ref,self.output,True,_fixture_executor=object(),_fixture_release=object())
        self.assertFalse(self.output.exists())
    def test_real_atomic_first_packet_triggers_only_owned_cancel_then_partial_resume(self):
        c=self.compiled();v=dict(self.v,track=dict(self.v['track'],output_root=str(self.root/'interrupted/labels')))
        store,rows,callback=self.partial(c,v);cancel=threading.Event();state={};evidence=self.root/'evidence';evidence.mkdir()
        P.watch_first_packet(c,v,cancel,threading.Event(),threading.Event(),time.monotonic()+2,evidence,state)
        self.assertTrue(cancel.is_set());self.assertIn('trigger',state);self.assertNotIn('error',state)
        before=P.snapshot_partial(c,v);self.assertEqual(len(before['committed_packets']),1);self.assertEqual(len(before['uncommitted_cohorts']),1)
        seen=[]
        result=store.run_batch(rows,infer_batch=lambda batch:(seen.append([r['row_id']for r in batch])or callback(batch)))
        self.assertEqual(len(seen),1);self.assertEqual(len(seen[0]),1);self.assertEqual(store.metrics['stage_cache_hits'],8)
        self.assertEqual(sum(v['durable_native_batch']['cache_hit']for v in result.values()),8);P.verify_preserved(before)
    def test_changed_atomic_packet_or_fixed_plan_refused_without_cancelling_foreign_work(self):
        c=self.compiled();v=dict(self.v,track=dict(self.v['track'],output_root=str(self.root/'labels')));self.partial(c,v)
        plan=P.expected_plan(c,c['requests'],fixture=True);root=P.group_path(c,v)
        p=root/'batches'/(plan['cohorts'][0]['cohort_sha256']+'.json');packet=json.loads(p.read_text());packet['content']['outputs']={};self.put(p,packet)
        with self.assertRaisesRegex(RuntimeError,'identity'):P.checked_packet(root,plan,plan['cohorts'][0])
        cancel=threading.Event();state={};evidence=self.root/'evidence';evidence.mkdir()
        P.watch_first_packet(c,v,cancel,threading.Event(),threading.Event(),time.monotonic()+2,evidence,state)
        self.assertTrue(cancel.is_set());self.assertIn('error',state)
    def test_link_visible_target_waits_until_atomic_tmp_unlinked_before_cancel(self):
        c=self.compiled();v=dict(self.v,track=dict(self.v['track'],output_root=str(self.root/'labels')));self.partial(c,v)
        plan=P.expected_plan(c,c['requests'],fixture=True);root=P.group_path(c,v)
        packet=root/'batches'/(plan['cohorts'][0]['cohort_sha256']+'.json');tmp=packet.with_name('.pending-publication.tmp')
        os.link(packet,tmp);self.assertEqual(packet.stat().st_nlink,2)
        with patch.object(P,'file_identity',side_effect=AssertionError('pending packet must not be read as published')):
            self.assertIsNone(P.checked_packet(root,plan,plan['cohorts'][0]))
        cancel=threading.Event();state={};stop=threading.Event();evidence=self.root/'evidence';evidence.mkdir()
        watcher=threading.Thread(target=P.watch_first_packet,args=(c,v,cancel,threading.Event(),stop,time.monotonic()+2,evidence,state))
        watcher.start();time.sleep(.08);self.assertFalse(cancel.is_set())
        # 仅CPU夹具模拟原atomic_new正常unlink；生产控制器不会删除任何tmp。
        tmp.unlink();watcher.join(2);self.assertFalse(watcher.is_alive());self.assertTrue(cancel.is_set());self.assertIn('trigger',state)
    def test_orphan_sidecar_and_partial_file_are_retained_and_refused(self):
        c=self.compiled();v=dict(self.v,track=dict(self.v['track'],output_root=str(self.root/'labels')));self.partial(c,v)
        side=Path(v['track']['output_root'])/v['contract']['version']/c['sources'][0]['sidecar_relpath'];side.parent.mkdir(parents=True);side.write_bytes(b'orphan-evidence')
        with self.assertRaisesRegex(RuntimeError,'orphan sidecar'):P.snapshot_partial(c,v)
        self.assertEqual(side.read_bytes(),b'orphan-evidence')
    def test_orphan_atomic_tmp_is_not_deleted(self):
        c=self.compiled();v=dict(self.v,track=dict(self.v['track'],output_root=str(self.root/'labels')));self.partial(c,v)
        p=P.group_path(c,v)/'.packet.interrupted.tmp';p.write_bytes(b'partial-original')
        with self.assertRaisesRegex(RuntimeError,'partial file'):P.snapshot_partial(c,v)
        self.assertEqual(p.read_bytes(),b'partial-original')
    def test_previously_committed_packet_mutation_or_replacement_refused(self):
        c=self.compiled();v=dict(self.v,track=dict(self.v['track'],output_root=str(self.root/'labels')));self.partial(c,v)
        before=P.snapshot_partial(c,v);p=Path(next(iter(before['committed_packets'].values()))['path']);p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(RuntimeError,'packet bytes/inode/mtime'):P.verify_preserved(before)
    def test_full_group_completion_is_not_misrepresented_as_partial(self):
        c=self.compiled();v=dict(self.v,track=dict(self.v['track'],output_root=str(self.root/'labels')))
        store,rows,callback=self.native_store(c,v);store.run_batch(rows,infer_batch=callback)
        snap=P.snapshot_partial(c,v);self.assertEqual(snap['uncommitted_cohorts'],[])
        # 调度入口必须将此状态视为inconclusive，而不能把全命中恢复当半批恢复。
        self.assertEqual(len(snap['committed_packets']),2)
    def test_release_reacquires_same_uuid_locks_before_five_second_probe_and_releases(self):
        from gpu_vendor import resources as G
        events=[];uuids=['GPU-00000000-0000-0000-0000-000000000001','GPU-00000000-0000-0000-0000-000000000002']
        devices=[dict(uuid=u,index=i,name='NVIDIA A800',total_mib=81920)for i,u in enumerate(uuids)]
        class Locks:
            def __init__(self,root,assigned):
                self.evidence=[dict(uuid=u,synthetic_CPU_lock=True)for u in assigned];test.assertEqual(assigned,uuids)
            def __enter__(self):events.append('locks-held');return self
            def __exit__(self,*args):events.append('locks-released')
        def probe(indices,budget,**kwargs):
            self.assertEqual(events,['locks-held']);self.assertEqual(indices,[0,1]);self.assertEqual(kwargs['interval_seconds'],5.)
            self.assertEqual(budget,60000);events.append('double-probe');return dict(devices=devices,interval_seconds=5.)
        test=self
        with patch.object(G,'capture_snapshot',return_value=dict(devices=devices)),patch.object(G,'DeviceLocks',Locks),patch.object(G,'double_probe',side_effect=probe):
            result=P.prove_release(self.v,uuids)
        self.assertEqual(events,['locks-held','double-probe','locks-released']);self.assertTrue(result['no_process_signalled'])
    def test_normalizing_cross_tree_only_removes_own_packet_absolute_prefix(self):
        c=self.compiled();tree=self.root/'labels'/self.c['version'];packet=tree/'.native_torch_cohorts/a/b/batches/x.json'
        track=dict(spec=dict(output_root=str(self.root/'labels')),contract=self.c,records={'x':dict(windows=[dict(asr=dict(native={'tokens':[1,2]},durable_native_batch=dict(packet_path=str(packet),cache_hit=True)))])})
        view=P.normalized_results(track);ref=view['records']['x']['windows'][0]['asr']['durable_native_batch']
        self.assertEqual(ref['packet_path'],'.native_torch_cohorts/a/b/batches/x.json');self.assertTrue(ref['cache_hit'])
        self.assertEqual(view['records']['x']['windows'][0]['asr']['native'],{'tokens':[1,2]})
        track['records']['x']['windows'][0]['asr']['durable_native_batch']['packet_path']=str(self.root/'foreign/x.json')
        with self.assertRaisesRegex(RuntimeError,'escaped'):P.normalized_results(track)
    def test_resume_argv_keeps_same_queue_labels_contract_and_new_resource_attempt(self):
        c=self.compiled();root=self.root/'new-probe';index={'path':str(root/'inputs/index.json'),'sha256':'1'*64}
        first=P.variant_for(c['variant'],root,index,'resources-attempt1',uuids=['u0','u1']);second=P.variant_for(c['variant'],root,index,'resources-attempt2',uuids=['u0','u1'])
        for key in ('--queue-root','--output-root','--contract','--contract-sha','--index-sha'):
            self.assertEqual(first['fields'].get(key),second['fields'].get(key))
        self.assertNotEqual(first['fields']['--output'],second['fields']['--output']);self.assertEqual(second['fields']['--target-uuids'],'u0,u1')
    def test_fixture_serial_flow_waits_owned_interruption_release_then_resumes_real_whole_packets(self):
        c=self.compiled();events=[];tracks={};test=self
        def stage_variant(argv):
            flag=lambda key:argv[argv.index(key)+1]
            return dict(c['variant'],track=dict(output_root=flag('--output-root'),queue_root=flag('--queue-root'),resources_root=flag('--output')))
        def finish(v):
            records=copy.deepcopy(test.records);hits=0
            tree=Path(v['track']['output_root'])/test.c['version']
            for start in range(0,30,8):
                group=c['sources'][start:start+8];requests=P.planned_requests(test.records,group)
                identity=P.expected_plan(c,requests,fixture=True)['store_identity']
                store=S.NativeTransformersBatchStore(tree/'.native_torch_cohorts'/P.SHARD_ID/digest(group),identity['producer_sha256'],
                    synthetic_fixture=True,contract_sha256=identity['contract_sha256'],model_sha256=identity['model_sha256'],max_batch_rows=8)
                inputs=[dict(r,audio=test.waves[r['row_id']])for r in requests]
                outputs=store.run_batch(inputs,infer_batch=lambda rows:AuditFixtures.NativeAuditTests.infer(SimpleNamespace(c=test.c),rows))
                hits+=store.metrics['stage_cache_hits']
                for source in group:
                    record=records[source['source_id']];record.update(decision={'state':'unknown'},units=[],quality_axes={},errors=[],vad={},contract={},provenance={})
                    for w in record['windows']:
                        w.update(asr=outputs[source['source_id']+':'+w['window_id']],alignment={},asr_cache_hit=outputs[source['source_id']+':'+w['window_id']]['durable_native_batch']['cache_hit'],aligner_cache_hit=False)
            tracks[v['track']['resources_root']]=dict(contract=test.c,contract_sha256=digest(test.c),spec_sha256='a'*64,records=records,
                spec=v['track'],performance=dict(totals={'native_ASR_stage_cache_hits':hits}),decision_counts={'unknown':30})
            resource=Path(v['track']['resources_root']);resource.mkdir(parents=True)
            pp=test.put(resource/'parent_result.json',dict(UUIDs=['GPU-00000000-0000-0000-0000-000000000001','GPU-00000000-0000-0000-0000-000000000002']))
            tracks[v['track']['resources_root']]['parent_result_sha256']=pp['sha256']
        class Executor:
            def __init__(self,output,deadline,cancel):self.cancel=cancel;self.output=output
            def command(self,argv,name,code):
                events.append(name);v=stage_variant(argv)
                if name!='interrupted':finish(v);return 0
                test.partial(c,v)
                timeout=time.monotonic()+2
                while not self.cancel.is_set()and time.monotonic()<timeout:time.sleep(.01)
                test.assertTrue(self.cancel.is_set());test.put(self.output/'interrupted-TERM.json',dict(synthetic_exact_owned_parent=True))
                events.append('original-parent-cleanup-returned');raise RuntimeError('synthetic owned parent stopped and waited')
        def audit(path,sha):
            track=P.pin(dict(path=path,sha256=sha));return tracks[track['resources_root']]
        def release(v,uuids):
            self.assertEqual(events[-1],'original-parent-cleanup-returned');events.append('locked-five-second-double-idle-fixture')
            return dict(synthetic_CPU_only=True,interval_seconds=5,uuids=uuids)
        with patch.object(P,'compile_plan',return_value=c),patch.object(P,'verify_parent',return_value={}),\
             patch('audit_native_transformers_batch.load_track',side_effect=audit),patch('sqlite3.connect',side_effect=AssertionError('no database')):
            result=P.run(self.ref,self.output,True,_fixture_executor=Executor,_fixture_release=release)
        self.assertEqual(events,['cold','interrupted','original-parent-cleanup-returned','locked-five-second-double-idle-fixture','resumed'])
        self.assertTrue(result['partial_recovery_verified']);self.assertFalse(result['GPU_executed']);self.assertEqual(result['committed_ASR_rows_reused'],8)


if __name__=='__main__':unittest.main()
