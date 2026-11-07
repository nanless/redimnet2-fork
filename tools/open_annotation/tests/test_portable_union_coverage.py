"""跨节点旧内容账本的六源CPU联合测试；不使用模型、GPU或服务器。"""
import copy
import datetime
import json
import itertools
import sqlite3
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'tests')]
import union_coverage as U
import cloud_gate as G
import portable_source_identity as P
import prepare_speed_continuation as C
from test_union_coverage import UnionFixture,pin
from union_consumer import UnionConsumer


class PortableUnionFixture:
    def __init__(self):
        self.f=UnionFixture();b=self.f.base;self.root=self.f.root
        self.folder=self.root/'portable-evidence';self.folder.mkdir()
        target=self.folder/'shared';target.mkdir();alias=self.folder/'old-shared';alias.symlink_to(target,target_is_directory=True)
        self.policy={'schema':P.SPEC_SCHEMA,'purpose':'cpu_fixture','source_identity_mode':P.MODE,
            'old_contract':pin(b.contract),'full_index':pin(b.index),'old_queue_root':str(b.queue),
            'old_sidecar_tree':str(b.tree),'source_alias':{'alias_path':str(alias),'target_path':str(target)},
            'allowed_difference':['device'],'encoded_hash_algorithm':'sha256',
            'model_API_calls_allowed':0,'release_training_labels':False}
        self.policy_path=self.folder/'portable.json';self.policy_ref=self.write(self.policy_path,self.policy)
        source=b.sources[2];path=b.tree/source['sidecar_relpath'];commit_path=path.with_name(path.name+'.commit.json')
        record=json.loads(path.read_bytes())
        record['source_identity']['file_sha256']=U.sha_file(Path(source['audio_path']))
        for field in ('stat_before','stat_after'):record['source_identity'][field]['device']+=1
        path.write_bytes(C.canonical(record));marker=json.loads(commit_path.read_bytes())
        marker.update(sidecar_sha256=U.sha_file(path),bytes=path.stat().st_size);commit_path.write_bytes(C.canonical(marker))
        kept=self.f.plan['retained_partial_sources'][0];kept.update(sidecar_sha256=U.sha_file(path),commit_sha256=U.sha_file(commit_path))
        self.entries=[P.prove_source_identity(self.policy,self.policy_ref['sha256'],source,record,b.csha,
                                             U.sha_file(path),U.sha_file(commit_path))]
        self.ledger_path=self.folder/'evidence.jsonl'
        self.current_terminal_path=self.folder/'current-terminal.json'
        current=json.loads(b.terminal.read_bytes());current['active_workers']=0
        self.current_terminal_ref=self.write(self.current_terminal_path,current)
        self.f.spec['current_terminal_observation']=self.current_terminal_ref
        self.refresh()
    def write(self,path,value,lines=False):
        path.write_bytes(b''.join(C.canonical(v)for v in value)if lines else C.canonical(value));return pin(path)
    def refresh(self):
        self.policy_ref=self.write(self.policy_path,self.policy)
        ledger={**self.write(self.ledger_path,self.entries,True),'count':len(self.entries)}
        self.f.plan.update(source_identity_mode=P.MODE,portable_source_spec_ref=self.policy_ref,
            portable_source_ledger_ref=ledger,fresh_media_scope=U.PORTABLE_FRESH_SCOPE,
            completed_media_scope=U.PORTABLE_COMPLETED_SCOPE)
        self.f.spec.update(source_identity_mode=P.MODE,portable_source_spec_ref=self.policy_ref,portable_source_ledger_ref=ledger)
        self.f.refresh_plan()
    def add_inventory(self):
        path=self.folder/'inventory.sqlite';b=self.f.base
        inventory=P.SidecarInventory(path,b.tree,self.policy,self.policy_ref['sha256'],{'fixture':6})
        try:
            inventory.scan()
            for i,source in enumerate(b.sources):inventory.select(source,'completed'if i<2 else'partial'if i<4 else'unclaimed')
            summary=inventory.finish(6)
        finally:inventory.close()
        self.f.plan.update(portable_sidecar_inventory_ref=pin(path),portable_sidecar_inventory_summary=summary)
        self.f.spec['portable_sidecar_inventory_ref']=pin(path);self.f.refresh_plan();return path
    def audit(self,mode='plan',database=False):
        return U.validate_union(self.f.spec_path,self.f.spec_sha,mode,
            output_index=self.root/'consumer.sqlite'if database else None)
    def close(self):self.f.close()


class PortableUnionTests(unittest.TestCase):
    def setUp(self):self.x=PortableUnionFixture();self.addCleanup(self.x.close)
    def test_explicit_device_bridge_has_exact_full_partition_without_media_rehash_by_auditor(self):
        before=self.x.f.old_snapshot();r=self.x.audit()
        self.assertEqual((r['source_count'],r['retained_source_count'],r['remaining_source_count']),(6,3,3))
        self.assertEqual(before,self.x.f.old_snapshot());self.assertEqual(r['portable_source_evidence_count'],1)
        self.assertEqual(r['upstream_encoded_content_verified_count'],1);self.assertFalse(r['source_audio_rehashed'])
        self.assertEqual(r['completed_media_scope'],U.PORTABLE_COMPLETED_SCOPE);self.assertFalse(r['formal_runtime_admitted'])
    def test_real_preparer_portable_ledger_unclaimed_retention_to_complete_consumer(self):
        f=self.x.f;b=f.base
        path,commit=b.publish(4);record=json.loads(path.read_bytes())
        record['source_identity']['file_sha256']=U.sha_file(Path(b.sources[4]['audio_path']))
        path.write_bytes(C.canonical(record));marker=json.loads(commit.read_bytes())
        marker.update(sidecar_sha256=U.sha_file(path),bytes=path.stat().st_size);commit.write_bytes(C.canonical(marker))
        before=f.old_snapshot();output=self.x.root/'actual-portable-preparation'
        with patch.object(C,'EXPECTED_SOURCE_COUNT',6),patch.object(C,'EXPECTED_DOMAINS',{'fixture':6}):
            plan=C.prepare(b.index,b.isha,b.queue,b.contract,b.csha,b.tree,output,True,b.terminal,b.tsha,b.task,
                portable_spec_path=self.x.policy_path,portable_spec_sha=self.x.policy_ref['sha256'])
        self.assertEqual(before,f.old_snapshot());self.assertEqual(plan['portable_source_ledger_ref']['count'],2)
        f.plan=plan;f.plan_path=output/'coverage-plan.json';f.new_index=output/'index.json'
        f.spec.update(continuation_plan=pin(f.plan_path),portable_source_spec_ref=plan['portable_source_spec_ref'],
                      portable_source_ledger_ref=plan['portable_source_ledger_ref'],
                      portable_sidecar_inventory_ref=plan['portable_sidecar_inventory_ref'])
        f.spec['new_producer']['index']=pin(f.new_index);f.write_spec()
        r=self.x.audit();self.assertEqual((r['source_count'],r['retained_source_count'],r['remaining_source_count']),(6,4,2))
        self.assertEqual(r['upstream_encoded_content_verified_count'],2)
        self.assertTrue(r['portable_sidecar_inventory_verified'])
        f.complete();r=self.x.audit('complete',True)
        with UnionConsumer(f.spec_path,f.spec_sha,r['consumer_index']['path'],r['consumer_index']['sha256'])as reader:
            self.assertEqual([v['producer_id']for v in reader.iter_selected_sources()],['old','old','old','new','old','new'])
        self.assertEqual(before,f.old_snapshot())

    def test_complete_consumer_keeps_old_and_new_producers_and_original_bytes(self):
        before=self.x.f.old_snapshot();self.x.f.complete();r=self.x.audit('complete',True)
        with UnionConsumer(self.x.f.spec_path,self.x.f.spec_sha,r['consumer_index']['path'],r['consumer_index']['sha256'])as reader:
            rows=list(reader.iter_selected_sources())
        self.assertEqual([r['producer_id']for r in rows],['old']*3+['new']*3)
        self.assertEqual(before,self.x.f.old_snapshot());self.assertEqual(r['source_count'],6)
    def test_union_progress_has_phases_and_low_frequency_actual_counters(self):
        events=[];ticks=itertools.count(0,31)
        with patch.object(U.time,'monotonic',side_effect=lambda:next(ticks)):
            report=U.validate_union(self.x.f.spec_path,self.x.f.spec_sha,'plan',progress=events.append)
        self.assertEqual(events[0]['stage'],'started');self.assertEqual(events[-1]['stage'],'completed')
        self.assertGreater(len(events),4)
        self.assertEqual(events[-1]['full_rows_checked'],report['source_count'])
        self.assertEqual(events[-1]['retained_sources_checked'],report['retained_source_count'])
        self.assertEqual(events[-1]['remaining_rows_checked'],report['remaining_source_count'])

    def test_cloud_gate_is_only_data_readiness_and_retains_full_denominator(self):
        r=G.check_union_data(self.x.f.spec_path,self.x.f.spec_sha,'plan')
        self.assertTrue(r['data_checks_passed']);self.assertEqual(r['original_full_source_count'],6)
        self.assertEqual(r['remaining_source_count'],3);self.assertTrue(r['portable_source_ledger_verified'])
        self.assertFalse(r['GPU_execution_admitted']);self.assertFalse(r['cloud_submission_admitted'])
    def test_fixed_readonly_inventory_avoids_old_missing_PFS_exists(self):
        inventory=self.x.add_inventory();before=inventory.read_bytes();b=self.x.f.base
        missing=set()
        for source in b.sources[3:]:
            p=b.tree/source['sidecar_relpath'];missing.add(p);missing.add(p.with_name(p.name+'.commit.json'))
        exists=Path.exists
        def guarded(path):
            if path in missing:raise AssertionError('must query fixed SQLite, not missing PFS paths')
            return exists(path)
        with patch.object(Path,'exists',guarded):r=self.x.audit()
        self.assertTrue(r['portable_sidecar_inventory_verified']);self.assertEqual(inventory.read_bytes(),before)

    def test_inventory_source_or_file_pin_tamper_is_rejected(self):
        path=self.x.add_inventory()
        with sqlite3.connect(path)as db:
            db.execute("UPDATE paths SET source_id='wrong-source' WHERE kind='partial'")
        with self.assertRaisesRegex(ValueError,'inventory SHA'):self.x.audit()
        self.x.f.plan['portable_sidecar_inventory_ref']=pin(path)
        self.x.f.spec['portable_sidecar_inventory_ref']=pin(path);self.x.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'source/path/selection'):self.x.audit()

    def test_aged_current_observation_is_retained_for_complete_audit_without_forging_refresh(self):
        self.x.f.complete();current=json.loads(self.x.current_terminal_path.read_bytes())
        current['observed_utc']='2026-01-01T00:00:00+00:00'
        self.x.f.spec['current_terminal_observation']=self.x.write(self.x.current_terminal_path,current);self.x.f.write_spec()
        with self.assertRaisesRegex(ValueError,'current terminal observation stale'):self.x.audit()
        self.assertTrue(self.x.audit('complete')['exact_union_and_disjoint'])

    def test_default_reader_stays_strict_without_portable_context(self):
        """设计文档：默认（非 portable）读取必须严格，device 参与比较。"""
        b=self.x.f.base
        with self.assertRaisesRegex(ValueError,'stat changed'):
            C.retained_sidecar(b.sources[2],b.tree,b.csha,C.record_validator())
    def test_mode_must_be_explicit_in_both_spec_and_plan(self):
        self.x.f.spec.pop('source_identity_mode');self.x.f.write_spec()
        with self.assertRaisesRegex(ValueError,'explicit new union mode'):self.x.audit()
    def test_historical_terminal_may_age_but_explicit_current_proof_must_be_fresh(self):
        b=self.x.f.base;old=json.loads(b.terminal.read_bytes());old['observed_utc']='2026-01-01T00:00:00+00:00'
        b.terminal.write_bytes(C.canonical(old));self.x.f.plan['terminal_observation']['sha256']=U.sha_file(b.terminal);self.x.f.refresh_plan()
        self.assertTrue(self.x.audit()['portable_source_ledger_verified'])
        original=json.loads(self.x.current_terminal_path.read_bytes())
        for change in ({'observed_utc':'2026-01-01T00:00:00+00:00'},{'state':'Running'},
                       {'active_workers':1},{'active_workers':False},{'task_id':'different-task'}):
            with self.subTest(change=change):
                current={**original,**change}
                self.x.f.spec['current_terminal_observation']=self.x.write(self.x.current_terminal_path,current)
                self.x.f.write_spec()
                with self.assertRaisesRegex(ValueError,'current terminal'):self.x.audit()

    def test_portable_plan_cannot_omit_current_platform_observation(self):
        self.x.f.spec.pop('current_terminal_observation');self.x.f.write_spec()
        with self.assertRaisesRegex(ValueError,'current terminal'):self.x.audit()

    def test_policy_and_ledger_must_match_plan_pins(self):
        self.x.f.spec['portable_source_ledger_ref']=dict(self.x.f.spec['portable_source_ledger_ref'],sha256='0'*64);self.x.f.write_spec()
        with self.assertRaisesRegex(ValueError,'both union and preparation'):self.x.audit()
    def test_missing_duplicate_and_extra_completed_ledger_sources_refused(self):
        original=copy.deepcopy(self.x.entries)
        for mode in ('missing','duplicate','completed'):
            with self.subTest(mode=mode):
                self.x.entries=copy.deepcopy(original)
                if mode=='missing':self.x.entries=[]
                if mode=='duplicate':self.x.entries*=2
                if mode=='completed':
                    extra=copy.deepcopy(self.x.entries[0]);extra['source_id']=self.x.f.base.sources[0]['source_id'];self.x.entries.append(extra)
                self.x.refresh()
                with self.assertRaisesRegex(ValueError,'omits|duplicate portable|extra, completed'):self.x.audit()
    def test_encoded_content_and_record_binding_cannot_be_forged(self):
        original=copy.deepcopy(self.x.entries)
        for field in ('actual_encoded_file_sha256','sidecar_sha256','commit_sha256','source_record_sha256','portable_spec_sha256'):
            with self.subTest(field=field):
                self.x.entries=copy.deepcopy(original);self.x.entries[0][field]='0'*64;self.x.refresh()
                with self.assertRaises(ValueError):self.x.audit()
    def test_non_device_stat_difference_remains_an_error(self):
        e=self.x.entries[0];e['actual_stat_before']['inode']+=1;e['actual_stat_after']['inode']+=1
        e['stat_differences']=['device','inode'];self.x.refresh()
        with self.assertRaisesRegex(ValueError,'non-device'):self.x.audit()
    def test_policy_cannot_allow_extra_stat_fields_or_change_original_index(self):
        self.x.policy['allowed_difference']=['device','inode'];self.x.refresh()
        with self.assertRaisesRegex(ValueError,'only device'):self.x.audit()
    def test_original_and_successor_semantics_must_remain_equal(self):
        original=json.loads(self.x.f.new_contract.read_bytes())
        for key in ('models','runtime','decoder','alignment','quality_policy','source_files','engine_source_files'):
            with self.subTest(key=key):
                c=copy.deepcopy(original);c[key]={'unexpected_change':True};self.x.f.new_contract.write_bytes(C.canonical(c))
                self.x.f.spec['new_producer']['contract']=pin(self.x.f.new_contract);self.x.f.write_spec()
                with self.assertRaisesRegex(ValueError,'inference semantics'):self.x.audit()
    def test_completed_scope_cannot_claim_fresh_million_source_media_audit(self):
        self.x.f.plan['completed_media_scope']='all_completed_media_freshly_verified';self.x.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'media scope'):self.x.audit()
    def test_output_database_cannot_replace_portable_evidence_tree(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):
            U.validate_union(self.x.f.spec_path,self.x.f.spec_sha,'plan',output_index=self.x.folder/'bad.sqlite')


if __name__=='__main__':unittest.main()
