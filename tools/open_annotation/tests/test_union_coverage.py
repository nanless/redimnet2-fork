"""六个合成来源走旧发布、剩余准备、新发布和联合消费全流程。"""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import prepare_speed_continuation as P
import union_coverage as U
import union_consumer as C
import test_prepare_speed_continuation as fixtures
from work_queue import encode_completion_slot,WorkQueue
from sidecar import SidecarStore
import schema


def pin(path):return {'path':str(path),'sha256':U.sha_file(path)}


class UnionFixture:
    """其他集成测试可复用此工厂；原声从不解码，模型从不导入。"""
    def __init__(self,preseed_indices=()):
        self.base=fixtures.SpeedContinuationTests(methodName='runTest');self.base.setUp();b=self.base
        self.root=b.root
        receipts=[]
        for i in (0,1):
            path,commit=b.publish(i);record=json.loads(path.read_bytes())
            receipts.append({'source_id':b.sources[i]['source_id'],'job_key':P.old_job_key(b.sources[i],b.csha),
                'sidecar_relpath':b.sources[i]['sidecar_relpath'],'sidecar_sha256':U.sha_file(path),
                'sidecar_bytes':path.stat().st_size,'commit_sha256':U.sha_file(commit),'decision':record['decision']['state']})
        (b.queue/'completed').mkdir()
        ch={'schema':'redimnet2-chunk-commit/v1','index_sha256':b.isha,'contract_sha256':b.csha,
            'shard_id':'s0','shard_sha256':b.shards[0]['sha256'],'source_count':2,
            'attempt_id':'attempt-fixture','sources':receipts,'committed_utc':'2026-10-27T00:00:00Z'}
        chp=b.queue/'completed/s0.json';chp.write_bytes(P.canonical(ch))
        slot={k:ch[k] for k in ('index_sha256','contract_sha256','shard_id','shard_sha256','source_count')}
        slot.update(commit_sha256=U.sha_file(chp),source_ids_sha256=P.digest(sorted(s['source_id']for s in b.sources[:2])),decision_counts={'candidate':2,'unknown':0,'failed':0})
        (b.queue/'completion_map.bin').write_bytes(encode_completion_slot(slot)+bytes(2048))
        for i in preseed_indices:b.publish(i)
        self.plan=b.run_prepare(True,name='remaining')
        self.plan_path=b.root/'remaining/coverage-plan.json';self.new_index=b.root/'remaining/index.json'
        self.new_contract=b.root/'new-inputs/new-contract.json';self.new_contract.parent.mkdir()
        c=json.loads(b.contract.read_bytes());c['version']='new-v1';self.new_contract.write_bytes(P.canonical(c));self.nsha=U.sha_file(self.new_contract)
        self.new_tree=b.root/'new-labels/new-v1';self.new_queue=b.root/'new-queue'
        self.spec={'schema':U.SCHEMA,'purpose':'cpu_fixture','expected_full_source_count':6,
            'full_index':pin(b.index),'continuation_plan':pin(self.plan_path),
            'old_producer':{'id':'old','contract':pin(b.contract),'queue_root':str(b.queue),'sidecar_tree':str(b.tree)},
            'new_producer':{'id':'new','contract':pin(self.new_contract),'index':pin(self.new_index),
                            'queue_root':str(self.new_queue),'sidecar_tree':str(self.new_tree)},
            'selection_policy':'old_retained_else_new_remaining','release_training_labels':False,'model_API_calls_allowed':0}
        self.spec_path=b.root/'union-spec.json';self.write_spec()

    def write_spec(self):self.spec_path.write_bytes(P.canonical(self.spec));self.spec_sha=U.sha_file(self.spec_path)
    def refresh_plan(self):self.plan_path.write_bytes(P.canonical(self.plan));self.spec['continuation_plan']=pin(self.plan_path);self.write_spec()
    def close(self):self.base.tearDown()
    def old_snapshot(self):return self.base.snapshot()
    def complete(self):
        b=self.base;c=json.loads(self.new_contract.read_bytes())
        q=WorkQueue(self.new_index,U.sha_file(self.new_index),self.new_queue,self.nsha)
        store=SidecarStore(self.new_tree.parent,c,self.nsha)
        lookup={s['source_id']:i for i,s in enumerate(b.sources)}
        while True:
            claim=q.claim_next('new-worker')
            if claim is None:break
            with claim:
                receipts=[]
                for source in claim.sources():
                    args=copy.deepcopy(b.args[lookup[source['source_id']]])
                    args[2].update(sha256=self.nsha,version='new-v1');args[6].update(job_key=P.old_job_key(source,self.nsha),attempt_id=claim.attempt_id)
                    record=schema.build_record(*args)
                    receipts.append(store.publish(source,record,claim))
                claim.finish(receipts)


class UnionCoverageTests(unittest.TestCase):
    def setUp(self):self.f=UnionFixture()
    def tearDown(self):self.f.close()
    def check(self,mode='plan',database=False):
        return U.validate_union(self.f.spec_path,self.f.spec_sha,mode,output_index=self.f.root/'consumer.sqlite' if database else None)

    def _rewrite_result(self,change):
        source=self.f.base.sources[2];path=self.f.base.tree/source['sidecar_relpath']
        record=json.loads(path.read_bytes());change(record);path.write_bytes(P.canonical(record))
        commit_path=path.with_name(path.name+'.commit.json');commit=json.loads(commit_path.read_bytes())
        commit.update(sidecar_sha256=U.sha_file(path),bytes=path.stat().st_size)
        commit_path.write_bytes(P.canonical(commit));return source,record

    def test_result_contract_version_cannot_claim_another_producer(self):
        source,_=self._rewrite_result(lambda r:r['contract'].update(version='foreign-version'))
        contract=json.loads(self.f.base.contract.read_bytes())
        with self.assertRaisesRegex(ValueError,'source/producer'):
            U.read_result(source,self.f.spec['old_producer'],contract=contract)

    def test_result_runtime_and_model_provenance_bound_to_pinned_contract(self):
        source=self.f.base.sources[2];path=self.f.base.tree/source['sidecar_relpath']
        original=json.loads(path.read_bytes());contract=json.loads(self.f.base.contract.read_bytes())
        contract.update(models={k:{'path':v['path'],'files':v['files']}for k,v in original['provenance']['models'].items()})
        # 仅使用合成合同，不读模型/原声。先证明完整producer元数据可被重开。
        contract['runtime']={'normalization_profile':schema.POLICY}
        source,_=self._rewrite_result(lambda r:r['provenance'].update(runtime=copy.deepcopy(contract['runtime'])))
        U.read_result(source,self.f.spec['old_producer'],contract=contract)
        for field in ('runtime','models'):
            def drift(r):
                if field=='runtime':r['provenance']['runtime']['unexpected_drift']=True
                else:r['provenance']['models']['asr']['path']='/synthetic/foreign-asr'
            source,_=self._rewrite_result(drift)
            with self.assertRaisesRegex(ValueError,'provenance differs'):
                U.read_result(source,self.f.spec['old_producer'],contract=contract)
            self._rewrite_result(lambda r:r['provenance'].update(models=copy.deepcopy(original['provenance']['models']),runtime=copy.deepcopy(contract['runtime'])))

    def test_production_cannot_reuse_minimal_cpu_producer_metadata(self):
        self.f.spec['purpose']='production';self.f.write_spec()
        # 只缩小拒绝测试的分母，不创建成功正式报告；合成最小合同仍然被拒。
        with patch.object(U,'PRODUCTION_COUNT',6),patch.object(U,'PRODUCTION_DOMAINS',{'fixture':6}):
            with self.assertRaisesRegex(ValueError,'complete models/runtime'):self.check()

    def test_plan_exact_partition_does_not_write_old_or_require_new_inference(self):
        before=self.f.old_snapshot();r=self.check();self.assertEqual(before,self.f.old_snapshot())
        self.assertEqual((r['source_count'],r['retained_source_count'],r['remaining_source_count']),(6,3,3))
        self.assertFalse(r['complete_sidecars_reopened']);self.assertFalse(r['formal_runtime_admitted'])
        self.assertFalse(self.f.new_queue.exists())

    def test_complete_reads_old_new_and_consumer_returns_original_producers(self):
        before=self.f.old_snapshot();self.f.complete();r=self.check('complete',True)
        with C.UnionConsumer(self.f.spec_path,self.f.spec_sha,r['consumer_index']['path'],r['consumer_index']['sha256'])as consumer:
            rows=list(consumer.iter_selected_sources())
            self.assertEqual([a['producer_id'] for a in rows],['old']*3+['new']*3)
            self.assertEqual([a['source_id'] for a in rows],[a['source_id'] for a in self.f.base.sources])
            self.assertTrue(all(not a['record']['decision']['usable_for_training'] for a in rows))
            with self.assertRaisesRegex(ValueError,'absent'):consumer.resolve_source('not-present')
        self.assertEqual(before,self.f.old_snapshot())
        self.assertEqual(r['decision_counts'],{'candidate':6,'unknown':0,'failed':0})

    def test_provisional_and_missing_terminal_refused(self):
        for value in ('provisional_not_executable','all_sources_retained_no_successor_needed'):
            self.f.plan['status']=value;self.f.refresh_plan()
            with self.assertRaisesRegex(ValueError,'final'):self.check()

    def test_active_old_lock_refuses_even_a_fresh_terminal_file(self):
        import fcntl
        with (self.f.base.queue/'.parent.lock').open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'still owned'):self.check()

    def test_stale_observation_refuses_plan_but_not_completed_historical_audit(self):
        b=self.f.base;self.f.complete()
        terminal=json.loads(b.terminal.read_bytes());terminal['observed_utc']='2026-01-01T00:00:00+00:00';b.terminal.write_bytes(P.canonical(terminal))
        self.f.plan['terminal_observation']['sha256']=U.sha_file(b.terminal);self.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'stale'):self.check()
        self.assertTrue(self.check('complete')['exact_union_and_disjoint'])

    def test_old_broker_advance_cannot_use_old_final_plan(self):
        p=self.f.base.queue/'broker.json';value=json.loads(p.read_bytes());value['next']=3;p.write_bytes(P.canonical(value))
        with self.assertRaisesRegex(ValueError,'broker changed'):self.check()

    def test_modified_old_sidecar_never_repaired_or_silently_rerun(self):
        p=self.f.base.tree/self.f.base.sources[0]['sidecar_relpath'];p.write_bytes(p.read_bytes()+b' ')
        after=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'SHA'):self.check()
        self.assertEqual(after,p.read_bytes());self.assertFalse(self.f.new_queue.exists())

    def test_remaining_source_record_changed_with_same_ID_refused(self):
        index=json.loads(self.f.new_index.read_bytes());p=self.f.new_index.parent/index['shards'][0]['path'];rows=[json.loads(x)for x in p.read_bytes().splitlines()]
        rows[0]['audio_path']='/changed/same-id.wav';p.write_bytes(b''.join(P.canonical(x)for x in rows));index['shards'][0]['sha256']=U.sha_file(p)
        self.f.new_index.write_bytes(P.canonical(index));self.f.plan['remaining_index']=pin(self.f.new_index)
        self.f.spec['new_producer']['index']=pin(self.f.new_index);self.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'absent/changed'):self.check()

    def test_retained_and_remaining_overlap_refused(self):
        index=json.loads(self.f.new_index.read_bytes());p=self.f.new_index.parent/index['shards'][0]['path'];p.write_bytes(P.canonical(self.f.base.sources[2]));index['shards'][0]['sha256']=U.sha_file(p)
        self.f.new_index.write_bytes(P.canonical(index));self.f.plan['remaining_index']=pin(self.f.new_index);self.f.spec['new_producer']['index']=pin(self.f.new_index);self.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'overlap'):self.check()

    def test_missing_remaining_source_is_not_only_a_count_check(self):
        index=json.loads(self.f.new_index.read_bytes());index['shards']=index['shards'][:1];index['source_count']=1;index['domains']={'fixture':1}
        self.f.new_index.write_bytes(P.canonical(index));self.f.plan['remaining_index']=pin(self.f.new_index);self.f.spec['new_producer']['index']=pin(self.f.new_index);self.f.plan['remaining_source_count']=1;self.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'missing'):self.check()

    def test_producer_relabel_and_fixture_cannot_admit_formal_full(self):
        self.f.spec['new_producer']['id']='old';self.f.write_spec()
        with self.assertRaisesRegex(ValueError,'overlap'):self.check()
        self.f.spec['new_producer']['id']='new';self.f.spec['purpose']='production';self.f.write_spec()
        with self.assertRaisesRegex(ValueError,'formal full'):self.check()

    def test_incomplete_new_producer_refused_by_complete(self):
        with self.assertRaises((ValueError,FileNotFoundError)):self.check('complete')

    def test_plan_consumer_and_changed_database_refused(self):
        r=self.check(database=True)
        with self.assertRaisesRegex(ValueError,'complete'):C.UnionConsumer(self.f.spec_path,self.f.spec_sha,r['consumer_index']['path'],r['consumer_index']['sha256'])
        db=Path(r['consumer_index']['path']);db.write_bytes(db.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):C.UnionConsumer(self.f.spec_path,self.f.spec_sha,db,r['consumer_index']['sha256'])

    def test_consumer_rechecks_sidecar_and_does_not_follow_mtime(self):
        self.f.complete();r=self.check('complete',True)
        with C.UnionConsumer(self.f.spec_path,self.f.spec_sha,r['consumer_index']['path'],r['consumer_index']['sha256'])as c:
            sid=self.f.base.sources[0]['source_id'];p=self.f.base.tree/self.f.base.sources[0]['sidecar_relpath']
            p.write_bytes(p.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'SHA'):c.resolve_source(sid)

    def test_path_escape_and_output_in_old_tree_refused(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):
            U.validate_union(self.f.spec_path,self.f.spec_sha,'plan',output_index=self.f.base.queue/'consumer.sqlite')
        self.assertFalse((self.f.base.queue/'consumer.sqlite').exists())
        self.f.spec['old_producer']['sidecar_tree']=str(self.f.base.tree/'..'/'old-v1');self.f.write_spec()
        with self.assertRaisesRegex(ValueError,'traversal'):self.check()

    def test_omitting_completed_chunk_cannot_reclassify_it_as_new(self):
        self.f.plan['retained_complete_shards']=[];self.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'omits/adds completed'):self.check()

    def test_omitting_partial_published_result_cannot_recompute_it(self):
        f=self.f;index=json.loads(f.new_index.read_bytes());shard=index['shards'][0]
        path=f.new_index.parent/shard['path'];rows=f.base.sources[2:4]
        path.write_bytes(b''.join(P.canonical(r)for r in rows));shard.update(count=2,sha256=U.sha_file(path),source_ids_sha256=P.digest(sorted(r['source_id']for r in rows)))
        index.update(source_count=4,domains={'fixture':4});f.new_index.write_bytes(P.canonical(index))
        f.plan.update(retained_partial_sources=[],retained_source_count=2,remaining_source_count=4,remaining_index=pin(f.new_index))
        f.spec['new_producer']['index']=pin(f.new_index);f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'already published'):self.check()

    def test_old_result_in_unclaimed_shard_is_not_silently_recomputed(self):
        self.f.base.publish(4)
        with self.assertRaisesRegex(ValueError,'already published'):self.check()

    def test_preseed_full_prepare_gate_and_consumer_preserve_four_old_results(self):
        self.f.close();self.f=UnionFixture(preseed_indices=(4,));before=self.f.old_snapshot();self.f.complete()
        import cloud_gate
        r=cloud_gate.check_union_data(self.f.spec_path,self.f.spec_sha,'complete',output_index=self.f.root/'consumer.sqlite')
        index=r['union_validation']['consumer_index']
        with C.UnionConsumer(self.f.spec_path,self.f.spec_sha,index['path'],index['sha256'])as c:
            selected=list(c.iter_selected_sources())
        self.assertEqual([v['producer_id']for v in selected],['old','old','old','new','old','new'])
        self.assertEqual(r['retained_sources_not_scheduled_again'],4);self.assertEqual(r['remaining_source_count'],2)
        self.assertEqual(before,self.f.old_snapshot());self.assertFalse(r['cloud_submission_admitted'])

    def test_preseed_ledger_tamper_and_duplicate_source_refused(self):
        self.f.close();self.f=UnionFixture(preseed_indices=(4,))
        p=Path(self.f.plan['retained_unclaimed_sources_ref']['path']);raw=p.read_bytes();p.write_bytes(raw+b' ')
        with self.assertRaisesRegex(ValueError,'blank|SHA'):self.check()
        p.write_bytes(raw+raw);self.f.plan['retained_unclaimed_sources_ref'].update(sha256=U.sha_file(p),count=2)
        self.f.plan['retained_unclaimed_source_count']=2;self.f.refresh_plan()
        with self.assertRaisesRegex(ValueError,'duplicate unclaimed'):self.check()

    def test_existing_new_namespace_cannot_be_ignored_by_plan(self):
        self.f.new_tree.mkdir(parents=True)
        identity={'schema':'redimnet2-sidecar-tree/v1','version':'wrong-version','contract_sha256':self.f.nsha}
        (self.f.new_tree/'identity.json').write_bytes(P.canonical(identity))
        with self.assertRaisesRegex(ValueError,'namespace identity'):self.check()

    def test_existing_new_queue_must_bind_the_exact_remaining_index(self):
        self.f.new_queue.mkdir();(self.f.new_queue/'identity.json').write_bytes(P.canonical({
            'schema':'redimnet2-work-queue/v1','index_sha256':'f'*64,'contract_sha256':self.f.nsha,
            'source_count':3,'shard_count':2}))
        with self.assertRaisesRegex(ValueError,'queue identity'):self.check()

    def test_dangling_new_queue_identity_symlink_is_not_treated_as_absent(self):
        self.f.new_queue.mkdir();(self.f.new_queue/'identity.json').symlink_to(self.f.root/'missing-identity.json')
        with self.assertRaisesRegex(ValueError,'symlink'):self.check()


if __name__=='__main__':unittest.main()
