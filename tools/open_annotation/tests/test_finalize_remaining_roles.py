"""剩余角色元数据的独立版本化修复；真实小JSON文件，不读音频。"""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HERE),str(HERE/'tests')]
import finalize_remaining_roles as F
import prepare_speed_continuation as P
import union_coverage as U
from work_queue import encode_completion_slot
from test_portable_union_coverage import PortableUnionFixture
from test_union_coverage import pin


class RoleFixture:
    def __init__(self):
        self.x=PortableUnionFixture();f=self.x.f;b=f.base;self.root=f.root
        identity2=json.loads((b.tree/b.sources[2]['sidecar_relpath']).read_bytes())['source_identity']
        for source in b.sources:source['roles']=['train_ptn']
        for i in (0,3,5):b.sources[i]['roles']=['train_ptn','train_lm']
        b.args[2][1]=identity2
        for i in (0,1,2):
            b.args[i][6]['job_key']=P.old_job_key(b.sources[i],b.csha);b.publish(i)
        for i,shard in enumerate(b.shards):
            path=b.inputs/shard['path'];path.write_bytes(b''.join(P.canonical(s)for s in b.sources[i*2:i*2+2]));shard['sha256']=P.file_sha(path)
        full=json.loads(b.index.read_bytes());full.update(shards=b.shards,role_union_counts={'train_ptn':6,'train_lm':3})
        b.index.write_bytes(P.canonical(full));b.isha=P.file_sha(b.index)
        identity=json.loads((b.queue/'identity.json').read_bytes());identity['index_sha256']=b.isha;(b.queue/'identity.json').write_bytes(P.canonical(identity))
        chp=b.queue/'completed/s0.json';chunk=json.loads(chp.read_bytes());chunk.update(index_sha256=b.isha,shard_sha256=b.shards[0]['sha256'])
        for i in (0,1):
            p=b.tree/b.sources[i]['sidecar_relpath'];cp=p.with_name(p.name+'.commit.json')
            chunk['sources'][i].update(job_key=P.old_job_key(b.sources[i],b.csha),sidecar_sha256=P.file_sha(p),commit_sha256=P.file_sha(cp),sidecar_bytes=p.stat().st_size)
        chp.write_bytes(P.canonical(chunk));slot={k:chunk[k]for k in ('index_sha256','contract_sha256','shard_id','shard_sha256','source_count')}
        slot.update(commit_sha256=P.file_sha(chp),source_ids_sha256=P.digest(sorted(s['source_id']for s in b.sources[:2])),decision_counts={'candidate':2,'unknown':0,'failed':0})
        (b.queue/'completion_map.bin').write_bytes(encode_completion_slot(slot)+bytes(2048))
        self.x.policy['full_index']=pin(b.index);self.x.policy_ref=self.x.write(self.x.policy_path,self.x.policy)
        self.output=self.root/'actual-role-preparation'
        with patch.object(P,'EXPECTED_SOURCE_COUNT',6),patch.object(P,'EXPECTED_DOMAINS',{'fixture':6}):
            self.plan=P.prepare(b.index,b.isha,b.queue,b.contract,b.csha,b.tree,self.output,True,b.terminal,b.tsha,b.task,
                portable_spec_path=self.x.policy_path,portable_spec_sha=self.x.policy_ref['sha256'])
        self.index=self.output/'index.json';self.plan_path=self.output/'coverage-plan.json'
        self.future_counts=json.loads(self.index.read_bytes())['role_union_counts']
        # 模拟已完成v1继承full aggregate的原始缺陷；不是修改正式运行原件。
        index=json.loads(self.index.read_bytes());index['role_union_counts']=full['role_union_counts']
        for key in ('role_union_counts_scope','parent_role_union_counts','role_union_counts_parent_index'):index.pop(key,None)
        self.index.write_bytes(P.canonical(index));self.plan['remaining_index']=pin(self.index);self.plan_path.write_bytes(P.canonical(self.plan));self.sha=P.file_sha(self.plan_path)
    def close(self):self.x.close()
    def finalize(self,**kwargs):return F.finalize(self.plan_path,self.sha,**kwargs)
    def snapshot(self):return {str(p):(p.read_bytes(),p.stat().st_mtime_ns)for p in self.output.rglob('*')if p.is_file()}


class RoleFinalizeTests(unittest.TestCase):
    def setUp(self):self.f=RoleFixture();self.addCleanup(self.f.close)
    def test_future_preparer_counts_actual_overlapping_roles(self):
        self.assertEqual(self.f.future_counts,{'train_ptn':3,'train_lm':2})
    def test_versioned_metadata_keeps_v1_bytes_paths_and_exact_overlap(self):
        before=self.f.snapshot();r=self.f.finalize();self.assertEqual(r['remaining_role_union_counts'],{'train_ptn':3,'train_lm':2})
        self.assertEqual(r['retained_role_union_counts'],{'train_ptn':3,'train_lm':1})
        self.assertTrue(all(Path(p).read_bytes()==v[0] and Path(p).stat().st_mtime_ns==v[1]for p,v in before.items()))
        old=json.loads(self.f.index.read_bytes());new=U.ref(r['new_remaining_index'])
        self.assertEqual(old['shards'],new['shards']);self.assertEqual(new['coverage_plan_file'],'coverage-plan-roles-v2.json')
        self.assertEqual(new['parent_role_union_counts'],{'train_ptn':6,'train_lm':3});self.assertFalse(r['source_media_read'])
        for d in new['shards']:self.assertTrue((self.f.output/d['path']).is_file())
        with self.assertRaisesRegex(ValueError,'new and distinct'):self.f.finalize()
    def test_finalized_plan_and_index_pass_existing_portable_union(self):
        r=self.f.finalize();x=self.f.x;f=x.f;b=f.base;plan=U.ref(r['new_plan'])
        f.plan=plan;f.plan_path=Path(r['new_plan']['path']);f.new_index=Path(r['new_remaining_index']['path'])
        f.spec.update(full_index=pin(b.index),continuation_plan=r['new_plan'],portable_source_spec_ref=plan['portable_source_spec_ref'],
            portable_source_ledger_ref=plan['portable_source_ledger_ref'],portable_sidecar_inventory_ref=plan['portable_sidecar_inventory_ref'])
        f.spec['new_producer']['index']=r['new_remaining_index'];f.write_spec()
        self.assertTrue(x.audit()['exact_union_and_disjoint'])
    def test_wrong_pins_bad_names_or_existing_files_do_not_rewrite_v1(self):
        before=self.f.snapshot()
        with self.assertRaisesRegex(ValueError,'SHA'):F.finalize(self.f.plan_path,'0'*64)
        for name in ('../escaped.json','/tmp/escaped.json','index.json','coverage-plan.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):self.f.finalize(index_name=name)
        self.assertEqual(self.f.snapshot(),before)
    def test_inventory_or_ledger_tamper_refused_without_media_access(self):
        p=Path(self.f.plan['portable_source_ledger_ref']['path']);p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises((ValueError,KeyError)):self.f.finalize()
        self.assertFalse((self.f.output/'index-roles-v2.json').exists())
    def test_remaining_roles_mutation_is_rejected_by_portable_union(self):
        x=self.f.x;f=x.f;b=f.base
        f.spec.update(full_index=pin(b.index),continuation_plan=pin(self.f.plan_path),portable_source_spec_ref=self.f.plan['portable_source_spec_ref'],
            portable_source_ledger_ref=self.f.plan['portable_source_ledger_ref'],portable_sidecar_inventory_ref=self.f.plan['portable_sidecar_inventory_ref'])
        f.spec['new_producer']['index']=pin(self.f.index);f.write_spec()
        with self.assertRaisesRegex(ValueError,'role aggregate'):x.audit()
    def test_wrong_declared_source_ID_digest_is_not_preserved_into_v2(self):
        index=json.loads(self.f.index.read_bytes());index['shards'][0]['source_ids_sha256']='0'*64
        self.f.index.write_bytes(P.canonical(index));self.f.plan['remaining_index']=pin(self.f.index)
        self.f.plan_path.write_bytes(P.canonical(self.f.plan));self.f.sha=P.file_sha(self.f.plan_path)
        with self.assertRaisesRegex(ValueError,'source ID digest'):self.f.finalize()
        self.assertFalse((self.f.output/'index-roles-v2.json').exists())

    def test_progress_is_separate_from_data_and_has_counted_completion(self):
        events=[];self.f.finalize(progress=events.append)
        self.assertEqual(events[0]['stage'],'validating_saved_proofs');self.assertEqual(events[-1]['stage'],'completed')
        self.assertEqual(events[-1]['remaining_rows_checked'],3)


if __name__=='__main__':unittest.main()
