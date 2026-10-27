"""旧结果保留和剩余输入的真实文件系统反例；不实例化旧队列或模型。"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import prepare_speed_continuation as M
from test_quality_schema import fixture
import schema


class SpeedContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='redim-speed-cont-');self.root=Path(self.tmp.name).resolve()
        self.inputs=self.root/'inputs';self.inputs.mkdir();self.queue=self.root/'queue';self.queue.mkdir()
        self.tree=self.root/'labels'/'old-v1';self.tree.mkdir(parents=True)
        self.contract=self.inputs/'old-contract.json';self.contract.write_bytes(M.canonical({'schema':'redimnet2-production-contract/v2','purpose':'silver_annotation','version':'old-v1','model_API_calls_allowed':0,'release_training_labels':False}));self.csha=M.byte_sha(self.contract.read_bytes())
        (self.tree/'identity.json').write_bytes(M.canonical({'schema':'redimnet2-sidecar-tree/v1','version':'old-v1','contract_sha256':self.csha}))
        self.sources=[];self.args=[];self.shards=[]
        for i in range(6):
            a=list(copy.deepcopy(fixture()));s=a[0];s['relative_path']='speaker/%d.wav'%i;s['sidecar_relpath']='fixture/'+s['relative_path'];s['sidecar_relpath']+='.annotation.json';s['source_id']='src_'+hashlib.sha256(('fixture\0'+s['relative_path']).encode()).hexdigest()
            media=self.root/('source%d.wav'%i);media.write_bytes(b'fixture audio identity only');s['audio_path']=str(media)
            a[1].update(bytes=media.stat().st_size,stat_before=M.stat_identity(media),stat_after=M.stat_identity(media));a[2].update(sha256=self.csha,version='old-v1');a[6]['job_key']=M.old_job_key(s,self.csha)
            self.sources.append(s);self.args.append(a)
        for i in range(3):
            name='s%d'%i;p=self.inputs/(name+'.jsonl');p.write_bytes(b''.join(M.canonical(s)for s in self.sources[i*2:i*2+2]));self.shards.append({'id':name,'path':p.name,'count':2,'sha256':M.byte_sha(p.read_bytes())})
        self.index=self.inputs/'index.json';self.index.write_bytes(M.canonical({'schema':'redimnet2-shard-index/v1','source_count':6,'shards':self.shards,'domains':{'fixture':6},'RIR_or_MUSAN_or_Child_or_evaluation_in_this_job':False}));self.isha=M.byte_sha(self.index.read_bytes())
        (self.queue/'identity.json').write_bytes(M.canonical({'schema':'redimnet2-work-queue/v1','index_sha256':self.isha,'contract_sha256':self.csha,'source_count':6,'shard_count':3}))
        (self.queue/'broker.json').write_bytes(M.canonical({'next':2,'leases':{'s1':{'worker_id':'0'}},'completed_chunks':1}))
        slot={'shard_id':'s0','shard_sha256':self.shards[0]['sha256'],'source_count':2,'index_sha256':self.isha,'contract_sha256':self.csha,'source_ids_sha256':M.digest(sorted(s['source_id']for s in self.sources[:2])),'commit_sha256':'f'*64,'decision_counts':{'candidate':2,'unknown':0,'failed':0}}
        payload=M.canonical(slot);data=b'RDMAP001'+struct.pack('>I',len(payload))+hashlib.sha256(payload).digest()+payload+bytes(980-len(payload));(self.queue/'completion_map.bin').write_bytes(data+bytes(2048));(self.queue/'.parent.lock').touch()
        self.task='t-fixture-annotation';self.terminal=self.root/'terminal.json';self.write_terminal()
        self.publish(2)

    def tearDown(self):self.tmp.cleanup()

    def write_terminal(self,**changes):
        v={'schema':'annotation-writer-terminal-observation/v1','task_id':self.task,'queue_root':str(self.queue),'state':'Stopped','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':[{'name':self.task+'-worker-0','state':'Stopped'}]};v.update(changes);self.terminal.write_bytes(M.canonical(v));self.tsha=M.byte_sha(self.terminal.read_bytes())

    def publish(self,i):
        r=schema.build_record(*self.args[i]);p=self.tree/self.sources[i]['sidecar_relpath'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(M.canonical(r));marker={'schema':'redimnet2-sidecar-commit/v1','source_id':self.sources[i]['source_id'],'job_key':M.old_job_key(self.sources[i],self.csha),'contract_sha256':self.csha,'source_record_sha256':M.digest(self.sources[i]),'sidecar_sha256':M.byte_sha(p.read_bytes()),'bytes':p.stat().st_size};c=p.with_name(p.name+'.commit.json');c.write_bytes(M.canonical(marker));return p,c

    def run_prepare(self,final=False,name='out'):
        with patch.object(M,'EXPECTED_SOURCE_COUNT',6),patch.object(M,'EXPECTED_DOMAINS',{'fixture':6}):
            return M.prepare(self.index,self.isha,self.queue,self.contract,self.csha,self.tree,self.root/name,final,self.terminal,self.tsha,self.task)

    def snapshot(self):
        return {str(p.relative_to(self.root)):(p.read_bytes(),p.stat().st_mtime_ns)for home in [self.inputs,self.queue,self.tree]for p in home.rglob('*')if p.is_file()}

    def test_provisional_preserves_partial_results_without_relabel_or_copy(self):
        before=self.snapshot();r=self.run_prepare();self.assertEqual(self.snapshot(),before)
        self.assertEqual((r['retained_source_count'],r['remaining_source_count']),(3,3));self.assertEqual(r['status'],'provisional_not_executable');self.assertFalse(r['formal_runtime_admitted']);self.assertEqual(r['new_inference_count'],0)
        self.assertFalse((self.root/'out'/'index.json').exists());self.assertFalse((self.root/'out'/'shards').exists())
        self.assertEqual(r['retained_partial_sources'][0]['producer_contract_sha256'],self.csha)
        self.assertEqual([a['descriptor']['count']for a in r['remaining_shards']],[1,2])
        self.assertTrue(r['retained_complete_shards'][0]['chunk_receipt_sha256'])

    def test_final_copies_only_missing_sources_and_retains_old_bytes(self):
        before=self.snapshot();r=self.run_prepare(True);self.assertEqual(self.snapshot(),before)
        ix=json.loads((self.root/'out'/'index.json').read_text());rows=[]
        for s in ix['shards']:
            p=self.root/'out'/s['path'];self.assertFalse(p.is_symlink());self.assertEqual(M.byte_sha(p.read_bytes()),s['sha256']);rows.extend(json.loads(a)for a in p.read_text().splitlines())
        self.assertEqual({s['source_id']for s in rows},{s['source_id']for s in self.sources[3:]});self.assertEqual(ix['source_count'],3);self.assertEqual(ix['domains'],{'fixture':3});self.assertEqual(r['status'],'final_prepared_not_started');self.assertFalse(r['formal_runtime_admitted'])
        with self.assertRaisesRegex(ValueError,'fresh'):self.run_prepare(True)

    def test_final_preserves_preseed_result_in_unclaimed_shard(self):
        self.publish(4);before=self.snapshot();r=self.run_prepare(True)
        self.assertEqual(self.snapshot(),before)
        self.assertTrue(r['unclaimed_sidecars_scanned'])
        self.assertEqual((r['retained_source_count'],r['remaining_source_count']),(4,2))
        ledger=Path(r['retained_unclaimed_sources_ref']['path']);lines=[json.loads(v)for v in ledger.read_bytes().splitlines()]
        self.assertEqual([v['source_id']for v in lines],[self.sources[4]['source_id']])
        self.assertEqual(r['retained_unclaimed_sources_ref']['sha256'],M.file_sha(ledger))
        index=json.loads((self.root/'out/index.json').read_bytes());remaining=[]
        for shard in index['shards']:remaining.extend(json.loads(v)for v in (self.root/'out'/shard['path']).read_bytes().splitlines())
        self.assertEqual({s['source_id']for s in remaining},{self.sources[3]['source_id'],self.sources[5]['source_id']})

    def test_unclaimed_orphan_is_preserved_and_final_refused(self):
        p,c=self.publish(4);c.unlink();before=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'orphan'):self.run_prepare(True)
        self.assertEqual(p.read_bytes(),before);self.assertFalse(c.exists());self.assertFalse((self.root/'out/index.json').exists())

    def test_provisional_explicitly_does_not_claim_full_unclaimed_scan(self):
        self.publish(4);r=self.run_prepare(False)
        self.assertFalse(r['unclaimed_sidecars_scanned'])
        self.assertEqual((r['retained_source_count'],r['remaining_source_count']),(3,3))

    def test_active_parent_cannot_prepare_final_but_provisional_does_not_lock(self):
        with (self.queue/'.parent.lock').open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'still owned'):self.run_prepare(True)
            self.assertEqual(self.run_prepare(False)['status'],'provisional_not_executable')
        self.assertFalse((self.root/'out'/'index.json').exists())

    def test_missing_parent_lock_never_created(self):
        (self.queue/'.parent.lock').unlink()
        with self.assertRaises(FileNotFoundError):self.run_prepare(True)
        self.assertFalse((self.queue/'.parent.lock').exists())

    def test_orphan_sidecar_is_preserved_and_not_reprocessed(self):
        p,c=self.publish(2);c.unlink();raw=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'orphan sidecar'):self.run_prepare(True)
        self.assertEqual(p.read_bytes(),raw);self.assertFalse(c.exists());self.assertFalse((self.root/'out').exists())

    def test_partial_commit_digest_corruption_is_not_accepted(self):
        _,c=self.publish(2);v=json.loads(c.read_text());v['sidecar_sha256']='0'*64;c.write_bytes(M.canonical(v))
        with self.assertRaisesRegex(ValueError,'commit identity'):self.run_prepare()

    def test_changed_old_contract_in_sidecar_is_not_relabelled(self):
        p,c=self.publish(2);v=json.loads(p.read_text());v['contract']['sha256']='a'*64;p.write_bytes(M.canonical(v));marker=json.loads(c.read_text());marker.update(sidecar_sha256=M.byte_sha(p.read_bytes()),bytes=p.stat().st_size);c.write_bytes(M.canonical(marker));before=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'old contract'):self.run_prepare()
        self.assertEqual(before,p.read_bytes())

    def test_duplicate_sources_in_partial_shard_refused(self):
        s=self.shards[1];p=self.inputs/s['path'];p.write_bytes(M.canonical(self.sources[2])*2);s['sha256']=M.byte_sha(p.read_bytes());ix=json.loads(self.index.read_text());ix['shards']=self.shards;self.index.write_bytes(M.canonical(ix));self.isha=M.byte_sha(self.index.read_bytes());identity=json.loads((self.queue/'identity.json').read_text());identity['index_sha256']=self.isha;(self.queue/'identity.json').write_bytes(M.canonical(identity));mapping=bytearray((self.queue/'completion_map.bin').read_bytes());slot=json.loads(mapping[44:44+struct.unpack('>I',mapping[8:12])[0]]);slot['index_sha256']=self.isha;raw=M.canonical(slot);mapping[:1024]=b'RDMAP001'+struct.pack('>I',len(raw))+hashlib.sha256(raw).digest()+raw+bytes(980-len(raw));(self.queue/'completion_map.bin').write_bytes(mapping)
        with self.assertRaisesRegex(ValueError,'uniqueness'):self.run_prepare()

    def test_stale_future_and_active_terminal_observations_refused(self):
        for changes in [{'state':'Running'},{'observed_utc':(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(hours=2)).isoformat()},{'observed_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat()},{'workers':[{'name':self.task+'-worker-0','state':'Running'}]}]:
            with self.subTest(changes=changes):
                self.write_terminal(**changes)
                with self.assertRaises(ValueError):self.run_prepare(True)

    def test_terminal_hash_and_task_are_bound(self):
        self.tsha='f'*64
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.run_prepare(True)
        self.write_terminal(task_id='other')
        with self.assertRaisesRegex(ValueError,'identity differs'):self.run_prepare(True)

    def test_broker_change_after_partition_refuses_final(self):
        original=M.partition
        def changing(*args):
            r=original(*args);p=self.queue/'broker.json';p.write_bytes(p.read_bytes()+b' ');return r
        with patch.object(M,'partition',side_effect=changing):
            with self.assertRaisesRegex(ValueError,'broker changed'):self.run_prepare(True)
        self.assertFalse((self.root/'out'/'index.json').exists())

    def test_index_replaced_after_progress_snapshot_is_refused(self):
        original=M.inspect_progress
        def replacing(*args):
            r=original(*args);p=self.index;p.write_bytes(p.read_bytes()+b' ');return r
        with patch.object(M,'inspect_progress',side_effect=replacing):
            with self.assertRaisesRegex(ValueError,'index changed after progress'):
                self.run_prepare()
        self.assertFalse((self.root/'out').exists())

    def test_retained_slot_count_replaced_after_progress_is_refused(self):
        original=M.inspect_progress
        def replacing(*args):
            r=original(*args);p=self.queue/'completion_map.bin';data=bytearray(p.read_bytes());size=struct.unpack('>I',data[8:12])[0];v=json.loads(data[44:44+size]);v['source_count']=3;raw=M.canonical(v);data[:1024]=b'RDMAP001'+struct.pack('>I',len(raw))+hashlib.sha256(raw).digest()+raw+bytes(980-len(raw));p.write_bytes(data);return r
        with patch.object(M,'inspect_progress',side_effect=replacing):
            with self.assertRaisesRegex(ValueError,'completion identity changed'):
                self.run_prepare()
        self.assertFalse((self.root/'out').exists())

    def test_fixed_formal_denominator_has_no_cli_fixture_override(self):
        with self.assertRaisesRegex(ValueError,'11521178'):
            M.prepare(self.index,self.isha,self.queue,self.contract,self.csha,self.tree,self.root/'out')

    def test_output_in_old_tree_and_symlink_inputs_refused(self):
        with patch.object(M,'EXPECTED_SOURCE_COUNT',6),patch.object(M,'EXPECTED_DOMAINS',{'fixture':6}):
            with self.assertRaisesRegex(ValueError,'overlaps'):
                M.prepare(self.index,self.isha,self.queue,self.contract,self.csha,self.tree,self.queue/'out')
            alias=self.root/'alias';alias.symlink_to(self.tree,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):
                M.prepare(self.index,self.isha,self.queue,self.contract,self.csha,alias,self.root/'out')


if __name__=='__main__':unittest.main()
