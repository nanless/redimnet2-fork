"""只用已有合成旧任务原件验证有界交接；不连服务器、不调用模型。"""
import copy
import fcntl
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import bounded_speed_handoff as M
import test_prepare_speed_continuation as Fixtures


def enable_portable(case):
    f=case.f
    for args,source in zip(f.args,f.sources):
        media=Path(source['audio_path']);old=dict(M.P.stat_identity(media));old['device']+=100
        args[1].update(file_sha256=M.P.byte_sha(media.read_bytes()),stat_before=old,stat_after=dict(old))
    f.publish(2)
    target=f.root/'portable-target';target.mkdir();alias=f.root/'portable-alias';alias.symlink_to(target)
    value=dict(schema='annotation-portable-source-spec/v1',purpose='cpu_fixture',source_identity_mode='portable_encoded_sha256/v1',
        allowed_difference=['device'],encoded_hash_algorithm='sha256',old_contract=case.value['old_contract'],full_index=case.value['old_index'],
        old_queue_root=str(f.queue),old_sidecar_tree=str(f.tree),source_alias=dict(alias_path=str(alias),target_path=str(target)),model_API_calls_allowed=0,release_training_labels=False)
    p=f.root/'portable-spec.json';p.write_bytes(M.P.canonical(value));case.value['portable_spec']=dict(path=str(p),sha256=M.P.byte_sha(p.read_bytes()));case.save()


class BoundedHandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture=Fixtures.SpeedContinuationTests();self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        self.f=self.fixture
        (self.f.queue/'claims').mkdir();(self.f.queue/'claims/s1.lock').touch()
        self.spec=self.f.root/'bounded-spec.json';self.output=self.f.root/'bounded-output'
        self.value=dict(schema=M.SCHEMA,purpose='cpu_fixture',old_index=dict(path=str(self.f.index),sha256=self.f.isha),
                        old_contract=dict(path=str(self.f.contract),sha256=self.f.csha),old_task_id=self.f.task,
                        old_queue_root=str(self.f.queue),old_sidecar_tree=str(self.f.tree),
                        new_queue_root=str(self.f.root/'new-queue'),new_sidecar_root=str(self.f.root/'new-labels'),
                        global_source_count=6,previous_retained_count=0)
        self.save()

    def save(self):
        self.spec.write_bytes(M.P.canonical(self.value));self.pin=M.P.byte_sha(self.spec.read_bytes())

    def run_plan(self, final=True):
        return M.prepare(self.spec,self.pin,self.output,execute_final=final,
                         terminal_pin=dict(path=str(self.f.terminal),sha256=self.f.tsha))

    def test_preview_is_metadata_only_even_while_parent_owned(self):
        with (self.f.queue/'.parent.lock').open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(M.P,'read_shard',side_effect=AssertionError('must not open source JSONL')):
                r=self.run_plan(False)
        self.assertEqual(r['partial_sources_read'],0);self.assertFalse(r['formal_runtime_admitted'])
        self.assertFalse((self.output/'index.json').exists())

    def test_final_only_opens_one_partial_and_preserves_old_producer(self):
        before=self.f.snapshot();calls=[];real=M.P.read_shard
        def observe(root,shard):calls.append(shard['id']);return real(root,shard)
        with patch.object(M.P,'read_shard',side_effect=observe):r=self.run_plan()
        self.assertEqual(calls,['s1']);self.assertEqual(self.f.snapshot(),before)
        self.assertEqual((r['retained_complete_source_count'],r['retained_partial_source_count'],r['exact_remaining_source_count']),(2,1,3))
        self.assertEqual(r['partial_sources_read'],2);self.assertEqual(r['unclaimed_JSONL_read_count'],0)
        self.assertFalse(r['database_created']);self.assertFalse(r['formal_runtime_admitted'])
        reader=M.BoundedShardReader(Path(r['new_index']['path']),r['new_index']['sha256'])
        rows=[]
        for shard in reader.index['shards']:rows.extend(reader.read(shard['id']))
        self.assertEqual(rows,self.f.sources[3:])
        ref=json.loads((self.output/'retained-partial.jsonl').read_text())
        self.assertEqual(ref['producer_contract_sha256'],self.f.csha)
        self.assertEqual((self.output/'retained-completed-slots.bin').stat().st_size,1024)

    def test_unclaimed_preseed_is_checked_only_when_claimed(self):
        self.f.publish(4);r=self.run_plan()
        self.assertEqual(r['retained_partial_source_count'],1)
        reader=M.BoundedShardReader(Path(r['new_index']['path']),r['new_index']['sha256'])
        source=reader.read('s2')[0]
        ref=reader.lookup_previous_for_claim('s2',source)
        self.assertEqual(ref['source_id'],self.f.sources[4]['source_id'])
        self.assertEqual(ref['producer_contract_sha256'],self.f.csha)
        self.assertTrue(reader.index['future_per_claim_previous_result_check_required'])
        self.assertIsNone(reader.lookup_previous_for_claim('s2',reader.read('s2')[1]))

    def test_unclaimed_orphan_is_preserved_and_refused_at_claim(self):
        p,c=self.f.publish(4);c.unlink();raw=p.read_bytes();r=self.run_plan()
        reader=M.BoundedShardReader(Path(r['new_index']['path']),r['new_index']['sha256']);source=reader.read('s2')[0]
        with self.assertRaisesRegex(ValueError,'orphan'):reader.lookup_previous_for_claim('s2',source)
        self.assertEqual(p.read_bytes(),raw);self.assertFalse(c.exists())

    def test_partial_orphan_refuses_plan_without_changing_any_old_file(self):
        p,c=self.f.publish(3);c.unlink();before=self.f.snapshot()
        with self.assertRaisesRegex(ValueError,'orphan'):self.run_plan()
        self.assertEqual(self.f.snapshot(),before);self.assertFalse((self.output/'report.json').exists())

    def test_active_parent_refused(self):
        with (self.f.queue/'.parent.lock').open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'still owned'):self.run_plan()
        self.assertFalse(self.output.exists())

    def test_active_claim_refused(self):
        with (self.f.queue/'claims/s1.lock').open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'still owned'):self.run_plan()
        self.assertFalse(self.output.exists())

    def test_wrong_or_stale_terminal_refused(self):
        self.f.write_terminal(state='Running')
        with self.assertRaisesRegex(ValueError,'not observed terminal'):self.run_plan()

    def test_no_terminal_refused(self):
        with self.assertRaisesRegex(ValueError,'terminal observation'):M.prepare(self.spec,self.pin,self.output,execute_final=True)

    def test_changed_broker_refused_before_publish(self):
        original=M.capture_slots;calls=[]
        def replace(*args):
            result=original(*args);calls.append(1)
            if len(calls)==1:
                b=json.loads((self.f.queue/'broker.json').read_text());b['leases']['s1']['tamper']=True
                (self.f.queue/'broker.json').write_bytes(M.P.canonical(b))
            return result
        with patch.object(M,'capture_slots',side_effect=replace):
            with self.assertRaisesRegex(ValueError,'broker changed'):self.run_plan()
        self.assertFalse(self.output.exists())

    def test_no_old_queue_reuse_with_new_contract(self):
        self.value['new_queue_root']=str(self.f.queue);self.save()
        with self.assertRaisesRegex(ValueError,'old queue'):self.run_plan()

    def test_old_input_output_overlap_refused(self):
        self.output=self.f.inputs/'bad'
        with self.assertRaisesRegex(ValueError,'old input'):self.run_plan()

    def test_completed_slot_checksum_checked(self):
        p=self.f.queue/'completion_map.bin';raw=bytearray(p.read_bytes());raw[80]^=1;p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'checksum'):self.run_plan()

    def test_unclaimed_descriptor_tamper_refused(self):
        r=self.run_plan();p=Path(r['new_index']['path']);v=json.loads(p.read_text());v['shards'][-1]['sha256']='0'*64;p.write_bytes(M.P.canonical(v))
        with self.assertRaisesRegex(ValueError,'descriptor changed'):M.BoundedShardReader(p,M.P.byte_sha(p.read_bytes()))

    def test_partial_foreign_source_refused_even_if_local_hashes_updated(self):
        r=self.run_plan();p=Path(r['new_index']['path']);v=json.loads(p.read_text());s=v['shards'][0];q=p.parent/s['path'];foreign=self.f.sources[5];raw=M.P.canonical(foreign);q.write_bytes(raw);s['sha256']=M.P.byte_sha(raw);s['source_record_sha256s']=[M.P.digest(foreign)];s['source_ids_sha256']=M.P.digest([foreign['source_id']]);p.write_bytes(M.P.canonical(v))
        with self.assertRaisesRegex(ValueError,'foreign'):M.BoundedShardReader(p,M.P.byte_sha(p.read_bytes()))

    def test_changed_parent_shard_refused_lazily(self):
        r=self.run_plan();reader=M.BoundedShardReader(Path(r['new_index']['path']),r['new_index']['sha256']);p=self.f.inputs/self.f.shards[-1]['path'];p.write_bytes(b''.join(M.P.canonical(dict(json.loads(line),tampered=True))for line in p.read_bytes().splitlines()))
        with self.assertRaisesRegex(ValueError,'SHA differs'):reader.read('s2')

    def test_lookup_foreign_or_not_selected_source_refused(self):
        r=self.run_plan();reader=M.BoundedShardReader(Path(r['new_index']['path']),r['new_index']['sha256'])
        with self.assertRaisesRegex(ValueError,'outside selected'):reader.lookup_previous_for_claim('s2',self.f.sources[4])
        reader.read('s2')
        with self.assertRaisesRegex(ValueError,'outside selected'):reader.lookup_previous_for_claim('s2',self.f.sources[2])

    def test_partial_budget_refused_and_no_database_construction(self):
        with patch.object(M,'MAX_PARTIAL_ROWS',1):
            with self.assertRaisesRegex(ValueError,'budget'):self.run_plan()
        self.assertFalse(self.output.exists())

    def test_relative_pin_and_output_roots_refused(self):
        self.value['old_index']['path']='relative/index.json';self.save()
        with self.assertRaisesRegex(ValueError,'absolute'):self.run_plan()

    def test_idempotent_attempt_does_not_overwrite_report(self):
        self.run_plan();raw=(self.output/'report.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'fresh'):self.run_plan()
        self.assertEqual((self.output/'report.json').read_bytes(),raw)

    def test_portable_final_preserves_old_bytes_and_proves_only_device(self):
        enable_portable(self);before=self.f.snapshot();r=self.run_plan()
        self.assertEqual(self.f.snapshot(),before);self.assertEqual(r['portable_content_proof_count'],1)
        proof=json.loads((self.output/'portable-retained-partial.jsonl').read_text())
        self.assertEqual(proof['stat_differences'],['device']);self.assertTrue(proof['encoded_content_verified']);self.assertFalse(proof['source_or_record_rewritten'])

    def test_portable_refuses_nondevice_drift(self):
        enable_portable(self)
        for key in ('stat_before','stat_after'):self.f.args[2][1][key]['mtime_ns']+=1
        self.f.publish(2);before=self.f.snapshot()
        with self.assertRaisesRegex(ValueError,'non-device'):self.run_plan()
        self.assertEqual(self.f.snapshot(),before)

    def test_portable_refuses_actual_encoded_sha_mismatch(self):
        enable_portable(self);self.f.args[2][1]['file_sha256']='f'*64;self.f.publish(2)
        with self.assertRaisesRegex(ValueError,'encoded source SHA differs'):self.run_plan()

    def test_portable_media_budget_before_read_and_no_database(self):
        import portable_source_identity as portable
        import sqlite3
        enable_portable(self)
        with patch.object(M,'MAX_PORTABLE_ENCODED_BYTES',1),patch.object(portable,'_hash_encoded_fd')as hash_read:
            with self.assertRaisesRegex(ValueError,'budget'):self.run_plan()
            hash_read.assert_not_called()
        with patch.object(sqlite3,'connect',side_effect=AssertionError('database construction forbidden')):
            r=self.run_plan();self.assertFalse(r['database_created'])

    def test_unclaimed_portable_hash_only_on_actual_claim_lookup(self):
        import portable_source_identity as portable
        enable_portable(self);self.f.publish(4);r=self.run_plan();reader=M.BoundedShardReader(Path(r['new_index']['path']),r['new_index']['sha256'])
        with patch.object(portable,'_hash_encoded_fd',wraps=portable._hash_encoded_fd)as hash_read:
            rows=reader.read('s2');hash_read.assert_not_called()
            ref=reader.lookup_previous_for_claim('s2',rows[0]);self.assertEqual(hash_read.call_count,1)
            self.assertEqual(ref['portable_source_evidence']['stat_differences'],['device'])
            self.assertEqual(ref['producer_contract_sha256'],self.f.csha)

    def test_thin_metadata_cannot_drop_unclaimed_shard_and_shrink_count(self):
        r=self.run_plan();p=Path(r['new_index']['path']);v=json.loads(p.read_text())
        v['shards']=[x for x in v['shards']if x['storage']!='parent_unclaimed'];v['source_count']=sum(x['count']for x in v['shards']);v['shard_count']=len(v['shards']);p.write_bytes(M.P.canonical(v))
        with self.assertRaisesRegex(ValueError,'coverage is incomplete'):M.BoundedShardReader(p,M.P.byte_sha(p.read_bytes()))

    def test_thin_cannot_drop_missing_partial_and_claim_more_retained(self):
        r=self.run_plan();p=Path(r['new_index']['path']);v=json.loads(p.read_text())
        v['shards']=[x for x in v['shards']if x['storage']!='local_partial'];v['lease_partitions']['s1'].update(missing_count=0,retained_count=2)
        v['source_count']=2;v['shard_count']=1;p.write_bytes(M.P.canonical(v))
        with self.assertRaisesRegex(ValueError,'retained ledger'):M.BoundedShardReader(p,M.P.byte_sha(p.read_bytes()))

    def test_thin_cannot_change_cursor_to_include_completed_source(self):
        r=self.run_plan();p=Path(r['new_index']['path']);v=json.loads(p.read_text());v['parent_cursor']=0;v['parent_lease_shard_ids']=[];p.write_bytes(M.P.canonical(v))
        with self.assertRaisesRegex(ValueError,'cursor/lease|partial origin'):M.BoundedShardReader(p,M.P.byte_sha(p.read_bytes()))

    def test_old_contract_replacement_after_plan_refused(self):
        r=self.run_plan();self.f.contract.write_bytes(self.f.contract.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'SHA differs'):M.BoundedShardReader(Path(r['new_index']['path']),r['new_index']['sha256'])

if __name__=='__main__':unittest.main()
