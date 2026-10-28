"""有界新队列与旧producer继承的CPU恢复／互斥反例。"""
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import bounded_speed_handoff as H
import bounded_speed_queue as Q
import test_bounded_speed_handoff as Fixtures


class BoundedQueueTests(unittest.TestCase):
    def setUp(self):
        self.fixture=Fixtures.BoundedHandoffTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture.f;self.report=self.fixture.run_plan();self.index=Path(self.report['new_index']['path']);self.isha=self.report['new_index']['sha256']
        self.terminal=dict(path=str(self.f.terminal),sha256=self.f.tsha)
        self.newroot=self.f.root/'new-queue';self.newtree=self.f.root/'new-labels'/'speed-v1';self.newtree.mkdir(parents=True)
        self.lease=Q.ExistingProducerLease(self.index,self.isha,self.terminal).__enter__();self.addCleanup(self.lease.__exit__,None,None,None)
        self.queue=Q.BoundedWorkQueue(self.index,self.isha,self.newroot,'a'*64,self.newtree,self.lease)

    def receipt(self,source,**changes):
        value=dict(source_id=source['source_id'],job_key=H.P.old_job_key(source,'a'*64),sidecar_relpath=source['sidecar_relpath'],sidecar_sha256='1'*64,sidecar_bytes=100,commit_sha256='2'*64,decision='candidate')
        value.update(changes);return value

    def test_new_profile_must_use_new_queue_and_identity(self):
        with self.assertRaisesRegex(ValueError,'explicit new contract'):Q.BoundedWorkQueue(self.index,self.isha,self.newroot,self.f.csha,self.newtree,self.lease)
        with self.assertRaisesRegex(ValueError,'root differs'):Q.BoundedWorkQueue(self.index,self.isha,self.f.queue,'a'*64,self.newtree,self.lease)
        self.assertEqual(self.queue.identity['schema'],'redimnet2-bounded-work-queue/v1')
        self.assertNotEqual(json.loads((self.f.queue/'identity.json').read_text()),self.queue.identity)

    def test_old_parent_and_claim_locks_remain_held(self):
        for p in (self.f.queue/'.parent.lock',self.f.queue/'claims/s1.lock'):
            with p.open('rb')as f:
                with self.assertRaises(BlockingIOError):fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertEqual(len(self.lease.pass_fds),2)

    def test_adopted_inherited_same_open_description_survives_parent_close(self):
        child=Q.ExistingProducerLease.adopt_inherited(self.index,self.isha,self.terminal,self.lease.fd_evidence)
        self.lease.__exit__(None,None,None)
        try:
            child.assert_held()
            with (self.f.queue/'.parent.lock').open('rb')as f:
                with self.assertRaises(BlockingIOError):fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:child.__exit__(None,None,None)
        with (self.f.queue/'.parent.lock').open('rb')as f:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_inherited_forged_inode_refused(self):
        e=self.lease.fd_evidence;e[0]['inode']+=1
        with self.assertRaisesRegex(ValueError,'identity differs'):Q.ExistingProducerLease.adopt_inherited(self.index,self.isha,self.terminal,e)

    def test_broker_change_after_plan_refused(self):
        p=self.f.queue/'broker.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'broker changed'):self.lease.verify_frozen_state()

    def test_claim_fencing_and_incomplete_finish(self):
        with self.queue.claim_next('w0')as claim:
            self.assertEqual(claim.shard['id'],'partial-s1')
            with self.assertRaisesRegex(ValueError,'incomplete'):claim.finish([])
            s=claim.sources()[0]
            with self.assertRaisesRegex(ValueError,'foreign'):claim.assert_member(self.f.sources[0])
            claim.finish([self.receipt(s)])
        saved=json.loads(self.queue.chunk_receipt_path('partial-s1').read_text())
        self.assertFalse(saved['sources'][0]['retained_existing']);self.assertEqual(saved['sources'][0]['producer_contract_sha256'],'a'*64)
        self.assertEqual(saved['sources'][0]['producer_sidecar_tree'],str(self.newtree))

    def test_released_unfinished_claim_retried_before_future_shard(self):
        first=self.queue.claim_next('w0');generation=first.token['generation'];first.close()
        with self.queue.claim_next('w1')as second:
            self.assertEqual(second.shard['id'],'partial-s1');self.assertGreater(second.token['generation'],generation)
            with self.assertRaisesRegex(ValueError,'released'):first.assert_current()

    def test_live_claim_not_stolen_by_other_worker(self):
        first=self.queue.claim_next('w0')
        try:
            second=self.queue.claim_next('w1')
            try:self.assertEqual(second.shard['id'],'s2')
            finally:second.close()
        finally:first.close()

    def test_unclaimed_old_result_keeps_old_job_and_producer(self):
        self.f.publish(4)
        with self.queue.claim_next('w0')as claim:claim.finish([self.receipt(claim.sources()[0])])
        with self.queue.claim_next('w1')as claim:
            a,b=claim.sources();old=self.queue.previous_receipt(claim,a);self.assertTrue(old['retained_existing']);self.assertEqual(old['producer_contract_sha256'],self.f.csha)
            claim.finish([old,self.receipt(b)])
        self.assertTrue(self.queue.is_complete());value=self.queue.receipt('s2');a=value['sources'][0]
        self.assertEqual(a['job_key'],H.P.old_job_key(self.f.sources[4],self.f.csha));self.assertNotEqual(a['job_key'],H.P.old_job_key(self.f.sources[4],'a'*64))
        self.assertEqual(a['producer_sidecar_tree'],str(self.f.tree));self.assertEqual(self.queue.write_completion('s2',value)['retained_source_count'],1)

    def test_foreign_producer_and_relabelled_old_job_refused(self):
        with self.queue.claim_next('w')as claim:
            s=claim.sources()[0]
            with self.assertRaisesRegex(ValueError,'undeclared'):claim.finish([self.receipt(s,producer_contract_sha256='b'*64)])
            with self.assertRaisesRegex(ValueError,'tree/retention'):claim.finish([self.receipt(s,producer_contract_sha256=self.f.csha)])
            old=self.receipt(s,producer_contract_sha256=self.f.csha,producer_sidecar_tree=str(self.f.tree),retained_existing=True)
            with self.assertRaisesRegex(ValueError,'job key'):claim.finish([old])

    def test_crash_after_chunk_receipt_recovers_without_new_inference(self):
        with self.queue.claim_next('w0')as claim:
            s=claim.sources()[0]
            real=self.queue.write_completion
            with patch.object(self.queue,'write_completion',side_effect=RuntimeError('synthetic crash after chunk receipt')):
                with self.assertRaisesRegex(RuntimeError,'synthetic crash'):claim.finish([self.receipt(s)])
        # Next worker reads the complete receipt, repairs only its own new completion accounting, then advances.
        with self.queue.claim_next('w1')as nxt:self.assertEqual(nxt.shard['id'],'s2')
        b=json.loads((self.newroot/'broker.json').read_text());self.assertEqual(b['completed_chunks'],1)

    def test_new_generation_refuses_stale_publication(self):
        claim=self.queue.claim_next('w');p=self.newroot/'attempt_state'/('partial-s1.json');v=json.loads(p.read_text());v['generation']+=1;p.write_bytes(H.P.canonical(v))
        try:
            with self.assertRaisesRegex(ValueError,'stale'):claim.assert_current()
        finally:claim.close()

    def test_old_lock_inode_replacement_refused(self):
        p=self.f.queue/'.parent.lock';p.rename(p.with_name('old-parent-evidence.lock'));p.touch()
        with self.assertRaisesRegex(ValueError,'replaced'):self.queue.claim_next('w')

    def test_old_bytes_never_modified_by_new_queue_work(self):
        before=self.f.snapshot()
        with self.queue.claim_next('w')as claim:claim.finish([self.receipt(claim.sources()[0])])
        self.assertEqual(self.f.snapshot(),before)

    def test_parent_lease_close_refuses_further_new_claim(self):
        self.lease.__exit__(None,None,None)
        with self.assertRaisesRegex(ValueError,'not held'):self.queue.claim_next('w')

    def test_portable_previous_receipt_keeps_proof_and_original_bytes(self):
        self.lease.__exit__(None,None,None)
        # This is a fresh synthetic successor after prior CPU preparation, not a real task mutation.
        import shutil
        shutil.rmtree(self.fixture.output);shutil.rmtree(self.newroot)
        Fixtures.enable_portable(self.fixture);self.f.publish(4);before=self.f.snapshot()
        self.report=self.fixture.run_plan();self.index=Path(self.report['new_index']['path']);self.isha=self.report['new_index']['sha256']
        self.lease=Q.ExistingProducerLease(self.index,self.isha,self.terminal).__enter__()
        self.addCleanup(self.lease.__exit__,None,None,None)
        self.queue=Q.BoundedWorkQueue(self.index,self.isha,self.newroot,'a'*64,self.newtree,self.lease)
        with self.queue.claim_next('w0')as claim:claim.finish([self.receipt(claim.sources()[0])])
        with self.queue.claim_next('w1')as claim:
            a,b=claim.sources();old=self.queue.previous_receipt(claim,a)
            self.assertEqual(old['portable_source_evidence']['stat_differences'],['device'])
            bad=copy.deepcopy(old);bad['portable_source_evidence']['old_contract_sha256']='f'*64
            with self.assertRaisesRegex(ValueError,'binding differs'):claim.finish([bad,self.receipt(b)])
            claim.finish([old,self.receipt(b)])
        self.assertEqual(self.f.snapshot(),before)
        self.assertTrue(self.queue.receipt('s2')['sources'][0]['portable_source_evidence']['encoded_content_verified'])

    def test_hardlinked_new_completion_map_refused(self):
        p=self.newroot/'completion_map.bin';os.link(p,self.f.root/'hardlink-evidence.bin')
        with self.assertRaisesRegex(ValueError,'link differs'):Q.BoundedWorkQueue(self.index,self.isha,self.newroot,'a'*64,self.newtree,self.lease)

    def finish_all(self,preseed=False):
        if preseed:self.f.publish(4)
        while True:
            claim=self.queue.claim_next('worker')
            if claim is None:break
            with claim:
                refs=[]
                for source in claim.sources():refs.append(self.queue.previous_receipt(claim,source)or self.receipt(source))
                claim.finish(refs)

    def test_readonly_progress_excludes_active_and_does_not_construct_queue(self):
        with self.queue.claim_next('active')as claim:
            with patch.object(Q,'BoundedWorkQueue',side_effect=AssertionError('must not construct new queue')):
                r=Q.inspect_bounded_progress(self.index,self.isha,self.newroot,'a'*64)
        self.assertEqual(r['committed_source_lower_bound'],0);self.assertEqual(len(r['broker_snapshot']['leases']),1)
        self.assertEqual(r['sidecars_reopened'],0);self.assertEqual(r['unclaimed_JSONL_read_count'],0);self.assertFalse(r['database_created'])

    def test_readonly_mixed_producer_counts_and_global_scope(self):
        self.finish_all(True);before=self.f.snapshot()
        r=Q.inspect_bounded_progress(self.index,self.isha,self.newroot,'a'*64)
        self.assertEqual((r['committed_source_lower_bound'],r['current_producer_source_count'],r['previous_producer_retained_in_this_queue']),(3,2,1))
        self.assertEqual(r['global_committed_source_lower_bound'],6);self.assertEqual(r['decision_counts'],{'candidate':3,'unknown':0,'failed':0})
        self.assertEqual(r['completion_map_bytes_read'],2048);self.assertEqual(self.f.snapshot(),before)

    def test_readonly_completion_checksum_and_producer_fields_refused(self):
        import hashlib,struct
        self.finish_all();p=self.newroot/'completion_map.bin';original=p.read_bytes();data=bytearray(original);data[60]^=1;p.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'checksum'):Q.inspect_bounded_progress(self.index,self.isha,self.newroot,'a'*64)
        p.write_bytes(original);data=bytearray(original);size=struct.unpack('>I',data[8:12])[0];v=json.loads(data[44:44+size]);v['retained_producer_contract_sha256']='f'*64;raw=H.P.canonical(v)
        data[:1024]=b'RDMAP001'+struct.pack('>I',len(raw))+hashlib.sha256(raw).digest()+raw+bytes(980-len(raw));p.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'producer/input'):Q.inspect_bounded_progress(self.index,self.isha,self.newroot,'a'*64)

    def test_terminal_internal_parent_fd_is_metadata_only(self):
        self.finish_all(True);lock=self.newroot/'.parent.lock';lock.touch()
        with lock.open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            r=Q.verify_bounded_terminal_metadata(self.index,self.isha,self.newroot,'a'*64,parent_lock_fd=f.fileno())
        self.assertTrue(r['metadata_complete']);self.assertFalse(r['GPU_resource_return_verified_by_this_tool']);self.assertFalse(r['full_sidecar_verification']);self.assertFalse(r['formal_runtime_admitted'])

    def test_terminal_external_requires_actual_terminal_and_free_parent(self):
        self.finish_all();lock=self.newroot/'.parent.lock';lock.touch();p=self.f.root/'new-terminal.json'
        value=dict(schema='annotation-writer-terminal-observation/v1',task_id='t-new-fixture',queue_root=str(self.newroot),state='Stopped',observed_utc=Q.datetime.datetime.now(Q.datetime.timezone.utc).isoformat(),workers=[{'name':'t-new-fixture-worker-0','state':'Stopped'}]);p.write_bytes(H.P.canonical(value));pin=dict(path=str(p),sha256=H.P.byte_sha(p.read_bytes()))
        with self.assertRaisesRegex(ValueError,'actual terminal'):Q.verify_bounded_terminal_metadata(self.index,self.isha,self.newroot,'a'*64)
        with lock.open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'still owned'):Q.verify_bounded_terminal_metadata(self.index,self.isha,self.newroot,'a'*64,terminal_pin=pin,task_id='t-new-fixture')
        r=Q.verify_bounded_terminal_metadata(self.index,self.isha,self.newroot,'a'*64,terminal_pin=pin,task_id='t-new-fixture');self.assertTrue(r['metadata_complete'])

    def test_terminal_incomplete_and_bad_parent_fd_refused(self):
        lock=self.newroot/'.parent.lock';lock.touch()
        with lock.open('rb')as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'not completely'):Q.verify_bounded_terminal_metadata(self.index,self.isha,self.newroot,'a'*64,parent_lock_fd=f.fileno())
        with (self.f.root/'random-file').open('wb')as f:
            with self.assertRaisesRegex(ValueError,'identity differs'):Q.verify_bounded_terminal_metadata(self.index,self.isha,self.newroot,'a'*64,parent_lock_fd=f.fileno())

    def test_inherited_adoption_rechecks_terminal_pin_and_contract(self):
        e=self.lease.fd_evidence
        p=self.f.terminal;p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'terminal observation SHA'):Q.ExistingProducerLease.adopt_inherited(self.index,self.isha,self.terminal,e)

if __name__=='__main__':unittest.main()
