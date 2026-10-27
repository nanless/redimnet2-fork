"""旧完成片有界纯读线程检查；小文件、真锁，绝不读媒体或运行模型。"""
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import sys

HERE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HERE),str(HERE/'tests')]
import union_coverage as U
from test_union_coverage import UnionFixture
from work_queue import encode_completion_slot
from contracts import canonical_bytes,digest,job_key


class RetainedReadConcurrencyTests(unittest.TestCase):
    def setUp(self):self.f=UnionFixture();self.addCleanup(self.f.close)
    def chunk(self,executor=None,stats=None,rows=None,descriptor=None,queue=None):
        b=self.f.base
        return list(U._complete_chunk(b.isha,descriptor or b.shards[0],0,queue or b.queue,
            self.f.spec['old_producer'],rows or b.sources[:2],contract=U.ref(self.f.spec['old_producer']['contract']),
            read_executor=executor,read_stats=stats))
    def test_parallel_reads_finish_out_of_order_but_return_exact_serial_source_proofs(self):
        expected=self.chunk();real=U.read_result;first,second=[s['source_id']for s in self.f.base.sources[:2]]
        finished=threading.Event();completion=[];guard=threading.Lock()
        def read(source,*args,**kwargs):
            if source['source_id']==first:self.assertTrue(finished.wait(5),'second independent read never started')
            result=real(source,*args,**kwargs)
            with guard:completion.append(source['source_id'])
            if source['source_id']==second:finished.set()
            return result
        with ThreadPoolExecutor(max_workers=2)as pool,patch.object(U,'read_result',side_effect=read):actual=self.chunk(pool)
        self.assertEqual(actual,expected);self.assertEqual(completion,[second,first])
    def test_default_and_configured_workers_have_exact_scope_and_preserve_old_bytes(self):
        before=self.f.old_snapshot()
        baseline=U.validate_union(self.f.spec_path,self.f.spec_sha)
        self.assertEqual(baseline['retained_read_workers'],1);self.assertEqual(baseline['retained_read_executor'],'serial')
        for workers in (2,4):
            self.f.spec['retained_read_workers']=workers;self.f.write_spec()
            r=U.validate_union(self.f.spec_path,self.f.spec_sha)
            for key in ('source_count','retained_source_count','remaining_source_count','decision_counts','exact_union_and_disjoint'):
                self.assertEqual(r[key],baseline[key])
            self.assertEqual(r['retained_read_workers'],workers);self.assertEqual(r['retained_read_scope'],'old_completed_chunks_only')
            self.assertEqual(r['retained_completed_source_reads'],2);self.assertEqual(r['retained_read_max_submitted_batch'],2)
        self.assertEqual(before,self.f.old_snapshot())
    def test_unknown_types_and_out_of_bounds_refuse_before_database_creation_for_both_purposes(self):
        for purpose in ('production','cpu_fixture'):
            for value in (True,False,None,0,-1,5,8,1.0,'2',[],{}):
                with self.subTest(purpose=purpose,value=value):
                    self.f.spec.update(purpose=purpose,retained_read_workers=value);self.f.write_spec()
                    with patch.object(U.sqlite3,'connect')as connect:
                        with self.assertRaisesRegex(ValueError,'retained_read_workers'):
                            U.validate_union(self.f.spec_path,self.f.spec_sha,output_index=self.f.root/'never-created.sqlite')
                        connect.assert_not_called()
                    self.assertFalse((self.f.root/'never-created.sqlite').exists())
    def test_sqlite_and_partial_sources_remain_on_main_thread(self):
        self.f.spec['retained_read_workers']=2;self.f.write_spec();main=threading.get_ident()
        connect=U.sqlite3.connect;read=U.read_result;old_completed={s['source_id']for s in self.f.base.sources[:2]};observed=[]
        class Connection:
            def __init__(self,db):self.db=db
            def execute(self,*a,**k):
                if threading.get_ident()!=main:raise AssertionError('SQLite escaped main thread')
                return self.db.execute(*a,**k)
            def commit(self):
                if threading.get_ident()!=main:raise AssertionError('SQLite commit escaped main thread')
                return self.db.commit()
            def close(self):return self.db.close()
        def wrapped(source,*a,**k):
            thread=threading.get_ident();observed.append((source['source_id'],thread))
            if source['source_id']in old_completed:self.assertNotEqual(thread,main)
            else:self.assertEqual(thread,main)
            return read(source,*a,**k)
        with patch.object(U.sqlite3,'connect',side_effect=lambda *a,**k:Connection(connect(*a,**k))),patch.object(U,'read_result',side_effect=wrapped):
            U.validate_union(self.f.spec_path,self.f.spec_sha)
        self.assertEqual(len(observed),3)
    def test_failure_drains_inflight_reads_under_old_lock_and_never_publishes_final_database(self):
        self.f.spec['retained_read_workers']=2;self.f.write_spec();barrier=threading.Barrier(2)
        release=threading.Event();done=threading.Event();checked=[];real=U.read_result;first=self.f.base.sources[0]['source_id']
        old_completed={s['source_id']for s in self.f.base.sources[:2]};lock=self.f.base.queue/'.parent.lock'
        class Pool(ThreadPoolExecutor):
            def shutdown(pool,wait=True,*,cancel_futures=False):
                if cancel_futures:
                    with lock.open('rb')as probe:
                        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:checked.append(True)
                        else:raise AssertionError('old parent lock released before inflight reads drained')
                    release.set()
                return super().shutdown(wait=wait,cancel_futures=cancel_futures)
        def read(source,*a,**k):
            if source['source_id']in old_completed:
                barrier.wait(5)
                if source['source_id']==first:raise ValueError('synthetic parallel proof failure')
                self.assertTrue(release.wait(5),'inflight read was not drained under old lock')
                result=real(source,*a,**k);done.set();return result
            return real(source,*a,**k)
        output=self.f.root/'consumer.sqlite';before=self.f.old_snapshot()
        with patch.object(U,'ThreadPoolExecutor',Pool),patch.object(U,'read_result',side_effect=read):
            with self.assertRaisesRegex(ValueError,'synthetic parallel proof failure'):
                U.validate_union(self.f.spec_path,self.f.spec_sha,output_index=output)
        self.assertTrue(checked);self.assertTrue(done.is_set());self.assertFalse(output.exists())
        self.assertTrue(list(self.f.root.glob('.consumer-building-*.sqlite')));self.assertEqual(before,self.f.old_snapshot())
    def test_large_chunk_submissions_are_bounded_and_ordered_without_retaining_records(self):
        b=self.f.base;count=600;rows=[];receipts=[];path=b.tree/b.sources[0]['sidecar_relpath'];size=path.stat().st_size
        for i in range(count):
            source=copy.deepcopy(b.sources[0]);source['source_id']='src_'+('%064x'%i);rows.append(source)
            receipts.append({'source_id':source['source_id'],'job_key':job_key(source,b.csha),
                'sidecar_relpath':source['sidecar_relpath'],'sidecar_bytes':size,'sidecar_sha256':'a'*64,'commit_sha256':'b'*64,'decision':'candidate'})
        queue=self.f.root/'bounded-queue';(queue/'completed').mkdir(parents=True)
        descriptor={'id':'bounded','sha256':'c'*64,'count':count}
        chunk={'schema':'redimnet2-chunk-commit/v1','index_sha256':b.isha,'contract_sha256':b.csha,
            'shard_id':'bounded','shard_sha256':'c'*64,'source_count':count,'sources':receipts}
        cp=queue/'completed/bounded.json';cp.write_bytes(canonical_bytes(chunk))
        slot={k:chunk[k]for k in ('index_sha256','contract_sha256','shard_id','shard_sha256','source_count')}
        slot.update(commit_sha256=U.sha_file(cp),source_ids_sha256=digest(sorted(s['source_id']for s in rows)),
            decision_counts={'candidate':count,'unknown':0,'failed':0})
        (queue/'completion_map.bin').write_bytes(encode_completion_slot(slot));batches=[]
        class Pool(ThreadPoolExecutor):
            def map(pool,fn,batch,*a,**k):batches.append(len(batch));return super().map(fn,batch,*a,**k)
        stats={'max_submitted_batch':0,'old_completed_source_reads':0}
        fake=({'decision':{'state':'candidate'}},'a'*64,'b'*64)
        with Pool(max_workers=2)as pool,patch.object(U,'read_result',return_value=fake):
            result=self.chunk(pool,stats,rows,descriptor,queue)
        self.assertEqual(batches,[256,256,88]);self.assertEqual(stats['max_submitted_batch'],256)
        self.assertEqual(stats['old_completed_source_reads'],count);self.assertEqual([r[0]['source_id']for r in result],[s['source_id']for s in rows])


if __name__=='__main__':unittest.main()
