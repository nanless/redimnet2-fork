"""有限pilot完整文件复核使用set，不创建SQLite且不漏跨片重复源。"""
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(Path(__file__).resolve().parent))
import report as R
from contracts import canonical_bytes,sha_file
from worker import main as worker_main
import test_audit_outputs as F

class BoundedPilotReport(unittest.TestCase):
    def setUp(self):
        self.f=F.AuditTests(methodName='test_clean_tree_passes_and_cache_is_explicitly_excluded');self.f.setUp();self.addCleanup(self.f.tearDown)
        self.contract=json.loads(self.f.cp.read_text());self.contract['purpose']='pilot_validation'
    def test_real_file_readback_has_no_sqlite_or_temp_index(self):
        with patch.object(R.sqlite3,'connect',side_effect=AssertionError('database forbidden'))as connect,patch.object(R.tempfile,'TemporaryDirectory',side_effect=AssertionError('scratch index forbidden')):
            result=R.verify_run(self.contract,self.f.ip,self.f.q,self.f.output,full=True)
        self.assertTrue(result['passed']);self.assertEqual(result['sidecars_individually_reopened'],2)
        self.assertEqual(result['source_disjointness_storage'],'bounded_pilot_memory_set');self.assertFalse(result['database_created']);connect.assert_not_called()
    def test_duplicate_source_across_two_real_shards_is_rejected_without_db(self):
        f=self.f;source=json.loads(f.first.read_text())['source'];descriptors=[]
        for i in range(2):
            p=f.root/f'dup-{i}.jsonl';p.write_bytes(canonical_bytes(source));descriptors.append({'id':f'dup-{i}','path':p.name,'count':1,'sha256':sha_file(p)})
        index=f.root/'duplicate-index.json';index.write_bytes(canonical_bytes({'schema':'redimnet2-shard-index/v1','source_count':2,'shards':descriptors}))
        queue=f.root/'duplicate-queue'
        worker_main(['--contract',str(f.cp),'--contract-sha',sha_file(f.cp),'--shard-index',str(index),'--index-sha',sha_file(index),
            '--queue-root',str(queue),'--output-root',str(f.output),'--worker-id','CPU','--worker-output',str(f.root/'duplicate-worker'),'--backend','fixture'])
        with patch.object(R.sqlite3,'connect',side_effect=AssertionError('database forbidden')):
            result=R.verify_run(self.contract,index,queue,f.output,full=True)
        self.assertFalse(result['passed']);self.assertFalse(result['database_created'])
        self.assertTrue(any('duplicate source ID' in e.get('traceback','')for e in result['errors']))
    def test_source_memory_cap_and_real_full_branch_preserved(self):
        with R.unique_source_scope({'purpose':'pilot_validation'},{'source_count':4096})as(add,count,commit,storage):
            for i in range(4096):add(str(i))
            with self.assertRaisesRegex(RuntimeError,'limit'):add('overflow')
            with self.assertRaisesRegex(RuntimeError,'duplicate'):add('0')
            self.assertEqual(count(),4096);self.assertEqual(storage,'bounded_pilot_memory_set')
        with patch.object(R.sqlite3,'connect',side_effect=RuntimeError('legacy branch retained')):
            for purpose,count in [('silver_annotation',3),('pilot_validation',4097),('cpu_fixture',3)]:
                with self.subTest(purpose=purpose,count=count),self.assertRaisesRegex(RuntimeError,'legacy branch retained'):
                    with R.unique_source_scope({'purpose':purpose},{'source_count':count}):pass
    def test_tampered_sidecar_still_fails_full_memory_verification(self):
        self.f.first.write_bytes(b'{}')
        with patch.object(R.sqlite3,'connect',side_effect=AssertionError('database forbidden')):
            result=R.verify_run(self.contract,self.f.ip,self.f.q,self.f.output,full=True)
        self.assertFalse(result['passed']);self.assertFalse(result['full_sidecar_readback_verified'])

if __name__=='__main__':unittest.main()
