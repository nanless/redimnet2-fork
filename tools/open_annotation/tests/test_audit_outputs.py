"""Real small-file final-audit regression: no model/GPU; explicit cpu_fixture."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('complete_cpu_audit',HERE/'audit_outputs.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
from contracts import canonical_bytes,sha_file
from source_registry import source_id
from worker import main as worker_main
from work_queue import ProductionRunLock


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();r=self.root=Path(self.temp.name)
        model=r/'fake-model';model.mkdir();(model/'fake.bin').write_bytes(b'CPU fixture, not model')
        pin={'path':str(model),'files':{'fake.bin':{'bytes':(model/'fake.bin').stat().st_size,'sha256':sha_file(model/'fake.bin')}}}
        c={'schema':'redimnet2-production-contract/v2','version':'audit-fixture-v1','purpose':'cpu_fixture',
            'model_API_calls_allowed':0,'release_training_labels':False,'models':{'asr':pin,'aligner':pin},'source_files':{},
            'runtime':{'dtype':'float32','attention':'eager','TF32':False,'max_batch':1,'length_ratio_limit':1,
                'max_new_tokens':512,'source_batch_rows':2,'window':{'core_seconds':24,'context_seconds':2}}}
        self.cp=r/'contract.json';self.cp.write_bytes(canonical_bytes(c));self.output=r/'output';self.q=r/'queue'
        self.tree=self.output/c['version'];sources=[]
        for i in range(2):
            rel='speaker/audio%d.wav'%i;sources.append({'schema':'redimnet2-source/v1','source_id':source_id('fixture',rel),
                'dataset':'fixture','version':'test','relative_path':rel,'audio_path':None,'resolution_status':'unresolved',
                'roles':['evaluation:fixture'],'original_splits':['test'],'parent_id':None,'speaker_evidence':None,
                'protection':{'training_forbidden':True,'reasons':['CPU_fixture']},'channel_policy':'mean_channels_to_mono',
                'source_interval':None,'old_vad':None,'sidecar_relpath':'fixture/'+rel+'.annotation.json'})
        self.first=self.tree/sources[0]['sidecar_relpath'];shard=r/'chunk.jsonl';shard.write_bytes(b''.join(canonical_bytes(s) for s in sources))
        self.ip=r/'index.json';self.ip.write_bytes(canonical_bytes({'schema':'redimnet2-shard-index/v1',
            'source_count':2,'shards':[{'id':'chunk','path':shard.name,'count':2,'sha256':sha_file(shard)}]}))
        worker_main(['--contract',str(self.cp),'--contract-sha',sha_file(self.cp),'--shard-index',str(self.ip),
            '--index-sha',sha_file(self.ip),'--queue-root',str(self.q),'--output-root',str(self.output),
            '--worker-id','CPU','--worker-output',str(r/'worker'),'--backend','fixture'])

    def tearDown(self):self.temp.cleanup()

    def run_audit(self):return audit.audit_run(self.cp,self.ip,self.q,self.output,self.root/'audit')

    def test_clean_tree_passes_and_cache_is_explicitly_excluded(self):
        cache=self.tree/'.stage_cache';cache.mkdir(exist_ok=True);(cache/'not-a-reviewed-cache.bin').write_bytes(b'x')
        result=self.run_audit();self.assertTrue(result['passed']);self.assertEqual(result['source_count'],2)
        self.assertEqual(result['counts']['expected_files_seen'],5)
        self.assertEqual(result['counts']['stage_cache_subtrees_excluded'],1)
        self.assertFalse(result['stage_cache_integrity_claimed'])
        self.assertTrue(result['same_parent_lock_held_for_entire_audit'])
        for manifest in result['manifests'].values():
            self.assertEqual(sha_file(manifest['path']),manifest['sha256'])

    def test_unregistered_extra_sidecar_is_rejected(self):
        p=self.first.with_name('extra.wav.annotation.json');p.write_bytes(b'{}')
        result=self.run_audit();self.assertFalse(result['passed']);self.assertEqual(result['counts']['unexpected_files'],1)
        self.assertTrue(p.exists())

    def test_orphan_commit_is_rejected_and_preserved(self):
        p=self.first.with_name('orphan.annotation.json.commit.json');p.write_bytes(b'{}')
        result=self.run_audit();self.assertFalse(result['passed'])
        self.assertTrue(any(e['kind']=='orphan_commit' for e in result['errors']));self.assertTrue(p.exists())

    def test_directory_symlink_is_rejected_without_following(self):
        external=self.root/'external';external.mkdir();(external/'not-in-tree.bin').write_bytes(b'no follow')
        link=self.tree/'danger';link.symlink_to(external,target_is_directory=True)
        result=self.run_audit();self.assertFalse(result['passed']);self.assertEqual(result['counts']['symlinks'],1)
        with gzip.open(result['manifests']['inventory']['path'],'rt') as f:rows=[json.loads(line) for line in f]
        self.assertFalse(any(x['path'].endswith('not-in-tree.bin') for x in rows));self.assertTrue(link.is_symlink())

    def test_known_atomic_temp_retained_listed_and_not_admitted(self):
        p=self.first.with_name('.'+self.first.name+'.'+'a'*24+'.tmp');p.write_bytes(b'partial or losing attempt')
        result=self.run_audit();self.assertTrue(result['passed']);self.assertEqual(result['counts']['known_atomic_temporary_files'],1)
        self.assertEqual(result['counts']['expected_files_seen'],5);self.assertTrue(p.exists())
        with gzip.open(result['manifests']['temporaries']['path'],'rt') as f:rows=[json.loads(line) for line in f]
        self.assertEqual(len(rows),1);self.assertFalse(rows[0]['published']);self.assertFalse(rows[0]['deleted'])
        self.assertEqual(rows[0]['sha256'],sha_file(p))

    def test_temp_for_unregistered_target_is_not_hidden(self):
        p=self.first.with_name('.unregistered.annotation.json.'+'b'*24+'.tmp');p.write_bytes(b'orphan')
        result=self.run_audit();self.assertFalse(result['passed']);self.assertEqual(result['counts']['known_atomic_temporary_files'],0)

    def test_same_parent_lock_blocks_audit_while_run_owns_it(self):
        with ProductionRunLock(self.q):
            with self.assertRaises(Exception):self.run_audit()
        self.assertFalse((self.root/'audit/full_report.json').exists())


if __name__=='__main__':unittest.main()
