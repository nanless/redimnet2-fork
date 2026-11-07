"""跨节点内容证明和真正剩余准备：真实小文件读SHA，禁止模型/GPU。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import prepare_speed_continuation as P
import portable_source_identity as V
import test_prepare_speed_continuation as F
import schema


def pin(path):return {'path':str(path),'sha256':P.file_sha(path)}


class PortableFixture:
    """合成旧device与当前不同，其余stat真实相同；没有音频解码或假称生产。"""
    def __init__(self,preseed=True):
        self.base=F.SpeedContinuationTests(methodName='runTest');self.base.setUp();b=self.base;self.root=b.root
        for args in b.args:
            identity=args[1];media=Path(args[0]['audio_path']);identity['file_sha256']=P.file_sha(media)
            for k in ('stat_before','stat_after'):identity[k]['device']+=900
        for i in (0,1,2):b.publish(i)
        if preseed:b.publish(4)
        target=b.root/'shared-target';target.mkdir();alias=b.root/'shared-alias';alias.symlink_to(target,target_is_directory=True)
        self.spec={'schema':V.SPEC_SCHEMA,'purpose':'cpu_fixture','source_identity_mode':V.MODE,
            'old_contract':pin(b.contract),'full_index':pin(b.index),'old_queue_root':str(b.queue),'old_sidecar_tree':str(b.tree),
            'source_alias':{'alias_path':str(alias),'target_path':str(target)},'allowed_difference':['device'],
            'encoded_hash_algorithm':'sha256','model_API_calls_allowed':0,'release_training_labels':False}
        self.spec_path=b.root/'portable-spec.json';self.refresh()
    def refresh(self):self.spec_path.write_bytes(P.canonical(self.spec));self.spec_sha=P.file_sha(self.spec_path)
    def close(self):self.base.tearDown()
    def record(self,i=2):
        path=self.base.tree/self.base.sources[i]['sidecar_relpath'];commit=path.with_name(path.name+'.commit.json')
        return self.base.sources[i],json.loads(path.read_bytes()),P.file_sha(path),P.file_sha(commit)
    def prove(self):
        s,r,rs,cs=self.record();return V.prove_source_identity(self.spec,self.spec_sha,s,r,self.base.csha,rs,cs)
    def prepare(self,name='portable-remaining'):
        b=self.base
        with patch.object(P,'EXPECTED_SOURCE_COUNT',6),patch.object(P,'EXPECTED_DOMAINS',{'fixture':6}):
            return P.prepare(b.index,b.isha,b.queue,b.contract,b.csha,b.tree,self.root/name,True,b.terminal,b.tsha,b.task,
                portable_spec_path=self.spec_path,portable_spec_sha=self.spec_sha)


class PortableSourceIdentityTests(unittest.TestCase):
    def setUp(self):self.f=PortableFixture()
    def tearDown(self):self.f.close()

    def test_device_only_difference_requires_real_hash_and_preserves_both_stats(self):
        before=self.f.base.snapshot();entry=self.f.prove();source,record,_,_=self.f.record()
        self.assertEqual(entry['status'],V.VERIFIED);self.assertEqual(entry['stat_differences'],['device'])
        self.assertNotEqual(entry['old_stat_after']['device'],entry['actual_stat_after']['device'])
        self.assertEqual(entry['old_encoded_file_sha256'],entry['actual_encoded_file_sha256'])
        self.assertFalse(entry['PCM_redecoded']);self.assertEqual(entry['encoded_bytes_read'],Path(source['audio_path']).stat().st_size)
        self.assertTrue(V.validate_ledger_entry(entry,source,record,self.f.base.csha,self.f.spec,spec_sha=self.f.spec_sha)['content_verified'])
        self.assertEqual(before,self.f.base.snapshot())

    def test_device_difference_is_accepted_and_does_not_touch_old_bytes(self):
        """2026-11-08 契约：device 漂移不再拒绝；旧生产者字节不得被改写。"""
        before=self.f.base.snapshot()
        self.f.base.run_prepare(True,name='strict')
        self.assertTrue((self.f.root/'strict/index.json').exists(),'device 漂移应被接受并产出新索引')

    def test_final_excludes_partial_and_unclaimed_without_rehashing_completed(self):
        before=self.f.base.snapshot()
        with patch.object(V,'prove_source_identity',wraps=V.prove_source_identity)as verify:
            report=self.f.prepare()
        self.assertEqual(verify.call_count,2);self.assertEqual(before,self.f.base.snapshot())
        self.assertEqual((report['retained_source_count'],report['remaining_source_count']),(4,2))
        self.assertEqual(report['fresh_media_scope'],'retained_partial_and_unclaimed_only')
        self.assertEqual(report['completed_media_scope'],'original_completed_chunk_reference_not_media_rehashed')
        self.assertFalse(report['completed_media_rehashed']);self.assertFalse(report['formal_runtime_admitted'])
        summary=report['portable_sidecar_inventory_summary'];self.assertEqual(summary['original_sources_checked'],6)
        self.assertEqual(summary['matched_completed_sources'],2);self.assertEqual(summary['missing_source_lookups_without_PFS_stat'],2)
        self.assertFalse(summary['completed_sidecar_content_read_by_inventory'])
        self.assertEqual(report['portable_sidecar_inventory_ref']['sha256'],P.file_sha(Path(report['portable_sidecar_inventory_ref']['path'])))
        ledger=report['portable_source_ledger_ref'];rows=[json.loads(v)for v in Path(ledger['path']).read_bytes().splitlines()]
        self.assertEqual(ledger['count'],2);self.assertEqual(ledger['sha256'],P.file_sha(Path(ledger['path'])))
        self.assertEqual({r['source_id']for r in rows},{self.f.base.sources[i]['source_id']for i in (2,4)})
        index=json.loads(Path(report['remaining_index']['path']).read_bytes());remaining=[]
        for shard in index['shards']:
            remaining.extend(json.loads(v)for v in (Path(report['remaining_index']['path']).parent/shard['path']).read_bytes().splitlines())
        self.assertEqual({s['source_id']for s in remaining},{self.f.base.sources[i]['source_id']for i in (3,5)})
        self.assertEqual(remaining,[self.f.base.sources[3],self.f.base.sources[5]])

    def test_explicit_pinned_spec_and_final_old_writer_lock_are_required(self):
        b=self.f.base
        with patch.object(P,'EXPECTED_SOURCE_COUNT',6),patch.object(P,'EXPECTED_DOMAINS',{'fixture':6}):
            with self.assertRaisesRegex(ValueError,'explicit final'):
                P.prepare(b.index,b.isha,b.queue,b.contract,b.csha,b.tree,self.f.root/'not-final',
                    portable_spec_path=self.f.spec_path,portable_spec_sha=self.f.spec_sha)
        self.f.spec['allowed_difference']=['device','inode'];self.f.refresh()
        with self.assertRaisesRegex(ValueError,'only device'):self.f.prepare()
        self.assertFalse((self.f.root/'portable-remaining').exists())

    def test_nondevice_drift_is_rejected_before_reading_content(self):
        s,r,rs,cs=self.f.record();r['source_identity']['stat_before']['inode']+=1;r['source_identity']['stat_after']['inode']+=1
        with patch.object(V,'_hash_encoded_fd')as read:
            with self.assertRaisesRegex(ValueError,'non-device'):
                V.prove_source_identity(self.f.spec,self.f.spec_sha,s,r,self.f.base.csha,V.digest(r),cs)
            read.assert_not_called()

    def test_metadata_agreement_cannot_replace_actual_encoded_sha(self):
        s,r,rs,cs=self.f.record();r['source_identity']['file_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'actual encoded source SHA'):
            V.prove_source_identity(self.f.spec,self.f.spec_sha,s,r,self.f.base.csha,V.digest(r),cs)

    def test_media_changed_during_hash_is_refused_with_no_success_proof(self):
        real=V._hash_encoded_fd;s,r,rs,cs=self.f.record();media=Path(s['audio_path'])
        def mutate(fd):
            result=real(fd);st=media.stat();os.utime(media,ns=(st.st_atime_ns,st.st_mtime_ns+1));return result
        with patch.object(V,'_hash_encoded_fd',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'stat changed during SHA'):self.f.prove()

    def test_wrong_alias_and_nested_symlink_are_not_general_path_exceptions(self):
        alias=Path(self.f.spec['source_alias']['alias_path']);alias.unlink();other=self.f.root/'other';other.mkdir();alias.symlink_to(other,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'alias target'):self.f.prove()
        alias.unlink();target=Path(self.f.spec['source_alias']['target_path']);alias.symlink_to(target,target_is_directory=True)
        (target/'nested.wav').symlink_to(self.f.base.sources[2]['audio_path'])
        with self.assertRaisesRegex(ValueError,'physical absolute'):V.resolve_media_path(self.f.spec,str(alias/'nested.wav'))

    def test_saved_proof_tamper_cannot_hide_old_device_or_claim_PCM_redecode(self):
        entry=self.f.prove();source,record,_,_=self.f.record()
        for change in ('old_device','actual_inode','actual_hash','PCM','spec'):
            bad=copy.deepcopy(entry)
            if change=='old_device':bad['old_stat_after']['device']=bad['actual_stat_after']['device']
            elif change=='actual_inode':bad['actual_stat_before']['inode']+=1;bad['actual_stat_after']['inode']+=1
            elif change=='actual_hash':bad['actual_encoded_file_sha256']='f'*64
            elif change=='PCM':bad['PCM_redecoded']=True
            else:bad['portable_spec_sha256']='f'*64
            with self.subTest(change=change),self.assertRaises(ValueError):
                V.validate_ledger_entry(bad,source,record,self.f.base.csha,self.f.spec,spec_sha=self.f.spec_sha)

    def test_historical_decode_failure_is_preserved_without_claiming_fresh_content(self):
        s,r,rs,cs=self.f.record();args=copy.deepcopy(self.f.base.args[2]);args[1]['decode_status']='decode_failed'
        args[4]=[];args[5]=[];args[7]=[{'stage':'source','reason':'synthetic original decode failure'}]
        record=schema.build_record(*args);self.assertEqual(record['decision']['state'],'failed')
        entry=V.prove_source_identity(self.f.spec,self.f.spec_sha,s,record,self.f.base.csha,V.digest(record),cs)
        self.assertEqual(entry['status'],V.FAILED);self.assertFalse(entry['encoded_content_verified'])
        self.assertIsNone(entry['actual_encoded_file_sha256']);self.assertEqual(entry['encoded_bytes_read'],0)

    def test_orphan_rejected_and_failure_evidence_is_new_only(self):
        b=self.f.base;p=b.tree/b.sources[4]['sidecar_relpath'];p.with_name(p.name+'.commit.json').unlink();raw=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'orphan'):self.f.prepare()
        self.assertEqual(p.read_bytes(),raw);self.assertFalse(p.with_name(p.name+'.commit.json').exists())
        failures=list(self.f.root.glob('.continuation-build-*/portable-preparation-failure.json'))
        self.assertEqual(len(failures),1);self.assertIn('orphan',failures[0].read_text())
        self.assertFalse((self.f.root/'portable-remaining/index.json').exists())

    def test_portable_spec_cannot_retarget_old_inputs(self):
        self.f.spec['old_queue_root']=str(self.f.root/'foreign-queue');self.f.refresh()
        with self.assertRaisesRegex(ValueError,'differs from original input'):self.f.prepare()

    def test_missing_inventory_lookup_never_calls_filesystem_path_checks(self):
        b=self.f.base;inventory=V.SidecarInventory(self.f.root/'inventory.sqlite',b.tree,self.f.spec,self.f.spec_sha,{'fixture':6})
        try:
            inventory.scan()
            with patch.object(P,'child',side_effect=AssertionError('missing source must not query PFS')):
                self.assertIsNone(P.retained_sidecar(b.sources[5],b.tree,b.csha,schema.assert_valid_record,
                    inventory=inventory,inventory_kind='unclaimed'))
        finally:inventory.close()

    def test_unknown_residual_file_is_preserved_and_refused(self):
        leftover=self.f.base.tree/'fixture/.owned-old-leftover.tmp';leftover.write_bytes(b'preserve original failure evidence')
        with self.assertRaisesRegex(ValueError,'unknown residual'):self.f.prepare()
        self.assertEqual(leftover.read_bytes(),b'preserve original failure evidence')

    def test_inventory_rejects_symlink_in_source_tree(self):
        link=self.f.base.tree/'fixture/alias.annotation.json';link.symlink_to(self.f.base.tree/self.f.base.sources[2]['sidecar_relpath'])
        with self.assertRaisesRegex(ValueError,'inventory symlink'):self.f.prepare()
        self.assertTrue(link.is_symlink())

    def test_known_stage_cache_is_explicitly_not_traversed_or_reused(self):
        cache=self.f.base.tree/'.stage_cache';cache.mkdir();p=cache/'old-unrelated-cache';p.write_bytes(b'old identity unchanged')
        before=p.read_bytes();report=self.f.prepare()
        summary=report['portable_sidecar_inventory_summary'];self.assertEqual(summary['known_stage_cache_roots_skipped'],1)
        self.assertFalse(summary['stage_cache_traversed']);self.assertEqual(p.read_bytes(),before)

    def test_inventory_extra_pair_not_in_source_index_is_refused(self):
        source=self.f.base.tree/self.f.base.sources[2]['sidecar_relpath'];extra=source.with_name('foreign.wav.annotation.json')
        extra.write_bytes(source.read_bytes());extra.with_name(extra.name+'.commit.json').write_bytes(source.with_name(source.name+'.commit.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'absent from original index'):self.f.prepare()
        self.assertTrue(extra.exists())

    def test_missing_completed_pair_not_treated_as_portable_remaining(self):
        p=self.f.base.tree/self.f.base.sources[0]['sidecar_relpath'];p.unlink();p.with_name(p.name+'.commit.json').unlink()
        with self.assertRaisesRegex(ValueError,'completed source sidecar pair missing'):self.f.prepare()

    def test_inventory_progress_is_bounded_between_phase_boundaries(self):
        now=[0.0];events=[];b=self.f.base
        with patch.object(V.time,'monotonic',side_effect=lambda:now[0]):
            inventory=V.SidecarInventory(self.f.root/'progress.sqlite',b.tree,self.f.spec,self.f.spec_sha,{'fixture':6},
                progress=lambda stage,counts:events.append((stage,counts)))
            try:
                inventory.scan();self.assertEqual([e[0]for e in events],['inventory_scan_started','inventory_scan_finished'])
                now[0]=29;inventory.emit_progress();self.assertEqual(len(events),2)
                now[0]=30;inventory.emit_progress();self.assertEqual(len(events),3)
                now[0]=31;inventory.emit_progress();self.assertEqual(len(events),3)
            finally:inventory.close()

    def test_independent_deployed_preparer_allows_sibling_output_not_arbitrary_ancestor_ban(self):
        b=self.f.base;code=self.f.root/'standalone-code';code.mkdir()
        for name in ('prepare_speed_continuation.py','portable_source_identity.py','readonly_progress.py',
                     'schema.py','normalization.py','quality_axes.py','source_clock.py'):
            shutil.copyfile(HERE/name,code/name)
        # 仅此合成测试进程缩小分母；正式CLI仍固定11521178，未提供放宽开关。
        script='''import sys,json,pathlib
sys.path.insert(0,sys.argv[1]);import prepare_speed_continuation as P
P.EXPECTED_SOURCE_COUNT=6;P.EXPECTED_DOMAINS={'fixture':6}
a=json.loads(sys.argv[2]);r=P.prepare(*a,portable_spec_path=sys.argv[3],portable_spec_sha=sys.argv[4])
assert r['retained_source_count']==4 and r['remaining_source_count']==2
assert 'torch'not in sys.modules
'''
        args=[str(b.index),b.isha,str(b.queue),str(b.contract),b.csha,str(b.tree),str(self.f.root/'standalone-output'),
              True,str(b.terminal),b.tsha,b.task]
        env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
        before=b.snapshot();result=subprocess.run([sys.executable,'-B','-c',script,str(code),json.dumps(args),
            str(self.f.spec_path),self.f.spec_sha],cwd=self.f.root,env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(before,b.snapshot())
        self.assertIn('inventory_scan_started',result.stdout);self.assertIn('remaining_write_finished',result.stdout)


if __name__=='__main__':unittest.main()
