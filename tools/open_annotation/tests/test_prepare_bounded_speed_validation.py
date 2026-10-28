"""只用三条合成来源验证小批速度准备；不接服务器、不加载模型。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_bounded_speed_validation as P
import contracts as C
from source_registry import source_id
from test_maintenance_entries import external_contract


class BoundedPreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        inputs = self.root / 'inputs'; inputs.mkdir()
        rows = [{'source_id': source_id(ds, 's/a.wav'), 'dataset': ds,
                 'relative_path': 's/a.wav', 'roles': ['train_ptn']} for ds in sorted(P.check_index.__globals__['DOMAINS'])]
        shard = inputs / 's.jsonl'; shard.write_bytes(b''.join(C.canonical_bytes(r) for r in rows))
        index = inputs / 'index.json'
        index.write_bytes(C.canonical_bytes({'schema':'redimnet2-shard-index/v1', 'source_count':3,
            'shards':[{'id':'s','path':'s.jsonl','sha256':C.sha_file(shard),'count':3}]}))
        base = inputs / 'base.json'
        base.write_bytes(C.canonical_bytes({'runtime':{},'purpose':'silver_annotation',
                                         'model_API_calls_allowed':0,'release_training_labels':False}))
        self.spec = {'schema':P.SCHEMA,'source_count':3,
            'base_contract':{'path':str(base),'sha256':C.sha_file(base)},
            'index':{'path':str(index),'sha256':C.sha_file(index)},
            'target_uuids':'auto8','device_lock_root':str(self.root/'locks'),
            'protected_roots':[str(self.root/'running-annotation'),str(self.root/'running-PTN')],
            'python':'/offline/python'}
        self.spec_path = self.root / 'spec.json'
        self.calls = []
    def freeze(self, base, base_sha, output, version_prefix, asr_compute):
        self.calls.append(asr_compute)
        output.mkdir(parents=True)
        return {'frozen_copy':str(output),'bundle_sha256':'b'*64,
                'contracts':{'pilot_validation':{'path':str(output/'contracts/pilot.json'),'sha256':'c'*64}}}
    def run_prepare(self, output=None):
        self.spec_path.write_bytes(C.canonical_bytes(self.spec))
        with patch.object(P.freezer, 'freeze', side_effect=self.freeze):
            return P.prepare(self.spec_path,C.sha_file(self.spec_path),output or self.root/'new-task')
    def test_small_three_domain_auto8_commands_are_independent_and_not_executed(self):
        result = self.run_prepare()
        self.assertEqual(result['source_count'],3); self.assertEqual(result['worker_count'],8)
        self.assertEqual(self.calls,[{'dtype':'float32','attention':'eager'},{'dtype':'float32','attention':'sdpa'}, {'dtype':'bfloat16','attention':'sdpa'}])
        self.assertFalse(result['GPU_executed']);self.assertFalse(result['database_created'])
        self.assertEqual(set(result['comparisons']),{'asr-fp32-sdpa','asr-bf16-sdpa'})
        roots=[]
        for v in result['variants'].values():
            self.assertIn('--full-verify',v['argv']); self.assertIn('auto8',v['argv'])
            track=json.loads(Path(v['track']['path']).read_text())
            self.assertEqual(track['index'],self.spec['index']);self.assertIn('pilot.json',track['contract']['path'])
            roots.append(track['queue_root'])
        self.assertEqual(len(set(roots)),3)
    def test_real_freeze_pins_same_engine_and_changes_only_ASR_profile(self):
        base=Path(self.spec['base_contract']['path']);base.write_bytes(C.canonical_bytes(external_contract()))
        self.spec['base_contract']['sha256']=C.sha_file(base)
        self.spec_path.write_bytes(C.canonical_bytes(self.spec))
        result=P.prepare(self.spec_path,C.sha_file(self.spec_path),self.root/'real-freeze')
        contracts=[]
        for v in result['variants'].values():
            raw=Path(v['track']['path']).read_bytes()
            self.assertEqual(C.digest(json.loads(raw)),v['track']['sha256'])
            contracts.append(json.loads(Path(v['freeze']['contracts']['pilot_validation']['path']).read_text()))
        for c in contracts[1:]:
            for key in ('engine_source_files','alignment','models','quality_policy','source_files'):
                self.assertEqual(c[key],contracts[0][key])
            left=dict(c['runtime']);right=dict(contracts[0]['runtime'])
            left.pop('asr_compute');right.pop('asr_compute');self.assertEqual(left,right)
        self.assertEqual(contracts[0]['runtime']['asr_compute'],{'dtype':'float32','attention':'eager'})
        self.assertEqual(contracts[2]['runtime']['asr_compute'],{'dtype':'bfloat16','attention':'sdpa'})
        self.assertNotIn('torch',sys.modules)
        self.assertFalse(list((self.root/'real-freeze').rglob('*.sqlite')))

    def test_explicit_two_cards_and_selected_candidate_only(self):
        self.spec['target_uuids']=['GPU-00000000-0000-0000-0000-000000000001','GPU-00000000-0000-0000-0000-000000000002']
        self.spec['profiles']=['asr-fp32-eager','asr-bf16-sdpa']
        result=self.run_prepare();self.assertEqual(result['worker_count'],2)
        self.assertEqual(len(result['variants']),2)
    def test_full_library_input_refused_before_freeze_or_audio(self):
        for count in (True,0,4097,11521178):
            with self.subTest(count=count):
                self.spec['source_count']=count
                with self.assertRaisesRegex(RuntimeError,'finite'):self.run_prepare()
        self.assertEqual(self.calls,[])
    def test_no_profile_duplicates_missing_baseline_or_unknown(self):
        for profiles in (['asr-fp32-eager','asr-fp32-eager'],['asr-bf16-sdpa'],['asr-fp32-eager','bogus'],['asr-fp32-eager',{}]):
            with self.subTest(profiles=profiles):
                self.spec['profiles']=profiles
                with self.assertRaisesRegex(RuntimeError,'baseline first'):self.run_prepare()
        self.assertEqual(self.calls,[])
    def test_false_small_denominator_cannot_list_many_or_large_shards(self):
        path=Path(self.spec['index']['path']);original=json.loads(path.read_text())
        for mutate in ('many','large'):
            obj=copy.deepcopy(original)
            if mutate=='many':obj['shards']*=4
            else:obj['shards'][0]['count']=10000
            path.write_bytes(C.canonical_bytes(obj));self.spec['index']['sha256']=C.sha_file(path)
            with self.subTest(mutate=mutate),patch.object(P,'check_index')as checker:
                with self.assertRaisesRegex(RuntimeError,'metadata'):self.run_prepare()
                checker.assert_not_called()
        self.assertEqual(self.calls,[])

    def test_explicit_production_and_PTn_roots_are_protected(self):
        for root in self.spec['protected_roots']:
            with self.subTest(root=root):
                with self.assertRaisesRegex(RuntimeError,'historical'):self.run_prepare(Path(root)/'child')
        self.assertEqual(self.calls,[])
    def test_reused_output_old_input_and_lock_overlap_refused(self):
        old=self.root/'existing';old.mkdir()
        with self.assertRaisesRegex(RuntimeError,'fresh'):self.run_prepare(old)
        with self.assertRaisesRegex(RuntimeError,'historical'):self.run_prepare(self.root/'inputs/child')
        self.spec['device_lock_root']=str(self.root/'new-task/locks')
        with self.assertRaisesRegex(RuntimeError,'locks'):self.run_prepare()
        self.assertEqual(self.calls,[])
    def test_auto2_prepares_pilot_only_and_same_guarded_launcher(self):
        self.spec['target_uuids']='auto2'
        result=self.run_prepare();self.assertEqual(result['worker_count'],2)
        for value in result['variants'].values():
            self.assertIn('auto2',value['argv'])
            self.assertTrue(value['argv'][2].endswith('/code/gpu_launch.py'))
            self.assertTrue(any(x.endswith('/contracts/pilot.json') for x in value['argv']))

    def test_source_identity_and_three_domain_coverage_are_required(self):
        index_path=Path(self.spec['index']['path']);idx=json.loads(index_path.read_text());shard=index_path.parent/'s.jsonl'
        rows=[json.loads(x) for x in shard.read_text().splitlines()];rows[0]['dataset']='musan'
        shard.write_bytes(b''.join(C.canonical_bytes(r) for r in rows));idx['shards'][0]['sha256']=C.sha_file(shard)
        index_path.write_bytes(C.canonical_bytes(idx));self.spec['index']['sha256']=C.sha_file(index_path)
        with self.assertRaisesRegex(RuntimeError,'three-training'):self.run_prepare()
        self.assertEqual(self.calls,[])
    def test_source_or_parent_SHA_tampering_refused(self):
        for key in ('index','base_contract'):
            old=self.spec[key]['sha256'];self.spec[key]['sha256']='0'*64
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'SHA'):self.run_prepare()
            self.spec[key]['sha256']=old
        self.assertEqual(self.calls,[])
    def test_unknown_device_bad_runtime_and_no_protected_roots_refused(self):
        for key,value in [('target_uuids',['auto8']),('max_runtime_seconds',7201),('protected_roots',[])]:
            old=copy.deepcopy(self.spec.get(key));self.spec[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.run_prepare()
            if old is None:self.spec.pop(key)
            else:self.spec[key]=old
        self.assertEqual(self.calls,[])
    def test_soft_link_input_or_output_ancestor_refused(self):
        alias=self.root/'alias';alias.symlink_to(self.root/'inputs',target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError,'physical'):self.run_prepare(alias/'new')
        self.assertEqual(self.calls,[])

if __name__=='__main__':unittest.main()
