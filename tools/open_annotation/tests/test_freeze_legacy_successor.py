"""固定真实23文件副本的CPU冻结测试；模型/数据路径只是格式样本。"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import freeze_legacy_successor as F


def base_contract():
    _,files,_,_=F.verify_snapshot()
    return {'schema':'redimnet2-production-contract/v2','purpose':'silver_annotation','version':'old-production',
        'model_API_calls_allowed':0,'release_training_labels':False,'runtime':copy.deepcopy(F.LEGACY_RUNTIME),
        'engine_source_files':{n:files[n]for n in F.ENGINE_MODULES},'source_files':{'/synthetic/qwen/inference.py':'c'*64},
        'models':{'asr':{'path':'/synthetic/qwen','files':{'model.safetensors':{'bytes':1,'sha256':'a'*64}}},
                  'aligner':{'path':'/synthetic/ultra','files':{'model.safetensors':{'bytes':1,'sha256':'e6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f'}}}},
        'alignment':{'mode':'ultra_no_head','head_used':False,'repo_id':'17slever17/Qwen3-ForcedAligner-Ultra',
                     'revision':'765dc1d714e4c80196782a5df5bfbcb3e1416736'},
        'decoder':{'soundfile_version':'CPU-format-only'},'parent_contract_sha256':'d'*64,'parent_bundle_sha256':'e'*64}


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.base=base_contract()
        self.path=self.root/'base-contract.json';self.write_base();self.out=self.root/'frozen-successor'
    def tearDown(self):self.tmp.cleanup()
    def write_base(self):self.path.write_bytes(F.canonical(self.base));self.pin=hashlib.sha256(self.path.read_bytes()).hexdigest()
    def run_freeze(self,**kw):return F.freeze(self.path,self.pin,self.out,'new-production',**kw)
    def test_original_23_sources_and_13_semantic_modules_are_identical(self):
        before={str(p.relative_to(F.SNAPSHOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in F.SNAPSHOT.rglob('*')if p.is_file()}
        receipt=self.run_freeze();after={str(p.relative_to(F.SNAPSHOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in F.SNAPSHOT.rglob('*')if p.is_file()}
        self.assertEqual(before,after);self.assertEqual(len(before),24)
        self.assertEqual(receipt['bundle']['sha256'],F.BUNDLE_SHA256);self.assertEqual(receipt['bundle_members_copied'],23)
        for name in before:self.assertEqual((F.SNAPSHOT/name).read_bytes(),(self.out/'code'/name).read_bytes())
        new=json.loads((self.out/'production-contract.json').read_bytes());expected=copy.deepcopy(self.base);expected['version']='new-production'
        self.assertEqual(new,expected);self.assertEqual(new['parent_contract_sha256'],self.base['parent_contract_sha256'])
        self.assertFalse(receipt['new_GPU_execution_verified']);self.assertFalse(receipt['runtime_admission_granted'])
        self.assertTrue(receipt['unfinished_stage_work_may_be_recomputed']);self.assertTrue(receipt['old_stage_cache_preserved_not_rekeyed'])
    def test_reject_mutated_runtime_even_with_recalculated_contract_hash(self):
        for key,value in [('dtype','bfloat16'),('TF32',True),('max_batch',2),('max_new_tokens',256),('boundary_mapping','new-clock')]:
            with self.subTest(key=key):
                self.base=base_contract();self.base['runtime'][key]=value;self.write_base()
                with self.assertRaisesRegex(ValueError,'profile changed'):self.run_freeze()
                self.assertFalse(self.out.exists())
    def test_reject_cpu_fixture_and_wrong_source_pin(self):
        self.base['purpose']='cpu_fixture';self.write_base()
        with self.assertRaisesRegex(ValueError,'CPU fixture'):self.run_freeze()
        self.base=base_contract();self.base['engine_source_files']['worker.py']='0'*64;self.write_base()
        with self.assertRaisesRegex(ValueError,'engine pins differ'):self.run_freeze()
    def test_model_source_metadata_preserved_but_not_rehashed_or_admitted(self):
        r=self.run_freeze();new=json.loads((self.out/'production-contract.json').read_bytes())
        for key in ('models','decoder','alignment','source_files','engine_source_files','runtime'):
            self.assertEqual(new[key],self.base[key])
        self.assertFalse(r['model_weight_files_rehashed']);self.assertFalse(r['external_runtime_sources_reopened'])
        self.assertFalse(r['historical_W2_evidence_verified_by_this_tool'])
    def test_reject_bad_pin_same_or_unsafe_version_and_existing_path(self):
        with self.assertRaisesRegex(ValueError,'SHA differs'):F.freeze(self.path,'0'*64,self.out,'new-production')
        for version in ('old-production','../escape','a/b','.','',True):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError,'version required'):F.freeze(self.path,self.pin,self.out,version)
        self.out.mkdir();(self.out/'preserve').write_text('old')
        with self.assertRaisesRegex(ValueError,'fresh external'):self.run_freeze()
        self.assertEqual((self.out/'preserve').read_text(),'old')
    def test_reject_changed_snapshot_bytes_even_if_contract_rehashed(self):
        snapshot=self.root/'snapshot';shutil.copytree(F.SNAPSHOT,snapshot)
        (snapshot/'worker.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bundle member differs'):self.run_freeze(snapshot=snapshot)
        self.assertFalse(self.out.exists())
    def test_reject_changed_bundle_extra_file_and_snapshot_overlap(self):
        snapshot=self.root/'snapshot';shutil.copytree(F.SNAPSHOT,snapshot)
        (snapshot/'unexpected').write_text('extra')
        with self.assertRaisesRegex(ValueError,'extra files'):self.run_freeze(snapshot=snapshot)
        (snapshot/'unexpected').unlink();(snapshot/'bundle_files.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'bundle SHA'):self.run_freeze(snapshot=snapshot)
        with self.assertRaisesRegex(ValueError,'overlaps fixed inputs'):
            F.freeze(self.path,self.pin,F.SNAPSHOT/'new-output','new-production')
    def test_reject_Git_output_and_symlink(self):
        repo=self.root/'repo';repo.mkdir();(repo/'.git').mkdir()
        with self.assertRaisesRegex(ValueError,'outside Git'):F.freeze(self.path,self.pin,repo/'output','new-production')
        alias=self.root/'alias';alias.symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'without symlink'):F.freeze(self.path,self.pin,alias/'output','new-production')
    def test_real_temporary_snapshot_can_freeze_without_writing_bytecode(self):
        snapshot=self.root/'snapshot';shutil.copytree(F.SNAPSHOT,snapshot)
        result=self.run_freeze(snapshot=snapshot)
        self.assertTrue(result['semantic_engine_bytes_unchanged']);self.assertFalse(list(snapshot.rglob('__pycache__')))
    def test_CLI_help_and_freeze_do_not_import_GPU_libraries(self):
        help_result=subprocess.run([sys.executable,'-B',str(HERE/'freeze_legacy_successor.py'),'--help'],capture_output=True,text=True,check=True)
        self.assertIn('--base-contract',help_result.stdout)
        code="import sys;sys.path.insert(0,%r);import freeze_legacy_successor as F;F.freeze(%r,%r,%r,'new-production');assert not {'torch','qwen_asr','vllm'}&set(sys.modules)"%(str(HERE),str(self.path),self.pin,str(self.out))
        subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,check=True)


if __name__=='__main__':unittest.main()
