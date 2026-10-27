"""新run挂载门的CPU模拟；旧run真实存在，但不执行任何GPU或实际findmnt。"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import verify_portable_successor_mounts as M


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


class MountFixture:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.user=self.root/'user';self.user.mkdir()
        self.base=self.user/'task';self.base.mkdir();self.oldrun=self.base/'old-run';self.oldrun.mkdir()
        self.oldcode=self.base/'old-code';self.oldcode.mkdir();self.newcode=self.base/'new-code';self.newcode.mkdir()
        self.shared=self.root/'shared';self.shared.mkdir();self.alias=self.user/'group-shared';self.alias.symlink_to(self.shared,target_is_directory=True)
        self.newrun=self.base/'new-run';self.oldqueue=self.oldrun/'queue';self.oldqueue.mkdir();self.labels=self.oldrun/'labels';self.labels.mkdir()
        self.models={}
        for key in ('asr','aligner'):
            p=self.shared/key;p.mkdir();asset=p/'model.safetensors';asset.write_bytes(b'CPU bytes, not model weights')
            self.models[key]={'path':str((self.alias if key=='aligner'else self.shared)/key),
                'files':{'model.safetensors':{'bytes':asset.stat().st_size,'sha256':pin(asset)['sha256']}}}
        oldcontract=self.save('old-contract.json',{'models':self.models,'version':'old'})
        newcontract=self.save('new-contract.json',{'models':self.models,'version':'new'})
        full=self.save('full-index.json',{'schema':'CPU fixture original full registry'})
        self.mounts=[{'target':'/root','source':'GPFS-user','fstype':'gpfs','mode':'rw'},
                     {'target':'/shared/group-shared','source':'GPFS-shared','fstype':'gpfs','mode':'ro'}]
        self.cloud={'task_root':str(self.base),'run_root':str(self.oldrun),'mounts':self.mounts}
        admission=self.save('old-admission.json',{'cloud':self.cloud,'silver_contract':oldcontract,'full_index':full})
        first=self.user/'ultra_w8_gate_v2.py';first.write_text('original W2 not executed in CPU test')
        second=self.user/'verify_ultra_cloud_mounts_v2.py';second.write_text('original fresh mount checker preserved')
        self.old={'purpose':'cpu_fixture','task_root':str(self.base),'run_root':str(self.oldrun),'code_root':str(self.oldcode),
            'queue_root':str(self.oldqueue),'output_root':str(self.labels),'admission':admission,'contract':oldcontract,'full_index':full,
            'legacy_source_alias':{'alias_path':str(self.alias),'target_path':str(self.shared)},
            'original_claim':self.save('old-claim.json',{}),'terminal_observation':self.save('terminal.json',{}),
            'writer_absence':self.save('absence.json',{}),'checks':[{'script':pin(first)},{'script':pin(second)}]}
        oldref=self.save('old-control.json',self.old);bundle=self.newcode/'bundle_files.json';bundle.write_bytes(b'{}\n')
        freeze=self.save('freeze.json',{'base_contract':oldcontract,'new_contract':newcontract,'code_root':str(self.newcode),'bundle':pin(bundle)})
        remaining=self.save('remaining.json',{'parent_full_index':full})
        self.union={'full_index':full,'new_producer':{'contract':newcontract,'index':remaining},
            'continuation_plan':self.save('plan.json',{}),'portable_source_spec_ref':self.save('portable-spec.json',{}),
            'portable_source_ledger_ref':self.save('ledger.json',{}),'portable_sidecar_inventory_ref':self.save('inventory.json',{})}
        union=self.save('union.json',self.union);consumer=self.save('consumer.json',{})
        self.spec={'schema':M.SCHEMA,'purpose':'cpu_fixture','model_API_calls_allowed':0,'release_training_labels':False,
            'old_control':oldref,'freeze_receipt':freeze,'union_spec':union,'union_attestation':self.save('attestation.json',{'consumer_index':consumer}),
            'current_platform_observation':self.save('current-platform.json',{}),'new_run_root':str(self.newrun),
            'evidence_root':str(self.base/'runtime-evidence')}
        self.path=self.user/'new-spec.json';self.refresh();self.calls=[]
    def save(self,name,value):
        p=self.user/name;p.write_bytes(encoded(value));return pin(p)
    def refresh(self):self.path.write_bytes(encoded(self.spec));self.sha=pin(self.path)['sha256']
    def fake(self,command,**kwargs):
        self.calls.append(command);path=Path(command[3])
        expected=self.mounts[1]if str(path)=='/shared/group-shared'or path==self.shared or self.shared in path.parents else self.mounts[0]
        return SimpleNamespace(stdout=json.dumps({'filesystems':[{'target':expected['target'],'source':expected['source'],
            'fstype':expected['fstype'],'options':expected['mode']+',relatime'}]}),stderr='')
    def verify(self,runner=None):return M.verify(self.path,self.sha,runner=runner or self.fake)
    def close(self):self.tmp.cleanup()


class PortableMountTests(unittest.TestCase):
    def setUp(self):self.f=MountFixture()
    def tearDown(self):self.f.close()
    def test_old_run_exists_is_valid_but_new_run_is_fresh_and_old_admission_unchanged(self):
        before=Path(self.f.old['admission']['path']).read_bytes();r=self.f.verify()
        self.assertTrue(r['passed']);self.assertTrue(r['old_run_exists']);self.assertFalse(r['old_run_required_fresh'])
        self.assertFalse(self.f.newrun.exists());self.assertFalse(r['historical_fresh_mount_checker_executed'])
        self.assertEqual(Path(self.f.old['admission']['path']).read_bytes(),before)
        self.assertEqual(r['model_asset_paths_and_bytes_checked'],2);self.assertFalse(r['GPU_or_model_execution'])
        self.assertFalse(r['runtime_GPU_admission_granted']);self.assertFalse(r['model_content_SHA_rechecked'])
        paths={c[3]for c in self.f.calls}
        self.assertIn(str(self.f.newcode),paths);self.assertIn(self.f.union['full_index']['path'],paths)
        self.assertIn(str(self.f.shared/'asr'/'model.safetensors'),paths)
    def test_new_existing_run_refused_before_findmnt(self):
        self.f.newrun.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh independent'):self.f.verify()
        self.assertEqual(self.f.calls,[])
    def test_shared_RW_or_wrong_GPFS_source_refused(self):
        for mode in ('rw','wrong-source','wrong-fstype'):
            def wrong(cmd,**kw):
                value=self.f.fake(cmd,**kw);rows=json.loads(value.stdout)
                if cmd[3]=='/shared/group-shared':
                    if mode=='rw':rows['filesystems'][0]['options']='rw'
                    if mode=='wrong-source':rows['filesystems'][0]['source']='foreign'
                    if mode=='wrong-fstype':rows['filesystems'][0]['fstype']='overlay'
                value.stdout=json.dumps(rows);return value
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'persistent mount'):self.f.verify(wrong)
    def test_actual_model_path_mounted_on_user_RW_is_refused(self):
        def wrong(cmd,**kw):
            value=self.f.fake(cmd,**kw)
            if cmd[3]==str(self.f.shared/'asr'):
                value.stdout=json.dumps({'filesystems':[{'target':'/root','source':'GPFS-user','fstype':'gpfs','options':'rw'}]})
            return value
        with self.assertRaisesRegex(ValueError,'persistent mount'):self.f.verify(wrong)
    def test_actual_remaining_reference_mount_must_match_original_user_profile(self):
        def wrong(cmd,**kw):
            value=self.f.fake(cmd,**kw)
            if cmd[3]==self.f.union['new_producer']['index']['path']:
                rows=json.loads(value.stdout);rows['filesystems'][0]['source']='different-user-PFS';value.stdout=json.dumps(rows)
            return value
        with self.assertRaisesRegex(ValueError,'persistent mount'):self.f.verify(wrong)
    def test_alias_retarget_and_model_missing_or_nested_symlink_refused(self):
        other=self.f.root/'wrong-shared';other.mkdir();self.f.alias.unlink();self.f.alias.symlink_to(other,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'alias target'):self.f.verify()
        self.f.alias.unlink();self.f.alias.symlink_to(self.f.shared,target_is_directory=True)
        p=self.f.shared/'asr/model.safetensors';p.unlink()
        with self.assertRaisesRegex(ValueError,'model asset'):self.f.verify()
        p.symlink_to(self.f.shared/'aligner/model.safetensors')
        with self.assertRaisesRegex(ValueError,'physical absolute'):self.f.verify()
    def test_spec_SHA_and_frozen_historical_checker_are_bound(self):
        with self.assertRaisesRegex(ValueError,'pinned JSON'):M.verify(self.f.path,'0'*64,runner=self.f.fake)
        Path(self.f.old['checks'][1]['script']['path']).write_text('changed old checker')
        with self.assertRaisesRegex(ValueError,'pinned file'):self.f.verify()
    def test_new_run_appearing_during_findmnt_is_refused(self):
        def create(cmd,**kw):
            if not self.f.newrun.exists():self.f.newrun.mkdir()
            return self.f.fake(cmd,**kw)
        with self.assertRaisesRegex(ValueError,'appeared during mount'):self.f.verify(create)
    def test_module_and_help_do_not_import_GPU_libraries(self):
        code="import sys;sys.path.insert(0,%r);import verify_portable_successor_mounts;assert not {'torch','qwen_asr','vllm'}&set(sys.modules)"%str(HERE)
        subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,check=True)
        result=subprocess.run([sys.executable,'-B',str(HERE/'verify_portable_successor_mounts.py'),'--help'],capture_output=True,text=True,check=True)
        self.assertIn('--spec-sha',result.stdout)

    def test_mount_gate_rejects_new_run_inside_old_code_namespace(self):
        self.f.spec['new_run_root']=str(self.f.oldcode/'mistyped-new-run');self.f.refresh()
        with self.assertRaisesRegex(ValueError,'namespace'):self.f.verify()
        self.assertFalse(Path(self.f.spec['new_run_root']).exists())
    def test_mount_gate_evidence_cannot_be_inside_frozen_or_new_output(self):
        for directory in (self.f.oldcode,self.f.newcode,self.f.newrun):
            with self.subTest(directory=str(directory)):
                self.f.spec['evidence_root']=str(directory/'forbidden-evidence');self.f.refresh()
                with self.assertRaisesRegex(ValueError,'namespace'):self.f.verify()
                self.assertFalse(Path(self.f.spec['evidence_root']).exists())

    def test_shared_input_tree_is_never_an_execution_evidence_destination(self):
        self.f.spec['evidence_root']=str(self.f.shared/'forbidden-evidence');self.f.refresh()
        with self.assertRaisesRegex(ValueError,'namespace|persistent mount'):self.f.verify()
        self.assertFalse(Path(self.f.spec['evidence_root']).exists())
    def test_production_evidence_cannot_be_moved_to_ephemeral_tmp(self):
        old={**self.f.old,'purpose':'production'}
        freeze={'code_root':str(self.f.newcode)}
        with self.assertRaisesRegex(ValueError,'persistent user /root'):
            M.validate_output_namespaces(old,freeze,self.f.union,self.f.newrun,self.f.root/'ephemeral',controller_root=HERE)
