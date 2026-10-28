"""控制树复制和默认CPU检查；合成冻结源码，不接触GPU、模型或平台。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_native_partial_recovery_control as P


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.source=self.root/'actual-original-build';self.source.mkdir();self.candidate=self.root/'existing-native/screen/b4/code';self.candidate.mkdir(parents=True)
        members={}
        for name in P.DEPENDENCIES:
            p=self.source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic source '+name+'\n')
            copy=self.candidate/name;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(p.read_bytes());members[name]=P.sha(copy)
        self.bundle=P.write(self.candidate/'bundle_files.json',members)
        self.prepared=P.write(self.root/'existing-native/screen/prepared.json',dict(schema='annotation-native-batch-prepared/v1',source_count=30,
            variants={'native-fp32-sdpa-b4':dict(freeze=dict(frozen_copy=str(self.candidate),bundle_sha256=self.bundle['sha256']),argv=[sys.executable])}))
        self.control=self.root/'new-control';self.output=self.root/'new-probe';self.protected=[self.root/name for name in ('PTN','old-annotation','old-pilot','existing-native')]
        self.raw=("import json,sys\nCONTROL_DEPENDENCIES="+repr(P.DEPENDENCIES)+"\nif '--execute'in sys.argv:raise RuntimeError('GPU execute must never be sent')\nprint(json.dumps(dict(checked=True,GPU_executed=False,source_count=30,first_group_windows=9,partial_recovery_verified=False)))\n").encode()
        self.approved=hashlib.sha256(self.raw).hexdigest();mock=patch.object(P,'APPROVED_CONTROLLER_SHA',self.approved);mock.start();self.addCleanup(mock.stop)
    def stage(self,**kwargs):return P.stage(self.source,self.prepared,self.control,self.output,self.protected,'a'*64,**kwargs)
    def test_exact_source_copy_and_real_subprocess_default_check_never_execute(self):
        before=P.sources(self.source);stage=self.stage();(self.control/P.CONTROLLER).write_bytes(self.raw)
        value=P.finish(self.control)
        self.assertEqual(before,P.sources(self.source));self.assertFalse(self.output.exists());self.assertFalse(value['GPU_executed'])
        stdout=P.pin(value['stdout']);self.assertNotIn('--execute',stdout['argv']);self.assertEqual(stdout['returncode'],0)
        self.assertIn('--execute',value['entry_command_for_review_only']);self.assertEqual(value['dependencies'],stage['dependencies'])
    def test_original_or_copied_source_change_refused(self):
        self.stage();(self.control/P.CONTROLLER).write_bytes(self.raw);(self.control/'gpu_common.py').write_text('#changed')
        with self.assertRaisesRegex(ValueError,'source changed'):P.finish(self.control)
        self.assertFalse(self.output.exists())
    def test_nine_dependency_mismatch_refused_before_creating_control(self):
        (self.source/'gpu_common.py').write_text('# changed source')
        with self.assertRaisesRegex(ValueError,'dependency mismatch'):self.stage()
        self.assertFalse(self.control.exists())
    def test_protected_or_preexisting_roots_refused(self):
        self.protected[0]=self.output
        with self.assertRaisesRegex(ValueError,'overlaps'):self.stage()
        self.protected[0]=self.root/'PTN';self.output.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):self.stage()
    def test_unapproved_controller_cannot_run_cpu_or_make_future_entry(self):
        self.stage();(self.control/P.CONTROLLER).write_bytes(b'raise RuntimeError("must not run")')
        with self.assertRaisesRegex(ValueError,'approved controller'):P.finish(self.control)
        self.assertFalse((self.control/'control-ready-receipt-v1.json').exists())
    def add_b8(self):
        matrix=P.pin(self.prepared);matrix['variants']['native-fp32-sdpa-b8']=dict(matrix['variants']['native-fp32-sdpa-b4'])
        (Path(self.prepared['path'])).unlink();self.prepared=P.write(Path(self.prepared['path']),matrix)
    def test_explicit_B8_uses_copied_approved_controller_and_keeps_source_bytes(self):
        self.add_b8();(self.source/P.CONTROLLER).write_bytes(self.raw);before=P.sources(self.source)
        stage=self.stage(profile='native-fp32-sdpa-b8');value=P.finish(self.control)
        self.assertEqual(stage['profile'],value['profile']);self.assertEqual(value['profile'],'native-fp32-sdpa-b8')
        self.assertEqual(P.pin(value['plan'])['profile'],value['profile']);self.assertEqual(before,P.sources(self.source))
        self.assertEqual(value['controller_origin'],'copied_unchanged_from_source')
        self.assertEqual(value['controller_source_ref'],dict(path=str(self.source/P.CONTROLLER),sha256=self.approved))
        self.assertEqual((self.control/P.CONTROLLER).read_bytes(),self.raw);self.assertFalse(self.output.exists())
    def test_existing_wrong_controller_refused_before_creating_control(self):
        (self.source/P.CONTROLLER).write_text('# unreviewed controller')
        with self.assertRaisesRegex(ValueError,'unapproved recovery controller'):self.stage()
        self.assertFalse(self.control.exists())
    def test_profile_cannot_be_undeclared_or_changed_only_in_stage_receipt(self):
        for profile in ('native-fp32-sdpa-b1',None,True):
            with self.subTest(profile=profile),self.assertRaisesRegex(ValueError,'B4/B8'):self.stage(profile=profile)
        self.add_b8();self.stage();(self.control/P.CONTROLLER).write_bytes(self.raw)
        path=self.control/'control-stage-receipt-v1.json';s=json.loads(path.read_bytes());s['profile']='native-fp32-sdpa-b8';path.write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,'fixed plan/profile'):P.finish(self.control)
    def test_legacy_B4_stage_receipt_without_new_fields_still_checks(self):
        self.stage();(self.control/P.CONTROLLER).write_bytes(self.raw)
        path=self.control/'control-stage-receipt-v1.json';s=json.loads(path.read_bytes())
        for key in ('profile','controller_origin','controller_source_ref'):s.pop(key)
        path.write_text(json.dumps(s));value=P.finish(self.control)
        self.assertEqual(value['profile'],'native-fp32-sdpa-b4');self.assertEqual(value['controller_origin'],'separate_approved_upload_required')


if __name__=='__main__':unittest.main()
