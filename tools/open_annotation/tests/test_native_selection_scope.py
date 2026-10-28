"""同次CLI的实际有限证明复用；完整CPU原件首次核过后才建立不序列化scope。"""
from pathlib import Path
import copy
import os
import pickle
import sys
import unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import native_batch_admission as A
from test_native_formal_admission import NativeFixture


class SelectionScopeTests(unittest.TestCase):
    def setUp(self):self.f=NativeFixture(self)
    def get(self,scope=None,**kwargs):return A.check_native_selected_pilot(kwargs.get('pin',self.f.pin),kwargs.get('contract',self.f.new),kwargs.get('purpose','cpu_fixture'),selection_scope=scope)
    def test_default_and_explicit_disabled_always_do_full_check(self):
        full=A._full_check_native_selected_pilot
        with patch.object(A,'_full_check_native_selected_pilot',wraps=full)as run:
            self.get();self.get();self.assertEqual(run.call_count,2)
            with A.NativeSelectionScope()as scope:self.get(scope);self.get(scope);self.assertEqual(scope.facts()['full_checks'],2);self.assertEqual(scope.facts()['reuses'],0)
        self.assertFalse(scope.facts()['enabled']);self.assertTrue(scope.facts()['closed'])
    def test_enabled_scope_only_after_actual_full_check_then_all_SHA_without_JSON_reparse(self):
        full=A._full_check_native_selected_pilot
        with A.NativeSelectionScope(enabled=True)as scope,patch('sqlite3.connect',side_effect=AssertionError('no DB')):
            with patch.object(A,'_full_check_native_selected_pilot',wraps=full)as run:self.get(scope);self.assertEqual(run.call_count,1)
            count=scope.facts()['fingerprint_file_count'];pinned_file=A.R.pinned_file
            with patch.object(A,'_full_check_native_selected_pilot',side_effect=AssertionError('no repeated full JSON reconstruction')),patch.object(A,'pinned',side_effect=AssertionError('no JSON reparse')),patch.object(Path,'open',side_effect=AssertionError('no Path.open JSON reader')),patch.object(A.R,'pinned_file',wraps=pinned_file)as hash_file:
                value=self.get(scope)
            self.assertEqual(hash_file.call_count,count)
            self.assertEqual(value['schema'],A.SCHEMA);self.assertEqual(scope.facts()['full_checks'],1);self.assertEqual(scope.facts()['reuses'],1)
    def test_changed_selected_pin_contract_or_purpose_refused(self):
        for args in ({'pin':dict(self.f.pin,sha256='f'*64)},{'contract':dict(self.f.new,version='changed')},{'purpose':'production'}):
            with self.subTest(args=args),A.NativeSelectionScope(enabled=True)as scope:
                self.get(scope)
                with self.assertRaisesRegex(ValueError,'binding changed'):self.get(scope,**args)
                self.assertTrue(scope.facts()['closed'])
    def test_quality_file_content_or_mtime_change_refused(self):
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);path=Path(self.f.selected['quality_review']['path']);old=path.stat();os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):self.get(scope)
    def test_selected_admission_original_file_change_refused(self):
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);Path(self.f.pin['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):self.get(scope)
    def test_quality_evidence_named_bundle_is_not_treated_as_GPU_manifest(self):
        evidence=self.f.put('quality-material/bundle_files.json',{'synthetic_CPU_only':True})
        quality=copy.deepcopy(self.f.quality);quality['review_evidence']=[evidence]
        self.f.selected['quality_review']=self.f.put('quality-role.json',quality);self.f.save()
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);self.get(scope);self.assertEqual(scope.facts()['full_checks'],1);self.assertEqual(scope.facts()['reuses'],1)
    def test_changed_bundle_member_without_changing_bundle_JSON_refused(self):
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);(self.f.code/'gpu_launch.py').write_text('# changed member, same bundle JSON')
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):self.get(scope)
    def test_replaced_same_bytes_inode_and_restored_mtime_refused(self):
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);path=Path(self.f.selected['comparison']['path']);st=path.stat();new=path.with_name('replacement.json');new.write_bytes(path.read_bytes());os.utime(new,ns=(st.st_atime_ns,st.st_mtime_ns));new.replace(path)
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):self.get(scope)

    def test_same_size_content_change_even_if_entire_metadata_fingerprint_collides(self):
        original=A._selection_fingerprint;path=Path(self.f.selected['quality_review']['path'])
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);expected=original(self.f.selected['quality_review'],verify_sha=False)
            raw=path.read_bytes();old=path.stat();path.write_bytes((b'x'if raw[:1]!=b'x'else b'y')+raw[1:]);os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
            def collided(pin,*,verify_sha):
                if not verify_sha and pin['path']==str(path):return expected
                return original(pin,verify_sha=verify_sha)
            with patch.object(A,'_selection_fingerprint',side_effect=collided),self.assertRaisesRegex(ValueError,'pinned file differs'):self.get(scope)
            self.assertTrue(scope.facts()['closed']);self.assertEqual(scope.facts()['reuses'],0)
    def test_returned_dict_is_not_the_capability_or_mutable_cache(self):
        with A.NativeSelectionScope(enabled=True)as scope:
            first=self.get(scope);first['verified_evidence_pins'].clear();first['schema']='forged'
            second=self.get(scope);self.assertEqual(second['schema'],A.SCHEMA);self.assertTrue(second['verified_evidence_pins'])
            with self.assertRaisesRegex(ValueError,'issued'):self.get(second)
    def test_dict_boolean_and_unissued_object_refused(self):
        for fake in ({'passed':True},True,object.__new__(A.NativeSelectionScope)):
            with self.subTest(fake_type=type(fake).__name__),self.assertRaisesRegex(ValueError,'actually issued'):self.get(fake)
        with self.assertRaisesRegex(ValueError,'boolean opt-in'):A.NativeSelectionScope(enabled=1)
    def use_unique_evidence_directory(self):
        evidence=self.f.put('separate-proof/evidence.json',{'synthetic_CPU_only':True});q=copy.deepcopy(self.f.quality);q['review_evidence']=[evidence]
        self.f.selected['quality_review']=self.f.put('scope-quality.json',q);self.f.save();return Path(evidence['path']).parent
    def test_ancestor_inode_change_is_refused_before_any_file_reuse(self):
        parent=self.use_unique_evidence_directory()
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);old=parent.with_name('original-evidence-directory');parent.rename(old);parent.mkdir()
            with patch.object(A,'_selection_fingerprint',side_effect=AssertionError('ancestor identity must reject first')):
                with self.assertRaisesRegex(ValueError,'ancestor directory identity'):self.get(scope)
    def test_ancestor_symlink_refused_and_unrelated_file_creation_allowed(self):
        parent=self.use_unique_evidence_directory()
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);(parent/'unrelated-new.json').write_text('{}');self.get(scope)
            self.assertGreater(scope.facts()['fingerprint_directory_count'],0)
            old=parent.with_name('original-evidence-directory');parent.rename(old);parent.symlink_to(old,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'ancestor directory identity'):self.get(scope)

    def test_closed_cross_process_and_pickle_refused(self):
        scope=A.NativeSelectionScope(enabled=True);self.get(scope)
        with self.assertRaisesRegex(TypeError,'cannot be serialized'):pickle.dumps(scope)
        pid=os.getpid()
        with patch.object(A.os,'getpid',return_value=pid+1),self.assertRaisesRegex(ValueError,'another process'):self.get(scope)
        scope.close()
        with self.assertRaisesRegex(ValueError,'closed'):self.get(scope)
    def test_changed_input_between_full_check_and_fingerprint_capture_closes_scope(self):
        full=A._full_check_native_selected_pilot;path=Path(self.f.selected['quality_review']['path'])
        def change_after_actual_full(*args,**kwargs):
            value=full(*args,**kwargs);path.write_text('{}');return value
        with A.NativeSelectionScope(enabled=True)as scope:
            with patch.object(A,'_full_check_native_selected_pilot',side_effect=change_after_actual_full),self.assertRaises(ValueError):self.get(scope)
            self.assertTrue(scope.facts()['closed']);self.assertEqual(scope.facts()['full_checks'],1);self.assertEqual(scope.facts()['reuses'],0)

    def test_later_pin_capture_changes_earlier_pin_before_capability_is_issued(self):
        original=A._selection_fingerprint;inputs=A._selection_inputs;first=[];changed=[]
        # 只修改本用例的临时质量文件，绝不修改真实维护源码。
        quality=self.f.selected['quality_review']['path']
        def ordered(*args):
            refs=inputs(*args);return sorted(refs,key=lambda ref:ref['path']!=quality)
        def race(pin,*,verify_sha):
            result=original(pin,verify_sha=verify_sha)
            if verify_sha:
                if not first:first.append(Path(pin['path']))
                elif not changed and Path(pin['path'])!=first[0]:
                    first[0].write_bytes(first[0].read_bytes()+b'\n');changed.append(True)
            return result
        with A.NativeSelectionScope(enabled=True)as scope,patch.object(A,'_selection_inputs',side_effect=ordered),patch.object(A,'_selection_fingerprint',side_effect=race):
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):self.get(scope)
            self.assertTrue(changed);self.assertTrue(scope.facts()['closed']);self.assertEqual(scope.facts()['reuses'],0)

    def test_detected_change_permanently_revokes_scope_even_after_restoring_mtime(self):
        path=Path(self.f.selected['quality_review']['path']);old=path.stat()
        with A.NativeSelectionScope(enabled=True)as scope:
            self.get(scope);os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):self.get(scope)
            os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
            self.assertTrue(scope.facts()['closed'])
            with self.assertRaisesRegex(ValueError,'closed'):self.get(scope)

    def test_failed_full_check_cannot_mint_capability(self):
        with A.NativeSelectionScope(enabled=True)as scope:
            with self.assertRaisesRegex(ValueError,'distinct version'):self.get(scope,contract=self.f.c)
            self.assertEqual(scope.facts()['full_checks'],0)
            value=self.get(scope);self.assertEqual(value['schema'],A.SCHEMA);self.assertEqual(scope.facts()['full_checks'],1)

if __name__=='__main__':unittest.main()
