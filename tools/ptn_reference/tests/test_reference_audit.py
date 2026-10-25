"""标准库合成反例；不会导入或执行归档入口。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reference_audit',HERE/'audit_reference.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


class ReferenceAuditTests(unittest.TestCase):
    def fixture(self, root, content=b'raise RuntimeError("MUST_NOT_EXECUTE")\n'):
        path=root/'synthetic.py';path.write_bytes(content)
        return {'synthetic.py':{'bytes':len(content),'sha256':A.sha(content)}}

    def test_syntax_only_never_executes_source(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);result=A.audit_tree(root,self.fixture(root))
            self.assertTrue(result['file_set_and_syntax_verified'])
            self.assertFalse(result['archived_module_imported']);self.assertFalse(result['runtime_admission'])
            self.assertFalse(result['historical_execution_proven'])

    def test_tampered_missing_and_extra_credentials_refused(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);pins=self.fixture(root);path=root/'synthetic.py'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes/SHA'):A.audit_tree(root,pins)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):A.audit_tree(root,pins)
            self.fixture(root);(root/'parent_terminal_attestation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unexpected'):A.audit_tree(root,pins)

    def test_symlink_and_unsafe_filename_refused(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);data=b'pass\n';outside=root/'outside';outside.write_bytes(data)
            folder=root/'archive';folder.mkdir();(folder/'synthetic.py').symlink_to(outside)
            pins={'synthetic.py':{'bytes':len(data),'sha256':A.sha(data)}}
            with self.assertRaisesRegex(ValueError,'symlink'):A.audit_tree(folder,pins)
            (folder/'synthetic.py').unlink()
            with self.assertRaises(ValueError):A.audit_tree(folder,{'../outside':pins['synthetic.py']})

    def test_matching_SHA_cannot_hide_invalid_python_syntax(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)
            with self.assertRaises(SyntaxError):A.audit_tree(root,self.fixture(root,b'def broken(:\n'))

    def test_shell_is_syntax_checked_without_running(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);sentinel=root/'NEVER_CREATE';raw=('touch '+str(sentinel)+'\n').encode()
            (root/'synthetic.sh').write_bytes(raw)
            result=A.audit_tree(root,{'synthetic.sh':{'bytes':len(raw),'sha256':A.sha(raw)}})
            self.assertTrue(result['file_set_and_syntax_verified']);self.assertFalse(sentinel.exists())

    def test_real_reference_bytes_pass_but_runtime_request_is_always_refused(self):
        result=A.verify_reference();self.assertEqual(len(result['files']),10)
        self.assertTrue(result['reference_source_integrity_verified']);self.assertEqual(result['historical_exact_files'],9);self.assertEqual(result['public_derived_files'],1);self.assertFalse(result['runtime_admission'])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(A.main(['--require-runtime-admission']),2)
        self.assertIn('cannot grant runtime admission',output.getvalue())


if __name__=='__main__':unittest.main()
