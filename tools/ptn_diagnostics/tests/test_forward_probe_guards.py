"""前向诊断在接触 CUDA 前必须拒绝的输入；不模拟 GPU 成功。"""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'ptn_forward_probe.py'
spec = importlib.util.spec_from_file_location('ptn_probe_guard_test', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeGuards(unittest.TestCase):
    def test_namespace_only_accepts_single_frozen_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve()
            expected = source / 'redimnet2/layers'
            expected.mkdir(parents=True)
            value = probe.import_identity('redimnet2.layers', SimpleNamespace(__file__=None, __path__=[str(expected)]), source)
            self.assertEqual(value['namespace_directory'], str(expected))
            for paths in ([], [str(expected), '/another'], ['/another']):
                with self.assertRaisesRegex(ValueError, '非冻结目录'):
                    probe.import_identity('redimnet2.layers', SimpleNamespace(__file__=None, __path__=paths), source)

    def test_truncated_screen_digest_rejected_before_read(self):
        with patch.object(probe, 'sha', side_effect=AssertionError('不得读取')):
            with self.assertRaisesRegex(ValueError, '64位'):
                probe.verify('/missing', 'a' * 63)

    def test_hash_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'input'
            f.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError, '身份不符'):
                probe.verify(f, '0' * 64)

    def test_change_while_hashing_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'input'
            f.write_bytes(b'original')
            def changed(path):
                path.write_bytes(b'changed after digest')
                return 'a' * 64
            with patch.object(probe, 'sha', side_effect=changed):
                with self.assertRaisesRegex(ValueError, '读取期间'):
                    probe.verify(f, 'a' * 64)

    def test_leaf_symlink_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'input'
            f.write_bytes(b'original')
            link = Path(tmp) / 'link'
            link.symlink_to(f)
            with patch.object(probe, 'sha', side_effect=AssertionError('不得读取')):
                with self.assertRaisesRegex(ValueError, '身份不符'):
                    probe.verify(link, 'a' * 64)

    def test_existing_output_refuses_before_source_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ['probe', '--output', tmp, '--lock-root', '/unused',
                    '--expected-uuids', 'a', 'b', '--case', 'baseline-amp']
            for name in ('source', 'manifest', 'manifest-sha', 'config', 'config-sha',
                         'checkpoint', 'checkpoint-sha', 'wave', 'wave-sha'):
                args.extend(['--' + name, '/must-not-read'])
            with patch('sys.argv', args), patch.object(probe, 'verify', side_effect=AssertionError('不得读取')):
                with self.assertRaisesRegex(ValueError, '全新输出'):
                    probe.main()


if __name__ == '__main__':
    unittest.main()
