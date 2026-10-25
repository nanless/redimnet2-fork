"""新标注入口的有限独立路径/诊断保护反例，全部使用合成来源。"""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/open_annotation'))
import gpu_launch
import word_adapter


class AnnotationReview(unittest.TestCase):
    def test_task_contains_code_but_alias_escape_and_ancestor_lock_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder).resolve()
            base = home / 'task'
            code = base / 'frozen-code'
            code.mkdir(parents=True)
            outside = home / 'outside'
            outside.mkdir()
            args = SimpleNamespace(run_root=str(base), device_lock_root=str(home / 'shared-locks'),
                                   output=str(base / 'run/resources'), queue_root=str(base / 'run/queue'),
                                   output_root=str(base / 'run/labels'))
            self.assertEqual(gpu_launch.validate_output_paths(args, here=code)[0], base)
            (base / 'escape').symlink_to(outside, target_is_directory=True)
            for key, target in (('output_root', base / 'escape/labels'),
                                ('device_lock_root', home), ('queue_root', code / 'queue')):
                changed = copy.copy(args)
                setattr(changed, key, str(target))
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    gpu_launch.validate_output_paths(changed, here=code)
            self.assertEqual(list(outside.iterdir()), [])

    def test_synthetic_protection_id_and_both_paths_precede_media(self):
        case = {'case_id': 'synthetic-diagnostic', 'source_id': 'src_private_fixture',
                'path': '/synthetic/original.wav', 'normalized_alias_path': '/synthetic/alias.wav'}
        reference = {'cases': [case]}
        context = {'diagnostic_protection': reference, 'diagnostic_protection_sha256': 'a' * 64,
                   'diagnostic_protection_content_sha256': word_adapter.digest(reference)}
        for source in ({'source_id': case['source_id'], 'audio_path': '/synthetic/other.wav'},
                       {'source_id': 'src_other', 'audio_path': case['path']},
                       {'source_id': 'src_other', 'audio_path': case['normalized_alias_path']}):
            proposal = word_adapter.propose({'source': source, 'decision': {'state': 'candidate'}}, context)
            self.assertEqual(proposal['status'], 'diagnostic_only')
            self.assertEqual(proposal['families'], [])
            # 即使外部错误加入family，导出仍在任何媒体读取前再次拒绝。
            proposal['families'] = [{'family_id': 'synthetic', 'records_pending_media_binding': []}]
            with patch.object(word_adapter, 'verify_media', side_effect=AssertionError('不得读取媒体')):
                records, refused = word_adapter.engineer_export([proposal], {}, context)
            self.assertEqual(records, [])
            self.assertEqual(refused[0]['reason'], 'frozen_benchmark_diagnostic_only')


if __name__ == '__main__':
    unittest.main()
