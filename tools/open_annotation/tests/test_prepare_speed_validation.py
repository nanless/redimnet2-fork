"""验证准备器不会执行GPU，以及来源身份和新输出的必要约束。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import prepare_speed_validation as P
import contracts as C
from source_registry import source_id


class PrepareSpeedValidation(unittest.TestCase):
    def index(self, root, rows):
        root.mkdir()
        shard = root / 's.jsonl'
        shard.write_bytes(b''.join(C.canonical_bytes(x) for x in rows))
        p = root / 'index.json'
        p.write_bytes(C.canonical_bytes({'schema': 'redimnet2-shard-index/v1', 'source_count': len(rows),
            'shards': [{'id': 's', 'path': shard.name, 'sha256': C.sha_file(shard), 'count': len(rows)}]}))
        return {'path': str(p), 'sha256': C.sha_file(p)}

    def rows(self):
        return [{'source_id': source_id(ds, 's/a.wav'), 'dataset': ds, 'relative_path': 's/a.wav',
                 'roles': ['train_ptn']} for ds in sorted(P.DOMAINS)]

    def test_index_rejects_duplicate_role_conflict_and_fake_source_id(self):
        for kind in ('duplicate', 'role', 'id'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                rows = self.rows()
                if kind == 'duplicate': rows.append(copy.deepcopy(rows[0]))
                elif kind == 'role': rows[0]['roles'].append('evaluation:test')
                else: rows[0]['source_id'] = 'wrong-id'
                ref = self.index(Path(d).resolve()/'input', rows)
                with self.assertRaises(RuntimeError): P.check_index(ref, len(rows))

    def test_valid_index_does_not_access_audio_and_wrong_sha_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ref = self.index(Path(d).resolve()/'input', self.rows())
            self.assertEqual(len(P.check_index(ref, 3)), 3)
            ref['sha256'] = '0'*64
            with self.assertRaises(RuntimeError): P.check_index(ref, 3)

    def test_profiles_keep_legacy_undeclared_and_aligner_unmodified(self):
        self.assertIsNone(P.VARIANTS['legacy'])
        self.assertEqual(P.VARIANTS['asr-fp32-sdpa'], {'dtype':'float32', 'attention':'sdpa'})
        self.assertEqual(P.VARIANTS['asr-bf16-sdpa'], {'dtype':'bfloat16', 'attention':'sdpa'})

    def test_hash_and_parsed_bytes_are_the_same_read_not_a_previous_observation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()/'input.json';p.write_bytes(b'{"old":true}')
            old_sha=C.sha_file(p);p.write_bytes(b'{"new":true}')
            # 模拟独立hash调用取得较早的旧观测；解析不能消费未经本次校验的新字节。
            with patch.object(P,'sha_file',return_value=old_sha,create=True):
                with self.assertRaisesRegex(RuntimeError,'SHA'):
                    P.pinned({'path':str(p),'sha256':old_sha})

    def test_freeze_parses_only_bytes_with_the_supplied_parent_SHA(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();p=root/'parent.json';p.write_bytes(b'{"old":true}')
            old_sha=C.sha_file(p);p.write_bytes(b'{"new":true}')
            with patch.object(P.freezer,'helpers',return_value=C), patch.object(C,'sha_file',return_value=old_sha):
                with self.assertRaisesRegex(RuntimeError,'SHA'):
                    P.freezer.freeze(p,old_sha,root/'new-code','speed-test')
            self.assertFalse((root/'new-code').exists())

    def test_prepare_only_builds_commands_and_rejects_reused_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); (root/'inputs').mkdir(); base = root/'inputs/base.json'
            base.write_bytes(C.canonical_bytes({'runtime': {}}))
            spec = {'schema': 'annotation-speed-matrix-preparation/v1',
                    'base_contract': {'path': str(base), 'sha256': C.sha_file(base)},
                    'indices': {n: {'path': '/synthetic/'+n+'.json','sha256':'a'*64}
                                for n in ('screen','main','recovery')},
                    'target_uuids': ['GPU-00000000-0000-0000-0000-000000000001',
                                     'GPU-00000000-0000-0000-0000-000000000002'],
                    'device_lock_root': str(root/'shared-locks'), 'python': '/offline/python'}
            p = root/'spec.json';p.write_bytes(C.canonical_bytes(spec))
            def freeze(base, sha, output, version, asr_compute):
                output.mkdir(parents=True)
                return {'frozen_copy': str(output), 'bundle_sha256':'b'*64,
                        'contracts': {'pilot_validation': {'path': str(output/'contracts/pilot.json'), 'sha256':'c'*64}}}
            with patch.object(P, 'check_index', return_value={'s': {'same':True}}), patch.object(P.freezer, 'freeze', side_effect=freeze) as f:
                result = P.prepare(p, C.sha_file(p), root/'new-task')
                self.assertFalse(result['commands_executed']);self.assertFalse(result['cloud_admission'])
                self.assertEqual(f.call_count, 3)
                self.assertEqual(len(result['variants']), 3)
                for v in result['variants'].values():
                    self.assertIn('--full-verify', v['screen_argv'])
                    self.assertEqual(v['screen_argv'][0], '/offline/python')
                with self.assertRaises(RuntimeError): P.prepare(p, C.sha_file(p), root/'new-task')
                with self.assertRaisesRegex(RuntimeError, 'historical'):
                    P.prepare(p, C.sha_file(p), root/'inputs/new-task')
                alias = root/'lock-alias';alias.symlink_to(root/'inputs', target_is_directory=True)
                spec['device_lock_root'] = str(alias/'locks');p.write_bytes(C.canonical_bytes(spec))
                with self.assertRaisesRegex(RuntimeError, 'symlink'):
                    P.prepare(p, C.sha_file(p), root/'second-task')


if __name__ == '__main__': unittest.main()
