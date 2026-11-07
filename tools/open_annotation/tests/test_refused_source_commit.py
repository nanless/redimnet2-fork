"""P0-1b 回归：含坏源（甚至全坏）的分片必须能 finish() 提交落盘。

CPU 合成，无模型、无 GPU：真实 WorkQueue + 真实 SidecarStore + 真实 build_record，
只有 inspect/VAD/plan/read 用确定性桩。覆盖三件事：
1) 1 个坏源 + N 个好源 → finish 成功、chunk commit 落盘、坏源 receipt=failed、好源侧车正常；
2) 全坏分片 → 同样成功提交，不无限重试、不卡后面的片；
3) 拒绝原因本身就是"这里已有不同/损坏产物"时 → 绝不覆盖，退回不带侧车的 failed receipt。
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'tests'))
import contracts as C
import schema as S
from sidecar import SidecarStore
from work_queue import WorkQueue, decode_completion_slot

CONTRACT_SHA = 'c' * 64


def contract():
    return {'purpose': 'cpu_fixture', 'version': 'synthetic-v1', 'sha256': CONTRACT_SHA,
            'runtime': {'execution_policy': C.NATIVE_BATCH_POLICY, 'max_batch': 1,
                        'asr_batching': {'policy': 'fixed_cohort_until_durable/v1', 'max_requests': 8,
                                         'ordering': 'frames_then_row_id_within_source_group/v1'}},
            'models': {'asr': {'path': '/synthetic/asr', 'files': {'w.bin': {'sha256': 'a' * 64, 'bytes': 1}}},
                       'aligner': {'path': '/synthetic/aligner', 'files': {'w.bin': {'sha256': 'b' * 64, 'bytes': 1}}}}}


def bound_source(name, root):
    """合法 source_id + 真实音频文件（只为可以被 stat 身份校验）。"""
    relative = name + '.wav'
    audio = Path(root) / 'audio' / relative
    audio.parent.mkdir(parents=True, exist_ok=True); audio.write_bytes(b'\0' * 64)
    return {'schema': 'redimnet2-source/v1',
            'source_id': 'src_' + hashlib.sha256(('synthetic' + '\0' + relative).encode()).hexdigest(),
            'dataset': 'synthetic', 'relative_path': relative,
            'sidecar_relpath': 'synthetic/' + relative + '.annotation.json',
            'audio_path': str(audio), 'roles': ['train'], 'original_splits': [],
            'protection': {'training_forbidden': False}, 'channel_policy': 'mean_channels_to_mono'}


def decoded_identity(stat):
    return {'decode_status': 'decoded_complete', 'file_sha256': 'e' * 64, 'pcm_sha256': 'f' * 64,
            'sample_rate': 16000, 'channels': 1, 'frames': 16000, 'bytes': 64,
            'duration_seconds': 1.0, 'stat_before': stat, 'stat_after': stat,
            'pcm_hash_policy': S.PCM_HASH_POLICY}


class FakeBackend:
    name = 'cpu-fixture-native'

    def prepare_native_group(self, group, *args):
        raise AssertionError('empty-window groups must not prepare a native cohort')


class FakeCache:
    def get(self, stage, inputs):
        return None

    def put(self, stage, inputs, value):
        return value


def counters():
    import worker
    return {'assigned_source_rows': 0, 'sidecars_committed': 0, 'sidecar_cache_hits': 0,
            'retained_previous_producer_hits': 0,
            'retained_previous_decision_counts': dict(candidate=0, unknown=0, failed=0),
            'decision_counts': dict(candidate=0, unknown=0, failed=0),
            'stage_cache_hits': dict(asr=0, aligner=0), 'model_windows': 0,
            'decoded_audio_seconds': 0.0, 'error_source_count': 0, 'errors': [],
            'pipeline_seconds': {key: 0.0 for key in worker.PIPELINE_TIMERS}}


class RefusedSourceCommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.claims = []

    def tearDown(self):
        for claim in self.claims:
            if claim is not None: claim.close()

    def queue_for(self, shards):
        descriptors = []
        for index, sources in enumerate(shards):
            path = self.root / ('shard-%03d.jsonl' % index)
            path.write_bytes(b''.join(C.canonical_bytes(source) for source in sources))
            descriptors.append({'id': 'chunk-%03d' % index, 'path': path.name,
                                'count': len(sources), 'sha256': C.sha_file(path)})
        index = self.root / 'index.json'
        index.write_bytes(C.canonical_bytes({'schema': 'redimnet2-shard-index/v1',
                                             'source_count': sum(len(s) for s in shards),
                                             'shards': descriptors}))
        return WorkQueue(index, C.sha_file(index), self.root / 'queue', CONTRACT_SHA)

    def patched(self, sources, identities, refuse_audit=()):
        import worker
        by_path = {s['audio_path']: s['source_id'] for s in sources}
        def inspect(row):
            if row['source_id'] in refuse_audit:
                raise ValueError('synthetic source decode failure')
            return identities[row['source_id']], []
        return patch.multiple(
            worker,
            inspect_source=Mock(side_effect=inspect),
            reuse_vad=Mock(return_value={'status': 'missing', 'reused': False, 'segments': [], 'errors': []}),
            plan_windows=Mock(return_value=[]),
            read_window=Mock(side_effect=lambda *a: np.zeros(400, dtype=np.float32)),
            units_from_windows=Mock(return_value=([], [])),
            stat_identity=Mock(side_effect=lambda p: identities[by_path[p]]['stat_after']))

    def run_group(self, queue, sources, identities, *, refuse_audit=(), claim=None):
        import worker
        claim = claim or queue.claim_next('cpu-fixture')
        self.claims.append(claim)
        store = SidecarStore(self.root / 'labels', contract(), CONTRACT_SHA)
        args = types.SimpleNamespace(contract_sha=CONTRACT_SHA, worker_id='cpu-fixture')
        descriptor = worker._descriptor(contract(), CONTRACT_SHA, '9' * 64, {'synthetic_fixture': True})
        observed = counters()
        with self.patched(sources, identities, refuse_audit):
            receipts = worker.process_group(sources, claim, store, FakeCache(), FakeBackend(),
                                            contract(), args, descriptor, observed)
        return claim, store, receipts, observed

    def test_mixed_refused_source_chunk_commits_with_failed_receipt(self):
        bad, good = bound_source('bad', self.root), bound_source('good', self.root)
        sources = [bad, good]
        identities = {s['source_id']: decoded_identity(C.stat_identity(s['audio_path'])) for s in sources}
        queue = self.queue_for([sources])
        claim, store, receipts, observed = self.run_group(queue, sources, identities,
                                                          refuse_audit={bad['source_id']})
        self.assertEqual([r['source_id'] for r in receipts], [bad['source_id'], good['source_id']])
        self.assertEqual(observed['source_refused'], 1)
        claim.finish(receipts); claim.close()

        path = queue.chunk_receipt_path('chunk-000')
        self.assertTrue(path.exists(), 'finish() 必须能提交含坏源的分片')
        value = json.loads(path.read_text())
        self.assertEqual(value['source_count'], 2)
        by_id = {r['source_id']: r for r in value['sources']}
        self.assertEqual(set(by_id), {bad['source_id'], good['source_id']})
        failed = by_id[bad['source_id']]
        self.assertEqual(failed['decision'], 'failed')
        self.assertEqual(failed['job_key'], C.job_key(bad, CONTRACT_SHA))
        self.assertEqual(by_id[good['source_id']]['decision'], 'unknown')
        self.assertEqual(by_id[good['source_id']]['job_key'], C.job_key(good, CONTRACT_SHA))

        record = json.loads(store.path(bad).read_text()); S.assert_valid_record(record)
        self.assertEqual(record['decision']['state'], 'failed')
        self.assertEqual(record['source_identity']['decode_status'], 'decode_failed')
        self.assertEqual(record['windows'], [])
        self.assertEqual([e['stage'] for e in record['errors']], ['source_refused'])
        good_record = json.loads(store.path(good).read_text()); S.assert_valid_record(good_record)
        self.assertEqual(good_record['source_identity']['decode_status'], 'decoded_complete')

        slot = decode_completion_slot(queue.path('completion_map.bin').read_bytes()[:1024])
        self.assertEqual(slot['source_count'], 2)
        self.assertEqual(slot['decision_counts'], {'candidate': 0, 'unknown': 1, 'failed': 1})
        self.assertTrue(queue.is_complete())
        self.assertIsNone(queue.claim_next('retry-worker'))
        self.assertIsNotNone(queue.receipt('chunk-000'))

    def test_fully_refused_shard_commits_and_worker_moves_to_next_shard(self):
        bad_a, bad_b = bound_source('bad_a', self.root), bound_source('bad_b', self.root)
        good = bound_source('good', self.root)
        identities = {s['source_id']: decoded_identity(C.stat_identity(s['audio_path']))
                      for s in (bad_a, bad_b, good)}
        queue = self.queue_for([[bad_a, bad_b], [good]])
        claim, store, receipts, observed = self.run_group(
            queue, [bad_a, bad_b], identities, refuse_audit={bad_a['source_id'], bad_b['source_id']})
        self.assertEqual([r['decision'] for r in receipts], ['failed', 'failed'])
        claim.finish(receipts); claim.close()

        nxt = queue.claim_next('cpu-fixture'); self.claims.append(nxt)
        self.assertEqual(nxt.shard['id'], 'chunk-001', '全坏片提交后必须继续领取下一片')
        _, _, receipts2, _ = self.run_group(queue, [good], identities, claim=nxt)
        nxt.finish(receipts2); nxt.close()
        self.assertTrue(queue.is_complete())
        self.assertIsNone(queue.claim_next('retry-worker'))

        value = json.loads(queue.chunk_receipt_path('chunk-000').read_text())
        self.assertEqual([r['decision'] for r in value['sources']], ['failed', 'failed'])
        slot0 = decode_completion_slot(queue.path('completion_map.bin').read_bytes()[:1024])
        self.assertEqual(slot0['decision_counts'], {'candidate': 0, 'unknown': 0, 'failed': 2})
        reopened = WorkQueue(queue.index_path, queue.index_sha, queue.root, CONTRACT_SHA)
        self.assertTrue(reopened.is_complete())
        self.assertIsNone(reopened.claim_next('restarted-worker'))

    def test_refused_source_never_overwrites_existing_sidecar(self):
        bad, good = bound_source('bad', self.root), bound_source('good', self.root)
        sources = [bad, good]
        identities = {s['source_id']: decoded_identity(C.stat_identity(s['audio_path'])) for s in sources}
        queue = self.queue_for([sources])
        corrupt = self.root / 'labels' / contract()['version'] / bad['sidecar_relpath']
        corrupt.parent.mkdir(parents=True, exist_ok=True); corrupt.write_bytes(b'{corrupt-sidecar')
        claim, store, receipts, observed = self.run_group(queue, sources, identities)
        self.assertEqual(observed['source_refused'], 1)
        self.assertEqual(receipts[0]['source_id'], bad['source_id'])
        self.assertEqual(receipts[0]['decision'], 'failed')
        self.assertEqual(receipts[0]['job_key'], C.job_key(bad, CONTRACT_SHA))
        self.assertNotIn('sidecar_relpath', receipts[0])
        self.assertTrue(receipts[0]['reason'])
        claim.finish(receipts); claim.close()
        self.assertEqual(corrupt.read_bytes(), b'{corrupt-sidecar', '既有产物不得被覆盖')
        self.assertTrue(queue.is_complete())


if __name__ == '__main__':
    unittest.main()
