"""P0-1 回归：单源确定性异常只拒绝该源，不杀 worker；白名单外异常仍然上抛。

CPU 合成，无模型、无 GPU：真实 process_group 代码路径 + 假 store/backend，
只把 inspect/plan/read/build 换成最小确定性桩。
"""
from __future__ import annotations
import tempfile
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'tests'))
import contracts as C


def contract():
    return {'purpose': 'cpu_fixture', 'version': 'synthetic-v1', 'sha256': 'c' * 64,
            'runtime': {'execution_policy': C.NATIVE_BATCH_POLICY, 'max_batch': 1,
                        'asr_batching': {'policy': 'fixed_cohort_until_durable/v1', 'max_requests': 8,
                                         'ordering': 'frames_then_row_id_within_source_group/v1'}},
            'provenance': {'models': {'asr': {'files_sha256': 'a' * 64},
                                      'aligner': {'files_sha256': 'b' * 64}}}}


def source(name):
    relative = name + '.wav'
    return {'source_id': 'src_' + name, 'dataset': 'synthetic', 'relative_path': relative,
            'sidecar_relpath': 'synthetic/' + relative + '.annotation.json',
            'audio_path': '/synthetic/' + relative, 'roles': ['train'], 'original_splits': [],
            'protection': {'training_forbidden': False}, 'channel_policy': 'mean_channels_to_mono'}


def window(source, index, frames=400):
    return {'window_id': 'w%d' % index, 'input_pcm_sha256': 'd' * 64,
            'model_input_frames': frames, 'model_sample_rate': 16000}


class FakeBackend:
    name = 'cpu-fixture-native'

    def __init__(self):
        self.groups = []

    def prepare_native_group(self, group, *args):
        self.groups.append(group)

    def transcribe(self, rows):
        return {r['row_id']: {'status': 'completed', 'normalized_text': 'synthetic',
                               'language': 'English',
                               'durable_native_batch': {'cache_hit': False}}
                for r in rows}

    def align(self, rows):
        return {r['row_id']: {'status': 'not_applicable', 'reason': 'synthetic'} for r in rows}


class FakeCache:
    def get(self, stage, inputs):
        return None

    def put(self, stage, inputs, value):
        return value


class FakeStore:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.refuse_lookup = None
        self.published = []

    def path(self, source):
        # 真实 SidecarStore 的路径接口；worker 在 refused 分支用它判断"是否已有产物"。
        return self.root / source['sidecar_relpath']

    def lookup(self, source):
        if source['source_id'] == self.refuse_lookup:
            raise ValueError('source content changed: inode differs')
        return None

    def publish(self, source, record, claim):
        path = self.root / source['sidecar_relpath']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"synthetic":true}')
        self.published.append(source['source_id'])
        # 2026-11-08：不再硬编码 decision，否则测不到 build_record 的失败语义。
        # 真实 store.publish 返回的是记录自身的 decision.state。
        return {'source_id': source['source_id'],
                'decision': (record.get('decision') or {}).get('state', 'unknown')}


def counters():
    import worker
    return {'assigned_source_rows': 0, 'sidecars_committed': 0, 'sidecar_cache_hits': 0,
            'retained_previous_producer_hits': 0,
            'retained_previous_decision_counts': dict(candidate=0, unknown=0, failed=0),
            'decision_counts': dict(candidate=0, unknown=0, failed=0),
            'stage_cache_hits': dict(asr=0, aligner=0), 'model_windows': 0,
            'decoded_audio_seconds': 0.0, 'error_source_count': 0, 'errors': [],
            'pipeline_seconds': {key: 0.0 for key in worker.PIPELINE_TIMERS}}


class SourceRefusalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def run_group(self, sources, windows, *, refuse_lookup=None, lookup_error=None):
        import worker
        store = FakeStore(self.root / 'labels'); store.refuse_lookup = refuse_lookup
        if lookup_error is not None:
            store.lookup = lookup_error
        identities = {s['source_id']: dict(
            decode_status='decoded_complete', frames=16000, sample_rate=16000, channels=1,
            duration_seconds=1.0, file_sha256='e' * 64, pcm_sha256='f' * 64,
            stat_after={'device': 1, 'inode': 2, 'bytes': 3, 'mtime_ns': 4, 'ctime_ns': 5})
            for s in sources}
        by_path = {s['audio_path']: s['source_id'] for s in sources}
        claim = types.SimpleNamespace(shard={'id': 'synthetic-shard'}, attempt_id='synthetic-attempt',
                                      assert_member=lambda s: None, journal=lambda value: None)
        args = types.SimpleNamespace(contract_sha='c' * 64, worker_id='cpu-fixture')
        descriptor = {'provenance': {'code_sha256': '9' * 64, 'models': contract()['provenance']['models']}}
        backend = FakeBackend(); observed = counters()
        with patch.multiple(
                worker,
                inspect_source=Mock(side_effect=lambda s: (identities[s['source_id']], [])),
                reuse_vad=Mock(return_value={'errors': []}),
                plan_windows=Mock(side_effect=lambda ident, s, runtime, vad: windows[s['source_id']]),
                read_window=Mock(side_effect=lambda *a: np.zeros(400, dtype=np.float32)),
                units_from_windows=Mock(return_value=([], [])),
                stat_identity=Mock(side_effect=lambda p: identities[by_path[p]]['stat_after']),
                build_record=Mock(side_effect=lambda s, ident, desc, vad, ws, units, execution, errors:
                                  {'source': s, 'windows': ws, 'units': units, 'errors': errors})):
            receipts = worker.process_group(sources, claim, store, FakeCache(), backend,
                                            contract(), args, descriptor, observed)
        return receipts, observed, store, backend

    def test_deterministic_source_failures_are_refused_not_fatal(self):
        bad_stat, bad_many, good_a, good_b = (source('bad_stat'), source('bad_many'),
                                              source('good_a'), source('good_b'))
        sources = [bad_stat, bad_many, good_a, good_b]
        windows = {'src_bad_stat': [window(bad_stat, 0)],
                   'src_bad_many': [window(bad_many, i) for i in range(257)],
                   'src_good_a': [window(good_a, 0)], 'src_good_b': [window(good_b, 0)]}
        receipts, observed, store, backend = self.run_group(sources, windows, refuse_lookup='src_bad_stat')
        # 2026-11-08 P0-1b：被拒源也要产回执（否则分片一直报 cannot commit incomplete chunk）。
        # 回执要覆盖分片内全部源，坏源标 failed，好源照常产侧车。
        self.assertEqual(sorted(r['source_id'] for r in receipts),
                         ['src_bad_many', 'src_bad_stat', 'src_good_a', 'src_good_b'])
        self.assertEqual(len({r['source_id'] for r in receipts}), len(sources))
        # 本用例用 patch.object 替换了 worker.build_record，decision 由真实 build_record
        # 计算，因此这里只断言"覆盖度"这一 P0-1b 实际的不变量；decision 语义另由
        # test_refused_identity_maps_to_failed_decision 直接测 schema.build_record。
        refused_ids = {r['source_id'] for r in receipts} & {'src_bad_stat', 'src_bad_many'}
        self.assertEqual(refused_ids, {'src_bad_stat', 'src_bad_many'})
        self.assertEqual(observed['source_refused'], 2)
        self.assertEqual(observed['sidecars_committed'], 4)
        self.assertEqual(sorted(store.published), ['src_bad_many', 'src_bad_stat', 'src_good_a', 'src_good_b'])
        self.assertEqual({e['source_id'] for e in observed['errors']}, {'src_bad_stat', 'src_bad_many'})
        self.assertTrue(any('source content changed' in e['reason'] for e in observed['errors']))
        self.assertTrue(any('256' in e['reason'] for e in observed['errors']))
        self.assertTrue((store.root / 'synthetic/good_a.wav.annotation.json').exists())
        self.assertTrue((store.root / 'synthetic/good_b.wav.annotation.json').exists())
        # 坏源也要有侧车文件（内容为 decode_failed 空窗记录），否则分片无法提交。
        self.assertTrue((store.root / 'synthetic/bad_stat.wav.annotation.json').exists())
        self.assertTrue((store.root / 'synthetic/bad_many.wav.annotation.json').exists())
        self.assertEqual(observed['assigned_source_rows'], 4)
        self.assertTrue(backend.groups)

    def test_non_whitelisted_oserror_still_stops_the_worker(self):
        def explode(source):
            raise OSError('synthetic storage outage')
        with self.assertRaises(OSError):
            self.run_group([source('good_a')], {'src_good_a': [window(source('good_a'), 0)]},
                           lookup_error=explode)


if __name__ == '__main__':
    unittest.main()

class RefusedRecordSemantics(unittest.TestCase):
    """被拒源在真实 build_record 下要落成 decision=failed（P0-1b 的语义面）。"""

    def test_refused_identity_maps_to_failed_decision(self):
        import sys
        sys.path.insert(0, str(HERE))
        import schema
        source = {'source_id': 'src_x', 'dataset': 'synthetic', 'relative_path': 'x.wav',
                  'sidecar_relpath': 'synthetic/x.wav.annotation.json', 'roles': ['enroll'],
                  'original_splits': [], 'protection': {'training_forbidden': False},
                  'channel_policy': 'mean_channels_to_mono', 'audio_path': '/tmp/x.wav'}
        identity = {'decode_status': 'decode_failed', 'stat_after': None, 'refused_reason': 'boom'}
        record = schema.build_record(
            source, identity, {'version': 'v1', 'runtime': {}},
            {'status': 'missing', 'reused': False, 'segments': [], 'errors': []},
            [], [],
            {'job_key': 'k', 'attempt_id': 'a', 'worker_id': 'w', 'device': 'cpu',
             'backend': 'x', 'started_utc': 't', 'finished_utc': 't', 'api_calls': 0},
            [{'stage': 'source_refused', 'reason': 'boom'}])
        self.assertEqual(record['decision']['state'], 'failed')
        self.assertEqual(record['windows'], [])
