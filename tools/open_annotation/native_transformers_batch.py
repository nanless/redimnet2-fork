"""原生Transformers固定批次的文件事务，不导入模型、不创建数据库。

先固定一个有限输入计划，再按原顺序保存整批原生JSON，最后才给调用者
逐行结果。恢复不能删除未提交批次中的行后重新组成不同批次。
真实多行generate、padding与EOS观测由明确的推理回调负责；本模块不把
CPU合成测试解释成模型、CUDA、字界或正式数据质量通过。
"""
from __future__ import annotations
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time
import traceback

if __package__:
    from .contracts import atomic_new, canonical_bytes, digest, need, utcnow
    from .readonly_progress import strict_json
else:
    from contracts import atomic_new, canonical_bytes, digest, need, utcnow
    from readonly_progress import strict_json

POLICY = 'asr_fixed_cohort_fa_single/v1'
MAX_PLAN_ROWS = 256
MAX_PACKET_BYTES = 64 * 1024**2
REQUEST_FIELDS = ('row_id', 'source_id', 'window_id', 'source_record_sha256',
                  'input_pcm_sha256', 'model_input_frames', 'model_sample_rate')


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def physical(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and all(not x.is_symlink() for x in (p, *p.parents)),
         'physical absolute batch path required')
    return p


def read_json(path):
    path = physical(path); before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_PACKET_BYTES, 'bounded regular batch JSON required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    try:
        actual = os.fstat(fd); need(stamp(actual) == stamp(before), 'batch file changed before read')
        with os.fdopen(fd, 'rb', closefd=False) as stream: raw = stream.read(MAX_PACKET_BYTES + 1)
        need(len(raw) == actual.st_size and stamp(actual) == stamp(os.fstat(fd)) == stamp(path.stat()),
             'batch file changed during read')
    finally: os.close(fd)
    return strict_json(raw), hashlib.sha256(raw).hexdigest()


def request_identity(row):
    need(isinstance(row, dict) and all(k in row for k in REQUEST_FIELDS), 'complete native batch request fields required')
    value = {k: row[k] for k in REQUEST_FIELDS}
    need(all(isinstance(value[k], str) and value[k] and '\x00' not in value[k] for k in ('row_id','source_id','window_id')),
         'explicit source/window/row IDs required')
    need(value['row_id'] == value['source_id'] + ':' + value['window_id'], 'row ID must bind source and window')
    need(is_sha(value['source_record_sha256']) and is_sha(value['input_pcm_sha256']), 'source and PCM SHA required')
    need(type(value['model_input_frames']) is int and 0 < value['model_input_frames'] <= 480000
         and type(value['model_sample_rate']) is int and value['model_sample_rate'] == 16000,
         'one native ASR window must be 16k mono and at most 30 seconds')
    source = row.get('context', {}).get('source')
    if source is not None:
        need(source.get('source_id') == value['source_id'] and digest(source) == value['source_record_sha256'],
             'request does not bind complete actual source record')
    return value


def verify_pcm(row):
    """仅对本次要推理的原输入核实际PCM；缓存命中可不再提供音频。"""
    import numpy as np
    wave = row.get('audio')
    need(isinstance(wave, np.ndarray) and wave.ndim == 1 and wave.dtype == np.dtype('float32'),
         'native batch requires actual mono float32 numpy PCM')
    need(len(wave) == row['model_input_frames'] and np.isfinite(wave).all(), 'actual PCM length/nonfinite mismatch')
    raw = np.ascontiguousarray(wave, dtype='<f4').tobytes()
    need(hashlib.sha256(raw).hexdigest() == row['input_pcm_sha256'], 'actual model PCM SHA differs')


def snapshot(value):
    """异常证据保留非JSON类型的明确描述，不把未知值冒充有效模型输出。"""
    if value is None or type(value) in (str, int, bool): return value
    if type(value) is float:
        import math
        return value if math.isfinite(value) else {'nonfinite_native_float': repr(value)}
    if isinstance(value, dict): return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [snapshot(v) for v in value]
    return {'unsupported_native_type': type(value).__module__ + '.' + type(value).__qualname__, 'repr': repr(value)}


class NativeTransformersBatchStore:
    """每个明确claim／来源组一个稳定目录；严禁以剩余cache-miss列表另起根。"""
    def __init__(self, root, producer_sha256, *, synthetic_fixture, contract_sha256,
                 model_sha256, max_batch_rows, policy=POLICY):
        need(all(is_sha(v) for v in (producer_sha256, contract_sha256, model_sha256)), 'producer/contract/model SHA required')
        need(type(synthetic_fixture) is bool and type(max_batch_rows) is int and max_batch_rows in (1,4,8),
             'explicit synthetic flag and native batch 1/4/8 required')
        need(policy == POLICY, 'explicit new ASR policy required; single-window policy cannot be relabeled')
        self.root = physical(root); self.max_batch_rows = max_batch_rows
        self.identity = dict(schema='annotation-native-transformers-file-store/v1', producer_sha256=producer_sha256,
            contract_sha256=contract_sha256, model_sha256=model_sha256, synthetic_fixture=synthetic_fixture,
            policy=policy, max_batch_rows=max_batch_rows, alignment=dict(dtype='float32', attention='eager', max_batch_rows=1),
            database_created=False, model_API_calls=0, release_training_labels=False)
        if self.root.exists():
            need(self.root.is_dir() and (self.root/'identity.json').is_file(), 'unknown existing batch directory; preserve it')
        else:
            self.root.mkdir(parents=True)
            atomic_new(physical(self.root/'identity.json'), self.identity)
        need(read_json(self.root/'identity.json')[0] == self.identity, 'batch store belongs to another producer/contract/profile')
        self.metrics = {}

    @contextlib.contextmanager
    def locked(self):
        path = physical(self.root/'.batch.lock')
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'batch lock must be one regular inode')
            fcntl.flock(fd, fcntl.LOCK_EX)
            after = path.stat(); need((before.st_dev,before.st_ino) == (after.st_dev,after.st_ino), 'batch lock inode changed')
            need(read_json(self.root/'identity.json')[0] == self.identity, 'batch identity changed')
            yield
        finally: os.close(fd)

    def _plan(self, identities):
        path = self.root/'plan.json'
        if path.exists():
            plan, plan_sha = read_json(path)
            need(plan.get('schema') == 'annotation-native-transformers-plan/v1' and plan.get('store_identity') == self.identity,
                 'stored fixed plan identity differs')
            expected = self._make_plan(plan['requests'])
            need(plan == expected, 'stored cohort plan is incomplete or changed')
            by_id = {v['row_id']: v for v in plan['requests']}
            need(all(by_id.get(v['row_id']) == v for v in identities), 'request is outside fixed plan or changed input; no regrouping')
        else:
            plan = self._make_plan(identities); plan_sha = atomic_new(physical(path), plan)
        return plan, plan_sha

    def _make_plan(self, identities):
        need(isinstance(identities,list) and 1 <= len(identities) <= MAX_PLAN_ROWS, 'fixed plan must contain 1..256 rows')
        need(len({v['row_id'] for v in identities}) == len(identities), 'duplicate planned row')
        need(all(request_identity(v) == v for v in identities), 'stored request fields differ')
        common = dict(store_identity=self.identity, requests=identities)
        cohorts = []
        for start in range(0,len(identities),self.max_batch_rows):
            rows = identities[start:start+self.max_batch_rows]
            intent = dict(schema='annotation-native-transformers-cohort/v1', store_identity_sha256=digest(self.identity),
                          requests=rows, original_plan_requests_sha256=digest(identities))
            cohorts.append(dict(cohort_sha256=digest(intent), intent=intent))
        return dict(schema='annotation-native-transformers-plan/v1', **common, cohorts=cohorts)

    def _validate_outputs(self, outputs, identities):
        ids = [r['row_id'] for r in identities]
        need(isinstance(outputs,dict) and set(outputs) == set(ids), 'whole native cohort result coverage differs')
        for position, row_id in enumerate(ids):
            value = outputs[row_id]; need(isinstance(value,dict) and isinstance(value.get('native'),dict), 'native row JSON required')
            need(value.get('status') in {'completed','truncated','unknown','failed'} and isinstance(value.get('raw_text'),str),
                 'native row status/text missing')
            native = value['native']
            need(type(native.get('actual_generate_rows')) is int and native['actual_generate_rows'] == len(ids)
                 and type(native.get('actual_generate_row_index')) is int and native['actual_generate_row_index'] == position,
                 'actual native generate width/order differs from fixed cohort')
        canonical_bytes(outputs)  # 非有限数值／非JSON对象不得成为有效缓存。

    def _packet(self, cohort, plan_sha):
        path = self.root/'batches'/(cohort['cohort_sha256']+'.json')
        if not path.exists(): return None
        packet, sha = read_json(path); content = packet.get('content')
        need(packet.get('schema') == 'annotation-native-transformers-atomic-result/v1' and isinstance(content,dict)
             and packet.get('content_sha256') == digest(content), 'whole native batch packet digest differs')
        need(content.get('store_identity') == self.identity and content.get('plan_sha256') == plan_sha
             and content.get('cohort') == cohort, 'native batch producer/plan/cohort changed')
        self._validate_outputs(content['outputs'],cohort['intent']['requests'])
        return content['outputs'], dict(plan_sha256=plan_sha,cohort_sha256=cohort['cohort_sha256'],packet_path=str(path),packet_sha256=sha)

    def _failure(self, cohort, plan_sha, outputs, *, committed):
        atomic_new(physical(self.root/'attempts'/(secrets.token_hex(16)+'.json')), dict(schema='annotation-native-transformers-failed-attempt/v1',
            observed_utc=utcnow(),store_identity=self.identity,plan_sha256=plan_sha,cohort=cohort,
            complete_batch_committed_before_error=committed,traceback=traceback.format_exc(),native_outputs=snapshot(outputs),
            native_callback_evidence=snapshot(getattr(sys.exc_info()[1],'native_batch_evidence',None)),
            no_per_position_retry=True, database_created=False))

    def run_batch(self, rows, *, infer_batch, after_durable_commit=None):
        """返回逐row副本前，必须完成整批落盘及读回；异常整批保留并向外抛。

        首次提供此固定来源组全部未发布窗口。恢复可请求已提交批次的子集，
        但未提交批次必须提供原cohort全部行与原顺序，不按剩余行重排。
        """
        need(isinstance(rows,list) and 1 <= len(rows) <= MAX_PLAN_ROWS and callable(infer_batch), 'finite rows and explicit native callback required')
        identities = [request_identity(r) for r in rows]
        need(len({r['row_id'] for r in rows}) == len(rows), 'duplicate requested row')
        verified = set()
        for row in rows:
            if row.get('audio') is not None:
                verify_pcm(row); verified.add(row['row_id'])
        by_id = {r['row_id']:r for r in rows}; outputs = {}; new_rows = 0; new_batches = 0; started = time.monotonic()
        with self.locked():
            plan, plan_sha = self._plan(identities)
            for cohort in plan['cohorts']:
                expected = cohort['intent']['requests']; ids = [r['row_id'] for r in expected]
                selected = [v['row_id'] for v in identities if v['row_id'] in set(ids)]
                if not selected: continue
                saved = self._packet(cohort,plan_sha)
                cache_hit = saved is not None
                if saved is None:
                    need(selected == ids, 'uncommitted cohort requires all original rows in original order; no partial retry')
                    native = None; committed = False
                    try:
                        batch = [by_id[k] for k in ids]
                        need(all(row['row_id'] in verified for row in batch), 'uncommitted cohort needs every original actual PCM')
                        native = infer_batch(batch); self._validate_outputs(native,expected)
                        content = dict(store_identity=self.identity,plan_sha256=plan_sha,cohort=cohort,outputs=native)
                        packet = dict(schema='annotation-native-transformers-atomic-result/v1',content_sha256=digest(content),content=content)
                        need(len(canonical_bytes(packet)) <= MAX_PACKET_BYTES, 'native batch JSON exceeds bounded packet size')
                        atomic_new(physical(self.root/'batches'/(cohort['cohort_sha256']+'.json')),packet)
                        committed = True; new_rows += len(ids); new_batches += 1
                        saved = self._packet(cohort,plan_sha)
                        if after_durable_commit is not None: after_durable_commit(cohort['cohort_sha256'])
                    except BaseException:
                        self._failure(cohort,plan_sha,native,committed=committed); raise
                data, ref = saved
                for position, row_id in enumerate(ids):
                    if row_id in by_id:
                        outputs[row_id] = copy.deepcopy(data[row_id])
                        outputs[row_id]['durable_native_batch'] = dict(ref,position=position,cache_hit=cache_hit)
            need(set(outputs) == set(by_id), 'durable native batch union incomplete')
        self.metrics = dict(requested_rows=len(rows),new_model_rows=new_rows,new_batches=new_batches,
            stage_cache_hits=len(rows)-new_rows,stage_wall_seconds=time.monotonic()-started,
            provided_PCM_verified_rows=len(verified),
            model_API_calls=0,database_created=False,synthetic_fixture=self.identity['synthetic_fixture'],formal_admission=False)
        return outputs
