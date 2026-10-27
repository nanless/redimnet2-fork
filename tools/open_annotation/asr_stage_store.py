"""新ASR阶段的事务存储：整批结果持久化后，才允许逐请求读取。

本目录不兼容旧StageCache，也不修改旧结果。每个明确的工作分片使用独立
store；同一store的调用串行，模型内部仍可批处理。SQLite事务用于一次预约
完整批次以及一次提交完整结果，避免半批发布后重新计算已持久化的结果。
"""
from __future__ import annotations
import contextlib
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import traceback
import uuid
from urllib.parse import quote


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def physical(path):
    p = Path(path).absolute()
    require('..' not in p.parts and all(not v.is_symlink() for v in (p, *p.parents)),
            'physical path without parent traversal required')
    return p


class ASRStageStore:
    """只保存新producer的批次；不会修复、覆盖或重贴旧缓存。"""
    def __init__(self, root, producer_sha256, *, synthetic_fixture):
        require(isinstance(producer_sha256, str) and len(producer_sha256) == 64 and
                all(c in '0123456789abcdef' for c in producer_sha256), 'producer SHA256 required')
        require(type(synthetic_fixture) is bool, 'explicit synthetic flag required')
        self.root = physical(root)
        identity = {'schema': 'annotation-asr-stage-store/v1', 'producer_sha256': producer_sha256,
                    'synthetic_fixture': synthetic_fixture}
        if self.root.exists():
            require(self.root.is_dir(), 'stage store is not a directory')
            require((self.root/'identity.json').is_file() or not any(self.root.iterdir()),
                    'nonempty existing tree is not this stage store; preserve it')
        else:
            self.root.mkdir(parents=True)
        self.producer_sha256 = producer_sha256
        self.synthetic_fixture = synthetic_fixture
        self.identity = identity
        self.database = physical(self.root/'stage.sqlite3')
        with self.locked():
            p = physical(self.root/'identity.json')
            existed = p.exists()
            if p.exists():
                require(json.loads(p.read_bytes()) == identity, 'stage store belongs to another producer')
                require(self.database.is_file(), 'existing stage store database missing; preserve and repair explicitly')
            else:
                self._write_new(p, canonical(identity))
            with self.connect(create=not existed) as con:
                con.execute('CREATE TABLE IF NOT EXISTS batches (batch_key TEXT PRIMARY KEY, intent BLOB NOT NULL, result BLOB, result_sha256 TEXT)')
                con.execute('CREATE TABLE IF NOT EXISTS requests (request_key TEXT PRIMARY KEY, batch_key TEXT NOT NULL, identity BLOB NOT NULL)')

    def _write_new(self, path, data):
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try: os.fsync(fd)
        finally: os.close(fd)

    @contextlib.contextmanager
    def locked(self):
        lock = physical(self.root/'.stage.lock')
        fd = os.open(str(lock), os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def connect(self, *, create=False):
        for suffix in ('', '-wal', '-shm', '-journal'):
            physical(str(self.database)+suffix)
        if not create:
            require(self.database.is_file(), 'existing stage store database missing; no automatic recreation')
        uri='file:'+quote(str(self.database),safe='/')+'?mode='+('rwc' if create else 'rw')
        con = sqlite3.connect(uri, timeout=30, uri=True)
        con.execute('PRAGMA journal_mode=DELETE')
        con.execute('PRAGMA synchronous=FULL')
        try:
            with con:
                yield con
        finally:
            con.close()

    def _batch(self, con, key):
        row = con.execute('SELECT intent,result,result_sha256 FROM batches WHERE batch_key=?', (key,)).fetchone()
        require(row is not None, 'request reservation has no complete batch intent')
        intent = json.loads(row[0])
        require(digest(intent) == key and intent['producer_sha256'] == self.producer_sha256,
                'batch intent identity differs')
        if row[1] is None:
            require(row[2] is None, 'partial database commit refused')
            return intent, None
        require(hashlib.sha256(row[1]).hexdigest() == row[2], 'stored batch result SHA differs')
        result = json.loads(row[1])
        require(result['cohort_sha256'] == key and result['producer_sha256'] == self.producer_sha256 and
                [v['request_id'] for v in result['results']] == intent['request_keys'],
                'stored result request coverage differs')
        self._validate_results(intent, result['results'])
        return intent, result

    def _validate_results(self, intent, results):
        require(len(results) == len(intent['request_keys']), 'stored result count differs')
        for key, identity, result in zip(intent['request_keys'], intent['request_identities'], results):
            require(digest(identity) == key and result.get('request_id') == key and
                    result.get('request_identity') == identity and result.get('producer_sha256') == self.producer_sha256 and
                    result.get('cohort_sha256') == digest(intent) and
                    all(result.get(k) == v for k,v in identity.items() if k != 'schema') and
                    result.get('synthetic_fixture') is self.synthetic_fixture and
                    result.get('usable_for_training') is False and result.get('usable_for_evaluation') is False,
                    'stored per-request producer/input or qualification differs')

    def lookup(self, identity):
        key = digest(identity)
        with self.connect() as con:
            row = con.execute('SELECT batch_key,identity FROM requests WHERE request_key=?', (key,)).fetchone()
            if row is None: return None
            require(row[1] == canonical(identity), 'request identity differs')
            _, result = self._batch(con, row[0])
            if result is None: return None
            found = next(v for v in result['results'] if v['request_id'] == key)
            return copy.deepcopy(found)

    def plans(self, identities):
        """调用者持有store锁；恢复pending批次时必须提供原批完整请求。"""
        by_key = {digest(x): x for x in identities}
        require(len(by_key) == len(identities), 'duplicate request identity')
        pending, unseen = {}, []
        with self.connect() as con:
            for key, identity in by_key.items():
                row = con.execute('SELECT batch_key,identity FROM requests WHERE request_key=?', (key,)).fetchone()
                if row is None:
                    unseen.append(key); continue
                require(row[1] == canonical(identity), 'reserved request identity changed')
                intent, result = self._batch(con, row[0])
                if result is None:
                    require(set(intent['request_keys']) <= set(by_key),
                            'pending cohort requires its original complete inputs; no changed-cohort retry')
                    pending[row[0]] = intent
            if unseen:
                intent = {'schema': 'annotation-asr-cohort/v1', 'producer_sha256': self.producer_sha256,
                          'request_keys': unseen, 'request_identities': [by_key[k] for k in unseen],
                          'batch_policy': 'fixed_cohort_until_durable; immutable_published_result_reuse/v1'}
                key = digest(intent)
                con.execute('INSERT INTO batches(batch_key,intent) VALUES (?,?)', (key, canonical(intent)))
                for rid in unseen:
                    con.execute('INSERT INTO requests(request_key,batch_key,identity) VALUES (?,?,?)',
                                (rid, key, canonical(by_key[rid])))
                pending[key] = intent
        return list(pending.items())

    def commit(self, key, results):
        """单个事务保存整批原生结果；已提交时只接受完全相同字节。"""
        packet = {'schema': 'annotation-asr-cohort-result/v1', 'producer_sha256': self.producer_sha256,
                  'cohort_sha256': key, 'synthetic_fixture': self.synthetic_fixture, 'results': results}
        raw = canonical(packet)
        with self.connect() as con:
            intent, previous = self._batch(con, key)
            require([v['request_id'] for v in results] == intent['request_keys'], 'batch result denominator/order differs')
            self._validate_results(intent, results)
            if previous is not None:
                require(canonical(previous) == raw, 'immutable batch result differs; preserve existing result')
                return
            con.execute('UPDATE batches SET result=?,result_sha256=? WHERE batch_key=? AND result IS NULL',
                        (raw, hashlib.sha256(raw).hexdigest(), key))

    def preserve_failure(self, key, outputs, exc, *, stage_committed=False):
        directory = physical(self.root/'attempts'); directory.mkdir(exist_ok=True)
        data = {'schema': 'annotation-asr-failed-attempt/v1', 'producer_sha256': self.producer_sha256,
                'cohort_sha256': key, 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                'native_outputs': outputs, 'synthetic_fixture': self.synthetic_fixture,
                'stage_committed_before_error': stage_committed}
        self._write_new(directory/(uuid.uuid4().hex+'.json'), canonical(data))
