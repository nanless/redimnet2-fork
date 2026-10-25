"""Read one broker snapshot and immutable completed slots; never construct a queue.

Only the requested new report is written. No locks, models, shard JSONL, labels or
logs are opened. This is a committed lower bound, not full sidecar verification.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

SLOT_BYTES = 1024
SLOT_MAGIC = b'RDMAP001'
DECISIONS = ('candidate', 'unknown', 'failed')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        require(math.isfinite(parsed), 'nonfinite JSON number')
        return parsed
    def invalid(value):
        raise ValueError('invalid JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)


def open_readonly(path):
    path = Path(path)
    require(not path.is_symlink(), 'input symlink refused: ' + str(path))
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), 'input is not a regular file')
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_small(path, maximum):
    fd = open_readonly(path)
    try:
        before = os.fstat(fd)
        require(before.st_size <= maximum, 'JSON exceeds declared size bound')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(fd)
        # A normal atomic broker replacement unlinks this still-readable old
        # inode and may change ctime; its pinned FD/bytes remain the snapshot.
        require(len(raw) == before.st_size and
                (before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), 'JSON changed during read')
        return strict_json(raw), raw
    finally:
        os.close(fd)


def inspect_progress(index_path, index_sha, queue_root, contract_sha):
    require(sha(index_sha) and sha(contract_sha), 'explicit lowercase SHA256 pins required')
    index, index_raw = read_small(index_path, 64 * 1024**2)
    require(hashlib.sha256(index_raw).hexdigest() == index_sha, 'index SHA differs')
    require(isinstance(index, dict) and index.get('schema') == 'redimnet2-shard-index/v1', 'wrong index schema')
    shards = index.get('shards')
    require(isinstance(shards, list) and bool(shards), 'missing shards')
    by_id = {}
    for ordinal, shard in enumerate(shards):
        require(isinstance(shard, dict), 'invalid shard descriptor')
        sid = shard.get('id')
        require(isinstance(sid, str) and sid and sid not in ('.', '..') and
                not any(c in sid for c in '/\\\x00') and sid not in by_id, 'unsafe/duplicate shard ID')
        require(type(shard.get('count')) is int and shard['count'] > 0 and sha(shard.get('sha256')), 'invalid shard count/SHA')
        by_id[sid] = ordinal
    require(type(index.get('source_count')) is int and index['source_count'] == sum(s['count'] for s in shards), 'index denominator differs')
    root = Path(queue_root).absolute()
    require(root.is_dir() and not root.is_symlink(), 'physical existing queue root required')
    identity, identity_raw = read_small(root / 'identity.json', 64 * 1024)
    expected = {'schema': 'redimnet2-work-queue/v1', 'index_sha256': index_sha,
                'contract_sha256': contract_sha, 'source_count': index['source_count'], 'shard_count': len(shards)}
    require(identity == expected, 'queue identity differs')
    # Exactly one atomic broker file is opened: later queue advances are excluded.
    broker, broker_raw = read_small(root / 'broker.json', 1024**2)
    require(isinstance(broker, dict) and set(broker) == {'next', 'leases', 'completed_chunks'}, 'invalid broker structure')
    nxt, completed, leases = broker['next'], broker['completed_chunks'], broker['leases']
    require(type(nxt) is int and 0 <= nxt <= len(shards), 'invalid broker cursor')
    require(type(completed) is int and 0 <= completed <= nxt and isinstance(leases, dict), 'invalid broker count/leases')
    require(all(sid in by_id and by_id[sid] < nxt and isinstance(token, dict) for sid, token in leases.items()), 'lease outside snapshot cursor')
    require(nxt - len(leases) == completed, 'broker completed count differs from nonleased prefix')
    counts = {key: 0 for key in DECISIONS}
    sources = slots = byte_count = 0
    fd = open_readonly(root / 'completion_map.bin')
    try:
        map_before = os.fstat(fd)
        require(map_before.st_size == len(shards) * SLOT_BYTES, 'completion map size differs')
        for ordinal in range(nxt):
            shard = shards[ordinal]
            if shard['id'] in leases:
                continue
            raw = os.pread(fd, SLOT_BYTES, ordinal * SLOT_BYTES)
            byte_count += len(raw)
            require(len(raw) == SLOT_BYTES and raw[:8] == SLOT_MAGIC, 'missing/torn completion slot')
            size = struct.unpack('>I', raw[8:12])[0]
            require(0 < size <= SLOT_BYTES - 44, 'bad completion payload length')
            payload = raw[44:44 + size]
            require(hashlib.sha256(payload).digest() == raw[12:44], 'completion checksum differs')
            require(raw[44 + size:] == bytes(SLOT_BYTES - 44 - size), 'nonzero slot padding')
            value = strict_json(payload)
            require(isinstance(value, dict) and value.get('shard_id') == shard['id'] and
                    value.get('shard_sha256') == shard['sha256'] and value.get('index_sha256') == index_sha and
                    value.get('contract_sha256') == contract_sha and type(value.get('source_count')) is int and
                    value['source_count'] == shard['count'], 'slot does not bind pinned shard')
            require(sha(value.get('source_ids_sha256')) and sha(value.get('commit_sha256')), 'slot commit/source-ID digest missing')
            if 'source_ids_sha256' in shard:
                require(value['source_ids_sha256'] == shard['source_ids_sha256'], 'source-ID digest differs')
            decisions = value.get('decision_counts')
            require(isinstance(decisions, dict) and set(decisions) == set(DECISIONS) and
                    all(type(n) is int and n >= 0 for n in decisions.values()) and
                    sum(decisions.values()) == shard['count'], 'decision counts invalid')
            for key in DECISIONS:
                counts[key] += decisions[key]
            sources += shard['count']; slots += 1
        require(os.fstat(fd).st_size == map_before.st_size, 'completion map size changed')
    finally:
        os.close(fd)
    require(slots == completed, 'verified slot count differs from broker completed chunks')
    return {'schema': 'redimnet2-readonly-progress/v1', 'passed': True,
            'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'index_sha256': index_sha, 'contract_sha256': contract_sha,
            'identity_sha256': hashlib.sha256(identity_raw).hexdigest(),
            'broker_snapshot_sha256': hashlib.sha256(broker_raw).hexdigest(), 'broker_snapshot': broker,
            'source_count': index['source_count'], 'shard_count': len(shards),
            'committed_source_lower_bound': sources, 'verified_completed_chunks': slots,
            'decision_counts': counts, 'completion_map_bytes_read': byte_count,
            'snapshot_leased_chunks_excluded': len(leases), 'queue_writes': 0, 'locks_acquired': 0,
            'sidecars_individually_reopened': 0, 'full_sidecar_verification': False,
            'human_gold_verified': False, 'GPU_model_imported_or_executed': False,
            'meaning': 'One broker snapshot; completed nonleased prefix slots only. Active work and later commits excluded. Checks producer publication accounting, not current sidecar bytes, audio, transcription or complete-word accuracy. Candidate is not human gold; unknown is unresolved; failed is a publication decision, not all operational failures.',
            'ETA': None, 'ETA_reason': 'One snapshot cannot establish comparable throughput or remaining time.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--index-sha', required=True)
    parser.add_argument('--queue-root', required=True, type=Path)
    parser.add_argument('--contract-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), 'output must be a new file')
    out = args.output.resolve(); queue = args.queue_root.resolve()
    require(out != queue and queue not in out.parents, 'output must be outside queue tree')
    require(out != args.index.resolve(), 'output overlaps index')
    result = inspect_progress(args.index, args.index_sha, args.queue_root, args.contract_sha)
    result['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Never create a queue or report directory implicitly.
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'output': str(out), 'committed_source_lower_bound': result['committed_source_lower_bound'],
                      'verified_completed_chunks': result['verified_completed_chunks']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
