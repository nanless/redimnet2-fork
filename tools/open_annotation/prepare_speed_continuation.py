"""只读保留旧生产结果，为新转写配置准备不重复的剩余输入。

默认只写独立的 provisional 准备报告。只有显式 --execute-final、真实旧
writer 终态观察、既有父锁独占和稳定快照同时满足时，才复制剩余文本分片。
不启动 GPU、不实例化 WorkQueue/SidecarStore、不修改旧合同、队列或结果。
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import collections
import struct
import os
from pathlib import Path, PurePosixPath
import stat
import sys

if __package__:
    from .readonly_progress import inspect_progress, read_small, strict_json
else:
    from readonly_progress import inspect_progress, read_small, strict_json

EXPECTED_SOURCE_COUNT = 11521178
EXPECTED_DOMAINS = {'voxceleb2': 1092009, 'voxblink2': 9904382, 'cnceleb2': 524787}
MAX_PARTIAL_SHARDS = 8
TERMINAL_STATES = {'Stopped', 'Failed', 'Succeeded', 'Completed'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def byte_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def physical(path):
    p = Path(path).absolute()
    need('..' not in p.parts, 'parent traversal in path refused')
    for item in [*reversed(p.parents), p]:
        need(not item.is_symlink(), 'symlink in input/output path refused')
    return p


def child(root, relative):
    need(isinstance(relative, str) and relative and '\\' not in relative and '\x00' not in relative,
         'invalid relative path')
    p = PurePosixPath(relative)
    need(not p.is_absolute() and all(a not in ('', '.', '..') for a in relative.split('/')),
         'relative path escapes root')
    return physical(root / relative)


def read_json(path, maximum=64 * 1024**2):
    return read_small(physical(path), maximum)


def write_new(path, value):
    data = value if isinstance(value, bytes) else canonical(value)
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return byte_sha(data)


def old_job_key(source, contract_sha):
    return digest({'source_id': source['source_id'], 'source_record_sha256': digest(source),
                   'contract_sha256': contract_sha})


def record_validator():
    # schema.py has script-style dependency imports. Load only this maintained
    # directory; never import modules from the old writer's frozen tree.
    here = str(Path(__file__).resolve().parent)
    sys.path.insert(0, here)
    try:
        import schema
        need(Path(schema.__file__).resolve() == Path(here)/'schema.py', 'unexpected schema module identity')
        return schema.assert_valid_record
    finally:
        sys.path.pop(0)


def stat_identity(path):
    s = path.stat()
    return {'device': s.st_dev, 'inode': s.st_ino, 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def read_shard(index_root, shard):
    path = child(index_root, shard['path'])
    value, raw = read_json_lines(path)
    need(byte_sha(raw) == shard['sha256'], 'partial shard SHA differs')
    need(len(value) == shard['count'] and len({s['source_id'] for s in value}) == len(value),
         'partial shard count or source uniqueness differs')
    return value


def read_json_lines(path):
    fd = os.open(str(physical(path)), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd); need(stat.S_ISREG(before.st_mode), 'shard must be regular')
        need(before.st_size <= 32 * 1024**2, 'shard exceeds bounded input size')
        with os.fdopen(fd, 'rb', closefd=False) as f:
            raw = f.read(32 * 1024**2 + 1)
        after = os.fstat(fd)
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns) and len(raw) == before.st_size,
             'shard changed while read')
        rows = raw.splitlines(); need(all(x.strip() for x in rows), 'blank source row')
        return [strict_json(x) for x in rows], raw
    finally:
        os.close(fd)


def retained_sidecar(source, tree, contract_sha, validate_record):
    path = child(tree, source['sidecar_relpath'])
    commit = path.with_name(path.name + '.commit.json')
    physical(commit)
    if not path.exists():
        need(not commit.exists(), 'orphan sidecar commit refused')
        return None
    need(commit.is_file(), 'orphan sidecar preserved; explicit old-contract repair required')
    record, raw = read_json(path, 32 * 1024**2)
    marker, marker_raw = read_json(commit, 64 * 1024)
    validate_record(record)
    need(record.get('source') == source and record['contract']['sha256'] == contract_sha,
         'partial result source or old contract differs')
    key = old_job_key(source, contract_sha)
    need(record['execution']['job_key'] == key, 'partial result job identity differs')
    expected = {'schema': 'redimnet2-sidecar-commit/v1', 'source_id': source['source_id'],
                'job_key': key, 'contract_sha256': contract_sha, 'source_record_sha256': digest(source),
                'sidecar_sha256': byte_sha(raw), 'bytes': len(raw)}
    need(marker == expected, 'partial result commit identity differs')
    identity = record['source_identity']
    if identity['decode_status'] == 'decoded_complete':
        media = physical(source['audio_path'])
        need(media.is_file() and stat_identity(media) == identity['stat_after'],
             'source stat changed since old result; no silent reinference')
    need(record['decision']['usable_for_training'] is False and record['decision']['usable_for_evaluation'] is False,
         'automatic training/evaluation release refused')
    return {'source_id': source['source_id'], 'source_record_sha256': digest(source),
            'sidecar_path': str(path), 'sidecar_sha256': byte_sha(raw),
            'commit_path': str(commit), 'commit_sha256': byte_sha(marker_raw),
            'producer_contract_sha256': contract_sha, 'decision': record['decision']['state']}


def terminal_observation(path, expected_sha, task_id, queue_root):
    v, raw = read_json(path, 1024**2)
    need(byte_sha(raw) == expected_sha, 'terminal observation SHA differs')
    need(v.get('schema') == 'annotation-writer-terminal-observation/v1' and
         v.get('task_id') == task_id and v.get('queue_root') == str(queue_root),
         'terminal observation task/queue identity differs')
    need(v.get('state') in TERMINAL_STATES, 'old writer not observed terminal')
    now = datetime.datetime.now(datetime.timezone.utc)
    observed = datetime.datetime.fromisoformat(v['observed_utc'])
    need(observed.tzinfo is not None and 0 <= (now - observed).total_seconds() <= 3600,
         'terminal observation stale or future')
    workers = v.get('workers')
    need(isinstance(workers, list) and len(workers) == 1 and
         workers[0].get('name') == task_id + '-worker-0' and
         workers[0].get('state') in TERMINAL_STATES,
         'single platform worker terminal observation missing')
    return {'path': str(path), 'sha256': expected_sha, 'task_id': task_id,
            'observed_utc': v['observed_utc'],
            'platform_authentication_performed_by_this_tool': False}


@contextlib.contextmanager
def old_parent_lock(queue):
    lock = physical(queue / '.parent.lock')
    fd = os.open(str(lock), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        s = os.fstat(fd); need(stat.S_ISREG(s.st_mode), 'old parent lock not regular')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('old writer parent lock still owned; no final preparation') from exc
        current = lock.stat()
        need((s.st_dev, s.st_ino) == (current.st_dev, current.st_ino), 'old parent lock replaced')
        yield {'path': str(lock), 'device': s.st_dev, 'inode': s.st_ino,
               'existing_lock_opened_readonly': True, 'lock_created': False}
    finally:
        os.close(fd)


def partition(index_path, index_sha, queue, contract_sha, tree):
    """Read-only algorithm; caller separately admits fixed production denominator."""
    progress = inspect_progress(index_path, index_sha, queue, contract_sha)
    index, index_raw = read_json(index_path)
    need(byte_sha(index_raw) == index_sha, 'index changed after progress snapshot')
    state = progress['broker_snapshot']; nxt = state['next']; leases = state['leases']
    need(len(leases) <= MAX_PARTIAL_SHARDS, 'more than eight partial shards refused')
    validate = record_validator()
    retained_chunks = []; partial_references = []; remaining = []; partial_rows = []; partial_seen = set()
    for ordinal, shard in enumerate(index['shards']):
        if ordinal < nxt and shard['id'] not in leases:
            fd = os.open(str(physical(queue / 'completion_map.bin')), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            try:
                slot = os.pread(fd, 1024, ordinal * 1024)
            finally:
                os.close(fd)
            need(len(slot) == 1024 and slot[:8] == b'RDMAP001', 'retained completion missing or torn')
            size = struct.unpack('>I', slot[8:12])[0]
            need(0 < size <= 980, 'retained completion payload size invalid')
            payload = slot[44:44+size]
            need(byte_sha(payload) == slot[12:44].hex() and slot[44+size:] == bytes(980-size), 'retained completion changed')
            proof = strict_json(payload)
            need(proof['shard_id'] == shard['id'] and proof['shard_sha256'] == shard['sha256'] and
                 proof['contract_sha256'] == contract_sha and proof['index_sha256'] == index_sha and
                 type(proof.get('source_count')) is int and proof['source_count'] == shard['count'],
                 'retained completion identity changed')
            retained_chunks.append({**shard, 'completion_slot_sha256': byte_sha(slot),
                'chunk_receipt_path': str(queue/'completed'/(shard['id']+'.json')),
                'chunk_receipt_sha256':proof['commit_sha256'], 'source_ids_sha256':proof['source_ids_sha256']})
        elif ordinal >= nxt:
            remaining.append({'mode': 'copy_original_shard', 'descriptor': dict(shard)})
        else:
            rows = read_shard(index_path.parent, shard); missing = []
            for source in rows:
                need(source['source_id'] not in partial_seen, 'duplicate source across partial shards')
                partial_seen.add(source['source_id'])
                kept = retained_sidecar(source, tree, contract_sha, validate)
                if kept is None:
                    missing.append(source)
                else:
                    partial_references.append(kept)
            if missing:
                name = 'partial-' + shard['id']
                data = b''.join(canonical(s) for s in missing)
                remaining.append({'mode': 'new_partial_shard', 'descriptor': {
                    'id': name, 'path': 'shards/' + name + '.jsonl', 'count': len(missing),
                    'sha256': byte_sha(data), 'source_ids_sha256':digest(sorted(s['source_id'] for s in missing))}, 'origin_shard_id': shard['id']})
                partial_rows.append((name, data))
    need(len({v['descriptor']['id'] for v in remaining}) == len(remaining), 'remaining shard ID collision')
    kept = progress['committed_source_lower_bound'] + len(partial_references)
    left = sum(v['descriptor']['count'] for v in remaining)
    need(kept + left == index['source_count'], 'retained and remaining do not cover original denominator')
    return index, progress, {'retained_source_count': kept, 'remaining_source_count': left,
        'retained_complete_shards': retained_chunks, 'retained_partial_sources': partial_references,
        'remaining_shards': remaining}, partial_rows


def prepare(index_path, index_sha, queue, old_contract_path, contract_sha, tree, output,
            execute_final=False, terminal_path=None, terminal_sha=None, old_task_id=None):
    index_path, queue, old_contract_path, tree, output = map(physical,
        (index_path, queue, old_contract_path, tree, output))
    need(not output.exists(), 'output must be a fresh external directory')
    for protected in (queue, tree, index_path.parent, old_contract_path.parent,
                      Path(__file__).resolve().parents[2]):
        need(output != protected and protected not in output.parents,
             'output overlaps old input, result, queue or repository')
    need(output.parent.is_dir(), 'output parent must already exist')
    contract, contract_raw = read_json(old_contract_path)
    need(byte_sha(contract_raw) == contract_sha and contract.get('schema') == 'redimnet2-production-contract/v2' and
         contract.get('purpose') == 'silver_annotation' and contract.get('model_API_calls_allowed') == 0 and
         contract.get('release_training_labels') is False,
         'explicit old production contract differs')
    identity, _ = read_json(tree / 'identity.json')
    need(identity == {'schema':'redimnet2-sidecar-tree/v1', 'version':contract['version'],
                      'contract_sha256':contract_sha}, 'old sidecar namespace identity differs')
    expected_index, expected_raw = read_json(index_path)
    need(byte_sha(expected_raw) == index_sha, 'original index SHA differs')
    need(expected_index.get('source_count') == EXPECTED_SOURCE_COUNT,
         'formal three-training denominator must remain 11521178')
    need(expected_index.get('domains') == EXPECTED_DOMAINS and
         expected_index.get('RIR_or_MUSAN_or_Child_or_evaluation_in_this_job') is False,
         'formal input must remain the three training domains only')
    terminal = None; lock_evidence = None
    if execute_final:
        need(terminal_path and terminal_sha and old_task_id, 'final preparation needs terminal observation pins')
        terminal = terminal_observation(terminal_path, terminal_sha, old_task_id, queue)
    ctx = old_parent_lock(queue) if execute_final else contextlib.nullcontext(None)
    with ctx as lock_evidence:
        index, progress, plan, partial_rows = partition(index_path, index_sha, queue, contract_sha, tree)
        if execute_final:
            current_lock = physical(queue/'.parent.lock').stat()
            need((current_lock.st_dev,current_lock.st_ino) == (lock_evidence['device'],lock_evidence['inode']), 'old parent lock replaced')
            _, broker_after = read_json(queue / 'broker.json', 1024**2)
            need(byte_sha(broker_after) == progress['broker_snapshot_sha256'],
                 'broker changed while final snapshot held')
        report = {'schema':'annotation-speed-continuation-plan/v1',
            'status':'final_prepared_not_started' if execute_final else 'provisional_not_executable',
            'formal_runtime_admitted':False, 'models_executed':False, 'new_inference_count':0,
            'old_queue_mutated':False, 'old_results_rewritten':False, 'stage_cache_relabelled':False,
            'old_index':{'path':str(index_path),'sha256':index_sha},
            'old_contract':{'path':str(old_contract_path),'sha256':contract_sha,'version':contract['version']},
            'old_sidecar_tree':str(tree), 'old_queue':str(queue),
            'old_progress_snapshot':progress, 'terminal_observation':terminal,
            'old_parent_lock':lock_evidence, **plan,
            'all_completed_sidecars_reopened':False, 'human_gold_verified':False,
            'limitations':['Completed chunks are retained by validated completion-slot/index references, not a full sidecar re-audit.',
                'Terminal JSON is external observation, not independently authenticated platform state.',
                'Released preparation lock is not a future execution lease. Recheck writer, platform and resource exclusion before launch.',
                'Unpublished stage-cache results remain under the old contract; they are not relabelled as new-profile cache hits.']}
        output.mkdir()
        write_new(output / 'preparation-started.json', {'execute_final':execute_final,
            'old_index_sha256':index_sha,'old_contract_sha256':contract_sha})
        if execute_final:
            (output / 'shards').mkdir()
            partial = dict(partial_rows); descriptors = []; domains = collections.Counter()
            for item in plan['remaining_shards']:
                descriptor = dict(item['descriptor'])
                if item['mode'] == 'copy_original_shard':
                    source_path = child(index_path.parent, descriptor['path'])
                    rows, data = read_json_lines(source_path)
                    need(byte_sha(data) == descriptor['sha256'] and len(rows) == descriptor['count'] and
                         len({s['source_id'] for s in rows}) == len(rows), 'remaining source shard changed')
                    descriptor['path'] = 'shards/' + descriptor['id'] + '.jsonl'
                else:
                    data = partial[descriptor['id']]; rows = [strict_json(v) for v in data.splitlines()]
                if 'source_ids_sha256' in descriptor:
                    need(digest(sorted(s['source_id'] for s in rows)) == descriptor['source_ids_sha256'], 'remaining source-ID digest differs')
                domains.update(s['dataset'] for s in rows)
                write_new(child(output, descriptor['path']), data); descriptors.append(descriptor)
            current_lock = physical(queue/'.parent.lock').stat()
            need((current_lock.st_dev,current_lock.st_ino) == (lock_evidence['device'],lock_evidence['inode']), 'old parent lock replaced')
            _, broker_after = read_json(queue / 'broker.json', 1024**2)
            need(byte_sha(broker_after) == progress['broker_snapshot_sha256'],
                 'broker changed before remaining index publication')
            new_index = {**index, 'source_count':plan['remaining_source_count'], 'shards':descriptors, 'domains':dict(domains),
                'parent_full_index':{'path':str(index_path),'sha256':index_sha},
                'coverage_plan_file':'coverage-plan.json', 'source_count_is_remaining_not_full':True}
            if descriptors:
                report['remaining_index'] = {'path':str(output/'index.json'),
                    'sha256':write_new(output/'index.json', new_index)}
            else:
                report['status'] = 'all_sources_retained_no_successor_needed'
        write_new(output / 'coverage-plan.json', report)
        return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index','queue-root','old-contract','old-sidecar-tree','output'):
        p.add_argument('--'+name, required=True, type=Path)
    for name in ('index-sha','old-contract-sha'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--execute-final',action='store_true')
    p.add_argument('--terminal-observation',type=Path);p.add_argument('--terminal-sha')
    p.add_argument('--old-task-id')
    a=p.parse_args(argv)
    r=prepare(a.index,a.index_sha,a.queue_root,a.old_contract,a.old_contract_sha,
              a.old_sidecar_tree,a.output,a.execute_final,a.terminal_observation,a.terminal_sha,a.old_task_id)
    print(json.dumps({'output':str(a.output),'status':r['status'],
        'retained_source_count':r['retained_source_count'],'remaining_source_count':r['remaining_source_count'],
        'new_inference_count':0,'formal_runtime_admitted':False},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
