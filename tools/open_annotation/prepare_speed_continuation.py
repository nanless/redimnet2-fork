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
import tempfile
import traceback
import time

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


def file_sha(path):
    h=hashlib.sha256()
    with physical(path).open('rb')as f:
        for raw in iter(lambda:f.read(1048576),b''):h.update(raw)
    return h.hexdigest()


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


def source_roles(source):
    """按来源并集计角色；PTN与LM可重叠，拒绝重复或未知角色。"""
    roles=source.get('roles')
    need(isinstance(roles,list) and roles and all(isinstance(r,str)for r in roles) and
         len(roles)==len(set(roles)) and set(roles)<={'train_ptn','train_lm'},'invalid/duplicate training source roles')
    return roles


def old_job_key(source, contract_sha):
    return digest({'source_id': source['source_id'], 'source_record_sha256': digest(source),
                   'contract_sha256': contract_sha})


def portable_progress(stage,counts):
    print(json.dumps({'event':'portable_preparation_progress','stage':stage,
        'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**counts},ensure_ascii=False),flush=True)


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

CONTENT_STAT_KEYS = ('inode', 'bytes', 'mtime_ns', 'ctime_ns')


def identity_matches(actual, recorded):
    """内容身份：device 是挂载命名空间标签，不参与；其余字段要完全相等。"""
    relaxed = actual.get('device') != recorded.get('device')
    for key in CONTENT_STAT_KEYS:
        if key not in recorded:
            raise ValueError('recorded stat identity missing field: ' + key)
        if actual.get(key) != recorded[key]:
            raise ValueError('source content changed: %s expected=%r actual=%r' % (key, recorded[key], actual.get(key)))
    return relaxed


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


def retained_sidecar(source, tree, contract_sha, validate_record, *, portable_context=None, portable_sink=None,
                     inventory=None,inventory_kind=None):
    # missing先查新目录项库；不能先child()/physical()，否则仍会逐来源lstat PFS。
    if inventory is not None and not inventory.select(source,inventory_kind):return None
    path = child(tree, source['sidecar_relpath'])
    commit = path.with_name(path.name + '.commit.json')
    physical(commit)
    if inventory is None:
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
    if portable_context is not None:
        need(callable(portable_sink),'explicit portable evidence sink required')
        need(record['contract']['version']==portable_context['contract_version'],'portable old contract version differs')
        verifier=portable_context['module']
        proof=verifier.prove_source_identity(portable_context['spec'],portable_context['spec_sha256'],source,record,
            contract_sha,byte_sha(raw),byte_sha(marker_raw))
        portable_sink(proof)
    elif identity['decode_status'] == 'decoded_complete':
        media = physical(source['audio_path'])
        need(media.is_file(), 'original media missing')
        actual = stat_identity(media)
        need(actual == identity['stat_after'],
             'source stat changed since old result; no silent reinference: ' +
             json.dumps({'source_id': source['source_id'], 'expected': identity['stat_after'],
                         'actual': actual}, sort_keys=True))
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


def partition(index_path, index_sha, queue, contract_sha, tree, scan_unclaimed=False,
              unclaimed_sink=None, remaining_sink=None, *, portable_context=None, portable_sink=None,inventory=None):
    """Read-only algorithm; caller separately admits fixed production denominator."""
    progress = inspect_progress(index_path, index_sha, queue, contract_sha)
    index, index_raw = read_json(index_path)
    need(byte_sha(index_raw) == index_sha, 'index changed after progress snapshot')
    state = progress['broker_snapshot']; nxt = state['next']; leases = state['leases']
    need(len(leases) <= MAX_PARTIAL_SHARDS, 'more than eight partial shards refused')
    validate = record_validator()
    retained_chunks = []; partial_references = []; remaining = []; partial_rows = []; partial_seen = set(); unclaimed_kept = 0
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
            if inventory is not None:
                for source in read_shard(index_path.parent,shard):inventory.select(source,'completed')
        elif ordinal >= nxt:
            if not scan_unclaimed:
                remaining.append({'mode': 'copy_original_shard', 'descriptor': dict(shard)})
                continue
            need(callable(unclaimed_sink) and callable(remaining_sink), 'final unclaimed scan sinks missing')
            rows=read_shard(index_path.parent,shard);missing=[];kept_count=0
            for source in rows:
                options={}if portable_context is None else {'portable_context':portable_context,'portable_sink':portable_sink,
                    'inventory':inventory,'inventory_kind':'unclaimed'}
                kept=retained_sidecar(source,tree,contract_sha,validate,**options)
                if kept is None:missing.append(source)
                else:
                    unclaimed_sink(kept);unclaimed_kept+=1;kept_count+=1
            if not kept_count:
                remaining.append({'mode':'copy_original_shard','descriptor':dict(shard)})
            elif missing:
                name='unclaimed-'+shard['id'];data=b''.join(canonical(s)for s in missing)
                remaining_sink(name,data)
                remaining.append({'mode':'new_unclaimed_shard','descriptor':{
                    'id':name,'path':'shards/'+name+'.jsonl','count':len(missing),'sha256':byte_sha(data),
                    'source_ids_sha256':digest(sorted(s['source_id']for s in missing))},'origin_shard_id':shard['id']})
        else:
            rows = read_shard(index_path.parent, shard); missing = []
            for source in rows:
                need(source['source_id'] not in partial_seen, 'duplicate source across partial shards')
                partial_seen.add(source['source_id'])
                options={}if portable_context is None else {'portable_context':portable_context,'portable_sink':portable_sink,
                    'inventory':inventory,'inventory_kind':'partial'}
                kept = retained_sidecar(source, tree, contract_sha, validate,**options)
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
    kept = progress['committed_source_lower_bound'] + len(partial_references) + unclaimed_kept
    left = sum(v['descriptor']['count'] for v in remaining)
    need(kept + left == index['source_count'], 'retained and remaining do not cover original denominator')
    return index, progress, {'retained_source_count': kept, 'remaining_source_count': left,
        'retained_complete_shards': retained_chunks, 'retained_partial_sources': partial_references,
        'retained_unclaimed_source_count':unclaimed_kept,'unclaimed_sidecars_scanned':scan_unclaimed,
        'remaining_shards': remaining}, partial_rows


def prepare(index_path, index_sha, queue, old_contract_path, contract_sha, tree, output,
            execute_final=False, terminal_path=None, terminal_sha=None, old_task_id=None, *,
            portable_spec_path=None, portable_spec_sha=None):
    index_path, queue, old_contract_path, tree, output = map(physical,
        (index_path, queue, old_contract_path, tree, output))
    need(not output.exists(), 'output must be a fresh external directory')
    here=Path(__file__).resolve().parent
    repository=next((p for p in here.parents if (p/'AGENTS.md').is_file()and(p/'redimnet2').is_dir()),None)
    protected_roots=[queue,tree,index_path.parent,old_contract_path.parent,here]
    if repository is not None:protected_roots.append(repository)
    if old_contract_path.parent.name=='contracts'and(old_contract_path.parent.parent/'bundle_files.json').is_file():
        protected_roots.append(old_contract_path.parent.parent)
    for protected in protected_roots:
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
    portable=None
    need((portable_spec_path is None)==(portable_spec_sha is None),'portable source spec path and SHA must be supplied together')
    if portable_spec_path is not None:
        need(execute_final,'portable content preparation requires explicit final mode and old writer lock')
        if __package__:
            from . import portable_source_identity as V
        else:
            import portable_source_identity as V
        portable_path=physical(portable_spec_path)
        portable_value=V.load_portable_spec(portable_path,portable_spec_sha,
            contract_pin={'path':str(old_contract_path),'sha256':contract_sha},index_pin={'path':str(index_path),'sha256':index_sha},
            queue=queue,tree=tree)
        portable={'module':V,'spec':portable_value,'spec_sha256':portable_spec_sha,'path':str(portable_path),
                  'contract_version':contract['version']}
    terminal = None; lock_evidence = None
    if execute_final:
        need(terminal_path and terminal_sha and old_task_id, 'final preparation needs terminal observation pins')
        terminal = terminal_observation(terminal_path, terminal_sha, old_task_id, queue)
    ctx = old_parent_lock(queue) if execute_final else contextlib.nullcontext(None)
    with ctx as lock_evidence:
        scratch=None;ledger=None;content_ledger=None;content_counts=collections.Counter();inventory=None;inventory_summary=None
        if execute_final:
            # 仅在旧writer互斥后一次流式扫描；临时文件是新的准备证据，不在旧树内。
            scratch=Path(tempfile.mkdtemp(prefix='.continuation-build-',dir=output.parent))
            ledger=(scratch/'retained-unclaimed.jsonl').open('xb')
            if portable is not None:content_ledger=(scratch/'portable-source-evidence.jsonl').open('xb')
        def retain_content(item):
            need(content_ledger is not None,'portable content ledger not open')
            content_ledger.write(canonical(item));content_counts[item['status']]+=1
        try:
            if portable is not None:
                inventory=portable['module'].SidecarInventory(scratch/'portable-sidecar-inventory.sqlite',tree,
                    portable['spec'],portable_spec_sha,expected_index['domains'],progress=portable_progress)
                inventory.scan()
                inventory.phase='partition';inventory.emit_progress('partition_started',True)
            options={}if portable is None else {'portable_context':portable,'portable_sink':retain_content,'inventory':inventory}
            index, progress, plan, partial_rows = partition(index_path, index_sha, queue, contract_sha, tree,
                execute_final, (lambda item:ledger.write(canonical(item))) if ledger else None,
                (lambda name,data:write_new(scratch/(name+'.jsonl'),data)) if scratch else None,
                **options)
            if inventory is not None:
                inventory_summary=inventory.finish(index['source_count']);inventory.emit_progress('partition_finished',True)
        except BaseException:
            if portable is not None and scratch is not None:
                write_new(scratch/'portable-preparation-failure.json',{'traceback':traceback.format_exc(),
                    'portable_spec_sha256':portable_spec_sha,'old_files_modified_by_this_tool':False,
                    'inventory_scan_counts':dict(inventory.counts)if inventory is not None else None,
                    'phase':inventory.phase if inventory is not None else 'before_inventory',
                    'partial_content_rows_written':sum(content_counts.values())})
            raise
        finally:
            if ledger is not None:
                ledger.flush();os.fsync(ledger.fileno());ledger.close()
            if content_ledger is not None:
                content_ledger.flush();os.fsync(content_ledger.fileno());content_ledger.close()
            if inventory is not None:inventory.close()
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
            ledger_path=output/'retained-unclaimed.jsonl'
            os.rename(scratch/'retained-unclaimed.jsonl',ledger_path)
            report['retained_unclaimed_sources_ref']={'path':str(ledger_path),'sha256':file_sha(ledger_path),
                'count':plan['retained_unclaimed_source_count']}
            if portable is not None:
                content_path=output/'portable-source-evidence.jsonl'
                os.rename(scratch/'portable-source-evidence.jsonl',content_path)
                content_count=sum(content_counts.values())
                need(content_count==len(plan['retained_partial_sources'])+plan['retained_unclaimed_source_count'],
                     'portable content evidence does not exactly cover retained partial/unclaimed sources')
                report.update(source_identity_mode=portable['module'].MODE,
                    portable_source_spec_ref={'path':portable['path'],'sha256':portable_spec_sha},
                    portable_source_ledger_ref={'path':str(content_path),'sha256':file_sha(content_path),'count':content_count},
                    portable_source_evidence_counts=dict(content_counts),
                    fresh_media_scope='retained_partial_and_unclaimed_only',
                    completed_media_scope='original_completed_chunk_reference_not_media_rehashed',
                    completed_media_rehashed=False,source_PCM_redecoded=False)
                inventory_path=output/'portable-sidecar-inventory.sqlite'
                os.rename(scratch/'portable-sidecar-inventory.sqlite',inventory_path)
                report['portable_sidecar_inventory_ref']={'path':str(inventory_path),'sha256':file_sha(inventory_path)}
                report['portable_sidecar_inventory_summary']=inventory_summary
            partial = dict(partial_rows); descriptors = []; domains = collections.Counter(); roles=collections.Counter()
            if portable is not None:
                portable_progress('remaining_write_started',{'remaining_shards_total':len(plan['remaining_shards']),
                    'remaining_source_count':plan['remaining_source_count']});last_progress=time.monotonic()
            for item in plan['remaining_shards']:
                descriptor = dict(item['descriptor'])
                if item['mode'] == 'copy_original_shard':
                    source_path = child(index_path.parent, descriptor['path'])
                    rows, data = read_json_lines(source_path)
                    need(byte_sha(data) == descriptor['sha256'] and len(rows) == descriptor['count'] and
                         len({s['source_id'] for s in rows}) == len(rows), 'remaining source shard changed')
                    descriptor['path'] = 'shards/' + descriptor['id'] + '.jsonl'
                elif item['mode']=='new_unclaimed_shard':
                    rows,data=read_json_lines(scratch/(descriptor['id']+'.jsonl'))
                    need(byte_sha(data)==descriptor['sha256'],'filtered unclaimed shard changed')
                else:
                    data = partial[descriptor['id']]; rows = [strict_json(v) for v in data.splitlines()]
                if 'source_ids_sha256' in descriptor:
                    need(digest(sorted(s['source_id'] for s in rows)) == descriptor['source_ids_sha256'], 'remaining source-ID digest differs')
                domains.update(s['dataset'] for s in rows)
                for source in rows:roles.update(source_roles(source))
                write_new(child(output, descriptor['path']), data); descriptors.append(descriptor)
                if item['mode']=='new_unclaimed_shard':(scratch/(descriptor['id']+'.jsonl')).unlink()
                if portable is not None and time.monotonic()-last_progress>=30:
                    portable_progress('remaining_write_progress',{'remaining_shards_written':len(descriptors),
                        'remaining_shards_total':len(plan['remaining_shards'])});last_progress=time.monotonic()
            current_lock = physical(queue/'.parent.lock').stat()
            need((current_lock.st_dev,current_lock.st_ino) == (lock_evidence['device'],lock_evidence['inode']), 'old parent lock replaced')
            _, broker_after = read_json(queue / 'broker.json', 1024**2)
            need(byte_sha(broker_after) == progress['broker_snapshot_sha256'],
                 'broker changed before remaining index publication')
            new_index = {**index, 'source_count':plan['remaining_source_count'], 'shards':descriptors, 'domains':dict(domains),
                'parent_full_index':{'path':str(index_path),'sha256':index_sha},
                'coverage_plan_file':'coverage-plan.json', 'source_count_is_remaining_not_full':True,
                'role_union_counts':{role:roles[role]for role in ('train_ptn','train_lm')},
                'role_union_counts_scope':'remaining_index_sources'}
            if 'role_union_counts'in index:
                new_index['parent_role_union_counts']=index['role_union_counts']
                new_index['role_union_counts_parent_index']={'path':str(index_path),'sha256':index_sha}
            if descriptors:
                report['remaining_index'] = {'path':str(output/'index.json'),
                    'sha256':write_new(output/'index.json', new_index)}
            else:
                report['status'] = 'all_sources_retained_no_successor_needed'
            if portable is not None:portable_progress('remaining_write_finished',{'remaining_shards_written':len(descriptors),
                'remaining_source_count':plan['remaining_source_count']})
        write_new(output / 'coverage-plan.json', report)
        if scratch is not None:scratch.rmdir()
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
    p.add_argument('--portable-source-spec',type=Path,help='显式新跨节点内容协议；默认严格stat不变，只在final模式使用')
    p.add_argument('--portable-source-spec-sha')
    a=p.parse_args(argv)
    r=prepare(a.index,a.index_sha,a.queue_root,a.old_contract,a.old_contract_sha,
              a.old_sidecar_tree,a.output,a.execute_final,a.terminal_observation,a.terminal_sha,a.old_task_id,
              portable_spec_path=a.portable_source_spec,portable_spec_sha=a.portable_source_spec_sha)
    print(json.dumps({'output':str(a.output),'status':r['status'],
        'retained_source_count':r['retained_source_count'],'remaining_source_count':r['remaining_source_count'],
        'new_inference_count':0,'formal_runtime_admitted':False},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
