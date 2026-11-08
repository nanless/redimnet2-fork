"""为转写提速生成有界、无数据库的交接清单，不启动或停止任何任务。

已封账片只读固定计数槽；终态后只核最多八个在途片的侧车/commit。
未领取片仅保留原索引描述，实际消费该片时才读原JSONL并核摘要。
新合同不能写旧queue；本模块的计划不是云资源、模型质量或GPU启动许可。
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

if __package__:
    from . import prepare_speed_continuation as P
    from .readonly_progress import inspect_progress
else:
    import prepare_speed_continuation as P
    from readonly_progress import inspect_progress

SCHEMA = 'annotation-bounded-speed-handoff/v1'
INDEX_SCHEMA = 'annotation-bounded-source-index/v1'
MAX_PARTIAL_SHARDS = 8
MAX_PARTIAL_ROWS = 4096
MAX_SIDECAR_BYTES = 512 * 1024**2
MAX_PORTABLE_ENCODED_BYTES = 4 * 1024**3


def need(value, message):
    if not value:
        raise ValueError(message)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read_pin(pin, maximum=64*1024**2):
    need(isinstance(pin, dict) and set(pin) == {'path', 'sha256'} and is_sha(pin['sha256']), 'explicit input pin required')
    need(isinstance(pin['path'], str) and Path(pin['path']).is_absolute(), 'absolute pinned input path required')
    value, raw = P.read_json(Path(pin['path']), maximum)
    need(P.byte_sha(raw) == pin['sha256'], 'input SHA differs')
    return value, raw


@contextlib.contextmanager
def lease_locks(queue, leases):
    """只打开已存在的claim锁，不创建、删除或更换inode。"""
    with contextlib.ExitStack() as stack:
        for sid in sorted(leases):
            path = P.child(queue, 'claims/' + sid + '.lock')
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            stack.callback(os.close, fd)
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode), 'claim lock must be regular')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('old claimed shard still owned') from exc
            now = path.stat()
            need((before.st_dev, before.st_ino) == (now.st_dev, now.st_ino), 'claim lock inode replaced')
        yield


def capture_slots(queue, shards, broker):
    """复制已封账固定槽的少量原字节，不打开完整侧车或未领取片。"""
    chunks = []
    fd = os.open(P.physical(queue/'completion_map.bin'), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        for i in range(broker['next']):
            if shards[i]['id'] not in broker['leases']:
                data = os.pread(fd, 1024, i*1024)
                need(len(data) == 1024, 'short completed slot')
                chunks.append(data)
    finally:
        os.close(fd)
    return b''.join(chunks)



class PortableReadBudget:
    """只在已有侧车时委托既有严格内容证明；额外施加本次有界读量。"""
    def __init__(self,module):
        self.module=module;self.encoded_bytes=0

    def prove_source_identity(self,*args,**kwargs):
        record=args[3]
        expected=record['source_identity'].get('bytes') if record['source_identity']['decode_status']=='decoded_complete' else 0
        need(type(expected)is int and expected>=0 and self.encoded_bytes+expected<=MAX_PORTABLE_ENCODED_BYTES,'portable encoded read budget exceeded')
        result=self.module.prove_source_identity(*args,**kwargs)
        self.encoded_bytes+=result['encoded_bytes_read']
        return result


def load_portable_context(container,contract_version):
    pin=container.get('portable_spec')
    if pin is None:return None
    if __package__:
        from . import portable_source_identity as module
    else:
        import portable_source_identity as module
    spec=module.load_portable_spec(Path(pin['path']),pin['sha256'],contract_pin=container['old_contract'],
        index_pin=container.get('old_index',container.get('parent_index')),queue=container['old_queue_root'],tree=container['old_sidecar_tree'])
    need(spec['purpose']==container['purpose'],'portable spec purpose differs from handoff')
    return dict(module=PortableReadBudget(module),spec=spec,spec_sha256=pin['sha256'],contract_version=contract_version)

def validate_spec(spec):
    need(isinstance(spec, dict) and spec.get('schema') == SCHEMA and spec.get('purpose') in ('production', 'cpu_fixture'), 'explicit bounded handoff schema/purpose required')
    for key in ('old_queue_root', 'old_sidecar_tree'):
        need(Path(spec[key]).is_absolute(), 'absolute producer path required')
        P.physical(Path(spec[key]))
    need(isinstance(spec.get('old_task_id'), str) and spec['old_task_id'].startswith('t-'), 'explicit old task required')
    need(type(spec.get('global_source_count')) is int and type(spec.get('previous_retained_count')) is int and spec['previous_retained_count'] >= 0, 'explicit scope counts required')
    if spec['purpose'] == 'production':
        need(spec['global_source_count'] == 11521178, 'production global scope must remain unchanged')
        need(isinstance(spec.get('portable_spec'),dict),'production cross-node handoff requires explicit portable content spec')
    need(spec.get('new_queue_root') != spec['old_queue_root'], 'new contract cannot write old queue')
    for key in ('new_queue_root', 'new_sidecar_root'):
        need(Path(spec[key]).is_absolute(), 'absolute new producer path required')
        p = P.physical(Path(spec[key]))
        old = P.physical(Path(spec['old_queue_root']))
        tree = P.physical(Path(spec['old_sidecar_tree']))
        need(p != old and old not in p.parents and p not in old.parents and p != tree and tree not in p.parents and p not in tree.parents, 'new output overlaps old producer')
    a=Path(spec['new_queue_root']);b=Path(spec['new_sidecar_root'])
    need(a!=b and a not in b.parents and b not in a.parents,'new queue and new sidecar roots must be disjoint')


def prepare(spec_path, spec_sha, output, *, terminal_pin=None, execute_final=False):
    spec, _ = read_pin({'path': str(spec_path), 'sha256': spec_sha}, 1024**2)
    validate_spec(spec)
    index, _ = read_pin(spec['old_index'])
    contract, _ = read_pin(spec['old_contract'])
    index_path = P.physical(Path(spec['old_index']['path']))
    queue = P.physical(Path(spec['old_queue_root']))
    tree = P.physical(Path(spec['old_sidecar_tree']))
    need(index.get('schema') == 'redimnet2-shard-index/v1' and index['source_count'] + spec['previous_retained_count'] == spec['global_source_count'], 'parent index/global denominator differs')
    need(contract.get('version') == tree.name, 'old sidecar version differs')
    output = P.physical(Path(output))
    need(not output.exists() and output.parent.is_dir(), 'fresh output with existing parent required')
    for forbidden in (queue, tree, index_path.parent, Path(spec['old_contract']['path']).parent):
        need(output != forbidden and forbidden not in output.parents, 'plan output inside old input/producer refused')
    progress = inspect_progress(index_path, spec['old_index']['sha256'], queue, spec['old_contract']['sha256'])
    broker = progress['broker_snapshot']
    need(len(broker['leases']) <= MAX_PARTIAL_SHARDS, 'more than eight old claimed shards')
    base = dict(schema='annotation-bounded-speed-handoff-report/v1', spec_sha256=spec_sha,
                old_index=spec['old_index'], old_contract=spec['old_contract'], old_task_id=spec['old_task_id'],
                global_source_count=spec['global_source_count'], previous_retained_count=spec['previous_retained_count'],
                progress=progress, full_sidecar_audit=False, database_created=False,
                unclaimed_JSONL_read_count=0, GPU_executed=False, formal_runtime_admitted=False,
                old_producer_modified=False, training_labels_released=False)
    if not execute_final:
        output.mkdir()
        base.update(status='preview_metadata_only_not_executable', partial_sources_read=0,
                    terminal_verified=False, exact_remaining_source_count=None)
        P.write_new(output/'preview.json', base)
        return base
    need(terminal_pin is not None, 'fresh actual terminal observation required')
    terminal = P.terminal_observation(Path(terminal_pin['path']), terminal_pin['sha256'], spec['old_task_id'], queue)
    with P.old_parent_lock(queue) as parent_lock, contextlib.ExitStack() as locks:
        # First active snapshot is only a hint; bind a new stable snapshot under actual locks.
        progress = inspect_progress(index_path, spec['old_index']['sha256'], queue, spec['old_contract']['sha256'])
        broker = progress['broker_snapshot']
        need(len(broker['leases']) <= MAX_PARTIAL_SHARDS, 'more than eight old claimed shards')
        locks.enter_context(lease_locks(queue, broker['leases']))
        slots = capture_slots(queue, index['shards'], broker)
        partials = []
        retained = []
        partitions = {}
        rows_read = 0
        sidecar_bytes = 0
        validate = P.record_validator()
        portable = load_portable_context(spec,contract['version'])
        proofs = []
        for shard in index['shards'][:broker['next']]:
            if shard['id'] not in broker['leases']:
                continue
            rows = P.read_shard(index_path.parent, shard)
            rows_read += len(rows)
            need(rows_read <= MAX_PARTIAL_ROWS, 'partial source read budget exceeded')
            missing = []
            for source in rows:
                path = P.child(tree, source['sidecar_relpath'])
                if path.exists():
                    sidecar_bytes += path.stat().st_size
                    need(sidecar_bytes <= MAX_SIDECAR_BYTES, 'partial sidecar byte budget exceeded')
                ref = P.retained_sidecar(source, tree, spec['old_contract']['sha256'], validate,
                                         portable_context=portable,portable_sink=proofs.append if portable else None)
                if ref is None:
                    missing.append(source)
                else:
                    retained.append(dict(ref,parent_shard_id=shard['id']))
            partitions[shard['id']]=dict(parent_shard_sha256=shard['sha256'],source_count=shard['count'],retained_count=len(rows)-len(missing),missing_count=len(missing))
            if missing:
                raw = b''.join(P.canonical(s) for s in missing)
                partials.append((shard, missing, raw))
        need(P.byte_sha(P.read_json(queue/'broker.json', 1024**2)[1]) == progress['broker_snapshot_sha256'], 'broker changed while preparing')
        need(capture_slots(queue, index['shards'], broker) == slots, 'completed slots changed while preparing')
        read_pin(spec['old_index'])
        held=(queue/'.parent.lock').stat()
        need((held.st_dev,held.st_ino)==(parent_lock['device'],parent_lock['inode']),'old parent lock replaced during plan')
        pending = []
        for shard, missing, raw in partials:
            pending.append(dict(id='partial-'+shard['id'], count=len(missing), sha256=P.byte_sha(raw),
                                path='partials/'+shard['id']+'.jsonl', storage='local_partial',
                                parent_shard_id=shard['id'], parent_shard_sha256=shard['sha256'],
                                source_record_sha256s=[P.digest(s) for s in missing],source_ids_sha256=P.digest(sorted(s['source_id']for s in missing))))
        for ordinal, shard in enumerate(index['shards'][broker['next']:], broker['next']):
            pending.append(dict(shard, storage='parent_unclaimed', parent_ordinal=ordinal))
        need(len({s['id'] for s in pending}) == len(pending), 'duplicate remaining shard identifier')
        remaining = sum(s['count'] for s in pending)
        preserved = progress['committed_source_lower_bound'] + len(retained)
        need(preserved + remaining == index['source_count'], 'bounded partition does not conserve original scope')
        output.mkdir()
        (output/'partials').mkdir()
        for shard, _, raw in partials:
            P.write_new(output/'partials'/(shard['id']+'.jsonl'), raw)
        P.write_new(output/'retained-partial.jsonl', b''.join(P.canonical(r) for r in retained))
        P.write_new(output/'retained-completed-slots.bin', slots)
        P.write_new(output/'portable-retained-partial.jsonl',b''.join(P.canonical(v)for v in proofs))
        thin = dict(schema=INDEX_SCHEMA, purpose=spec['purpose'], parent_index=spec['old_index'],
                    old_contract=spec['old_contract'], old_queue_root=str(queue), old_sidecar_tree=str(tree),
                    old_task_id=spec['old_task_id'], portable_spec=spec.get('portable_spec'), old_contract_version=contract['version'], new_queue_root=spec['new_queue_root'], new_sidecar_root=spec['new_sidecar_root'],
                    retained_completed_slots_sha256=P.byte_sha(slots), old_queue_identity_sha256=progress['identity_sha256'],
                    parent_cursor=broker['next'], parent_lease_shard_ids=sorted(broker['leases']), lease_partitions=partitions,
                    retained_partial_ref=dict(path='retained-partial.jsonl',sha256=P.byte_sha((output/'retained-partial.jsonl').read_bytes()),count=len(retained)), broker_snapshot_sha256=progress['broker_snapshot_sha256'],
                    source_count=remaining, shard_count=len(pending), shards=pending,
                    global_source_count=spec['global_source_count'], previous_retained_count=spec['previous_retained_count'],
                    parent_preserved_source_count=preserved, retained_partial_source_count=len(retained),
                    retains_previous_producer=True, copy_unclaimed_shards=False,
                    parent_input_uniqueness_is_inherited_not_reaudited=True,
                    future_per_claim_previous_result_check_required=True)
        index_sha = P.write_new(output/'index.json', thin)
        base.update(status='terminal_thin_plan_not_gpu_admission', progress=progress, terminal=terminal,
                    partial_sources_read=rows_read, partial_sidecar_bytes_read=sidecar_bytes,
                    portable_content_proof_count=len(proofs), portable_encoded_bytes_read=sum(v['encoded_bytes_read']for v in proofs),
                    portable_proofs_sha256=P.byte_sha((output/'portable-retained-partial.jsonl').read_bytes()),
                    retained_complete_source_count=progress['committed_source_lower_bound'],
                    retained_partial_source_count=len(retained), exact_remaining_source_count=remaining,
                    old_retained_plus_remaining_equals_parent=True,
                    retained_completed_slots_sha256=P.byte_sha(slots),
                    retained_partial_sha256=P.byte_sha((output/'retained-partial.jsonl').read_bytes()),
                    new_index={'path':str(output/'index.json'), 'sha256':index_sha},
                    new_queue_root=spec['new_queue_root'], new_sidecar_root=spec['new_sidecar_root'])
        P.write_new(output/'report.json', base)
        return base


class BoundedShardReader:
    """仅在消费具体分片时打开该片，不能直接交给旧v1 WorkQueue。"""
    def __init__(self, index_path, index_sha):
        self.path = P.physical(Path(index_path))
        self.index, _ = read_pin({'path':str(self.path), 'sha256':index_sha})
        need(self.index.get('schema') == INDEX_SCHEMA, 'explicit thin index reader required')
        need(self.index.get('purpose')in ('production','cpu_fixture'),'explicit thin index purpose required')
        if self.index['purpose']=='production':need(isinstance(self.index.get('portable_spec'),dict),'production thin index requires actual portable content proof')
        self.parent, _ = read_pin(self.index['parent_index'])
        self.parent_by_id = {s['id']:s for s in self.parent['shards']}
        lease_ids = self.index['parent_lease_shard_ids']
        need(isinstance(lease_ids, list) and len(lease_ids) <= MAX_PARTIAL_SHARDS and len(lease_ids) == len(set(lease_ids)), 'bounded unique parent leases required')
        need(all(sid in {s['id'] for s in self.parent['shards'][:self.index['parent_cursor']]} for sid in lease_ids), 'parent lease outside claimed cursor')
        need(len(self.parent_by_id) == len(self.parent['shards']), 'duplicate parent shard')
        self.by_id = {s['id']:s for s in self.index['shards']}
        self._selected = {}
        self._portable = {}
        partial_origins=set()
        need(len(self.by_id) == len(self.index['shards']) and sum(s['count'] for s in self.index['shards']) == self.index['source_count'], 'thin index count/uniqueness differs')
        for shard in self.index['shards']:
            if shard['storage'] == 'parent_unclaimed':
                ordinal = shard['parent_ordinal']
                need(type(ordinal) is int and self.index['parent_cursor'] <= ordinal < len(self.parent['shards']), 'unclaimed descriptor before parent cursor')
                expected = dict(self.parent['shards'][ordinal], storage='parent_unclaimed', parent_ordinal=ordinal)
                need(shard == expected, 'parent shard descriptor changed')
            else:
                need(shard['parent_shard_id'] not in partial_origins,'duplicate partial parent shard')
                partial_origins.add(shard['parent_shard_id'])
                need(shard['storage'] == 'local_partial' and shard['parent_shard_id'] in self.parent_by_id and
                     shard['parent_shard_id'] in lease_ids and self.parent_by_id[shard['parent_shard_id']]['sha256'] == shard['parent_shard_sha256'], 'invalid local partial origin')
        self._validate_closed_scope()

    def _validate_closed_scope(self):
        """核真实父快照和完整元数据闭包；仅重读最多八个已领取源片。"""
        index=self.index;parent=self.parent;queue=Path(index['old_queue_root'])
        contract,_=read_pin(index['old_contract'])
        need(contract.get('version')==index['old_contract_version']==Path(index['old_sidecar_tree']).name,'old contract version/tree differs')
        progress=inspect_progress(Path(index['parent_index']['path']),index['parent_index']['sha256'],queue,index['old_contract']['sha256'])
        need(progress['broker_snapshot_sha256']==index['broker_snapshot_sha256']and progress['identity_sha256']==index['old_queue_identity_sha256'],'actual old broker/identity differs from thin index')
        broker=progress['broker_snapshot']
        need(broker['next']==index['parent_cursor']and sorted(broker['leases'])==index['parent_lease_shard_ids'],'actual parent cursor/lease differs')
        need(P.byte_sha(capture_slots(queue,parent['shards'],broker))==index['retained_completed_slots_sha256'],'actual old completed slots differ')
        expected=[dict(shard,storage='parent_unclaimed',parent_ordinal=i)for i,shard in enumerate(parent['shards'][broker['next']:],broker['next'])]
        actual=[shard for shard in index['shards']if shard['storage']=='parent_unclaimed']
        need(actual==expected,'unclaimed parent descriptor coverage is incomplete or changed')
        partitions=index['lease_partitions'];need(isinstance(partitions,dict)and set(partitions)==set(broker['leases']),'partial partition does not cover exact parent leases')
        pin=index['retained_partial_ref'];ledger_path=P.child(self.path.parent,pin['path'])
        need(stat.S_ISREG(ledger_path.lstat().st_mode),'retained partial ledger must be regular')
        refs,raw=P.read_json_lines(ledger_path)
        need(is_sha(pin['sha256'])and P.byte_sha(raw)==pin['sha256']and len(refs)==pin['count']and len({r['source_id']for r in refs})==len(refs),'retained partial ledger count/SHA/uniqueness differs')
        need(all(r.get('parent_shard_id')in partitions for r in refs),'retained partial ledger has foreign shard')
        partials={s['parent_shard_id']:s for s in index['shards']if s['storage']=='local_partial'}
        read_count=0;retained_total=0;missing_total=0
        for sid,part in partitions.items():
            descriptor=self.parent_by_id[sid]
            need(part.get('parent_shard_sha256')==descriptor['sha256']and part.get('source_count')==descriptor['count']and all(type(part.get(k))is int and part[k]>=0 for k in ('retained_count','missing_count'))and part['retained_count']+part['missing_count']==descriptor['count'],'partial missing/retained denominator differs')
            original=P.read_shard(Path(index['parent_index']['path']).parent,descriptor);read_count+=len(original)
            need(read_count<=MAX_PARTIAL_ROWS,'partial closure read budget exceeded')
            original_by_id={r['source_id']:r for r in original};selected=[r for r in refs if r['parent_shard_id']==sid]
            need(len(selected)==part['retained_count'],'retained ledger does not bind partial count')
            for ref in selected:
                need(ref['source_id']in original_by_id and ref['source_record_sha256']==P.digest(original_by_id[ref['source_id']])and ref['producer_contract_sha256']==index['old_contract']['sha256'],'retained source is not in original claimed shard')
                source=original_by_id[ref['source_id']];path=P.child(Path(index['old_sidecar_tree']),source['sidecar_relpath']);commit=path.with_name(path.name+'.commit.json')
                need(str(path)==ref['sidecar_path']and str(commit)==ref['commit_path'],'retained producer output path differs')
                marker,marker_raw=P.read_json(commit,65536)
                expected_marker=dict(schema='redimnet2-sidecar-commit/v1',source_id=source['source_id'],job_key=P.old_job_key(source,index['old_contract']['sha256']),contract_sha256=index['old_contract']['sha256'],source_record_sha256=P.digest(source),sidecar_sha256=ref['sidecar_sha256'],bytes=path.stat().st_size)
                need(marker==expected_marker and P.byte_sha(marker_raw)==ref['commit_sha256'],'retained producer commit binding differs')
            retained_ids={r['source_id']for r in selected}
            if part['missing_count']:
                need(sid in partials,'missing partial shard omitted')
                shard=partials[sid];rows=P.read_shard(self.path.parent,shard)
                need(shard['count']==part['missing_count']and [P.digest(r)for r in rows]==shard['source_record_sha256s']and P.digest(sorted(r['source_id']for r in rows))==shard['source_ids_sha256'],'partial missing rows digest/count differs')
                need(all(original_by_id.get(r['source_id'])==r for r in rows),'partial missing source is foreign')
                missing_ids={r['source_id']for r in rows}
            else:
                need(sid not in partials,'fully retained lease has unexpected missing shard');missing_ids=set()
            need(not(retained_ids&missing_ids)and retained_ids|missing_ids==set(original_by_id),'partial source coverage is not complete/disjoint')
            retained_total+=len(retained_ids);missing_total+=len(missing_ids)
        need(retained_total==index['retained_partial_source_count']and progress['committed_source_lower_bound']+retained_total==index['parent_preserved_source_count'],'parent preserved denominator differs')
        need(missing_total+sum(s['count']for s in expected)==index['source_count'],'remaining thin source denominator differs from exact parent coverage')
        need(type(index['global_source_count'])is int and type(index['previous_retained_count'])is int and index['previous_retained_count']>=0 and index['previous_retained_count']+parent['source_count']==index['global_source_count'],'inherited global denominator differs')
        if index['purpose']=='production':need(index['global_source_count']==11521178,'production inherited global scope differs')

    def read(self, shard_id):
        need(shard_id in self.by_id, 'unknown selected shard')
        shard = self.by_id[shard_id]
        root = Path(self.index['parent_index']['path']).parent if shard['storage'] == 'parent_unclaimed' else self.path.parent
        rows = P.read_shard(root, shard)
        if shard['storage'] == 'local_partial':
            need([P.digest(s) for s in rows] == shard['source_record_sha256s'], 'partial source record differs')
            original = P.read_shard(Path(self.index['parent_index']['path']).parent, self.parent_by_id[shard['parent_shard_id']])
            original_by_id = {s['source_id']:s for s in original}
            need(all(original_by_id.get(s['source_id']) == s for s in rows), 'partial source is foreign to bound parent shard')
        self._selected[shard_id] = {r['source_id']:r for r in rows}
        self._portable[shard_id] = load_portable_context(self.index,self.index['old_contract_version'])
        if len(self._selected) > 8:
            oldest=next(iter(self._selected));self._selected.pop(oldest);self._portable.pop(oldest,None)
        return rows

    def lookup_previous_for_claim(self, shard_id, source):
        """新worker领取片后、推理前检查旧结果；返回原producer，不生成新job。

        此方法不取得claim或GPU锁，调用者仍要持有真实新claim和旧writer父锁。
        """
        need(self._selected.get(shard_id, {}).get(source['source_id']) == source, 'source outside selected claimed shard')
        proofs=[];portable=self._portable.get(shard_id)
        ref=P.retained_sidecar(source,Path(self.index['old_sidecar_tree']),self.index['old_contract']['sha256'],
                              P.record_validator(),portable_context=portable,portable_sink=proofs.append if portable else None)
        if ref is not None and portable:
            need(len(proofs)==1,'one existing source requires one actual content proof')
            ref=dict(ref,portable_source_evidence=proofs[0])
        return ref


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True, type=Path)
    p.add_argument('--spec-sha', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--execute-final', action='store_true')
    p.add_argument('--terminal', type=Path)
    p.add_argument('--terminal-sha')
    a = p.parse_args()
    terminal = {'path':str(a.terminal),'sha256':a.terminal_sha} if a.terminal else None
    result = prepare(a.spec, a.spec_sha, a.output, terminal_pin=terminal, execute_final=a.execute_final)
    print(json.dumps({k:result[k] for k in ('status','formal_runtime_admitted','database_created','unclaimed_JSONL_read_count')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
