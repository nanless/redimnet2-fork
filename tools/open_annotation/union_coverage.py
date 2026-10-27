"""联合核对旧结果与新剩余输入，使用磁盘索引保证每源唯一、不重不漏。

只读取外部已固定输入。临时/显式SQLite仅是新的消费索引，不是生产队列；
不加载模型、不读原音、不创建旧锁或修复旧侧车，也不授GPU与训练资格。
"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import digest, job_key, model_provenance
from readonly_progress import read_small, strict_json, inspect_progress, open_readonly
from prepare_speed_continuation import physical, child, old_parent_lock, read_json_lines
from schema import assert_valid_record
from source_registry import source_id
from work_queue import decode_completion_slot, SLOT_BYTES

SCHEMA = 'annotation-union-coverage/v1'
PRODUCTION_COUNT = 11521178
PRODUCTION_DOMAINS = {'voxceleb2': 1092009, 'voxblink2': 9904382, 'cnceleb2': 524787}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read_json(path, expected=None, maximum=64*1024**2):
    p = physical(path); value, raw = read_small(p, maximum)
    pin = hashlib.sha256(raw).hexdigest()
    need(expected is None or expected == pin, 'pinned JSON SHA differs: ' + str(p))
    return value, pin


def ref(value):
    need(isinstance(value, dict) and set(value) == {'path','sha256'}, 'explicit path/SHA reference required')
    need(Path(value['path']).is_absolute(), 'absolute input path required')
    return read_json(value['path'], value['sha256'])[0]


def shards(index, index_path):
    need(index.get('schema') == 'redimnet2-shard-index/v1', 'wrong index schema')
    need(type(index.get('source_count')) is int and index['source_count'] > 0, 'positive source denominator required')
    seen = set(); total = 0
    for ordinal, item in enumerate(index['shards']):
        need(item['id'] not in seen, 'duplicate shard ID'); seen.add(item['id'])
        child(Path(index_path).parent, item['id'])
        need(type(item['count']) is int and item['count'] > 0, 'invalid shard count')
        rows, raw = read_json_lines(child(Path(index_path).parent, item['path']))
        need(hashlib.sha256(raw).hexdigest() == item['sha256'] and len(rows) == item['count'], 'shard bytes/count differ')
        need(len({s['source_id'] for s in rows}) == len(rows), 'duplicate source inside shard')
        for row in rows:
            need(row.get('schema') == 'redimnet2-source/v1' and
                 row['source_id'] == source_id(row['dataset'], row['relative_path']), 'source path/ID identity differs')
            need(row['sidecar_relpath']==row['dataset']+'/'+row['relative_path']+'.annotation.json',
                 'source sidecar path identity differs')
            child(Path('/'), row['sidecar_relpath'])
        total += len(rows)
        yield ordinal, item, rows
    need(total == index['source_count'], 'index source denominator differs')


def producer(value, allow_missing=False, index_pin=None):
    need(isinstance(value.get('id'),str) and value['id'] and '/' not in value['id'], 'invalid producer ID')
    c = ref(value['contract'])
    need(c.get('schema') == 'redimnet2-production-contract/v2' and
         c.get('model_API_calls_allowed') == 0 and c.get('release_training_labels') is False,
         'producer contract permits unsupported format/API/label release')
    queue, tree = physical(value['queue_root']), physical(value['sidecar_tree'])
    need(queue != tree and queue not in tree.parents and tree not in queue.parents, 'queue/labels overlap')
    if not allow_missing or tree.exists():
        identity, _ = read_json(tree/'identity.json')
        need(identity == {'schema':'redimnet2-sidecar-tree/v1','version':c['version'],
                          'contract_sha256':value['contract']['sha256']}, 'producer namespace identity differs')
    queue_identity=physical(queue/'identity.json')
    if queue_identity.exists():
        expected_index=index_pin if index_pin is not None else value.get('index')
        need(expected_index is not None,'queue index binding missing')
        ix=ref(expected_index);identity,_=read_json(queue_identity)
        need(identity=={'schema':'redimnet2-work-queue/v1','index_sha256':expected_index['sha256'],
             'contract_sha256':value['contract']['sha256'],'source_count':ix['source_count'],
             'shard_count':len(ix['shards'])},'producer queue identity differs')
    return c, queue, tree


def read_result(source, value, expected_record_sha=None, expected_commit_sha=None, *, contract):
    """重读已保存原声身份字段及提交绑定；不重读或哈希原声。"""
    path = child(physical(value['sidecar_tree']), source['sidecar_relpath'])
    record, record_sha = read_json(path, expected_record_sha, 32*1024**2)
    commit_path = path.with_name(path.name+'.commit.json')
    commit, commit_sha = read_json(commit_path, expected_commit_sha, 64*1024)
    assert_valid_record(record)
    contract_sha = value['contract']['sha256']; key = job_key(source, contract_sha)
    need(record['source'] == source and record['contract']['sha256'] == contract_sha and
         record['contract']['version'] == contract['version'] and
         record['execution']['job_key'] == key, 'selected sidecar source/producer differs')
    # 合同只在入口按pin读取一次；逐源检查实际模型和运行策略，不能仅相信合同SHA字段。
    if 'models' in contract or 'runtime' in contract:
        need(record['provenance'].get('models') == model_provenance(contract) and
             record['provenance'].get('runtime') == contract['runtime'],
             'selected sidecar model/runtime provenance differs from pinned producer')
    need(record['decision']['usable_for_training'] is False and record['decision']['usable_for_evaluation'] is False,
         'automatic training/evaluation eligibility release refused')
    expected = {'schema':'redimnet2-sidecar-commit/v1','source_id':source['source_id'],
                'job_key':key,'contract_sha256':contract_sha,'source_record_sha256':digest(source),
                'sidecar_sha256':record_sha,'bytes':physical(path).stat().st_size}
    need(commit == expected, 'sidecar commit identity differs')
    return record, record_sha, commit_sha


def _complete_chunk(index_pin, descriptor, ordinal, queue, producer_value, rows, expected_chunk_sha=None,
                    expected_slot_sha=None, *, contract):
    fd=open_readonly(physical(queue/'completion_map.bin'))
    try:raw=os.pread(fd,SLOT_BYTES,ordinal*SLOT_BYTES)
    finally:os.close(fd)
    slot = decode_completion_slot(raw)
    import struct
    size=struct.unpack('>I',raw[8:12])[0]
    need(raw[44+size:]==bytes(SLOT_BYTES-44-size),'completion slot trailing bytes differ')
    need(expected_slot_sha is None or hashlib.sha256(raw).hexdigest() == expected_slot_sha, 'retained completion slot changed')
    need(slot['index_sha256'] == index_pin and slot['shard_id'] == descriptor['id'] and
         slot['shard_sha256'] == descriptor['sha256'] and slot['source_count'] == descriptor['count'] and
         slot['contract_sha256'] == producer_value['contract']['sha256'], 'completion input identity differs')
    need(expected_chunk_sha is None or slot['commit_sha256'] == expected_chunk_sha, 'retained chunk pin differs')
    chunk, chunk_sha = read_json(child(queue, 'completed/'+descriptor['id']+'.json'), slot['commit_sha256'])
    need(chunk.get('schema') == 'redimnet2-chunk-commit/v1' and
         all(chunk.get(k) == slot[k] for k in ('index_sha256','contract_sha256','shard_id','shard_sha256','source_count')),
         'chunk receipt identity differs')
    receipts = {r['source_id']:r for r in chunk['sources']}
    need(len(receipts) == len(chunk['sources']) == len(rows) and set(receipts) == {s['source_id'] for s in rows}, 'chunk source coverage differs')
    need(digest(sorted(receipts)) == slot['source_ids_sha256'], 'chunk source ID digest differs')
    decisions = collections.Counter()
    for source in rows:
        receipt = receipts[source['source_id']]
        record, record_sha, commit_sha = read_result(source, producer_value,
            receipt['sidecar_sha256'], receipt['commit_sha256'], contract=contract)
        need(receipt['job_key'] == job_key(source,producer_value['contract']['sha256']) and
             receipt['sidecar_relpath'] == source['sidecar_relpath'] and
             receipt['sidecar_bytes'] == child(Path(producer_value['sidecar_tree']),source['sidecar_relpath']).stat().st_size and
             receipt['decision'] == record['decision']['state'], 'chunk per-source receipt differs')
        decisions[record['decision']['state']] += 1
        yield source, record_sha, commit_sha, record['decision']['state']
    need({k:decisions[k] for k in ('candidate','unknown','failed')} == slot['decision_counts'], 'chunk decision count differs')


def _terminal(plan, old, mode):
    proof = plan.get('terminal_observation'); lock = plan.get('old_parent_lock')
    need(isinstance(proof,dict) and isinstance(lock,dict) and proof.get('platform_authentication_performed_by_this_tool') is False,
         'final plan lacks external terminal/lock evidence')
    obs, _ = read_json(proof['path'],proof['sha256'],1024**2)
    need(obs.get('schema') == 'annotation-writer-terminal-observation/v1' and
         obs.get('task_id') == proof.get('task_id') and obs.get('queue_root') == old['queue_root'] and
         obs.get('state') in {'Stopped','Failed','Succeeded','Completed'}, 'old writer terminal observation differs')
    workers=obs.get('workers');need(isinstance(workers,list) and len(workers)==1 and
        workers[0].get('name') == obs['task_id']+'-worker-0' and
        workers[0].get('state') in {'Stopped','Failed','Succeeded','Completed'}, 'old worker still active/unobserved')
    observed=datetime.datetime.fromisoformat(obs['observed_utc'])
    need(observed.tzinfo is not None, 'terminal observation timezone missing')
    age=(datetime.datetime.now(datetime.timezone.utc)-observed).total_seconds()
    need(age>=0 and (mode!='plan' or age<=3600), 'plan terminal observation stale or future')
    need(lock.get('existing_lock_opened_readonly') is True and lock.get('lock_created') is False and
         lock.get('path') == str(Path(old['queue_root'])/'.parent.lock'), 'final preparation lock proof differs')
    return lock


def _load_unclaimed_ledger(plan,db):
    """逐行载入准备器的新保留账本，ID唯一性放SQLite，不整库读入RAM。"""
    db.execute('CREATE TABLE retained_extras(sid TEXT PRIMARY KEY NOT NULL,value TEXT NOT NULL)')
    item=plan.get('retained_unclaimed_sources_ref');expected=plan.get('retained_unclaimed_source_count',0)
    need(type(expected)is int and expected>=0,'invalid unclaimed retained count')
    if item is None:
        need(expected==0,'unclaimed retained sources lack pinned ledger');return
    need(set(item)=={'path','sha256','count'} and item['count']==expected,'unclaimed ledger count/ref differs')
    path=physical(item['path']);fd=open_readonly(path);h=hashlib.sha256();count=0
    try:
        before=os.fstat(fd)
        with os.fdopen(fd,'rb',closefd=False)as stream:
            while True:
                line=stream.readline(65537)
                if not line:break
                need(len(line)<=65536 and line.strip(),'unclaimed ledger row exceeds bound or is blank')
                h.update(line);value=strict_json(line)
                need(isinstance(value.get('source_id'),str),'unclaimed ledger source ID missing')
                try:db.execute('INSERT INTO retained_extras VALUES(?,?)',(value['source_id'],line.decode('utf-8')))
                except sqlite3.IntegrityError as exc:raise ValueError('duplicate unclaimed retained source')from exc
                count+=1
        after=os.fstat(fd)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'unclaimed ledger changed during read')
    finally:os.close(fd)
    need(count==expected and h.hexdigest()==item['sha256'],'unclaimed ledger SHA/count differs')


def _select_old_source(db,source,expected,old,tree,contract):
    need(expected['source_record_sha256']==digest(source) and expected['producer_contract_sha256']==old['contract']['sha256'] and
         expected['sidecar_path']==str(child(tree,source['sidecar_relpath'])) and
         expected['commit_path']==str(child(tree,source['sidecar_relpath']+'.commit.json')), 'retained old source identity differs')
    before=db.execute('SELECT selected FROM sources WHERE sid=?',(source['source_id'],)).fetchone()[0]
    need(before is None,'source retained twice')
    record,rs,cs=read_result(source,old,expected['sidecar_sha256'],expected['commit_sha256'],contract=contract)
    need(expected['decision']==record['decision']['state'],'retained decision changed')
    db.execute('UPDATE sources SET selected=?,record_sha=?,commit_sha=?,decision=? WHERE sid=?',(old['id'],rs,cs,record['decision']['state'],source['source_id']))
    return record['decision']['state']


def _validate(spec, spec_sha, mode, db):
    need(mode in {'plan','complete'}, 'unknown union mode')
    need(spec.get('schema') == SCHEMA and spec.get('purpose') in {'production','cpu_fixture'}, 'wrong union schema/purpose')
    need(spec.get('selection_policy') == 'old_retained_else_new_remaining', 'explicit source selection policy required')
    need(type(spec.get('expected_full_source_count')) is int and spec['expected_full_source_count']>0,
         'full source denominator must be a positive integer')
    need(spec.get('release_training_labels') is False and spec.get('model_API_calls_allowed') == 0,
         'union cannot release labels or call model APIs')
    full = ref(spec['full_index']); plan = ref(spec['continuation_plan'])
    need(full.get('source_count') == spec['expected_full_source_count'], 'original denominator differs')
    if spec['purpose']=='production':
        need(spec['expected_full_source_count']==PRODUCTION_COUNT and full.get('domains')==PRODUCTION_DOMAINS and
             full.get('RIR_or_MUSAN_or_Child_or_evaluation_in_this_job') is False, 'formal full denominator/domain scope differs')
    old,new=spec['old_producer'],spec['new_producer']
    need(old['id'] != new['id'], 'producer IDs overlap')
    oldc,oq,ot=producer(old,index_pin=spec['full_index']);newc,nq,nt=producer(new,allow_missing=mode=='plan')
    need(oq != nq and oq not in nq.parents and nq not in oq.parents and ot != nt and ot not in nt.parents and nt not in ot.parents,
         'old/new producer namespaces overlap')
    if spec['purpose']=='production':
        need(oldc.get('purpose')==newc.get('purpose')=='silver_annotation', 'formal union requires silver producer contracts')
        for c in (oldc,newc):
            need(isinstance(c.get('runtime'),dict) and bool(c['runtime']) and
                 isinstance(c.get('models'),dict) and set(c['models'])=={'asr','aligner'},
                 'formal producer requires complete models/runtime provenance')
            for m in c['models'].values():
                need(isinstance(m,dict) and isinstance(m.get('path'),str) and Path(m['path']).is_absolute() and
                     isinstance(m.get('files'),dict) and bool(m['files']), 'formal model file identity missing')
    need(plan.get('schema')=='annotation-speed-continuation-plan/v1' and
         plan.get('status')=='final_prepared_not_started' and plan.get('formal_runtime_admitted') is False and
         plan.get('old_index')==spec['full_index'] and plan.get('old_contract',{}).get('sha256')==old['contract']['sha256'] and
         plan.get('old_queue')==str(oq) and plan.get('old_sidecar_tree')==str(ot) and
         plan.get('remaining_index')==new['index'], 'final continuation plan/input identity differs')
    need(plan.get('new_inference_count')==0 and plan.get('old_queue_mutated') is False and
         plan.get('old_results_rewritten') is False and plan.get('stage_cache_relabelled') is False,
         'old producer preservation declaration missing')
    lock_proof=_terminal(plan,old,mode)
    db.execute('CREATE TABLE sources(sid TEXT PRIMARY KEY, source_sha TEXT NOT NULL, shard INTEGER NOT NULL, row_number INTEGER NOT NULL, relative TEXT NOT NULL, selected TEXT, record_sha TEXT, commit_sha TEXT, decision TEXT)')
    db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    _load_unclaimed_ledger(plan,db)
    domains=collections.Counter();total=0;retained=0;remaining=0;decisions=collections.Counter()
    retained_chunks={s['id']:s for s in plan['retained_complete_shards']}
    need(len(retained_chunks)==len(plan['retained_complete_shards']), 'duplicate retained chunk')
    partial={s['source_id']:s for s in plan['retained_partial_sources']}
    need(len(partial)==len(plan['retained_partial_sources']), 'duplicate retained partial source')
    remaining_index=ref(new['index']);need(remaining_index.get('parent_full_index')==spec['full_index'], 'remaining index parent differs')
    with old_parent_lock(oq) as current_lock:
        need(all(current_lock[k]==lock_proof[k] for k in ('path','device','inode')), 'old lock inode changed')
        _,broker_before=read_json(oq/'broker.json')
        need(broker_before==plan['old_progress_snapshot']['broker_snapshot_sha256'], 'old writer broker changed since final preparation')
        old_progress=inspect_progress(spec['full_index']['path'],spec['full_index']['sha256'],oq,old['contract']['sha256'])
        need(old_progress['broker_snapshot_sha256']==broker_before,'old queue snapshot changed before coverage check')
        state=old_progress['broker_snapshot'];expected_retained={s['id'] for i,s in enumerate(full['shards'])
            if i<state['next'] and s['id'] not in state['leases']}
        need(set(retained_chunks)==expected_retained,'final plan omits/adds completed old chunks')
        for ordinal,descriptor,rows in shards(full,spec['full_index']['path']):
            for number,source in enumerate(rows):
                if spec['purpose']=='production':need(set(source.get('roles',[])) and
                    set(source['roles'])<={'train_ptn','train_lm'},'production source has conflicting roles')
                try:db.execute('INSERT INTO sources(sid,source_sha,shard,row_number,relative)VALUES(?,?,?,?,?)',
                    (source['source_id'],digest(source),ordinal,number,source['sidecar_relpath']))
                except sqlite3.IntegrityError as exc:raise ValueError('duplicate source in full original index') from exc
                domains[source['dataset']]+=1;total+=1
            if descriptor['id'] in retained_chunks:
                expected=retained_chunks.pop(descriptor['id'])
                need(all(expected.get(k)==descriptor.get(k) for k in ('id','path','count','sha256')), 'retained shard descriptor differs')
                for source,rs,cs,decision in _complete_chunk(spec['full_index']['sha256'],descriptor,ordinal,oq,old,rows,
                        expected['chunk_receipt_sha256'],expected['completion_slot_sha256'],contract=oldc):
                    db.execute('UPDATE sources SET selected=?,record_sha=?,commit_sha=?,decision=? WHERE sid=?',(old['id'],rs,cs,decision,source['source_id']))
                    retained+=1;decisions[decision]+=1
            for source in rows:
                extra=db.execute('SELECT value FROM retained_extras WHERE sid=?',(source['source_id'],)).fetchone()
                if source['source_id'] in partial:
                    expected=partial.pop(source['source_id'])
                    need(extra is None and ordinal<state['next'] and descriptor['id'] in state['leases'],
                         'partial retained source is duplicated or not from a leased shard')
                    decision=_select_old_source(db,source,expected,old,ot,oldc)
                    retained+=1;decisions[decision]+=1
                elif extra is not None:
                    need(ordinal>=state['next'],'unclaimed retained source is not in an unclaimed shard')
                    decision=_select_old_source(db,source,strict_json(extra[0]),old,ot,oldc)
                    db.execute('DELETE FROM retained_extras WHERE sid=?',(source['source_id'],))
                    retained+=1;decisions[decision]+=1
                elif descriptor['id'] not in expected_retained:
                    # 包括未领取片中的预置旧结果。不能因准备器未列出而重新推理。
                    old_path=child(ot,source['sidecar_relpath'])
                    old_commit=old_path.with_name(old_path.name+'.commit.json')
                    physical(old_commit)
                    need(not old_path.exists() and not old_commit.exists(),
                         'final plan omits an already published/orphan old source')
            db.commit()
        need(not retained_chunks and not partial, 'retained source/chunk absent from full index')
        need(db.execute('SELECT COUNT(*) FROM retained_extras').fetchone()[0]==0,'unclaimed retained source absent from full index')
        need(total==full['source_count'] and dict(domains)==full.get('domains'), 'original full domain counts differ')
        if spec['purpose']=='production':need(dict(domains)==PRODUCTION_DOMAINS, 'production row domains differ')
        if mode=='complete':
            progress=inspect_progress(new['index']['path'],new['index']['sha256'],nq,new['contract']['sha256'])
            need(progress['committed_source_lower_bound']==remaining_index['source_count'] and progress['snapshot_leased_chunks_excluded']==0,
                 'new remaining producer not complete')
            new_broker=progress['broker_snapshot_sha256']
        for ordinal,descriptor,rows in shards(remaining_index,new['index']['path']):
            proofs={}
            if mode=='complete':proofs={s['source_id']:(rs,cs,d) for s,rs,cs,d in _complete_chunk(new['index']['sha256'],descriptor,ordinal,nq,new,rows,contract=newc)}
            for source in rows:
                found=db.execute('SELECT source_sha,relative,selected FROM sources WHERE sid=?',(source['source_id'],)).fetchone()
                need(found is not None and found[0]==digest(source) and found[1]==source['sidecar_relpath'], 'remaining source absent/changed from full index')
                need(found[2] is None, 'retained/new source overlap or duplicate remaining source')
                rs,cs,d=proofs.get(source['source_id'],(None,None,None))
                db.execute('UPDATE sources SET selected=?,record_sha=?,commit_sha=?,decision=? WHERE sid=?',(new['id'],rs,cs,d,source['source_id']))
                remaining+=1
                if d is not None:decisions[d]+=1
            db.commit()
        need(retained==plan['retained_source_count'] and remaining==plan['remaining_source_count']==remaining_index['source_count'], 'plan retained/remaining counts differ')
        missing=db.execute('SELECT COUNT(*) FROM sources WHERE selected IS NULL').fetchone()[0]
        need(missing==0 and retained+remaining==total, 'union leaves missing sources')
        _,broker_after=read_json(oq/'broker.json');need(broker_after==broker_before,'old writer changed during union validation')
        if mode=='complete':need(read_json(nq/'broker.json')[1]==new_broker,'new producer changed during complete validation')
    report={'schema':'annotation-union-validation/v1','spec_sha256':spec_sha,'mode':mode,'purpose':spec['purpose'],
            'source_count':total,'retained_source_count':retained,'remaining_source_count':remaining,
            'domains':dict(domains),'decision_counts':{k:decisions[k]for k in ('candidate','unknown','failed')},
            'decision_count_scope':'all_selected' if mode=='complete' else 'retained_only',
            'exact_union_and_disjoint':True,
            'complete_sidecars_reopened':mode=='complete','old_retained_sidecars_reopened':True,
            'formal_runtime_admitted':False,'training_or_evaluation_eligibility_released':False,
            'GPU_executed':False,'source_audio_rehashed':False,'platform_authentication_performed':False}
    if mode=='complete':
        # 消费千万条时按原顺序流式读取，避免每轮再对全表排序。
        db.execute('CREATE INDEX source_sequence ON sources(shard,row_number)')
    for key,value in {'spec':spec,'spec_sha256':spec_sha,'validation':report}.items():
        db.execute('INSERT INTO metadata VALUES(?,?)',(key,json.dumps(value,sort_keys=True,ensure_ascii=False)))
    db.commit();return report


def validate_union(spec_path, spec_sha, mode='plan', *, output_index=None):
    spec=ref({'path':str(spec_path),'sha256':spec_sha})
    def build(path):
        db=sqlite3.connect(str(path));db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-8192')
        try:return _validate(spec,spec_sha,mode,db)
        finally:db.close()
    if output_index is None:
        with tempfile.TemporaryDirectory(prefix='annotation-union-readonly-audit-') as tmp:
            return build(Path(tmp)/'consumer.sqlite')
    output=physical(output_index);need(not output.exists() and output.parent.is_dir(),'consumer index must be a new external file')
    protected=[physical(spec['full_index']['path']).parent,physical(spec['continuation_plan']['path']).parent]
    for p in (spec['old_producer'],spec['new_producer']):protected.extend([physical(p['queue_root']),physical(p['sidecar_tree']),physical(p['contract']['path']).parent])
    need(all(output!=p and p not in output.parents for p in protected),'consumer index overlaps producer input/output tree')
    fd,tmp=tempfile.mkstemp(prefix='.consumer-building-',suffix='.sqlite',dir=output.parent);os.close(fd)
    try:
        report=build(Path(tmp));os.link(tmp,output);os.unlink(tmp)
        parent_fd=os.open(str(output.parent),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
        report['consumer_index']={'path':str(output),'sha256':sha_file(output)}
        return report
    except BaseException:
        # 未通过的临时SQLite保留证据，不能留下貌似已通过的最终消费索引。
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True,type=Path)
    p.add_argument('--spec-sha',required=True);p.add_argument('--mode',choices=('plan','complete'),default='plan')
    p.add_argument('--consumer-index',type=Path);a=p.parse_args()
    print(json.dumps(validate_union(a.spec,a.spec_sha,a.mode,output_index=a.consumer_index),ensure_ascii=False))


if __name__=='__main__':main()
