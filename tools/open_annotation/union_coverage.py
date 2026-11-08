"""联合核对旧结果与新剩余输入，使用磁盘索引保证每源唯一、不重不漏。

只读取外部已固定输入。临时/显式SQLite仅是新的消费索引，不是生产队列；
不加载模型、不读原音、不创建旧锁或修复旧侧车，也不授GPU与训练资格。
"""
from __future__ import annotations
import argparse
import collections
import contextlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import datetime
import builtins
import ctypes
import hashlib
import json
import os
import multiprocessing
import signal
from pathlib import Path
import sqlite3
import tempfile
import time
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import digest, job_key, model_provenance
from readonly_progress import read_small, strict_json, inspect_progress, open_readonly
from prepare_speed_continuation import physical, child, old_parent_lock, read_json_lines, source_roles
from schema import assert_valid_record
from source_registry import source_id
from work_queue import decode_completion_slot, SLOT_BYTES

SCHEMA = 'annotation-union-coverage/v1'
PRODUCTION_COUNT = 11521178
PRODUCTION_DOMAINS = {'voxceleb2': 1092009, 'voxblink2': 9904382, 'cnceleb2': 524787}
MAX_RETAINED_READ_WORKERS = 4
MAX_RETAINED_READ_BATCH = 256


def need(ok, message):
    if not ok: raise ValueError(message)


def retained_read_workers(spec):
    """显式纯读容量；缺省保持串行，bool/float/字符串都不能充整数。"""
    workers=spec.get('retained_read_workers',1)
    need(type(workers)is int and 1<=workers<=MAX_RETAINED_READ_WORKERS,
         'retained_read_workers must be an explicit integer in 1..4')
    return workers


def retained_read_backend(spec):
    backend=spec.get('retained_read_backend','thread')
    need(type(backend)is str and backend in {'thread','process'},'retained_read_backend must be thread or process')
    if backend=='process' and retained_read_workers(spec)>1 and sys.platform!='linux':
        need(spec.get('purpose')=='cpu_fixture','production process readers require Linux parent-death guarding')
    return backend


_CPU_FORBIDDEN_MODULES={'torch','qwen_asr','vllm'}
_CPU_PROCESS_STATE=None
_CPU_SOURCE_NAMES=('union_coverage.py','contracts.py','readonly_progress.py','prepare_speed_continuation.py',
                   'schema.py','normalization.py','quality_axes.py','source_clock.py','source_registry.py',
                   'work_queue.py','alignment_observer.py')
_CPU_ENV={'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'none','HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':''}


def _linux_start_ticks(pid):
    raw=Path('/proc')/str(pid)/'stat';text=raw.read_text()
    fields=text[text.rfind(')')+2:].split()
    need(len(fields)>19 and fields[19].isdigit(),'CPU process start identity unavailable')
    return fields[19]


def _same_cpu_parent(pid,start,*,getppid=None,start_reader=None):
    getppid=getppid or os.getppid;start_reader=start_reader or _linux_start_ticks
    need(type(pid)is int and pid>1 and getppid()==pid,'CPU worker parent PID changed')
    if start is not None:need(start_reader(pid)==start,'CPU worker parent start identity changed')


def _linux_arm_parent_death(pid,start,*,getppid=None,start_reader=None,prctl=None):
    """SET/GET不可忽略SIGKILL；设置前后均核父PID/start，关闭先死再arm的竞态。"""
    need(isinstance(start,str)and start.isdigit(),'exact Linux parent start required')
    _same_cpu_parent(pid,start,getppid=getppid,start_reader=start_reader)
    if prctl is None:
        prctl=ctypes.CDLL(None,use_errno=True).prctl;prctl.restype=ctypes.c_int
    zero=ctypes.c_ulong(0)
    need(prctl(ctypes.c_int(1),ctypes.c_ulong(signal.SIGKILL),zero,zero,zero)==0,
         'PR_SET_PDEATHSIG failed: '+str(ctypes.get_errno()))
    actual=ctypes.c_int(0)
    need(prctl(ctypes.c_int(2),ctypes.byref(actual),zero,zero,zero)==0 and actual.value==signal.SIGKILL,
         'PR_GET_PDEATHSIG did not confirm SIGKILL')
    _same_cpu_parent(pid,start,getppid=getppid,start_reader=start_reader)


def _cpu_source_pins():
    here=physical(Path(__file__).resolve().parent)
    return {name:sha_file(here/name)for name in _CPU_SOURCE_NAMES}


def _no_heavy_imports():
    need(not any(name.split('.')[0]in _CPU_FORBIDDEN_MODULES for name in sys.modules),
         'CPU retained readers refuse a parent/worker with heavy inference modules imported')


def _init_retained_cpu_process(producer_value,contract,parent_pid,parent_start,source_root,source_pins,purpose):
    global _CPU_PROCESS_STATE
    _no_heavy_imports()
    for key,value in _CPU_ENV.items():
        need(os.environ.get(key)==value,'CPU spawn environment not hidden before initializer: '+key)
    _same_cpu_parent(parent_pid,parent_start)
    linux=sys.platform=='linux'
    if linux:_linux_arm_parent_death(parent_pid,parent_start)
    else:need(purpose=='cpu_fixture','non-Linux process reader is synthetic CPU only')
    here=physical(Path(__file__).resolve().parent)
    need(str(here)==source_root and _cpu_source_pins()==source_pins,'CPU worker validator source identity changed')
    original_import=builtins.__import__
    def cpu_import(name,*args,**kwargs):
        need(name.split('.')[0]not in _CPU_FORBIDDEN_MODULES,'heavy inference import forbidden in CPU reader: '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=cpu_import
    _CPU_PROCESS_STATE={'producer':producer_value,'contract':contract,'parent_pid':parent_pid,'parent_start':parent_start}
    receipt={'schema':'annotation-retained-cpu-worker/v1','backend':'process','start_method':'spawn',
        'pid':os.getpid(),'start_ticks':_linux_start_ticks(os.getpid())if linux else None,
        'parent_pid':parent_pid,'parent_start_ticks':parent_start,'actual_parent_pid':os.getppid(),
        'linux_parent_death_signal_verified':linux,'parent_death_signal':'SIGKILL'if linux else None,
        'lifecycle_platform':'linux_prctl'if linux else'unsupported_nonlinux_cpu_fixture',
        'source_root':source_root,'validator_source_files':source_pins,'producer_metadata_sha256':digest(producer_value),
        'contract_metadata_sha256':digest(contract),'GPU_visible':False,'heavy_inference_modules_imported':False}
    raw=(json.dumps(receipt,ensure_ascii=False,sort_keys=True)+'\n').encode()
    need(len(raw)<=4096 and os.write(2,raw)==len(raw),'CPU worker receipt was not saved atomically to stderr')


def _read_retained_cpu_process(job):
    state=_CPU_PROCESS_STATE;need(state is not None,'CPU reader initializer has not run')
    _same_cpu_parent(state['parent_pid'],state['parent_start'])
    source,record_sha,commit_sha=job
    record,rs,cs=read_result(source,state['producer'],record_sha,commit_sha,contract=state['contract'])
    return rs,cs,record['decision']['state']


def _shutdown_retained_pool(pool):
    """幂等排空；由持旧锁的_validate先调用，外层finally不重复释放后shutdown。"""
    if pool is not None and not getattr(pool,'_annotation_retained_drained',False):
        pool.shutdown(wait=True,cancel_futures=True)
        pool._annotation_retained_drained=True


@contextlib.contextmanager
def _drain_retained_pool_on_exit(pool):
    try:yield
    finally:_shutdown_retained_pool(pool)


@contextlib.contextmanager
def retained_read_pool(workers,backend='thread',*,producer_value=None,contract=None,purpose='production'):
    """共享的安全构造入口；union和有界实测CLI复用，不用fork。"""
    spec={'retained_read_workers':workers,'retained_read_backend':backend,'purpose':purpose}
    retained_read_workers(spec);retained_read_backend(spec)
    if workers==1:
        yield None;return
    if backend=='thread':
        pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='retained-sidecar-read')
        try:yield pool
        finally:_shutdown_retained_pool(pool)
        return
    need(purpose in {'production','cpu_fixture'} and isinstance(producer_value,dict)and isinstance(contract,dict),
         'process reader requires explicit immutable producer/contract metadata')
    _no_heavy_imports();parent=os.getpid();start=_linux_start_ticks(parent)if sys.platform=='linux'else None
    source_root=str(physical(Path(__file__).resolve().parent));pins=_cpu_source_pins()
    previous={key:os.environ.get(key)for key in _CPU_ENV}
    try:
        # spawn导入__main__之前即隐藏显卡；仅影响本次独立CPU审计进程，不改其他PID环境。
        for key,value in _CPU_ENV.items():os.environ[key]=value
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_retained_cpu_process,initargs=(producer_value,contract,parent,start,source_root,pins,purpose))
        try:yield pool
        finally:_shutdown_retained_pool(pool)
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


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
                    expected_slot_sha=None, *, contract, read_executor=None, read_stats=None, read_backend='thread'):
    need(read_backend in {'thread','process'},'unknown completed read backend')
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
    def read_one(source):
        # 线程内只做独立文件读取/解析/校验，不触SQLite、进度、选择集合，不修改共享合同。
        receipt=receipts[source['source_id']]
        record,record_sha,commit_sha=read_result(source,producer_value,
            receipt['sidecar_sha256'],receipt['commit_sha256'],contract=contract)
        return record_sha,commit_sha,record['decision']['state']
    for start in range(0,len(rows),MAX_RETAINED_READ_BATCH):
        batch=rows[start:start+MAX_RETAINED_READ_BATCH]
        if read_executor is None:
            results=map(read_one,batch)
            if read_stats is not None:read_stats['max_submitted_batch']=max(read_stats['max_submitted_batch'],int(bool(batch)))
        else:
            if read_stats is not None:read_stats['max_submitted_batch']=max(read_stats['max_submitted_batch'],len(batch))
            try:
                # map按source原序返回。每批最多256个小结果，整片结束才提交下一片。
                if read_backend=='process':
                    jobs=[(source,receipts[source['source_id']]['sidecar_sha256'],receipts[source['source_id']]['commit_sha256'])for source in batch]
                    results=list(read_executor.map(_read_retained_cpu_process,jobs,chunksize=1))
                else:results=list(read_executor.map(read_one,batch))
            except BaseException:
                # 失败时要在调用者仍持旧父锁期间排空/取消此pool，不能先放锁再留后台读取。
                _shutdown_retained_pool(read_executor)
                raise
        for source,(record_sha,commit_sha,decision)in zip(batch,results):
            receipt=receipts[source['source_id']]
            need(receipt['job_key']==job_key(source,producer_value['contract']['sha256']) and
                 receipt['sidecar_relpath']==source['sidecar_relpath'] and
                 receipt['sidecar_bytes']==child(Path(producer_value['sidecar_tree']),source['sidecar_relpath']).stat().st_size and
                 receipt['decision']==decision,'chunk per-source receipt differs')
            decisions[decision]+=1
            if read_stats is not None:read_stats['old_completed_source_reads']+=1
            yield source,record_sha,commit_sha,decision
    need({k:decisions[k] for k in ('candidate','unknown','failed')} == slot['decision_counts'], 'chunk decision count differs')


def _terminal(plan, old, mode, spec=None):
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
    portable=spec is not None and spec.get('source_identity_mode')==PORTABLE_SOURCE_MODE
    need(age>=0 and (portable or mode!='plan' or age<=3600), 'plan terminal observation stale or future')
    if portable:
        current_ref=spec.get('current_terminal_observation')
        need(current_ref is not None,'portable union requires an explicit current terminal observation')
        current=ref(current_ref)
        need(current.get('schema')=='annotation-writer-terminal-observation/v1' and
             current.get('task_id')==obs['task_id'] and current.get('queue_root')==old['queue_root'] and
             current.get('state')in {'Stopped','Failed','Succeeded','Completed'},
             'current terminal observation old task/queue/state differs')
        workers=current.get('workers')
        need(isinstance(workers,list) and len(workers)==1 and
             workers[0].get('name')==obs['task_id']+'-worker-0' and
             workers[0].get('state')in {'Stopped','Failed','Succeeded','Completed'} and
             type(current.get('active_workers'))is int and current['active_workers']==0,
             'current terminal observation has active or unbound worker')
        timestamp=datetime.datetime.fromisoformat(current['observed_utc'])
        need(timestamp.tzinfo is not None,'current terminal observation timezone missing')
        current_age=(datetime.datetime.now(datetime.timezone.utc)-timestamp).total_seconds()
        need(current_age>=0 and (mode!='plan' or current_age<=3600),
             'current terminal observation stale or future')
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


STRICT_SOURCE_MODE = 'recorded_source_stat/v1'
PORTABLE_SOURCE_MODE = 'portable_encoded_sha256/v1'
PORTABLE_FRESH_SCOPE = 'retained_partial_and_unclaimed_only'
PORTABLE_COMPLETED_SCOPE = 'original_completed_chunk_reference_not_media_rehashed'


def _portable_ledger(spec, plan, old, oldc, new, newc, db):
    """只核显式跨节点内容证据；旧strict分支没有skip或stat例外。"""
    mode=spec.get('source_identity_mode',STRICT_SOURCE_MODE)
    need(mode in {STRICT_SOURCE_MODE,PORTABLE_SOURCE_MODE},'unsupported explicit source identity mode')
    fields=('portable_source_spec_ref','portable_source_ledger_ref')
    if mode==STRICT_SOURCE_MODE:
        need(plan.get('source_identity_mode',STRICT_SOURCE_MODE)==STRICT_SOURCE_MODE and
             not any(k in spec or k in plan for k in fields+('portable_sidecar_inventory_ref',)),
             'portable evidence requires explicit new union mode')
        return None
    need(plan.get('source_identity_mode')==mode and
         plan.get('fresh_media_scope')==PORTABLE_FRESH_SCOPE and
         plan.get('completed_media_scope')==PORTABLE_COMPLETED_SCOPE,
         'portable preparation media scope differs')
    for key in fields:
        need(key in spec and spec[key]==plan.get(key),'portable evidence must be pinned in both union and preparation: '+key)
    policy=ref(spec['portable_source_spec_ref'])
    from portable_source_identity import validate_portable_spec
    validate_portable_spec(policy)
    need(policy.get('schema')=='annotation-portable-source-spec/v1' and policy.get('source_identity_mode')==mode and
         policy.get('purpose')==spec['purpose'] and policy.get('old_contract')==old['contract'] and
         policy.get('full_index')==spec['full_index'] and policy.get('old_queue_root')==old['queue_root'] and
         policy.get('old_sidecar_tree')==old['sidecar_tree'] and policy.get('allowed_difference')==['device'] and
         policy.get('encoded_hash_algorithm')=='sha256','portable policy source/producer/full denominator differs')
    need(old['contract']['sha256']!=new['contract']['sha256'] and oldc['version']!=newc['version'],
         'portable successor requires a separate contract and version')
    for key in ('models','runtime','decoder','alignment','quality_policy','source_files','engine_source_files'):
        need(oldc.get(key)==newc.get(key),'portable successor changed frozen inference semantics: '+key)
    if spec['purpose']=='production':
        runtime=oldc['runtime']
        need(runtime.get('dtype')=='float32' and runtime.get('attention')=='eager' and runtime.get('TF32') is False and
             runtime.get('asr_compute',{'dtype':'float32','attention':'eager'})=={'dtype':'float32','attention':'eager'},
             'portable successor must preserve original FP32/eager inference')
        need(isinstance(oldc.get('engine_source_files'),dict) and bool(oldc['engine_source_files']),
             'portable successor lacks original inference module pins')
    item=spec['portable_source_ledger_ref']
    need(isinstance(item,dict) and set(item)=={'path','sha256','count'} and type(item['count']) is int and item['count']>=0,
         'portable source ledger must have exact path/SHA/count')
    db.execute('CREATE TABLE portable_retained(sid TEXT PRIMARY KEY NOT NULL,value TEXT NOT NULL)')
    fd=open_readonly(physical(item['path']));h=hashlib.sha256();count=0
    try:
        before=os.fstat(fd)
        with os.fdopen(fd,'rb',closefd=False)as stream:
            while True:
                line=stream.readline(65537)
                if not line:break
                need(line.strip() and len(line)<=65536,'portable source ledger row blank or oversized')
                h.update(line);row=strict_json(line)
                need(isinstance(row.get('source_id'),str) and bool(row['source_id']),'portable source ledger ID missing')
                try:db.execute('INSERT INTO portable_retained VALUES(?,?)',(row['source_id'],line.decode('utf-8')))
                except sqlite3.IntegrityError as exc:raise ValueError('duplicate portable source ledger ID')from exc
                count+=1
        after=os.fstat(fd)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'portable ledger changed during read')
    finally:os.close(fd)
    need(count==item['count'] and h.hexdigest()==item['sha256'],'portable source ledger SHA/count differs')
    return {'policy':policy,'spec_sha':spec['portable_source_spec_ref']['sha256'],'count':count,
            'verified':collections.Counter()}


def _portable_inventory(spec,plan,portable,old,db):
    """读取准备器固定的新目录项库；missing只查新SQLite，不逐源stat旧PFS。"""
    key='portable_sidecar_inventory_ref';pin=plan.get(key)
    if pin is None:
        need(key not in spec,'union inventory ref absent from preparation')
        return
    need(spec.get(key)==pin,'portable inventory must be pinned in union and preparation')
    need(isinstance(pin,dict) and set(pin)=={'path','sha256'},'exact portable inventory path/SHA required')
    path=physical(pin['path']);stamp=path.stat()
    need(sha_file(path)==pin['sha256'],'portable inventory SHA differs')
    uri='file:'+quote(str(path),safe='/')+'?mode=ro'
    db.execute('ATTACH DATABASE ? AS portable_inventory',(uri,))
    summary={k:strict_json(v)for k,v in db.execute('SELECT key,value FROM portable_inventory.metadata')}
    need(summary==plan.get('portable_sidecar_inventory_summary') and
         summary.get('schema')=='annotation-portable-sidecar-inventory/v1' and
         summary.get('portable_spec_sha256')==portable['spec_sha'] and
         summary.get('old_contract')==old['contract'] and summary.get('full_index')==spec['full_index'] and
         summary.get('old_sidecar_tree')==old['sidecar_tree'] and
         summary.get('all_inventory_paths_matched_original_sources') is True and
         summary.get('source_audio_read_by_inventory') is False and
         summary.get('completed_sidecar_content_read_by_inventory') is False and
         summary.get('stage_cache_traversed') is False,
         'portable inventory metadata/spec/producer differs')
    bad=db.execute("SELECT relative FROM portable_inventory.paths WHERE sidecar!=1 OR commit_marker!=1 OR seen!=1 OR source_id IS NULL OR kind NOT IN ('completed','partial','unclaimed') LIMIT 1").fetchone()
    need(bad is None,'portable inventory contains orphan, unmatched or invalid source entry')
    portable['inventory']={'path':path,'stamp':stamp,'summary':summary,'matched':0}


def _inventory_source(db,portable,source,kind):
    if portable is None or 'inventory'not in portable:return False
    row=db.execute('SELECT sidecar,commit_marker,seen,source_id,kind FROM portable_inventory.paths WHERE relative=?',
                   (source['sidecar_relpath'],)).fetchone()
    if kind is None:
        need(row is None,'portable inventory exposes an omitted old published source')
    else:
        need(row==(1,1,1,source['source_id'],kind),'portable inventory source/path/selection differs')
        portable['inventory']['matched']+=1
    return True


def _inventory_finished(db,portable,total):
    if portable is None or 'inventory'not in portable:return
    inventory=portable['inventory'];summary=inventory['summary']
    count=db.execute('SELECT COUNT(*) FROM portable_inventory.paths').fetchone()[0]
    need(inventory['matched']==count and summary.get('original_sources_checked')==total and
         summary.get('missing_source_lookups_without_PFS_stat')==total-count,
         'portable inventory full source coverage/count differs')
    need(sum(summary.get('matched_'+kind+'_sources',-1)for kind in ('completed','partial','unclaimed'))==count,
         'portable inventory selected kind counts differ')
    now=inventory['path'].stat();old=inventory['stamp']
    need((old.st_dev,old.st_ino,old.st_size,old.st_mtime_ns,old.st_ctime_ns)==
         (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns),
         'portable inventory changed during union validation')


def _consume_portable_entry(db,source,record,record_sha,commit_sha,old,portable):
    if portable is None:return
    row=db.execute('SELECT value FROM portable_retained WHERE sid=?',(source['source_id'],)).fetchone()
    need(row is not None,'portable ledger omits retained partial/unclaimed source')
    entry=strict_json(row[0])
    need(entry.get('sidecar_sha256')==record_sha and entry.get('commit_sha256')==commit_sha,
         'portable ledger differs from retained original sidecar/commit bytes')
    from portable_source_identity import validate_ledger_entry
    validate_ledger_entry(entry,source,record,old['contract']['sha256'],portable['policy'],spec_sha=portable['spec_sha'])
    portable['verified'][entry['status']]+=1
    db.execute('DELETE FROM portable_retained WHERE sid=?',(source['source_id'],))


def _select_old_source(db,source,expected,old,tree,contract,portable=None):
    need(expected['source_record_sha256']==digest(source) and expected['producer_contract_sha256']==old['contract']['sha256'] and
         expected['sidecar_path']==str(child(tree,source['sidecar_relpath'])) and
         expected['commit_path']==str(child(tree,source['sidecar_relpath']+'.commit.json')), 'retained old source identity differs')
    before=db.execute('SELECT selected FROM sources WHERE sid=?',(source['source_id'],)).fetchone()[0]
    need(before is None,'source retained twice')
    record,rs,cs=read_result(source,old,expected['sidecar_sha256'],expected['commit_sha256'],contract=contract)
    need(expected['decision']==record['decision']['state'],'retained decision changed')
    _consume_portable_entry(db,source,record,rs,cs,old,portable)
    db.execute('UPDATE sources SET selected=?,record_sha=?,commit_sha=?,decision=? WHERE sid=?',(old['id'],rs,cs,record['decision']['state'],source['source_id']))
    return record['decision']['state']


def _validate(spec, spec_sha, mode, db, progress=None, retained_executor=None):
    workers=retained_read_workers(spec);backend=retained_read_backend(spec)
    read_stats={'max_submitted_batch':0,'old_completed_source_reads':0}
    last_progress=[0.]
    def emit(stage,force=False,**counts):
        now=time.monotonic()
        if progress is not None and (force or now-last_progress[0]>=30):
            progress({'schema':'annotation-union-progress/v1','stage':stage,**counts});last_progress[0]=now
    emit('started',True,full_rows_checked=0,retained_sources_checked=0,remaining_rows_checked=0)
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
    lock_proof=_terminal(plan,old,mode,spec)
    db.execute('CREATE TABLE sources(sid TEXT PRIMARY KEY, source_sha TEXT NOT NULL, shard INTEGER NOT NULL, row_number INTEGER NOT NULL, relative TEXT NOT NULL, selected TEXT, record_sha TEXT, commit_sha TEXT, decision TEXT)')
    db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    _load_unclaimed_ledger(plan,db)
    portable=_portable_ledger(spec,plan,old,oldc,new,newc,db)
    if portable is not None:_portable_inventory(spec,plan,portable,old,db)
    domains=collections.Counter();total=0;retained=0;remaining=0;decisions=collections.Counter()
    retained_chunks={s['id']:s for s in plan['retained_complete_shards']}
    need(len(retained_chunks)==len(plan['retained_complete_shards']), 'duplicate retained chunk')
    partial={s['source_id']:s for s in plan['retained_partial_sources']}
    need(len(partial)==len(plan['retained_partial_sources']), 'duplicate retained partial source')
    remaining_index=ref(new['index']);need(remaining_index.get('parent_full_index')==spec['full_index'], 'remaining index parent differs')
    with old_parent_lock(oq) as current_lock, _drain_retained_pool_on_exit(retained_executor):
        need(all(current_lock[k]==lock_proof[k] for k in ('path','device','inode')), 'old lock inode changed')
        _,broker_before=read_json(oq/'broker.json')
        need(broker_before==plan['old_progress_snapshot']['broker_snapshot_sha256'], 'old writer broker changed since final preparation')
        old_progress=inspect_progress(spec['full_index']['path'],spec['full_index']['sha256'],oq,old['contract']['sha256'])
        need(old_progress['broker_snapshot_sha256']==broker_before,'old queue snapshot changed before coverage check')
        state=old_progress['broker_snapshot'];expected_retained={s['id'] for i,s in enumerate(full['shards'])
            if i<state['next'] and s['id'] not in state['leases']}
        need(set(retained_chunks)==expected_retained,'final plan omits/adds completed old chunks')
        emit('original_full_and_retained',True,full_rows_checked=0,retained_sources_checked=0,remaining_rows_checked=0)
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
                        expected['chunk_receipt_sha256'],expected['completion_slot_sha256'],contract=oldc,
                        read_executor=retained_executor,read_stats=read_stats,read_backend=backend):
                    db.execute('UPDATE sources SET selected=?,record_sha=?,commit_sha=?,decision=? WHERE sid=?',(old['id'],rs,cs,decision,source['source_id']))
                    retained+=1;decisions[decision]+=1
                    emit('original_full_and_retained',full_rows_checked=total,retained_sources_checked=retained,remaining_rows_checked=0)
            for source in rows:
                extra=db.execute('SELECT value FROM retained_extras WHERE sid=?',(source['source_id'],)).fetchone()
                kind=('completed'if descriptor['id']in expected_retained else
                      'partial'if source['source_id']in partial else 'unclaimed'if extra is not None else None)
                inventory_used=_inventory_source(db,portable,source,kind)
                if source['source_id'] in partial:
                    expected=partial.pop(source['source_id'])
                    need(extra is None and ordinal<state['next'] and descriptor['id'] in state['leases'],
                         'partial retained source is duplicated or not from a leased shard')
                    decision=_select_old_source(db,source,expected,old,ot,oldc,portable)
                    retained+=1;decisions[decision]+=1
                    emit('original_full_and_retained',full_rows_checked=total,retained_sources_checked=retained,remaining_rows_checked=0)
                elif extra is not None:
                    need(ordinal>=state['next'],'unclaimed retained source is not in an unclaimed shard')
                    decision=_select_old_source(db,source,strict_json(extra[0]),old,ot,oldc,portable)
                    db.execute('DELETE FROM retained_extras WHERE sid=?',(source['source_id'],))
                    retained+=1;decisions[decision]+=1
                    emit('original_full_and_retained',full_rows_checked=total,retained_sources_checked=retained,remaining_rows_checked=0)
                elif descriptor['id'] not in expected_retained and not inventory_used:
                    # 包括未领取片中的预置旧结果。不能因准备器未列出而重新推理。
                    old_path=child(ot,source['sidecar_relpath'])
                    old_commit=old_path.with_name(old_path.name+'.commit.json')
                    physical(old_commit)
                    need(not old_path.exists() and not old_commit.exists(),
                         'final plan omits an already published/orphan old source')
            db.commit()
            emit('original_full_and_retained',full_rows_checked=total,retained_sources_checked=retained,remaining_rows_checked=0)
        need(not retained_chunks and not partial, 'retained source/chunk absent from full index')
        _inventory_finished(db,portable,total)
        if portable is not None:
            need(db.execute('SELECT COUNT(*) FROM portable_retained').fetchone()[0]==0,
                 'portable ledger contains extra, completed, or non-retained sources')
        need(db.execute('SELECT COUNT(*) FROM retained_extras').fetchone()[0]==0,'unclaimed retained source absent from full index')
        need(total==full['source_count'] and dict(domains)==full.get('domains'), 'original full domain counts differ')
        if spec['purpose']=='production':need(dict(domains)==PRODUCTION_DOMAINS, 'production row domains differ')
        if mode=='complete':
            new_progress=inspect_progress(new['index']['path'],new['index']['sha256'],nq,new['contract']['sha256'])
            need(new_progress['committed_source_lower_bound']==remaining_index['source_count'] and new_progress['snapshot_leased_chunks_excluded']==0,
                 'new remaining producer not complete')
            new_broker=new_progress['broker_snapshot_sha256']
        remaining_roles=collections.Counter()
        emit('remaining_inputs',True,full_rows_checked=total,retained_sources_checked=retained,remaining_rows_checked=0)
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
                if portable is not None:remaining_roles.update(source_roles(source))
                emit('remaining_inputs',full_rows_checked=total,retained_sources_checked=retained,remaining_rows_checked=remaining)
                if d is not None:decisions[d]+=1
            db.commit()
        need(retained==plan['retained_source_count'] and remaining==plan['remaining_source_count']==remaining_index['source_count'], 'plan retained/remaining counts differ')
        missing=db.execute('SELECT COUNT(*) FROM sources WHERE selected IS NULL').fetchone()[0]
        need(missing==0 and retained+remaining==total, 'union leaves missing sources')
        if portable is not None and (spec['purpose']=='production' or 'role_union_counts'in remaining_index):
            measured={role:remaining_roles[role]for role in ('train_ptn','train_lm')}
            need(remaining_index.get('role_union_counts')==measured and
                 remaining_index.get('role_union_counts_scope')=='remaining_index_sources',
                 'remaining role aggregate differs from actual remaining sources')
            if spec['purpose']=='production':
                need(measured['train_ptn']==remaining and remaining_index.get('parent_role_union_counts')==full.get('role_union_counts') and
                     remaining_index.get('role_union_counts_parent_index')==spec['full_index'],
                     'remaining role aggregate lacks original full scope binding')
        _,broker_after=read_json(oq/'broker.json');need(broker_after==broker_before,'old writer changed during union validation')
        if mode=='complete':need(read_json(nq/'broker.json')[1]==new_broker,'new producer changed during complete validation')
    report={'schema':'annotation-union-validation/v1','spec_sha256':spec_sha,'mode':mode,'purpose':spec['purpose'],
            'source_count':total,'retained_source_count':retained,'remaining_source_count':remaining,
            'retained_read_workers':workers,'retained_read_backend':backend,
            'retained_read_executor':(('process_pool'if backend=='process'else'thread_pool')if workers>1 else'serial'),
            'retained_read_process_start_method':'spawn'if workers>1 and backend=='process'else None,
            'retained_read_process_lifecycle_platform':(('linux_prctl_in_initializer'if sys.platform=='linux'else'unsupported_nonlinux_cpu_fixture')if workers>1 and backend=='process'else'not_used'),
            'retained_read_scope':'old_completed_chunks_only',
            'retained_read_submission_limit':MAX_RETAINED_READ_BATCH if workers>1 else 1,
            'retained_read_max_submitted_batch':read_stats['max_submitted_batch'],
            'retained_completed_source_reads':read_stats['old_completed_source_reads'],
            'domains':dict(domains),'decision_counts':{k:decisions[k]for k in ('candidate','unknown','failed')},
            'decision_count_scope':'all_selected' if mode=='complete' else 'retained_only',
            'exact_union_and_disjoint':True,
            'complete_sidecars_reopened':mode=='complete','old_retained_sidecars_reopened':True,
            'formal_runtime_admitted':False,'training_or_evaluation_eligibility_released':False,
            'GPU_executed':False,'source_audio_rehashed':False,'platform_authentication_performed':False}
    report.update(source_identity_mode=spec.get('source_identity_mode',STRICT_SOURCE_MODE),
                  portable_source_ledger_verified=portable is not None,
                  portable_source_evidence_count=portable['count'] if portable else 0,
                  portable_sidecar_inventory_verified=portable is not None and 'inventory'in portable,
                  upstream_encoded_content_verified_count=portable['verified']['encoded_content_verified'] if portable else 0,
                  historical_decode_failure_not_content_verified_count=portable['verified']['historical_decode_failure_preserved_not_content_verified'] if portable else 0,
                  fresh_media_scope=PORTABLE_FRESH_SCOPE if portable else None,
                  completed_media_scope=PORTABLE_COMPLETED_SCOPE if portable else None)
    if mode=='complete':
        # 消费千万条时按原顺序流式读取，避免每轮再对全表排序。
        db.execute('CREATE INDEX source_sequence ON sources(shard,row_number)')
    for key,value in {'spec':spec,'spec_sha256':spec_sha,'validation':report}.items():
        db.execute('INSERT INTO metadata VALUES(?,?)',(key,json.dumps(value,sort_keys=True,ensure_ascii=False)))
    db.commit();emit('completed',True,full_rows_checked=total,retained_sources_checked=retained,remaining_rows_checked=remaining)
    return report


def validate_union(spec_path, spec_sha, mode='plan', *, output_index=None, progress=None):
    spec=ref({'path':str(spec_path),'sha256':spec_sha})
    workers=retained_read_workers(spec);backend=retained_read_backend(spec)
    if progress is None and spec.get('purpose')=='production':
        progress=lambda event:print(json.dumps(event,ensure_ascii=False),file=sys.stderr,flush=True)
    def build(path):
        db=sqlite3.connect(str(path),uri=True);db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-8192')
        try:
            producer_value=spec['old_producer']
            contract=ref(producer_value['contract'])if backend=='process'and workers>1 else None
            with retained_read_pool(workers,backend,producer_value=producer_value,contract=contract,purpose=spec['purpose'])as executor:
                return _validate(spec,spec_sha,mode,db,progress,executor)
        finally:db.close()
    if output_index is None:
        with tempfile.TemporaryDirectory(prefix='annotation-union-readonly-audit-') as tmp:
            return build(Path(tmp)/'consumer.sqlite')
    output=physical(output_index);need(not output.exists() and output.parent.is_dir(),'consumer index must be a new external file')
    protected=[physical(spec['full_index']['path']).parent,physical(spec['continuation_plan']['path']).parent]
    for key in ('portable_source_spec_ref','portable_source_ledger_ref','portable_sidecar_inventory_ref'):
        if key in spec:protected.append(physical(spec[key]['path']).parent)
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
