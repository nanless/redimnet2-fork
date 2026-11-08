"""排队后在GPU调用前等待真实平台新鲜观察，原恢复门和计算引擎保持不变。

本程序不查询平台，不加载模型，也不创建CUDA context。等待发生在已分配
云实例内，仍可能占用及计费八卡；1800秒上限不能写成免费资源等待。
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import sys
import time
import traceback

BLUEPRINT_SCHEMA='annotation-queue-handoff-blueprint/v1'
REQUEST_SCHEMA='annotation-queue-handoff-request/v1'
REPLY_SCHEMA='annotation-queue-handoff-reply/v1'
PYTHON='/root/miniforge3/envs/qwen3-asr/bin/python3.10'
PYTHON_SHA='cdb9592ef64842d7528ac97c2e03c81f226e5d875d5cc9304b59131e7bbcd505'
RUNTIME_KEYS={'schema','purpose','task_name','model_API_calls_allowed','release_training_labels',
    'old_control','freeze_receipt','union_spec','union_attestation','old_broker','old_completion_map',
    'old_queue_identity','new_run_root','evidence_root','lineage_claim'}
PREPARED_RUNTIME_KEYS=(RUNTIME_KEYS-{'union_attestation'})|{'preparation_attestation'}
BLUEPRINT_KEYS={'schema','purpose','task_name','runtime_base','handoff_root','outer_controller',
    'runtime_controller','python_binary','wait_seconds','required_prior_attempt_ids'}
IDENTITY_KEYS={'hostname','task_id','pid','start_ticks','boot_id'}
TERMINAL={'Failed','Stopped','Succeeded','Completed'}
HEX=re.compile('[0-9a-f]{64}')
TASK=re.compile('t-[0-9]{14}-[a-z0-9]+')
INNER_SOURCE_NAMES={'alignment_observer.py','benchmark_retained_reads.py','contracts.py',
    'finalize_remaining_roles.py','freeze_legacy_successor.py','launch_portable_successor.sh',
    'normalization.py','portable_source_identity.py','prepare_speed_continuation.py',
    'probe_retained_process_parent_death.py','quality_axes.py','readonly_progress.py',
    'resume_existing_annotation.py','run_portable_successor.py','schema.py','source_clock.py',
    'source_registry.py','task_names.py','union_coverage.py','verify_portable_successor_mounts.py','work_queue.py'}


def need(value,message):
    if not value:raise ValueError(message)


def physical(value):
    p=Path(value)
    need(p.is_absolute() and '..'not in p.parts and all(not q.is_symlink()for q in (p,*p.parents)),
         'physical absolute path without symlink required')
    return p


def _stamp(value):return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_nlink)


def read_file(path,*,private=False,limit=64*1024**2):
    """先拒软/硬链，再核打开FD与路径稳定；不把开发机PID当云进程证明。"""
    p=physical(path);initial=p.lstat()
    need(stat.S_ISREG(initial.st_mode)and initial.st_nlink==1,'regular file with one link required')
    if private:need(stat.S_IMODE(initial.st_mode)==0o600 and initial.st_uid==os.getuid(),'private owned file mode0600 required')
    # 先lstat避免直接打开FIFO阻塞；NONBLOCK再防检查后被置换为FIFO的窗口。
    fd=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    try:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_nlink==1,'regular file with one link required')
        need(_stamp(initial)==_stamp(before),'file replaced between lstat and open')
        if private:need(stat.S_IMODE(before.st_mode)==0o600 and before.st_uid==os.getuid(),'private owned file mode0600 required')
        need(before.st_size<=limit,'bounded evidence file exceeded')
        with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(limit+1)
        after=os.fstat(fd);current=physical(p).lstat()
        need(len(raw)<=limit and len(raw)==before.st_size and _stamp(before)==_stamp(after)==_stamp(current),
             'file changed while reading')
        return raw
    finally:os.close(fd)


def _pairs(rows):
    value={}
    for key,item in rows:need(key not in value,'duplicate JSON key');value[key]=item
    return value


def json_value(raw):
    def finite_float(text):
        value=float(text);need(math.isfinite(value),'nonfinite JSON');return value
    value=json.loads(raw,object_pairs_hook=_pairs,parse_float=finite_float,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    need(isinstance(value,dict),'JSON object required');return value


def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _pin_shape(pin):
    need(isinstance(pin,dict)and set(pin)=={'path','sha256'}and isinstance(pin['sha256'],str)and HEX.fullmatch(pin['sha256']),
         'exact absolute file ref/SHA required')
    physical(pin['path']);return pin


def ref(pin,*,private=False):
    _pin_shape(pin);raw=read_file(pin['path'],private=private)
    need(hashlib.sha256(raw).hexdigest()==pin['sha256'],'pinned JSON differs');return json_value(raw)


def file_pin(path,*,private=False):
    raw=read_file(path,private=private);return {'path':str(physical(path)),'sha256':hashlib.sha256(raw).hexdigest()}


def _fsync_dir(path):
    fd=os.open(physical(path),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def write_new(path,value):
    """完整私有临时文件fsync后无覆盖发布；nlink2的过渡文件不可消费。"""
    p=physical(path);private_dir(p.parent)
    pending=p.with_name('.'+p.name+'.pending')
    fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        with os.fdopen(fd,'wb',closefd=False)as stream:
            stream.write(canonical(value));stream.flush();os.fsync(fd)
    finally:os.close(fd)
    # 保留失败时的pending原件；不覆盖目标或删除历史失败证据。
    os.link(pending,p,follow_symlinks=False);pending.unlink();_fsync_dir(p.parent)
    return file_pin(p,private=True)


def private_dir(path):
    p=physical(path);value=p.stat()
    need(stat.S_ISDIR(value.st_mode)and stat.S_IMODE(value.st_mode)==0o700 and value.st_uid==os.getuid(),
         'private owned directory mode0700 required');return p


def mkdir_new(path):
    p=physical(path);need(p.parent.is_dir(),'existing handoff parent required')
    p.mkdir(mode=0o700);private_dir(p);_fsync_dir(p.parent);return p


def manifest(controller,required,*,name,exact=False):
    need(isinstance(controller,dict)and set(controller)=={'root','manifest'},'fixed controller root/manifest required')
    root=physical(controller['root']);need(root.is_dir(),'controller directory missing')
    pin=_pin_shape(controller['manifest']);p=physical(pin['path'])
    need(p.parent==root and p.name==name,'fixed manifest filename inside controller root required')
    raw=read_file(p,limit=128*1024);need(hashlib.sha256(raw).hexdigest()==pin['sha256'],'controller manifest SHA differs')
    names=set()
    for line in raw.decode('utf-8').splitlines():
        need(len(line)>66 and HEX.fullmatch(line[:64])and line[64:66]=='  ','strict sha256sum line required')
        member=line[66:];rel=Path(member)
        need(not rel.is_absolute() and str(rel)==member and '..'not in rel.parts and member not in names and
             member!=p.name and '\\'not in member and all(c.isprintable()for c in member),'unsafe/duplicate manifest member')
        need(hashlib.sha256(read_file(root/rel)).hexdigest()==line[:64],'controller member SHA differs: '+member);names.add(member)
    need(required<=names and (not exact or names==required),'controller member set differs')
    # 原v6还保留一份非执行JSON映射；其它额外成员可能遮蔽Python导入，拒绝。
    allowed=names|{name}|({'controller-files.json'}if name=='controller-files.sha256'else set())
    entries=list(root.iterdir());actual={p.name for p in entries}
    need(actual<=allowed and all(stat.S_ISREG(p.lstat().st_mode)and p.lstat().st_nlink==1 for p in entries),
         'controller directory has extra or linked import members')
    return root


def _task_name(name):
    need(isinstance(name,str)and name.startswith('zhounan_')and len(name)>8 and name==name.strip()and
         all(ord(c)>=32 and ord(c)!=127 for c in name),'task name must start zhounan_')


def _disjoint(a,b):return a!=b and a not in b.parents and b not in a.parents


def check(path,pin):
    """默认只检查外层固定输入，不创建request、不执行原控制器或GPU。"""
    blueprint=ref({'path':str(path),'sha256':pin})
    need(set(blueprint)==BLUEPRINT_KEYS and blueprint['schema']==BLUEPRINT_SCHEMA and
         blueprint['purpose']in {'production','cpu_fixture'},'exact handoff blueprint schema required')
    base=blueprint['runtime_base']
    prepared=isinstance(base,dict)and base.get('schema')=='annotation-portable-successor-run/v2'
    expected_keys=PREPARED_RUNTIME_KEYS if prepared else RUNTIME_KEYS
    expected_schema='annotation-portable-successor-run/v2'if prepared else'annotation-portable-successor-run/v1'
    need(isinstance(base,dict)and set(base)==expected_keys and
        base['schema']==expected_schema and base['purpose']==blueprint['purpose'],
        'complete fixed runtime_base without current_platform_observation required')
    _task_name(blueprint['task_name']);need(base['task_name']==blueprint['task_name'],'runtime task name differs')
    need(type(base['model_API_calls_allowed'])is int and base['model_API_calls_allowed']==0 and
         base['release_training_labels']is False,'offline API0/no automatic training release required')
    need(type(blueprint['wait_seconds'])is int and 1<=blueprint['wait_seconds']<=1800,'CPU wait must be bounded 1..1800 seconds')
    prior=blueprint['required_prior_attempt_ids']
    need(isinstance(prior,list)and prior and len(prior)==len(set(prior))and
         all(isinstance(x,str)and TASK.fullmatch(x)for x in prior),'explicit unique prior terminal task IDs required')
    outer=manifest(blueprint['outer_controller'],{'queue_handoff.py','launch_queue_handoff.sh'},
        name='queue-handoff-files.sha256',exact=True)
    inner_required=INNER_SOURCE_NAMES if blueprint['purpose']=='production'else {
        'launch_portable_successor.sh','run_portable_successor.py','verify_portable_successor_mounts.py'}
    inner=manifest(blueprint['runtime_controller'],inner_required,name='controller-files.sha256',
        exact=blueprint['purpose']=='production')
    need(_disjoint(outer,inner),'outer and frozen runtime controller roots must be independent')
    python=_pin_shape(blueprint['python_binary']);need(hashlib.sha256(read_file(python['path'])).hexdigest()==python['sha256'],
        'actual Python binary SHA differs')
    if blueprint['purpose']=='production':
        need(outer==Path(__file__).resolve().parent,'production outer source root differs from executing script')
        need(python=={'path':PYTHON,'sha256':PYTHON_SHA},'fixed production Python binary required')
    proof_key='preparation_attestation'if prepared else'union_attestation'
    old=ref(base['old_control']);freeze=ref(base['freeze_receipt']);union=ref(base['union_spec']);data=ref(base[proof_key])
    consumer=None
    if prepared:
        # v2复用已完成的逐源剩余清单检查；不把中断的联合库当成成功。
        plan=ref(union['continuation_plan'])
        need(data.get('schema')=='annotation-remaining-role-finalize/v1'and
            data.get('new_plan')==union['continuation_plan']and
            data.get('new_remaining_index')==union['new_producer']['index']and
            data.get('parent_full_index')==old['full_index']and
            data.get('inventory_ref')==union['portable_sidecar_inventory_ref']and
            type(data.get('remaining_sources_checked'))is int and
            type(plan.get('remaining_source_count'))is int and
            data['remaining_sources_checked']==plan.get('remaining_source_count',0)>0 and
            type(data.get('portable_proofs_reopened'))is int and
            type(plan['portable_source_ledger_ref'].get('count'))is int and
            plan['portable_source_ledger_ref']['count']>=0 and
            data.get('portable_proofs_reopened')==plan['portable_source_ledger_ref']['count']and
            all(data.get(k)is False for k in ('formal_runtime_admitted','GPU_executed',
                'source_shard_bytes_changed','v1_bytes_rewritten','source_media_read','models_executed')),
            'actual completed remaining preparation required; no partial union substitute')
    else:
        need(data.get('schema')=='annotation-union-validation/v1'and data.get('mode')=='plan'and
            data.get('spec_sha256')==base['union_spec']['sha256']and data.get('purpose')==base['purpose']and
            data.get('exact_union_and_disjoint')is True and data.get('formal_runtime_admitted')is False and
            data.get('GPU_executed')is False,'actual finished union plan required, never a partial database')
        consumer=_pin_shape(data['consumer_index']);cp=physical(consumer['path'])
        need(cp.is_file()and not cp.name.startswith('.consumer-building-'),'final union plan database required')
    oldroot=physical(old['task_root']);handoff=physical(blueprint['handoff_root'])
    need(handoff.parent.is_dir()and oldroot in handoff.parents,'handoff must be an independent child of original task ROOT')
    if blueprint['purpose']=='production':need(Path('/root')in handoff.parents,'production handoff must use persistent user /root mount')
    protected=[outer,inner,physical(old['run_root']),physical(old['code_root']),physical(freeze['code_root']).parent,
        physical(base['new_run_root']),physical(base['evidence_root']),physical(old['semantic_lock']).parent,
        physical(old['device_lock_root']),physical(old['legacy_source_alias']['target_path'])]
    # 通用task ROOT不是专用输入目录；其它固定控制/数据证明目录均双向保护。
    refs=[base[k]for k in ('old_control','freeze_receipt','union_spec',proof_key,'old_broker','old_completion_map','old_queue_identity')]
    refs.extend(old[k]for k in ('full_index','contract','admission','original_launcher','original_claim'))
    refs.extend(union[k]for k in ('continuation_plan','portable_source_spec_ref','portable_source_ledger_ref','portable_sidecar_inventory_ref'))
    refs.append(union['new_producer']['index'])
    if consumer is not None:refs.append(consumer)
    if prepared:refs.extend(data[k]for k in ('old_plan','old_remaining_index'))
    for item in refs:
        p=physical(item['path']).parent
        if p!=oldroot and p not in oldroot.parents:protected.append(p)
    need(all(_disjoint(handoff,p)for p in protected),'handoff overlaps old/new run, evidence, source, input, model or claim namespace')
    for key in ('new_run_root','evidence_root','lineage_claim'):
        p=physical(base[key]);need(not p.exists(),'runtime output/evidence/claim already exists; inspect prior task, do not retry')
    return {'blueprint':blueprint,'old':old,'outer_root':outer,'runtime_root':inner,'handoff_root':handoff,
        'protected_namespaces':list(dict.fromkeys(map(str,protected))),'GPU_started':False,'runtime_admitted':False,
        'controller_inputs_check_executed':False,'consumer_database_content_SHA_verified_by_outer':False}


def actual_identity():
    hostname=socket.gethostname();need(hostname.endswith('-worker-0'),'actual hostname must identify unique platform worker0')
    task=hostname[:-len('-worker-0')];need(TASK.fullmatch(task),'actual cloud hostname task ID invalid')
    raw=Path('/proc/self/stat').read_text();start=int(raw[raw.rfind(')')+2:].split()[19])
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return {'hostname':hostname,'task_id':task,'pid':os.getpid(),'start_ticks':start,'boot_id':boot}


def _identity(value):
    need(isinstance(value,dict)and set(value)==IDENTITY_KEYS and TASK.fullmatch(value.get('task_id',''))and
        value['hostname']==value['task_id']+'-worker-0'and type(value['pid'])is int and value['pid']>0 and
        type(value['start_ticks'])is int and value['start_ticks']>0 and
        isinstance(value['boot_id'],str)and re.fullmatch('[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',value['boot_id']),
        'bound actual cloud PID/start/boot/hostname identity required')


def _utc(now):return datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat()


def _age(value,now,limit):
    timestamp=datetime.datetime.fromisoformat(value);need(timestamp.tzinfo is not None,'timezone required')
    age=now-timestamp.timestamp();need(0<=age<=limit,'observation/request stale or future');return timestamp.timestamp()


def observation(pin,checked,request,now):
    value=ref(pin);old=checked['old'];blueprint=checked['blueprint']
    need(value.get('schema')=='annotation-portable-platform-observation/v1'and
        value.get('old_task_id')==old['old_task_id']and value.get('competing_successor_tasks')==[]and
        value.get('remote_proc_enumeration_performed')is False and
        value.get('evidence_basis')=='platform_terminal_and_current_shared_locks',
        'actual fresh platform observation with no competing successor required')
    observed=_age(value['observed_utc'],now,3600)
    need(observed>=datetime.datetime.fromisoformat(request['created_utc']).timestamp(),
         'platform observation must follow this actual cloud request')
    terminal=value['original_terminal']
    need(terminal.get('schema')=='annotation-writer-terminal-observation/v1'and
        terminal.get('task_id')==old['old_task_id']and terminal.get('queue_root')==old['queue_root']and
        terminal.get('state')in TERMINAL and type(terminal.get('active_workers'))is int and terminal['active_workers']==0 and
        terminal.get('observed_utc')==value['observed_utc'],'original task terminal/time/queue binding differs')
    workers=terminal.get('workers')
    need(isinstance(workers,list)and len(workers)==1 and isinstance(workers[0],dict)and
        workers[0].get('name')==old['old_task_id']+'-worker-0'and workers[0].get('state')in TERMINAL,
        'unique original worker must be terminal')
    priors=value.get('prior_attempts');need(isinstance(priors,list),'prior terminal observations required')
    ids=[]
    for prior in priors:
        need(isinstance(prior,dict)and isinstance(prior.get('task_id'),str)and TASK.fullmatch(prior['task_id'])and
            prior.get('state')in TERMINAL and type(prior.get('active_workers'))is int and prior['active_workers']==0,
            'prior successor must be terminal')
        ids.append(prior['task_id'])
    need(len(ids)==len(set(ids))and set(blueprint['required_prior_attempt_ids'])<=set(ids),'required prior task missing or duplicated')
    own=value.get('current_successor_task');identity=request['identity']
    need(isinstance(own,dict)and own.get('task_id')==identity['task_id']and own.get('task_name')==blueprint['task_name']and
        own.get('state')=='Running'and type(own.get('active_workers'))is int and own['active_workers']==1 and
        own.get('workers')==[{'name':identity['hostname'],'state':'Running'}]and
        identity['task_id']not in set(ids)|{old['old_task_id']},'actual own task/unique worker Running observation required')
    return value


def _request(checked,pin,now):
    request=ref(pin,private=True);root=private_dir(checked['handoff_root']);directory=private_dir(root/'request-v1')
    need(physical(pin['path'])==directory/'request.json','request path differs from fixed handoff namespace')
    need(set(request)=={'schema','blueprint','identity','nonce','created_utc','wait_seconds','reply_path'}and
        request['schema']==REQUEST_SCHEMA and request['blueprint']==checked['blueprint_pin']and
        type(request['wait_seconds'])is int and request['wait_seconds']==checked['blueprint']['wait_seconds']and
        request['reply_path']==str(directory/'reply.json')and
        isinstance(request['nonce'],str)and HEX.fullmatch(request['nonce']),'request blueprint/nonce/path binding differs')
    _identity(request['identity']);_age(request['created_utc'],now,request['wait_seconds']);return request


def publish(path,pin,request_pin,observation_pin,*,clock=None):
    checked=check(path,pin);blueprint=checked['blueprint'];checked['blueprint_pin']={'path':str(path),'sha256':pin}
    need(clock is None or blueprint['purpose']=='cpu_fixture','production publisher cannot use a synthetic clock')
    now=(clock or time.time)();request=_request(checked,request_pin,now);observation(observation_pin,checked,request,now)
    directory=physical(request_pin['path']).parent
    need(not physical(request['reply_path']).exists(),'reply already exists; no implicit re-publication')
    publication=mkdir_new(directory/'publication-v1')
    try:
        runtime={**blueprint['runtime_base'],'current_platform_observation':observation_pin}
        specpin=write_new(publication/'runtime-spec.json',runtime)
        # 再读固定输入与实际观察，避免文件在生成spec期间置换；不执行原入口。
        check(path,pin);_request(checked,request_pin,(clock or time.time)())
        observation(observation_pin,checked,request,(clock or time.time)())
        reply={'schema':REPLY_SCHEMA,'request':request_pin,'identity':request['identity'],'request_nonce':request['nonce'],
            'blueprint':checked['blueprint_pin'],'runtime_spec':specpin,'observation':observation_pin,
            'published_utc':_utc((clock or time.time)()),'platform_authentication_performed':False,'GPU_admission_granted':False}
        replypin=write_new(directory/'reply.json',reply)
        return {'schema':'annotation-queue-handoff-publication/v1','runtime_spec':specpin,'reply':replypin,
            'platform_authentication_performed':False,'GPU_admission_granted':False,'controller_inputs_check_executed':False,
            'consumer_database_content_SHA_verified_by_outer':False}
    except BaseException:
        _failure(publication,'publisher-failure.json');raise


def _failure(directory,name):
    value={'schema':'annotation-queue-handoff-failure/v1','traceback':traceback.format_exc(),
        'GPU_context_created_by_outer':False,'automatic_retry_allowed':False,'observed_utc':_utc(time.time())}
    try:write_new(directory/name,value)
    except BaseException:print(json.dumps(value,ensure_ascii=False),file=sys.stderr,flush=True)


def _reply_pending(path):
    """只等待本协议原子发布的nlink2过渡；其它硬链不能作为正式reply读取。"""
    p=physical(path)
    if not p.exists():return True
    current=p.stat()
    if current.st_nlink==2:
        pending=p.with_name('.'+p.name+'.pending')
        if pending.exists():
            temporary=physical(pending).stat()
            if (current.st_dev,current.st_ino)==(temporary.st_dev,temporary.st_ino):return True
    return False


def _reply(checked,request_pin,request,identity,now):
    _request(checked,request_pin,now)
    need(identity==request['identity'],'actual cloud process identity changed during wait')
    replypin=file_pin(request['reply_path'],private=True);reply=ref(replypin,private=True)
    _identity(reply.get('identity'))
    need(set(reply)=={'schema','request','identity','request_nonce','blueprint','runtime_spec','observation',
        'published_utc','platform_authentication_performed','GPU_admission_granted'}and reply['schema']==REPLY_SCHEMA and
        reply['request']==request_pin and reply['identity']==identity and reply['request_nonce']==request['nonce']and
        reply['blueprint']==checked['blueprint_pin']and reply['platform_authentication_performed']is False and
        reply['GPU_admission_granted']is False,'reply request/nonce/identity/blueprint binding differs')
    published=_age(reply['published_utc'],now,checked['blueprint']['wait_seconds'])
    need(published>=datetime.datetime.fromisoformat(request['created_utc']).timestamp(),'reply predates actual request')
    specpin=reply['runtime_spec'];need(physical(specpin['path'])==physical(request_pin['path']).parent/'publication-v1/runtime-spec.json',
        'runtime spec must stay in fixed private publication directory')
    private_dir(Path(specpin['path']).parent);runtime=ref(specpin,private=True)
    need(canonical(runtime)==canonical({**checked['blueprint']['runtime_base'],'current_platform_observation':reply['observation']}),
         'runtime spec may replace only current_platform_observation')
    observation(reply['observation'],checked,request,now);return specpin,replypin


def wait(path,pin,*,execute=False,runner=None,clock=None,monotonic=None,sleeper=None,identity_provider=None):
    checked=check(path,pin);blueprint=checked['blueprint'];checked['blueprint_pin']={'path':str(path),'sha256':pin}
    injected=any(x is not None for x in (runner,clock,monotonic,sleeper,identity_provider))
    need(not injected or blueprint['purpose']=='cpu_fixture','production waiter cannot use synthetic execution/clock/process hooks')
    if not execute:return {'schema':'annotation-queue-handoff-check/v1','passed':True,'request_created':False,
        'GPU_started':False,'runtime_admitted':False,'controller_inputs_check_executed':False,
        'consumer_database_content_SHA_verified_by_outer':False}
    need(blueprint['purpose']=='production'or runner is not None,'CPU fixture cannot execute the real runtime launcher')
    clock=clock or time.time;monotonic=monotonic or time.monotonic;sleeper=sleeper or time.sleep
    identify=identity_provider or actual_identity;identity=identify();_identity(identity)
    need(identity['task_id']not in set(blueprint['required_prior_attempt_ids'])|{checked['old']['old_task_id']},
         'waiter must be an independent new platform task')
    root=None;directory=None;previous={}
    def interrupted(signum,frame):raise RuntimeError('handoff interrupted before GPU: signal '+str(signum))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,interrupted)
        root=mkdir_new(checked['handoff_root']);directory=mkdir_new(root/'request-v1')
        request={'schema':REQUEST_SCHEMA,'blueprint':checked['blueprint_pin'],'identity':identity,'nonce':secrets.token_hex(32),
            'created_utc':_utc(clock()),'wait_seconds':blueprint['wait_seconds'],'reply_path':str(directory/'reply.json')}
        request_pin=write_new(directory/'request.json',request)
        print(json.dumps({'schema':'annotation-queue-handoff-awaiting/v1','request':request_pin,
            'cloud_instance_already_allocated':True,'GPU_context_created_by_outer':False},ensure_ascii=False),flush=True)
        started=monotonic()
        while True:
            need(monotonic()-started<blueprint['wait_seconds'],'handoff timed out before GPU; preserve request and inspect task')
            need(identify()==identity,'actual cloud process identity changed during wait')
            _request(checked,request_pin,clock())
            if not _reply_pending(request['reply_path']):break
            remaining=blueprint['wait_seconds']-(monotonic()-started)
            need(remaining>0,'handoff timed out before CPU sleep')
            sleeper(min(2.0,remaining))
        # 原launch再完整检查输入、时效、W2、挂载和旧锁。本层不替代任何原门。
        refreshed=check(path,pin);refreshed['blueprint_pin']=checked['blueprint_pin']
        specpin,replypin=_reply(refreshed,request_pin,request,identify(),clock())
        need(monotonic()-started<blueprint['wait_seconds'],'handoff timed out during final checks')
        need(identify()==identity,'actual cloud process identity changed before exec')
        argv=['/bin/bash',str(refreshed['runtime_root']/'launch_portable_successor.sh'),specpin['path'],
            specpin['sha256'],blueprint['runtime_controller']['manifest']['sha256']]
        write_new(directory/'exec-intent.json',{'argv':argv,'request':request_pin,'runtime_spec':specpin,
            'GPU_started_not_proven_by_intent':True,'platform_authentication_performed':False})
        need(monotonic()-started<blueprint['wait_seconds'],'handoff timed out while saving exec intent')
        need(identify()==identity,'actual cloud process identity changed immediately before exec')
        final=check(path,pin);final['blueprint_pin']=checked['blueprint_pin']
        finalspec,finalreply=_reply(final,request_pin,request,identify(),clock())
        need(finalspec==specpin and finalreply==replypin,'fixed reply/runtime spec changed while saving exec intent')
        need(monotonic()-started<blueprint['wait_seconds'],'handoff timed out during final source/reply verification')
        need(identify()==identity,'actual cloud process identity changed at final exec boundary')
        # exec会把捕获型信号处理器重置为默认；CPU注入执行返回后由finally恢复。
        env=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','BASH_ENV','ENV'):env.pop(key,None)
        if runner is not None:return runner(argv,env)
        os.execve('/bin/bash',argv,env)
        raise RuntimeError('runtime exec unexpectedly returned')
    except BaseException:
        if directory is not None:_failure(directory,'waiter-failure.json')
        elif root is not None:_failure(root,'waiter-failure.json')
        raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','wait','publish'),nargs='?',default='check')
    parser.add_argument('--blueprint',required=True,type=Path);parser.add_argument('--blueprint-sha',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--request',type=Path);parser.add_argument('--request-sha')
    parser.add_argument('--observation',type=Path);parser.add_argument('--observation-sha')
    args=parser.parse_args()
    if args.action=='publish':
        need(not args.execute and all((args.request,args.request_sha,args.observation,args.observation_sha)),
             'publish requires explicit actual request and fresh observation refs, never --execute')
        result=publish(args.blueprint,args.blueprint_sha,{'path':str(args.request),'sha256':args.request_sha},
            {'path':str(args.observation),'sha256':args.observation_sha})
    else:
        need(not any((args.request,args.request_sha,args.observation,args.observation_sha)),'request/observation arguments are publish-only')
        need(args.action=='wait'or not args.execute,'only wait may execute')
        result=wait(args.blueprint,args.blueprint_sha,execute=args.execute)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
