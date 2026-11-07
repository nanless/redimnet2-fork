"""显式接续既有同合同标注任务，不重建队列、不绕过原GPU守卫。

默认只检查固定原件。执行仅创建新的接续证据和一次声明，调用原冻结
gpu_launch；原声明、合同、全量索引、标签及资源记录均保留。
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

if __package__:
    from .readonly_progress import read_small, inspect_progress, open_readonly, sha
else:
    from readonly_progress import read_small, inspect_progress, open_readonly, sha

SCHEMA='annotation-existing-resume/v1'
FULL_COUNT=11521178
DOMAINS={'voxceleb2':1092009,'voxblink2':9904382,'cnceleb2':524787}


def need(ok,message):
    if not ok:raise ValueError(message)


def physical(value):
    p=Path(value)
    need(p.is_absolute() and '..' not in p.parts and all(not x.is_symlink() for x in (p,*p.parents)),
         'explicit physical absolute path required')
    return p


def file_sha(path):
    fd=open_readonly(physical(path));h=hashlib.sha256()
    try:
        before=os.fstat(fd)
        with os.fdopen(fd,'rb',closefd=False)as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
        after=os.fstat(fd)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'input changed while hashing')
    finally:os.close(fd)
    return h.hexdigest()


def ref(pin):
    need(isinstance(pin,dict) and set(pin)=={'path','sha256'} and sha(pin['sha256']), 'explicit file SHA required')
    value,raw=read_small(physical(pin['path']),64*1024**2)
    need(hashlib.sha256(raw).hexdigest()==pin['sha256'],'pinned JSON differs')
    return value


def pinned_file(pin):
    need(isinstance(pin,dict) and set(pin)=={'path','sha256'} and sha(pin['sha256']), 'explicit file SHA required')
    p=physical(pin['path']);need(file_sha(p)==pin['sha256'],'pinned file differs: '+str(p));return p


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def write_new(path,value):
    p=physical(path);need(p.parent.is_dir(),'existing evidence parent required')
    with p.open('xb')as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


@contextlib.contextmanager
def existing_lock(path):
    p=physical(path);fd=open_readonly(p)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=os.fstat(fd);after=p.stat()
        need((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'existing lock inode replaced')
        yield fd
    finally:os.close(fd)


def fresh(value):
    observed=datetime.datetime.fromisoformat(value['observed_utc'])
    need(observed.tzinfo is not None,'timezone required')
    age=(datetime.datetime.now(datetime.timezone.utc)-observed).total_seconds()
    need(0<=age<=3600,'terminal/absence observation stale or future')


def inspect(spec,*,_semantic_owned=False,_evidence_owned=False):
    """纯读原件；临时检查既有锁，不创建/修复任何生产对象。"""
    need(spec.get('schema')==SCHEMA and spec.get('purpose') in {'production','cpu_fixture'},'wrong resume schema/purpose')
    protocol=spec.get('protocol');need(protocol in {'legacy_frozen_v2','explicit_roots_v3'},'explicit deployed launcher protocol required')
    need(spec.get('model_API_calls_allowed')==0 and type(spec['model_API_calls_allowed'])is int and
         spec.get('release_training_labels')is False,'offline API0 and no label release required')
    need(spec.get('attempt_number')==2 and type(spec['attempt_number'])is int,
         'only explicit first recovery attempt2 supported; preserve prior attempts')
    base=physical(spec['task_root']);run=physical(spec['run_root']);code=physical(spec['code_root'])
    queue=physical(spec['queue_root']);labels=physical(spec['output_root']);resources=physical(spec['resource_output'])
    evidence=physical(spec['evidence_root']);locks=physical(spec['device_lock_root']);python=Path(spec['python'])
    need(python.is_absolute() and '..'not in python.parts,'original absolute Python path required')
    physical(python.parent)
    need(base.is_dir() and run.is_dir() and code.is_dir() and base in run.parents and base in code.parents and
         run!=code and run not in code.parents and code not in run.parents,'original task/code/run layout differs')
    need(queue==run/'queue' and labels==run/'labels' and resources==run/'resources-attempt2' and
         queue.is_dir() and labels.is_dir() and not resources.exists(),'original queue/labels and fresh attempt2 resources required')
    protected=(run,code,locks)
    need((not evidence.exists() or (_evidence_owned and evidence.is_dir())) and evidence.parent.is_dir() and
         all(evidence!=p and p not in evidence.parents and evidence not in p.parents for p in protected),
         'fresh external evidence directory required')
    need(locks.is_dir() and locks!=base and base not in locks.parents and locks not in base.parents,
         'original shared device lock root required')
    need(python.is_file(),'original explicit Python executable missing')
    python_target=python.resolve(strict=True);physical(python_target)
    need(python_target.parent==python.parent and pinned_file(spec['python_binary'])==python_target,
         'original interpreter must resolve to the explicitly pinned binary in the same bin directory')
    python_sha=spec['python_binary']['sha256']
    contract=ref(spec['contract']);index=ref(spec['full_index']);admission=ref(spec['admission'])
    need(contract.get('schema')=='redimnet2-production-contract/v2' and contract.get('purpose')=='silver_annotation' and
         contract.get('model_API_calls_allowed')==0 and contract.get('release_training_labels')is False,
         'original silver offline contract required')
    version=contract.get('version');need(isinstance(version,str) and version and '/' not in version and version not in {'.','..'},'invalid original label version')
    if spec['purpose']=='production':
        need(index.get('source_count')==FULL_COUNT and index.get('domains')==DOMAINS and
             index.get('RIR_or_MUSAN_or_Child_or_evaluation_in_this_job')is False,'original formal denominator/domain scope differs')
    original_claim=ref(spec['original_claim']);key=spec['contract']['sha256']+'-'+spec['full_index']['sha256']
    semantic=physical(spec['semantic_lock']);resume_claim=physical(spec['resume_claim'])
    need(semantic==base/'.formal-parent-locks'/(key+'.lock') and
         physical(spec['original_claim']['path'])==semantic.with_name(key+'.claim.json') and
         resume_claim==semantic.with_name(key+'.resume-attempt2.claim.json'), 'original semantic lock/claim path differs')
    need(not resume_claim.exists(),'permanent resume claim already exists; inspect actual prior attempt, never retry blindly')
    need(original_claim.get('schema')=='ultra-three-train-semantic-job-claim/v1' and
         original_claim.get('run_root')==str(run) and original_claim.get('spec_sha256')==spec['admission']['sha256'] and
         original_claim.get('contract_sha256')==spec['contract']['sha256'] and
         original_claim.get('index_sha256')==spec['full_index']['sha256'],'original semantic claim identity differs')
    cloud=admission['cloud']
    need(admission.get('schema')==('ultra-eight-three-train-spec/v2'if protocol=='legacy_frozen_v2'else'ultra-eight-three-train-spec/v3') and admission.get('code_root')==str(code) and
         admission.get('silver_contract')==spec['contract'] and admission.get('full_index')==spec['full_index'] and
         admission.get('bundle')==spec['bundle'] and cloud.get('task_root')==str(base) and cloud.get('run_root')==str(run) and
         cloud.get('python')==str(python), 'original admission differs from resume paths/pins')
    if protocol=='explicit_roots_v3':need(cloud.get('device_lock_root')==str(locks),'original explicit shared lock root differs')
    need(cloud.get('platform_workers')==1 and cloud.get('gpus_per_worker')==8 and cloud.get('priority')==6 and
         cloud.get('max_runtime_seconds')==2592000,'original one-worker eight-GPU P6 profile differs')
    pinned_file(spec['original_launcher'])
    frozen_bundle(spec)
    if protocol=='legacy_frozen_v2':
        frozen_roots(code/'gpu_launch.py',base,locks)
        legacy_media_path(spec,None)
    for name in ('identity.json','broker.json','completion_map.bin','.parent.lock','broker.lock'):
        need(physical(queue/name).is_file(),'original queue metadata missing; no automatic initialization: '+name)
    label_identity=ref_unpinned(physical(labels/version/'identity.json'))
    need(label_identity=={'schema':'redimnet2-sidecar-tree/v1','version':version,
                          'contract_sha256':spec['contract']['sha256']},'original label tree identity differs')
    progress=inspect_progress(spec['full_index']['path'],spec['full_index']['sha256'],queue,spec['contract']['sha256'])
    need(progress['committed_source_lower_bound']<index['source_count'],'old queue already complete; no successor needed')
    terminal=ref(spec['terminal_observation']);fresh(terminal)
    need(terminal.get('schema')=='annotation-writer-terminal-observation/v1' and terminal.get('task_id')==spec['old_task_id'] and
         terminal.get('queue_root')==str(queue) and terminal.get('state')in {'Stopped','Failed','Succeeded','Completed'},'old task not bound to a real terminal observation')
    workers=terminal.get('workers');need(isinstance(workers,list) and len(workers)==1 and
         workers[0].get('name')==spec['old_task_id']+'-worker-0' and
         workers[0].get('state')in {'Stopped','Failed','Succeeded','Completed'} and
         type(terminal.get('active_workers'))is int and terminal['active_workers']==0,'old platform worker still active/unobserved')
    absence=ref(spec['writer_absence']);fresh(absence)
    need(absence.get('schema')=='annotation-writer-absence-observation/v1' and absence.get('task_id')==spec['old_task_id'] and
         absence.get('run_root')==str(run) and absence.get('other_successor_tasks')==[],
         'old parent/worker absence and no competing successor evidence required')
    if protocol=='legacy_frozen_v2':
        need(absence.get('evidence_basis')=='platform_terminal_and_stable_locks' and
             absence.get('remote_proc_enumeration_performed')is False and
             absence.get('platform_instance_state')in {'Stopped','Failed','Succeeded','Completed'},
             'legacy absence must state real platform terminal plus stable-lock basis, not invented remote /proc enumeration')
    else:
        need(all(type(absence.get(k))is int and absence[k]==0 for k in ('active_parent_count','active_worker_count')),
             'explicit observed parent/worker absence required')
    checks=spec['checks'];need(isinstance(checks,list)and len(checks)==2,'original gate and mount commands required')
    commands=[]
    names=('ultra_w8_gate_v2.py','verify_ultra_cloud_mounts_v2.py')if protocol=='legacy_frozen_v2'else('cloud_gate.py','verify_cloud_mounts.py')
    for item,name,prefix in zip(checks,names,(['--live'],[])):
        script=pinned_file(item['script']);need(script.name==name,'original check script differs')
        expected=prefix+['--spec',spec['admission']['path'],'--spec-sha',spec['admission']['sha256']]
        need(item.get('args')==expected,'gate/mount arguments must bind exact original admission')
        commands.append([str(python),'-B',str(script),*expected])
    argv=[str(python),'-B',str(code/'gpu_launch.py'),'--execute','--bundle-sha',spec['bundle']['sha256'],
        '--contract',spec['contract']['path'],'--contract-sha',spec['contract']['sha256'],
        '--shard-index',spec['full_index']['path'],'--index-sha',spec['full_index']['sha256'],
        '--queue-root',str(queue),'--output-root',str(labels),'--output',str(resources),
        '--target-uuids','auto8',
        '--memory-budget-mib','60000','--max-runtime-seconds','2592000']
    if protocol=='explicit_roots_v3':argv+=['--run-root',str(base),'--device-lock-root',str(locks)]
    with contextlib.nullcontext()if _semantic_owned else existing_lock(semantic):
        with existing_lock(queue/'.parent.lock'):
            latest=inspect_progress(spec['full_index']['path'],spec['full_index']['sha256'],queue,spec['contract']['sha256'])
            need(latest['broker_snapshot_sha256']==progress['broker_snapshot_sha256'],'old broker changed before lock validation')
            partial=partial_snapshot(spec,index,contract,progress)
    return {'schema':'annotation-existing-resume-check/v1','purpose':spec['purpose'],'protocol':protocol,'progress':progress,
        'checks':commands,'gpu_command':argv,'same_queue_and_labels':True,'old_bytes_rewritten':False,
        'partial_sources':partial,'python_resolved_path':str(python_target),'python_sha256':python_sha,
        'platform_authenticated_by_this_tool':False,'GPU_started':False,'execution_admitted_by_check_only':False}


def frozen_roots(path,base,locks):
    """只解析已核旧源码的Path字面常量，不import/exec冻结程序。"""
    tree=ast.parse(physical(path).read_bytes());values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'ROOT','GLOBAL_LOCKS'}:
            value=node.value
            need(isinstance(value,ast.Call) and isinstance(value.func,ast.Name)and value.func.id=='Path' and
                 len(value.args)==1 and not value.keywords and isinstance(value.args[0],ast.Constant)and
                 isinstance(value.args[0].value,str),'legacy frozen root is not an explicit Path literal')
            need(node.targets[0].id not in values,'duplicate legacy root definition')
            values[node.targets[0].id]=value.args[0].value
    need(values=={'ROOT':str(base),'GLOBAL_LOCKS':str(locks)},'legacy frozen ROOT/GLOBAL_LOCKS differ from spec')
    need(any(isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and
        node.func.value.id=='G'and node.func.attr=='DeviceLocks'and node.args and isinstance(node.args[0],ast.Name)and
        node.args[0].id=='GLOBAL_LOCKS'for node in ast.walk(tree)),'legacy code does not use pinned GLOBAL_LOCKS')


def frozen_bundle(spec):
    """每次都重开原包及全部成员，不能只相信启动前的一次摘要。"""
    code=physical(spec['code_root'])
    need(physical(spec['bundle']['path'])==code/'bundle_files.json','original bundle path differs')
    members=ref(spec['bundle'])
    need(isinstance(members,dict)and 'gpu_launch.py'in members,'original GPU launcher absent from bundle')
    for relative,expected in members.items():
        path=Path(relative)
        need(not path.is_absolute()and '..'not in path.parts and path.parts,'unsafe frozen bundle member')
        need(sha(expected)and file_sha(code/path)==expected,'original frozen bundle member changed: '+relative)
    return members


def verify_execution_inputs(spec,result,semantic_fd):
    """准入命令之后、原GPU入口之前重核所有可执行原件与路径关系。"""
    frozen_bundle(spec)
    pinned_file(spec['original_launcher'])
    for item in spec['checks']:pinned_file(item['script'])
    pinned_file(spec['python_binary']);verify_interpreter(spec,result)
    if spec['protocol']=='legacy_frozen_v2':
        frozen_roots(physical(spec['code_root'])/'gpu_launch.py',physical(spec['task_root']),physical(spec['device_lock_root']))
        legacy_media_path(spec,None)
    current=physical(spec['semantic_lock']).stat();held=os.fstat(semantic_fd)
    need((current.st_dev,current.st_ino)==(held.st_dev,held.st_ino),'semantic lock inode replaced during checks')


def partial_snapshot(spec,index,contract,progress):
    """只读最多八个活动片；拒绝孤儿，不调用会补commit的lookup。"""
    if __package__:
        from .prepare_speed_continuation import read_shard
        from .schema import assert_valid_record
    else:
        from prepare_speed_continuation import read_shard
        from schema import assert_valid_record
    leases=progress['broker_snapshot']['leases'];need(len(leases)<=8,'old task has more than eight leased shards')
    by_id={s['id']:s for s in index['shards']};kept=[]
    for sid in sorted(leases):
        with existing_lock(physical(spec['queue_root'])/'claims'/(sid+'.lock')):
            for source in read_shard(physical(spec['full_index']['path']).parent,by_id[sid]):
                proof=retained_for_resume(source,physical(spec['output_root'])/contract['version'],spec,contract,assert_valid_record)
                if proof is not None:kept.append(proof)
    return kept


def legacy_media_path(spec,source_path):
    """仅旧协议显式固定共享挂载别名；不改source，也不放宽一般路径检查。"""
    item=spec.get('legacy_source_alias')
    need(isinstance(item,dict)and set(item)=={'alias_path','target_path'},'explicit legacy source mount alias required')
    alias=Path(item['alias_path']);target=physical(item['target_path'])
    need(alias.is_absolute()and '..'not in alias.parts,'explicit absolute legacy mount alias required')
    physical(alias.parent)
    if spec['purpose']=='production':need(str(alias)=='/root/group-shared'and str(target)=='/shared/group-shared','unrecognized production mount alias')
    need(alias.is_symlink()and target.is_dir()and alias.resolve(strict=True)==target and
         (alias.parent/alias.readlink()).resolve(strict=True)==target,'actual legacy mount alias target differs')
    if source_path is None:return target
    source=Path(source_path);need(source.is_absolute()and '..'not in source.parts,'unsafe original source path')
    if alias in source.parents:return physical(target/source.relative_to(alias))
    return physical(source)


def retained_for_resume(source,tree,spec,contract,validate):
    """局部纯读原提交；唯一例外是已明确固定的旧共享挂载别名。"""
    if __package__:
        from .prepare_speed_continuation import child,old_job_key,digest,stat_identity,identity_matches
    else:
        from prepare_speed_continuation import child,old_job_key,digest,stat_identity,identity_matches
    path=child(tree,source['sidecar_relpath']);commit=physical(path.with_name(path.name+'.commit.json'))
    if not path.exists():need(not commit.exists(),'orphan commit without sidecar');return None
    need(commit.is_file(),'orphan sidecar preserved; no implicit commit repair')
    record,raw=read_small(path,32*1024**2);marker,marker_raw=read_small(commit,64*1024);validate(record)
    contract_sha=spec['contract']['sha256'];key=old_job_key(source,contract_sha)
    need(record.get('source')==source and record['contract']['sha256']==contract_sha and
         record['contract']['version']==contract['version']and record['execution']['job_key']==key,
         'partial sidecar original source/contract/job differs')
    expected={'schema':'redimnet2-sidecar-commit/v1','source_id':source['source_id'],'job_key':key,
        'contract_sha256':contract_sha,'source_record_sha256':digest(source),
        'sidecar_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    need(marker==expected,'partial sidecar commit binding differs')
    identity=record['source_identity']
    if identity['decode_status']=='decoded_complete':
        media=legacy_media_path(spec,source['audio_path'])if spec['protocol']=='legacy_frozen_v2'else physical(source['audio_path'])
        need(media.is_file(),'original source missing')
        actual=stat_identity(media)
        identity_matches(actual, identity['stat_after'])
        need(True,'original source stat differs (device/inode/bytes/mtime/ctime preserved): '+
             json.dumps({'source_id':source['source_id'],'expected':identity['stat_after'],'actual':actual},sort_keys=True))
    need(record['decision']['usable_for_training']is False and record['decision']['usable_for_evaluation']is False,
         'partial sidecar automatically released labels')
    return {'source_id':source['source_id'],'source_record_sha256':digest(source),'sidecar_path':str(path),
        'sidecar_sha256':expected['sidecar_sha256'],'commit_path':str(commit),'commit_sha256':hashlib.sha256(marker_raw).hexdigest(),
        'producer_contract_sha256':contract_sha,'decision':record['decision']['state']}


def ref_unpinned(path):return read_small(path,1024**2)[0]


def _real_runner(argv,log,semantic_fd):
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(name,None)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    with log.open('xb')as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,pass_fds=(semantic_fd,))
        try:
            write_new(log.with_name(log.name+'.process.json'),{'schema':'annotation-resume-command-process/v1',
                'pid':process.pid,'argv':argv,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'synthetic_fixture':False})
            return process.wait()
        except BaseException:
            if process.poll()is None:
                # 原父入口收到TERM后按它自己的精确身份清理worker；不按进程名杀。
                process.send_signal(signal.SIGTERM)
                process.wait()
            raise


def run(spec_path,spec_sha,*,execute=False,runner=None):
    spec=ref({'path':str(spec_path),'sha256':spec_sha})
    need(runner is None or spec['purpose']=='cpu_fixture','injected runner is synthetic CPU only')
    need(not execute or spec['purpose']=='production' or runner is not None,'CPU fixture cannot execute real commands')
    result=inspect(spec)
    if not execute:return result
    evidence=physical(spec['evidence_root']);evidence.mkdir()
    write_new(evidence/'prepared.json',{'spec':spec,'spec_sha256':spec_sha,'check':result})
    started=0;check_started=0;claim_written=False
    try:
        # 不持独立queue FD跨exec；原gpu_launch将自己获取原父锁。
        with existing_lock(spec['semantic_lock'])as semantic_fd:
            # 已有语义FD贯穿执行；队列只读锁在检查后释放，由原GPU父入口自己取得。
            result=inspect(spec,_semantic_owned=True,_evidence_owned=True)
            need(not physical(spec['resume_claim']).exists(),'permanent resume claim appeared')
            with existing_lock(physical(spec['queue_root'])/'.parent.lock'):
                current=inspect_progress(spec['full_index']['path'],spec['full_index']['sha256'],spec['queue_root'],spec['contract']['sha256'])
                need(current['broker_snapshot_sha256']==result['progress']['broker_snapshot_sha256'],'old broker changed before claim')
            for pin in ('contract','full_index','admission','original_claim','terminal_observation','writer_absence'):
                ref(spec[pin])
            fresh(ref(spec['terminal_observation']));fresh(ref(spec['writer_absence']))
            claim={'schema':'annotation-existing-resume-claim/v1','attempt_number':2,'spec_sha256':spec_sha,
                'old_task_id':spec['old_task_id'],'original_claim':spec['original_claim'],'run_root':spec['run_root'],
                'queue_root':spec['queue_root'],'output_root':spec['output_root'],'resource_output':spec['resource_output'],
                'contract_sha256':spec['contract']['sha256'],'index_sha256':spec['full_index']['sha256'],
                'status':'explicit_resume_claimed_no_implicit_retry','synthetic_fixture':spec['purpose']=='cpu_fixture'}
            write_new(spec['resume_claim'],claim);claim_written=True
            write_new(evidence/'claimed.json',claim)
            invoke=runner or _real_runner
            for i,command in enumerate(result['checks']):
                pinned_file(spec['checks'][i]['script']);check_started+=1
                verify_interpreter(spec,result)
                write_new(evidence/('check-%d-started.json'%i),{'argv':command,'ordinal':i})
                rc=invoke(command,evidence/('check-%d.log'%i),semantic_fd)
                write_new(evidence/('check-%d-result.json'%i),{'returncode':rc})
                need(type(rc)is int and rc==0,'original gate/mount command failed')
            # 门耗时后再核旧终态时效、关键旧字节、稳定broker和仍不存在的新资源目录。
            fresh(ref(spec['terminal_observation']));fresh(ref(spec['writer_absence']))
            for pin in ('contract','full_index','admission','original_claim'):
                ref(spec[pin])
            verify_execution_inputs(spec,result,semantic_fd)
            need(not physical(spec['resource_output']).exists(),'resource attempt output appeared before launch')
            with existing_lock(physical(spec['queue_root'])/'.parent.lock'):
                latest=inspect_progress(spec['full_index']['path'],spec['full_index']['sha256'],spec['queue_root'],spec['contract']['sha256'])
                need(latest['broker_snapshot_sha256']==current['broker_snapshot_sha256'],'old broker changed after gate')
                need(partial_snapshot(spec,ref(spec['full_index']),ref(spec['contract']),latest)==result['partial_sources'],
                     'old partial results changed during gate; preserve evidence before any GPU call')
            verify_execution_inputs(spec,result,semantic_fd)
            started=1;write_new(evidence/'gpu-start-intent.json',{'launch_attempt_count':1,'argv':result['gpu_command']})
            rc=invoke(result['gpu_command'],evidence/'gpu-parent.log',semantic_fd)
            write_new(evidence/'gpu-exit.json',{'returncode':rc,'launch_count':started})
            need(type(rc)is int and rc==0,'original GPU parent failed; inspect preserved attempt, do not auto retry')
            receipt=physical(spec['resource_output'])/'parent_result.json';value=ref_unpinned(receipt)
            need(value.get('schema')=='open-annotation-independent-gpu-run/v1' and value.get('contract_sha256')==spec['contract']['sha256'] and
                 value.get('index_sha256')==spec['full_index']['sha256'] and value.get('bundle_sha256')==spec['bundle']['sha256'] and
                 value.get('worker_count')==8 and value.get('resource_returned')is True and
                 value.get('verification',{}).get('passed',value.get('verification',{}).get('pass',False))is True,
                 'original parent full receipt incomplete; exit0 is not success evidence')
            final={'schema':'annotation-existing-resume-result/v1','launch_count':started,'check_count':check_started,
                'parent_result':{'path':str(receipt),'sha256':file_sha(receipt)},'synthetic_fixture':spec['purpose']=='cpu_fixture',
                'original_claim_preserved':True,'same_contract_index_queue_labels':True,'training_labels_released':False}
            write_new(evidence/'completed.json',final);return final
    except BaseException:
        actual=started if spec['purpose']=='cpu_fixture'else int((evidence/'gpu-parent.log.process.json').is_file())
        write_new(evidence/'failure.json',{'traceback':traceback.format_exc(),'launch_count':actual,'launch_attempt_count':started,
            'check_count':check_started,'permanent_resume_claim_written':claim_written,'automatic_retry_allowed':False})
        raise


def verify_interpreter(spec,result):
    p=Path(spec['python']).resolve(strict=True)
    need(str(p)==result['python_resolved_path'] and file_sha(p)==result['python_sha256'],
         'original interpreter changed before execution')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.spec,a.spec_sha,execute=a.execute),ensure_ascii=False))


if __name__=='__main__':main()
