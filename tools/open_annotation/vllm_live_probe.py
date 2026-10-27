"""真实模型构造已关闭；目前仅检查文件和读取完整阶段缓存。

新目录保存实验身份、整批阶段结果和每次恢复回执。这里不授予共享GPU的许可，
闲卡与原独占锁门只保留为CPU协议检查，不能代替尚未实现的CUDA完整生命周期。
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
import tempfile
import traceback
import uuid

if __package__:
    from . import vllm_backend as V
    from .asr_stage_store import ASRStageStore, canonical, digest, physical, require
else:
    import vllm_backend as V
    from asr_stage_store import ASRStageStore, canonical, digest, physical, require

SCHEMA='annotation-bounded-vllm-probe/v1'
MAX_WALL_SECONDS=600


def read_ref(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'exact file reference required')
    p=physical(ref['path']);raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'file SHA differs: '+str(p))
    return json.loads(raw)


def write_once(path,value):
    raw=canonical(value);p=physical(path)
    if p.exists():
        require(p.read_bytes()==raw,'immutable experiment output differs: '+str(p));return
    # hard-link publishes complete bytes without overwriting another result.
    tmp=p.parent/('.publishing-'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    try:os.link(tmp,p)
    finally:tmp.unlink()
    fd=os.open(str(p.parent),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def inspect_experiment(ref):
    """CPU检查会核模型、代码、PCM文件，不导入GPU库，也不新建输出目录。"""
    config=read_ref(ref)
    require(config.get('schema')==SCHEMA and config.get('model_API_calls_allowed')==0 and
            type(config['model_API_calls_allowed']) is int and config.get('release_training_labels') is False,
            'bounded offline pilot required')
    spec=V.validate_spec(read_ref(config['adapter_spec']))
    require(spec['max_batch_rows']<=2 and spec['decode']['max_tokens']<=512 and
            spec['engine']['gpu_memory_utilization']<=0.20,'single-card pilot exceeds hard batch/token/memory cap')
    require(type(config.get('wall_seconds')) is int and 1<=config['wall_seconds']<=MAX_WALL_SECONDS,
            'explicit finite wall timeout required')
    pins=config.get('probe_source_files',{})
    own={str(physical(__file__)),str(physical(V.__file__)),str(physical(sys.modules[ASRStageStore.__module__].__file__))}
    require(isinstance(pins,dict) and own<=set(pins),'probe/adapter/store source pins required')
    for p,h in pins.items():require(V._file_sha(p)==h,'probe source SHA differs')
    V.verify_local_snapshot(spec['model'])
    for p,h in spec.get('source_files',{}).items():require(V._file_sha(p)==h,'engine source SHA differs')
    require(spec['purpose']=='cpu_fixture' or bool(spec.get('source_files')),'real engine source pins required')
    items=read_ref(config['inputs']);require(isinstance(items,list) and 1<=len(items)<=8,'one to eight fixed requests required')
    import numpy as np
    rows=[];seen=set();total=0
    for item in items:
        require(isinstance(item,dict) and isinstance(item.get('source_id'),str) and item['source_id'] and
                isinstance(item.get('window_id'),str) and item['window_id'],'source/window identity missing')
        key=(item['source_id'],item['window_id']);require(key not in seen,'duplicate source/window');seen.add(key)
        frames=item.get('model_input_frames');require(type(frames) is int and 0<frames<=30*16000 and
              item.get('model_sample_rate')==16000,'PCM exceeds 30 seconds or changes source clock')
        p=physical(item['pcm_path']);require(p.suffix=='.npy' and V._file_sha(p)==item['pcm_file_sha256'],'fixed PCM file SHA differs')
        # 文件上限避免损坏的npy文件先分配巨大数组；绝不启用pickle。
        require(p.stat().st_size<=frames*4+65536,'PCM file larger than declared bound')
        wave=np.load(p,allow_pickle=False)
        require(wave.dtype==np.dtype('float32') and wave.ndim==1 and len(wave)==frames and np.isfinite(wave).all(),
                'finite mono float32 PCM required')
        require(hashlib.sha256(wave.astype('<f4',copy=False).tobytes()).hexdigest()==item['input_pcm_sha256'],'decoded PCM SHA differs')
        rows.append({k:item[k] for k in ('source_id','window_id','model_input_frames','model_sample_rate','input_pcm_sha256')})
        rows[-1]['audio']=wave;total+=frames
    require(total<=120*16000,'total PCM exceeds 120 seconds')
    out=physical(config['output_root']);protected=config.get('protected_roots')
    require(isinstance(protected,list) and protected and all(isinstance(p,str) and Path(p).is_absolute() for p in protected),
            'explicit old producer protected roots required')
    for p in protected+[spec['model']['path']]:
        p=physical(p);require(out!=p and p not in out.parents and out not in p.parents,'output overlaps protected tree')
    require(out!=physical(ref['path']) and all(out!=physical(x['pcm_path']) for x in items),'output replaces experiment input')
    identity={'schema':SCHEMA,'experiment_reference':ref,'config_sha256':digest(config),'producer_sha256':digest(spec),
              'inputs_sha256':config['inputs']['sha256'],'synthetic_fixture':spec['purpose']=='cpu_fixture'}
    if out.exists():require((out/'experiment.json').is_file() and json.loads((out/'experiment.json').read_bytes())==identity,
                           'existing output is not this exact experiment')
    versions={}
    for name in ('torch','transformers','qwen-asr','vllm'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    report={'schema':'annotation-bounded-vllm-inspection/v1','request_count':len(rows),'PCM_seconds':total/16000,
            'installed_versions_metadata_only':versions,'GPU_libraries_imported_by_probe':False,
            'GPU_execution_admitted':False,'real_engine_execution_enabled':False,'GPU_lifecycle_cleanup_verified':False,'quality_verified':False,'source_audio_identity_verified':False,'input_model_PCM_verified':True,
            'formal_admission':False,'model_API_calls':0,'synthetic_fixture':identity['synthetic_fixture'],
            'existing_compute_sharing_supported':False}
    return config,spec,rows,out,identity,report


def validate_numeric_mapping(mapping,gpu_uuid,index):
    """只支持已实证NVML连续编号恰与PCI总线排序一致的整卡机器。"""
    require(type(index) is int and index>=0 and isinstance(mapping,list) and bool(mapping),
            'physical index/PCI mapping missing')
    pci=[]
    for row in mapping:
        require(type(row.get('physical_index')) is int and isinstance(row.get('gpu_uuid'),str) and
                isinstance(row.get('pci_bus_id'),str),'physical index/PCI mapping fields missing')
        match=re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',row['pci_bus_id'])
        require(match is not None,'physical index/PCI bus identity malformed')
        pci.append(tuple(int(v,16) for v in match.groups()))
    require([v['physical_index'] for v in mapping]==list(range(len(mapping))) and
            len({v['gpu_uuid'] for v in mapping})==len(mapping) and len(set(pci))==len(mapping) and pci==sorted(pci),
            'physical index order does not uniquely match contiguous PCI bus order')
    selected=[v for v in mapping if v['gpu_uuid']==gpu_uuid]
    require(len(selected)==1 and selected[0]['physical_index']==index,
            'physical index is not bound to the exact GPU UUID in live mapping')


def capture_snapshot(gpu_uuid):
    """现场短命只读命令；有任意计算PID即保持忙碌，不用映射证明给共享放行。"""
    def query(flag,fields):
        text=subprocess.run(['nvidia-smi',flag+'='+fields,'--format=csv,noheader,nounits'],check=True,
                            capture_output=True,text=True,timeout=15).stdout
        return [[x.strip() for x in line.split(',')] for line in text.splitlines() if line.strip()]
    gpus=query('--query-gpu','index,uuid,pci.bus_id,memory.total,memory.used')
    require(all(len(v)==5 and v[0].isdigit() for v in gpus),'numeric NVIDIA physical indices required')
    mapping=sorted([{'physical_index':int(v[0]),'gpu_uuid':v[1],'pci_bus_id':v[2]} for v in gpus],key=lambda v:v['physical_index'])
    require(len({v['physical_index'] for v in mapping})==len(mapping) and
            len({v['gpu_uuid'] for v in mapping})==len(mapping),'ambiguous NVIDIA index/UUID mapping')
    found=[v for v in gpus if v[1]==gpu_uuid];require(len(found)==1,'exact GPU UUID not found')
    validate_numeric_mapping(mapping,gpu_uuid,int(found[0][0]))
    procs=[]
    for row in query('--query-compute-apps','gpu_uuid,pid,used_gpu_memory'):
        if row[0]!=gpu_uuid:continue
        procs.append({'pid':int(row[1]),'used_memory_mib':int(row[2])})
    return {'gpu_uuid':gpu_uuid,'physical_index':int(found[0][0]),'index_uuid_map':sorted(mapping,key=lambda v:v['physical_index']),
            'total_memory_mib':int(found[0][3]),'used_memory_mib':int(found[0][4]),
            'hostname':socket.gethostname(),'observed_unix':time.time(),'processes':sorted(procs,key=lambda x:x['pid'])}


def check_admission(config,spec,*,snapshot=capture_snapshot,pause=time.sleep):
    grant=read_ref(config['resource_admission'])
    require(grant.get('schema')=='annotation-idle-GPU-exclusive-admission/v1' and
            grant.get('idle_gpu_test_authorized') is True and grant.get('shared_gpu_test_authorized') is False and
            grant.get('allowed_existing_processes')==[] and grant.get('task_id') and
            grant.get('hostname')==socket.gethostname(),
            'shared GPU requests are unsupported; exact idle-only exclusive permission required')
    expires=dt.datetime.fromisoformat(grant['expires_utc']);require(expires.tzinfo is not None and
        dt.datetime.now(dt.timezone.utc)<expires,'resource permission expired')
    require(grant.get('producer_sha256')==digest(spec) and grant.get('inputs_sha256')==config['inputs']['sha256'],
            'resource permission belongs to another model/input experiment')
    gpu=grant.get('gpu_uuid');require(isinstance(gpu,str) and gpu.startswith('GPU-'),'exact GPU UUID required')
    require(type(grant.get('headroom_mib')) is int and grant['headroom_mib']>=8192 and
            type(grant.get('max_probe_memory_mib')) is int and grant['max_probe_memory_mib']>0,'explicit bounded memory/headroom required')
    require(type(grant.get('driver_used_limit_mib')) is int and 0<=grant['driver_used_limit_mib']<=256,
            'conservative idle driver memory limit <=256 MiB required')
    snapshots=[]
    for i in range(2):
        if i:pause(5)
        current=snapshot(gpu);snapshots.append(current)
        require(current['hostname']==grant['hostname'] and current['gpu_uuid']==gpu and
                0<=time.time()-current['observed_unix']<=10,'GPU snapshot host/time differs')
        validate_numeric_mapping(current.get('index_uuid_map'),gpu,current.get('physical_index'))
        require(isinstance(current.get('processes'),list) and not current['processes'],
                'existing compute PID on requested GPU; never share or bypass old exclusive guards')
        require(0<=current['used_memory_mib']<=grant['driver_used_limit_mib'],
                'GPU is not idle: driver-used memory exceeds conservative limit')
        budget=spec['engine']['gpu_memory_utilization']*current['total_memory_mib']
        require(budget<=grant['max_probe_memory_mib'] and current['used_memory_mib']+budget+grant['headroom_mib']<=current['total_memory_mib'],
                'explicit total-card budget/headroom not satisfied')
    require(snapshots[0]['total_memory_mib']==snapshots[1]['total_memory_mib'],'GPU memory identity changed')
    require(snapshots[0]['physical_index']==snapshots[1]['physical_index'] and snapshots[0]['index_uuid_map']==snapshots[1]['index_uuid_map'],
            'live UUID to physical index mapping changed between snapshots')
    require(dt.datetime.now(dt.timezone.utc)<expires,'resource permission expired during double snapshot')
    return grant,snapshots


@contextlib.contextmanager
def existing_exclusive_lock(config,spec):
    """沿用真实旧票据的同一个锁inode，不能自建一套路径声称兼容。"""
    grant=read_ref(config['resource_admission'])
    require(grant.get('schema')=='annotation-idle-GPU-exclusive-admission/v1' and
            grant.get('idle_gpu_test_authorized') is True and grant.get('shared_gpu_test_authorized') is False and
            grant.get('allowed_existing_processes')==[],
            'shared GPU requests are unsupported; idle-only exclusive permission required')
    gpu=grant.get('gpu_uuid');require('existing_exclusive_ticket' in grant,'actual existing exclusive lock ticket required')
    ticket=read_ref(grant['existing_exclusive_ticket'])
    require(ticket.get('schema')=='open-annotation-resource-ticket/v1' and ticket.get('target_uuid')==gpu and
            isinstance(grant.get('exclusive_protocol_bundle_sha256'),str) and
            re.fullmatch('[0-9a-f]{64}',grant['exclusive_protocol_bundle_sha256']) and
            ticket.get('bundle_sha256')==grant['exclusive_protocol_bundle_sha256'] and
            isinstance(ticket.get('locks'),list),'fixed existing exclusive protocol ticket required')
    for name in ('gpu_launch.py','gpu_challenge.py','gpu_vendor/resources.py'):
        source=physical(Path(__file__).parent/name)
        require(config['probe_source_files'].get(str(source))==V._file_sha(source),
                'existing exclusive launcher/challenge/lock source pins required')
    if spec['purpose']!='cpu_fixture':
        require(ticket.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'exclusive lock ticket belongs to another machine boot')
    selected=[v for v in ticket['locks'] if v.get('uuid')==gpu]
    require(len(selected)==1,'existing exclusive ticket UUID lock missing or ambiguous')
    old=selected[0];path=physical(old['path'])
    require(path.name==gpu+'.lock','existing exclusive lock filename differs')
    # 不创建、不写入、不删除旧锁。原parent占有同一锁时，下面必须立即拒绝。
    fd=os.open(str(path),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    try:
        info=os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.getuid() and
                (info.st_dev,info.st_ino)==(old['device'],old['inode']),
                'existing exclusive lock identity changed')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('existing exclusive GPU owner still holds original lock') from exc
        yield {'uuid':gpu,'path':str(path),'device':info.st_dev,'inode':info.st_ino,
               'original_protocol_lock_held_until_probe_end':True,
               'ticket_sha256':grant['existing_exclusive_ticket']['sha256']}
    finally:os.close(fd)


class _NoInference:
    def enqueue(self,*args,**kwargs):raise RuntimeError('cache-only restore must not enqueue')
    def wait_for_completion(self,*args,**kwargs):raise RuntimeError('cache-only restore must not wait')


@contextlib.contextmanager
def cooperative_lock(path):
    """只使用本助手独立锁；不读写旧标注器或PTN锁。"""
    fd=os.open(str(physical(path)),os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('another bounded probe owns this experiment/GPU lock') from exc
        yield
    finally:os.close(fd)


def run_experiment(ref,*,factory=None,snapshot=capture_snapshot,pause=time.sleep,after_durable_commit=None):
    require(factory is not None or os.environ.get('REDIM_PROBE_SUPERVISOR_PID')==str(os.getppid()),'real execution requires bounded CLI supervisor')
    config,spec,_,out,identity,_=inspect_experiment(ref)
    require(factory is None or spec['purpose']=='cpu_fixture','injected factory is restricted to explicit CPU fixtures')
    out.mkdir(parents=True,exist_ok=True);write_once(out/'experiment.json',identity)
    with cooperative_lock(out/'.bounded-probe.lock'):
        with contextlib.ExitStack() as locks:
            return _run_experiment(ref,factory=factory,snapshot=snapshot,pause=pause,
                                   after_durable_commit=after_durable_commit,locks=locks)


def _run_experiment(ref,*,factory,snapshot,pause,after_durable_commit,locks):
    config,spec,rows,out,identity,inspection=inspect_experiment(ref)
    require(factory is not None or spec['purpose']=='pilot_validation','real entry cannot execute CPU fixture')
    out.mkdir(parents=True,exist_ok=True);write_once(out/'experiment.json',identity)
    store=ASRStageStore(out/'stage',digest(spec),synthetic_fixture=spec['purpose']=='cpu_fixture')
    cached=V.VLLMBackend(spec,_NoInference(),store,parse_answer=lambda *_:None,prompt_builder=lambda *_:'never',
                         synthetic_fixture=spec['purpose']=='cpu_fixture')
    metadata=[{k:v for k,v in row.items() if k!='audio'} for row in rows]
    with store.locked():complete=all(store.lookup(cached.request_identity(row)) is not None for row in metadata)
    snapshots=[];grant=None;device_selection=None;exclusive_lock=None
    if complete:adapter=cached
    else:
        # 禁止真实构造：原锁尚不能由supervisor持有到全部自身上下文排空。
        # 完整cache恢复走上方分支；不得靠许可JSON或自定义pilot factory重开。
        require(factory is not None,'real GPU construction is disabled until supervisor/group/context lifetime is verified; inspect or full-cache restore only')
        preview=read_ref(config['resource_admission']);gpu=preview.get('gpu_uuid')
        require(isinstance(gpu,str) and gpu.startswith('GPU-') and
                all(c.isalnum() or c=='-' for c in gpu),'safe exact GPU UUID required')
        # 新命名空间的协作锁仅约束本助手；现有标注器没有参与此锁协议。
        locks.enter_context(cooperative_lock(Path(tempfile.gettempdir()).resolve()/('redimnet2-bounded-vllm-'+gpu+'.lock')))
        exclusive_lock=locks.enter_context(existing_exclusive_lock(config,spec))
        grant,snapshots=check_admission(config,spec,snapshot=snapshot,pause=pause)
        current_lock=physical(exclusive_lock['path']).stat()
        require((current_lock.st_dev,current_lock.st_ino)==(exclusive_lock['device'],exclusive_lock['inode']),
                'original exclusive lock path changed during idle checks')
        numeric=str(snapshots[-1]['physical_index'])
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in (None,numeric,grant['gpu_uuid']),'CUDA device selection conflicts with permission')
        require(os.environ.get('CUDA_DEVICE_ORDER') in (None,'PCI_BUS_ID'),'CUDA physical device order conflicts with inspected mapping')
        # vLLM0.19.1会把该环境字段转成int。资源身份和锁仍使用UUID，数字只来自双快照。
        device_selection={'gpu_uuid':grant['gpu_uuid'],'physical_index':snapshots[-1]['physical_index'],
                          'CUDA_VISIBLE_DEVICES':numeric,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
                          'mapping_evidence':'two_fresh_nvidia_smi_index_UUID_PCI_snapshots_with_contiguous_PCI_order',
                          'CUDA_runtime_UUID_verified':False}
        os.environ['CUDA_VISIBLE_DEVICES']=numeric;os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        adapter=(factory or V.build_real_engine)(spec,out/'stage',execute=True)
    runs=[];results=[]
    for start in range(0,len(rows),spec['max_batch_rows']):
        run=adapter.run_batch((metadata if complete else rows)[start:start+spec['max_batch_rows']],after_durable_commit=after_durable_commit)
        runs.append(run['metrics']);results.extend(run['results'])
    packet={'schema':'annotation-bounded-vllm-native-results/v1','experiment':identity,'results':results,
            'formal_admission':False,'quality_verified':False,'model_API_calls':0}
    write_once(out/'native-results.json',packet)
    receipt={'schema':'annotation-bounded-vllm-attempt/v1','inspection':inspection,'stage_metrics':runs,
             'cache_only_restoration':complete,'GPU_snapshots_before_construction':snapshots,'device_selection':device_selection,
             'original_exclusive_lock':exclusive_lock,'existing_compute_sharing_supported':False,
             'real_engine_execution_enabled':False,'GPU_lifecycle_cleanup_verified':False,
             'resource_admission_sha256':config.get('resource_admission',{}).get('sha256') if grant else None,
             'runtime_facts':adapter.runtime_facts,'native_results_sha256':V._file_sha(out/'native-results.json'),
             'formal_admission':False,'quality_verified':False,'model_API_calls':0}
    attempts=out/'attempts';attempts.mkdir(exist_ok=True);write_once(attempts/(uuid.uuid4().hex+'.json'),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True)
    group=parser.add_mutually_exclusive_group();group.add_argument('--inspect-only',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS);a=parser.parse_args();ref={'path':a.spec,'sha256':a.spec_sha}
    config,*_,report=inspect_experiment(ref)
    if not a.execute:print(json.dumps(report,ensure_ascii=False));return
    if a._worker:
        require(os.environ.get('REDIM_PROBE_SUPERVISOR_PID')==str(os.getppid()),'bounded supervisor required')
        try:print(json.dumps(run_experiment(ref),ensure_ascii=False))
        except BaseException:
            out=physical(config['output_root'])
            if out.is_dir() and (out/'experiment.json').is_file():
                errors=out/'errors';errors.mkdir(exist_ok=True)
                write_once(errors/(uuid.uuid4().hex+'.json'),{'error':traceback.format_exc(),'formal_admission':False})
            raise
        return
    env=dict(os.environ,REDIM_PROBE_SUPERVISOR_PID=str(os.getpid()))
    command=[sys.executable,'-B',str(physical(__file__)),'--spec',a.spec,'--spec-sha',a.spec_sha,'--execute','--_worker']
    child=subprocess.Popen(command,env=env,start_new_session=True)
    try:code=child.wait(timeout=config['wall_seconds'])
    except subprocess.TimeoutExpired:
        # 只结束本助手新建的进程组；绝不发送信号给现场已存在的PID。
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        out=physical(config['output_root'])
        if out.is_dir() and (out/'experiment.json').is_file():
            errors=out/'errors';errors.mkdir(exist_ok=True)
            write_once(errors/(uuid.uuid4().hex+'.json'),{'error':'bounded_probe_wall_timeout','wall_seconds':config['wall_seconds'],
                'terminated_own_process_group':child.pid,'formal_admission':False})
        raise SystemExit(124)
    except BaseException:
        # 用户中断时也只清理本次新进程组；保留已有训练和标注进程。
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        raise
    raise SystemExit(code)


if __name__=='__main__':main()
