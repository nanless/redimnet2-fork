"""Read only 8 registered ranks + 192 registered loader processes on the cloud.

No Torch, CUDA, audio, model, lock acquisition or signals. Never enumerate /proc
or unrelated environments. Evidence files describe past admission; known PIDs
are checked against current /proc twice before a live-process claim is made.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess


def need(value,message):
    if not value:raise ValueError(message)


def sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None


def physical(path):
    p=Path(path).absolute();current=Path(p.anchor)
    for part in p.parts[1:]:
        need(part not in ('.','..'),'unsafe path');current=current/part
        need(not current.is_symlink(),'evidence/output path symlink refused')
    return p


def read_json(path):
    p=physical(path);before=p.stat();need(stat.S_ISREG(before.st_mode)and before.st_size<=4*1024**2,'bounded regular evidence required')
    raw=p.read_bytes();after=p.stat()
    snap=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(snap(before)==snap(after),'evidence changed during read')
    def pairs(items):
        value={}
        for k,v in items:need(k not in value,'duplicate JSON key');value[k]=v
        return value
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    return value,hashlib.sha256(raw).hexdigest()


def hash_small(path):
    p=physical(path);before=p.stat();need(stat.S_ISREG(before.st_mode)and before.st_size<=4*1024**2,'bounded small code file required')
    raw=p.read_bytes();after=p.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'small code file changed')
    return hashlib.sha256(raw).hexdigest()


def proc_identity(pid,proc=Path('/proc')):
    need(type(pid)is int and pid>0,'positive integer PID required')
    raw=(proc/str(pid)/'stat').read_text();end=raw.rfind(')');need(end>0,'malformed proc stat')
    need(int(raw.split('(',1)[0].strip())==pid,'proc PID differs')
    values=raw[end+2:].split();need(len(values)>=20 and values[0]not in {'Z','X','x'},'process dead/zombie or malformed')
    return {'pid':pid,'state':values[0],'ppid':int(values[1]),'pgid':int(values[2]),'sid':int(values[3]),'start_ticks':int(values[19])}


def stable_identity(value):
    return {key:value[key] for key in ('pid','ppid','pgid','sid','start_ticks')}


def env_fields(pid,proc):
    keep={'PTN_RESUME_NONCE','PTN_BUNDLE_MANIFEST_SHA','RANK','LOCAL_RANK','WORLD_SIZE','CUDA_VISIBLE_DEVICES'}
    raw=(proc/str(pid)/'environ').read_bytes();need(len(raw)<=1024**2,'rank environment exceeds bound')
    fields={}
    for item in raw.split(b'\0'):
        key,sep,value=item.partition(b'=')
        if sep and key.decode(errors='replace')in keep:
            key=key.decode();need(key not in fields,'duplicate required environment key');fields[key]=value.decode()
    return fields


def gpu_query():
    commands=[['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
              ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']]
    rows=[]
    for command in commands:
        p=subprocess.run(command,capture_output=True,text=True,timeout=8)
        rows.append({'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    need(all(r['returncode']==0 and not r['stderr'].strip()for r in rows),'nvidia-smi query failed')
    devices=[]
    for row in csv.reader(io.StringIO(rows[0]['stdout'])):
        need(len(row)==5,'GPU query format differs');index,uuid,used,total,util=[s.strip()for s in row]
        need(all(x.isdigit()for x in (index,used,total,util)),'GPU query numeric fields invalid')
        devices.append({'index':int(index),'uuid':uuid,'used_MiB':int(used),'total_MiB':int(total),'utilization_percent':int(util)})
    return {'devices':devices,'raw_queries':rows,'compute_PID_is_not_container_PID_translation':True}


def probe(run_root,contract_sha,package_sha,expected_hostname,epoch=48,*,proc=Path('/proc'),gpu_reader=gpu_query):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    need(socket.gethostname()==expected_hostname,'wrong cloud hostname; developer process evidence is not admissible')
    need(sha(contract_sha)and sha(package_sha)and type(epoch)is int and 48<=epoch<=120,'invalid explicit identity/epoch')
    root=physical(run_root);checks=root/'checks';admission,admission_sha=read_json(checks/'parent_admission.json')
    need(admission['contract_sha256']==contract_sha and admission['package_manifest_sha256']==package_sha,'parent contract/package differs')
    package=physical(Path(admission['source']['source']).parent)
    contract,contract_actual_sha=read_json(package/'resume-contract.json');manifest,manifest_actual_sha=read_json(package/'bundle_files.json')
    need(contract_actual_sha==contract_sha and manifest_actual_sha==package_sha
         and manifest['resume-contract.json']==contract_sha and contract['schema']=='ptn-resume-model47/v1'
         and Path(contract['new_run_root'])==root and Path(contract['source_root'])==package/'source','actual package/contract bytes or RUN differ')
    gate_sha=hash_small(package/'runtime/state_gate.py')
    need(gate_sha==contract['runtime_files']['state_gate.py']==manifest['runtime/state_gate.py'],'actual state gate SHA differs')
    nonce=admission['nonce'];need(isinstance(nonce,str)and re.fullmatch('[0-9a-f]{48}',nonce),'parent nonce invalid')
    uuids=admission['GPU']['UUIDs'];need(isinstance(uuids,list)and len(uuids)==len(set(uuids))==8,'eight admitted GPU UUIDs required')
    parent=proc_identity(admission['parent']['pid'],proc)
    need(parent['start_ticks']==admission['parent']['start_ticks'],'parent PID generation changed')
    gates=[];evidence={str(checks/'parent_admission.json'):admission_sha,str(package/'resume-contract.json'):contract_actual_sha,
                      str(package/'bundle_files.json'):manifest_actual_sha,str(package/'runtime/state_gate.py'):gate_sha}
    for rank in range(8):
        path=checks/'loader_workers'/('gate_epoch%d_rank%d.json'%(epoch,rank));gate,pin=read_json(path);evidence[str(path)]=pin
        need(gate.get('schema')=='ptn-loader-gate/v1'and gate['rank']==rank and gate['epoch']==epoch
             and gate['world_size']==8 and gate['workers_per_rank']==24 and gate['global_verified_workers']==192
             and gate['before_first_batch_yield']is True and len(gate['ranks'])==8,'loader gate identity/denominator differs')
        gates.append(gate)
    gathered=gates[0]['ranks'];need(all(g['ranks']==gathered for g in gates),'eight loader gathered states differ')
    ranks=[];workers=[];pids=set();initial={}
    for rank in range(8):
        ready_path=checks/('worker_ready_rank%d.json'%rank);ready,rpin=read_json(ready_path);evidence[str(ready_path)]=rpin
        restore_path=checks/('restore_verified_rank%d.json'%rank);restore,spin=read_json(restore_path);evidence[str(restore_path)]=spin
        need(ready['rank']==rank and ready['world_size']==8 and ready['nonce']==nonce
             and ready['contract_sha256']==contract_sha and ready['package_sha256']==package_sha,'rank ready binding differs')
        need(restore.get('schema')=='ptn-model47-exact-state-proof/v1'and restore['rank']==rank and restore['world_size']==8
             and restore['contract_sha256']==contract_sha and restore['model_exact']is True
             and restore['ordered_optimizer_exact']is True and restore['scaler_exact']is True
             and restore['state_gate_sha256']==gate_sha,'rank restore proof/gate SHA differs')
        actual=proc_identity(ready['pid'],proc)
        need(all(actual[k]==ready[k]for k in ('pid','ppid','pgid','sid','start_ticks')),'rank PID generation/group changed')
        leader=proc_identity(actual['pgid'],proc);need(leader['start_ticks']==ready['group_start_ticks']and leader['sid']==actual['sid'],'rank process group generation differs')
        need(actual['pid']not in pids,'duplicate rank PID');pids.add(actual['pid']);initial[actual['pid']]=actual
        environment=env_fields(actual['pid'],proc)
        need(environment=={'PTN_RESUME_NONCE':nonce,'PTN_BUNDLE_MANIFEST_SHA':package_sha,'RANK':str(rank),
             'LOCAL_RANK':str(rank),'WORLD_SIZE':'8','CUDA_VISIBLE_DEVICES':','.join(uuids)},'actual rank required environment differs')
        lock=ready['gpu_lock'];expected=admission['GPU']['lock_evidence'][rank]
        need(lock['path']==expected['path']and lock['uuid']==expected['uuid']==uuids[rank]
             and type(lock['fd'])is int and lock['fd']>=0,'rank GPU FD selection differs')
        target=physical(lock['path']);fdpath=proc/str(actual['pid'])/'fd'/str(lock['fd']);fd=fdpath.stat();st=target.stat()
        need(fdpath.resolve()==target.resolve()and stat.S_ISREG(fd.st_mode)and fd.st_nlink==1
             and (fd.st_dev,fd.st_ino)==(st.st_dev,st.st_ino)==(expected['device'],expected['inode'])
             ==(lock['device'],lock['inode']),'actual GPU FD/inode differs')
        local=gathered[rank]
        need(local['rank']==rank and local['world_size']==8 and local['epoch']==epoch and local['parent_rank_pid']==actual['pid']
             and local['expected_workers']==local['worker_count']==len(local['proofs'])==24 and local['error']is None,'rank loader parent/gate differs')
        need(isinstance(local['nonce'],str)and re.fullmatch('[0-9a-f]{48}',local['nonce']),'loader nonce invalid')
        need({x['worker_id']for x in local['proofs']}==set(range(24)),'loader worker IDs missing/duplicate')
        for item in sorted(local['proofs'],key=lambda v:v['worker_id']):
            worker_id=item['worker_id'];path=checks/'loader_workers'/('epoch%d'%epoch)/('rank%d'%rank)/('worker%d.json'%worker_id)
            need(Path(item['path'])==path,'loader proof path differs from RUN/epoch/rank')
            v,pin=read_json(path);need(pin==item['sha256'],'loader proof SHA differs');evidence[str(path)]=pin
            need(v.get('schema')=='ptn-loader-worker-audit/v1'and v['rank']==rank and v['worker_id']==worker_id
                 and v['epoch']==epoch and v['world_size']==v['actual_dist_world_size']==v['sampler_inherited_world_size']==8
                 and v['num_workers']==24 and v['actual_dist_rank']==v['sampler_inherited_rank']==rank
                 and v['sampler_epoch']==epoch and v['nonce']==local['nonce']and v['bundle_sha256']==package_sha
                 and v['ppid']==actual['pid']and v['pid']==item['pid'],'loader evidence identity differs')
            need(v['actual_context']=='fork'and v['actual_dist_initialized']is True and v['sampler_partition']is True
                 and v['legacy_aug_behavior']is True and v['original_seed_worker_called']is True
                 and v['worker_collective_called']is False and v['worker_GPU_operation_called']is False,'loader inherited behavior differs')
            live=proc_identity(v['pid'],proc)
            need(live['pid']not in pids and live['ppid']==actual['pid']and live['start_ticks']==v['start_ticks']
                 and live['start_ticks']>=actual['start_ticks']and live['pgid']==actual['pgid']and live['sid']==actual['sid'],
                 'loader PID reused/reparented/group changed/duplicate')
            pids.add(live['pid']);initial[live['pid']]=live;workers.append(dict(live,rank=rank,worker_id=worker_id,epoch=epoch,proof_sha256=pin))
        ranks.append(dict(actual,rank=rank,environment_checked_keys=sorted(environment),ready_sha256=rpin,restore_sha256=spin,GPU_UUID=uuids[rank],GPU_fd_identity_verified=True))
    need(len(ranks)==8 and len(workers)==192 and len(pids)==200,'live process denominator incomplete')
    GPU=gpu_reader();need([d['index']for d in GPU['devices']]==list(range(8))and [d['uuid']for d in GPU['devices']]==uuids,'actual NVML GPUs differ from admitted UUID order')
    final_states={}
    for pid,identity in initial.items():
        after=proc_identity(pid,proc);need(stable_identity(after)==stable_identity(identity),'registered process changed during collection')
        final_states[str(pid)]={'before':identity['state'],'after':after['state']}
    after_parent=proc_identity(parent['pid'],proc)
    need(stable_identity(after_parent)==stable_identity(parent),'parent changed during collection')
    return {'schema':'ptn-known-live-process-proof/v1','passed':True,'status':'verified_live_8_plus_192','capture_started_UTC':started,
        'capture_finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),
        'run_root':str(root),'contract_sha256':contract_sha,'package_sha256':package_sha,'epoch':epoch,'parent':parent,
        'rank_count':8,'loader_worker_count':192,'unique_rank_and_loader_PID_count':200,'ranks':ranks,'loader_workers':workers,
        'scheduling_states_at_two_reads':final_states,'parent_state_after':after_parent['state'],
        'GPU':GPU,'evidence_file_sha256':evidence,'proc_directory_enumerated':False,'rank_environments_read':8,
        'unrelated_environments_read':0,'loader_environments_read':0,'CUDA_initialized':False,'Torch_imported':False,
        'locks_acquired_or_signals_sent':False,'existing_run_files_written':False,
        'limits':['Short collection interval with two live-stat reads, not a permanent liveness guarantee.',
                  'Evidence of process identities and inherited registration, not acoustic quality or numerical stability.',
                  'FD identity was read; no lock was acquired/reacquired by this probe; NVML compute PID is not equated to container PID.']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','contract-sha','package-sha','expected-hostname','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--epoch',type=int,default=48);a=p.parse_args(argv)
    need(socket.gethostname()==a.expected_hostname,'wrong cloud hostname')
    run=physical(a.run_root);admission,_=read_json(run/'checks/parent_admission.json')
    package=physical(Path(admission['source']['source']).parent);output=physical(a.output)
    for root in (run,package,Path(__file__).resolve().parent):need(output!=root and root not in output.parents,'output overlaps RUN/package/code')
    need(not output.exists()and not output.is_symlink()and output.parent.is_dir(),'output must be fresh in existing external directory')
    try:
        result=probe(run,a.contract_sha,a.package_sha,a.expected_hostname,a.epoch)
    except (ValueError,OSError,KeyError,subprocess.SubprocessError) as error:
        result={'schema':'ptn-known-live-process-proof/v1','passed':False,'status':'indeterminate',
            'capture_finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),
            'run_root':str(run),'contract_sha256':a.contract_sha,'package_sha256':a.package_sha,'epoch':a.epoch,
            'error_type':type(error).__name__,'error':str(error),
            'meaning':'Live identity proof incomplete. An epoch transition, exited worker, absent evidence or identity mismatch is not by itself a training-failure diagnosis. No signal, lock or RUN write was performed.'}
    result['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with output.open('x')as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'output':str(output),'passed':result['passed'],'status':result.get('status','verified_live_8_plus_192')}));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
