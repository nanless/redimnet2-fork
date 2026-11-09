#!/usr/bin/env python3
"""唯一model47后继父入口：显式执行、稳定锁、CPU恢复、空闲八卡、零自动重试。"""
import argparse
import contextlib
import csv
import datetime
from decimal import Decimal
import fcntl
import io
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
import strict_restore as S

FATAL = re.compile(r'read_failed|Traceback \(most recent call last\)|could NOT restore|freshly initialised|丢弃了|Non-finite training', re.I)


class PublicationPending(RuntimeError):
    pass


def write_new(path, value):
    path=Path(path);data=S.C.canonical(value)
    tmp=path.with_name('.'+path.name+'.'+secrets.token_hex(12)+'.tmp')
    with tmp.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.link(tmp,path);tmp.unlink()
    fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def owned_json(path, maximum=16*1024**2):
    path=S.C.physical(path)
    try:
        a=path.lstat()
        S.need(stat.S_ISREG(a.st_mode)and a.st_uid==os.getuid()and not a.st_mode&0o022,'unsafe owned publication')
        if a.st_nlink!=1:raise PublicationPending('publication link not settled')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb')as stream:
            b=os.fstat(stream.fileno());raw=stream.read(maximum+1);c=os.fstat(stream.fileno())
        d=path.lstat()
    except FileNotFoundError as error:raise PublicationPending('publication absent')from error
    key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_nlink)
    if key(a)!=key(b)or key(b)!=key(c)or key(c)!=key(d):raise PublicationPending('publication changed')
    S.need(len(raw)==a.st_size and len(raw)<=maximum,'publication exceeds size bound')
    return S.C.json_bytes(raw),hashlib.sha256(raw).hexdigest()


def proc_identity(pid):
    raw=Path('/proc/%d/stat'%pid).read_text();v=raw[raw.rfind(')')+2:].split()
    return {'pid':pid,'ppid':int(v[1]),'pgid':int(v[2]),'sid':int(v[3]),'start_ticks':int(v[19])}


def ancestor(pid, parent_pid):
    seen=set()
    while pid>1 and pid not in seen:
        if pid==parent_pid:return True
        seen.add(pid);pid=proc_identity(pid)['ppid']
    return False


def validate_ready(value, rank, child_pid, nonce, admission):
    expected={'rank':rank,'world_size':8,'nonce':nonce,'contract_sha256':S.CONTRACT_SHA,'package_sha256':S.PACKAGE_SHA}
    S.need(isinstance(value,dict)and all(type(value.get(k))is type(v)and value.get(k)==v for k,v in expected.items()),'worker ready binding differs')
    for name in ('pid','ppid','pgid','sid','start_ticks','group_start_ticks'):
        S.need(type(value.get(name))is int and value[name]>0,'invalid worker process identity')
    actual=proc_identity(value['pid'])
    S.need(all(actual[k]==value[k]for k in actual),'worker ready is not actual process state')
    leader=proc_identity(value['pgid'])
    S.need(leader['start_ticks']==value['group_start_ticks']and leader['sid']==value['sid'],'worker group generation differs')
    S.need(ancestor(value['pid'],child_pid),'worker is not actual torchrun descendant')
    env=Path('/proc/%d/environ'%value['pid']).read_bytes().split(b'\0')
    for key,expected_value in {'PTN_RESUME_NONCE':nonce,'RANK':str(rank),'LOCAL_RANK':str(rank),'WORLD_SIZE':'8',
        'CUDA_VISIBLE_DEVICES':','.join(admission['GPU']['UUIDs'])}.items():
        S.need((key+'='+expected_value).encode()in env,'actual worker environment differs: '+key)
    lock=value.get('gpu_lock');expected_lock=admission['GPU']['lock_evidence'][rank]
    S.need(isinstance(lock,dict)and lock.get('path')==expected_lock['path']and lock.get('uuid')==expected_lock['uuid']and
           type(lock.get('fd'))is int and lock['fd']>=0,'worker GPU lock selection differs')
    target=Path(lock['path']);S.C.physical(target)
    fdpath=Path('/proc/%d/fd/%d'%(value['pid'],lock['fd']));st=fdpath.stat();current=target.stat()
    S.need(fdpath.resolve()==target.resolve()and stat.S_ISREG(st.st_mode)and st.st_nlink==1 and
        (st.st_dev,st.st_ino)==(expected_lock['device'],expected_lock['inode'])==(current.st_dev,current.st_ino),'actual worker GPU FD/inode differs')
    return {'pgid':value['pgid'],'sid':value['sid'],'start_ticks':value['group_start_ticks']}


def worker_ready_before_cuda(rank):
    nonce=os.environ.get('PTN_RESUME_NONCE','');S.need(re.fullmatch('[0-9a-f]{48}',nonce)is not None,'fresh resume nonce missing')
    path=S.RUN/'checks/parent_admission.json';admission,_=owned_json(path)
    S.need(admission['nonce']==nonce and admission['contract_sha256']==S.CONTRACT_SHA and
           admission['package_manifest_sha256']==S.PACKAGE_SHA,'parent admission binding differs')
    parent=admission['parent'];actual=proc_identity(parent['pid'])
    S.need(actual['start_ticks']==parent['start_ticks']and ancestor(os.getpid(),parent['pid']),'actual parent generation differs')
    S.need(os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(admission['GPU']['UUIDs']),'visible GPU UUID mapping differs')
    lock=admission['GPU']['lock_evidence'][rank];target=S.C.physical(lock['path'])
    S.need(target.name==lock['uuid']+'.lock' and target.parent==Path('/root/experiments/redimnet2/.word-gpu-device-locks'),'GPU stable lock path differs')  # storage-policy-allow: 八进程共享的GPU UUID锁目录（协调文件，非大数据产物），换位置会与既有段失去互斥
    parent_fd=Path('/proc/%d/fd/%d'%(parent['pid'],lock['fd']));st=parent_fd.stat();current=target.stat()
    S.need(parent_fd.resolve()==target.resolve()and (st.st_dev,st.st_ino)==(lock['device'],lock['inode'])==(current.st_dev,current.st_ino),'parent GPU FD/inode differs')
    stream=target.open('rb');fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
    st=os.fstat(stream.fileno());S.need((st.st_dev,st.st_ino)==(lock['device'],lock['inode']),'worker GPU inode changed')
    value=proc_identity(os.getpid());leader=proc_identity(value['pgid'])
    value.update(rank=rank,world_size=8,group_start_ticks=leader['start_ticks'],nonce=nonce,
        contract_sha256=S.CONTRACT_SHA,package_sha256=S.PACKAGE_SHA,
        gpu_lock={'path':str(target),'uuid':lock['uuid'],'fd':stream.fileno(),'device':st.st_dev,'inode':st.st_ino})
    ready=S.RUN/'checks'/('worker_ready_rank%d.json'%rank);write_new(ready,value);pin=S.sha(ready)
    deadline=time.monotonic()+180
    while True:
        S.need(time.monotonic()<deadline,'pre-CUDA parent ACK timeout')
        actual=proc_identity(parent['pid']);S.need(actual['start_ticks']==parent['start_ticks'],'parent exited before CUDA ACK')
        try:ack,_=owned_json(S.RUN/'checks/cuda_ack.json')
        except PublicationPending:time.sleep(.1);continue
        S.need(ack['nonce']==nonce and ack['contract_sha256']==S.CONTRACT_SHA and ack['package_sha256']==S.PACKAGE_SHA and
            set(ack['ready_sha256'])=={str(i)for i in range(8)}and ack['ready_sha256'][str(rank)]==pin,'CUDA ACK differs from actual ready')
        return stream


def wait_worker_registration(child, nonce, admission, groups, timeout=180):
    pending=set(range(8));pins={};deadline=time.monotonic()+timeout
    while pending:
        S.need(child.poll()is None,'torchrun exited before all pre-CUDA registrations')
        S.need(time.monotonic()<deadline,'all-rank pre-CUDA registration timeout')
        for rank in sorted(pending):
            try:value,pin=owned_json(S.RUN/'checks'/('worker_ready_rank%d.json'%rank))
            except PublicationPending:continue
            group=validate_ready(value,rank,child.pid,nonce,admission)
            groups[(group['pgid'],group['sid'],group['start_ticks'])]=group;pins[str(rank)]=pin
            pending.remove(rank)
        if pending:time.sleep(.1)
    fresh=gpu_snapshot()
    key=lambda d:(d['index'],d['uuid'],d['name'],d['driver_version'],d['total_mib'])
    S.need([key(d)for d in fresh['devices']]==[key(d)for d in admission['GPU']['snapshots'][-1]['devices']],'pre-CUDA GPU identity drift')
    write_new(S.RUN/'checks/cuda_ack.json',{'nonce':nonce,'contract_sha256':S.CONTRACT_SHA,
        'package_sha256':S.PACKAGE_SHA,'ready_sha256':pins,'fresh_idle_snapshot':fresh})
    return pins


def acquire(stack, path, *, existing):
    p = Path(path)
    S.C.physical(p)
    if existing: S.need(p.is_file(), 'existing stable lock missing: ' + str(p))
    fd = os.open(p, os.O_RDWR | os.O_NOFOLLOW | (0 if existing else os.O_CREAT), 0o600)
    stream = stack.enter_context(os.fdopen(fd, 'r+b'))
    st = os.fstat(fd)
    S.need(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_uid == os.getuid(), 'unsafe stable lock')
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    S.need((st.st_dev, st.st_ino) == (p.stat().st_dev, p.stat().st_ino), 'lock inode changed')
    return stream


def gpu_snapshot():
    commands = [['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits']]
    raw = []
    for cmd in commands:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        S.need(p.returncode == 0 and not p.stderr.strip(), 'GPU query failed')
        raw.append({'command': cmd, 'stdout': p.stdout})
    rows = list(csv.reader(io.StringIO(raw[0]['stdout'])))
    devices = []
    for row in rows:
        S.need(len(row) == 7, 'unknown GPU row'); idx, uuid, name, driver, total, used, free = [x.strip() for x in row]
        S.need(idx.isdigit() and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid) and 'A800' in name
               and re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', driver),
               'unknown/non-A800 GPU identity')
        S.need(all(x.isdigit() for x in (total, used, free)), 'unknown GPU memory')
        S.need(int(total) == 81920 and int(used) <= 256 and 62048 <= int(free) <= int(total)
               and int(used) + int(free) <= int(total) + 8, 'GPU not idle or insufficient60000MiB budget+2048MiB reserve')
        devices.append({'index': int(idx), 'uuid': uuid, 'name': name, 'driver_version': driver, 'total_mib': int(total),
                        'used_mib': int(used), 'free_mib': int(free)})
    S.need([d['index'] for d in devices] == list(range(8)) and len({d['uuid'] for d in devices}) == 8,
           'exactly eight assigned distinct GPUs required')
    S.need(not raw[1]['stdout'].strip(), 'existing/unknown compute PID; leave it untouched')
    return {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'devices': devices, 'raw': raw}


def gpu_admit(stack):
    first = gpu_snapshot()
    lock_root = Path('/root/experiments/redimnet2/.word-gpu-device-locks')  # storage-policy-allow: 同上，GPU UUID锁目录，非大数据产物
    S.C.physical(lock_root); lock_root.mkdir(exist_ok=True)
    locks = [acquire(stack, lock_root / (d['uuid'] + '.lock'), existing=False) for d in sorted(first['devices'], key=lambda d: d['uuid'])]
    start = time.monotonic(); time.sleep(5); second = gpu_snapshot()
    S.need(time.monotonic() - start >= 5, 'GPU independent probe spacing not elapsed')
    keys = lambda snap: [(d['index'], d['uuid'], d['name'], d['driver_version'], d['total_mib']) for d in snap['devices']]
    S.need(keys(first) == keys(second), 'GPU identities changed between snapshots')
    by_uuid={d['uuid']:h for d,h in zip(sorted(first['devices'],key=lambda d:d['uuid']),locks)}
    evidence=[]
    for rank,d in enumerate(second['devices']):
        h=by_uuid[d['uuid']];st=os.fstat(h.fileno())
        evidence.append({'rank':rank,'uuid':d['uuid'],'path':str(lock_root/(d['uuid']+'.lock')),
                         'fd':h.fileno(),'device':st.st_dev,'inode':st.st_ino})
    return {'snapshots':[first,second],'lock_evidence':evidence,'UUIDs':[d['uuid']for d in second['devices']]},locks


def ensure_mounts():
    mounts = {}
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1); fields, extra = left.split(), right.split()
        if fields[4] in ('/root', '/shared/group-shared'):
            S.need(fields[4] not in mounts, 'ambiguous overlapping mount at required target')
            mounts[fields[4]] = {'type': extra[0], 'options': fields[5].split(','), 'super_options': extra[2].split(',')}
    S.need(set(mounts) == {'/root', '/shared/group-shared'}, 'required persistent mounts absent')
    S.need(all(row['type'] == 'gpfs' for row in mounts.values()), 'expected actual vePFS/gpfs mounts, not container overlay')
    S.need('rw' in mounts['/root']['options'] and 'ro' not in mounts['/root']['options']
           and 'ro' in mounts['/shared/group-shared']['options'] and 'rw' not in mounts['/shared/group-shared']['options'],
           'formal root must be writable and group-shared mounted read-only')
    group, alias = Path('/shared/group-shared'), Path('/root/group-shared')
    if alias.exists() or alias.is_symlink():
        S.need(alias.is_dir() and alias.samefile(group), 'existing group-shared alias conflicts')
    else:
        alias.symlink_to(group, target_is_directory=True)
    return {'persistent_root': '/root', 'shared_group': str(group), 'alias': str(alias.resolve()), 'actual_mounts': mounts}


def own_members(pgid, started, nonce):
    members = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            raw = (p / 'stat').read_text(); v = raw[raw.rfind(')') + 2:].split()
            if v[0] == 'Z' or int(v[2]) != pgid: continue
            S.need(int(v[3]) == pgid and int(v[19]) >= started, 'unproven own process session/start')
            S.need(('PTN_RESUME_NONCE=' + nonce).encode() in (p / 'environ').read_bytes().split(b'\0'), 'unproven own process nonce')
            members.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError): continue
    return members


def stop_own(child, started, nonce):
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not own_members(child.pid, started, nonce): return
        os.killpg(child.pid, sig)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            child.poll()
            if not own_members(child.pid, started, nonce): return
            time.sleep(.1)
    S.need(not own_members(child.pid, started, nonce), 'owned workers remain; retain locks')


def group_presence(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return False
    except OSError:return None
    return True


def members_in_group(group, nonce):
    members=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            actual=proc_identity(int(p.name))
            # Read environment only for an already registered process group.
            if actual['pgid']!=group['pgid']:continue
            raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
            if v[0]=='Z':continue
            S.need(actual['sid']==group['sid']and actual['start_ticks']>=group['start_ticks'],'known group generation differs')
            env=(p/'environ').read_bytes().split(b'\0')
            S.need(('PTN_RESUME_NONCE='+nonce).encode()in env,'known group nonce differs')
            members.append(actual['pid'])
        except(FileNotFoundError,ProcessLookupError):continue
    return members


def wait_group_absent(group, child=None, *, presence=group_presence, sleep=time.sleep):
    quiet=0
    while quiet<2:
        if child is not None:child.poll()
        try:value=presence(group['pgid'])
        except Exception:value=None
        quiet=quiet+1 if value is False else 0
        if quiet<2:sleep(.2)


def stop_registered_group(group, nonce, child=None):
    for sig,grace in ((signal.SIGTERM,10),(signal.SIGKILL,5)):
        if not members_in_group(group,nonce):break
        os.killpg(group['pgid'],sig);deadline=time.monotonic()+grace
        while time.monotonic()<deadline:
            if child is not None:child.poll()
            if not members_in_group(group,nonce):break
            time.sleep(.1)
    wait_group_absent(group,child)


def cleanup_registered(child, groups, nonce, *, stop=stop_registered_group, wait=wait_group_absent):
    # Groups are actual pre-CUDA registrations, not a global environment search.
    # Torchrun ranks usually create their own sessions; the launcher PG is insufficient.
    launcher=None
    for group in groups.values():
        if group['pgid']==child.pid:launcher=group
    if launcher is None:
        try:
            actual=proc_identity(child.pid)
            S.need(actual['pgid']==child.pid and actual['sid']==child.pid,'launcher session differs')
            env=Path('/proc/%d/environ'%child.pid).read_bytes().split(b'\0')
            S.need(('PTN_RESUME_NONCE='+nonce).encode()in env,'launcher nonce differs')
            launcher={'pgid':child.pid,'sid':child.pid,'start_ticks':actual['start_ticks']}
        except Exception:
            # Before a complete ACK unregistered workers cannot initialize CUDA.
            # Never infer their identity or signal them; retain locks while the
            # original launcher PG remains present/unknown.
            wait({'pgid':child.pid},child)
    ordered=[g for g in groups.values()if g['pgid']!=child.pid]
    if launcher is not None:ordered.append(launcher)
    for group in ordered:
        try:stop(group,nonce,child)
        except BaseException as error:
            print('CLEANUP_UNKNOWN_GROUP_KEEPING_LOCKS '+repr(error),flush=True)
            # Independent signal-0 observation cannot repeat the failing nonce
            # read and accidentally unwind the lock context.
            wait(group,child)


def wait_gpu_return(expected):
    while True:
        try:
            first=gpu_snapshot();time.sleep(5);second=gpu_snapshot()
            key=lambda row:(row['index'],row['uuid'],row['name'],row['driver_version'],row['total_mib'])
            initial=[key(x)for x in expected['snapshots'][-1]['devices']]
            S.need([key(x)for x in first['devices']]==initial==[key(x)for x in second['devices']],'GPU return identity drift')
            write_new(S.RUN/'checks/resource_return.json',{'snapshots':[first,second],'same_initial_devices':True,'GPU_compute_PID_absent':True})
            return
        except Exception as error:
            print('RESOURCE_RETURN_UNPROVEN_KEEPING_LOCKS '+repr(error),flush=True)
            time.sleep(5)


@contextlib.contextmanager
def cleanup_signal_guard():
    """Neither TERM nor a repeated Ctrl-C may unwind live-worker lock ownership."""
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def environment(uuids, nonce):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('SKIP_', 'PET_', 'TORCHELASTIC_')) or key in {
            'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_AUDIT', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE',
            'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'MODEL_CKPT', 'CHECKPOINT', 'EXP_DIR',
            'RUN_ROOT', 'RECIPE', 'NUM_EPOCHS', 'STAGE', 'SAMPLE_NUM_PER_EPOCH', 'BATCH_SIZE'}:
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PTN_RESUME_NONCE=nonce, PTN_BUNDLE_MANIFEST_SHA=S.PACKAGE_SHA)
    return env


def postflight():
    import math
    import torch
    S.verify_bundle()
    for rank in range(8):
        restore = json.loads((S.RUN / 'checks' / f'restore_verified_rank{rank}.json').read_text())
        update = json.loads((S.RUN / 'checks' / f'first_update_rank{rank}.json').read_text())
        first = json.loads((S.RUN / 'checks' / f'first_iteration_rank{rank}.json').read_text())
        S.need(restore.get('schema') == 'ptn-model47-exact-state-proof/v1' and type(restore.get('rank')) is int and restore['rank'] == rank
               and restore.get('model_exact') is True and restore.get('ordered_optimizer_exact') is True and restore.get('scaler_exact') is True
               and restore.get('contract_sha256') == S.CONTRACT_SHA and restore.get('checkpoint_sha256') == S.CK_SHA
               and restore.get('start_epoch') == 48 and restore.get('absolute_iteration') == 705047, 'rank restoration missing')
        S.need(first == {'rank': rank, 'absolute_iteration': 705047, 'lr': .72, 'contract_sha256': S.CONTRACT_SHA, 'before_first_forward': True},
               'first forward clock/contract differs')
        S.need(type(update.get('rank')) is int and update['rank'] == rank and type(update.get('epoch')) is int and update['epoch'] == 48
               and update.get('optimizer_step_completed') is True and type(update.get('absolute_iteration')) is int
               and 705047 <= update['absolute_iteration'] < 720048 and update.get('contract_sha256') == S.CONTRACT_SHA
               and update.get('checkpoint_sha256') == S.CK_SHA,
               'first actual optimizer update missing')
        for epoch in range(48, 121):
            proof = json.loads((S.RUN / 'checks' / f'epoch_complete_epoch{epoch}_rank{rank}.json').read_text())
            S.need(proof == {'rank': rank, 'epoch': epoch, 'batches_per_rank': 15001,
                             'ranks': [{'rank': j, 'epoch': epoch, 'batches': 15001} for j in range(8)],
                             'contract_sha256': S.CONTRACT_SHA, 'before_validation_and_save': True},
                   'missing actual full epoch proof')
    with (S.RUN / 'ptn/val_log.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    epochs = [Decimal(r['epoch']) for r in rows]
    S.need(all(v.is_finite() and v == v.to_integral_value() for v in epochs), 'validation epoch must be exact integer')
    S.need([int(v) for v in epochs] == list(range(48, 121)), 'new validation lineage must be48..120')
    S.need(all(math.isfinite(float(r['vox1-eer-8s'])) and 0 <= float(r['vox1-eer-8s']) <= 100 for r in rows), 'validation nonfinite')
    logs = list((S.RUN / 'rank_logs').rglob('stdout.log')) + list((S.RUN / 'rank_logs').rglob('stderr.log'))
    S.need(len(logs) == 16, 'exactly16 rank logs required')
    for p in logs: S.need(not FATAL.search(p.read_text(errors='replace')), 'unreviewed new failure: ' + str(p))
    final = S.RUN / 'ptn/models/model_120.pt'; before = S.snapshot(final)
    payload = torch.load(final, map_location='cpu', weights_only=True, mmap=True)
    S.need(set(payload) == {'epoch', 'model', 'optimizer', 'scaler'} and type(payload.get('epoch')) is int and payload['epoch'] == 120
           and len(payload['model']) == 823 and len(payload['optimizer']['state']) == 584 and payload.get('scaler'), 'final checkpoint incomplete')
    tensors = list(payload['model'].values()) + [v for row in payload['optimizer']['state'].values() for v in row.values()]
    S.need(all(not torch.is_tensor(v) or not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in tensors), 'final state nonfinite')
    S.need(S.snapshot(final) == before, 'final checkpoint changed')
    return {'ptn_complete': True, 'epochs': list(range(48, 121)), 'checkpoint_sha256': S.sha(final),
            'checkpoint': str(final), 'LM_started': False, 'three_segment_LM_contract_required': True,
            'historical_numeric_root_cause_proven': False}


def execute():
    S.need(int(os.environ.get('WORLD_SIZE', '1')) == 1 and int(os.environ.get('RANK', '0')) == 0 and 'LOCAL_RANK' not in os.environ,
           'platform must start one parent, not eight parents')
    S.need(not S.RUN.exists(), 'new run already claimed; never repeat this entry')
    from predecessor_gate import wait_for_terminal, verify_unchanged
    terminal = wait_for_terminal(); mounts = ensure_mounts()
    with contextlib.ExitStack() as stack:
        admission = acquire(stack, S.CONFIG['admission_lock'], existing=True)
        logical = acquire(stack, S.CONFIG['logical_chain_lock'], existing=True)
        old = [acquire(stack, p, existing=True) for p in S.CONFIG['required_existing_locks']]
        verify_unchanged(terminal); S.verify_bundle()
        previous = os.environ.get('CUDA_VISIBLE_DEVICES'); os.environ['CUDA_VISIBLE_DEVICES'] = ''
        try:
            from cpu_check import run
            cpu = run()
        finally:
            if previous is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else: os.environ['CUDA_VISIBLE_DEVICES'] = previous
        gpu, gpu_handles = gpu_admit(stack)
        S.RUN.mkdir(); (S.RUN / 'ptn').mkdir(); (S.RUN / 'checks').mkdir()
        new = [acquire(stack, S.RUN / p, existing=False) for p in ('.reproduction.lock', 'ptn/.cloud_train.lock')]
        nonce = secrets.token_hex(24)
        admission_record={'terminal':terminal,'mounts':mounts,'CPU':cpu,'GPU':gpu,'package_manifest_sha256':S.PACKAGE_SHA,
            'contract_sha256':S.CONTRACT_SHA,'source':S.verify_source(),'nonce':nonce,'parent':proc_identity(os.getpid())}
        write_new(S.RUN/'checks/parent_admission.json',admission_record)
        cmd = [S.PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=8',
               '--max-restarts=0', '--log-dir', str(S.RUN / 'rank_logs'), '--redirects=3', '--tee=3',
               str(S.BUNDLE / 'worker.py'), '--contract', str(S.CONTRACT_PATH), '--contract-sha', S.CONTRACT_SHA,
               '--package-sha', S.PACKAGE_SHA]
        for h in [logical,*old,*gpu_handles]:fcntl.flock(h,fcntl.LOCK_SH|fcntl.LOCK_NB)
        fds=[h.fileno()for h in [admission,logical,*old,*new,*gpu_handles]]
        groups={}
        with (S.RUN / 'launch.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=S.SOURCE, env=environment(gpu['UUIDs'], nonce), pass_fds=fds,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            started = int(time.monotonic() * os.sysconf('SC_CLK_TCK')) - 1
            def interrupted(signum, frame): raise KeyboardInterrupt('parent signal ' + str(signum))
            old_handler = signal.signal(signal.SIGTERM, interrupted)
            try:
                raw = Path(f'/proc/{child.pid}/stat').read_text(); v = raw[raw.rfind(')') + 2:].split()
                S.need(int(v[2]) == child.pid and int(v[3]) == child.pid, 'new torchrun session ownership differs')
                started = int(v[19])
                group={'pgid':child.pid,'sid':child.pid,'start_ticks':started};groups[(child.pid,child.pid,started)]=group
                wait_worker_registration(child,nonce,admission_record,groups)
                for line in child.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                    S.need(not FATAL.search(line), 'new controlled PTN failure requires review; abort only owned group')
                S.need(child.wait() == 0, 'own torchrun failed')
                S.need(not own_members(child.pid, started, nonce), 'own worker residuals remain')
            finally:
                try:
                    with cleanup_signal_guard():
                        cleanup_registered(child,groups,nonce)
                        wait_gpu_return(gpu)
                finally:signal.signal(signal.SIGTERM,old_handler)
        result = postflight(); write_new(S.RUN / 'ptn_success.json', result)


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(); p.add_argument('--contract', required=True); p.add_argument('--contract-sha', required=True)
    p.add_argument('--package-sha', required=True); p.add_argument('--preflight-report')
    mode = p.add_mutually_exclusive_group(required=True); mode.add_argument('--execute', action='store_true'); mode.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(); S.configure(a.contract, a.contract_sha, a.package_sha)
    output = None
    if a.preflight_report:
        S.need(a.preflight_only, 'preflight report option requires preflight-only')
        output = S.C.physical(a.preflight_report)
        protected = [S.PACKAGE, S.RUN, S.OLD, S.PREDECESSOR, Path(S.C.SOURCE), Path(S.CONFIG['legacy']['bundle_root'])]
        S.need(all(output != root and root not in output.parents for root in protected), 'CPU report overlaps protected package/run/source')
        S.need(not output.exists() and output.parent.is_dir(), 'CPU report must be new in existing audit directory')
    if a.execute: execute()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from cpu_check import run
        result = run()
        if output is not None: write_new(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try: main()
    except SystemExit: raise
    except BaseException:
        traceback.print_exc(); sys.exit(2)
