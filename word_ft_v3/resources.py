"""Conservative read-only W2/W8 admission. No torch/CUDA import here.

A flock coordinates cooperating launchers, never authorizes touching old PIDs.
The parent holds all assigned device locks throughout its own torchrun process group.
"""
from __future__ import annotations
from contextlib import ExitStack
import csv,fcntl,hashlib,io,json,math,os,re,socket,stat,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

TICKET_SCHEMA='word-gpu-admission/v3'
from .topology import topology,validate as validate_topology
UUID_RE=re.compile(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z')
GPU_QUERY=['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free','--format=csv,noheader,nounits']
PROCESS_QUERY=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']

class AdmissionRefused(RuntimeError):pass

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def positive_int(text,name):
    if not re.fullmatch(r'[0-9]+',str(text)) or int(text)<=0:raise AdmissionRefused('invalid '+name)
    return int(text)
def _rows(raw,count):
    rows=[]
    for row in csv.reader(io.StringIO(raw)):
        if not row or all(not value.strip() for value in row):continue
        row=[value.strip() for value in row]
        if len(row)!=count:raise AdmissionRefused('unknown nvidia-smi CSV row')
        rows.append(row)
    return rows

def parse_snapshot(gpu_stdout,process_stdout):
    devices=[]
    for r in _rows(gpu_stdout,7):
        if not re.fullmatch(r'[0-9]+',r[0]) or not UUID_RE.fullmatch(r[1]) or not r[2] or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+',r[3]):
            raise AdmissionRefused('unknown GPU identity/driver')
        total,used,free=[positive_int(r[4],'total memory'),int(r[5]) if re.fullmatch(r'[0-9]+',r[5]) else -1,int(r[6]) if re.fullmatch(r'[0-9]+',r[6]) else -1]
        if used<0 or free<0 or used>total or free>total or used+free>total+8:
            raise AdmissionRefused('unknown/inconsistent GPU memory')
        devices.append({'index':int(r[0]),'uuid':r[1],'name':r[2],'driver_version':r[3],'total_mib':total,'used_mib':used,'free_mib':free})
    if not devices or len({x['uuid'] for x in devices})!=len(devices) or len({x['index'] for x in devices})!=len(devices):
        raise AdmissionRefused('empty/duplicate GPU inventory')
    processes=[];known={x['uuid'] for x in devices}
    for r in _rows(process_stdout,4):
        if r[0] not in known or not r[2]:raise AdmissionRefused('unknown compute process/GPU mapping')
        pid=positive_int(r[1],'compute PID')
        memory=int(r[3]) if re.fullmatch(r'[0-9]+',r[3]) else None
        # A real PID is always busy even if its memory counter is unavailable.
        processes.append({'gpu_uuid':r[0],'pid':pid,'process_name':r[2],'used_mib':memory})
    return {'devices':devices,'compute_processes':processes}

def capture_snapshot(runner=subprocess.run):
    raw=[]
    for cmd in [GPU_QUERY,PROCESS_QUERY]:
        try:r=runner(cmd,capture_output=True,text=True,timeout=8,check=False)
        except Exception as exc:raise AdmissionRefused('nvidia-smi query failed: '+str(exc)) from exc
        raw.append({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        if r.returncode or r.stderr.strip():raise AdmissionRefused('nvidia-smi query not clean: '+canonical(raw[-1]))
    parsed=parse_snapshot(raw[0]['stdout'],raw[1]['stdout'])
    return dict(parsed,utc=utc(),monotonic=time.monotonic(),raw=raw)

def selected_devices(snapshot,indices):
    if len(indices) not in (2,8) or len(set(indices))!=len(indices) or any(type(i) is not int or i<0 for i in indices):
        raise AdmissionRefused('exactly two or eight distinct GPU indices required')
    found={d['index']:d for d in snapshot['devices']}
    if any(i not in found for i in indices):raise AdmissionRefused('requested GPU not found')
    return [dict(found[i]) for i in indices]

def check_idle(snapshot,indices,budget_mib,driver_used_limit_mib=256,reserve_mib=2048):
    if type(budget_mib) is not int or budget_mib<512:raise AdmissionRefused('explicit memory budget >=512 MiB required')
    if type(driver_used_limit_mib) is not int or not 0<=driver_used_limit_mib<=512:
        raise AdmissionRefused('driver-used limit must be conservative <=512 MiB')
    if type(reserve_mib) is not int or reserve_mib<2048:raise AdmissionRefused('context/NCCL reserve must be >=2048 MiB')
    selected=selected_devices(snapshot,indices);uuids={d['uuid'] for d in selected}
    if any(p['gpu_uuid'] in uuids for p in snapshot['compute_processes']):
        raise AdmissionRefused('existing compute PID on requested GPU; leave it untouched')
    for d in selected:
        if d['used_mib']>driver_used_limit_mib:raise AdmissionRefused('GPU driver-used memory exceeds idle limit')
        if d['free_mib']<budget_mib+reserve_mib or budget_mib+reserve_mib>=d['total_mib']:
            raise AdmissionRefused('insufficient free memory for budget plus context/NCCL reserve')
    return selected

def stable_identity(devices):
    return [{k:d[k] for k in ['index','uuid','name','driver_version','total_mib']} for d in devices]

def double_probe(indices,budget_mib,*,interval_seconds=5.,driver_used_limit_mib=256,reserve_mib=2048,
                 capture=capture_snapshot,sleep=time.sleep,clock=time.monotonic):
    if not math.isfinite(interval_seconds) or not 5<=interval_seconds<=30:raise AdmissionRefused('probe spacing must be 5–30 seconds')
    def fresh_capture():
        before=clock();snap=capture();after=clock();stamp=snap.get('monotonic')
        if type(stamp) not in (int,float) or not math.isfinite(stamp) or not math.isfinite(before) or not math.isfinite(after) or not before-.02<=stamp<=after+.02:
            raise AdmissionRefused('capture is stale/future/nonfinite')
        return snap
    snapshots=[fresh_capture()];start=clock();sleep(interval_seconds);elapsed=clock()-start
    if elapsed<interval_seconds-.02:raise AdmissionRefused('probe spacing not actually elapsed')
    snapshots.append(fresh_capture());errors=[];identities=[]
    for snap in snapshots:
        try:identities.append(check_idle(snap,indices,budget_mib,driver_used_limit_mib,reserve_mib))
        except AdmissionRefused as exc:errors.append(str(exc))
    if errors:raise AdmissionRefused(canonical({'reasons':errors,'snapshots':snapshots}))
    if stable_identity(identities[0])!=stable_identity(identities[1]):raise AdmissionRefused('GPU identity/driver/total changed between probes')
    if snapshots[1]['monotonic']-snapshots[0]['monotonic']<interval_seconds-.02:
        raise AdmissionRefused('snapshot timestamps not independently fresh')
    return {'snapshots':snapshots,'devices':identities[1],'interval_seconds':elapsed}

def process_start_time(pid):
    # comm may contain spaces/parentheses: split after its last right bracket.
    data=Path(f'/proc/{pid}/stat').read_text();parts=data[data.rfind(')')+2:].split()
    return int(parts[19])

def protected_path(path,repo=None):
    root=Path(path).resolve()
    forbidden=[Path('/root/code/github_repos/redimnet2-fork'),Path('/root/experiments/redimnet2/b6-mix-official-20261019'),
               Path('/root/group-shared/voiceprint/data'),Path('/shared/group-shared/voiceprint/data'),
               Path('/root/group-shared/voiceprint/models'),Path('/shared/group-shared/voiceprint/models')]
    if repo:forbidden.append(Path(repo).resolve())
    for f in forbidden:
        if root==f or f in root.parents or root in f.parents:raise AdmissionRefused('protected output/lock path')
    return root

class DeviceLocks:
    def __init__(self,root,uuids):
        self.root=protected_path(root);self.uuids=sorted(uuids);self.stack=ExitStack();self.evidence=[]
        if len(self.uuids) not in (2,8) or len(set(self.uuids))!=len(self.uuids) or any(not UUID_RE.fullmatch(u) for u in self.uuids):
            raise AdmissionRefused('W2/W8 valid UUID locks required')
    def __enter__(self):
        self.root.mkdir(parents=True,exist_ok=True)
        try:
            for uuid in self.uuids:
                p=self.root/(uuid+'.lock')
                fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
                stream=self.stack.enter_context(os.fdopen(fd,'a+'))
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                st=os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_uid!=os.getuid():
                    raise AdmissionRefused('lock must be a singly linked owned regular file')
                self.evidence.append({'uuid':uuid,'path':str(p.resolve()),'fd':fd,'device':st.st_dev,'inode':st.st_ino})
        except Exception:
            self.stack.close();raise
        return self
    def __exit__(self,*args):return self.stack.__exit__(*args)

def create_ticket(probe,locks,budget_mib,reserve_mib,driver_used_limit_mib,source_identity):
    return {'schema':TICKET_SCHEMA,'topology':topology(len(probe['devices'])),'parent_pid':os.getpid(),'parent_start_time':process_start_time(os.getpid()),
        'hostname':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_utc':utc(),'created_monotonic':time.monotonic(),'devices':probe['devices'],'probe':probe,
        'memory_budget_mib':budget_mib,'reserve_mib':reserve_mib,'driver_used_limit_mib':driver_used_limit_mib,
        'locks':locks.evidence,'source_identity':source_identity,'execute':True}

def validate_ticket(ticket_path,expected_sha,source_identity,*,max_age=60.,allow_running=False):
    path=Path(ticket_path)
    if not re.fullmatch('[0-9a-f]{64}',expected_sha):raise AdmissionRefused('invalid ticket SHA')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read();after=os.fstat(stream.fileno())
    observed=path.stat()
    attrs=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if attrs(before)!=attrs(after) or attrs(after)!=attrs(observed) or not stat.S_ISREG(after.st_mode) or hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise AdmissionRefused('admission ticket file/SHA differs')
    ticket=json.loads(raw)
    if ticket.get('schema')!=TICKET_SCHEMA or ticket.get('execute') is not True or ticket.get('source_identity')!=source_identity:
        raise AdmissionRefused('admission schema/execution/source differs')
    if ticket.get('hostname')!=socket.gethostname() or ticket.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise AdmissionRefused('admission host/boot differs')
    age=time.monotonic()-ticket['created_monotonic']
    if not math.isfinite(age) or age<0 or (not allow_running and age>max_age):raise AdmissionRefused('admission ticket is stale/future')
    verify_ticket_probe(ticket)
    world=validate_topology(ticket['topology'])
    pid=ticket['parent_pid']
    if type(pid) is not int or pid<=1 or process_start_time(pid)!=ticket['parent_start_time']:
        raise AdmissionRefused('admission parent no longer matches')
    if not is_descendant(os.getpid(),pid):raise AdmissionRefused('admission parent is not this worker ancestor')
    if len(ticket.get('locks',[]))!=world:raise AdmissionRefused('missing device lock proof')
    uuids={d['uuid'] for d in ticket['devices']}
    if {lock['uuid'] for lock in ticket['locks']}!=uuids:raise AdmissionRefused('admission lock UUIDs differ')
    for lock in ticket['locks']:
        fd_path=Path(f'/proc/{pid}/fd/{lock["fd"]}')
        st=fd_path.stat();local=Path(lock['path'])
        if local.is_symlink() or str(fd_path.resolve())!=lock['path'] or (st.st_dev,st.st_ino)!=(lock['device'],lock['inode']):
            raise AdmissionRefused('admission parent lock FD changed')
        # The parent's still-open FD must actually hold an exclusive lock.
        with local.open('r+') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                fcntl.flock(stream,fcntl.LOCK_UN);raise AdmissionRefused('device lock is not held')
    return ticket

def allowed_process_group(snapshot,devices,process_group):
    uuids={d['uuid'] for d in devices}
    for process in snapshot['compute_processes']:
        if process['gpu_uuid'] not in uuids:continue
        try:group=os.getpgid(process['pid'])
        except Exception as exc:raise AdmissionRefused('cannot identify compute PID ownership; fail closed') from exc
        if group!=process_group:raise AdmissionRefused('external compute PID appeared; stop only this experiment')
    selected=[d for d in snapshot['devices'] if d['uuid'] in uuids]
    if sorted(stable_identity(selected),key=lambda x:x['uuid'])!=sorted(stable_identity(devices),key=lambda x:x['uuid']):
        raise AdmissionRefused('GPU identity/driver changed during own job')


def training_source_identity(repo):
    root=Path(repo).resolve()
    required=['word_ft_v3/__init__.py','word_ft_v3/train.py','word_ft_v3/resources.py','word_ft_v3/topology.py','scripts/launch_word_v3.py',
              'word_ft/__init__.py','word_ft/schema.py','duration_ft/__init__.py','duration_ft/config_checks.py','duration_ft/train.py','duration_ft/data.py',
              'applications/speaker_rec/wespeaker_lite/models/projections.py']
    backbone_files=sorted((root/'redimnet2').rglob('*.py'))
    if not backbone_files:raise AdmissionRefused('backbone source tree missing')
    package_inits=[root/'applications/speaker_rec/wespeaker_lite/__init__.py',root/'applications/speaker_rec/wespeaker_lite/models/__init__.py']
    files=[root/name for name in required]+backbone_files+[p for p in package_inits if p.is_file()]
    if any(not p.is_file() for p in files):raise AdmissionRefused('GPU training source file missing')
    return {str(p.relative_to(root)):sha_file(p) for p in files}


def is_descendant(pid,ancestor):
    seen=set()
    for _ in range(128):
        if pid==ancestor:return True
        if pid<=1 or pid in seen:return False
        seen.add(pid)
        try:
            raw=Path(f'/proc/{pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split();pid=int(parts[1])
        except Exception:return False
    return False


def verify_ticket_probe(ticket):
    try:
        probe=ticket['probe'];snapshots=probe['snapshots'];devices=ticket['devices']
        world=validate_topology(ticket['topology'])
        if len(snapshots)!=2 or len(devices)!=world:raise AdmissionRefused('ticket requires two temporal probes and topology-sized devices')
        indices=[d['index'] for d in devices]
        checked=[check_idle(s,indices,ticket['memory_budget_mib'],ticket['driver_used_limit_mib'],ticket['reserve_mib']) for s in snapshots]
        stamps=[s['monotonic'] for s in snapshots];created=ticket['created_monotonic']
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in stamps+[created]):
            raise AdmissionRefused('ticket probe timestamps nonfinite')
        spacing=stamps[1]-stamps[0]
        if not 5-.02<=probe['interval_seconds']<=30.5 or not 5-.02<=spacing<=60 or not 0<=created-stamps[1]<=15:
            raise AdmissionRefused('ticket probe spacing/freshness differs')
        identities=[stable_identity(d) for d in checked]
        if identities[0]!=identities[1] or identities[1]!=stable_identity(devices):
            raise AdmissionRefused('ticket probe UUID/driver identity differs')
    except (KeyError,TypeError,ValueError) as exc:
        raise AdmissionRefused('malformed admission probe') from exc
