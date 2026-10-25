"""Fixed model47 lineage and bounded, read-only identity checks; stdlib only."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat

BASE = '/root/experiments/redimnet2/b6-mix-official-20261019'
ORIGINAL = BASE + '/formal-main-7afb821-zero-20261019-2327'
PREDECESSOR = BASE + '/ptn-resume-model35-20261024-v3'
SOURCE = BASE + '/lm-continuation-20261022/source-7afb821'
CHECKPOINT = PREDECESSOR + '/ptn/models/model_47.pt'
CHECKPOINT_BYTES = 650777251
CK_SHA = '91c1b76ac9d867286a860011a4912b4316d4d987dc30316329d84dc699235e95'
SPK_SHA = '559a92fa71e8f0c1048fd5e916a18476f7503d20e5e70cf8452ea36d013e2548'
LEGACY_BUNDLE_SHA = '840235e7b5963ada274531a29401b9f2bd75fe92e985a8e8e560341ac48350d8'
SOURCE189_SHA = 'b342964138b7ff72b5403add10f8ac4b962f6b5346d501e757176504c89196b0'
SOURCE95_SHA = '5160f99e33caccf373563b6bc695d10f304044b780d10b5c40f6a4b0fab102d9'
PATCH_NAMES = ('applications/speaker_rec/wespeaker_lite/utils/executor.py',
               'applications/speaker_rec/wespeaker_lite/utils/numerical_failure.py')
RUNTIME_NAMES = ('entry.py','worker.py','strict_restore.py','cpu_check.py','epoch_guard.py',
                 'loader_worker_audit.py','predecessor_gate.py','launch.sh','decoder_runtime.py','audio_review.py')
CLOCK = {'saved_epoch':47,'start_epoch':48,'target_epoch':120,'epoch_iter':15001,
         'absolute_iteration':705047,'world_size':8,'batch_size_per_gpu':96,'workers_per_rank':24,
         'lr':0.72,'margin':0.2,'stage':'ptn','optimizer':'restore','scaler':'restore','rng_restored':False}


def need(ok, message):
    if not ok:raise ValueError(message)


def is_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef'for c in value)


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def json_bytes(raw):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'duplicate JSON key');result[key]=value
        return result
    def finite(raw):
        value=float(raw);need(math.isfinite(value),'nonfinite JSON number');return value
    def invalid(raw):raise ValueError('nonfinite JSON constant: '+raw)
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=invalid)


def physical(path):
    p=Path(path);need(p.is_absolute(),'absolute path required')
    current=Path(p.anchor)
    for part in p.parts[1:]:
        need(part not in ('.','..'),'unsafe path');current=current/part
        need(not current.is_symlink(),'symlink evidence/source refused: '+str(current))
    return p


def relative(value):
    need(isinstance(value,str) and value and '\x00'not in value and '\\'not in value,'invalid relative name')
    p=PurePosixPath(value)
    need(p.parts and not p.is_absolute() and '..'not in p.parts and str(p)==value,'unsafe relative name')
    return p


def snapshot(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def hash_file(path, maximum=None):
    p=physical(path);need(p.is_file(),'regular evidence file required')
    h=hashlib.sha256()
    with p.open('rb')as stream:
        before=os.fstat(stream.fileno());need(stat.S_ISREG(before.st_mode),'regular file required')
        if maximum is not None:need(before.st_size<=maximum,'evidence exceeds size bound')
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
        after=os.fstat(stream.fileno())
    need(snapshot(before)==snapshot(after)==snapshot(p.stat()),'file changed while hashing')
    return h.hexdigest(),snapshot(before)


def read_ref(ref, expected_sha=None, maximum=16*1024**2):
    need(isinstance(ref,dict) and set(ref)=={'path','sha256'} and is_sha(ref['sha256']),'exact path/SHA reference required')
    if expected_sha:need(ref['sha256']==expected_sha,'reference does not match fixed historical SHA')
    p=physical(ref['path']);pin,before=hash_file(p,maximum);need(pin==ref['sha256'],'reference bytes differ')
    data=p.read_bytes();need(hashlib.sha256(data).hexdigest()==pin and snapshot(p.stat())==before,'evidence changed while reading')
    return data


def terminal(ref, task, run, allowed, now=None):
    value=json_bytes(read_ref(ref,maximum=65536));need(isinstance(value,dict),'terminal object required')
    need(value.get('schema')=='ptn-predecessor-terminal/v1' and value.get('task_id')==task and value.get('run_root')==run,'wrong predecessor task/run')
    need(value.get('state')in allowed and type(value.get('active_workers'))is int and value['active_workers']==0,'predecessor not terminal')
    workers=value.get('worker_states');need(isinstance(workers,list)and len(workers)==1 and isinstance(workers[0],dict)and
        workers[0].get('name')=='worker_0' and workers[0].get('state')==value['state'],'worker terminal mismatch')
    observed=dt.datetime.fromisoformat(value['observed_utc'].replace('Z','+00:00'));need(observed.tzinfo is not None,'terminal timezone missing')
    now=now or dt.datetime.now(dt.timezone.utc);age=(now-observed).total_seconds();need(-60<=age<=3600,'terminal evidence stale/future')
    return {'reference':ref,'payload':value,'age_seconds':age,'authority':'externally obtained control-plane observation; file validation is not platform authentication'}


def validate_clock(value):
    need(isinstance(value,dict) and canonical(value)==canonical(CLOCK),'only fixed same-stage model47→48 clock is supported')


def read_contract(path, pin):
    value=json_bytes(read_ref({'path':str(path),'sha256':pin}))
    need(isinstance(value,dict) and value.get('schema')=='ptn-resume-model47/v1','wrong model47 contract')
    validate_clock(value.get('clock'))
    need(value.get('runtime_admission')is False,'a build contract cannot grant runtime admission')
    need(value['checkpoint']=={'path':CHECKPOINT,'sha256':CK_SHA,'bytes':CHECKPOINT_BYTES},'fixed model47 identity differs')
    need(value['speaker_map']['sha256']==SPK_SHA,'speaker map pin differs')
    return value


def verify_fresh_terminals(contract, now=None, evidence=None):
    if evidence is not None:
        need(isinstance(evidence,dict)and set(evidence)=={'original','predecessor'},'fresh terminal evidence requires two explicit refs')
        terms={key:{'reference':ref}for key,ref in evidence.items()}
    else:terms=contract['terminals']
    return {'original':terminal(terms['original']['reference'],'t-20261019232824-8zqrl',ORIGINAL,{'Terminated'},now),
            'predecessor':terminal(terms['predecessor']['reference'],'t-20261024154424-vggvm',PREDECESSOR,{'Failed'},now)}
