#!/usr/bin/env python3
"""核验真实新观察并刷新未启动run的control引用；默认只审计，不改规范引用。"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import traceback
try:
    from . import contracts as C
except ImportError:
    import contracts as C

ROLES={'original':('t-20261019232824-8zqrl',C.ORIGINAL,{'Terminated'}),
       'predecessor':('t-20261024154424-vggvm',C.PREDECESSOR,{'Failed'})}


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def new_bytes(path,raw):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())


def current(path):
    if not os.path.lexists(path):return {'exists':False}
    C.physical(path);s=path.lstat()
    C.need(stat.S_ISREG(s.st_mode)and s.st_uid==os.getuid()and s.st_nlink==1 and not s.st_mode&0o022,'unsafe canonical control file')
    pin,before=C.hash_file(path,65536);raw=path.read_bytes()
    C.need(C.snapshot(path.stat())==before and hashlib.sha256(raw).hexdigest()==pin,'control changed while reading')
    return {'exists':True,'sha256':pin,'stat':before,'bytes':len(raw),'raw':raw}


def summary(value):return {k:v for k,v in value.items()if k!='raw'}


def overlap(a,b):return a==b or a in b.parents or b in a.parents


def validate_paths(run,control,audit,refs):
    run,control,audit=map(C.physical,(run,control,audit));base=C.physical(C.BASE)
    C.need(run.parent==base and run.name.startswith('ptn-resume-model47-capture-')and
           not run.name.startswith('ptn-resume-model47-capture-package-')and not run.name.endswith('-control'),'unexpected new-run namespace')
    C.need(not os.path.lexists(run),'run already exists or claimed; never refresh a live run')
    C.need(control==run.with_name(run.name+'-control'),'control not bound to exact new RUN-control')
    s=control.lstat();C.need(stat.S_ISDIR(s.st_mode)and s.st_uid==os.getuid()and not s.st_mode&0o022,'unsafe/foreign control directory')
    C.need(not os.path.lexists(audit)and audit.parent.is_dir(),'audit must be a new external directory')
    blocked=[run,control,C.physical(C.ORIGINAL),C.physical(C.PREDECESSOR),C.physical(C.SOURCE),Path(__file__).resolve().parent]
    blocked+=list(base.glob('ptn-resume-model47-capture-package-*'))+list(base.glob('ptn-resume-code-*'))+[base/'lm-continuation-20261022']
    C.need(all(not overlap(audit,p)for p in blocked),'audit overlaps control/run/frozen/source/code')
    for ref in refs.values():
        p=C.physical(ref['path']);C.need(C.is_sha(ref['sha256']),'explicit64 SHA required')
        C.need(not overlap(p,audit)and not overlap(p,control)and not overlap(p,run),'input/output overlap')
    return run,control,audit


def replace_one(target,raw,*,before_publish):
    temporary=target.with_name('.'+target.name+'.'+secrets.token_hex(12)+'.tmp')
    new_bytes(temporary,raw);before_publish();os.replace(temporary,target);sync_dir(target.parent)


def refresh(*,run,control,audit,refs,execute=False):
    run,control,audit=validate_paths(run,control,audit,refs)
    audit.mkdir(mode=0o700);before={};applied=[];error=None;proofs={};inputs={}
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        for role in ROLES:
            path=control/(role+'-terminal.json');value=current(path);before[role]=value
            if value['exists']:new_bytes(audit/(role+'-before.json'),value['raw'])
        new_bytes(audit/'before-index.json',C.canonical({k:summary(v)for k,v in before.items()}));sync_dir(audit)
        for role,(task,oldrun,allowed) in ROLES.items():
            proofs[role]=C.terminal(refs[role],task,oldrun,allowed)
            raw=C.read_ref(refs[role],maximum=65536);inputs[role]=raw
            new_bytes(audit/(role+'-input.json'),raw)
        new_bytes(audit/'input-index.json',C.canonical({'refs':refs,'validation':proofs}));sync_dir(audit)
        if execute:
            expected=dict(before)
            for role,(task,oldrun,allowed) in ROLES.items():
                # No training lock is taken. Every mutation rechecks the run and
                # both canonical refs against the archived/just-written generation.
                def still_unclaimed():
                    C.need(not os.path.lexists(run),'run appeared before replacement; stop')
                    for key in ROLES:C.need(summary(current(control/(key+'-terminal.json')))==summary(expected[key]),'control generation changed before replace')
                still_unclaimed()
                C.terminal(refs[role],task,oldrun,allowed)  # Fresh TTL at this mutation, never backdate observed_utc.
                C.need(hashlib.sha256(inputs[role]).hexdigest()==refs[role]['sha256'],'input archive binding differs')
                target=control/(role+'-terminal.json');replace_one(target,inputs[role],before_publish=still_unclaimed);applied.append(role)
                value=current(target);C.need(value['raw']==inputs[role],'published bytes differ from original input')
                expected[role]=value
                new_bytes(audit/(role+'-after.json'),value['raw']);sync_dir(audit)
    except Exception:error=traceback.format_exc()
    after={}
    for role in ROLES:
        try:after[role]=summary(current(control/(role+'-terminal.json')))
        except Exception:after[role]={'read_error':traceback.format_exc()}
    changed=[role for role in before if summary(before[role])!=after.get(role)]
    report={'schema':'ptn-terminal-reference-refresh/v1','started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'run':str(run),'control':str(control),'execute':execute,'dry_run':not execute,'passed':error is None,'error':error,
        'input_refs':refs,'validated_inputs':proofs,'before':{k:summary(v)for k,v in before.items()},'after':after,
        'replaced_roles':applied,'actual_changed_roles':changed,
        'partial_update_possible':execute and error is not None and bool(applied or changed),
        'atomic_file_publication':True,'atomic_two_file_transaction':False,'observed_utc_modified':False,
        'training_locks_taken':False,'runtime_admission':False,'task_submitted':False,
        'authority':'validates supplied control-plane observation bytes; does not authenticate platform or invent a fresh observation'}
    new_bytes(audit/'result.json',C.canonical(report));sync_dir(audit)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run-root','control-dir','audit-dir','original','original-sha','predecessor','predecessor-sha'):p.add_argument('--'+n,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    refs={role:{'path':getattr(a,role),'sha256':getattr(a,role+'_sha')}for role in ROLES}
    result=refresh(run=a.run_root,control=a.control_dir,audit=a.audit_dir,refs=refs,execute=a.execute)
    print(json.dumps({'passed':result['passed'],'execute':a.execute,'audit':a.audit_dir,'runtime_admission':False}))
    return 0 if result['passed']else 2


if __name__=='__main__':raise SystemExit(main())
