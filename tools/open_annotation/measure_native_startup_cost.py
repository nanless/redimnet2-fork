"""测量固定正式代码的有限证明启动成本；默认只检查，不产生运行资格。

执行时一次完整 A 检查、同 scope 八次复用，再测 M 真实底层全 SHA 路径。
不构造 RuntimeLease、memfd、锁或 claim；不加载模型，不读取原音或全库。
唯一子进程受九百秒以内预算监督，阶段进度、异常和实际 RSS 均保留。
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback

SCHEMA='annotation-native-startup-cost-plan/v1'
REPEATS=8
DENIED_SUFFIXES={'.wav','.flac','.mp3','.m4a','.aac','.ogg','.opus','.mp4','.webm','.mkv',
    '.safetensors','.pt','.pth','.ckpt','.onnx','.bin','.sqlite','.sqlite3','.db','.mdb','.lmdb'}
CHILD_CODE="""import sys,hashlib
b=open(sys.argv[1],'rb').read()
if hashlib.sha256(b).hexdigest()!=sys.argv[4]:raise RuntimeError('cost controller bytes changed')
n={'__file__':sys.argv[1],'__name__':'_actual_cost_child'}
exec(compile(b,sys.argv[1],'exec'),n)
n['_child'](sys.argv[2],sys.argv[3])
"""


def need(value,message):
    if not value:raise ValueError(message)


def physical(value):
    p=Path(value);need(p.is_absolute()and '..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents)),'physical absolute path required');return p


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def read_pin(ref,maximum=2*1024**2):
    need(isinstance(ref,dict)and set(ref)=={'path','sha256'}and isinstance(ref['sha256'],str)and len(ref['sha256'])==64
         and all(c in '0123456789abcdef'for c in ref['sha256']),'exact path/SHA pin required')
    p=physical(ref['path']);need(p.suffix.lower()not in DENIED_SUFFIXES,'media/model/database cannot be a control JSON')
    s=p.lstat();need(stat.S_ISREG(s.st_mode)and s.st_size<=maximum,'bounded regular JSON required')
    raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'input SHA differs: '+str(p));return json.loads(raw)


def write_new(path,value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())


def check(plan_pin,*,created=False):
    p=read_pin(plan_pin);need(set(p)=={'schema','selected','formal_contract','formal_bundle','code_root','output_root','max_seconds'}and p['schema']==SCHEMA,'explicit cost plan fields required')
    need(type(p['max_seconds'])is int and 0<p['max_seconds']<=900,'CPU controller budget must be 1..900 seconds')
    code=physical(p['code_root']);out=physical(p['output_root']);need(code.is_dir()and out.parent.is_dir()and (out.is_dir()if created else not out.exists()),'fresh output root required')
    need(code!=out and code not in out.parents and out not in code.parents,'output overlaps frozen code')
    need(out.parent==code.parent and out.name.startswith('cpu-startup-cost-'),'dedicated independent cpu-startup-cost-* sibling output required')
    for ref in (plan_pin,p['selected'],p['formal_contract'],p['formal_bundle']):
        q=physical(ref['path']);need(out!=q and out not in q.parents,'output overlaps fixed input')
    selected=read_pin(p['selected']);contract=read_pin(p['formal_contract']);bundle=read_pin(p['formal_bundle'],8*1024**2)
    need(selected.get('schema')=='annotation-native-selected-pilot-admission/v2'and selected.get('purpose')=='production'
         and selected.get('formal_execution_bundle')==p['formal_bundle'],'actual production selected admission/formal bundle required')
    need(contract.get('schema')=='redimnet2-production-contract/v2'and contract.get('purpose')=='silver_annotation'
         and contract.get('native_formal',{}).get('default_enabled')is False,'explicit disabled native formal contract required')
    need(p['formal_bundle']['path']==str(code/'bundle_files.json')and isinstance(bundle,dict)and 0<len(bundle)<=512
         and {'native_batch_admission.py','bounded_speed_runtime.py','gpu_launch.py'}<=set(bundle),'actual finite frozen bundle required')
    for name,digest in bundle.items():
        rel=Path(name);need(rel.parts and not rel.is_absolute()and '..'not in rel.parts and rel.suffix.lower()not in DENIED_SUFFIXES,'unsafe/media/model/database bundle member')
        q=physical(code/rel);s=q.lstat();need(stat.S_ISREG(s.st_mode)and s.st_size<=16*1024**2,'bounded regular frozen source required')
        need(sha_file(q)==digest,'frozen source changed: '+name)
    allowed={};counts={}
    for role in ('baseline','main','recovery'):
        track=read_pin(selected[role+'_track']);need(track.get('schema')=='redimnet2-annotation-native-batch-track/v1','fixed native track required')
        for key in ('queue_root','output_root','resources_root'):
            q=physical(track[key]);need(out!=q and out not in q.parents and q not in out.parents,'measurement output overlaps existing producer tree')
        index=read_pin(track['index'],8*1024**2);n=index.get('source_count');shards=index.get('shards')
        need(type(n)is int and 0<n<=4096 and isinstance(shards,list)and 0<len(shards)<=n
             and all(type(s.get('count'))is int and s['count']>0 for s in shards)and sum(s['count']for s in shards)==n,'finite source denominator must be complete and <=4096')
        counts[role]=n
        for s in shards:
            rel=Path(s['path']);need(rel.parts and not rel.is_absolute()and '..'not in rel.parts,'unsafe finite shard path')
            need(isinstance(s.get('sha256'),str)and len(s['sha256'])==64 and all(x in '0123456789abcdef'for x in s['sha256']),'explicit finite shard SHA required')
            q=physical(Path(track['index']['path']).parent/rel);need(str(q)not in allowed or allowed[str(q)]==s['sha256'],'conflicting finite shard SHA')
            allowed[str(q)]=s['sha256']
    need(counts['baseline']==counts['main']and counts['recovery']<=counts['main'],'finite track denominators differ')
    return dict(plan=p,contract=contract,bundle=bundle,allowed_jsonl=allowed,source_counts=counts)


def _guard(output,code,allowed_jsonl):
    """sqlite3 可正常导入；只禁止连接、真实媒体/权重、非有限JSONL及副作用。"""
    result_path=str(output/'child-result.json')
    def audit(event,args):
        if event in {'sqlite3.connect','fcntl.flock','subprocess.Popen','os.system','os.fork','os.forkpty','socket.connect','ctypes.dlopen',
                     'os.mkdir','os.remove','os.rename','os.symlink','os.link','os.rmdir','os.chmod','os.chown','os.truncate','os.utime'}or event.startswith('os.exec'):
            raise PermissionError('cost measurement forbids side effect: '+event)
        if event=='import'and args[0].split('.')[0]in {'torch','qwen_asr','vllm','tensorflow','cupy','onnxruntime'}:
            raise PermissionError('cost measurement forbids model/GPU import')
        if event!='open'or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]));flags=args[2]if len(args)>2 and isinstance(args[2],int)else 0
        writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:
            if str(path)!=result_path:raise PermissionError('cost measurement forbids input/claim/database writes')
            return
        if path.suffix.lower()in DENIED_SUFFIXES:raise PermissionError('cost measurement forbids original media/model/database input')
        if str(path).endswith(('.annotation.json','.annotation.json.commit.json')):raise PermissionError('original sidecar/commit reopen is not needed for sealed proof measurement')
        if path.suffix.lower()=='.jsonl'and str(path)not in allowed_jsonl:raise PermissionError('JSONL is outside the fixed finite track indexes')
        if path.suffix=='.pyc'and code in path.parents:raise PermissionError('unbound formal-root bytecode refused')
    sys.addaudithook(audit)


def _rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin'else 1024)


def _parse_proc_stat(raw,pid):
    tail=raw.rpartition(')')[2].split()
    need(type(pid)is int and pid>0 and int(raw.split(' ',1)[0])==pid and len(tail)>=20,'own process stat identity missing')
    value=dict(pid=pid,parent_pid=int(tail[1]),start_ticks=int(tail[19]));need(value['start_ticks']>0,'process start identity missing');return value


def _proc_identity(pid):
    return _parse_proc_stat(Path('/proc/%d/stat'%pid).read_text(),pid)


def _child(plan_path,plan_sha):
    c=check(dict(path=plan_path,sha256=plan_sha),created=True);p=c['plan'];out=Path(p['output_root']);code=Path(p['code_root'])
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.nice(0)==15,'CPU-only nice15 child required')
    allowed=os.sched_getaffinity(0);need(allowed,'one allowed CPU required');os.sched_setaffinity(0,{min(allowed)})
    result=dict(schema='annotation-native-startup-cost-measurement/v1',completed=False,source_counts=c['source_counts'],CPU_affinity=sorted(os.sched_getaffinity(0)),nice=15,phases=[],GPU_admitted=False,production_admitted=False,database_created=False,measurement_not_startup_or_ACK_qualification=True)
    _guard(out,code,c['allowed_jsonl'])
    try:
        need('native_batch_admission'not in sys.modules and 'bounded_speed_runtime'not in sys.modules,'ambient actual modules refused')
        sys.path.insert(0,str(code));start=time.monotonic();A=importlib.import_module('native_batch_admission');M=importlib.import_module('bounded_speed_runtime')
        need(Path(A.__file__)==code/'native_batch_admission.py'and Path(M.__file__)==code/'bounded_speed_runtime.py','actual module origins differ')
        result['import_seconds']=time.monotonic()-start
        def timed(name,call,files=None):
            started=time.monotonic();print(json.dumps(dict(phase_started=name)),flush=True)
            try:value=call()
            except BaseException:
                result['failed_phase']=dict(name=name,seconds=time.monotonic()-started,peak_RSS_bytes_so_far=_rss());raise
            row=dict(name=name,seconds=time.monotonic()-started,peak_RSS_bytes_so_far=_rss())
            if files is not None:row.update(files)
            result['phases'].append(row);print(json.dumps(row),flush=True);return value
        with A.NativeSelectionScope(enabled=True)as scope:
            selected=timed('full_selected_check',lambda:A.check_native_selected_pilot(p['selected'],c['contract'],'production',selection_scope=scope))
            refs=A._selection_inputs(selected,c['contract'],'production',p['selected'])
            counts=dict(unique_input_files=len(refs),unique_input_bytes=sum(Path(v['path']).stat().st_size for v in refs),unique_bytes_are_not_total_IO_bytes=True)
            result['scope_input_counts']=counts;result['phases'][0].update(counts)
            for i in range(REPEATS):timed('same_scope_reuse_'+str(i+1),lambda:A.check_native_selected_pilot(p['selected'],c['contract'],'production',selection_scope=scope),counts)
            result['scope_facts']=scope.facts();need(scope.facts()['full_checks']==1 and scope.facts()['reuses']==REPEATS,'actual scope execution counts differ')
            refs=M._unique_pins([*refs,p['formal_contract'],p['formal_bundle']])
            counts=dict(unique_input_files=len(refs),unique_input_bytes=sum(Path(v['path']).stat().st_size for v in refs),unique_bytes_are_not_total_IO_bytes=True)
            snap=timed('M_snapshot_capture',lambda:M._snapshot_verified_pins(refs),counts);counts['ancestor_directories']=len(snap['directories'])
            for i in range(REPEATS):timed('M_full_SHA_directory_verify_'+str(i+1),lambda:M._verify_snapshot(snap),counts)
        scope_rows=[r['seconds']for r in result['phases']if r['name'].startswith('same_scope_reuse_')];low=[r['seconds']for r in result['phases']if r['name'].startswith('M_full_SHA_directory_verify_')]
        result.update(completed=True,scope_eight_seconds=sum(scope_rows),lowlevel_eight_seconds=sum(low),lowlevel_last_seven_seconds=sum(low[1:]),ACK20_risk_proxy_exceeds= sum(low[1:])>20,
            proxy_excludes='RuntimeLease lineage/platform/small-control rechecks, locks/claim, memfd/PID authentication, child spawn/context/ACK; never qualifies the real 20/120/900 limits')
    except BaseException:result['exception']=traceback.format_exc()
    finally:result['peak_RSS_bytes']=_rss();write_new(out/'child-result.json',result)
    if not result['completed']:raise SystemExit(1)


def run(plan_pin,*,execute=False):
    started=time.monotonic();c=check(plan_pin);p=c['plan']
    if not execute:return dict(checked=True,executed=False,source_counts=c['source_counts'],GPU_admitted=False)
    need(sys.platform.startswith('linux')and hasattr(os,'wait4'),'actual Linux wait4 required');out=Path(p['output_root']);out.mkdir(mode=0o700)
    env={k:v for k,v in os.environ.items()if k not in {'PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','BASH_ENV','ENV'}}
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    controller=Path(__file__).resolve();controller_sha=sha_file(controller);argv=[sys.executable,'-I','-S','-B','-c',CHILD_CODE,str(controller),plan_pin['path'],plan_pin['sha256'],controller_sha]
    result=dict(schema='annotation-native-startup-cost-controller/v1',plan=plan_pin,controller_sha256=controller_sha,argv=argv,timeout=False,child_reaped=False,GPU_admitted=False,production_admitted=False)
    child=None;handlers={};usage=None
    def interrupted(sig,frame):raise RuntimeError('cost controller interrupted: '+str(sig))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        parent_identity=_proc_identity(os.getpid());boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();need(boot,'actual boot identity required')
        write_new(out/'intent.json',dict(result,max_seconds=p['max_seconds'],single_CPU=True,nice=15,CUDA_VISIBLE_DEVICES='',parent_identity=parent_identity,boot_id=boot,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        current=os.nice(0);need(current<=15,'exact nice15 unavailable');os.nice(15-current)
        with (out/'stdout.log').open('xb')as stdout,(out/'stderr.log').open('xb')as stderr:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,shell=False,start_new_session=True);termination=None
            child_identity=_proc_identity(child.pid);need(child_identity['parent_pid']==parent_identity['pid'],'Popen child actual parent differs')
            identity=dict(schema='annotation-native-cost-owned-processes/v1',parent=parent_identity,child=child_identity,boot_id=boot,plan=plan_pin,controller_sha256=controller_sha,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_new(out/'child-identity.json',identity);result['child_identity']={'path':str(out/'child-identity.json'),'sha256':sha_file(out/'child-identity.json')}
            while True:
                pid,status,ru=os.wait4(child.pid,os.WNOHANG)
                if pid:result.update(child_reaped=True,returncode=os.waitstatus_to_exitcode(status));usage=ru;child.returncode=result['returncode'];break
                if termination is None and time.monotonic()-started>=p['max_seconds']:
                    result['timeout']=True;os.kill(child.pid,signal.SIGTERM);termination=time.monotonic()
                elif termination is not None and time.monotonic()-termination>=1:os.kill(child.pid,signal.SIGKILL);termination=float('inf')
                time.sleep(.05)
    except BaseException:result['exception']=traceback.format_exc()
    finally:
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        if child is not None and not result['child_reaped']:
            try:os.kill(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:_,status,usage=os.wait4(child.pid,0);result.update(child_reaped=True,returncode=os.waitstatus_to_exitcode(status));child.returncode=result['returncode']
            except BaseException:result['cleanup_exception']=traceback.format_exc()
        result['elapsed_seconds']=time.monotonic()-started
        if usage is not None:result.update(peak_RSS_bytes=int(usage.ru_maxrss)*1024,child_user_CPU_seconds=usage.ru_utime,child_system_CPU_seconds=usage.ru_stime)
        result['completed']=False
        q=out/'child-result.json'
        if q.exists():
            try:
                s=q.lstat();need(stat.S_ISREG(s.st_mode)and s.st_size<=2*1024**2,'bounded regular child cost result required')
                raw=q.read_bytes();result['child_result']={'path':str(q),'sha256':hashlib.sha256(raw).hexdigest()}
                result['completed']=result.get('returncode')==0 and result['child_reaped']and not result['timeout']and not result.get('exception')and json.loads(raw).get('completed')is True
            except BaseException:result['child_result_parse_exception']=traceback.format_exc()
        try:write_new(out/'receipt.json',result)
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    value=run(dict(path=a.plan,sha256=a.plan_sha),execute=a.execute);print(json.dumps(value,ensure_ascii=False));raise SystemExit(0 if not a.execute or value['completed']else 1)
