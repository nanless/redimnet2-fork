"""固定原生试验封存器的单CPU诊断入口；默认只检查，执行也不加载GPU。

保存启动意图后才启动固定Python命令。每60秒保存原子进程调用栈，
正常、报错和超时都保存输出、退出码及wait4取得的实际峰值内存。
本工具不生成试验成功或正式标注准入证明，不修改任何运行中的任务。
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback

SCHEMA='annotation-native-cpu-sealer-run/v1'
MAX_OUTPUT_BYTES=256*1024**2
INJECTION_KEYS={'PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','DYLD_INSERT_LIBRARIES','DYLD_LIBRARY_PATH','BASH_ENV','ENV','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'}
# 固定代码不解释命令字符串：runpy保留旧sealer的__file__，显式补原入口目录。
CHILD_CODE="""import faulthandler,json,os,runpy,sys
from pathlib import Path
faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(60,repeat=True)
meta={'schema':'annotation-native-cpu-sealer-child/v1','pid':os.getpid(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'single_cpu_affinity_enforced':False}
if hasattr(os,'sched_getaffinity') and hasattr(os,'sched_setaffinity'):
    allowed=os.sched_getaffinity(0)
    if not allowed:raise RuntimeError('no allowed CPU')
    os.sched_setaffinity(0,{min(allowed)})
    meta.update(single_cpu_affinity_enforced=True,CPU_affinity=sorted(os.sched_getaffinity(0)))
Path(sys.argv[1]).write_text(json.dumps(meta,sort_keys=True)+'\\n')
sys.argv=sys.argv[2:]
sys.path.insert(0,str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0],run_name='__main__')
"""


def need(ok,message):
    if not ok:raise ValueError(message)


def physical(path):
    p=Path(path);need(p.is_absolute()and '..'not in p.parts,'absolute physical path required')
    need(all(not parent.is_symlink()for parent in (p,*p.parents)),'linked input/output ancestor refused')
    return p


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def pin(value,maximum):
    need(isinstance(value,dict)and set(value)=={'path','sha256'},'exact file path/SHA pin required')
    p=physical(value['path']);st=p.lstat();need(stat.S_ISREG(st.st_mode)and st.st_size<=maximum,'bounded regular input required')
    digest=value['sha256'];need(isinstance(digest,str)and len(digest)==64 and all(c in '0123456789abcdef'for c in digest),'lowercase SHA256 required')
    need(sha_file(p)==digest,'input SHA differs: '+str(p));return p


def read_pin(value,maximum):
    p=pin(value,maximum);raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==value['sha256'],'input changed while reading')
    return json.loads(raw)


def disjoint(a,b):return a!=b and a not in b.parents and b not in a.parents


def check(plan_pin,*,created_root=None):
    plan=read_pin(plan_pin,1024**2)
    keys={'schema','sealer','track','audit_code_root','audit_bundle','execution_root','deadline_seconds'}
    need(set(plan)==keys and plan.get('schema')==SCHEMA,'explicit CPU-sealer plan schema/fields required')
    deadline=plan['deadline_seconds'];need(type(deadline)in (int,float)and math.isfinite(deadline)and 0<deadline<=300,'CPU deadline must be positive and at most300 seconds')
    sealer=pin(plan['sealer'],16*1024**2);need(sealer.name=='native_batch_readback.py','only fixed native_batch_readback.py entry is allowed')
    track=pin(plan['track'],1024**2);spec=read_pin(plan['track'],1024**2)
    need(spec.get('schema')=='redimnet2-annotation-native-batch-track/v1','only finite native track may be sealed')
    audit=physical(plan['audit_code_root']);need(plan['audit_bundle']['path']==str(audit/'bundle_files.json'),'actual audit root/bundle path differs')
    bundle=read_pin(plan['audit_bundle'],8*1024**2)
    need(isinstance(bundle,dict)and {'audit_native_transformers_batch.py','audit_speed_comparison.py','gpu_launch.py'}<=set(bundle),'actual frozen audit bundle incomplete')
    for name,digest in bundle.items():
        rel=Path(name);need(rel.parts and not rel.is_absolute()and '..'not in rel.parts,'unsafe frozen bundle member')
        pin({'path':str(audit/rel),'sha256':digest},64*1024**2)
    root=physical(plan['execution_root']);need(root.parent.is_dir()and (not root.exists()if created_root is None else root==created_root and root.is_dir()),'fresh execution root with existing parent required')
    protected=[audit,sealer.parent,track.parent,physical(plan_pin['path'])]
    need(all(disjoint(root,p)for p in protected),'execution output overlaps frozen code or fixed input')
    for name in ('queue_root','output_root','resources_root'):
        need(name in spec and disjoint(root,physical(spec[name])),'CPU execution evidence overlaps producer tree')
    return plan


def write_new(path,value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def environment():
    env={k:v for k,v in os.environ.items()if k not in INJECTION_KEYS}
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    return env


def execute(plan_pin,*,allow=False):
    plan=check(plan_pin)
    if not allow:return dict(schema='annotation-native-cpu-sealer-check/v1',checked=True,executed=False,GPU_started=False,execution_root_created=False)
    need(hasattr(os,'wait4'),'wait4 is required for exact child peakRSS accounting')
    root=physical(plan['execution_root']);root.mkdir(mode=0o700)
    output=root/'sealed-readback.json';metadata=root/'child-control.json'
    argv=[sys.executable,'-I','-B','-u','-c',CHILD_CODE,str(metadata),plan['sealer']['path'],'--track',plan['track']['path'],'--track-sha',plan['track']['sha256'],
        '--audit-code-root',plan['audit_code_root'],'--audit-bundle-sha',plan['audit_bundle']['sha256'],'--output',str(output)]
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=dict(schema='annotation-native-cpu-sealer-execution/v1',plan=plan_pin,host=os.uname().nodename,parent_pid=os.getpid(),created_utc=now(),
        child_started=False,timeout=False,returncode=None,peak_RSS_bytes=None,peak_RSS_basis='unavailable_before_wait4',sealed_output=None,
        GPU_started_by_controller=False,full_library_audit=False,database_created=False,old_task_modified=False,production_admitted=False)
    child=None;reaped=False;pidfd=None;usage=None;started=time.monotonic();handlers={}
    def interrupted(signum,frame):raise RuntimeError('CPU sealer controller interrupted: signal '+str(signum))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
        intent=dict(result,argv=argv,python_binary={'path':str(Path(sys.executable).resolve()),'sha256':sha_file(sys.executable)},
            controller_source_sha256=sha_file(__file__),child_code_sha256=hashlib.sha256(CHILD_CODE.encode()).hexdigest(),source_pins={'sealer':plan['sealer'],'track':plan['track'],'audit_bundle':plan['audit_bundle']},
            deadline_seconds=plan['deadline_seconds'],stack_dump_interval_seconds=60,CUDA_VISIBLE_DEVICES='',shell_used=False)
        write_new(root/'intent.json',intent)
        # 意图写盘后再核固定输入；不得把已改源码或旧输出传给子进程。
        check(plan_pin,created_root=root)
        with (root/'stdout.log').open('xb')as stdout,(root/'stderr.log').open('xb')as stderr:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=environment(),shell=False,start_new_session=True)
            result.update(child_started=True,child_pid=child.pid,started_utc=now())
            if hasattr(os,'pidfd_open'):
                try:pidfd=os.pidfd_open(child.pid)
                except OSError:pass
            def send(sig):
                if pidfd is not None and hasattr(signal,'pidfd_send_signal'):signal.pidfd_send_signal(pidfd,sig)
                else:os.kill(child.pid,sig)  # 尚未wait4收割的本次child PID不能被系统复用。
            terminate_at=None
            while True:
                pid,status,ru=os.wait4(child.pid,os.WNOHANG)
                if pid:
                    reaped=True;usage=ru;child.returncode=os.waitstatus_to_exitcode(status);result['returncode']=child.returncode;break
                elapsed=time.monotonic()-started
                if terminate_at is None and elapsed>=plan['deadline_seconds']:
                    result['timeout']=True;send(signal.SIGTERM);terminate_at=time.monotonic()
                elif terminate_at is not None and time.monotonic()-terminate_at>=1:
                    send(signal.SIGKILL);terminate_at=float('inf')
                time.sleep(.05)
    except BaseException:
        result['exception']=traceback.format_exc()
    finally:
        # 清理与收据写完前忽略重复中断，避免第二次TERM使finally再次跳出。
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        if child is not None and not reaped:
            try:
                if pidfd is not None and hasattr(signal,'pidfd_send_signal'):signal.pidfd_send_signal(pidfd,signal.SIGKILL)
                else:os.kill(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:
                _,status,usage=os.wait4(child.pid,0);reaped=True;child.returncode=os.waitstatus_to_exitcode(status);result['returncode']=child.returncode
            except BaseException:result['cleanup_exception']=traceback.format_exc()
        if pidfd is not None:os.close(pidfd)
        result.update(finished_utc=now(),elapsed_seconds=time.monotonic()-started,child_reaped=reaped)
        if usage is not None:result.update(peak_RSS_bytes=int(usage.ru_maxrss)*(1 if sys.platform=='darwin' else 1024),peak_RSS_basis='exact_direct_child_wait4_ru_maxrss',child_user_CPU_seconds=usage.ru_utime,child_system_CPU_seconds=usage.ru_stime)
        if output.exists():
            try:
                st=output.lstat();need(stat.S_ISREG(st.st_mode)and st.st_size<=MAX_OUTPUT_BYTES,'unexpected sealed output type/size')
                result['sealed_output']={'path':str(output),'bytes':st.st_size,'sha256':sha_file(output)}
            except BaseException:result['sealed_output_exception']=traceback.format_exc()
        for name in ('stdout.log','stderr.log','child-control.json'):
            path=root/name
            if path.exists():result[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha_file(path)}
        result['completed']=result['returncode']==0 and not result['timeout']and not result.get('exception')and result['sealed_output']is not None
        try:write_new(root/'receipt.json',result)
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    value=execute({'path':a.plan,'sha256':a.plan_sha},allow=a.execute)
    print(json.dumps(value,ensure_ascii=False));return 0 if not a.execute or value['completed'] else 1

if __name__=='__main__':raise SystemExit(main())
