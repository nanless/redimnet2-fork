"""在自有 CPU 子进程比较两种栈采集；只运行合成有限 JSON，不加载模型。

旧 watchdog 与 Python 线程分别执行相同的递归校验，最多各十五秒。
实际 schema.py 只按固定 SHA 抽取 _finite_tree 函数，不导入其余模块。
默认只检查输入；执行保存原始 stdout/stderr、退出码与源码摘要。
"""
from __future__ import annotations
import argparse
import ast
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def need(value,message):
    if not value:raise ValueError(message)


def source_function(path,expected):
    p=Path(path);need(p.is_absolute()and '..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents)), 'physical schema source required')
    raw=p.read_bytes();need(len(raw)<=8*1024**2 and hashlib.sha256(raw).hexdigest()==expected,'actual schema SHA differs')
    tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_finite_tree']
    need(len(nodes)==1 and not nodes[0].decorator_list,'exact finite-tree function required')
    namespace={'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
    return namespace['_finite_tree']


def validate(seconds,interval):
    need(type(seconds)in(int,float)and math.isfinite(seconds)and 0<seconds<=15,'synthetic duration must be at most fifteen seconds')
    need(type(interval)in(int,float)and math.isfinite(interval)and .005<=interval<=.5,'bounded stack capture interval required')


def child(mode,schema_path,schema_sha,seconds,interval,controller_sha):
    validate(seconds,interval);need(mode in('legacy_watchdog','python_GIL_thread'),'explicit stack mode required')
    need(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==controller_sha,'diagnostic controller changed')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU child must have no visible CUDA device')
    allowed=os.sched_getaffinity(0);need(allowed,'one allowed CPU required');os.sched_setaffinity(0,{min(allowed)})
    need(os.nice(0)==15,'explicit nice15 CPU child required')
    finite=source_function(schema_path,schema_sha)
    raw=json.dumps({'schema':'synthetic-only/v1','records':[{'nested':{'windows':[{'mask':[1]*64,'ids':list(range(32)),'numbers':[.25,.5,1.0]}for _ in range(4)]}}for _ in range(8)]},allow_nan=False)
    started=time.monotonic();stop=threading.Event();thread=None;counts={'dumps':0}
    print(json.dumps(dict(mode=mode,python=sys.version,CPU_affinity=sorted(os.sched_getaffinity(0)),nice=os.nice(0),CUDA_VISIBLE_DEVICES='',schema_sha256=schema_sha,controller_sha256=controller_sha)),flush=True)
    faulthandler.enable(all_threads=True)
    if mode=='legacy_watchdog':faulthandler.dump_traceback_later(interval,repeat=True,file=sys.stderr)
    else:
        def capture():
            while not stop.wait(interval):
                faulthandler.dump_traceback(file=sys.stderr,all_threads=True);counts['dumps']+=1
        thread=threading.Thread(target=capture,name='synthetic-GIL-stack-capture',daemon=True);thread.start()
    loops=0
    try:
        while time.monotonic()-started<seconds:
            need(not finite(json.loads(raw)),'synthetic finite JSON unexpectedly invalid');loops+=1
    finally:
        if mode=='legacy_watchdog':faulthandler.cancel_dump_traceback_later()
        else:stop.set();thread.join(2);need(not thread.is_alive(),'owned stack capture thread did not exit')
    print(json.dumps(dict(completed=True,loops=loops,dumps=counts['dumps']if thread else None,elapsed_seconds=time.monotonic()-started,no_models_loaded=not any(n.split('.')[0]in {'torch','qwen_asr','vllm','numpy'}for n in sys.modules))),flush=True)


def run(schema_path,schema_sha,output,*,seconds=10,interval=.01,execute=False):
    validate(seconds,interval);source_function(schema_path,schema_sha)
    output=Path(output);need(output.is_absolute()and '..'not in output.parts and not output.exists()and output.parent.is_dir(),'fresh bounded CPU diagnostic output required')
    need(all(not q.is_symlink()for q in output.parents),'linked diagnostic output refused')
    for p in (Path(schema_path).parent,Path(__file__).resolve().parent):
        need(output!=p and output not in p.parents and p not in output.parents,'diagnostic output overlaps source')
    if not execute:return dict(checked=True,executed=False,GPU_executed=False)
    output.mkdir(mode=0o700);current=os.nice(0);need(current<=15,'cannot claim exact nice15 from a lower-priority parent');os.nice(15-current)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for key in('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(key,None)
    result=dict(schema='synthetic-stack-capture-comparison/v1',controller_sha256=source_sha,schema_source={'path':str(schema_path),'sha256':schema_sha},seconds=seconds,interval=interval,runs=[],real_audio_or_model_read=False,GPU_executed=False)
    for mode in('legacy_watchdog','python_GIL_thread'):
        directory=output/mode;directory.mkdir();argv=[sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--child-mode',mode,'--schema',str(schema_path),'--schema-sha',schema_sha,'--seconds',str(seconds),'--interval',str(interval),'--controller-sha',source_sha]
        started=time.monotonic();timed_out=False
        try:r=subprocess.run(argv,cwd=directory,env=env,capture_output=True,timeout=seconds+4);rc=r.returncode;stdout=r.stdout;stderr=r.stderr
        except subprocess.TimeoutExpired as e:timed_out=True;rc=None;stdout=e.stdout or b'';stderr=e.stderr or b''
        (directory/'stdout.log').write_bytes(stdout);(directory/'stderr.log').write_bytes(stderr)
        result['runs'].append(dict(mode=mode,argv=argv,returncode=rc,timeout=timed_out,elapsed_seconds=time.monotonic()-started,stdout=stdout.decode(errors='replace'),stderr_bytes=len(stderr),stderr_sha256=hashlib.sha256(stderr).hexdigest(),stderr_path=str(directory/'stderr.log')))
    result['legacy_SIGSEGV_reproduced']=result['runs'][0]['returncode']==-11
    result['GIL_thread_completed']=result['runs'][1]['returncode']==0 and not result['runs'][1]['timeout']
    (output/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--schema',required=True);p.add_argument('--schema-sha',required=True);p.add_argument('--output');p.add_argument('--seconds',type=float,default=10);p.add_argument('--interval',type=float,default=.01);p.add_argument('--execute',action='store_true');p.add_argument('--child-mode',choices=('legacy_watchdog','python_GIL_thread'));p.add_argument('--controller-sha');a=p.parse_args()
    if a.child_mode:child(a.child_mode,a.schema,a.schema_sha,a.seconds,a.interval,a.controller_sha)
    else:print(json.dumps(run(a.schema,a.schema_sha,a.output,seconds=a.seconds,interval=a.interval,execute=a.execute),ensure_ascii=False))
