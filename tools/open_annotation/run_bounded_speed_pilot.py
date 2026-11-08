"""独立云端小批转写对照：固定收据、串行GPU父进程、完整审计后再前进。

本程序不安装软件、不选择生产剩余来源、不读取当前生产队列、不创建
数据库。只有--execute才启动固定gpu_launch；资源准入、设备锁、上下文
挑战及GPU归还仍由原冻结父进程负责。失败仅通知本次精确子父进程TERM，
等待其清理，不给其它进程发信号，不提前开始下一轨。
"""
from __future__ import annotations
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import stat
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from contracts import atomic_new, need, safe_join
from readonly_progress import strict_json
from gpu_common import open_pidfd, process_start
from prepare_speed_validation import check_index

SCHEMA='annotation-bounded-speed-pilot/v1'
PROFILES=('asr-fp32-eager','asr-fp32-sdpa','asr-bf16-sdpa')
HEX=re.compile(r'[0-9a-f]{64}')


def physical(value):
    p=Path(value)
    need(p.is_absolute() and '..' not in p.parts and all(not x.is_symlink() for x in (p,*p.parents)),
         'physical absolute path required')
    return p


def disjoint(a,b):
    return a!=b and a not in b.parents and b not in a.parents


def stable_bytes(path,maximum):
    path=physical(path);initial=path.lstat()
    need(stat.S_ISREG(initial.st_mode) and initial.st_size<=maximum,'bounded regular file required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    stamp=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    try:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and stamp(before)==stamp(initial),'input changed before open')
        with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(maximum+1)
        need(len(raw)==before.st_size and len(raw)<=maximum and stamp(before)==stamp(os.fstat(fd))==stamp(path.stat()),'input changed while reading')
        return raw,before
    finally:os.close(fd)


def pin(ref,maximum=32*1024**2):
    need(isinstance(ref,dict) and set(ref)=={'path','sha256'} and isinstance(ref['sha256'],str)
         and HEX.fullmatch(ref['sha256']), 'explicit file path/SHA required')
    raw,_=stable_bytes(ref['path'],maximum)
    need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'pinned file SHA differs: '+str(ref['path']))
    return strict_json(raw)


def file_identity(path):
    raw,before=stable_bytes(path,64*1024**2)
    return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),mtime_ns=before.st_mtime_ns,
                inode=before.st_ino,device=before.st_dev)


def argv_fields(argv):
    need(isinstance(argv,list) and len(argv)>=3 and all(isinstance(x,str) and '\x00'not in x for x in argv),
         'explicit argv required; no shell command accepted')
    args={};i=3
    while i<len(argv):
        key=argv[i];need(key.startswith('--') and key not in args,'duplicate/unknown argument layout')
        if key in {'--execute','--full-verify'}:args[key]=True;i+=1
        else:
            need(i+1<len(argv) and not argv[i+1].startswith('--'),'flag value missing')
            args[key]=argv[i+1];i+=2
    return args


def load_matrix(ref,count,*,cpu_fixture=False):
    matrix=pin(ref);root=physical(ref['path']).parent
    need(matrix.get('schema')=='annotation-bounded-speed-prepared/v1' and matrix.get('source_count')==count,
         'wrong finite prepared denominator/schema')
    need(matrix.get('execution_order')==list(PROFILES) and set(matrix.get('variants',{}))==set(PROFILES),
         'exact explicit baseline/FP32SDPA/BF16SDPA order required')
    need(matrix.get('GPU_executed')is False and matrix.get('commands_executed')is False
         and matrix.get('database_created')is False,'preparation is not execution proof')
    need(matrix.get('worker_count') in (2,8),'pilot worker count must be 2 or 8')
    index_rows=check_index(matrix['index'],count)
    result={'matrix':matrix,'root':root,'rows':index_rows,'variants':{}}
    for name in PROFILES:
        v=matrix['variants'][name];code=physical(v['freeze']['frozen_copy']);track_path=physical(v['track']['path'])
        need(code==root/name/'code' and track_path==root/name/'speed-track.json','prepared directory layout differs')
        bundle_ref={'path':str(code/'bundle_files.json'),'sha256':v['freeze']['bundle_sha256']}
        members=pin(bundle_ref)
        need(isinstance(members,dict) and all(k in members for k in ('gpu_launch.py','report.py','worker.py','audit_speed_comparison.py')),
             'frozen executable bundle incomplete')
        for rel,digest in members.items():
            path=physical(safe_join(code,rel));need(path.is_file() and file_identity(path)['sha256']==digest,'frozen code member changed')
        if not cpu_fixture:
            for module in ('gpu_launch.py','report.py','audit_speed_comparison.py'):
                need(members[module]==file_identity(HERE/module)['sha256'],'driver and frozen pilot control source differ: '+module)
        track=pin(v['track']);contract=pin(track['contract'])
        need(track.get('schema')=='redimnet2-annotation-speed-track/v1' and track['index']==matrix['index']
             and track['contract']==v['freeze']['contracts']['pilot_validation'],'track inputs differ')
        need(contract.get('purpose')=='pilot_validation' and contract.get('model_API_calls_allowed')==0
             and contract.get('release_training_labels')is False,'only offline pilot contracts permitted')
        need(physical(track['contract']['path']).parent==code/'contracts','contract outside frozen code')
        expected_profile={'dtype':'bfloat16' if name=='asr-bf16-sdpa' else 'float32',
                          'attention':'eager' if name=='asr-fp32-eager' else 'sdpa'}
        need(contract['runtime'].get('asr_compute')==expected_profile,'explicit compute profile differs')
        run=root/name/'run'
        need(track['queue_root']==str(run/'queue') and track['output_root']==str(run/'labels')
             and track['resources_root']==str(run/'resources-attempt1'),'finite run roots differ')
        argv=v['argv'];fields=argv_fields(argv)
        need(physical(argv[0]).is_file() and argv[1]=='-B' and physical(argv[2])==code/'gpu_launch.py',
             'only frozen gpu_launch argv permitted')
        expected={'--execute':True,'--bundle-sha':bundle_ref['sha256'],'--contract':track['contract']['path'],
                  '--contract-sha':track['contract']['sha256'],'--shard-index':matrix['index']['path'],
                  '--index-sha':matrix['index']['sha256'],'--queue-root':track['queue_root'],
                  '--output-root':track['output_root'],'--output':track['resources_root'],
                  '--run-root':str(root/name),'--full-verify':True}
        need(all(fields.get(k)==x for k,x in expected.items()) and set(fields)==set(expected)|{
            '--target-uuids','--device-lock-root','--max-runtime-seconds'},'frozen launcher flags differ')
        cap=1800 if count==30 else 7200
        need(fields['--max-runtime-seconds'].isdigit() and 60<=int(fields['--max-runtime-seconds'])<=cap,
             'phase runtime exceeds bounded pilot budget')
        need(fields['--target-uuids'] in ('auto2','auto8') or len(fields['--target-uuids'].split(','))==2,
             'unsupported pilot card selection')
        need(matrix['worker_count']==(8 if fields['--target-uuids']=='auto8' else 2),'prepared worker count/card selection differs')
        physical(fields['--device-lock-root'])
        result['variants'][name]={'argv':list(argv),'fields':fields,'track':track,'track_ref':v['track'],
                                  'contract':contract,'code':code,'members':members}
    first=result['variants'][PROFILES[0]]
    for v in result['variants'].values():
        for key in ('--target-uuids','--device-lock-root'):
            need(v['fields'][key]==first['fields'][key],'different device group across finite profiles')
        need(v['argv'][0]==first['argv'][0],'interpreter differs across profiles')
        need(v['members']==first['members'],'executable source bundle differs across profiles')
    return result


def compile_plan(plan_ref,output):
    plan=pin(plan_ref)
    need(plan.get('schema')==SCHEMA and plan.get('purpose') in ('production','cpu_fixture'),'wrong bounded pilot schema/purpose')
    need(plan.get('model_API_calls_allowed')==0 and plan.get('production_switch_allowed')is False,
         'pilot cannot release production or API use')
    budget=plan.get('wall_budget_seconds',21000)
    need(type(budget)is int and 900<=budget<=21600,'overall pilot budget must be 900..21600 seconds')
    matrices={key:load_matrix(plan[key+'_prepared'],count,cpu_fixture=plan['purpose']=='cpu_fixture')for key,count in [('screen',30),('main',3000)]}
    screen,main=matrices['screen'],matrices['main']
    need(all(main['rows'].get(k)==r for k,r in screen['rows'].items()),'screen must be exact main source subset')
    for name in PROFILES:
        a,b=(matrices[k]['variants'][name]for k in ('screen','main'))
        need(a['contract']==b['contract'] and a['members']==b['members'],'screen/main software or model contract differs')
        for key in ('--target-uuids','--device-lock-root'):
            need(a['fields'][key]==b['fields'][key],'screen/main resource group differs')
        need(a['argv'][0]==b['argv'][0],'screen/main interpreter differs')
    output=physical(output)
    need(not output.exists() and output.parent.is_dir() and all(disjoint(output,m['root'])for m in matrices.values()),
         'new external evidence output required')
    need(disjoint(screen['root'],main['root']),'screen/main trees must be independent')
    protected=plan.get('protected_roots')
    need(isinstance(protected,list) and 1<=len(protected)<=16,'explicit current production/PTN roots required')
    need(all(disjoint(output,physical(q)) and all(disjoint(m['root'],physical(q))for m in matrices.values())for q in protected),
         'pilot evidence/compute tree overlaps protected production or PTN')
    recovery=None
    if plan.get('recovery_index')is not None:
        index=pin(plan['recovery_index']);count=index.get('source_count')
        need(type(count)is int and count in (500,512),'recovery must explicitly contain 500 or 512 sources')
        rows=check_index(plan['recovery_index'],count)
        need(all(main['rows'].get(k)==r for k,r in rows.items()),'recovery source records differ from main')
        profiles=plan.get('recovery_profiles',list(PROFILES))
        need(isinstance(profiles,list) and profiles and all(isinstance(x,str)for x in profiles)
             and len(profiles)==len(set(profiles)) and set(profiles)<=set(PROFILES),'unknown recovery profiles')
        recovery={'index':plan['recovery_index'],'rows':rows,'profiles':profiles,'count':count}
    return {'plan':plan,'plan_ref':plan_ref,'budget':budget,'matrices':matrices,'recovery':recovery,'output':output}


def _pidfd_send(fd,sig):
    if hasattr(signal,'pidfd_send_signal'):
        signal.pidfd_send_signal(fd,sig,None,0);return
    need(platform.system()=='Linux' and platform.machine()=='x86_64','verified pidfd signal unavailable')
    header=Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    need(re.search(r'^#define\s+__NR_pidfd_send_signal\s+424\s*$',header,re.M),'pidfd signal syscall not verified')
    lib=ctypes.CDLL(None,use_errno=True);lib.syscall.restype=ctypes.c_long
    rc=lib.syscall(ctypes.c_long(424),ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))
    if rc<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal failed')


class SerialExecutor:
    """只持有本次Popen子父进程身份，TERM后等待原GPU父自行清理。"""
    def __init__(self,output,deadline,cancel):
        self.output,self.deadline,self.cancel=output,deadline,cancel
    def command(self,argv,name,code):
        need(not self.cancel.is_set() and time.monotonic()<self.deadline,'pilot cancelled or overall budget exhausted')
        env=dict(os.environ)
        for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):
            env.pop(key,None)
        nonce=secrets.token_hex(24);env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',BOUNDED_PILOT_NONCE=nonce,
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        identity=None;fd=None;proc=None;terminated=False
        def identify():
            need(identity is not None and proc.pid==identity['pid'],'owned Popen identity missing')
            if identity['start_ticks'] is not None:
                need(process_start(proc.pid)==identity['start_ticks'],'owned child start changed')
            else:
                # 刚创建且尚未回收的Popen加先打开的内核pidfd，仍精确绑定本次子父。
                # 这个分支只用于失败清理，不授运行成功或资源归还资格。
                need(fd is not None,'neither start ticks nor kernel pidfd available for cleanup')
            need(os.getpgid(proc.pid)==os.getsid(proc.pid)==proc.pid,'owned child session changed')
            raw=Path('/proc')/str(proc.pid)
            need((raw/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')==[s.encode()for s in argv],'owned child argv differs')
            need(('BOUNDED_PILOT_NONCE='+nonce).encode()in(raw/'environ').read_bytes().split(b'\0'),'owned child nonce differs')
            stat=(raw/'stat').read_text();need(int(stat[stat.rfind(')')+2:].split()[1])==os.getpid(),'owned child parent differs')
        def stop_and_wait():
            nonlocal terminated
            if proc is not None and proc.poll()is None:
                identify()
                if fd is not None:_pidfd_send(fd,signal.SIGTERM)
                else:proc.terminate()  # 未回收Popen子进程的start/argv/nonce/PPID仍逐项证明
                terminated=True
                atomic_new(self.output/(name+'-TERM.json'),dict(pid=proc.pid,start_ticks=identity['start_ticks'],scope='this exact child parent only; original parent owns GPU cleanup'))
                while proc.poll()is None:time.sleep(.2)
        try:
            with (self.output/(name+'.log')).open('xb')as log:
                proc=subprocess.Popen(argv,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity={'pid':proc.pid,'start_ticks':None,'argv':argv,'nonce':nonce,'driver_pid':os.getpid()}
                # Popen尚未被wait/poll回收，pidfd先绑定，避免首次/proc读失败时丢掉清理句柄。
                fd=open_pidfd(proc.pid);identity['start_ticks']=process_start(proc.pid);identify();atomic_new(self.output/(name+'-child.json'),identity)
                print(json.dumps({'event':'pilot_step_started','step':name,'pid':proc.pid}),flush=True)
                while proc.poll()is None:
                    if self.cancel.is_set()or time.monotonic()>=self.deadline:
                        stop_and_wait();raise RuntimeError('pilot stopped; original parent cleanup awaited; no next step')
                    time.sleep(.2)
                rc=proc.wait()
            atomic_new(self.output/(name+'-exit.json'),dict(returncode=rc,TERM_sent=terminated))
            need(rc==0,'bounded pilot child failed: '+name)
            return rc
        except BaseException:
            if proc is not None and proc.poll()is None:
                # 无法证明身份时不向陌生PID退化发送信号，故障向外保留。
                stop_and_wait()
            raise
        finally:
            if fd is not None:os.close(fd)


def verify_parent(v):
    root=Path(v['track']['resources_root']);parent=pin({'path':str(root/'parent_result.json'),'sha256':file_identity(root/'parent_result.json')['sha256']})
    need(parent.get('resource_returned')is True and parent.get('full_sidecar_readback')is True
         and parent.get('verification',{}).get('passed')is True
         and parent['verification'].get('database_created')is False
         and parent['verification'].get('source_disjointness_storage')=='bounded_pilot_memory_set',
         'parent did not complete bounded no-database verification/resource return')
    need(parent['contract_sha256']==v['track']['contract']['sha256'] and parent['index_sha256']==v['track']['index']['sha256']
         and parent['bundle_sha256']==v['fields']['--bundle-sha'],'parent result identity differs')
    returned=json.loads((root/'resource_return_double_probe.json').read_text())
    need(returned.get('interval_seconds',0)>=5 and len(returned.get('snapshots',[]))==2,'resource return double probe missing')
    for snap in returned['snapshots']:
        need(not snap['compute_processes'] and all(d['used_mib']<=256 for d in snap['devices']),'resources did not return idle')
    drain=json.loads((root/'terminal_drain_complete.json').read_text())
    need(drain.get('all_attested_contexts_absent')is True,'CUDA context drain not proved')
    return {'parent_result_sha256':file_identity(root/'parent_result.json')['sha256'],'resource_returned':True}


def audited_track(v):
    """完整有限readback；只读Python字典计账，不使用联合库或ASR阶段库。"""
    from audit_speed_comparison import load_track
    return load_track(v['track_ref']['path'],v['track_ref']['sha256'])


def snapshot_recovery(v,rows):
    tree=Path(v['track']['output_root'])/v['contract']['version'];result={}
    for sid,row in rows.items():
        p=safe_join(tree,row['sidecar_relpath'])
        result[sid]={str(p):file_identity(p),str(p)+'.commit.json':file_identity(str(p)+'.commit.json')}
    return result


def run(plan_ref,output,execute=False,*,_fixture_executor=None):
    compiled=compile_plan(plan_ref,output)
    if not execute:return dict(checked=True,GPU_executed=False,database_created=False,source_counts=[30,3000],recovery_count=compiled['recovery']['count']if compiled['recovery']else 0)
    fixture=compiled['plan']['purpose']=='cpu_fixture'
    need(_fixture_executor is None or fixture,'custom executors prohibited for real pilot')
    need(not fixture or _fixture_executor is not None,'cpu_fixture execution requires fake executor; real commands refused')
    need(fixture or platform.system()=='Linux','real pilot requires Linux pidfd resources')
    need(int(os.environ.get('WORLD_SIZE','1'))==1 and int(os.environ.get('RANK','0'))==0 and 'LOCAL_RANK'not in os.environ,
         'driver must run once outside DDP ranks')
    out=compiled['output'];out.mkdir(mode=0o700)
    cancel=threading.Event();old_handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):old_handlers[sig]=signal.signal(sig,lambda s,f:cancel.set())
    deadline=time.monotonic()+compiled['budget'];executor=_fixture_executor or SerialExecutor(out,deadline,cancel)
    result={'schema':'annotation-bounded-speed-pilot-result/v1','plan':plan_ref,'purpose':compiled['plan']['purpose'],
            'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'steps':[],'database_created':False,'production_switch_allowed':False,'human_accuracy_certified':False,
            'GPU_runtime_verified':False,'all_finite_audits_completed':False}
    try:
        atomic_new(out/'effective-plan.json',compiled['plan'])
        from audit_speed_comparison import analyze
        for stage in ('screen','main'):
            matrix=compiled['matrices'][stage];tracks={}
            for name in PROFILES:
                # 每条命令前重新核固定准备和全部冻结源，拒执行中被改写。
                need(pin(plan_ref)==compiled['plan'],'outer pilot plan changed before execution')
                matrix=load_matrix(compiled['plan'][stage+'_prepared'],30 if stage=='screen' else 3000,cpu_fixture=fixture)
                v=matrix['variants'][name];executor.command(v['argv'],stage+'-'+name,v['code'])
                need(not cancel.is_set(),'pilot cancelled after current parent cleanup')
                returned=verify_parent(v);tracks[name]=audited_track(v)
                result['steps'].append(dict(stage=stage,profile=name,**returned,source_count=len(tracks[name]['records'])))
                atomic_new(out/(stage+'-'+name+'-verified.json'),result['steps'][-1])
            for name in PROFILES[1:]:
                comparison=analyze(tracks[PROFILES[0]],tracks[name],mode='speed')
                atomic_new(out/(stage+'-compare-'+name+'.json'),comparison)
                need(comparison['all_planned_sources_accounted']and comparison['cold_speed_ratio_valid'],
                     'finite comparison incomplete or true cold measurement unavailable')
        recovery=compiled['recovery']
        if recovery:
            for name in recovery['profiles']:
                need(pin(plan_ref)==compiled['plan'],'outer pilot plan changed before recovery')
                main=load_matrix(compiled['plan']['main_prepared'],3000,cpu_fixture=fixture);v=main['variants'][name]
                before=snapshot_recovery(v,recovery['rows']);root=main['root']/name/('recovery'+str(recovery['count']))
                # 不复制或改写主结果；新队列只复用同合同已发布完整侧车。
                track=dict(v['track'],index=recovery['index'],queue_root=str(root/'queue'),resources_root=str(root/'resources-attempt1'))
                track_path=out/('recovery-'+name+'-track.json');track_sha=atomic_new(track_path,track)
                flags=dict(v['fields']);flags.update({'--shard-index':recovery['index']['path'],'--index-sha':recovery['index']['sha256'],
                    '--queue-root':track['queue_root'],'--output':track['resources_root'],'--max-runtime-seconds':'1800'})
                argv=list(v['argv'][:3])
                for k,value in flags.items():argv.extend([k]if value is True else[k,value])
                rv=dict(v,track=track,track_ref={'path':str(track_path),'sha256':track_sha},fields=flags)
                executor.command(argv,'recovery-'+name,v['code']);need(not cancel.is_set(),'pilot cancelled after recovery parent cleanup');returned=verify_parent(rv);resumed=audited_track(rv)
                baseline=audited_track(v);comparison=analyze(baseline,resumed,mode='recovery')
                atomic_new(out/('recovery-'+name+'-comparison.json'),comparison)
                need(snapshot_recovery(v,recovery['rows'])==before,'published finite results changed during recovery')
                totals=resumed['performance']['totals']
                need(comparison['strict_recovery_native_results_equal']and totals['sidecars_committed']==0
                     and totals['sidecar_cache_hits']==recovery['count']and totals['asr_rows']==totals['aligner_rows']==0,
                     'recovery reran model or changed native result')
                result['steps'].append(dict(stage='recovery',profile=name,sources=recovery['count'],new_ASR_calls=0,new_FA_calls=0,**returned))
        need(not cancel.is_set(),'pilot cancelled before final result')
        result['all_finite_audits_completed']=True
        atomic_new(out/'result.json',result);return result
    except BaseException:
        atomic_new(out/'failure.json',dict(traceback=traceback.format_exc(),completed_steps=result['steps'],no_next_step_started=True))
        raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();print(json.dumps(run({'path':str(a.plan),'sha256':a.plan_sha},a.output,a.execute),ensure_ascii=False))

if __name__=='__main__':main()
