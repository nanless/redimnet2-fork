"""仅用自有合成CPU进程实证Linux父死亡保护，不打开训练数据或GPU。

控制器终止本次创建的等待父进程，观察两个已登记PID/start的空闲读进程
是否由内核结束。失败时仅清理nonce和启动时钟均匹配的自有进程，原件保留。
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import subprocess
import sys
import time
import traceback

import union_coverage as U

NONCE='REDIM_CPU_PARENT_DEATH_PROBE_NONCE'


def verified_syscall(number,name):
    U.need(sys.platform=='linux'and platform.machine()=='x86_64','no verified pidfd fallback for this architecture')
    header=Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    U.need(re.search(r'^#define\s+__NR_'+name+r'\s+'+str(number)+r'\s*$',header,re.M),
           'pidfd syscall number not verified by local header')
    library=ctypes.CDLL(None,use_errno=True);library.syscall.restype=ctypes.c_long
    return library


def verify_pidfd_interfaces():
    if not hasattr(os,'pidfd_open'):verified_syscall(434,'pidfd_open')
    if not hasattr(signal,'pidfd_send_signal'):verified_syscall(424,'pidfd_send_signal')


def open_exact_pidfd(pid):
    if hasattr(os,'pidfd_open'):descriptor=os.pidfd_open(pid,0)
    else:
        library=verified_syscall(434,'pidfd_open')
        descriptor=library.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
        if descriptor<0:raise OSError(ctypes.get_errno(),'pidfd_open failed')
    try:
        os.set_inheritable(descriptor,False)
        info=Path('/proc/self/fdinfo')/str(descriptor)
        U.need(re.search(r'^Pid:\s*'+str(pid)+r'\s*$',info.read_text(),re.M)and
               os.readlink('/proc/self/fd/'+str(descriptor))=='anon_inode:[pidfd]','pidfd does not bind exact owned worker')
    except BaseException:os.close(descriptor);raise
    return descriptor


def kill_exact_pidfd(descriptor):
    if hasattr(signal,'pidfd_send_signal'):return signal.pidfd_send_signal(descriptor,signal.SIGKILL)
    library=verified_syscall(424,'pidfd_send_signal')
    result=library.syscall(ctypes.c_long(424),ctypes.c_int(descriptor),ctypes.c_int(signal.SIGKILL),
                           ctypes.c_void_p(),ctypes.c_uint(0))
    if result<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal failed')


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8')as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def identity_job():
    # 短暂等待只用于使两个自有CPU读进程都被真实调度；不调用read_result。
    time.sleep(.25)
    return {'pid':os.getpid(),'start_ticks':U._linux_start_ticks(os.getpid()),
            'parent_pid':os.getppid(),'nonce':os.environ[NONCE]}


def parent(output):
    U.need(sys.platform=='linux','Linux kernel probe only')
    with U.retained_read_pool(2,'process',producer_value={},contract={},purpose='cpu_fixture')as pool:
        futures=[pool.submit(identity_job)for _ in range(8)]
        identities={value['pid']:value for value in (future.result(timeout=15)for future in futures)}
        U.need(len(identities)==2,'both synthetic CPU workers must be observed')
        staged=Path(output)/'parent-ready.staged.json'
        write_new(staged,{'parent_pid':os.getpid(),
            'parent_start_ticks':U._linux_start_ticks(os.getpid()),'workers':list(identities.values()),
            'synthetic_fixture':True,'source_media_read':False,'GPU_executed':False})
        os.link(staged,Path(output)/'parent-ready.json')
        # 控制器持有stdin，随后发送TERM。默认信号退出不运行Python池清理，
        # 因而子进程消失要靠已设置的内核父死亡信号。
        sys.stdin.buffer.read(1)


def current_identity(item):
    proc=Path('/proc')/str(item['pid'])
    try:
        text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
    except FileNotFoundError:return {'same_process':False,'live':False,'state':'absent'}
    same=fields[19]==item['start_ticks']
    return {'same_process':same,'live':same and fields[0]!='Z','state':fields[0]}


def run(output):
    U.need(sys.platform=='linux','this probe requires actual Linux; no mock success')
    output=U.physical(output);U.need(not output.exists()and output.parent.is_dir(),'fresh persistent evidence directory required')
    here=Path(__file__).resolve().parent
    U.need(here!=output and here not in output.parents and output not in here.parents,'probe output overlaps its source tree')
    output.mkdir(mode=0o700);nonce=secrets.token_hex(24);env=os.environ.copy()
    env.update(U._CPU_ENV);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    env[NONCE]=nonce
    ready=None;owned=None;cleaned=[];result=None;failure=None;pidfds={};cleanup_errors=[];process=None
    with (output/'parent.stdout.log').open('xb')as out,(output/'worker.stderr.log').open('xb')as err:
        try:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--parent',str(output)],
                stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,start_new_session=True)
            owned={'pid':process.pid,'start_ticks':U._linux_start_ticks(process.pid)}
            write_new(output/'controller-created-parent.json',owned)
            deadline=time.monotonic()+25;path=output/'parent-ready.json'
            while not path.exists():
                U.need(process.poll()is None and time.monotonic()<deadline,'synthetic parent failed or was not ready')
                time.sleep(.05)
            ready=U.ref({'path':str(path),'sha256':U.sha_file(path)})
            U.need(ready['parent_pid']==owned['pid']and ready['parent_start_ticks']==owned['start_ticks'],
                   'created parent identity differs')
            verify_pidfd_interfaces()
            for worker in ready['workers']:
                U.need(worker['parent_pid']==owned['pid']and worker['nonce']==nonce and current_identity(worker)['live'],
                       'synthetic CPU worker identity differs before parent termination')
                pidfds[worker['pid']]=open_exact_pidfd(worker['pid'])
                U.need(current_identity(worker)['live'],'worker changed while opening its exact pidfd')
            U.need(current_identity(owned)['live'],'created parent no longer live')
            process.terminate();returncode=process.wait(timeout=10)
            deadline=time.monotonic()+10
            while any(current_identity(w)['live']for w in ready['workers'])and time.monotonic()<deadline:time.sleep(.05)
            observed=[{**w,**current_identity(w)}for w in ready['workers']]
            U.need(returncode==-signal.SIGTERM and not any(w['live']for w in observed),
                   'Linux parent death did not end all exact synthetic workers')
            result={'schema':'annotation-cpu-parent-death-kernel-probe/v1','passed':True,
                'parent':owned,'parent_returncode':returncode,'workers_after_parent_death':observed,
                'parent_python_finally_executed':False,'worker_cleanup_signals_sent':[],
                'pidfd_open_python_API':hasattr(os,'pidfd_open'),
                'pidfd_send_signal_python_API':hasattr(signal,'pidfd_send_signal'),
                'exact_worker_pidfds_verified':True,
                'fallback_syscall_numbers_verified_from_local_header':not(
                    hasattr(os,'pidfd_open')and hasattr(signal,'pidfd_send_signal')),
                'synthetic_fixture':True,'production_data_opened':False,'models_executed':False,
                'GPU_executed':False,'GPU_lifecycle_verified':False,'formal_runtime_admitted':False}
        except BaseException:
            failure=traceback.format_exc()
        finally:
            if process is not None:
                try:
                    if process.poll()is None:
                        try:process.terminate()
                        except ProcessLookupError:pass
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                except BaseException:cleanup_errors.append(traceback.format_exc())
            if ready is not None:
                for worker in ready['workers']:
                    try:
                        if current_identity(worker)['live']:
                            # 仅清理本次具备精确start与随机nonce的合成CPU子进程。
                            marker=(NONCE+'='+nonce).encode()
                            U.need(marker in (Path('/proc')/str(worker['pid'])/'environ').read_bytes().split(b'\0'),
                                   'refuse cleanup of process without this probe nonce')
                            U.need(current_identity(worker)['live']and worker['pid']in pidfds,'owned worker changed before cleanup')
                            kill_exact_pidfd(pidfds[worker['pid']]);cleaned.append(worker)
                    except (FileNotFoundError,ProcessLookupError):pass
                    except BaseException:cleanup_errors.append(traceback.format_exc())
            for descriptor in pidfds.values():
                try:os.close(descriptor)
                except BaseException:cleanup_errors.append(traceback.format_exc())
            if process is not None and process.stdin is not None:
                try:process.stdin.close()
                except BaseException:cleanup_errors.append(traceback.format_exc())
            if process is not None:
                try:
                    if process.poll()is None:
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                except BaseException:cleanup_errors.append(traceback.format_exc())
    if failure is not None or cleanup_errors:
        payload={'traceback':failure,'parent':owned,'ready':ready,
            'exact_owned_cleanup':cleaned,'cleanup_errors':cleanup_errors}
        try:write_new(output/'failure.json',payload)
        except BaseException:
            payload['failure_receipt_write_error']=traceback.format_exc()
            print(json.dumps(payload,ensure_ascii=False),file=sys.stderr,flush=True)
            raise RuntimeError('kernel probe failed; receipt could not be saved; details emitted to stderr')from None
        raise RuntimeError('kernel probe failed; preserved '+str(output/'failure.json'))
    write_new(output/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path);group.add_argument('--parent',type=Path)
    args=parser.parse_args()
    if args.parent is not None:parent(args.parent)
    else:print(json.dumps(run(args.output),ensure_ascii=False))


if __name__=='__main__':main()
