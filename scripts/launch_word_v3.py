#!/usr/bin/env python3
"""Explicit, conservative GPU launcher. Probe-only never imports torch.

Default torchrun binary workers create separate sessions; --run-path/spawn is
intentional so this launcher's own new torchrun session retains all workers.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from word_ft_v3 import resources as G

REPO=Path(__file__).resolve().parents[1]
GLOBAL_LOCK_ROOT=Path('/root/experiments/redimnet2/.word-gpu-device-locks')
FIXED_INFRA_LIBS=['/usr/local/nccl-rdma-sharp-plugins/lib','/usr/local/nvidia/lib','/usr/local/nvidia/lib64']

def write_once(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def preflight_inputs(args):
    from word_ft import schema
    from word_ft_v3 import train as T
    if not all([args.config,args.dataset,args.official,args.output]):raise G.AdmissionRefused('execute requires config/dataset/official/new output')
    config=json.loads(Path(args.config).read_text());T.validate_config(config)
    if config['topology']!=G.topology(args.world_size):raise G.AdmissionRefused('CLI/config topology differs')
    root=G.protected_path(args.output,REPO)
    if root.exists() and not args.resume:raise G.AdmissionRefused('fresh GPU output must not exist')
    if args.resume:T.validate_resume_target(root,args.resume,args.world_size,0)
    data=schema.load_dataset(args.dataset,verify=True,allow_candidates=config['allow_candidates'])
    if data.get('metadata',{}).get('split')!='train' or any(r.get('split')!='train' for r in data['records']):
        raise G.AdmissionRefused('only train split allowed')
    pool=T.build_family_index(data['records']) if config['sampling']=='matched_prefix_family' else data['records']
    if len(pool)<args.world_size*config['items_per_rank'] or len(data['speaker_map'])<2:raise G.AdmissionRefused('global step requires sufficient items/families and two speakers')
    for record in data['records']:T.predicted_vad_seconds(record)
    if G.sha_file(args.official)!=T.OFFICIAL_SHA:raise G.AdmissionRefused('official backbone SHA differs')
    decoder=config['decoder'];library=Path(decoder['library_root'])/'libsndfile.so.1'
    if G.sha_file(library)!=decoder['library_sha256']:raise G.AdmissionRefused('private decoder file SHA differs')
    return config,root,data['dataset_id']

def child_environment(ticket_path,ticket_sha,devices,config):
    env=dict(os.environ)
    for key in ['LD_PRELOAD','LD_AUDIT','PYTHONPATH','PYTHONHOME','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','TORCHELASTIC_RUN_ID']:env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in devices),
        WORD_GPU_TICKET=str(ticket_path),WORD_GPU_TICKET_SHA256=ticket_sha,
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        LD_LIBRARY_PATH=':'.join([config['decoder']['library_root']]+FIXED_INFRA_LIBS))
    return env

def worker_command(args):
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={args.world_size}',
         '--start-method=spawn','--run-path',str(Path(__file__).resolve()),'--worker','--world-size',str(args.world_size),
         '--config',str(Path(args.config).resolve()),'--dataset',str(Path(args.dataset).resolve()),
         '--official',str(Path(args.official).resolve()),'--output',str(Path(args.output).resolve())]
    if args.resume:cmd+=['--resume',str(Path(args.resume).resolve())]
    if args.stop_after is not None:cmd+=['--stop-after',str(args.stop_after)]
    return cmd

def owned_group_members(child):
    """Only our newly created session, with an inherited per-launch nonce."""
    proof=getattr(child,'_word_group_owner',None)
    if not proof or proof['pgid']!=child.pid or child.pid<=1 or child.pid==os.getpgrp():
        raise G.AdmissionRefused('missing own new-session ownership proof')
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            raw=(path/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
            if int(parts[2])!=child.pid or parts[0]=='Z':continue
            if int(parts[3])!=child.pid or int(parts[19])<proof['start_time']:
                raise G.AdmissionRefused('process group/session no longer belongs to this launch')
            env=(path/'environ').read_bytes().split(b'\0')
            token=('WORD_GPU_SESSION_NONCE='+proof['nonce']).encode()
            if token not in env:raise G.AdmissionRefused('own process group contains unproven member')
            members.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return members


def stop_own_group(child,grace_seconds=10.):
    """Clean only proven members of our session, even after its leader exits."""
    members=owned_group_members(child)
    if not members:return False
    os.killpg(child.pid,signal.SIGTERM)
    deadline=time.monotonic()+grace_seconds
    while time.monotonic()<deadline:
        child.poll()
        if not owned_group_members(child):return True
        time.sleep(.1)
    if owned_group_members(child):os.killpg(child.pid,signal.SIGKILL)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        child.poll()
        if not owned_group_members(child):return True
        time.sleep(.1)
    raise G.AdmissionRefused('own process group did not terminate; preserve evidence')


def supervise(cmd,env,log,devices,max_seconds=3600.,watch_seconds=3.,expected_source=None):
    if not 30<=max_seconds<=86400 or not 1<=watch_seconds<=10:raise G.AdmissionRefused('invalid supervision interval')
    start=time.monotonic();snapshots=[];child=None
    with Path(log).open('x') as stream:
        try:
            nonce=os.urandom(24).hex();env=dict(env,WORD_GPU_SESSION_NONCE=nonce)
            child=subprocess.Popen(cmd,env=env,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            child._word_group_owner={'pgid':child.pid,'start_time':G.process_start_time(child.pid),'nonce':nonce}
            while child.poll() is None:
                if expected_source is not None and G.training_source_identity(REPO)!=expected_source:
                    raise G.AdmissionRefused('GPU source changed while own job running')
                snapshot=G.capture_snapshot();G.allowed_process_group(snapshot,devices,child.pid)
                snapshots.append(snapshot)
                if time.monotonic()-start>max_seconds:raise G.AdmissionRefused('own experiment runtime limit reached')
                try:child.wait(timeout=watch_seconds)
                except subprocess.TimeoutExpired:pass
            if stop_own_group(child):raise G.AdmissionRefused('torchrun exited with residual workers; own group cleaned')
            return {'returncode':child.returncode,'pid':child.pid,'process_group':child.pid,
                    'elapsed_seconds':time.monotonic()-start,'watch_snapshots':snapshots}
        except BaseException:
            if child is not None:stop_own_group(child)
            raise


def launch(args):
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or int(os.environ.get('RANK','0'))!=0 or 'LOCAL_RANK' in os.environ:
        raise G.AdmissionRefused('parent launcher must run once, outside a platform rank group')
    G.validate_topology(G.topology(args.world_size))
    if args.gpu_indices is None:args.gpu_indices=list(range(args.world_size))
    if len(args.gpu_indices)!=args.world_size:raise G.AdmissionRefused('GPU index count differs from topology')
    report=G.protected_path(args.report,REPO)
    if report.exists():raise FileExistsError(report)
    result={'mode':'probe-only' if args.probe_only else 'execute','started_utc':G.utc(),
            'topology':G.topology(args.world_size),'gpu_hardware_validation_claimed':False,'source_identity':G.training_source_identity(REPO)}
    if args.probe_only:
        result.update(torchrun_spawned=False,torch_imported_in_probe_process=('torch' in sys.modules))
    try:
        config=root=dataset_id=None
        if args.execute:
            config,root,dataset_id=preflight_inputs(args)
            if report==root or root in report.parents:raise G.AdmissionRefused('report must remain outside fresh training output')
        # The global path is fixed, never a per-output selectable lock directory.
        if GLOBAL_LOCK_ROOT.resolve()!=GLOBAL_LOCK_ROOT:raise G.AdmissionRefused('global GPU lock root must not be an alias')
        initial=G.capture_snapshot();devices=G.selected_devices(initial,args.gpu_indices)
        with G.DeviceLocks(GLOBAL_LOCK_ROOT,[d['uuid'] for d in devices]) as locks:
            probe=G.double_probe(args.gpu_indices,args.memory_mib,interval_seconds=args.probe_seconds,
                    driver_used_limit_mib=args.driver_used_limit_mib,reserve_mib=args.reserve_mib)
            result['probe']=probe
            if args.probe_only:
                result.update(status='idle at two snapshots; no torchrun/CUDA started',finished_utc=G.utc())
                write_once(report,result);return 0
            final=G.capture_snapshot();current=G.check_idle(final,args.gpu_indices,args.memory_mib,args.driver_used_limit_mib,args.reserve_mib)
            if G.stable_identity(current)!=G.stable_identity(probe['devices']):raise G.AdmissionRefused('final GPU identity differs')
            if G.training_source_identity(REPO)!=result['source_identity']:raise G.AdmissionRefused('GPU source changed during admission')
            ticket=G.create_ticket(probe,locks,args.memory_mib,args.reserve_mib,args.driver_used_limit_mib,result['source_identity'])
            ticket.update(config_digest=G.digest(config),dataset_file_sha256=G.sha_file(args.dataset),dataset_id=dataset_id,
                output_root=str(root),resume_path=str(Path(args.resume).resolve()) if args.resume else None,stop_after=args.stop_after,
                launch_mode='torchrun_run_path_spawn_inherited_process_group',final_prelaunch_snapshot=final)
            ticket_path=report.with_name(report.stem+'.ticket.json');write_once(ticket_path,ticket)
            ticket_sha=G.sha_file(ticket_path)
            result.update(ticket_path=str(ticket_path),ticket_sha256=ticket_sha,final_prelaunch_snapshot=final)
            run=supervise(worker_command(args),child_environment(ticket_path,ticket_sha,probe['devices'],config),
                          report.with_name(report.stem+'.torchrun.log'),probe['devices'],args.max_runtime_seconds,expected_source=result['source_identity'])
            result.update(run,status='own torchrun completed' if run['returncode']==0 else 'own torchrun failed',finished_utc=G.utc())
            write_once(report,result);return run['returncode']
    except BaseException:
        result.update(status='refused_or_own_job_failed',failure_traceback=traceback.format_exc(),finished_utc=G.utc())
        write_once(report,result)
        if isinstance(sys.exc_info()[1],KeyboardInterrupt):raise
        print(result['failure_traceback'],file=sys.stderr);return 2


def main():
    p=argparse.ArgumentParser(description='完整字词单机 W2/W8 v3；显式探测或执行，忙卡绝不触碰。')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--probe-only',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--world-size',type=int,choices=(2,8),required=True)
    p.add_argument('--gpu-indices',nargs='+',type=int);p.add_argument('--memory-mib',type=int,default=4096)
    p.add_argument('--reserve-mib',type=int,default=2048);p.add_argument('--driver-used-limit-mib',type=int,default=256)
    p.add_argument('--probe-seconds',type=float,default=5);p.add_argument('--max-runtime-seconds',type=float,default=3600)
    p.add_argument('--report');p.add_argument('--config');p.add_argument('--dataset');p.add_argument('--official');p.add_argument('--output')
    p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    args=p.parse_args()
    if args.worker:
        from word_ft_v3.train import run
        print(json.dumps(run(json.loads(Path(args.config).read_text()),args.dataset,args.official,args.output,
            resume=args.resume,stop_after=args.stop_after,cpu_threads=1),ensure_ascii=False));return 0
    if not args.report:p.error('--report is required outside worker mode')
    return launch(args)
if __name__=='__main__':raise SystemExit(main())
