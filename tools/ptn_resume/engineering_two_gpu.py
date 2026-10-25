#!/usr/bin/env python3
"""真实B6的独立双卡工程检查；输入全为合成，不启动正式训练。"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path=Path(path);temporary=path.with_name(path.name+'.part')
    with temporary.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);temporary.unlink()


def imports(args):
    package=Path(args.package).absolute()
    need(package.is_dir() and not package.is_symlink(),'physical package directory required')
    need(sha(package/'bundle_files.json')==args.manifest_sha,'package manifest SHA differs')
    sys.path.insert(0,str(package/'runtime'))
    import strict_restore as S
    S.configure(package/'resume-contract.json',args.contract_sha,args.manifest_sha)
    need(S.PACKAGE==package,'configured package path differs')
    return S


def output_guard(path, package, source, run, protected_path):
    out=Path(path).absolute();protected_path(out)
    need(not out.exists() and not out.is_symlink(),'engineering output must be new')
    for value in (package,source,run,Path(__file__).resolve().parent):
        p=Path(value).resolve();r=out.resolve()
        need(r!=p and p not in r.parents and r not in p.parents,'engineering output overlaps source/package/production run')
    cursor=out.parent
    while cursor!=cursor.parent:
        need(not cursor.is_symlink(),'output ancestor symlink refused');cursor=cursor.parent
    need(out.parent.is_dir(),'engineering output parent must exist')
    return out


def group_members(pid, started, nonce):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            raw=(path/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
            if v[0]=='Z' or int(v[2])!=pid:continue
            need(int(v[3])==pid and int(v[19])>=started,'process group/session identity differs')
            env=(path/'environ').read_bytes().split(b'\0')
            need(('PTN_ENGINEERING_NONCE='+nonce).encode() in env,'own process group nonce differs')
            members.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return members


def process_proof(pid, nonce):
    raw=Path('/proc/%d/stat'%pid).read_text();v=raw[raw.rfind(')')+2:].split()
    env=Path('/proc/%d/environ'%pid).read_bytes().split(b'\0')
    need(('PTN_ENGINEERING_NONCE='+nonce).encode() in env,'process nonce differs/unreadable')
    pgid,sid,started=int(v[2]),int(v[3]),int(v[19]);need(pgid==sid,'only independent proved sessions supported')
    leader=Path('/proc/%d/stat'%pgid).read_text();lv=leader[leader.rfind(')')+2:].split()
    need(int(lv[2])==pgid and int(lv[3])==sid and int(lv[19])<=started,'session leader identity differs')
    need(('PTN_ENGINEERING_NONCE='+nonce).encode() in Path('/proc/%d/environ'%pgid).read_bytes().split(b'\0'),
         'session leader nonce differs')
    return {'pid':pid,'pgid':pgid,'sid':sid,'start':started,'leader_start':int(lv[19])}


def process_table(minimum_start, known_pids):
    rows={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            raw=(path/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
            if v[0]=='Z' or int(v[19])<minimum_start:continue
            rows[int(path.name)]={'ppid':int(v[1]),'pgid':int(v[2]),'start':int(v[19])}
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            if int(path.name) in known_pids:raise
    return rows


def discover_owned_groups(nonce, minimum_start, known, launcher_pid):
    """Kernel PG/PPID filter first: never read unrelated users' environ."""
    roots=set(known)|{launcher_pid}|{p['pid'] for p in known.values() if 'pid' in p}
    rows=process_table(minimum_start,roots);groups={}
    def candidate(pid):
        seen=set()
        while pid in rows and pid not in seen:
            seen.add(pid)
            if pid in roots or rows[pid]['pgid'] in roots:return True
            pid=rows[pid]['ppid']
        return False
    for pid in rows:
        if not candidate(pid):continue
        try:
            proof=process_proof(pid,nonce);groups[proof['pgid']]=proof
        except (FileNotFoundError,ProcessLookupError):continue
    return groups


def ready_groups(output, nonce, minimum_start):
    groups={}
    for rank in range(2):
        path=Path(output)/('ready-rank%d.json'%rank)
        if not path.is_file():continue
        row=json.loads(path.read_text());need(row['rank']==rank and row['nonce']==nonce,'private ready identity differs')
        expected=row['process']
        try:
            raw=Path('/proc/%d/stat'%expected['pid']).read_text();v=raw[raw.rfind(')')+2:].split()
            if int(v[19])!=expected['start']:continue  # PID reused after this original worker exited.
            actual=process_proof(expected['pid'],nonce)
        except (FileNotFoundError,ProcessLookupError):continue
        need(actual==expected and actual['start']>=minimum_start,'ready process evidence changed')
        groups[actual['pgid']]=actual
    return groups


def stop_owned(child, started, nonce):
    child.poll()
    for sig,seconds in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
        if not group_members(child.pid,started,nonce):return
        os.killpg(child.pid,sig);until=time.monotonic()+seconds
        while time.monotonic()<until:
            child.poll()
            if not group_members(child.pid,started,nonce):return
            time.sleep(.1)
    need(not group_members(child.pid,started,nonce),'own torchrun group did not exit; retain resource locks')


def group_presence(pgid):
    """Read-only signal-0 query, never TERM/KILL; errors mean unknown, not empty."""
    try:os.killpg(pgid,0)
    except ProcessLookupError:return False
    except (PermissionError,OSError):return None
    return True


def recover_child_identity(child, nonce):
    need(child.poll() is None,'original child already exited; do not infer a replacement identity')
    raw=Path('/proc/%d/stat'%child.pid).read_text();v=raw[raw.rfind(')')+2:].split()
    need(int(v[2])==child.pid and int(v[3])==child.pid,'unproved original child group/session')
    need(('PTN_ENGINEERING_NONCE='+nonce).encode() in Path('/proc/%d/environ'%child.pid).read_bytes().split(b'\0'),
         'original child nonce not readable/matching')
    return int(v[19])


def cleanup_child(child, started, nonce, output, *, recover=recover_child_identity,
                  stop=stop_owned, presence=group_presence, sleep=time.sleep):
    """Caller still holds UUID locks. Unknown ownership permits waiting only."""
    error=None
    try:
        if started is None:started=recover(child,nonce)
        stop(child,started,nonce)
    except BaseException:error=traceback.format_exc()
    if error is not None:
        try:write_new(Path(output)/('cleanup-unproven-%d.json'%child.pid),{'error':error,'locks_retained':True,'unknown_processes_signalled':False})
        except Exception:pass  # Diagnostics failure must not release the resource locks.
    quiet=0
    while quiet<2:
        child.poll()
        try:observed=presence(child.pid)
        except Exception:observed=None
        quiet=quiet+1 if observed is False else 0
        if quiet<2:sleep(.2)


def cleanup_all(child, started, nonce, output, known):
    """torchrun ranks may have separate sessions; never equate launcher exit to rank exit."""
    minimum=started
    if minimum is None:
        try:minimum=recover_child_identity(child,nonce)
        except Exception:minimum=0
    known=dict(known)
    while True:
        try:
            known.update(ready_groups(output,nonce,minimum))
            groups=discover_owned_groups(nonce,minimum,known,child.pid);break
        except Exception:
            try:write_new(Path(output)/'cleanup-discovery-unproven.json',{'error':traceback.format_exc(),'locks_retained':True})
            except Exception:pass
            time.sleep(.2)  # Unknown process inventory: retain locks, send no signal.
    groups.update(known)
    for pgid,proof in sorted(groups.items()):
        if pgid==child.pid:continue
        actor=SimpleNamespace(pid=pgid,poll=lambda:None)
        cleanup_child(actor,proof['leader_start'],nonce,output)
    cleanup_child(child,started,nonce,output)
    # Includes descendants not yet in ready at the instant the parent failed.
    while True:
        try:remaining=discover_owned_groups(nonce,minimum,groups,child.pid)
        except Exception:remaining=None
        if remaining=={}:break
        if remaining:
            for pgid,proof in remaining.items():
                cleanup_child(SimpleNamespace(pid=pgid,poll=lambda:None),proof['leader_start'],nonce,output)
        else:time.sleep(.2)
    if not (Path(output)/'cuda-ack.json').exists() and len(list(Path(output).glob('ready-rank*.json')))<2:
        time.sleep(120)  # Unregistered CPU workers cannot initialize CUDA without ACK; no guessed kill.


def make_step_state(model, optimizer, scaler, N):
    names={id(p):n for n,p in model.named_parameters()}
    ordered=[[names.get(id(p)) for p in g['params']] for g in optimizer.param_groups]
    need(all(n is not None for group in ordered for n in group),'optimizer contains foreign parameter')
    return {'schema':'synthetic-B6-step-state/v1','synthetic_engineering_only':True,
            'model':N._tree(model.state_dict(),cpu=True),
            'buffers':N._tree(dict(model.named_buffers(remove_duplicate=False)),cpu=True),
            'optimizer':N._tree(optimizer.state_dict(),cpu=True),'optimizer_parameter_names':ordered,
            'scaler':N._tree(scaler.state_dict(),cpu=True),
            'rng':N.rng_state(model)}


def restore_step_state(model, optimizer, scaler, state, N):
    need(state.get('schema')=='synthetic-B6-step-state/v1' and state.get('synthetic_engineering_only') is True,
         'engineering state is not a production checkpoint')
    names={id(p):n for n,p in model.named_parameters()}
    ordered=[[names.get(id(p)) for p in g['params']] for g in optimizer.param_groups]
    need(ordered==state['optimizer_parameter_names'],'restored optimizer order differs')
    model.load_state_dict(state['model'],strict=True)
    buffers=dict(model.named_buffers(remove_duplicate=False));need(set(buffers)==set(state['buffers']),'buffer members differ')
    import torch
    with torch.no_grad():
        for name,value in state['buffers'].items():
            need(buffers[name].shape==value.shape and buffers[name].dtype==value.dtype,'buffer geometry differs')
            buffers[name].copy_(value)
    optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);N.restore_rng(state['rng'])


def construct(T, cfg, torch, rank):
    model=T.SpeakerModel(T.get_speaker_model(cfg['model'])(**cfg['model_args']),T.get_projection(cfg['projection_args'])).cuda(rank).train()
    optimizer=torch.optim.SGD([
        {'params':model.backbone.parameters(),'weight_decay':cfg['optimizer_args']['weight_decay']},
        {'params':model.projection.parameters(),'weight_decay':cfg['loss_weight_decay_mult']*cfg['optimizer_args']['weight_decay']}],**cfg['optimizer_args'])
    scheduler=getattr(T.schedulers,cfg['scheduler'])(optimizer,**cfg['scheduler_args'])
    margin=getattr(T.schedulers,cfg['margin_scheduler'])(model=model,**cfg['margin_update'])
    scaler=torch.cuda.amp.GradScaler(enabled=True)
    return model,optimizer,scheduler,margin,scaler


def run_step(ddp, model, optimizer, scaler, criterion, wave, targets, scheduler, margin, iteration, N, rank, output, inject=False):
    import torch
    from wespeaker_lite.utils.executor import _check_training_numerics
    scheduler.step(iteration);margin.step(iteration)
    before={n:int(b) for n,b in model.named_buffers() if n.endswith('num_batches_tracked')}
    started=time.monotonic()
    with torch.cuda.amp.autocast(enabled=True):
        with N.ForwardCapture(ddp,targets=targets) as capture:
            embeds=ddp(wave)
        if isinstance(embeds,tuple):embeds=embeds[-1]
        if inject and rank==0:embeds=embeds*float('nan')
    with torch.cuda.amp.autocast(enabled=False):
        logits=model.projection(embeds.float(),targets,None)
        if isinstance(logits,tuple):logits,loss=logits
        else:loss=criterion(logits,targets)
    try:
        _check_training_numerics(wave,embeds,loss,epoch=48,batch_index=iteration-705047+1,rank=rank,
            optimizer=optimizer,scaler=scaler,model=model,targets=targets,capture_context=capture,
            diagnostic_dir=output/'injected-failure',original_batch={'wav':wave,'label':targets},
            enable_amp=True,absolute_iteration=iteration,epoch_iter=15001)
    finally:capture.release()
    optimizer.zero_grad()
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    need(gradients,'no gradients were computed')
    nonfinite=sum(not bool(torch.isfinite(g).all()) for g in gradients)
    before_scale=scaler.get_scale();called=[]
    handle=optimizer.register_step_post_hook(lambda *_:called.append(True))
    try:scaler.step(optimizer)
    finally:handle.remove()
    scaler.update();need(len(called) in (0,1),'GradScaler invoked SGD more than once')
    need((len(called)==1 and nonfinite==0 and scaler.get_scale()>=before_scale)
         or (not called and nonfinite>0 and scaler.get_scale()<before_scale),'AMP update/skip evidence inconsistent')
    after={n:int(b) for n,b in model.named_buffers() if n.endswith('num_batches_tracked')}
    deltas={k:after[k]-v for k,v in before.items()}
    need(before and set(before)==set(after) and all(x>=0 for x in deltas.values()) and any(x>0 for x in deltas.values()),
         'training BN counters did not advance monotonically')
    torch.cuda.synchronize()
    return {'loss':float(loss.detach()),'gradient_tensors_finite':len(gradients)-nonfinite,
            'gradient_tensors_nonfinite':nonfinite,'scaler_before':before_scale,
            'BN_counters_advanced':sum(x>0 for x in deltas.values()),'BN_counter_deltas':deltas,
            'scaler':scaler.state_dict(),'seconds':time.monotonic()-started,'absolute_iteration':iteration,
            'wave_frames':list(wave.shape),'actual_optimizer_step':len(called)==1,
            'normal_AMP_backoff_without_manual_scale_change':not called}


def run_until_update(*, ddp, model, optimizer, scaler, criterion, wave, targets, scheduler, margin,
                     iteration, N, rank, output, phase, maximum):
    attempts=[]
    for index in range(maximum):
        row=run_step(ddp,model,optimizer,scaler,criterion,wave,targets,scheduler,margin,iteration+index,N,rank,output)
        attempts.append(row);write_new(output/('attempt-%s-rank%d-%03d.json'%(phase,rank,index)),row)
        if row['actual_optimizer_step']:return attempts,iteration+index+1
    raise ValueError('bounded attempts exhausted without a real SGD update; all AMP skips preserved')


def update_consensus(dist, attempts, next_iteration, scaler):
    value={'attempt_count':len(attempts),'next_iteration':next_iteration,
           'updates':sum(a['actual_optimizer_step'] for a in attempts),'scaler':scaler.state_dict()}
    rows=[None]*2;dist.all_gather_object(rows,value)
    need(rows[0]==rows[1] and value['updates']==1,'ranks disagree on actual optimizer update/AMP attempt clock')


def worker(args):
    S=imports(args)
    sys.path.insert(0,str(REPO))
    from word_ft_v3 import resources as R
    identity=json.loads(Path(args.identity).read_text());need(sha(__file__)==identity['runner_sha256'],'runner drift')
    need(args.case==identity['case'] and args.manifest_sha==identity['package_manifest_sha256']
         and args.contract_sha==identity['contract_sha256'],'worker/package/case ticket binding differs')
    need(args.max_attempts_per_update==identity['max_attempts_per_update'],'attempt budget differs from parent ticket')
    need((sha(args.terminal_evidence) if args.terminal_evidence else None)==identity['terminal_evidence_sha256'],
         'terminal evidence selection changed')
    need(sha(REPO/'word_ft_v3/resources.py')==identity['resources_sha256'],'resource helper drift')
    for rel,pin in identity['resource_dependencies'].items():need(sha(REPO/rel)==pin,'resource dependency drift: '+rel)
    ticket=R.validate_ticket(args.ticket,args.ticket_sha,identity,max_age=120)
    rank=int(os.environ['RANK']);need(int(os.environ['WORLD_SIZE'])==2 and int(os.environ['LOCAL_RANK'])==rank,'one node W2 required')
    uuids=[d['uuid'] for d in ticket['devices']]
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(uuids),'logical CUDA/UUID binding differs')
    # No CUDA creation until both workers have verified the package/ticket and parent freshly checks idle.
    output=Path(args.output)
    nonce=os.environ['PTN_ENGINEERING_NONCE'];proof=process_proof(os.getpid(),nonce)
    write_new(output/('ready-rank%d.json'%rank),{'rank':rank,'ticket_sha256':args.ticket_sha,
        'process':proof,'nonce':nonce})
    until=time.monotonic()+120
    while not (output/'cuda-ack.json').is_file():
        need(time.monotonic()<until,'parent CUDA ACK timeout');time.sleep(.1)
    need(json.loads((output/'cuda-ack.json').read_text())['ticket_sha256']==args.ticket_sha,'ACK identity differs')
    sys.path[:0]=[str(S.SOURCE),str(S.SOURCE/'applications/speaker_rec')]
    import torch
    import torch.distributed as dist
    import yaml
    torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'worker inherited initialized CUDA')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(rank);need(torch.cuda.device_count()==2,'exactly two visible devices required')
    backend={'torch':str(torch.__version__),'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'extra_engineering_settings':['deterministic_algorithms=True','CUBLAS_WORKSPACE_CONFIG=:4096:8'],
        'production_numerical_trajectory_equivalence_claimed':False}
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=1000))
    try:
        from wespeaker_lite.bin import train as T
        from wespeaker_lite.utils import numerical_failure as N
        from types import SimpleNamespace
        import decoder_runtime
        decoder=decoder_runtime.audit_loaded_decoder(SimpleNamespace(decoder_library=str(S.DECODER)))
        cfg=yaml.safe_load(Path(S.CONFIG['resume_config_path']).read_bytes());S.validate_config(cfg)
        need(not cfg.get('compile_backbone') and cfg['model']=='ReDimNet2Custom','only uncompiled actual B6 supported')
        seed=int(cfg['seed'])+rank;__import__('random').seed(seed);__import__('numpy').random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        model,optimizer,scheduler,margin,scaler=construct(T,cfg,torch,rank)
        saved=S.checkpoint();model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        terminal=None
        if args.terminal_evidence:terminal=json.loads(Path(args.terminal_evidence).read_text())
        proof=S.state_gate.verify_restored(S.CONTRACT_PATH,S.CONTRACT_SHA,model=model,optimizer=optimizer,scaler=scaler,
            scheduler=scheduler,margin=margin,config=cfg,speaker_map_path=S.CONFIG['speaker_map']['path'],terminal_evidence=terminal)
        proof.update(engineering_world_size=2,production_world_size=8,formal_W8_admission=False,
            imports=S.verify_imports(),decoder=decoder,actual_backend=backend)
        write_new(output/('restore-rank%d.json'%rank),proof)
        allproof=[None]*2;dist.all_gather_object(allproof,{'rank':rank,'verified':True});need(all(x=={'rank':i,'verified':True} for i,x in enumerate(allproof)),'restore consensus differs')
        del saved
        ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],find_unused_parameters=False)
        generator=torch.Generator(device='cpu');generator.manual_seed(2026102600+rank)
        wave=(torch.randn((96,1,32240),generator=generator,dtype=torch.float32)*.03).to(rank)
        targets=((torch.arange(96,dtype=torch.int64)+rank*96)%357822).to(rank)
        write_new(output/('synthetic-input-rank%d.json'%rank),{'synthetic_engineering_only':True,'seed':2026102600+rank,
            'shape':[96,1,32240],'dtype':'float32','labels_range':[rank*96,rank*96+95],
            'wave_float32_sha256':hashlib.sha256(wave.cpu().numpy().tobytes()).hexdigest(),'model47_failure_reproduction':False})
        criterion=getattr(torch.nn,cfg['loss'])(**cfg['loss_args']);torch.cuda.reset_peak_memory_stats(rank)
        if args.case=='injected-failure':
            caught=False
            try:run_step(ddp,model,optimizer,scaler,criterion,wave,targets,scheduler,margin,705047,N,rank,output,inject=True)
            except FloatingPointError:caught=True
            need(caught,'injected NaN did not stop before backward')
            files=list((output/'injected-failure').glob('*/coordination.json'));need(len(files)==2,'both forensic coordination files required')
            for f in files:
                row=json.loads(f.read_text());need(row['all_rank_replay_complete'] is True,'injected export incomplete')
            write_new(output/('result-rank%d.json'%rank),{'passed':True,'synthetic_engineering_only':True,'injected_NaN':True,
                'original_fault_reproduced':False,'optimizer_updates':0,'actual_backend':backend})
        else:
            first,next_iteration=run_until_update(ddp=ddp,model=model,optimizer=optimizer,scaler=scaler,criterion=criterion,
                wave=wave,targets=targets,scheduler=scheduler,margin=margin,iteration=705047,N=N,rank=rank,output=output,
                phase='first',maximum=args.max_attempts_per_update)
            update_consensus(dist,first,next_iteration,scaler)
            state=make_step_state(model,optimizer,scaler,N);path=output/('step1-rank%d.pt'%rank)
            state['next_iteration']=next_iteration
            with path.open('xb') as stream:torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
            pin=sha(path);del state
            second,final_iteration=run_until_update(ddp=ddp,model=model,optimizer=optimizer,scaler=scaler,criterion=criterion,
                wave=wave,targets=targets,scheduler=scheduler,margin=margin,iteration=next_iteration,N=N,rank=rank,output=output,
                phase='second',maximum=args.max_attempts_per_update)
            update_consensus(dist,second,final_iteration,scaler)
            continuous=make_step_state(model,optimizer,scaler,N)
            need(sha(path)==pin,'step1 saved state changed');restored=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            replay_iteration=restored['next_iteration'];need(type(replay_iteration)is int and replay_iteration==next_iteration,'saved attempt clock differs')
            dist.barrier();del ddp,model,optimizer,scaler,scheduler,margin
            torch.cuda.empty_cache()
            model,optimizer,scheduler,margin,scaler=construct(T,cfg,torch,rank)
            restore_step_state(model,optimizer,scaler,restored,N)
            ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],find_unused_parameters=False)
            N.restore_rng(restored['rng']);del restored
            replay,replay_final_iteration=run_until_update(ddp=ddp,model=model,optimizer=optimizer,scaler=scaler,criterion=criterion,
                wave=wave,targets=targets,scheduler=scheduler,margin=margin,iteration=replay_iteration,N=N,rank=rank,output=output,
                phase='replayed-second',maximum=args.max_attempts_per_update)
            update_consensus(dist,replay,replay_final_iteration,scaler)
            actual=make_step_state(model,optimizer,scaler,N);S.state_gate.exact(actual,continuous,'continuous_vs_restored_step2')
            need(sha(path)==pin,'first-step checkpoint changed during restoration')
            need(final_iteration==replay_final_iteration and len(second)==len(replay),'attempt trajectory differs after restore')
            for old,new in zip(second,replay):
                for key in ('loss','actual_optimizer_step','absolute_iteration','scaler','gradient_tensors_nonfinite','BN_counter_deltas'):
                    need(old[key]==new[key],'step2 attempt differs after restore: '+key)
            write_new(output/('result-rank%d.json'%rank),{'passed':True,'synthetic_engineering_only':True,'case':'finite',
                'formal_training_admission':False,'continuous_updates':2,'replayed_updates':1,'step1_sha256':pin,
                'fresh_model_optimizer_DDP_reconstructed':True,'process_restart_tested':False,
                'step1':first,'continuous_step2':second,'restored_step2':replay,'state_exact':True,
                'normal_AMP_skips':sum(not a['actual_optimizer_step'] for a in first+second),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(rank),'peak_reserved_bytes':torch.cuda.max_memory_reserved(rank),
                'attempted_waveforms_per_second':96*len(first+second)/sum(a['seconds'] for a in first+second),
                'actual_update_waveforms_per_second':192/sum(a['seconds'] for a in first+second),
                'deterministic_engineering_policy':True,'actual_backend':backend,
                'model_AMP_optimizer_LR_loss_config_unchanged':True,
                'production_numerical_trajectory_equivalence_claimed':False})
        dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def parent(args):
    need('LOCAL_RANK'not in os.environ and int(os.environ.get('WORLD_SIZE','1'))==1
         and int(os.environ.get('RANK','0'))==0,'one parent only')
    S=imports(args)
    sys.path.insert(0,str(REPO));from word_ft_v3 import resources as R
    output=output_guard(args.output,S.PACKAGE,S.SOURCE,S.RUN,R.protected_path)
    need(len(set(args.expected_uuids))==2,'two unique target UUIDs required')
    need(1<=args.max_seconds<=1200,'engineering wall time must be <=1200 seconds')
    need(1<=args.max_attempts_per_update<=32,'finite AMP attempts must be bounded 1..32')
    need('torch'not in sys.modules,'parent must not import torch/CUDA')
    need(Path(sys.executable).resolve()==Path(S.PYTHON).resolve(),'use the fixed PTN interpreter, not Qwen/system Python')
    with R.DeviceLocks(args.lock_root,args.expected_uuids) as locks:
        probe=R.double_probe([0,1],60000)
        need([d['uuid'] for d in probe['devices']]==args.expected_uuids,'actual target UUIDs differ')
        output.mkdir(mode=0o700)
        identity={'runner_sha256':sha(__file__),'resources_sha256':sha(REPO/'word_ft_v3/resources.py'),
            'resource_dependencies':{rel:sha(REPO/rel) for rel in ('word_ft_v3/resources.py','word_ft_v3/topology.py','word_ft_v3/__init__.py')},
            'package_manifest_sha256':args.manifest_sha,'contract_sha256':args.contract_sha,
            'terminal_evidence_sha256':sha(args.terminal_evidence) if args.terminal_evidence else None,
            'case':args.case,'synthetic_engineering_only':True}
        identity['max_attempts_per_update']=args.max_attempts_per_update
        write_new(output/'identity.json',identity)
        ticket=R.create_ticket(probe,locks,60000,2048,256,identity);write_new(output/'ticket.json',ticket)
        ticket_sha=sha(output/'ticket.json');nonce=secrets.token_hex(24)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(args.expected_uuids),PTN_ENGINEERING_NONCE=nonce,
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env['LD_LIBRARY_PATH']=str(S.DECODER.parent)+((':'+env['LD_LIBRARY_PATH']) if env.get('LD_LIBRARY_PATH') else '')
        cmd=[sys.executable,'-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2','--max-restarts=0',
            str(Path(__file__).resolve()),'--worker','--package',str(S.PACKAGE),'--manifest-sha',args.manifest_sha,'--contract-sha',args.contract_sha,
            '--output',str(output),'--case',args.case,'--ticket',str(output/'ticket.json'),'--ticket-sha',ticket_sha,'--identity',str(output/'identity.json')]
        cmd+=['--max-attempts-per-update',str(args.max_attempts_per_update)]
        if args.terminal_evidence:cmd+=['--terminal-evidence',args.terminal_evidence]
        write_new(output/'launch.json',{'command':cmd,'identity':identity,'probe':probe,'locks':locks.evidence,'parent_CUDA_initialized':False})
        child=None;error=None;started=None;ack=False;start=time.monotonic();rank_groups={}
        def interrupted(signum,frame):raise KeyboardInterrupt('engineering parent signal '+str(signum))
        handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            with (output/'launch.log').open('x') as log:
                child=subprocess.Popen(cmd,env=env,cwd=S.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=R.process_start_time(child.pid)
                need(os.getpgid(child.pid)==child.pid and os.getsid(child.pid)==child.pid,'own torchrun session differs')
                while child.poll() is None:
                    need(time.monotonic()-start<args.max_seconds,'engineering bounded timeout')
                    if not ack and all((output/('ready-rank%d.json'%i)).is_file() for i in range(2)):
                        for i in range(2):
                            row=json.loads((output/('ready-rank%d.json'%i)).read_text())
                            need(row['rank']==i and row['ticket_sha256']==ticket_sha and row['nonce']==nonce,'worker ready pin differs')
                            actual=process_proof(row['process']['pid'],nonce)
                            need(actual==row['process'] and actual['start']>=started and R.is_descendant(actual['pid'],child.pid),
                                 'worker pid/session/start/nonce not independently proved')
                            rank_groups[actual['pgid']]=actual
                        write_new(output/'proved-rank-groups.json',list(rank_groups.values()))
                        snapshot=R.capture_snapshot();devices=R.check_idle(snapshot,[0,1],60000)
                        need(R.stable_identity(devices)==R.stable_identity(probe['devices']),'pre-CUDA GPU identity changed')
                        write_new(output/'cuda-ack.json',{'ticket_sha256':ticket_sha,'pre_CUDA_snapshot':snapshot});ack=True
                    time.sleep(.2)
                need(child.returncode==0 and ack,'own engineering torchrun failed')
                need(not group_members(child.pid,started,nonce),'own workers remain after torchrun')
                for i in range(2):need(json.loads((output/('result-rank%d.json'%i)).read_text())['passed'] is True,'rank result failed')
        except BaseException:
            error=traceback.format_exc()
        finally:
            for sig in handlers:signal.signal(sig,signal.SIG_IGN)
            try:
                if child is not None:cleanup_all(child,started,nonce,output,rank_groups)
                # 两种可处理信号在清理和归还证明期间保持忽略，不能提前离开设备锁。
                while True:
                    try:returned=R.double_probe([0,1],60000);break
                    except Exception:
                        try:write_new(output/'resource-return-unproven.json',{'error':traceback.format_exc(),'locks_retained':True})
                        except Exception:pass
                        time.sleep(5)  # A busy/unknown PID cannot cause lock release or be killed.
                need(R.stable_identity(returned['devices'])==R.stable_identity(probe['devices']),'resource return UUIDs differ')
                write_new(output/'parent-result.json',{'passed':error is None,'error':error,'resource_returned':True,'return_probe':returned,
                    'identity':identity,'seconds':time.monotonic()-start,'formal_W8_admission':False,'synthetic_engineering_only':True})
            finally:
                for sig,handler in handlers.items():signal.signal(sig,handler)
        need(error is None,'engineering case failed; preserved original trace in parent-result.json')


def main(argv=None):
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','manifest-sha','contract-sha','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--case',choices=('finite','injected-failure'),required=True)
    p.add_argument('--lock-root');p.add_argument('--expected-uuids',nargs=2);p.add_argument('--max-seconds',type=int,default=1200)
    p.add_argument('--max-attempts-per-update',type=int,default=8)
    p.add_argument('--terminal-evidence');p.add_argument('--worker',action='store_true')
    for name in ('ticket','ticket-sha','identity'):p.add_argument('--'+name)
    a=p.parse_args(argv)
    if a.worker:need(a.ticket and a.ticket_sha and a.identity,'worker ticket arguments missing');worker(a)
    else:need(a.lock_root and a.expected_uuids,'parent requires shared lock root and exact UUIDs');parent(a)


if __name__=='__main__':main()
