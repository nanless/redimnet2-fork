"""Matched duration fine-tuning. No padding, mask, speed perturbation or mixup."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import hashlib
from datetime import timedelta
import traceback
from contextlib import ExitStack
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from .data import (canonical,digest,file_sha,load_dataset,length_for_step,
                   global_batch_indices,speech_crop_plan,read_speech_crop,validate_output_path)

CHECKPOINT_SCHEMA='duration-training/v1'
OFFICIAL_SHA='287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868'

def seed_all(seed):
    random.seed(seed);np.random.seed(seed % (2**32));torch.default_generator.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.default_generators[torch.cuda.current_device()].manual_seed(seed)

def freeze_bn(module):
    """Freeze running statistics, retain affine gradients; call after train()."""
    for layer in module.modules():
        if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()

def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':{'device':torch.cuda.current_device(),'state':torch.cuda.get_rng_state()} if torch.cuda.is_initialized() else None}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if state['cuda']['device']!=torch.cuda.current_device():raise ValueError('resume CUDA device mapping mismatch')
        torch.cuda.set_rng_state(state['cuda']['state'],device=torch.cuda.current_device())

def atomic_torch_save(obj,path):
    p=Path(path);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    try:torch.save(obj,tmp);os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()

def load_official(path,expected_sha=OFFICIAL_SHA):
    if file_sha(path)!=expected_sha:raise ValueError('official checkpoint SHA mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(payload,dict) or 'model_config' not in payload or 'state_dict' not in payload:
        raise ValueError('official public {model_config,state_dict} required')
    from redimnet2.redimnet2 import ReDimNet2Wrap
    backbone=ReDimNet2Wrap(**payload['model_config'])
    backbone.load_state_dict(payload['state_dict'],strict=True)
    return backbone,copy.deepcopy(payload['model_config'])

class FineTuneModel(nn.Module):
    def __init__(self,backbone,head):
        super().__init__();self.backbone=backbone;self.head=head
    def forward(self,wave,targets):
        emb=self.backbone(wave)
        if isinstance(emb,tuple):emb=emb[0]
        with torch.autocast(device_type=emb.device.type,enabled=False):
            logits,loss=self.head(emb.float(),targets)
        return emb,logits,loss

def build_head(embedding_dim,speakers,seed,args):
    repo=Path(__file__).resolve().parents[1]
    apps=str(repo/'applications/speaker_rec')
    if apps not in sys.path:sys.path.insert(0,apps)
    from wespeaker_lite.models.projections import SphereFace2
    # Forked RNG context ensures identical head and unchanged global dropout RNG.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return SphereFace2(embedding_dim,speakers,**args)

def configuration_identity(config,dataset_id,world_size):
    return digest({'config':config,'dataset_id':dataset_id,'world_size':world_size})

def checkpoint_payload(model,optimizer,scaler,config,dataset_id,world_size,step,
                       exposure,rank_rngs,model_config,initial_head_sha,official_sha):
    return {'schema':CHECKPOINT_SCHEMA,'state_dict':model.backbone.state_dict(),
            'model_config':model_config,'head':model.head.state_dict(),
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'config':config,'dataset_id':dataset_id,'world_size':world_size,
            'identity':configuration_identity(config,dataset_id,world_size),
            'global_step':step,'sampler_global_position':step*config['batch_size']*world_size,
            'exposure':exposure,'rank_rngs':rank_rngs,'initial_head_sha256':initial_head_sha,
            'official_sha256':official_sha,'cold_optimizer_at_step0':True,'source_identity':source_identity()}

def restore_checkpoint(payload,model,optimizer,scaler,config,dataset_id,world_size,rank):
    if payload.get('schema')!=CHECKPOINT_SCHEMA:raise ValueError('unsupported checkpoint')
    if payload.get('source_identity')!=source_identity():raise ValueError('resume source identity mismatch')
    if payload.get('identity')!=configuration_identity(config,dataset_id,world_size):
        raise ValueError('resume config/data/world-size identity mismatch')
    if payload.get('official_sha256')!=config['official_sha256']:
        raise ValueError('resume official identity mismatch')
    if payload.get('world_size')!=world_size:raise ValueError('resume world-size metadata mismatch')
    step=payload['global_step']
    if type(step) is not int or not 0<=step<=config['total_steps']:
        raise ValueError('invalid resume global_step')
    if payload['sampler_global_position']!=step*config['batch_size']*world_size:
        raise ValueError('resume sampler position mismatch')
    if len(payload['rank_rngs'])!=world_size:raise ValueError('missing rank RNG')
    model.backbone.load_state_dict(payload['state_dict'],strict=True)
    model.head.load_state_dict(payload['head'],strict=True)
    optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
    restore_rng(payload['rank_rngs'][rank])
    return step,copy.deepcopy(payload['exposure'])

def collective_check(error):
    if dist.is_initialized():
        errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
    else:errors=[error]
    if any(errors):raise RuntimeError('rank failure: '+canonical(errors))

def source_identity():
    root=Path(__file__).resolve().parents[1]
    files=[root/'duration_ft/config_checks.py',root/'duration_ft/data.py',root/'duration_ft/train.py',root/'duration_ft/resume_guard.py',
           root/'applications/speaker_rec/wespeaker_lite/models/projections.py']
    files+=sorted((root/'redimnet2').rglob('*.py'))
    return {str(p.relative_to(root)):file_sha(p) for p in files if p.is_file()}

def validate_config(c):
    from duration_ft.config_checks import validate_numeric_hyperparameters
    validate_numeric_hyperparameters(c)
    required={'seed','head_seed','batch_size','total_steps','checkpoint_every','lengths','probabilities',
              'optimizer','head','official_sha256','bn_policy','amp','decoder'}
    if set(c)!=required:raise ValueError(f'config keys differ: {set(c)^required}')
    if c.get('official_sha256')!=OFFICIAL_SHA:raise ValueError('unreviewed official backbone')
    if c['bn_policy']!='frozen_statistics_trainable_affine':raise ValueError('unsupported BN policy')
    for name in ['batch_size','total_steps','checkpoint_every']:
        if type(c[name]) is not int or c[name]<1:raise ValueError(name)
    if max(c['lengths'])>5.0:raise ValueError('common pool budget is five seconds')
    length_for_step(0,c['seed'],c['lengths'],c['probabilities'])
    if c['head'].get('margin_type')!='C':raise ValueError('only reviewed SphereFace2 cosine margin supported')

def decoder_identity(expected):
    import soundfile as sf
    paths=set()
    maps=Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            if 'libsndfile' in line:
                path=line.split()[-1]
                if path.startswith('/'):paths.add(str(Path(path).resolve()))
    if len(paths)!=1:raise ValueError(f'decoder mapping must be unique: {paths}')
    path=next(iter(paths))
    identity={'soundfile_version':sf.__version__,'libsndfile_version':sf.__libsndfile_version__,
              'library_realpath':path,'library_sha256':file_sha(path)}
    if identity['libsndfile_version']!=expected['version'] or identity['library_sha256']!=expected['library_sha256']:
        raise ValueError('decoder version/SHA mismatch')
    if Path(expected['library_root']).resolve() not in Path(path).parents:
        raise ValueError('decoder loaded outside private library root')
    return identity

def _run_impl(config,dataset_path,official_path,output,device='cpu',resume=None,stop_after=None,
        gpu_memory_mib=None,cpu_threads=2,_resources=None):
    validate_config(config)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if world<1 or not 0<=rank<world or local_rank<0:
        raise ValueError('invalid distributed rank/world-size environment')
    if resume is not None:
        from .resume_guard import check_resume_output
        check_resume_output(output,resume,world_size=world,rank=rank)
    if device=='cuda' and os.environ.get('DURATION_FT_GPU_AUTHORIZED')!='1':
        raise RuntimeError('GPU execution not enabled; obtain resource approval first')
    if cpu_threads<1:raise ValueError('CPU threads must be positive')
    if device=='cuda':
        if gpu_memory_mib is None or gpu_memory_mib<=0:raise ValueError('positive GPU memory budget required')
        torch.cuda.set_device(local_rank)
        total=torch.cuda.get_device_properties(local_rank).total_memory
        fraction=gpu_memory_mib*1024**2/total
        if not 0<fraction<1:raise ValueError('GPU budget must be below total memory')
        torch.cuda.set_per_process_memory_fraction(fraction,local_rank)
    target=torch.device('cuda',local_rank) if device=='cuda' else torch.device('cpu')
    torch.set_num_threads(cpu_threads)
    decoder=decoder_identity(config['decoder'])
    if world>1:dist.init_process_group('nccl' if device=='cuda' else 'gloo',timeout=timedelta(seconds=60))
    root=validate_output_path(output,Path(__file__).resolve().parents[1])
    data=load_dataset(dataset_path,verify=True);records=data['records']
    if data['selection'].get('split')!='train' or any(r.get('split')!='train' for r in records):
        raise ValueError('heldout data cannot be used for training')
    if rank==0:
        root.mkdir(parents=True,exist_ok=resume is not None)
        if resume is None and any(root.iterdir()):raise ValueError('fresh output must be empty')
    if world>1:dist.barrier()
    # Each rank locks only its own evidence stream; the root once-claim excludes a second launch.
    lock=None
    if rank==0:
        import fcntl
        lock=_resources.enter_context(open(root/'.training.lock','a+'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume is not None:
            check_resume_output(output,resume,world_size=world,rank=rank,writer_lock_held=True)
    seed_all(config['seed']+rank)
    backbone,model_config=load_official(official_path,config['official_sha256'])
    embedding_dim=int(model_config.get('embed_dim',192))
    head=build_head(embedding_dim,len(data['speaker_map']),config['head_seed'],config['head'])
    initial_head_sha=state_digest(head.state_dict())
    model=FineTuneModel(backbone,head).to(target)
    optimizer=torch.optim.SGD(model.parameters(),**config['optimizer'])
    scaler=torch.amp.GradScaler('cuda',enabled=bool(config['amp'] and device=='cuda'))
    step=0;exposure={'samples':0,'audio_seconds':0.0,'seconds_by_duration':{},'samples_by_duration':{}}
    if resume:
        payload=torch.load(resume,map_location='cpu',weights_only=False)
        if digest(payload['model_config'])!=digest(model_config):raise ValueError('resume model config mismatch')
        step,exposure=restore_checkpoint(payload,model,optimizer,scaler,config,data['dataset_id'],world,rank)
        if initial_head_sha!=payload['initial_head_sha256']:raise ValueError('initial head identity mismatch')
    if world>1:
        model_ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank] if device=='cuda' else None,
                                                          broadcast_buffers=False)
    else:model_ddp=model
    model.train();freeze_bn(model)
    identity={'dataset_id':data['dataset_id'],'config':config,'world_size':world,'model_config':model_config,
              'initial_head_sha256':initial_head_sha,'official_sha256':config['official_sha256'],
              'source':str(Path(__file__).resolve()),'resume':str(resume) if resume else None,
              'start_step':step,'rank':rank,'device':str(target),'decoder':decoder,'gpu_memory_mib':gpu_memory_mib,'cpu_threads':cpu_threads,'optimizer_state_entries_before_first_step':len(optimizer.state),
              'scaler_before_first_step':scaler.state_dict(),'BN_statistics':'fixed','loss':'SphereFace2',
              'transform':'VAD speech-axis concatenation; no wrap/pad/speed/mixup/augmentation'}
    (root/f'identity_rank{rank}_start_step_{step}.json').write_text(json.dumps(identity,indent=2,ensure_ascii=False)+'\n')
    if rank==0:
        (root/f'identity_start_step_{step}.json').write_text(json.dumps(identity,indent=2,ensure_ascii=False)+'\n')
    evidence=open(root/f'exposure_rank{rank}.jsonl','a',buffering=1)
    limit=config['total_steps'] if stop_after is None else min(stop_after,config['total_steps'])
    try:
        while step<limit:
            seconds=length_for_step(step,config['seed'],config['lengths'],config['probabilities'])
            if world>1:
                seen=[None]*world;dist.all_gather_object(seen,seconds)
                if any(s!=seconds for s in seen):raise RuntimeError('rank duration disagreement')
            indices=global_batch_indices(len(records),step,config['batch_size'],world,config['seed'])
            lo=rank*config['batch_size'];local=indices[lo:lo+config['batch_size']]
            plans=[];waves=[];labels=[];keys=[]
            decode_error=None
            try:
                for j,index in enumerate(local):
                    record=records[index];position=step*world*config['batch_size']+lo+j
                    plan=speech_crop_plan(record,seconds,config['seed'],position)
                    wave=read_speech_crop(record,plan)
                    plans.append(plan);waves.append(wave);labels.append(data['speaker_map'][record['speaker']]);keys.append(record['key'])
            except Exception:
                decode_error=traceback.format_exc()
                (root/f'failure_rank{rank}_step{step}.txt').write_text(decode_error)
            collective_check(decode_error)
            wave=torch.from_numpy(np.stack(waves)).to(target)
            labels=torch.tensor(labels,dtype=torch.long,device=target)
            optimizer.zero_grad(set_to_none=True)
            start=time.monotonic()
            with torch.autocast(device_type=target.type,enabled=scaler.is_enabled()):
                emb,logits,loss=model_ddp(wave,labels)
            collective_check('nonfinite forward' if not torch.isfinite(loss) or not torch.isfinite(emb).all() else None)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            collective_check('nonfinite gradient; update rejected' if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()) else None)
            scaler.step(optimizer);scaler.update()
            collective_check('nonfinite parameter after update' if any(
                not torch.isfinite(p).all() for p in model.parameters()) else None)
            step+=1;count=world*config['batch_size'];key=str(seconds)
            exposure['samples']+=count;exposure['audio_seconds']+=count*seconds
            exposure['seconds_by_duration'][key]=exposure['seconds_by_duration'].get(key,0)+count*seconds
            exposure['samples_by_duration'][key]=exposure['samples_by_duration'].get(key,0)+count
            evidence.write(json.dumps({'step':step,'rank':rank,'seconds':seconds,'samples':len(local),
                 'keys':keys,'labels':labels.cpu().tolist(),'crop_plans':plans,'loss':loss.item(),
                 'accuracy':float((logits.argmax(1)==labels).float().mean().item()),'margin':model.head.margin,
                 'optimizer_update':True,'elapsed_seconds':time.monotonic()-start,'exposure_global':exposure},ensure_ascii=False)+'\n')
            if step%config['checkpoint_every']==0 or step==limit:
                rng=capture_rng()
                if world>1:
                    rngs=[None]*world;dist.all_gather_object(rngs,rng)
                else:rngs=[rng]
                if rank==0:
                    payload=checkpoint_payload(model,optimizer,scaler,config,data['dataset_id'],world,step,exposure,
                                               rngs,model_config,initial_head_sha,config['official_sha256'])
                    atomic_torch_save(payload,root/f'train_step_{step}.pt')
                    atomic_torch_save({'model_config':model_config,'state_dict':model.backbone.state_dict()},root/f'embedding_step_{step}.pt')
                if world>1:dist.barrier()
    finally:
        evidence.close()
        if lock:lock.close()
        if world>1:dist.destroy_process_group()
    return {'global_step':step,'exposure':exposure,'initial_head_sha256':initial_head_sha}


def run(config,dataset_path,official_path,output,device='cpu',resume=None,stop_after=None,
        gpu_memory_mib=None,cpu_threads=2):
    with ExitStack() as resources:
        try:
            return _run_impl(config,dataset_path,official_path,output,device,resume,stop_after,
                             gpu_memory_mib,cpu_threads,_resources=resources)
        finally:
            if dist.is_initialized():dist.destroy_process_group()
