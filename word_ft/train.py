"""Whole lexical units: individual true-length forwards, equal-item accumulation.

No VAD concatenation, time crop, padding, wrapping, speed change or mixup.
Only complete optimizer boundaries are resumable; no partial gradients are saved.
"""
from __future__ import annotations
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from functools import lru_cache
import copy
import fcntl
import json
import os
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from duration_ft.data import canonical, digest, file_sha, validate_output_path
from duration_ft.train import (OFFICIAL_SHA, FineTuneModel, seed_all, freeze_bn,
    state_digest, capture_rng, restore_rng, atomic_torch_save, load_official,
    build_head, decoder_identity, collective_check)
from . import schema

CHECKPOINT_SCHEMA = 'word-training/v1'


def source_identity():
    root = Path(__file__).resolve().parents[1]
    files = [root/'word_ft/train.py', root/'word_ft/schema.py',
        root/'duration_ft/config_checks.py',root/'duration_ft/train.py', root/'duration_ft/data.py',
        root/'applications/speaker_rec/wespeaker_lite/models/projections.py']
    files += sorted((root/'redimnet2').rglob('*.py'))
    return {str(p.relative_to(root)): file_sha(p) for p in files if p.is_file()}


def validate_config(c):
    from duration_ft.config_checks import validate_numeric_hyperparameters
    validate_numeric_hyperparameters(c)
    required = {'seed','head_seed','items_per_rank','total_steps','checkpoint_every',
        'optimizer','head','official_sha256','bn_policy','amp','decoder','allow_candidates','sampling'}
    optional={'unit_counts','unit_probabilities'}
    if required-set(c) or set(c)-(required|optional):
        raise ValueError(f'config keys differ: {set(c)^required}')
    if c['sampling'] not in {'individual_units','matched_prefix_family'}:
        raise ValueError('unknown sampling policy')
    if c['sampling']=='matched_prefix_family':
        counts=c.get('unit_counts');probs=c.get('unit_probabilities')
        if not isinstance(counts,list) or len(set(counts))!=len(counts) or not counts or any(type(x) is not int or x not in (1,2,3,6) for x in counts):
            raise ValueError('invalid family unit counts')
        if not isinstance(probs,list) or len(probs)!=len(counts) or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in probs) or not math.isclose(sum(probs),1,abs_tol=1e-12):
            raise ValueError('invalid family count probabilities')
    elif optional&set(c):
        raise ValueError('individual-unit sampling cannot carry family count options')
    for name in ('seed','head_seed','items_per_rank','total_steps','checkpoint_every'):
        if type(c[name]) is not int or c[name] < (0 if name in ('seed','head_seed') else 1):
            raise ValueError(f'invalid {name}')
    if type(c['allow_candidates']) is not bool or type(c['amp']) is not bool:
        raise ValueError('policy flags must be boolean')
    if c['bn_policy'] != 'frozen_statistics_trainable_affine':
        raise ValueError('unsupported BN policy')
    if c['head'].get('margin_type') != 'C' or c['head'].get('dynamic_margin') is not False:
        raise ValueError('fixed cosine SphereFace2 margin required')
    if c['official_sha256'] != OFFICIAL_SHA:
        raise ValueError('unreviewed official backbone')


@lru_cache(maxsize=2)
def _permutation(count, epoch, seed):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count)


def step_indices(count, step, k, world, seed):
    from duration_ft.data import balanced_batch_indices
    # Retain this module's permutation seed while fixing epoch-boundary repeats.
    def order(n,seed_value,cycle):return _permutation(n,cycle,seed_value)
    return balanced_batch_indices(count,step,k,world,seed,order)


def build_family_index(records):
    families={}
    for index,r in enumerate(records):
        family=r.get('family_id')
        if not isinstance(family,str) or not family:raise ValueError('missing family_id')
        group=families.setdefault(family,{})
        if r['n_units'] in group:raise ValueError('duplicate family unit count')
        group[r['n_units']]=index
    identity_fields=['speaker_id','audio_sha256','audio_realpath','source_utterance_id',
        'start_sample','sample_rate','source_num_samples','source_channels','language','counting_policy']
    result=[]
    for family,group in sorted(families.items()):
        if set(group)!={1,2,3,6}:raise ValueError('family must cover exactly 1/2/3/6 units')
        long=records[group[6]];previous=long['start_sample']
        for n in [1,2,3,6]:
            r=records[group[n]]
            if any(r.get(key)!=long.get(key) for key in identity_fields):
                raise ValueError('family source/speaker/anchor/policy mismatch')
            if r['unit_texts']!=long['unit_texts'][:n]:raise ValueError('family text is not a prefix')
            if r['end_sample']<=previous:raise ValueError('family ends must strictly increase')
            previous=r['end_sample']
        result.append({'family_id':family,'indices':group})
    return result


def count_for_position(config,position):
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],0xC0FFEE,position]))
    return int(rng.choice(config['unit_counts'],p=config['unit_probabilities']))


def select_step(records,families,config,step,world):
    k=config['items_per_rank']
    if config['sampling']=='individual_units':
        return step_indices(len(records),step,k,world,config['seed'])
    selected=step_indices(len(families),step,k,world,config['seed'])
    return [families[i]['indices'][count_for_position(config,step*k*world+j)] for j,i in enumerate(selected)]


def predicted_vad_seconds(r):
    vad=r.get('vad')
    if vad is None:return None
    if not isinstance(vad,dict):raise ValueError('VAD evidence must be a dictionary')
    fields={'segments_source_samples','source_sha256'}
    if not fields&set(vad):return None
    if not fields<=set(vad) or vad['source_sha256']!=r['audio_sha256']:
        raise ValueError('VAD source identity differs')
    previous=0;covered=0;segments=vad['segments_source_samples']
    if not isinstance(segments,list):raise ValueError('VAD segments must be a list')
    for segment in segments:
        if not isinstance(segment,(list,tuple)) or len(segment)!=2 or any(type(x) is not int for x in segment):
            raise ValueError('invalid VAD interval')
        a,b=segment
        if not previous<=a<b<=r['source_num_samples']:raise ValueError('invalid/overlapping VAD geometry')
        covered+=max(0,min(b,r['end_sample'])-max(a,r['start_sample']));previous=b
    return covered/r['sample_rate']


def experiment_identity(config, dataset_id, world):
    return digest({'config':config,'dataset_id':dataset_id,'world_size':world})


def validate_resume_target(output, resume, world, rank, writer_lock_held=False):
    """CPU precheck, then repeat under the actual writer lock (TOCTOU closure)."""
    root=Path(output).resolve(); path=Path(resume).resolve()
    if path.parent != root or not path.is_file():
        raise ValueError('resume must be a checkpoint in this output')
    if type(world) is not int or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('invalid resume rank/world')
    probe=None
    try:
        if rank==0 and not writer_lock_held:
            probe=open(root/'.training.lock','r')
            fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        step=payload.get('global_step')
        if payload.get('schema') != CHECKPOINT_SCHEMA or type(step) is not int or step < 1:
            raise ValueError('invalid word checkpoint')
        if payload.get('world_size') != world or payload.get('micro_cursor') != 0:
            raise ValueError('resume world-size/partial-step mismatch')
        if path.name != f'train_step_{step}.pt':
            raise ValueError('checkpoint filename/step mismatch')
        train={p.name for p in root.glob('train_step_*.pt')}
        embeddings={p.name.replace('embedding_','train_',1) for p in root.glob('embedding_step_*.pt')}
        if train != embeddings or not train:
            raise ValueError('checkpoint/public export set mismatch')
        steps=[]
        for name in train:
            try: steps.append(int(name.removeprefix('train_step_').removesuffix('.pt')))
            except ValueError as exc: raise ValueError('invalid checkpoint filename') from exc
        if max(steps) != step:
            raise ValueError('stale resume would overwrite newer output')
        if any(root.glob('*.tmp.*')) or any(root.glob('failure_*')):
            raise ValueError('incomplete/failed output requires explicit review')
        streams=[]
        for r in range(world):
            evidence=root/f'exposure_rank{r}.jsonl'
            raw=evidence.read_bytes()
            if not raw.endswith(b'\n'):
                raise ValueError('partial evidence line')
            rows=[json.loads(line) for line in raw.splitlines()]
            if len(rows)!=step or [x.get('step') for x in rows]!=list(range(1,step+1)):
                raise ValueError('evidence steps not unique/continuous')
            if any(x.get('rank')!=r or x.get('optimizer_update') is not True for x in rows):
                raise ValueError('invalid rank/update evidence')
            if rows[-1].get('exposure_global') != payload.get('exposure'):
                raise ValueError('evidence/checkpoint exposure mismatch')
            if (root/f'identity_rank{r}_start_step_{step}.json').exists():
                raise ValueError('resume attempt already claimed this checkpoint')
            streams.append(rows)
        # Verify every committed row, not only the final exposure counter.
        exposure=empty_exposure();k=payload['config']['items_per_rank']
        for j in range(step):
            for r in range(world):
                row=streams[r][j];items=row.get('items',[])
                if len(items)!=k or len(row.get('labels',[]))!=k or row.get('keys')!=[x.get('unit_id') for x in items]:
                    raise ValueError('incomplete item evidence')
                losses=row.get('item_losses',[])
                if len(losses)!=k or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in losses):
                    raise ValueError('invalid item loss evidence')
                if not math.isclose(row.get('loss',float('nan')),sum(losses)/k,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('mean item loss evidence mismatch')
                for z,item in enumerate(items):
                    schema.validate_unit(item,allow_candidates=payload['config']['allow_candidates'])
                    if item.get('sampler_global_position')!=j*world*k+r*k+z or item.get('input_samples')!=item['n_samples_16k'] or item.get('input_sample_rate')!=16000:
                        raise ValueError('item position/length evidence mismatch')
                    add_exposure(exposure,item)
            if any(streams[r][j]['exposure_global']!=exposure for r in range(world)):
                raise ValueError('committed exposure is inconsistent with item records')
        return payload
    finally:
        if probe is not None:probe.close()


def empty_exposure():
    return {'items':0,'audio_seconds':0.0,'units':0,'units_by_type':{},'items_by_evidence':{},
            'predicted_vad_seconds':0.0,'items_with_vad':0}


def add_exposure(exposure,r):
    exposure['items']+=1;exposure['units']+=r['n_units']
    exposure['audio_seconds']+=(r['end_sample']-r['start_sample'])/r['sample_rate']
    unit_type=r['language']+':'+r['counting_policy']
    exposure['units_by_type'][unit_type]=exposure['units_by_type'].get(unit_type,0)+r['n_units']
    level=r['evidence']
    exposure['items_by_evidence'][level]=exposure['items_by_evidence'].get(level,0)+1
    seconds=predicted_vad_seconds(r)
    if seconds is not None:
        exposure['predicted_vad_seconds']+=seconds;exposure['items_with_vad']+=1


def _read_record(record):
    # Schema owns policy/header/SHA verification; reading never changes the span.
    wave = schema.read_unit(record)
    if not isinstance(wave,np.ndarray) or wave.ndim != 1 or wave.dtype != np.float32:
        raise ValueError('read_unit must return float32 mono waveform')
    if len(wave) != record['n_samples_16k'] or not np.isfinite(wave).all():
        raise ValueError('unit waveform length/finite mismatch')
    return wave


def item_evidence(record, position):
    return dict(copy.deepcopy(record), sampler_global_position=position,
                input_samples=record['n_samples_16k'], input_sample_rate=16000,
                predicted_vad_seconds=predicted_vad_seconds(record))


def accumulate_step(model_ddp, model, optimizer, scaler, waves, labels, device):
    """Same K on every rank. DDP means K-normalized gradients across ranks."""
    k=len(waves)
    if k < 1 or len(labels)!=k:raise ValueError('empty/mismatched accumulation step')
    optimizer.zero_grad(set_to_none=True)
    losses=[]; correct=0
    for j,(wave,label) in enumerate(zip(waves,labels)):
        # no_sync must include forward, not merely backward.
        sync_context=model_ddp.no_sync() if j<k-1 and hasattr(model_ddp,'no_sync') else nullcontext()
        with sync_context:
            forward_error=None; emb=logits=loss=None
            try:
                with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
                    emb,logits,loss=model_ddp(torch.from_numpy(wave).unsqueeze(0).to(device),
                                            torch.tensor([label],dtype=torch.long,device=device))
                if loss.ndim != 0 or not torch.isfinite(loss) or not torch.isfinite(emb).all():
                    raise ValueError('nonfinite or non-scalar item forward')
            except Exception:forward_error=traceback.format_exc()
            collective_check(forward_error)
            losses.append(float(loss.detach()))
            correct+=int(logits.detach().argmax(1).item()==label)
            scaler.scale(loss/k).backward()
    scaler.unscale_(optimizer)
    collective_check('nonfinite gradient; no update' if any(
        p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()) else None)
    scaler.step(optimizer);scaler.update()
    collective_check('nonfinite parameter after update' if any(
        not torch.isfinite(p).all() for p in model.parameters()) else None)
    return {'item_losses':losses,'loss':sum(losses)/k,'accuracy':correct/k}


def checkpoint_payload(model,optimizer,scaler,config,dataset_id,world,step,exposure,
                       rngs,model_config,head_sha):
    return {'schema':CHECKPOINT_SCHEMA,'global_step':step,'micro_cursor':0,
        'state_dict':model.backbone.state_dict(),'head':model.head.state_dict(),
        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
        'model_config':model_config,'config':config,'dataset_id':dataset_id,
        'world_size':world,'identity':experiment_identity(config,dataset_id,world),
        'sampler_global_position':step*config['items_per_rank']*world,
        'exposure':copy.deepcopy(exposure),'rank_rngs':rngs,'initial_head_sha256':head_sha,
        'official_sha256':config['official_sha256'],'source_identity':source_identity(),
        'cold_optimizer_at_step0':True,'boundary_policy':'whole-contiguous-lexical-units'}


def restore_checkpoint(p,model,optimizer,scaler,config,dataset_id,world,rank,head_sha):
    if p.get('schema')!=CHECKPOINT_SCHEMA or p.get('source_identity')!=source_identity():
        raise ValueError('resume schema/source identity mismatch')
    if p.get('identity') != experiment_identity(config,dataset_id,world):
        raise ValueError('resume config/data/world-size identity mismatch')
    if p.get('initial_head_sha256')!=head_sha or p.get('official_sha256')!=config['official_sha256']:
        raise ValueError('resume initialization mismatch')
    step=p['global_step']
    if type(step) is not int or not 0<step<=config['total_steps'] or p.get('micro_cursor')!=0:
        raise ValueError('invalid/partial resume step')
    if p.get('sampler_global_position')!=step*world*config['items_per_rank'] or len(p['rank_rngs'])!=world:
        raise ValueError('resume order/RNG mismatch')
    model.backbone.load_state_dict(p['state_dict'],strict=True);model.head.load_state_dict(p['head'],strict=True)
    optimizer.load_state_dict(p['optimizer']);scaler.load_state_dict(p['scaler']);restore_rng(p['rank_rngs'][rank])
    return step,copy.deepcopy(p['exposure'])


def _run_impl(config,dataset_path,official_path,output,device='cpu',resume=None,
              stop_after=None,cpu_threads=2,gpu_memory_mib=None,resources=None):
    validate_config(config)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if world<1 or not 0<=rank<world or local_rank<0:raise ValueError('invalid rank/world environment')
    root=validate_output_path(output,Path(__file__).resolve().parents[1])
    if resume:validate_resume_target(root,resume,world,rank)
    # No GPU authorization has been granted for this word protocol.
    if device!='cpu':raise RuntimeError('word fine-tuning GPU execution is not authorized')
    if type(cpu_threads) is not int or cpu_threads<1:raise ValueError('CPU threads must be positive')
    if stop_after is not None and (type(stop_after) is not int or stop_after<1):raise ValueError('invalid stop-after')
    torch.set_num_threads(cpu_threads);target=torch.device('cpu')
    decoder=decoder_identity(config['decoder'])
    if world>1:dist.init_process_group('gloo',timeout=timedelta(seconds=60))
    data_error=None;data=None
    try:
        data=schema.load_dataset(dataset_path,verify=True,allow_candidates=config['allow_candidates'])
        if data.get('metadata',{}).get('split')!='train' or any(r.get('split')!='train' for r in data['records']):
            raise ValueError('only train split is allowed')
        families=build_family_index(data['records']) if config['sampling']=='matched_prefix_family' else None
        pool_size=len(families) if families is not None else len(data['records'])
        if pool_size<world*config['items_per_rank']:raise ValueError('insufficient items/families for global step')
        for record in data['records']:predicted_vad_seconds(record)
    except Exception:data_error=traceback.format_exc()
    collective_check(data_error)
    if world>1:
        identities=[None]*world;dist.all_gather_object(identities,data['dataset_id'])
        if len(set(identities))!=1:raise ValueError('rank dataset identities differ')
    records=data['records']
    if rank==0:
        root.mkdir(parents=True,exist_ok=resume is not None)
        if not resume and any(root.iterdir()):raise ValueError('fresh output must be empty')
    if world>1:dist.barrier()
    if rank==0:
        lock=resources.enter_context(open(root/'.training.lock','a+'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:validate_resume_target(root,resume,world,rank,writer_lock_held=True)
    seed_all(config['seed']+rank)
    backbone,model_config=load_official(official_path,config['official_sha256'])
    head=build_head(int(model_config.get('embed_dim',192)),len(data['speaker_map']),config['head_seed'],config['head'])
    head_sha=state_digest(head.state_dict());model=FineTuneModel(backbone,head).to(target)
    optimizer=torch.optim.SGD(model.parameters(),**config['optimizer'])
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    step=0;exposure=empty_exposure()
    if resume:
        p=torch.load(resume,map_location='cpu',weights_only=False)
        if digest(p['model_config'])!=digest(model_config):raise ValueError('resume model config mismatch')
        step,exposure=restore_checkpoint(p,model,optimizer,scaler,config,data['dataset_id'],world,rank,head_sha)
    model_ddp=torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False) if world>1 else model
    model.train();freeze_bn(model)
    identity={'dataset_id':data['dataset_id'],'config':config,'world_size':world,'rank':rank,
        'device':'cpu','start_step':step,'source_identity':source_identity(),'decoder':decoder,
        'initial_head_sha256':head_sha,'official_sha256':config['official_sha256'],
        'optimizer_state_entries_before_first_step':len(optimizer.state),'model_config':model_config,
        'transform':'original contiguous unit span including pauses; no VAD concatenation/crop/pad/wrap',
        'equal_weight':'one item; fixed items_per_rank per optimizer step',
        'sampling':config['sampling'],'common_family_pool':len(families) if families is not None else None,
        'candidate_policy':'silver model proposals permitted' if config['allow_candidates'] else 'reference/reviewed only'}
    (root/f'identity_rank{rank}_start_step_{step}.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
    evidence=resources.enter_context(open(root/f'exposure_rank{rank}.jsonl','a',buffering=1))
    limit=min(config['total_steps'],stop_after) if stop_after is not None else config['total_steps']
    while step<limit:
        indices=select_step(records,families,config,step,world)
        lo=rank*config['items_per_rank'];local=indices[lo:lo+config['items_per_rank']]
        waves=[];labels=[];items=[];error=None
        try:
            for j,index in enumerate(local):
                record=records[index];waves.append(_read_record(record))
                position=step*world*config['items_per_rank']+lo+j
                labels.append(data['speaker_map'][record['speaker_id']]);items.append(item_evidence(record,position))
        except Exception:
            error=traceback.format_exc();(root/f'failure_rank{rank}_step{step}.txt').write_text(error)
        collective_check(error)
        started=time.monotonic()
        try:stats=accumulate_step(model_ddp,model,optimizer,scaler,waves,labels,target)
        except Exception:
            (root/f'failure_rank{rank}_step{step}.txt').write_text(traceback.format_exc());raise
        # All ranks can derive the same actual exposure from immutable record spans.
        for index in indices:
            add_exposure(exposure,records[index])
        step+=1
        evidence.write(json.dumps({'step':step,'rank':rank,'items':items,'labels':labels,
            'keys':[x['unit_id'] for x in items],'optimizer_update':True,
            'elapsed_seconds':time.monotonic()-started,'exposure_global':exposure,**stats},ensure_ascii=False)+'\n')
        if step%config['checkpoint_every']==0 or step==limit:
            rng=capture_rng()
            if world>1:rngs=[None]*world;dist.all_gather_object(rngs,rng)
            else:rngs=[rng]
            if rank==0:
                payload=checkpoint_payload(model,optimizer,scaler,config,data['dataset_id'],world,step,exposure,rngs,model_config,head_sha)
                atomic_torch_save(payload,root/f'train_step_{step}.pt')
                atomic_torch_save({'model_config':model_config,'state_dict':model.backbone.state_dict()},root/f'embedding_step_{step}.pt')
            if world>1:dist.barrier()
    return {'global_step':step,'exposure':exposure,'initial_head_sha256':head_sha}


def run(config,dataset_path,official_path,output,device='cpu',resume=None,stop_after=None,
        cpu_threads=2,gpu_memory_mib=None):
    with ExitStack() as resources:
        try:
            return _run_impl(config,dataset_path,official_path,output,device,resume,
                             stop_after,cpu_threads,gpu_memory_mib,resources)
        finally:
            if dist.is_initialized():dist.destroy_process_group()
