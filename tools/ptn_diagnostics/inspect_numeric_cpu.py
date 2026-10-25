"""CPU-only SHA-bound checkpoint magnitude scan; never constructs or trains a model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import torch

torch.set_num_threads(2)
if torch.cuda.is_initialized():
    raise RuntimeError('CPU probe requires a fresh process without an existing CUDA context')


def identity(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


class Budget:
    def __init__(self, seconds, rate):
        self.started=time.monotonic();self.seconds=seconds;self.rate=rate
        self.phase=time.monotonic();self.bytes=0
    def check(self):
        if time.monotonic()-self.started>self.seconds:raise RuntimeError('bounded CPU scan time exceeded')
    def phase_start(self):self.phase=time.monotonic();self.bytes=0
    def read(self,n):
        self.bytes+=n;delay=self.bytes/self.rate-(time.monotonic()-self.phase)
        if delay>0:time.sleep(min(delay,1))
        self.check()


def blocks(tensor, elements=131072):
    # Do not reshape an entire noncontiguous checkpoint tensor into a new copy.
    if tensor.is_contiguous():
        flat=tensor.reshape(-1)
        for start in range(0,flat.numel(),elements):yield flat[start:start+elements]
    elif tensor.ndim==1:
        for start in range(0,tensor.numel(),elements):yield tensor[start:start+elements]
    else:
        for row in tensor:yield from blocks(row,elements)


def tensor_stats(name,tensor,budget):
    if tensor.device.type!='cpu' or tensor.layout!=torch.strided or tensor.is_complex():
        raise ValueError('unsupported non-CPU/sparse/complex tensor: '+name)
    count=nan=posinf=neginf=negative=0;lo=hi=None;scale=ssq=0.0
    for block in blocks(tensor.detach()):
        budget.check();x=block.to(torch.float64);finite=torch.isfinite(x)
        nan+=int(torch.isnan(x).sum());posinf+=int(torch.isposinf(x).sum());neginf+=int(torch.isneginf(x).sum())
        values=x[finite];count+=values.numel()
        if values.numel():
            a=float(values.min());b=float(values.max());lo=a if lo is None else min(lo,a);hi=b if hi is None else max(hi,b)
            negative+=int((values<0).sum());peak=max(abs(a),abs(b))
            next_scale=max(scale,peak)
            if next_scale:
                ssq=ssq*(scale/next_scale)**2+float((values/next_scale).square().sum())
                scale=next_scale
        budget.read(block.numel()*block.element_size())
    return {'name':name,'shape':list(tensor.shape),'stride':list(tensor.stride()),'dtype':str(tensor.dtype),
            'numel':tensor.numel(),'floating_point':tensor.is_floating_point(),'finite_count':count,'nan':nan,'posinf':posinf,'neginf':neginf,
            'max_abs':scale,'min':lo,'max':hi,'negative_finite_count':negative,
            'rms_finite_only':scale*math.sqrt(ssq/count) if count else None,
            'integer_counter_value':int(tensor.item()) if tensor.numel()==1 and not tensor.is_floating_point() else None}


def tensor_members(value,prefix='optimizer',depth=0):
    if depth>16:raise ValueError('optimizer container nesting exceeds bound')
    if torch.is_tensor(value):yield prefix,value
    elif isinstance(value,dict):
        for key,child in value.items():yield from tensor_members(child,prefix+'.'+str(key),depth+1)
    elif isinstance(value,(list,tuple)):
        for index,child in enumerate(value):yield from tensor_members(child,prefix+'.'+str(index),depth+1)


def load_bound(path,pin,budget):
    if not path.is_file() or len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('physical file and explicit SHA256 required')
    budget.phase_start();before=identity(path);digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block);budget.read(len(block))
    if digest.hexdigest()!=pin or identity(path)!=before:raise ValueError('input SHA or stable stat differs')
    value=torch.load(str(path),map_location='cpu',weights_only=True,mmap=True)
    if identity(path)!=before:raise ValueError('input changed during load')
    return value,before


def inspect_wave(payload,sample_rate,budget):
    if not isinstance(payload,dict) or not torch.is_tensor(payload.get('wav')):
        raise ValueError('saved wave must be a dictionary containing wav tensor')
    w=payload['wav'];keys=payload.get('utterance_keys')
    if w.device.type!='cpu' or w.layout!=torch.strided or w.ndim not in (2,3) or w.is_complex():
        raise ValueError('invalid CPU waveform tensor geometry/type')
    if w.numel()>32*1024**2 or not isinstance(keys,(list,tuple)) or len(keys)!=w.shape[0]:
        raise ValueError('waveform exceeds CPU bound or utterance-key denominator differs')
    rows=[];budget.phase_start()
    for i in range(w.shape[0]):
        x=w[i].detach().reshape(-1).to(torch.float64);ok=torch.isfinite(x);z=x[ok]
        row={'index':i,'utterance_key_sha256':hashlib.sha256(str(keys[i]).encode()).hexdigest(),
          'nan':int(torch.isnan(x).sum()),'posinf':int(torch.isposinf(x).sum()),'neginf':int(torch.isneginf(x).sum()),
          'finite_count':int(ok.sum()),'total_samples':x.numel(),'exact_zero_count':int((z==0).sum())}
        row.update(mean=float(z.mean()) if z.numel() else None,std_population=float(z.std(unbiased=False)) if z.numel() else None,
          rms_finite_only=float(z.square().mean().sqrt()) if z.numel() else None,
          min_finite=float(z.min()) if z.numel() else None,max_finite=float(z.max()) if z.numel() else None,
          max_abs_finite=float(z.abs().max()) if z.numel() else None,
          abs_ge1_fraction_finite=float((z.abs()>=1).double().mean()) if z.numel() else None)
        if sample_rate is not None:row['declared_duration_seconds']=x.numel()/sample_rate
        rows.append(row);budget.read(w[i].numel()*w.element_size())
    return {'shape':list(w.shape),'stride':list(w.stride()),'dtype':str(w.dtype),
       'storage_nbytes':w.untyped_storage().nbytes(),'logical_nbytes':w.numel()*w.element_size(),
       'contiguous':w.is_contiguous(),'utterance_count':len(keys),'rows':rows,
       'declared_sample_rate':sample_rate,'sample_rate_source':'explicit argument, not inferred from saved tensor',
       'waveform_normalized_clipped_resampled_or_augmented':False}


def inspect_checkpoint(payload,budget):
    if not isinstance(payload,dict):raise ValueError('checkpoint must be a tensor or training dict')
    if isinstance(payload.get('model'),dict):state=payload['model'];layout='training_dict'
    elif payload and all(torch.is_tensor(x) for x in payload.values()):state=payload;layout='bare_state_dict'
    else:raise ValueError('ambiguous checkpoint model payload')
    if not state or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):raise ValueError('invalid model state')
    budget.phase_start();model=[tensor_stats(k,v,budget)for k,v in state.items()]
    optimizer=[tensor_stats(k,v,budget)for k,v in tensor_members(payload.get('optimizer',{}))]
    bn_prefixes={name[:-len('running_var')]for name in state if name.endswith('running_var')}
    affine={prefix+tail for prefix in bn_prefixes for tail in ('weight','bias')}
    return {'layout':layout,'model_tensor_count':len(model),
        'model_nonfinite':[x['name']for x in model if x['finite_count']!=x['numel']],
        'optimizer_tensor_count':len(optimizer),'optimizer_nonfinite':[x['name']for x in optimizer if x['finite_count']!=x['numel']],
        'model_float_top_max_abs':sorted((x for x in model if x['floating_point']),key=lambda x:x['max_abs'],reverse=True)[:30],
        'optimizer_float_top_max_abs':sorted((x for x in optimizer if x['floating_point']),key=lambda x:x['max_abs'],reverse=True)[:20],
        'integer_state_counters':[x for x in model if not x['floating_point']],
        'running_var':[x for x in model if x['name'].endswith('running_var')],
        'norm_affine':[x for x in model if x['name'] in affine or ('norm' in x['name'].casefold() and x['name'].endswith(('weight','bias')))],
        'norm_affine_identification':'BN prefixes inferred from running_var plus norm-name heuristic; no model construction',
        'saved_epoch':payload.get('epoch') if layout=='training_dict'else None,
        'saved_scaler':payload.get('scaler') if layout=='training_dict'else None,
        'model_constructed':False,'optimizer_or_scheduler_applied':False}


def main(argv=None):
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--checkpoint-sha')
    p.add_argument('--wav',type=Path);p.add_argument('--wav-sha');p.add_argument('--sample-rate',type=int)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=900)
    p.add_argument('--rate-mib',type=float,default=16);a=p.parse_args(argv)
    sources=[('wave',a.wav,a.wav_sha),('checkpoint',a.checkpoint,a.checkpoint_sha)]
    if not any(x[1] is not None for x in sources):raise ValueError('at least one pinned input required')
    if any((path is None)!=(pin is None)for _,path,pin in sources):raise ValueError('each input requires its explicit SHA')
    if not(0<a.max_seconds<=1800 and 0<a.rate_mib<=16) or (a.sample_rate is not None and a.sample_rate<=0):raise ValueError('invalid bounded CPU budget/sample rate')
    # Resolve a known parent alias only after rejecting an existing/dangling
    # output leaf; otherwise a dangling symlink would create its target.
    if a.output.exists() or a.output.is_symlink():raise FileExistsError(a.output)
    out=a.output.resolve()
    for _,path,_ in sources:
        if path is not None and (path.resolve().parent==out.parent or path.resolve().parent in out.parents):
            raise ValueError('fresh output must be outside each evidence/checkpoint directory')
    budget=Budget(a.max_seconds,a.rate_mib*1024**2)
    result={'schema':'ptn-numeric-CPU-inspection/v1','torch_version':str(torch.__version__),
            'recorded_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'CPU_threads':2,'GPU_used':False,'CUDA_initialization_requested':False,'input_modified':False}
    for kind,path,pin in sources:
        if path is None:continue
        source=path.resolve();payload,before=load_bound(source,pin,budget)
        result[kind]=inspect_wave(payload,a.sample_rate,budget) if kind=='wave'else inspect_checkpoint(payload,budget)
        if identity(source)!=before:raise ValueError('input changed during numeric scan')
        result[kind].update(file_sha256=pin,file_stat=before,original_requested_path=str(path),resolved_path=str(source))
    result['elapsed_seconds']=time.monotonic()-budget.started
    data=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:f.write(data)
    print(json.dumps({'output':str(out),'GPU_used':False,'model_constructed':False}))
    return result


if __name__=='__main__':main()
