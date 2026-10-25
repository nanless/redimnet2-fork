#!/usr/bin/env python3
"""只读复核工程保存文件；先实际SHA，再CPU weights_only/mmap；绝不前向或CUDA。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import time
import traceback

APPROVED_RUNNER_SHA='54e089818583d69bf1c5b93c45363b6603ab17b5d8124b24282ed78bcdccb9e4'


def need(ok,msg):
    if not ok:raise ValueError(msg)


def validate_runner(identity,pin):
    need(isinstance(pin,str)and re.fullmatch('[0-9a-f]{64}',pin)is not None,'full external runner SHA required')
    need(pin==APPROVED_RUNNER_SHA,'runner is not the independently approved engineering implementation')
    need(identity.get('runner_sha256')==pin,'case runner differs from external approved pin')


def physical(value):
    p=Path(value);need(p.is_absolute(),'absolute physical path required')
    q=Path(p.anchor)
    for part in p.parts[1:]:
        need(part not in ('.','..'),'unsafe path');q=q/part;need(not q.is_symlink(),'symlink refused')
    return p


def snap(p):
    s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def hash_file(path,rate=32*1024**2):
    p=physical(path);before=snap(p);need(p.is_file(),'regular input required')
    h=hashlib.sha256();start=time.monotonic();total=0
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block);total+=len(block)
            if rate:time.sleep(max(0,total/rate-(time.monotonic()-start)))
    need(snap(p)==before,'input changed while hashing')
    return {'sha256':h.hexdigest(),'bytes':before[2],'stat':before}


def read_json(path):
    p=physical(path);before=snap(p);need(before[2]<=16*1024**2,'JSON evidence too large')
    raw=p.read_bytes();need(snap(p)==before,'JSON changed')
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    def bad(x):raise ValueError('nonfinite JSON')
    def number(x):
        value=float(x);need(math.isfinite(value),'nonfinite JSON exponent');return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=number)


def child(base,rel):
    p=PurePosixPath(rel);need(str(p)==rel and not p.is_absolute()and '..'not in p.parts,'unsafe member')
    return physical(Path(base)/rel)


def inspect_package(path,pin,rate):
    path=physical(path);actual=hash_file(path/'bundle_files.json',rate);need(actual['sha256']==pin,'package index differs')
    bundle=read_json(path/'bundle_files.json')
    observed={str(p.relative_to(path))for p in path.rglob('*')if p.is_file()or p.is_symlink()}
    need(observed==set(bundle)|{'bundle_files.json'},'package membership differs')
    for rel,digest in bundle.items():need(hash_file(child(path,rel),rate)['sha256']==digest,'package member drift: '+rel)
    source=read_json(path/'source_manifest.json');delta=read_json(path/'source_delta.json');contract=read_json(path/'resume-contract.json')
    need(len(source)==190 and len({r['relative']for r in source})==190,'source190 denominator differs')
    for row in source:
        p=child(path/'source',row['relative']);s=p.stat()
        need(bundle['source/'+row['relative']]==row['sha256']and s.st_size==row['bytes']and stat.S_IMODE(s.st_mode)==row['mode'],'source metadata differs')
    need(hash_file(path/'source_manifest.json',rate)['sha256']==contract['source_manifest_sha256'],'source ledger not contract bound')
    need(hash_file(path/'source_delta.json',rate)['sha256']==contract['source_delta_sha256'],'source delta not contract bound')
    return {'path':str(path),'bundle_sha256':pin,'source_manifest_sha256':contract['source_manifest_sha256'],
        'source_delta_sha256':contract['source_delta_sha256'],'source190':source,'delta':delta,'bundle':bundle,
        'contract_sha256':bundle['resume-contract.json'],'runtime_files':contract['runtime_files']}


def validate_rng(value,torch,rank):
    need(isinstance(value,dict)and set(value)=={'python','numpy','torch_cpu','torch_cuda'},'RNG fields incomplete')
    random.Random().setstate(value['python'])
    n=value['numpy'];need(n['name']=='MT19937'and len(n['keys'])==624 and all(type(x)is int and 0<=x<2**32 for x in n['keys'])
        and type(n['position'])is int and 0<=n['position']<=624 and n['has_gauss']in(0,1)and math.isfinite(n['cached_gaussian']),'NumPy RNG invalid')
    for t in [value['torch_cpu'],*value['torch_cuda'].values()]:
        need(torch.is_tensor(t)and t.device.type=='cpu'and t.dtype==torch.uint8 and t.ndim==1 and t.numel()>=16,'Torch RNG tensor invalid')
    try:torch.Generator(device='cpu').set_state(value['torch_cpu'])
    except RuntimeError as error:raise ValueError('CPU RNG cannot be restored by an independent generator')from error
    need(set(value['torch_cuda'])==({str(rank)}if rank is not None else set()),'rank CUDA RNG membership differs')


def validate_state(model,opt,scaler,names,torch,*,counts=(823,584),groups=(583,2),head=(357822,192)):
    need(isinstance(model,dict)and len(model)==counts[0]and tuple(model['projection.weight'].shape)==head,'model geometry differs')
    need(all(torch.is_tensor(v)and v.device.type=='cpu'and (not(v.is_floating_point()or v.is_complex())or bool(torch.isfinite(v).all()))for v in model.values()),'model tensor missing/nonfinite/nonCPU')
    need(len(opt['state'])==counts[1]and [len(g['params'])for g in opt['param_groups']]==list(groups),'optimizer geometry differs')
    need([len(g)for g in names]==list(groups)and len({n for g in names for n in g})==sum(groups),'optimizer ordered names differ')
    lookup={pid:name for group,ns in zip(opt['param_groups'],names)for pid,name in zip(group['params'],ns)}
    need(len(lookup)==sum(groups)and set(lookup.values())<=set(model),'optimizer model binding incomplete')
    for pid,entry in opt['state'].items():
        need(pid in lookup and set(entry)=={'momentum_buffer'},'optimizer state fields differ')
        m=entry['momentum_buffer'];p=model[lookup[pid]]
        need(torch.is_tensor(m)and m.shape==p.shape and m.dtype==p.dtype and bool(torch.isfinite(m).all()),'momentum geometry/nonfinite')
    need(set(scaler)=={'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}and
        all(type(v)in(int,float)and math.isfinite(v)for v in scaler.values())and scaler['scale']>0,'scaler incomplete/nonfinite')
    return {'model_tensor_count':len(model),'optimizer_state_count':len(opt['state']),
        'ordered_parameter_count':sum(groups),'scaler':scaler,'head_shape':list(head)}


def load_bound(path,pin,torch):
    need(snap(path)==pin['stat'],'binary changed after initial stdlib hash')
    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    need(snap(path)==pin['stat'],'binary changed during mmap load')
    return value


def validate_buffers(model,buffers,torch):
    need(isinstance(buffers,dict)and buffers,'registered buffers missing')
    for name,v in buffers.items():
        need(torch.is_tensor(v)and v.device.type=='cpu'and (not(v.is_floating_point()or v.is_complex())or bool(torch.isfinite(v).all())),'buffer invalid')
        if name in model:need(torch.equal(model[name],v),'model/buffer value binding differs')
    return {n:int(v)for n,v in buffers.items()if n.endswith('num_batches_tracked')}


def inspect(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU inspector must explicitly hide CUDA before import')
    rate=args.hash_mib_per_second*1024**2
    packages=[inspect_package(args.candidate2,args.candidate2_manifest_sha,rate),inspect_package(args.candidate3,args.candidate3_manifest_sha,rate)]
    a,b=packages;need(a['source190']==b['source190']and a['delta']==b['delta'],'candidate model source/delta changed')
    runtime_changes={name:{'candidate2':a['runtime_files'].get(name),'candidate3':b['runtime_files'].get(name)}
        for name in set(a['runtime_files'])|set(b['runtime_files'])if a['runtime_files'].get(name)!=b['runtime_files'].get(name)}
    finite=physical(args.finite);injected=physical(args.injected);binaries=[];case_rows={}
    for case,root in [('finite',finite),('injected',injected)]:
        identity=read_json(root/'identity.json');parent=read_json(root/'parent-result.json')
        validate_runner(identity,args.runner_sha)
        need(parent['passed']is True and parent['error']is None and parent['resource_returned']is True and
            parent['identity']==identity and identity['package_manifest_sha256']==a['bundle_sha256']and identity['contract_sha256']==a['contract_sha256'],
            'actual case not bound to candidate2 or parent failed')
        snaps=parent['return_probe']['snapshots'];need(len(snaps)==2 and snaps[1]['monotonic']-snaps[0]['monotonic']>=4.98,'return probes not independently spaced')
        need(all(not s['compute_processes']and len(s['devices'])==2 and all(d['used_mib']<=256 for d in s['devices'])for s in snaps),'resource return not idle')
        admitted=read_json(root/'launch.json')['probe']['devices']
        keys=lambda ds:[{k:d[k]for k in ('index','uuid','driver_version','total_mib')}for d in ds]
        need(all(keys(s['devices'])==keys(admitted)for s in snaps),'returned physical devices differ from admission')
        case_rows[case]={'parent_seconds':parent['seconds'],'identity':identity,'return_probe':parent['return_probe'],'ranks':[]}
        if case=='finite':
            for rank in range(2):
                result=read_json(root/('result-rank%d.json'%rank));path=root/('step1-rank%d.pt'%rank)
                pin=hash_file(path,rate);need(pin['sha256']==result['step1_sha256'],'step1 actual file SHA differs')
                binaries.append((case,rank,path,pin,result,None))
        else:
            completes=list((root/'injected-failure').glob('*/complete.json'));need(len(completes)==2,'exactly2 failure bundles required')
            ranks=set()
            for p in completes:
                complete=read_json(p);rank=complete['rank'];need(type(rank)is int and rank in(0,1)and rank not in ranks,'failure rank duplicates');ranks.add(rank)
                path=p.parent/'payload.pt';pin=hash_file(path,rate)
                need(pin['sha256']==complete['payload_sha256']and pin['bytes']==complete['payload_bytes'],'actual payload size/SHA differs')
                coord=read_json(p.parent/'coordination.json');need(coord['coherent']is True and coord['all_local_payloads_published']is True and coord['all_rank_replay_complete']is True,'failure coordination incomplete')
                need([r['rank']for r in coord['participants']]==[0,1]and [r['numerical_finite']for r in coord['participants']]==[False,True]
                    and all(r['write_error']is None and r['artifact']['replay_complete']is True for r in coord['participants']),'bad/finite peer participation differs')
                binaries.append((case,rank,path,pin,complete,coord))
    # Every large input has been actually hashed before Torch is imported.
    import torch
    torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'CPU inspector inherited CUDA')
    for case,rank,path,pin,metadata,coord in binaries:
        value=load_bound(path,pin,torch)
        if case=='finite':
            need(value['schema']=='synthetic-B6-step-state/v1'and value['synthetic_engineering_only']is True,'wrong step state purpose')
            report=validate_state(value['model'],value['optimizer'],value['scaler'],value['optimizer_parameter_names'],torch)
            validate_rng(value['rng'],torch,rank);bn=validate_buffers(value['model'],value['buffers'],torch)
            need(bn and value['next_iteration']==metadata['step1'][-1]['absolute_iteration']+1,'saved next attempt clock differs')
            need(metadata['state_exact']is True and metadata['fresh_model_optimizer_DDP_reconstructed']is True and metadata['process_restart_tested']is False,'fresh reconstruction/exact result missing')
            for phase in ('step1','continuous_step2','restored_step2'):
                rows=metadata[phase];need(sum(r['actual_optimizer_step']for r in rows)==1 and rows[-1]['actual_optimizer_step']is True,'not exactly1 actual SGD per phase')
                for i,row in enumerate(rows):
                    name={'step1':'first','continuous_step2':'second','restored_step2':'replayed-second'}[phase]
                    need(read_json(finite/('attempt-%s-rank%d-%03d.json'%(name,rank,i)))==row,'attempt original differs from summary')
            left,right=metadata['continuous_step2'],metadata['restored_step2'];need(len(left)==len(right),'restore attempt count differs')
            for x,y in zip(left,right):
                for key in ('loss','actual_optimizer_step','absolute_iteration','scaler','gradient_tensors_nonfinite','BN_counter_deltas'):need(x[key]==y[key],'restore attempt comparison differs')
            need(value['scaler']==metadata['step1'][-1]['scaler'],'step1 saved scaler not actual update')
            report.update(BN_counter_count=len(bn),next_iteration=value['next_iteration'],
                full_step2_compare_scope='pinned runner internal full-state exact check; final tensors were not separately saved',result=metadata)
        else:
            need(value['schema']=='ptn-forward-failure/v1'and value['training_resume_checkpoint']is False and
                value['replay_complete']is True and value['replay_incomplete_reasons']==[] and value['pre_capture_error']is None,'forensic schema/capture incomplete')
            report=validate_state(value['model_pre_forward'],value['optimizer_state'],value['scaler_state'],value['optimizer_parameter_names'],torch)
            need(value['scaler_state']==read_json(injected/('restore-rank%d.json'%rank))['actual_scaler'],
                 'failure scaler changed before backward/update')
            need(len(value['model_post_forward'])==823,'post model missing')
            pre=validate_buffers(value['model_pre_forward'],value['buffers_pre_forward'],torch)
            post=validate_buffers(value['model_post_forward'],value['buffers_post_forward'],torch)
            need(pre and set(pre)==set(post)and all(post[k]==pre[k]+1 for k in pre),'forensic BN pre/post advance differs')
            for key in value['model_pre_forward']:
                if key not in value['buffers_pre_forward']:need(torch.equal(value['model_pre_forward'][key],value['model_post_forward'][key]),'parameters changed before backward')
            for key in ('rng_pre_forward','rng_at_failure_capture'):validate_rng(value[key],torch,rank)
            inputs=value['inputs'];wave=inputs['wav'];targets=inputs['targets']
            need(tuple(wave.shape)==(96,1,32240)and wave.dtype==torch.float32 and bool(torch.isfinite(wave).all()),'wave geometry/nonfinite')
            need(targets.shape==(96,)and targets.dtype==torch.int64 and int(targets.min())>=0 and int(targets.max())<357822,'targets invalid')
            need(torch.equal(value['wav'],wave)and torch.equal(value['targets'],targets)and
                torch.equal(value['original_batch']['wav'],wave)and torch.equal(value['original_batch']['label'],targets),'prepared/original input aliases differ')
            need(value['location']['rank']==rank and value['location']['world_size']==2 and value['saved_before_current_batch_backward_or_optimizer']is True,'forensic location/clock differs')
            flags=[bool(torch.isfinite(t).all())for t in(wave,value['embeds_at_failure'],value['loss_at_failure'])]
            need(flags==metadata['local_finite']==([True,False,False]if rank==0 else[True,True,True]),'actual failure tensor flags differ')
            synthetic=read_json(injected/('synthetic-input-rank%d.json'%rank))
            need(hashlib.sha256(wave.numpy().tobytes()).hexdigest()==synthetic['wave_float32_sha256'],'saved waveform differs from fixed synthetic input')
            report.update(BN_counter_count=len(pre),BN_each_advanced_by=1,actual_local_finite=flags,coordination_verified=True,
                prepared_wave_shape=list(wave.shape),RNG_before_and_after_structure_verified=True,
                CPU_rng_states_independent_set_state_tested=True,cuda_rng_restoration_tested=False,not_an_epoch_checkpoint=True)
        need(snap(path)==pin['stat'],'binary changed during full CPU inspection')
        report.update(rank=rank,path=str(path),actual_file=pin);case_rows[case]['ranks'].append(report);del value
    need(not torch.cuda.is_initialized(),'CPU inspection touched CUDA')
    return {'schema':'PTN-engineering-real-binary-readback/v1','passed':True,'cpu_only':True,'model_forward_or_update':False,
        'cuda_initialized':False,'candidate_source190_exactly_same':True,'candidate_source_delta_exactly_same':True,
        'candidate2':{k:v for k,v in a.items()if k not in('bundle','source190')},
        'candidate3':{k:v for k,v in b.items()if k not in('bundle','source190')},'runtime_differences':runtime_changes,
        'cases':case_rows,'formal_W8_admission':False,'original_fault_reproduced_or_fixed':False,
        'external_approved_runner_sha256':args.runner_sha,'CPU_rng_independent_set_state_tested':True,
        'cuda_rng_restoration_tested':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('candidate2','candidate2-manifest-sha','candidate3','candidate3-manifest-sha','finite','injected','report','runner-sha'):p.add_argument('--'+n,required=True)
    p.add_argument('--hash-mib-per-second',type=int,default=32);args=p.parse_args(argv)
    need(1<=args.hash_mib_per_second<=64,'bounded hash rate 1..64 MiB/s')
    out=physical(args.report);need(not out.exists()and not out.is_symlink()and out.parent.is_dir(),'new report in existing external directory required')
    for value in (args.candidate2,args.candidate3,args.finite,args.injected,str(Path(__file__).resolve().parent)):
        root=physical(value);need(out!=root and root not in out.parents and out not in root.parents,'report overlaps inputs/code')
    try:result=inspect(args);code=0
    except Exception:result={'schema':'PTN-engineering-real-binary-readback/v1','passed':False,'traceback':traceback.format_exc()};code=2
    tmp=out.with_name(out.name+'.part')
    with tmp.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(tmp,out);tmp.unlink();print(json.dumps({'passed':result['passed'],'report':str(out)}));return code


if __name__=='__main__':raise SystemExit(main())
