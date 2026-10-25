#!/usr/bin/env python3
"""Four bounded CPU workers, shared strict cache, then cache-only formal dev/test runs."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse,json,multiprocessing as mp,shutil,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from duration_ft.evaluate import EmbeddingCache,load_model,write_json,sha_file,source_identity
from word_ft import schema
from word_ft.evaluate import (PREPROCESS,code_identity,extract_unit,read_protocol,evaluate_words,calibrate_words)

_CONTEXT={}

def emit(value):print(json.dumps(dict(at_unix=time.time(),**value),ensure_ascii=False),flush=True)

def initialize(checkpoint,cache_dir,expected):
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if code_identity()!=expected['word_code'] or source_identity(Path(__file__).resolve().parents[1])!=expected['model_source']:
            raise ValueError('source changed before worker model load')
        model,identity=load_model(Path(checkpoint),'cpu')
        if identity['checkpoint_sha256']!=expected['checkpoint_sha256']:raise ValueError('checkpoint changed')
        _CONTEXT.update(model=model,identity=identity,cache=EmbeddingCache(Path(cache_dir)))
    except BaseException:
        _CONTEXT['error']=traceback.format_exc()


def one(unit):
    started=time.monotonic()
    if 'error' in _CONTEXT:return {'ok':False,'unit_id':unit['unit_id'],'traceback':_CONTEXT['error'],'worker_pid':os.getpid()}
    try:
        _,hit=extract_unit(_CONTEXT['model'],_CONTEXT['identity'],unit,_CONTEXT['cache'])
        return {'ok':True,'unit_id':unit['unit_id'],'cache_hit':hit,'n_samples_16k':unit['n_samples_16k'],
            'worker_pid':os.getpid(),'seconds':time.monotonic()-started,'model_identity':_CONTEXT['identity']}
    except BaseException:
        return {'ok':False,'unit_id':unit['unit_id'],'traceback':traceback.format_exc(),'worker_pid':os.getpid(),'seconds':time.monotonic()-started}


def evaluation_request(checkpoint,data,proto,split,calibration):
    return {'checkpoint_sha256':sha_file(checkpoint),'dataset_id':data['dataset_id'],'protocol_id':proto['protocol_id'],
        'split':split,'device':'cpu','gpu_memory_mib':None,'allow_exploratory':False,
        'word_code':code_identity(),'preprocess':PREPROCESS,'model_source':source_identity(Path(__file__).resolve().parents[1]),
        'calibration_sha256':sha_file(calibration) if calibration else None}


def publish(cache_dir,identity,data,proto,split,output,checkpoint,calibration=None):
    output=schema.protect_output(output)
    request=evaluation_request(checkpoint,data,proto,split,calibration)
    if output.exists():
        if not (output/'request_identity.json').is_file() or json.loads((output/'request_identity.json').read_text())!=request:
            raise ValueError('formal output exists with another identity')
    else:output.mkdir(parents=True)
    shared=EmbeddingCache(cache_dir);target=EmbeddingCache(output/'embedding-cache')
    copied=0
    for unit in data['records']:
        if unit['split']!=split:continue
        key={'schema':'word-complete-embedding/v1','model':identity,'preprocess':PREPROCESS,'word_code':code_identity(),
            'unit':schema.unit_identity(unit),'output_samples':unit['n_samples_16k']}
        if shared.get(key) is None:raise ValueError('missing verified prefill vector '+unit['unit_id'])
        dest=target.path(key);source=shared.path(key)
        if dest.exists():
            if dest.read_bytes()!=source.read_bytes():raise ValueError('different formal cache bytes')
        else:shutil.copyfile(source,dest)
        copied+=1
    write_json(output/'request_identity.json',request)
    emit({'phase':'formal_cache_published','split':split,'cache_records':copied,'output':str(output)})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','dataset','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args(argv)
    if not 1<=a.workers<=4:raise ValueError('1-4 CPU workers, one thread each')
    output=schema.protect_output(a.output)
    data,proto=read_protocol(a.dataset,a.protocol,verify_audio=True)
    request={'schema':'word-cpu-prefill/v1','checkpoint_sha256':sha_file(a.checkpoint),
        'dataset_id':data['dataset_id'],'protocol_id':proto['protocol_id'],'preprocess':PREPROCESS,
        'word_code':code_identity(),'model_source':source_identity(Path(__file__).resolve().parents[1]),'workers':a.workers,
        'cuda_visible_devices':'','threads_per_worker':1}
    if output.exists() and any(output.iterdir()):
        if not (output/'prefill_request.json').is_file() or json.loads((output/'prefill_request.json').read_text())!=request:
            raise ValueError('prefill output belongs to another immutable request')
    output.mkdir(parents=True,exist_ok=True);write_json(output/'prefill_request.json',request)
    results=[];identities={};shared=output/'shared-embedding-cache';shared.mkdir(exist_ok=True)
    emit({'phase':'prefill_start','units':len(data['records']),'workers':a.workers,'output':str(output),'request':request})
    pool=mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=(str(a.checkpoint),str(shared),request))
    try:
        iterator=pool.imap_unordered(one,data['records'],chunksize=1)
        last_progress=time.monotonic()
        while len(results)<len(data['records']):
            try:result=iterator.next(timeout=30)
            except mp.TimeoutError:
                if time.monotonic()-last_progress>600:raise TimeoutError('no completed CPU unit for 600 seconds; stop bounded worker pool')
                emit({'phase':'prefill_progress','completed':len(results),'planned':len(data['records']),'cache_hits':sum(r.get('cache_hit',False) for r in results)});continue
            last_progress=time.monotonic()
            identity=result.pop('model_identity',None)
            if identity is not None:identities[str(result['worker_pid'])]=identity
            results.append(result)
            emit({'phase':'unit_complete',**result})
            write_json(output/'prefill_progress.json',{'complete':False,'completed':len(results),'planned':len(data['records']),'results':results,'worker_model_identities':identities})
            if not result['ok']:raise RuntimeError('worker refused; full traceback preserved above')
        pool.close();pool.join()
        if not identities or len({schema.digest(v) for v in identities.values()})!=1:raise ValueError('worker model/runtime identities differ')
        identity=next(iter(identities.values()))
        if code_identity()!=request['word_code'] or source_identity(Path(__file__).resolve().parents[1])!=request['model_source'] or sha_file(a.checkpoint)!=request['checkpoint_sha256']:
            raise ValueError('source/model changed during extraction')
        write_json(output/'prefill_progress.json',{'complete':True,'completed':len(results),'planned':len(data['records']),'results':results,'worker_model_identities':identities})
        for split in ('dev-cal','test'):
            calibration=None if split=='dev-cal' else output/'fixed-thresholds.json'
            formal=output/('formal-dev-cal' if split=='dev-cal' else 'formal-test-fixed')
            publish(shared,identity,data,proto,split,formal,a.checkpoint,calibration)
            summary=evaluate_words(a.checkpoint,a.dataset,a.protocol,formal,split=split,device='cpu',calibration=calibration)
            if not summary['complete'] or summary['cache_hits']!=summary['planned_units']:
                raise ValueError('formal evaluation incomplete or not all strict prefill cache hits')
            emit({'phase':'formal_complete','split':split,'units':summary['planned_units'],'cache_hits':summary['cache_hits'],
                'available_conditions':summary['available_condition_count'],'missing_conditions':len(summary['missing_conditions'])})
            if split=='dev-cal':
                if (output/'fixed-thresholds.json').exists():raise ValueError('existing thresholds require explicit review; do not overwrite')
                value=calibrate_words(formal/'summary.json',a.dataset,output/'fixed-thresholds.json')
                emit({'phase':'calibration_complete','calibration_id':value['calibration_id']})
        write_json(output/'workflow_result.json',{'complete':True,'request':request,'model_identity':identity,
            'units':len(results),'formal_dev_summary':str(output/'formal-dev-cal/summary.json'),
            'formal_test_summary':str(output/'formal-test-fixed/summary.json'),'fixed_thresholds':str(output/'fixed-thresholds.json')})
        emit({'phase':'workflow_complete','units':len(results)});return 0
    except BaseException:
        pool.terminate();pool.join()
        failure={'complete':False,'traceback':traceback.format_exc(),'completed':len(results),'results':results,'worker_model_identities':identities}
        write_json(output/('workflow_failure-'+str(time.time_ns())+'.json'),failure);emit({'phase':'workflow_failure',**failure});return 2

if __name__=='__main__':raise SystemExit(main())
