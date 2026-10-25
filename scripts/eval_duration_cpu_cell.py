#!/usr/bin/env python3
"""One raw/no-VAD duration cell, fresh CPU workers and the shared evaluator cache.

The parent indexes/scores only. Workers each own one CPU-only model with one
thread. Any missing/error embedding makes metrics unavailable for the full cell.
Long real extraction is an explicit invocation, never part of --self-test.
"""
from __future__ import annotations
import os
# Must precede numpy/torch imports in spawned children, even when caller has CUDA.
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_name]='1'
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from duration_ft import evaluate as E

_MODEL=None
_MODEL_ID=None
_INIT_ERROR=None
_SENT_ID=False
_CACHE=None

def resources():
    visible=os.cpu_count() or 1
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else visible
    quota=None;raw={}
    for root in [Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup/cpu'),Path('/sys/fs/cgroup/cpu,cpuacct')]:
        maxfile=root/'cpu.max';q=root/'cpu.cfs_quota_us';p=root/'cpu.cfs_period_us'
        try:
            if maxfile.is_file():
                value=maxfile.read_text().strip();raw[str(maxfile)]=value
                amount,period=value.split()
                if amount!='max':quota=min(quota or math.inf,float(amount)/float(period))
            elif q.is_file() and p.is_file():
                amount,period=float(q.read_text()),float(p.read_text())
                raw[str(q)]=amount;raw[str(p)]=period
                if amount>0:quota=min(quota or math.inf,amount/period)
        except (OSError,ValueError):pass
    available=min(visible,affinity,quota if quota is not None else math.inf)
    memory={}
    if Path('/proc/meminfo').is_file():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.split(':')[0] in ['MemTotal','MemAvailable','MemFree','SwapFree']:
                memory[line.split(':')[0]]=line.split(':')[1].strip()
    return {'observed_utc':datetime.now(timezone.utc).isoformat(),'hostname':os.uname().nodename,
            'cpu_count':visible,'affinity_count':affinity,'cgroup_cpu_quota':quota,'cgroup_files':raw,
            'quota_status':'readable' if quota is not None else 'unavailable; cap uses conservative visible CPU count',
            'conservative_available_cpus':available,'worker_limit_25pct':min(6,math.floor(available*.25)),
            'loadavg':list(os.getloadavg()),'memory':memory}

def check_workers(workers,resource):
    if not 1<=workers<=6 or workers>resource['worker_limit_25pct']:
        raise E.EvaluationRefused(f"workers={workers} exceeds min(6, floor(visible available CPUs*25%))={resource['worker_limit_25pct']}")

def worker_init(checkpoint,cache_dir):
    global _MODEL,_MODEL_ID,_INIT_ERROR,_SENT_ID,_CACHE
    _SENT_ID=False;_INIT_ERROR=None
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        _MODEL,_MODEL_ID=E.load_model(Path(checkpoint),'cpu')
        _CACHE=E.EmbeddingCache(Path(cache_dir))
        if torch.cuda.is_initialized():raise RuntimeError('CPU worker initialized CUDA')
    except BaseException as exc:
        _INIT_ERROR={'phase':'worker_model_init','type':type(exc).__name__,'message':str(exc),
                     'traceback':traceback.format_exc(),'pid':os.getpid()}

def worker_one(item):
    global _SENT_ID
    record,crop=item
    result={'relative':record['relative'],'crop':crop,'pid':os.getpid()}
    if _INIT_ERROR:
        return dict(result,status='error',error=_INIT_ERROR)
    try:
        vector,hit=E.extract_one(_MODEL,_MODEL_ID,record,crop,_CACHE)
        result.update(status='ok',vector=vector.tolist(),cache_hit=hit,model_identity_sha256=E.digest(_MODEL_ID))
        if not _SENT_ID:
            result['model_identity']=_MODEL_ID;_SENT_ID=True
        return result
    except BaseException as exc:
        return dict(result,status='error',error={'phase':'extract','type':type(exc).__name__,
                    'message':str(exc),'traceback':traceback.format_exc(),
                    'fatal':not isinstance(exc,E.AudioDecodeError)})

def write_scores(path,rows,duration,cohort,model_sha):
    for row in rows:
        row.update(test_speaker=row['verify_speaker'],enroll_duration=duration,test_duration=duration,
                   cohort_id=cohort,model_id=model_sha)
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run(checkpoint,protocol,audio_root,output,duration=.5,workers=1,resume=False,
        strict_standard=True,max_errors=3,task_timeout=180):
    import numpy as np
    from duration_ft.metrics import score_metrics
    resource=resources();check_workers(workers,resource)
    if duration!='full' and (not math.isfinite(duration) or duration<=0):raise E.EvaluationRefused('positive finite duration or full required')
    if max_errors<1 or task_timeout<=0:raise E.EvaluationRefused('positive error/timeout bounds required')
    output=Path(output);E.protect_output(output)
    repo=Path(__file__).resolve().parents[1]
    if repo==output.resolve() or repo in output.resolve().parents:raise E.EvaluationRefused('output inside source tree')
    output.mkdir(parents=True,exist_ok=resume)
    if not resume and any(output.iterdir()):raise E.EvaluationRefused('new output directory required')
    with (output/'.cpu-cell.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        E.write_json(output/f'cpu_cell_resources-{time.time_ns()}.json',resource)
        trials,protocol_id=E.read_trials(Path(protocol))
        if strict_standard and (protocol_id['sha256']!=E.STANDARD_O_SHA256 or len(trials)!=37611):
            raise E.EvaluationRefused('standard all-O protocol identity/count differ')
        records={};crops={};errors=[]
        unique_audio=sorted({t[k] for t in trials for k in ['enroll','verify']})
        for rel in unique_audio:
            try:
                record=E.audio_record(Path(audio_root),rel)
                records[rel]=record;crops[rel]=E.center_crop(record,duration)
            except Exception as exc:
                errors.append({'phase':'index','relative':rel,'type':type(exc).__name__,
                               'message':str(exc),'traceback':traceback.format_exc()})
        E.write_json(output/'cpu_cell_index_errors.json',errors)
        if strict_standard and len(records)!=4708 and not errors:
            raise E.EvaluationRefused('standard O expected 4708 unique audio')
        request={'schema':'duration-cpu-single-cell/v1','checkpoint_sha256':E.sha_file(Path(checkpoint)),
                 'protocol':protocol_id,'trialset_sha256':E.digest(trials),'duration':duration,
                 'preprocess':E.PREPROCESS,'audio_identity_sha256':E.digest({r:E.audio_content_identity(v) for r,v in records.items()}),
                 'crops_sha256':E.digest([{k:v for k,v in crops[r].items() if k!='source_stat'} for r in sorted(crops)]),
                 'model_code':E.source_identity(repo),'entry_sha256':E.sha_file(Path(__file__)),
                 'strict_standard':strict_standard,'runtime':'CPU float32; one thread per spawned worker',
                 'calibration':'unavailable; no independent frozen threshold supplied'}
        identity_file=output/'cpu_cell_identity.json'
        if identity_file.exists() and json.loads(identity_file.read_text())!=request:
            raise E.EvaluationRefused('resume request identity differs; choose a new output')
        E.write_json(identity_file,request);E.write_json(output/'cpu_cell_audio_index.json',records)
        E.write_json(output/'cpu_cell_crops.json',[crops[r] for r in sorted(crops)])
        vectors={};hits=0;attempted=0;model_identity=None;model_digest=None;aborted=bool(errors)
        started=time.monotonic()
        if not errors:
            # Spawn avoids forking a loaded model/BLAS thread pool from the parent.
            ctx=mp.get_context('spawn')
            with ctx.Pool(workers,initializer=worker_init,
                          initargs=(str(checkpoint),str(output/'embedding-cache'))) as pool:
                iterator=pool.imap_unordered(worker_one,((records[r],crops[r]) for r in sorted(crops)),chunksize=1)
                for _ in range(len(crops)):
                    try:result=iterator.next(timeout=task_timeout)
                    except Exception as exc:
                        errors.append({'phase':'worker_timeout_or_pool','type':type(exc).__name__,
                                       'message':str(exc),'traceback':traceback.format_exc()});aborted=True;break
                    attempted+=1
                    if result['status']=='error':
                        errors.append(result)
                        E.write_json(output/'failures'/f"crop-{time.time_ns()}.json",result)
                        if result['error'].get('fatal',True) or len(errors)>=max_errors:
                            aborted=True;break
                    else:
                        identity=result.get('model_identity')
                        try:
                            if identity:
                                if identity['checkpoint_sha256']!=request['checkpoint_sha256'] or identity['code']!=request['model_code']:
                                    raise E.EvaluationRefused('worker loaded changed checkpoint/model code')
                                if model_identity is not None and model_identity!=identity:
                                    raise E.EvaluationRefused('worker model/runtime identities differ')
                                model_identity=identity;model_digest=E.digest(identity)
                            if model_digest is not None and result['model_identity_sha256']!=model_digest:
                                raise E.EvaluationRefused('worker identity digest differs')
                        except Exception as exc:
                            errors.append({'phase':'worker_identity','type':type(exc).__name__,
                                           'message':str(exc),'traceback':traceback.format_exc()})
                            aborted=True;break
                        vector=np.asarray(result['vector'],dtype=np.float32)
                        vectors[result['relative']]=vector;hits+=int(result['cache_hit'])
                    if attempted%100==0:
                        progress={'complete':False,'attempted_crops':attempted,'total':len(crops),'errors':len(errors),
                                  'cache_hits':hits,'elapsed_seconds':time.monotonic()-started}
                        E.write_json(output/'cpu_cell_progress.json',progress);print(json.dumps(progress),flush=True)
                # Leaving this context terminates only this entry's own model workers,
                # including on bounded timeout/error; no unrelated process is touched.
        rows,scores,labels=E.score_cell(trials,vectors,vectors)
        complete=not errors and len(scores)==len(trials) and len(vectors)==len(unique_audio)
        write_scores(output/'cpu_cell_scores.csv',rows,duration,request['trialset_sha256'],request['checkpoint_sha256'])
        summary={'request_identity':request,'model_identity':model_identity,'complete':complete,
                 'aborted':aborted,'duration_enroll':duration,'duration_test':duration,'protocol':'raw/no VAD',
                 'expected_trials':len(trials),'scored_trials':len(scores),'expected_unique_audio':len(unique_audio),'indexed_unique_audio':len(records),
                 'completed_unique_audio':len(vectors),'attempted_crops':attempted,'cache_hits':hits,
                 'positive':sum(t['label'] for t in trials),'negative':sum(1-t['label'] for t in trials),
                 'speakers':len({t[k] for t in trials for k in ['enroll_speaker','verify_speaker']}),
                 'metrics':score_metrics(labels,scores) if complete else None,
                 'fixed_threshold':{'status':'unavailable','reason':'no independent frozen calibration supplied'},
                 'errors':errors,'workers':workers,'threads_per_worker':1,'elapsed_seconds':time.monotonic()-started,
                 'scores_sha256':E.sha_file(output/'cpu_cell_scores.csv'),
                 'note':'EER/minDCF test-score thresholds describe discrimination, not calibrated deployment thresholds; no successful-subset metrics.'}
        E.write_json(output/'cpu_cell_extraction_errors.json',errors);E.write_json(output/'cpu_cell_summary.json',summary)
        return summary

def self_test():
    """Synthetic single-item cache/decode and refusal checks; no released model."""
    import numpy as np
    import soundfile as sf
    import torch
    global _MODEL,_MODEL_ID,_INIT_ERROR,_CACHE,_SENT_ID
    torch.set_num_threads(1)
    class Toy(torch.nn.Module):
        def forward(self,wave):
            return torch.stack((wave.mean(1),wave.square().mean(1)+.1),1)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);audio=root/'idA/v/a.wav';audio.parent.mkdir(parents=True)
        sf.write(audio,np.linspace(-.2,.5,16000,dtype=np.float32),16000,subtype='FLOAT')
        record=E.audio_record(root,'idA/v/a.wav');crop=E.center_crop(record,.5)
        assert crop['actual_samples']==8000 and crop['start_sample']==4000
        _MODEL=Toy().eval();_MODEL_ID={'embedding_dim':2,'checkpoint_sha256':'synthetic-only','device':'cpu'}
        _CACHE=E.EmbeddingCache(root/'cache');_INIT_ERROR=None;_SENT_ID=False
        a=worker_one((record,crop));b=worker_one((record,crop))
        assert a['status']=='ok' and not a['cache_hit'] and b['cache_hit']
        assert a['vector']==b['vector']
        try:E.center_crop(record,2)
        except E.EvaluationRefused:pass
        else:raise AssertionError('short audio was not refused')
        try:check_workers(17,{'worker_limit_25pct':16})
        except E.EvaluationRefused:pass
        else:raise AssertionError('worker bound ignored')
        audio.write_bytes(b'broken')
        failure=worker_one((record,crop));assert failure['status']=='error' and failure['error']['traceback']
        assert not torch.cuda.is_initialized()
        for rel in ['idA/v/b.wav','idB/v/c.wav']:
            dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            sf.write(dest,np.ones(16000,dtype=np.float32)*.1,16000)
        sf.write(audio,np.ones(16000,dtype=np.float32)*.2,16000)
        protocol=root/'trials.txt';protocol.write_text('1 idA/v/a.wav idA/v/b.wav\n0 idA/v/a.wav idB/v/c.wav\n')
        fake=root/'invalid.pt';fake.write_bytes(b'not a model')
        failed=run(fake,protocol,root,root/'failed-cell',duration=.5,workers=1,strict_standard=False,task_timeout=30)
        assert not failed['complete'] and failed['metrics'] is None and failed['expected_trials']==2
        assert failed['scored_trials']==0 and failed['errors'][0]['error']['traceback']
        assert failed['fixed_threshold']['status']=='unavailable'
    return {'self_test':'passed','checks':['exact 0.5 seconds','shared center crop','shared extraction/cache hit',
            'insufficient duration refused','worker cap refused','source change full traceback','CUDA never initialized','spawn model-init traceback retained','incomplete full-cell metrics unavailable'],
            'real_model_extracted':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint');p.add_argument('--protocol');p.add_argument('--audio-root');p.add_argument('--output')
    p.add_argument('--duration',type=lambda v: 'full' if v=='full' else float(v),default=.5);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--resume',action='store_true');p.add_argument('--nonstandard-test-protocol',action='store_true')
    p.add_argument('--max-errors',type=int,default=3);p.add_argument('--task-timeout',type=float,default=180)
    p.add_argument('--resources-only',action='store_true');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.resources_only:print(json.dumps(resources()));return
    if a.self_test:print(json.dumps(self_test()));return
    for key in ['checkpoint','protocol','audio_root','output']:
        if not getattr(a,key):p.error('--'+key.replace('_','-')+' is required for real evaluation')
    summary=run(a.checkpoint,a.protocol,a.audio_root,a.output,a.duration,a.workers,a.resume,
                not a.nonstandard_test_protocol,a.max_errors,a.task_timeout)
    print(json.dumps({'complete':summary['complete'],'metrics':summary['metrics'],'output':a.output}))
    if not summary['complete']:raise SystemExit(2)

if __name__=='__main__':main()
