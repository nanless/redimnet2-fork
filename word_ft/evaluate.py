"""Exact-length, whole linguistic-unit evaluation. No test-set threshold fitting."""
from __future__ import annotations
import csv
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
from duration_ft.evaluate import (load_model, EmbeddingCache, ModelExecutionError, AudioDecodeError,
    write_json, sha_file, digest, source_identity)
from word_ft import schema
from word_ft.prepare import build_count_matrix

PREPROCESS={'schema':'word-natural-v1','sample_rate':16000,'channels':'arithmetic-mean',
    'resample':'scipy.signal.resample_poly-gcd-default','interval':'declared-complete-unit',
    'padding':False,'repeat':False,'vad_concatenation':False,'precision':'float32','embedding':'L2'}


def read_protocol(dataset_path, protocol_path, *, allow_exploratory=False, verify_audio=True):
    data=schema.load_dataset(dataset_path,verify=verify_audio,allow_candidates=allow_exploratory)
    protocol=json.loads(Path(protocol_path).read_text())
    body={k:v for k,v in protocol.items() if k!='protocol_id'}
    if protocol.get('schema')!='word-count-matrix/v1' or protocol.get('protocol_id')!=digest(body):
        raise ValueError('word protocol identity differs')
    if protocol.get('dataset_id')!=data['dataset_id']:raise ValueError('protocol/dataset mismatch')
    for r in data['records']:
        if r.get('template_slot') not in (0,1) or isinstance(r.get('template_slot'),bool):raise ValueError('invalid template slot')
    for split in ('dev-cal','test'):
        if protocol['matrices'].get(split)!=build_count_matrix(data['records'],split):raise ValueError('matrix differs from frozen units/templates')
    speakers={s:{r['speaker_id'] for r in data['records'] if r['split']==s} for s in ('dev-cal','test')}
    if not speakers['dev-cal'] or not speakers['test'] or speakers['dev-cal']&speakers['test']:
        raise ValueError('development/test speakers overlap or empty')
    by_source={s:{r['source_utterance_id'] for r in data['records'] if r['split']==s} for s in speakers}
    by_audio={s:{r['audio_sha256'] for r in data['records'] if r['split']==s} for s in speakers}
    if by_source['dev-cal']&by_source['test'] or by_audio['dev-cal']&by_audio['test']:raise ValueError('development/test source overlap')
    if len({r['language'] for r in data['records']})!=1:raise ValueError('Chinese characters and English words require separate matrices')
    return data,protocol


def code_identity():
    return {p.name:sha_file(p) for p in (Path(__file__),Path(schema.__file__),Path(__file__).with_name('prepare.py'),Path(__file__).resolve().parents[1]/'duration_ft/metrics.py')}


def extract_unit(model, model_identity, unit, cache):
    identity={'schema':'word-complete-embedding/v1','model':model_identity,'preprocess':PREPROCESS,
        'word_code':code_identity(),'unit':schema.unit_identity(unit),
        'output_samples':unit['n_samples_16k']}
    # Validate the live source even when the vector was cached in an earlier attempt.
    schema.validate_unit(unit,allow_candidates=True,verify_audio=True)
    cached=cache.get(identity)
    if cached is not None:return cached,True
    try:wave=schema.read_unit(unit)
    except (MemoryError, KeyboardInterrupt):raise
    except Exception as error:raise AudioDecodeError('complete-unit decode failed: '+unit['unit_id']) from error
    import torch
    try:
        with torch.inference_mode():out=model(torch.from_numpy(wave).unsqueeze(0).to(model_identity['device']))
        if out.shape!=(1,model_identity['embedding_dim']):raise ValueError('wrong embedding shape')
        vector=out.detach().float().cpu().numpy()[0]
        norm=float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or not math.isfinite(norm) or norm<=1e-12:raise ValueError('invalid embedding')
        vector=vector/norm
    except Exception as error:raise ModelExecutionError(f'unit model/device failure: {type(error).__name__}: {error}') from error
    cache.put(identity,vector)
    return vector,False


def check_calibration(calibration, model_identity, test_units):
    if calibration is None:return None
    value=json.loads(Path(calibration).read_text()) if isinstance(calibration,(str,Path)) else calibration
    body={k:v for k,v in value.items() if k!='calibration_id'}
    if value.get('schema')!='word-fixed-threshold/v1' or value.get('calibration_id')!=digest(body):raise ValueError('calibration identity differs')
    if value.get('model_sha256')!=model_identity['checkpoint_sha256'] or value.get('model_identity')!=model_identity:raise ValueError('calibration model/runtime mismatch')
    if not value.get('dev_speakers') or set(value['dev_speakers'])&{u['speaker_id'] for u in test_units}:raise ValueError('calibration/test speakers overlap or unknown')
    if set(value.get('dev_audio_sha256',[]))&{u['audio_sha256'] for u in test_units}:raise ValueError('calibration/test audio overlaps')
    if value.get('preprocess')!=PREPROCESS or value.get('word_code')!=code_identity():raise ValueError('calibration preprocessing/code differs')
    for key in ('far_0.01','far_0.001'):
        threshold=value.get('thresholds',{}).get(key,{}).get('threshold')
        if isinstance(threshold,bool) or not isinstance(threshold,(float,int)) or not math.isfinite(threshold):raise ValueError('missing finite fixed threshold')
    return value


def write_scores(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=['trial_id','template_id','label','enroll','verify','enroll_speaker','verify_speaker','test_speaker',
        'content_condition','status','score','enroll_duration','test_duration','enroll_n_units','test_n_units',
        'enroll_source_utterance_id','test_source_utterance_id','enroll_audio_sha256','test_audio_sha256','model_id','cohort_id']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def evaluate_words(checkpoint,dataset,protocol,output,*,split='test',device='cpu',allow_gpu=False,
    gpu_memory_mib=None,calibration=None,allow_exploratory=False,max_audio_errors=10,max_consecutive_audio_errors=3):
    from duration_ft.metrics import score_metrics,fixed_threshold_metrics
    if split not in ('dev-cal','test'):raise ValueError('unknown split')
    if max_audio_errors<1 or max_consecutive_audio_errors<1:raise ValueError('positive error budgets required')
    out=schema.protect_output(output)
    data,proto=read_protocol(dataset,protocol,allow_exploratory=allow_exploratory)
    intent={'checkpoint_sha256':sha_file(Path(checkpoint)),'dataset_id':data['dataset_id'],
        'protocol_id':proto['protocol_id'],'split':split,'device':device,'gpu_memory_mib':gpu_memory_mib,
        'allow_exploratory':allow_exploratory,'word_code':code_identity(),'preprocess':PREPROCESS,
        'model_source':source_identity(Path(__file__).resolve().parents[1]),
        'calibration_sha256':sha_file(Path(calibration)) if calibration is not None else None}
    if out.exists() and any(out.iterdir()):
        if not (out/'request_identity.json').is_file() or json.loads((out/'request_identity.json').read_text())!=intent:
            raise ValueError('output belongs to another request; existing evidence preserved')
    out.mkdir(parents=True,exist_ok=True);write_json(out/'request_identity.json',intent)
    units={u['unit_id']:u for u in data['records'] if u['split']==split}
    comparison={'dataset_id':data['dataset_id'],'protocol_id':proto['protocol_id'],'split':split,
        'preprocess':PREPROCESS,'word_code':code_identity(),'unit_identities':{k:schema.unit_identity(v) for k,v in units.items()}}
    summary={'schema':'word-evaluation/v1','complete':False,'split':split,'exploratory':allow_exploratory,
        'comparison_policy':proto['matrices'][split]['comparison_policy'],'cells':[],'errors':[],
        'comparison_identity_sha256':digest(comparison),'comparison_identity':comparison,
        'quality_scope':'provider whole-utterance reference; no independently audited boundary claim',
        'duration_scope':'natural elapsed whole-file duration; no VAD speech-time substitution'}
    try:
        model,model_id=load_model(checkpoint,device,allow_gpu=allow_gpu,gpu_memory_mib=gpu_memory_mib)
        cal=check_calibration(calibration,model_id,list(units.values())) if split=='test' else None
        if cal and (cal['dev_dataset_id']!=data['dataset_id'] or cal['dev_protocol_id']!=proto['protocol_id']):raise ValueError('calibration must use this frozen disjoint dev/test protocol')
        identity={'model':model_id,'comparison_identity_sha256':digest(comparison),'calibration_id':cal['calibration_id'] if cal else None}
        existing=out/'run_identity.json'
        if existing.exists() and json.loads(existing.read_text())!=identity:raise ValueError('output belongs to another model/protocol/calibration')
        write_json(existing,identity);summary['run_identity']=identity
        cache=EmbeddingCache(out/'embedding-cache');vectors={};cache_hits=0;consecutive=0
        for uid,unit in sorted(units.items()):
            try:
                vector,hit=extract_unit(model,model_id,unit,cache);vectors[uid]=vector;cache_hits+=int(hit);consecutive=0
            except AudioDecodeError as error:
                consecutive+=1;summary['errors'].append({'unit_id':uid,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
                if len(summary['errors'])>=max_audio_errors or consecutive>=max_consecutive_audio_errors:
                    summary['abort_reason']='bounded audio failure budget';break
            except Exception as error:
                summary['errors'].append({'unit_id':uid,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
                summary['abort_reason']='model/device/system/source-identity failure';break
        summary.update(extracted_units=len(vectors),planned_units=len(units),cache_hits=cache_hits)
        for cell in proto['matrices'][split]['cells']:
            for condition,group in cell['groups'].items():
                rows=[]
                for trial in group['trials']:
                    e,t=units[trial['enroll']],units[trial['verify']]
                    row=dict(trial,test_speaker=trial['verify_speaker'],model_id=model_id['checkpoint_sha256'],cohort_id=split,
                        enroll_duration=e['duration_sec'],test_duration=t['duration_sec'],enroll_n_units=e['n_units'],test_n_units=t['n_units'],
                        enroll_source_utterance_id=e['source_utterance_id'],test_source_utterance_id=t['source_utterance_id'],
                        enroll_audio_sha256=e['audio_sha256'],test_audio_sha256=t['audio_sha256'])
                    if trial['enroll'] in vectors and trial['verify'] in vectors:
                        row.update(status='ok',score=float(np.dot(vectors[trial['enroll']],vectors[trial['verify']])))
                    else:row.update(status='missing_embedding',score=None)
                    rows.append(row)
                path=out/'cells'/cell['name']/condition/'scores.csv';write_scores(path,rows)
                complete=group['available'] and all(r['status']=='ok' for r in rows)
                item={k:v for k,v in cell.items() if k!='groups'}
                item.update(content_condition=condition,available=group['available'],complete=complete,
                    unavailable_reason=group['unavailable_reason'],trial_count=len(rows),scored_trials=sum(r['status']=='ok' for r in rows),
                    positive=group['positive'],negative=group['negative'],scores_csv=str(path.resolve()),scores_csv_sha256=sha_file(path),
                    trialset_sha256=group['trialset_sha256'],comparison_identity_sha256=digest(comparison),metrics=None,fixed_threshold=None)
                if complete:
                    labels=[r['label'] for r in rows];scores=[r['score'] for r in rows]
                    item['metrics']=score_metrics(labels,scores)
                    if cal:item['fixed_threshold']={k:fixed_threshold_metrics(labels,scores,v['threshold']) for k,v in cal['thresholds'].items()}
                summary['cells'].append(item)
        summary['complete']=not summary['errors'] and len(vectors)==len(units) and all(c['complete'] for c in summary['cells'] if c['available'])
        summary['all_requested_conditions_available']=all(c['available'] for c in summary['cells'])
        summary['available_condition_count']=sum(c['available'] for c in summary['cells'])
        summary['requested_condition_count']=len(summary['cells'])
        summary['completion_scope']='all available conditions scored; unavailable natural-data cells remain missing'
        summary['missing_conditions']=[{'cell':c['name'],'content_condition':c['content_condition'],'reason':c['unavailable_reason']} for c in summary['cells'] if not c['available']]
    except Exception as error:
        summary['errors'].append({'phase':'initialization','type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    if summary['errors']:write_json(out/'failures'/f'attempt-{time.time_ns()}.json',summary)
    write_json(out/'summary.json',summary)
    return summary


def calibrate_words(summary_path,dataset_path,output,*,cell='4-6x4-6'):
    from duration_ft.metrics import calibrate_far_threshold
    summary=json.loads(Path(summary_path).read_text());data=schema.load_dataset(dataset_path,verify=False)
    if summary.get('split')!='dev-cal' or not summary.get('complete') or summary.get('exploratory'):raise ValueError('complete reference-only dev-cal evaluation required')
    if summary['comparison_identity']['dataset_id']!=data['dataset_id']:raise ValueError('calibration dataset differs')
    found=[x for x in summary['cells'] if x['name']==cell and x['content_condition']=='all']
    if len(found)!=1 or not found[0]['complete']:raise ValueError('preregistered dev cell unavailable')
    info=found[0];path=Path(info['scores_csv'])
    if sha_file(path)!=info['scores_csv_sha256']:raise ValueError('dev scores changed')
    with path.open() as stream:rows=list(csv.DictReader(stream))
    labels=[int(r['label']) for r in rows];scores=[float(r['score']) for r in rows]
    value={'schema':'word-fixed-threshold/v1','model_sha256':summary['run_identity']['model']['checkpoint_sha256'],
        'model_identity':summary['run_identity']['model'],'dev_dataset_id':data['dataset_id'],'dev_protocol_id':summary['comparison_identity']['protocol_id'],
        'dev_speakers':sorted({u['speaker_id'] for u in data['records'] if u['split']=='dev-cal'}),
        'dev_audio_sha256':sorted({u['audio_sha256'] for u in data['records'] if u['split']=='dev-cal'}),
        'dev_cell':cell,'content_condition':'all','dev_scores_sha256':info['scores_csv_sha256'],
        'preprocess':PREPROCESS,'word_code':code_identity(),
        'thresholds':{'far_'+str(f):calibrate_far_threshold(labels,scores,target_far=f) for f in (.01,.001)},
        'scope':'speaker/source-disjoint within this protocol; external/pretraining identity overlap not proven; same thresholds for every test count cell'}
    value['calibration_id']=digest(value);output=schema.protect_output(output)
    if output.exists():raise FileExistsError(output)
    write_json(output,value);return value
