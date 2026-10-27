"""读取完整有限样本的跨引擎结果，比较文字、边界及已发布结果保持。

仅写本次新报告。没有模型、GPU或队列操作；保存的计时不自认证GPU运行。
"""
from __future__ import annotations
import argparse
from collections import Counter
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import atomic_new, digest, job_key, need, model_provenance
from schema import assert_valid_record, SUCCESS
from engine_protocol import (read_ref, physical, validate_protocol, validate_producer, validate_input,
    controls_from_contract, code_identity, finite, native_result_identity, WINDOW_FIELDS, SOURCE_FIELDS,
    validate_native_spec, MAX_SOURCES, KINDS)


def _map(rows, key, label):
    need(isinstance(rows,list) and rows and all(isinstance(x,dict) and key in x for x in rows), 'invalid ' + label)
    result = {x[key]:x for x in rows}
    need(len(result) == len(rows), 'duplicate ' + label)
    return result


def load_inputs(ref):
    rows, size = read_ref(ref, lines=True)
    need(0 < len(rows) <= MAX_SOURCES, 'finite comparison source count exceeded')
    for row in rows: validate_input(row)
    by_id = {x['source']['source_id']:x for x in rows}
    need(len(by_id) == len(rows), 'duplicate declared source')
    return by_id, size


def _validate_sidecar(row, producer, contract, contract_sha, planned):
    record, record_bytes = read_ref(row['sidecar'], maximum=32*1024**2)
    marker, _ = read_ref(row['commit'], maximum=1024**2)
    assert_valid_record(record)
    source = planned['source']; sid = source['source_id']
    need(record['source'] == source and row['source_id'] == sid, 'saved source record differs from fixed input')
    need(record['contract']['sha256'] == contract_sha and record['contract']['version'] == contract['version'],
         'sidecar producer contract differs')
    need(record['provenance']['runtime'] == contract['runtime'] and record['provenance']['models'] == model_provenance(contract),
         'saved runtime/weights differ from producer contract')
    need(record['provenance']['code_sha256']==producer['bundle_sha256'] and record['contract'].get('provenance')==record['provenance'],
         'sidecar frozen bundle/duplicated provenance differs from producer')
    expected_marker = {'schema':'redimnet2-sidecar-commit/v1','source_id':sid,
        'job_key':job_key(source,contract_sha),'contract_sha256':contract_sha,'source_record_sha256':digest(source),
        'sidecar_sha256':row['sidecar']['sha256'], 'bytes':record_bytes}
    need(marker == expected_marker and record['execution']['job_key'] == expected_marker['job_key'], 'sidecar commit identity differs')
    expected_windows = {x['window_id']:x for x in planned['windows']}
    windows = {x['window_id']:x for x in record['windows']}
    need(len(windows) == len(record['windows']) and set(windows) <= set(expected_windows), 'unexpected/duplicate saved window')
    identity_comparable = record['source_identity']['decode_status'] == planned['source_identity'].get('decode_status') == 'decoded_complete'
    if identity_comparable:
        need(all(record['source_identity'].get(k) == planned['source_identity'].get(k) for k in SOURCE_FIELDS),
             'decoded source identity differs from fixed input')
    else:
        need(record['decision']['state'] in {'unknown','failed'}, 'undecoded result cannot be a clean candidate')
    for wid, w in windows.items():
        pw = expected_windows[wid]
        if w.get('preprocessing_status','completed') == 'completed':
            need(all(w.get(k) == pw[k] for k in WINDOW_FIELDS), 'saved PCM/window/preprocessing identity differs')
        else:
            need(w.get('asr',{}).get('status') == 'failed', 'failed preprocessing lacks failed ASR evidence')
        asr = w.get('asr',{})
        limit = asr.get('native',{}).get('max_new_tokens')
        if asr.get('status') in SUCCESS | {'truncated'}:
            need(limit == producer['fixed_controls']['generation']['value']['max_new_tokens'], 'actual saved token cap differs')
    if set(windows) != set(expected_windows):
        need(record['decision']['state'] in {'unknown','failed'}, 'candidate silently omitted planned windows')
    return record, windows, identity_comparable


def load_track(track, fixed, planned):
    need(isinstance(track,dict) and all(k in track for k in ('producer','results','metrics')), 'track references missing')
    producer, _ = read_ref(track['producer'])
    validate_producer(producer,fixed)
    contract, _ = read_ref(producer['contract'])
    contract_sha = producer['contract']['sha256']
    need(contract.get('purpose')==producer['purpose'], 'contract purpose differs from producer')
    need(controls_from_contract(contract) == fixed, 'producer contract changed fixed controls')
    need(producer['variables']['code_sha256'] == code_identity(contract), 'engine code declaration differs from producer contract')
    expected_dtype = contract['runtime'].get('asr_compute',{}).get('dtype',contract['runtime'].get('dtype','float32'))
    need(producer['variables']['precision'] == expected_dtype, 'engine precision declaration differs from contract')
    native_spec=validate_native_spec(producer,contract) if producer['variables']['backend']=='vllm' else None
    rows, _ = read_ref(track['results'], lines=True)
    by_id = _map(rows,'source_id','source result')
    need(set(by_id) == set(planned), 'result source union differs; failed sources must remain')
    records, totals, decisions, missing, native = {}, Counter(), Counter(), 0, {}
    expected_windows = sum(len(v['windows']) for v in planned.values())
    input_comparable = True
    for sid, row in by_id.items():
        record, windows, comparable = _validate_sidecar(row,producer,contract,contract_sha,planned[sid])
        input_comparable &= comparable
        missing += len(planned[sid]['windows']) - len(windows)
        execution = row.get('execution')
        need(isinstance(execution,dict) and set(execution) == set(windows), 'per-window execution accounting differs')
        refs = row.get('native_asr_refs',{})
        need(isinstance(refs,dict) and set(refs) <= set(windows), 'native result references have unknown windows')
        for wid, stages in execution.items():
            need(isinstance(stages,dict) and set(stages) == {'asr','alignment'} and all(v in KINDS for v in stages.values()),
                 'invalid actual per-window stage accounting')
            for stage, kind in stages.items(): totals[stage+'_'+kind+'_windows'] += 1
            if producer['variables']['backend']=='vllm' and stages['asr']=='new':
                need(wid in refs, 'new vLLM execution must preserve native request result')
            if wid in refs:
                value, _ = read_ref(refs[wid],maximum=16*1024**2)
                pw = next(w for w in planned[sid]['windows'] if w['window_id'] == wid)
                need(native_spec is not None, 'independent native ASR refs need an explicit supported adapter spec')
                native_result_identity(value,pw,sid,producer,native_spec)
                need(all(value.get(k) == windows[wid]['asr'].get(k) for k in ('status','raw_text','normalized_text','language','normalization_audit')),
                     'native output and final ASR text/language differ')
                need(all(value['native'].get(k)==windows[wid]['asr'].get('native',{}).get(k)
                         for k in ('generated_token_ids','eos_observed','max_new_tokens')),
                     'native output and final generation evidence differ')
                native[(sid,wid)] = value
        totals['source_results'] += 1;decisions[record['decision']['state']] += 1;records[sid] = record
    metrics, _ = read_ref(track['metrics'])
    need(metrics.get('schema') == 'annotation-engine-metrics/v1' and metrics.get('source_count') == len(planned), 'metrics source denominator differs')
    need(metrics.get('producer_sha256') == track['producer']['sha256'], 'metrics producer differs')
    need(metrics.get('time_basis') == 'whole_run_wall_including_loading_and_publication', 'whole-run wall basis required')
    wall = finite(metrics.get('wall_seconds'),'whole run wall',positive=True)
    for stage in ('asr','alignment'):
        for kind in sorted(KINDS):
            key = stage+'_'+kind+'_windows';v = metrics.get(key)
            need(type(v) is int and v >= 0 and v == totals[key], 'metrics execution accounting differs: ' + key)
    need(metrics.get('model_API_calls') == 0 and type(metrics['model_API_calls']) is int, 'nonlocal calls observed')
    need(type(metrics.get('synthetic_fixture')) is bool and
         (producer['purpose']!='cpu_fixture' or metrics['synthetic_fixture']), 'synthetic execution marker missing or relabelled')
    fresh = (not missing and input_comparable and expected_windows > 0 and totals['asr_new_windows'] == expected_windows
             and all(totals[s+'_'+k+'_windows'] == 0 for s in ('asr','alignment') for k in ('cache','reused')))
    return {'spec':track,'producer':producer,'contract':contract,'records':records,'raw_rows':by_id,
            'native_results':native,'decisions':dict(decisions),'totals':dict(totals),'metrics':metrics,
            'wall_seconds':wall,'all_sources_and_windows_present':not missing,'missing_windows':missing,
            'all_saved_decoded_inputs_comparable':input_comparable,'reported_all_new_execution':fresh}


def compare_window(a, b, planned, mode, native_a=None, native_b=None):
    result={'window_id':planned['window_id'],'both_present':a is not None and b is not None}
    if a is None or b is None:
        return result|{'endpoint_comparison_status':'missing_window_preserved','endpoint_delta_ms':None}
    aa, bb = a.get('asr',{}), b.get('asr',{})
    result['ASR_equal']={k:aa.get(k)==bb.get(k) for k in ('status','raw_text','normalized_text','language')}
    result['native_ASR']={'baseline':native_a or aa.get('native'), 'candidate':native_b or bb.get('native'),
                          'cross_engine_bitwise_equivalence_required':False}
    al, bl = a.get('alignment',{}),b.get('alignment',{})
    if mode == 'fixed_text_alignment':
        text = planned.get('fixed_text')
        need(isinstance(text,str) and text, 'fixed-text input is missing')
        for original in (al,bl):
            if original.get('status') in SUCCESS or original.get('target_text'):
                need(original.get('target_text')==text, 'fixed-text alignment did not use the declared identical text')
    else:
        for w, asr in ((al,aa),(bl,bb)):
            if w.get('status') in SUCCESS:
                need(w.get('target_text') == asr.get('normalized_text'), 'end-to-end FA target differs from own ASR')
    words_a = al.get('encoded_target_words',[v.get('text') for v in al.get('items',[])])
    words_b = bl.get('encoded_target_words',[v.get('text') for v in bl.get('items',[])])
    same_target = bool(words_a) and words_a == words_b and al.get('target_text') == bl.get('target_text')
    result['alignment_targets_equal'] = same_target
    result['alignment_observations']={'baseline':al,'candidate':bl}
    result['endpoint_comparison_status']='target_or_geometry_not_comparable'
    result['endpoint_delta_ms']=None
    if same_target:
        measured={}
        for key in ('native_timestamps_ms','fixed_timestamps_ms'):
            x,y=al.get(key),bl.get(key)
            if (isinstance(x,list) and isinstance(y,list) and x and len(x)==len(y)==2*len(words_a)
                and all(type(z) in (int,float) and math.isfinite(z) for z in x+y)):
                measured[key]=max(abs(v-w) for v,w in zip(x,y))
        if measured:result.update(endpoint_comparison_status='same_target_items',endpoint_delta_ms=measured)
    result['observed_states']={'baseline':{'asr':aa.get('status'),'alignment':al.get('status')},
                               'candidate':{'asr':bb.get('status'),'alignment':bl.get('status')}}
    return result


def check_published_before(ref, candidate_rows):
    value,_=read_ref(ref)
    need(value.get('schema')=='annotation-published-before/v1' and isinstance(value.get('entries'),list), 'invalid prior publication snapshot')
    seen=set()
    for row in value['entries']:
        sid=row['source_id'];need(sid not in seen and sid in candidate_rows,'duplicate/missing previously published source');seen.add(sid)
        for key in ('sidecar','commit'):
            need(row[key] == candidate_rows[sid][key], 'previously published producer reference was replaced')
            read_ref(row[key])
    return {'published_before_sources':len(seen),'published_references_and_bytes_unchanged':True,
            'unpublished_recomputation_bitwise_required':False}


def audit(spec_path, spec_sha, output=None):
    spec,_=read_ref({'path':str(spec_path),'sha256':spec_sha})
    validate_protocol(spec)
    planned,_=load_inputs(spec['inputs'])
    if spec['mode']=='fixed_text_alignment':
        need(all(isinstance(w.get('fixed_text'),str) and w['fixed_text'] for p in planned.values() for w in p['windows']),
             'all planned windows need explicit fixed text')
    a=load_track(spec['baseline'],spec['fixed_controls'],planned)
    b=load_track(spec['candidate'],spec['fixed_controls'],planned)
    need(a['producer']['purpose']==b['producer']['purpose']==spec['purpose'], 'track purpose differs from protocol')
    actual_changes=[k for k in a['producer']['variables'] if a['producer']['variables'][k]!=b['producer']['variables'][k]]
    need(set(actual_changes)<=set(spec['allowed_changes']), 'undeclared engine variable changed')
    rows=[];counts=Counter();max_delta={}
    for sid, source in planned.items():
        ar,br=a['records'][sid],b['records'][sid]
        aw={x['window_id']:x for x in ar['windows']};bw={x['window_id']:x for x in br['windows']}
        details=[]
        counts['sources']+=1;counts['state_changed_sources']+=ar['decision']['state']!=br['decision']['state']
        for w in source['windows']:
            item=compare_window(aw.get(w['window_id']),bw.get(w['window_id']),w,spec['mode'],
                a['native_results'].get((sid,w['window_id'])),b['native_results'].get((sid,w['window_id'])))
            counts['planned_windows']+=1;counts['missing_windows']+=not item['both_present']
            for k,equal in item.get('ASR_equal',{}).items():counts['ASR_'+k+'_changed_windows']+=not equal
            if item.get('endpoint_delta_ms'):
                counts['same_target_measured_windows']+=1
                for k,v in item['endpoint_delta_ms'].items():max_delta[k]=max(max_delta.get(k,0),v)
            details.append(item)
        rows.append({'source_id':sid,'baseline_state':ar['decision']['state'],'candidate_state':br['decision']['state'],
                     'baseline_errors':ar['errors'],'candidate_errors':br['errors'],'windows':details})
    eligible = (spec['purpose']!='cpu_fixture' and not a['metrics']['synthetic_fixture'] and not b['metrics']['synthetic_fixture']
                and not any(v.get('synthetic_fixture') for t in (a,b) for v in t['native_results'].values())
                and a['reported_all_new_execution'] and b['reported_all_new_execution'])
    recovery=check_published_before(spec['published_before'],b['raw_rows']) if 'published_before' in spec else None
    result={'schema':'annotation-engine-comparison-report/v1','mode':spec['mode'],'spec_sha256':spec_sha,
        'source_count':len(planned),'actual_declared_changes':actual_changes,'counts':dict(counts),
        'baseline_decision_counts':a['decisions'],'candidate_decision_counts':b['decisions'],
        'maximum_same_target_endpoint_delta_ms':max_delta or None,
        'reported_new_execution_wall_ratio':a['wall_seconds']/b['wall_seconds'] if eligible else None,
        'reported_wall_ratio_is_GPU_speedup_proof':False,'GPU_runtime_verified':False,'validated_GPU_speedup':None,
        'saved_metrics_not_runtime_authenticated':True,
        'baseline_performance':{'totals':a['totals'],'wall_seconds':a['wall_seconds'],'all_new':a['reported_all_new_execution']},
        'candidate_performance':{'totals':b['totals'],'wall_seconds':b['wall_seconds'],'all_new':b['reported_all_new_execution']},
        'recovery':recovery,'all_planned_sources_accounted':True,'saved_sidecars_and_commit_bindings_verified':True,
        'source_audio_rehashed_in_this_audit':False,'human_quality_verified':False,'formal_admission':False,
        'usable_for_training':False,'usable_for_evaluation':False,'native_cross_engine_bitwise_required':False,
        'rows':rows,'contains_private_model_text_and_source_ids':True}
    if output is not None:
        out=physical(output);need(out.parent.is_dir() and not out.exists(),'fresh report in existing physical directory required')
        protected={physical(spec_path),physical(spec['inputs']['path'])}
        for track in (a,b):
            for key in ('producer','results','metrics'):protected.add(physical(track['spec'][key]['path']))
            protected.add(physical(track['producer']['contract']['path']))
            if 'native_asr_spec' in track['producer']:protected.add(physical(track['producer']['native_asr_spec']['path']))
            for row in track['raw_rows'].values():
                for key in ('sidecar','commit'):protected.add(physical(row[key]['path']))
                for ref in row.get('native_asr_refs',{}).values():protected.add(physical(ref['path']))
        if 'published_before' in spec:protected.add(physical(spec['published_before']['path']))
        need(all(out!=p and p.parent not in out.parents for p in protected),'report overlaps producer/input tree')
        atomic_new(out,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);result=audit(a.spec,a.spec_sha,a.output)
    print({'source_count':result['source_count'],'GPU_runtime_verified':False,'formal_admission':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
