"""有限原生Transformers合批的独立完整审计；旧单窗审计条件不变。

只读固定来源、已结束worker和整批JSON。不读取原音、不创建模型或数据库。
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_speed_comparison as A
from audit_speed_comparison import bound, pin, compute_execution, _finite_seconds, DECISIONS
from contracts import (atomic_new, canonical_bytes, digest, job_key, need, safe_join, utcnow,
                       model_provenance, asr_batch_profile, asr_feature_profile)
from readonly_progress import inspect_progress, SLOT_BYTES, strict_json
from work_queue import decode_completion_slot
from schema import assert_valid_record
from normalization import normalize_with_audit, POLICY
from native_transformers_batch import POLICY as NATIVE_POLICY, request_identity, MAX_PLAN_ROWS

SPEC_SCHEMA='redimnet2-annotation-native-batch-track/v1'
MAX_SOURCES=4096
MAX_READBACK_BYTES=64*1024**2
PREPROCESSING_SOURCE_ROLES={'whisper_feature_extractor','sequence_feature_extractor','batch_feature',
    'processor_mixin','audio_utils','qwen_processor','sdk_normalizer'}


def feature_source_bindings(contract):
    """离线只核声明和原模型／SDK pins；实际类及文件由活体worker核验。"""
    value=contract.get('asr_preprocessing_sources')
    if asr_feature_profile(contract)is None:
        need(value is None,'preprocessing sources require explicit B1 feature protocol')
        return None
    need(isinstance(value,dict)and set(value)=={'schema','files','processor_config'}
         and value['schema']=='annotation-official-b1-feature-sources/v1'
         and isinstance(value['files'],dict)and set(value['files'])==PREPROCESSING_SOURCE_ROLES,
         'explicit seven preprocessing source bindings required')
    for ref in [*value['files'].values(),value['processor_config']]:
        need(isinstance(ref,dict)and set(ref)=={'path','sha256'}and isinstance(ref['path'],str)
             and Path(ref['path']).is_absolute()and '..'not in Path(ref['path']).parts
             and isinstance(ref['sha256'],str)and len(ref['sha256'])==64
             and all(c in '0123456789abcdef'for c in ref['sha256']), 'invalid preprocessing source/config pin')
    for role in ('qwen_processor','sdk_normalizer'):
        ref=value['files'][role]
        # 合同中的SDK路径已固定为规范绝对路径；离线逐row校验不做resolve/stat。
        pins=contract.get('source_files',{})
        need(pins.get(ref['path'])==ref['sha256'],
             'B1 preprocessing source differs from pinned original SDK')
    model=contract['models']['asr'];ref=value['processor_config']
    need(Path(ref['path']).name=='preprocessor_config.json'
         and model['files'].get('preprocessor_config.json',{}).get('sha256')==ref['sha256'],
         'B1 processor config differs from actual model manifest')
    return value


def validate_contract(contract):
    need(contract.get('schema')=='redimnet2-production-contract/v2' and contract.get('purpose') in {'pilot_validation','cpu_fixture'} and
         contract.get('model_API_calls_allowed') == 0 and contract.get('release_training_labels') is False,
         'native finite audit requires offline unreleased pilot contract')
    profile=asr_batch_profile(contract)
    asr_feature_profile(contract)
    feature_source_bindings(contract)
    need(profile is not None and contract['runtime']['execution_policy']==NATIVE_POLICY,
         'native auditor refuses old single-window or undeclared batching')
    need(contract['runtime'].get('dtype')=='float32' and contract['runtime'].get('attention')=='eager'
         and contract['runtime'].get('max_batch')==1 and contract['runtime'].get('TF32')is False,
         'Ultra precision and single-window policy changed')
    need(type(contract['runtime'].get('max_new_tokens'))is int and contract['runtime']['max_new_tokens']>0,
         'native token limit must be a positive integer')
    return profile


def compare_contracts(a,b,mode):
    for c in (a,b):validate_contract(c)
    if mode=='recovery':need(a==b,'native recovery requires exact same contract');return
    for key in ('models','source_files','alignment','decoder','quality_policy','engine_source_files','asr_preprocessing_sources'):
        need(a.get(key)==b.get(key),'native fixed producer field differs: '+key)
    left,right=copy.deepcopy(a['runtime']),copy.deepcopy(b['runtime'])
    for c in (left,right):c['asr_batching'].pop('max_requests')
    need(left==right,'native comparison may change only declared ASR batch width')


def validate_call_counts(metrics,contract):
    profile=validate_contract(contract)
    calls=metrics.get('asr_call_rows',{})
    need(isinstance(calls,dict) and all(isinstance(k,str) and k.isdigit() and 1<=int(k)<=profile['max_requests']
         and str(int(k))==k and type(v)is int and v>=0 for k,v in calls.items()),'actual native ASR call width invalid')
    need(sum(int(k)*v for k,v in calls.items())==metrics['asr_rows'] and sum(calls.values())==metrics['asr_batches'],
         'actual native ASR width/row/batch denominator differs')
    fa=metrics.get('aligner_call_rows',{})
    need(set(fa)<={'1'} and metrics['aligner_rows']==metrics['aligner_batches']==fa.get('1',0),
         'actual Ultra calls are not independent single rows')


def validate_preprocessing_sources_observation(metrics,contract):
    bindings=feature_source_bindings(contract)
    if bindings is None:return None
    observed=metrics.get('asr_preprocessing_sources_observed')
    if contract['purpose']=='cpu_fixture'and observed is None:return None
    need(isinstance(observed,dict)and observed.get('verified_from_live_objects')is True
         and {k:observed.get(k)for k in ('schema','files','processor_config')}==bindings,
         'actual worker preprocessing source/config observation differs from contract')
    config=observed.get('actual_feature_extractor_config')
    need(isinstance(config,dict)and config.get('sampling_rate')==16000
         and all(type(config.get(k))is int and config[k]>0 for k in ('feature_size','hop_length','n_fft')),
         'actual live feature extractor configuration missing')
    return observed


def validate_native_preprocessing(native,contract,rows,position,request=None):
    """核独立B1参考及真正generate张量观察；PCM相同不能代替此证明。"""
    profile=asr_feature_profile(contract)
    if profile is None:
        need('preprocessing'not in native,'undeclared native B1 feature preprocessing proof')
        return None
    p=native.get('preprocessing')
    need(isinstance(p,dict)and p.get('schema')=='annotation-official-b1-feature-proof/v1'
         and p.get('policy')==profile['policy'],'actual independent B1 preprocessing proof missing')
    bindings=feature_source_bindings(contract)
    if contract['purpose']=='cpu_fixture'and p.get('source_bindings_verified')is False:
        need(p.get('source_bindings_sha256')is None,'synthetic preprocessing source flag contradicts digest')
    else:
        need(p.get('source_bindings_verified')is True and p.get('source_bindings_sha256')==digest(bindings),
             'actual B1 preprocessing source bindings are not the declared live sources')
    need(type(rows)is int and 1<=rows<=asr_batch_profile(contract)['max_requests']
         and type(position)is int and 0<=position<rows,'actual B1 preprocessing batch width/position invalid')
    sha=lambda v:isinstance(v,str)and len(v)==64 and all(c in '0123456789abcdef'for c in v)
    need(all(sha(p.get(k))for k in ('original_input_pcm_sha256','sdk_input_pcm_sha256',
                                   'feature_valid_sha256','actual_generate_valid_feature_sha256')),
         'B1 feature/SDK PCM SHA missing')
    need(isinstance(p.get('row_id'),str)and p['row_id']and type(p.get('frames'))is int
         and 0<p['frames']<=480000 and type(p.get('position'))is int and p['position']==position,
         'B1 preprocessing source/frame/position identity differs')
    need(p.get('official_processor_rows')==1 and type(p['official_processor_rows'])is int
         and p.get('official_processor_kwargs')==dict(return_tensors='pt',padding=True),
         'B1 reference did not use independent official single-row processor')
    shape=p.get('feature_shape');valid=p.get('valid_feature_frames');padded=p.get('actual_generate_feature_padded_frames')
    need(isinstance(shape,list)and len(shape)==2 and all(type(n)is int and n>0 for n in shape)
         and type(valid)is int and 0<valid<=10000 and shape[1]==valid and shape[0]<=1024
         and p.get('feature_dtype')=='float32','B1 valid mel shape/dtype differs')
    need(type(padded)is int and valid<=padded<=10000
         and p.get('actual_generate_feature_shape')==[shape[0],padded]
         and p.get('actual_generate_feature_dtype')=='float32'
         and p.get('actual_generate_feature_padding_is_zero')is True,
         'actual generate mel shape/dtype/padding proof differs')
    mask=p.get('feature_attention_mask');actual_mask=p.get('actual_generate_feature_mask')
    need(isinstance(mask,list)and all(type(v)is int for v in mask)and mask==[1]*valid
         and isinstance(actual_mask,list)and all(type(v)is int for v in actual_mask)
         and actual_mask==[1]*valid+[0]*(padded-valid),'actual valid feature mask differs from independent B1')
    need(p.get('feature_attention_mask_dtype')in {'int32','int64'}
         and p.get('actual_generate_feature_mask_dtype')==p['feature_attention_mask_dtype'],
         'actual feature mask dtype differs from independent official B1')
    need(p['actual_generate_valid_feature_sha256']==p['feature_valid_sha256'],
         'actual generate valid mel differs from independent B1 reference')
    ids=p.get('baseline_input_token_ids');count=p.get('baseline_nonpad_prompt_tokens');audio_id=p.get('audio_token_id')
    need(isinstance(ids,list)and ids and all(type(v)is int and v>=0 for v in ids)
         and type(count)is int and count==len(ids)and type(audio_id)is int and audio_id>=0,
         'independent B1 nonpadding prompt identity missing')
    actual_ids=native.get('input_token_ids');prompt_mask=native.get('input_attention_mask')
    need(isinstance(actual_ids,list)and isinstance(prompt_mask,list)and len(actual_ids)==len(prompt_mask)
         and all(type(v)is int and v in (0,1)for v in prompt_mask)
         and [v for v,m in zip(actual_ids,prompt_mask)if m]==ids
         and p.get('actual_generate_nonpad_prompt_token_ids')==ids
         and native.get('input_nonpad_tokens')==count,
         'actual generate nonpadding prompt differs from independent B1')
    placeholders=sum(v==audio_id for v in ids)
    need(type(p.get('audio_placeholder_count'))is int and placeholders>0
         and p['audio_placeholder_count']==placeholders==p.get('actual_generate_audio_placeholder_count'),
         'actual audio placeholder count differs from independent B1')
    need(type(p.get('actual_generate_rows'))is int and p['actual_generate_rows']==rows
         and type(p.get('actual_generate_row_index'))is int and p['actual_generate_row_index']==position
         and p.get('verified_from_actual_generate_tensors')is True,
         'B1 proof is not bound to actual generate row tensors')
    need(p.get('SDK_normalization_changed_PCM')is(p['original_input_pcm_sha256']!=p['sdk_input_pcm_sha256']),
         'official SDK normalization PCM observation contradicts hashes')
    if request is not None:
        need(p['row_id']==request['row_id']and p['frames']==request['model_input_frames']
             and p['original_input_pcm_sha256']==request['input_pcm_sha256'],
             'B1 feature proof refers to another source PCM/window')
    return p


def validate_native_tokens(native,contract,rows,position):
    need(isinstance(native,dict),'native generation evidence missing')
    raw=native.get('generated_token_ids');eos=native.get('eos_token_ids');mask=native.get('input_attention_mask')
    need(isinstance(raw,list) and all(type(v)is int and v>=0 for v in raw) and
         isinstance(eos,list) and eos and len(eos)==len(set(eos)) and all(type(v)is int and v>=0 for v in eos),
         'native token/EOS identity missing')
    limit=contract['runtime']['max_new_tokens'];need(native.get('max_new_tokens')==limit and len(raw)<=limit,'actual native token cap differs')
    first=next((i for i,v in enumerate(raw)if v in eos),None)
    effective=raw if first is None else raw[:first+1];post=[] if first is None else raw[first+1:]
    need(native.get('eos_observed')is(first is not None) and native.get('first_eos_index')==first and
         native.get('effective_generated_token_ids')==effective and native.get('effective_generated_token_count')==len(effective)
         and native.get('raw_returned_token_count')==len(raw) and native.get('post_eos_token_ids')==post,
         'native effective tokens/EOS/padding evidence contradicts raw matrix')
    pad=native.get('padding_token_id')
    need(type(pad)is int and pad>=0 and all(v==pad or v in eos for v in post),'unexplained native tokens after first EOS')
    padding=native.get('padding_policy')
    need(isinstance(padding,dict) and set(padding)=={'original_pad_token_id','pad_token_id','eos_token_ids','padding_side','policy'}
         and padding['pad_token_id']==pad and padding['eos_token_ids']==eos and padding['padding_side']=='left'
         and padding['policy']=='explicit_existing_pad_else_first_configured_EOS/v1',
         'native actual explicit padding policy missing')
    original=padding['original_pad_token_id']
    need((original is None and pad==eos[0])or(type(original)is int and original>=0 and pad==original),
         'native padding does not follow observed original configuration')
    need(type(native.get('actual_generate_rows'))is int and native['actual_generate_rows']==rows and
         type(native.get('actual_generate_row_index'))is int and native['actual_generate_row_index']==position,
         'actual native row map differs')
    width=native.get('input_padded_tokens')
    prompt=native.get('input_token_ids')
    need(type(width)is int and width>0 and isinstance(mask,list) and len(mask)==width and
         all(type(v)is int and v in (0,1)for v in mask) and sorted(mask)==mask and sum(mask)>0 and
         native.get('input_nonpad_tokens')==sum(mask),'native actual left-padding mask geometry differs')
    need(isinstance(prompt,list)and len(prompt)==width and all(type(v)is int and v>=0 for v in prompt),
         'native actual input token row missing or differs from padded geometry')
    need(native.get('model_training')is False and native.get('inference_mode')is True and native.get('grad_enabled')is False,
         'native actual generation was not eval/inference-only')
    validate_native_preprocessing(native,contract,rows,position)
    return first


def verify_native_window(record,window,contract,contract_sha,bundle_sha,tree,cache):
    asr=window.get('asr',{})
    if window.get('preprocessing_status','completed')!='completed':
        need(asr.get('status')=='failed' and not asr.get('durable_native_batch'),'preprocessing failure must retain failed evidence without invented batch')
        return None
    ref=asr.get('durable_native_batch')
    need(isinstance(ref,dict) and set(ref)=={'plan_sha256','cohort_sha256','packet_path','packet_sha256','position','cache_hit'}
         and type(ref['cache_hit'])is bool,'actual durable native batch reference missing')
    path=Path(ref['packet_path']);native_root=Path(tree)/'.native_torch_cohorts'
    need(path.is_absolute() and '..'not in path.parts and native_root in path.parents and path.parent.name=='batches' and path.name==ref['cohort_sha256']+'.json',
         'native batch packet outside this producer tree')
    key=(str(path),ref['packet_sha256'])
    if key not in cache:
        packet,_,_=bound(path,ref['packet_sha256']);content=packet.get('content')
        need(packet.get('schema')=='annotation-native-transformers-atomic-result/v1' and isinstance(content,dict)
             and packet.get('content_sha256')==digest(content),'whole native batch packet digest differs')
        plan,plan_sha,_=bound(path.parent.parent/'plan.json',ref['plan_sha256'])
        identity=content['store_identity'];saved_identity,_,_=bound(path.parent.parent/'identity.json')
        need(identity==saved_identity==plan.get('store_identity') and content.get('plan_sha256')==plan_sha,
             'native plan/store identity differs')
        profile=validate_contract(contract);model_sha=digest(model_provenance(contract)['asr'])
        producer=digest(dict(contract_sha256=contract_sha,bundle_sha256=bundle_sha,model_sha256=model_sha,
                            execution_policy=NATIVE_POLICY,asr_batching=profile))
        need(identity.get('schema')=='annotation-native-transformers-file-store/v1' and identity.get('producer_sha256')==producer
             and identity.get('contract_sha256')==contract_sha and identity.get('model_sha256')==model_sha
             and identity.get('policy')==NATIVE_POLICY and identity.get('max_batch_rows')==profile['max_requests']
             and identity.get('alignment')==dict(dtype='float32',attention='eager',max_batch_rows=1)
             and identity.get('database_created')is False and identity.get('model_API_calls')==0
             and identity.get('release_training_labels')is False
             and identity.get('synthetic_fixture')is(contract['purpose']=='cpu_fixture'),'native actual producer/model/contract differs')
        requests=plan.get('requests');need(isinstance(requests,list) and 1<=len(requests)<=MAX_PLAN_ROWS
             and len({r['row_id']for r in requests})==len(requests) and all(request_identity(r)==r for r in requests),'invalid fixed native plan requests')
        need(requests==sorted(requests,key=lambda r:(r['model_input_frames'],r['row_id'])),'native original group ordering differs')
        cohorts=[]
        for start in range(0,len(requests),profile['max_requests']):
            intent=dict(schema='annotation-native-transformers-cohort/v1',store_identity_sha256=digest(identity),
                        requests=requests[start:start+profile['max_requests']],original_plan_requests_sha256=digest(requests))
            cohorts.append(dict(cohort_sha256=digest(intent),intent=intent))
        need(plan.get('schema')=='annotation-native-transformers-plan/v1' and plan.get('cohorts')==cohorts
             and content.get('cohort')in cohorts and content['cohort']['cohort_sha256']==ref['cohort_sha256'],
             'native whole fixed cohort plan changed')
        expected=content['cohort']['intent']['requests'];outputs=content['outputs'];ids=[r['row_id']for r in expected]
        need(set(outputs)==set(ids),'native whole packet output coverage differs')
        generated_widths=set();input_widths=set()
        for i,row_id in enumerate(ids):
            value=outputs[row_id];end=validate_native_tokens(value.get('native'),contract,len(ids),i)
            validate_native_preprocessing(value['native'],contract,len(ids),i,expected[i])
            generated_widths.add(len(value['native']['generated_token_ids']));input_widths.add(value['native']['input_padded_tokens'])
            need(value.get('status')==('completed'if end is not None else'truncated'),'native status does not match actual EOS')
            need(value.get('batch_rows')==ids and value.get('batch_size')==len(ids),'native result does not bind original cohort order')
            call=value.get('inference_call',{})
            need(call.get('stage')=='asr' and call.get('rows')==len(ids) and call.get('producer_row_ids')==ids
                 and call.get('input_frames')==[r['model_input_frames']for r in expected]
                 and call.get('execution_policy')==NATIVE_POLICY,'native actual inference call differs from whole cohort')
            audit=normalize_with_audit(value['raw_text'],contract['runtime'].get('normalization_profile',POLICY))
            need(value.get('normalization_audit')==audit and value.get('normalized_text')==audit['normalized_text'],'native text normalization differs')
        need(len(generated_widths)==len(input_widths)==1,'actual native generate matrices were not rectangular')
        cache[key]=(content,expected)
    content,expected=cache[key];source=record['source'];row_id=source['source_id']+':'+window['window_id'];position=ref['position']
    need(type(position)is int and 0<=position<len(expected),'native cohort row position invalid')
    request=expected[position]
    need(request==dict(row_id=row_id,source_id=source['source_id'],window_id=window['window_id'],
         source_record_sha256=digest(source),input_pcm_sha256=window['input_pcm_sha256'],
         model_input_frames=window['model_input_frames'],model_sample_rate=window['model_sample_rate']),
         'native actual source/PCM window binding differs')
    need({k:v for k,v in asr.items()if k!='durable_native_batch'}==content['outputs'][row_id],
         'sidecar ASR is not the committed whole native packet row')
    return dict(packet_key=key,rows=len(expected),cache_hit=ref['cache_hit'],row_id=row_id)

def summarize_workers(parent, resource_root, expected_receipts, contract):
    """核实际worker终态摘要与发布集合；资源归还不代表此刻仍空闲。"""
    need(parent.get('resource_returned') is True and parent.get('full_sidecar_readback') is True,
         'run has no successful terminal publication/resource return')
    parts = parent.get('workers', [])
    need(parts and len(parts) == parent.get('worker_count'), 'worker terminal denominator differs')
    seen_workers, seen_sources = set(), set(); sums = Counter(); workers = []
    for part in parts:
        wid = part['worker_id']; need(wid not in seen_workers, 'duplicate worker'); seen_workers.add(wid)
        summary, summary_sha, _ = bound(safe_join(resource_root, wid + '/worker_summary.json'))
        terminal = part['terminal']
        need(summary == terminal['summary'] and summary_sha == terminal['summary_sha256'], 'parent/worker summary differs')
        ready, ready_sha, _ = bound(safe_join(resource_root, wid + '/terminal_ready.json'))
        ack, _, _ = bound(safe_join(resource_root, wid + '/terminal_ack.json'))
        need(ready['summary_sha256'] == summary_sha and terminal['ready_sha256'] == ready_sha == ack['ready_sha256'],
             'worker ready/ACK digest differs')
        need(ready['nonce'] == ack['nonce'] and summary['worker_id'] == wid, 'worker nonce/ID differs')
        need(ready['bundle_sha256'] == parent['bundle_sha256'], 'worker bundle identity differs')
        for key in ('contract_sha256', 'index_sha256'):
            need(summary[key] == parent[key] == ready[key], 'worker input identity differs: ' + key)
        need(summary['device_uuid'] == part['uuid'], 'actual worker device differs')
        if contract.get('purpose') != 'cpu_fixture':
            att, att_sha, _ = bound(safe_join(resource_root, wid + '/resource_attestation.json'))
            challenge, challenge_sha, _ = bound(safe_join(resource_root, wid + '/context_challenge/attestation.json'))
            need(att_sha == ready['resource_attestation_sha256'] and att['challenge_attestation_sha'] == challenge_sha,
                 'resource challenge/ready digest differs')
            need(att['worker_id'] == wid and att['nonce'] == ready['nonce'] and att['UUIDs'] == [part['uuid']]
                 and att['contract_sha256'] == parent['contract_sha256'] and att['index_sha256'] == parent['index_sha256']
                 and att['bundle_sha256'] == parent['bundle_sha256'], 'active resource identity differs')
            need(challenge['complete'] is True and challenge['all_free_phases_returned_to_baseline'] is True
                 and len(challenge['phases']) == 7 and challenge['UUIDs'] == [part['uuid']]
                 and challenge['known_container_worker_pid'] == att['known_container_worker_pid'] == ready['worker_pid'],
                 'saved seven-phase resource context proof differs')
        metrics = summary['backend_metrics']
        validate_preprocessing_sources_observation(metrics,contract)
        need(summary['model_API_calls'] == metrics['model_API_calls'] == 0, 'nonlocal model calls observed')
        source_ids = set(); fresh_count = 0
        for ref in summary['chunk_receipt_refs']:
            item = expected_receipts[ref['id']]
            need(ref['sha256'] == item['sha256'] and ref['source_count'] == item['value']['source_count'], 'worker chunk binding differs')
            ids = {r['source_id'] for r in item['value']['sources']}
            need(not source_ids & ids and not seen_sources & ids, 'worker source ownership overlap')
            source_ids.update(ids); fresh_count += len(item['fresh_source_ids'])
        seen_sources.update(source_ids)
        need(len(source_ids) == summary['assigned_source_rows'] == summary['sidecars_committed'] + summary['sidecar_cache_hits'],
             'new/whole-sidecar reuse denominator differs')
        need(fresh_count == summary['sidecars_committed'], 'new sidecar count differs from actual publication attempts')
        for key in ('assigned_source_rows', 'sidecars_committed', 'sidecar_cache_hits', 'model_windows', 'decoded_audio_seconds'):
            sums[key] += summary[key]
        for key in ('asr_rows', 'aligner_rows', 'asr_batches', 'aligner_batches', 'asr_seconds', 'aligner_seconds', 'model_load_seconds'):
            need(key in metrics, 'missing actual backend metric: ' + key)
            if key.endswith('seconds'): _finite_seconds(metrics[key], key)
            else: need(type(metrics[key]) is int and metrics[key] >= 0, 'invalid actual invocation count')
            sums[key] += metrics[key]
        validate_call_counts(metrics, contract)
        for stage in ('asr', 'aligner'):
            sums[stage + '_stage_cache_hits'] += summary['stage_cache_hits'][stage]
        native_hits=metrics.get('native_ASR_stage_cache_hits',0)
        need(type(native_hits)is int and native_hits>=0,'invalid native cache hit count')
        sums['native_ASR_stage_cache_hits']+=native_hits
        sums['worker_elapsed_seconds'] += _finite_seconds(summary['elapsed_seconds'], 'worker elapsed')
        workers.append({'worker_id': wid, 'summary_sha256': summary_sha, 'assigned_sources': len(source_ids),
                        'new_sidecars': summary['sidecars_committed'], 'whole_sidecar_reuse': summary['sidecar_cache_hits'],
                        'backend_metrics': metrics, 'compute_execution': compute_execution(metrics, contract),
                        'worker_elapsed_seconds': summary['elapsed_seconds']})
    all_expected = {r['source_id'] for item in expected_receipts.values() for r in item['value']['sources']}
    need(seen_sources == all_expected, 'worker publication union is incomplete')
    need(len({p['uuid'] for p in parts}) == len(parts), 'worker GPU identities repeat')
    if contract.get('purpose') != 'cpu_fixture':
        returned, _, _ = bound(Path(resource_root) / 'resource_return_double_probe.json')
        need(len(returned['snapshots']) == 2 and returned['interval_seconds'] >= 5, 'saved resource return double probe missing')
        for snapshot in returned['snapshots']:
            need(not snapshot['compute_processes'] and {d['uuid'] for d in snapshot['devices']} == {p['uuid'] for p in parts}
                 and all(d['used_mib'] <= 256 for d in snapshot['devices']), 'saved resource return was not idle')
    wall = _finite_seconds(parent['parent_elapsed_seconds'], 'parent elapsed')
    need(wall > 0, 'zero elapsed cannot establish throughput')
    return {'workers': workers, 'totals': dict(sums), 'parent_wall_seconds': wall,
            'gpu_lease_wall_seconds': _finite_seconds(parent['gpu_lease_wall_seconds'], 'GPU lease wall'),
            'all_assigned_sources_per_parent_second': len(all_expected) / wall,
            'new_sidecars_per_parent_second': sums['sidecars_committed'] / wall,
            'all_sources_recomputed_cold': sums['sidecar_cache_hits'] == sums['asr_stage_cache_hits'] == sums['aligner_stage_cache_hits'] == sums['native_ASR_stage_cache_hits'] == 0,
            'resource_returned_at_saved_run_end': True, 'current_GPU_availability_checked': False,
            'historical_resource_digest_chain_verified': contract.get('purpose') != 'cpu_fixture',
            'actual_compute_execution_verified': bool(sums['asr_rows'] > 0 and all(w['compute_execution']['verified']
                for w in workers if w['backend_metrics']['asr_rows'] > 0))}


def _load_track(spec_path, expected_sha):
    spec, spec_sha, _ = bound(spec_path, expected_sha)
    need(spec.get('schema') == SPEC_SCHEMA, 'wrong track spec schema')
    contract, contract_sha, _ = pin(spec, 'contract'); index, index_sha, _ = pin(spec, 'index')
    validate_contract(contract)
    need(type(index.get('source_count')) is int and 0 < index['source_count'] <= MAX_SOURCES,
         'finite validation source limit exceeded; this is not a full-library auditor')
    for key in ('queue_root', 'output_root', 'resources_root'):
        need(Path(spec[key]).is_absolute(), 'absolute track directory required: ' + key)
    queue, output, resource = (Path(spec[k]) for k in ('queue_root', 'output_root', 'resources_root'))
    progress = inspect_progress(spec['index']['path'], index_sha, queue, contract_sha)
    need(progress['committed_source_lower_bound'] == index['source_count'] and not progress['snapshot_leased_chunks_excluded'],
         'track is incomplete; running chunks cannot be silently omitted')
    tree = safe_join(output, contract['version'])
    identity, _, _ = bound(tree / 'identity.json')
    need(identity == {'schema': 'redimnet2-sidecar-tree/v1', 'version': contract['version'], 'contract_sha256': contract_sha},
         'versioned sidecar identity differs')
    records, receipts, decisions, audit_bytes = {}, {}, Counter(), 0
    source_file_refs={}
    with (queue / 'completion_map.bin').open('rb') as completion:
        for shard in index['shards']:
            slot = decode_completion_slot(completion.read(SLOT_BYTES))
            chunk, chunk_sha, size = bound(queue / 'completed' / (shard['id'] + '.json'), slot['commit_sha256']); audit_bytes += size
            need(chunk['source_count'] == shard['count'] and chunk['shard_id'] == shard['id']
                 and chunk['shard_sha256'] == shard['sha256'] and chunk['index_sha256'] == index_sha
                 and chunk['contract_sha256'] == contract_sha, 'committed chunk input identity differs')
            need(digest(sorted(r['source_id'] for r in chunk['sources'])) == slot['source_ids_sha256'], 'slot source IDs differ')
            receipt_by_id = {r['source_id']: r for r in chunk['sources']}
            need(len(receipt_by_id) == len(chunk['sources']) == shard['count'], 'receipt source denominator differs')
            shard_path = safe_join(Path(spec['index']['path']).parent, shard['path'])
            raw = shard_path.read_bytes(); need(hashlib.sha256(raw).hexdigest() == shard['sha256'], 'input shard changed')
            sources = [strict_json(line) for line in raw.splitlines() if line]
            need(len(sources) == shard['count'] and {s['source_id'] for s in sources} == set(receipt_by_id), 'source shard/receipt union differs')
            local_counts = Counter(); fresh_ids = set()
            for source in sources:
                sid = source['source_id']; need(sid not in records, 'duplicate source in validation set')
                receipt = receipt_by_id[sid]; path = safe_join(tree, source['sidecar_relpath'])
                record, record_sha, size = bound(path, receipt['sidecar_sha256']); audit_bytes += size
                assert_valid_record(record)
                need(record['source'] == source and record['contract']['sha256'] == contract_sha
                     and record['execution']['job_key'] == receipt['job_key'] == job_key(source, contract_sha), 'sidecar source/contract/job differs')
                need(record['provenance'].get('runtime') == contract['runtime'], 'saved effective runtime differs from track contract')
                need(record['provenance'].get('models') == model_provenance(contract), 'saved model identity differs from producer contract')
                if record['execution']['attempt_id'] == chunk['attempt_id']: fresh_ids.add(sid)
                need(not record['decision']['usable_for_training'] and not record['decision']['usable_for_evaluation'], 'automatic eligibility release refused')
                commit, commit_sha, _ = bound(path.with_name(path.name + '.commit.json'), receipt['commit_sha256'])
                need(commit == {'schema': 'redimnet2-sidecar-commit/v1', 'source_id': sid, 'job_key': receipt['job_key'],
                     'contract_sha256': contract_sha, 'source_record_sha256': digest(source), 'sidecar_sha256': record_sha, 'bytes': size}, 'per-source commit differs')
                need(receipt['sidecar_bytes'] == size and receipt['decision'] == record['decision']['state'], 'receipt byte/decision differs')
                source_file_refs[sid]=dict(sidecar=dict(path=str(path),sha256=record_sha),
                    commit=dict(path=str(path.with_name(path.name+'.commit.json')),sha256=commit_sha))
                local_counts[record['decision']['state']] += 1
                records[sid] = record
            need({k: local_counts[k] for k in DECISIONS} == slot['decision_counts'], 'sidecar/completion decision counts differ')
            decisions.update(local_counts); receipts[shard['id']] = {'sha256': chunk_sha, 'value': chunk, 'fresh_source_ids': fresh_ids}
    need(len(records) == index['source_count'], 'complete source denominator differs')
    parent, parent_sha, _ = bound(resource / 'parent_result.json')
    need(parent['contract_sha256'] == contract_sha and parent['index_sha256'] == index_sha
         and parent.get('model_API_calls') == 0, 'parent terminal identity differs')
    for record in records.values():
        need(record['provenance']['code_sha256'] == parent['bundle_sha256'], 'source producer bundle differs from this track; separate reused producer identity explicitly')
    performance = summarize_workers(parent, resource, receipts, contract)
    need(performance['totals']['assigned_source_rows'] == len(records), 'worker source denominator differs')
    return {'spec': spec, 'spec_sha256': spec_sha, 'contract': contract, 'contract_sha256': contract_sha,
            'index_sha256': index_sha, 'records': records, 'decision_counts': dict(decisions),
            'performance': performance, 'parent_result_sha256': parent_sha, 'sidecar_bytes_read': audit_bytes,
            '_fresh_source_ids':{sid for item in receipts.values()for sid in item['fresh_source_ids']},
            '_source_file_refs':source_file_refs}


def load_track(spec_path,expected_sha):
    track=_load_track(spec_path,expected_sha);contract=track['contract'];contract_sha=track['contract_sha256']
    fresh_source_ids=track.pop('_fresh_source_ids')
    tree=Path(track['spec']['output_root'])/contract['version'];cache={};new_packets={};cache_hits=Counter();new_row_ids={};fa_rows=Counter()
    for sid,record in track['records'].items():
        fresh=sid in fresh_source_ids;wid=record['execution']['worker_id']
        for window in record.get('windows',[]):
            proof=verify_native_window(record,window,contract,contract_sha,record['provenance']['code_sha256'],tree,cache)
            if proof is None:continue
            alignment=window.get('alignment',{});call=alignment.get('inference_call')
            if call is not None:
                need(call.get('stage')=='aligner' and call.get('rows')==1 and call.get('producer_row_ids')==[proof['row_id']]
                     and call.get('input_frames')==[window['model_input_frames']],'saved Ultra actual call changed from single-window')
            if fresh:
                if proof['cache_hit']:cache_hits[wid]+=1
                else:
                    new_packets.setdefault(wid,{})[proof['packet_key']]=proof['rows']
                    new_row_ids.setdefault(wid,set()).add(proof['row_id'])
                if call is not None and not window.get('aligner_cache_hit',False):fa_rows[wid]+=1
    for worker in track['performance']['workers']:
        wid=worker['worker_id'];metrics=worker['backend_metrics'];packets=new_packets.get(wid,{})
        widths=Counter(str(width)for width in packets.values())
        need(dict(widths)==metrics['asr_call_rows'] and sum(packets.values())==metrics['asr_rows']
             and len(packets)==metrics['asr_batches'] and sum(packets.values())==len(new_row_ids.get(wid,set())),
             'actual native model metrics do not match new durable cohort publications')
        need(cache_hits[wid]==metrics.get('native_ASR_stage_cache_hits',0),'native stage cache hits do not match fresh sidecar references')
        need(fa_rows[wid]==metrics['aligner_rows'],'actual Ultra metrics do not match fresh single-window publications')
    track['native_batch_packets_read']=len(cache);track['native_whole_batch_readback_verified']=True
    track['performance']['native_whole_batch_readback_verified']=True
    if asr_feature_profile(contract)is not None:
        count=verify_track_preprocessing(track)
        track['native_B1_preprocessing_verified_windows']=count
        track['native_B1_preprocessing_verified']=True
    track['spec_pin']=dict(path=str(Path(spec_path).absolute()),sha256=expected_sha)
    track['_native_readback_receipt']=_readback_from_verified(track,cache)
    track.pop('_source_file_refs')
    return track


def _readback_from_verified(track,cache):
    """只使用本次已核内存对象生成完整有限来源和批次引用，不再打开文件。"""
    sources=[]
    for sid,record in sorted(track['records'].items()):
        windows=[]
        for window in record.get('windows',[]):
            ref=window.get('asr',{}).get('durable_native_batch')
            request=None
            if ref is not None:
                content,expected=cache[(ref['packet_path'],ref['packet_sha256'])]
                request=copy.deepcopy(expected[ref['position']])
            windows.append(dict(window_id=window['window_id'],preprocessing_status=window.get('preprocessing_status','completed'),
                request=request,durable_native_batch=copy.deepcopy(ref)))
        sources.append(dict(source_id=sid,source_record_sha256=digest(record['source']),
            **track['_source_file_refs'][sid],windows=windows,errors_sha256=digest(record.get('errors',[])),
            decision_state=record['decision']['state']))
    packets=[]
    for (path,sha),(content,expected) in sorted(cache.items()):
        packets.append(dict(packet=dict(path=path,sha256=sha),
            plan=dict(path=str(Path(path).parent.parent/'plan.json'),sha256=content['plan_sha256']),
            cohort_sha256=content['cohort']['cohort_sha256'],store_identity_sha256=digest(content['store_identity']),
            requests=copy.deepcopy(expected)))
    bundle={r['provenance']['code_sha256']for r in track['records'].values()}
    need(len(bundle)==1 and len(sources)==track['performance']['totals']['assigned_source_rows'],
         'finite readback source/bundle denominator differs')
    return dict(schema='annotation-native-finite-readback/v1',purpose=track['contract']['purpose'],
        track_spec=track['spec_pin'],track_spec_sha256=track['spec_sha256'],
        contract_sha256=track['contract_sha256'],index_sha256=track['index_sha256'],bundle_sha256=next(iter(bundle)),
        source_count=len(sources),sources=sources,native_packets=packets,native_packet_count=len(packets),
        records_sha256=digest(track['records']),
        parent_result=dict(path=str(Path(track['spec']['resources_root'])/'parent_result.json'),sha256=track['parent_result_sha256']),
        performance=copy.deepcopy(track['performance']),full_sidecar_commit_readback_verified=True,
        native_whole_batch_readback_verified=True,model_API_calls=0,database_created=False,
        input_media_rehashed_in_this_audit=False,human_accuracy_certified=False,training_labels_released=False)


def native_readback_receipt(track):
    """供新独立选型协议固定本次有限读回；纯内存，不补造未发生的GPU证明。"""
    need(track.get('native_whole_batch_readback_verified')is True and isinstance(track.get('_native_readback_receipt'),dict),
         'native readback receipt requires actual completed finite load_track')
    value=copy.deepcopy(track['_native_readback_receipt'])
    need(value['source_count']==len(track['records']) and value['track_spec_sha256']==track['spec_sha256']
         and value['contract_sha256']==track['contract_sha256'] and value['index_sha256']==track['index_sha256'],
         'native readback receipt no longer matches track identity')
    need(value['records_sha256']==digest(track['records']),'native source observations changed after finite readback')
    need(len(canonical_bytes(value))<=MAX_READBACK_BYTES,'native finite readback receipt exceeds 64MiB; no truncation')
    return value


def verify_track_preprocessing(track):
    """只复算已读内存原件，不打开媒体、模型、侧车或整批文件。"""
    count=0
    for sid,record in track['records'].items():
        for w in record.get('windows',[]):
            if w.get('preprocessing_status','completed')!='completed':continue
            native=w.get('asr',{}).get('native',{})
            request=dict(row_id=sid+':'+w['window_id'],model_input_frames=w['model_input_frames'],
                         input_pcm_sha256=w['input_pcm_sha256'])
            validate_native_tokens(native,track['contract'],native.get('actual_generate_rows'),native.get('actual_generate_row_index'))
            validate_native_preprocessing(native,track['contract'],native.get('actual_generate_rows'),
                                          native.get('actual_generate_row_index'),request)
            count+=1
    return count


def compare_record(a,b,mode='speed'):
    result=A.compare_record(a,b,mode)
    result['complete_source_observations']={name:{key:record.get(key)for key in ('errors','units','decision','quality_axes','vad')}
                                          for name,record in (('baseline',a),('candidate',b))}
    wa={w['window_id']:w for w in a.get('windows',[])};wb={w['window_id']:w for w in b.get('windows',[])}
    for row in result['windows']:
        if not row['both_present']:continue
        left,right=wa[row['window_id']],wb[row['window_id']]
        for name,window in (('baseline',left),('candidate',right)):
            row['observations'][name]['ASR_native_complete']=copy.deepcopy(window.get('asr',{}).get('native'))
            row['observations'][name]['durable_native_batch']=copy.deepcopy(window.get('asr',{}).get('durable_native_batch'))
        for key in ('effective_generated_token_ids','post_eos_token_ids','raw_returned_token_count'):
            row['ASR'][key+'_equal']=left.get('asr',{}).get('native',{}).get(key)==right.get('asr',{}).get('native',{}).get(key)
        pl=left.get('asr',{}).get('native',{}).get('preprocessing')
        pr=right.get('asr',{}).get('native',{}).get('preprocessing')
        if pl is not None or pr is not None:
            # 批宽导致的总padding宽度允许不同，有效特征、mask、prompt必须逐条相同。
            for name,keys in (
                ('policy',('policy',)),
                ('source_bindings',('source_bindings_sha256','source_bindings_verified')),
                ('SDK_PCM',('original_input_pcm_sha256','sdk_input_pcm_sha256','frames')),
                ('valid_mel',('feature_shape','feature_dtype','feature_valid_sha256','valid_feature_frames')),
                ('valid_feature_mask',('feature_attention_mask',)),
                ('nonpad_prompt',('baseline_input_token_ids','baseline_nonpad_prompt_tokens','audio_token_id','audio_placeholder_count')),
                ('official_single_processor',('official_processor_rows','official_processor_kwargs'))):
                row['ASR']['preprocessing_'+name+'_equal']=bool(isinstance(pl,dict)and isinstance(pr,dict)
                    and all(pl.get(k)==pr.get(k)for k in keys))
        if mode=='recovery':
            def truth(window):
                value=copy.deepcopy(window)
                for key in ('asr_cache_hit','aligner_cache_hit'):value.pop(key,None)
                value.get('asr',{}).get('durable_native_batch',{}).pop('cache_hit',None)
                return value
            row['complete_native_window_equal_except_cache_flags']=truth(left)==truth(right)
    return result


def analyze(baseline, candidate, mode='speed'):
    need(mode in {'speed','recovery'}, 'unknown comparison mode')
    compare_contracts(baseline['contract'], candidate['contract'], mode)
    feature_protocol=asr_feature_profile(baseline['contract'])
    feature_counts=None
    if feature_protocol is not None:
        feature_counts=[verify_track_preprocessing(t)for t in (baseline,candidate)]
    left, right = baseline['records'], candidate['records']
    need(set(right) <= set(left) and (mode == 'recovery' or set(left) == set(right)), 'fixed input source union differs')
    rows = [compare_record(left[sid], right[sid], mode) for sid in sorted(right)]
    counts = Counter(); max_delta = {'native_timestamps_ms': None, 'fixed_timestamps_ms': None}; comparable = 0
    recovery_equal = True
    for row in rows:
        counts['sources'] += 1; counts['source_identity_changed'] += not row['source_identity_equal']
        comparable += bool(row['source_identity_equal'] and row['decoded_audio_identity_comparable'])
        counts['decision_changed_sources'] += row['baseline_state'] != row['candidate_state']
        recovery_equal &= not row['changed_fields'] and row['source_identity_equal'] and row['baseline_window_count'] == row['candidate_window_count']
        for w in row['windows']:
            counts['planned_window_union'] += 1
            if not w['both_present']: counts['missing_windows'] += 1; recovery_equal = False; continue
            counts['input_mismatch_windows'] += not w['input_identity_equal']
            for key,value in w['ASR'].items():
                if key.endswith('_equal'): counts['ASR_' + key.removesuffix('_equal') + '_changed_windows'] += not value
            if feature_protocol is not None:
                counts['preprocessing_effective_input_mismatch_windows']+=any(value is not True
                    for key,value in w['ASR'].items()if key.startswith('preprocessing_')and key.endswith('_equal'))
            for key,value in w['alignment'].items():
                if key.endswith('_equal'): counts['alignment_' + key.removesuffix('_equal') + '_changed_windows'] += not value
            for key in max_delta:
                value = w['alignment'].get(key + '_max_absolute_delta_ms')
                if value is not None:
                    max_delta[key] = value if max_delta[key] is None else max(max_delta[key], value)
                    counts[key + '_comparable_windows'] += 1
            if mode == 'recovery': recovery_equal &= w['complete_native_window_equal_except_cache_flags']
    pa, pb = baseline['performance'], candidate['performance']
    cold_speed_valid = bool(mode == 'speed' and baseline['contract'].get('purpose') != 'cpu_fixture' and candidate['contract'].get('purpose') != 'cpu_fixture' and pa['all_sources_recomputed_cold'] and pb['all_sources_recomputed_cold']
                            and pa.get('native_whole_batch_readback_verified')is True and pb.get('native_whole_batch_readback_verified')is True
                            and pa.get('actual_compute_execution_verified') and pb.get('actual_compute_execution_verified')
                            and pa['totals']['asr_rows'] > 0 and pb['totals']['asr_rows'] > 0
                            and comparable == len(rows) and not counts['missing_windows'] and not counts['input_mismatch_windows']
                            and (feature_protocol is None or (feature_counts[0]==pa['totals']['asr_rows']
                                 and feature_counts[1]==pb['totals']['asr_rows']
                                 and not counts['preprocessing_effective_input_mismatch_windows'])))
    return {'schema': 'redimnet2-annotation-native-batch-comparison/v1', 'recorded_utc': utcnow(), 'mode': mode,
            'source_count': len(rows), 'baseline_total_sources': len(left), 'candidate_total_sources': len(right),
            'subset_of_baseline_explicit_in_recovery_mode': mode == 'recovery' and set(left) != set(right),
            'counts': dict(counts), 'decoded_audio_identity_comparable_sources': comparable,
            'maximum_same_target_endpoint_delta_ms': max_delta, 'baseline_performance': pa, 'candidate_performance': pb,
            'cold_end_to_end_speed_ratio': pa['parent_wall_seconds'] / pb['parent_wall_seconds'] if cold_speed_valid else None,
            'cold_speed_ratio_valid': cold_speed_valid,
            'strict_recovery_native_results_equal': bool(recovery_equal) if mode == 'recovery' else None,
            'strict_native_results_equal': bool(recovery_equal) if mode == 'recovery' else None,
            'baseline_decision_counts': baseline['decision_counts'], 'candidate_decision_counts': candidate['decision_counts'],
            'baseline_spec_sha256': baseline['spec_sha256'], 'candidate_spec_sha256': candidate['spec_sha256'],
            'baseline_contract_sha256': baseline['contract_sha256'], 'candidate_contract_sha256': candidate['contract_sha256'],
            'all_planned_sources_accounted': True, 'full_sidecar_commit_readback_verified': True,
            'native_whole_batch_readback_verified': bool(pa.get('native_whole_batch_readback_verified') and pb.get('native_whole_batch_readback_verified')),
            'ASR_preprocessing_protocol':feature_protocol,
            'native_B1_preprocessing_verified_windows':feature_counts,
            'native_B1_effective_inputs_equal':(not counts['preprocessing_effective_input_mismatch_windows'])if feature_protocol is not None else None,
            'database_created': False,
            'source_audio_rehashed_in_this_audit': False, 'human_text_or_boundary_accuracy_certified': False,
            'training_or_evaluation_eligibility_released': False, 'formal_admission': False, 'changed_rows_are_accuracy_errors': False,
            'rows': rows, 'report_contains_private_source_ids_and_model_observations': True}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate'):
        p.add_argument('--'+name+'-spec',required=True);p.add_argument('--'+name+'-spec-sha',required=True)
    p.add_argument('--mode',choices=('speed','recovery'),default='speed');p.add_argument('--output',required=True)
    args=p.parse_args(argv);out=Path(args.output).absolute()
    need(out.parent.is_dir() and not out.exists() and all(not p.is_symlink()for p in (out,*out.parents)),'new physical report output required')
    a=load_track(args.baseline_spec,args.baseline_spec_sha);b=load_track(args.candidate_spec,args.candidate_spec_sha)
    for track in (a,b):
        for key in ('queue_root','output_root','resources_root'):
            root=Path(track['spec'][key]);need(root!=out and root not in out.parents,'native comparison report inside producer refused')
    result=analyze(a,b,args.mode);atomic_new(out,result)
    return 2 if args.mode=='recovery' and not result['strict_native_results_equal']else 0


if __name__=='__main__':raise SystemExit(main())
