"""跨引擎对照的只读协议。声明相同的输入条件，保留不同的原生输出。"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import digest, need, model_provenance
from readonly_progress import read_small, strict_json

SCHEMA = 'annotation-engine-comparison/v1'
PRODUCER_SCHEMA = 'annotation-engine-producer/v1'
CONTROL_KEYS = {'model_weights', 'preprocessing', 'window_policy', 'prompt_language_policy',
                'generation', 'quality_policy', 'alignment_identity'}
VARIABLE_KEYS = {'backend', 'code_sha256', 'precision', 'batching'}
WINDOW_FIELDS = ('window_id', 'input_pcm_sha256', 'model_input_frames', 'model_sample_rate',
                 'context_start_sample', 'context_end_sample', 'owned_start_sample',
                 'owned_end_sample', 'resampling_policy', 'time_mapping')
SOURCE_FIELDS = ('file_sha256', 'pcm_sha256', 'pcm_hash_policy', 'sample_rate', 'channels', 'frames')
MAX_SOURCES = 20000
KINDS = {'new', 'cache', 'reused', 'not_run'}
ENGINE_RUNTIME_KEYS = {'asr_compute','asr_backend','batching','max_batch','length_ratio_limit',
                       'execution_policy','source_batch_rows'}


def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def physical(path):
    p = Path(path).absolute()
    need('..' not in p.parts and all(not v.is_symlink() for v in (p, *p.parents)),
         'physical input path without symlinks required')
    return p


def read_ref(ref, *, lines=False, maximum=64 * 1024**2):
    need(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and sha(ref['sha256']),
         'explicit file path and lowercase SHA256 required')
    need(isinstance(ref['path'], str) and Path(ref['path']).is_absolute(), 'absolute reference required')
    p = physical(ref['path'])
    if lines:
        import os
        fd = os.open(str(p), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            import stat
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'regular bounded JSONL required')
            with os.fdopen(fd, 'rb', closefd=False) as f: raw = f.read(maximum + 1)
            after = os.fstat(fd)
            need(len(raw) == before.st_size and (before.st_ino,before.st_size,before.st_mtime_ns) ==
                 (after.st_ino,after.st_size,after.st_mtime_ns), 'JSONL changed while read')
        finally: os.close(fd)
        need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'JSONL SHA differs')
        rows = raw.splitlines()
        need(rows and all(x.strip() for x in rows), 'blank or empty result list')
        value = [strict_json(x) for x in rows]
    else:
        value, raw = read_small(p, maximum)
        need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'JSON SHA differs')
    return value, len(raw)


def finite(value, name, *, positive=False):
    need(type(value) in (int,float) and math.isfinite(value) and value >= 0 and
         (not positive or value > 0), 'invalid measured numeric value: ' + name)
    return value


def validate_controls(controls):
    need(isinstance(controls, dict) and set(controls) == CONTROL_KEYS, 'fixed control fields differ')
    for name, packet in controls.items():
        need(isinstance(packet, dict) and set(packet) == {'value','sha256'} and
             isinstance(packet['value'],dict) and packet['value'] and sha(packet['sha256']) and
             digest(packet['value']) == packet['sha256'], 'fixed control content/SHA differs: ' + name)
    generation = controls['generation']['value']
    need(type(generation.get('max_new_tokens')) is int and generation['max_new_tokens'] > 0 and
         generation.get('do_sample') is False, 'positive fixed token limit and non-sampling policy required')
    return controls


def controls_from_contract(contract):
    """从真实已固定合同提取控制条件，未知runtime字段也保留为固定条件。"""
    runtime = contract['runtime']
    values = {'model_weights': model_provenance(contract),
        'preprocessing': {'decoder':contract['decoder'],
                          'runtime_except_engine':{k:v for k,v in runtime.items() if k not in ENGINE_RUNTIME_KEYS}},
        'window_policy': {k:runtime[k] for k in ('window','ownership','boundary_mapping')},
        'prompt_language_policy': {'language':runtime.get('language'), 'prompt':runtime.get('prompt',''),
                                  'policy':runtime.get('prompt_language_policy','automatic-language-no-text-context'),
                                  'supported_languages':runtime.get('supported_languages')},
        'generation': {'max_new_tokens':runtime['max_new_tokens'], 'do_sample':runtime.get('do_sample',False),
                       'temperature':runtime.get('temperature'), 'top_p':runtime.get('top_p'), 'seed':runtime.get('seed')},
        'quality_policy': contract['quality_policy'], 'alignment_identity': contract['alignment']}
    return {k:{'value':v,'sha256':digest(v)} for k,v in values.items()}


def code_identity(contract):
    return digest({'source_files':contract.get('source_files',{}),
                   'engine_source_files':contract.get('engine_source_files',{}),
                   'engine_versions':contract.get('engine_versions',{})})


def validate_native_spec(producer, contract):
    """读取实际的适配配置；只有一个散列字符串不证明模型与解码声明相同。"""
    from vllm_backend import validate_spec
    need('native_asr_spec' in producer, 'vLLM producer requires the actual pinned adapter spec')
    value,_=read_ref(producer['native_asr_spec'])
    value=validate_spec(value)
    need(digest(value)==producer.get('native_asr_producer_sha256') and value['purpose']==producer['purpose'],
         'native adapter spec/producer digest or purpose differs')
    fixed=producer['fixed_controls'];generation=fixed['generation']['value'];prompt=fixed['prompt_language_policy']['value']
    need(value['decode']['max_tokens']==generation['max_new_tokens'] and value['decode']['temperature']==0 and
         generation['do_sample'] is False and type(generation.get('seed')) is int and value['decode']['seed']==generation['seed'],
         'native adapter changed fixed generation controls')
    need(value['force_language']==prompt['language'] and value['context']==prompt['prompt'] and
         value['supported_languages']==prompt['supported_languages'], 'native adapter changed prompt/language policy')
    need(value['engine']['dtype']==producer['variables']['precision'] and
         value['max_batch_rows']==producer['variables']['batching']['max_requests'], 'native dtype/batch profile differs')
    expected=contract['models']['asr']
    need(value['model']['path']==expected['path'] and value['model']['files_sha256']==
         {k:v['sha256'] for k,v in expected['files'].items()}, 'native adapter changed ASR weight identity')
    need(value['preprocessing']['window_policy_sha256']==fixed['window_policy']['sha256'], 'native adapter changed window policy')
    from normalization import POLICY
    need(value.get('normalization_profile',POLICY)==contract['runtime'].get('normalization_profile',POLICY),
         'native adapter changed normalization policy')
    need(value.get('source_files')==contract.get('source_files') and
         value['engine']['version']==contract.get('engine_versions',{}).get('vllm') and
         value.get('qwen_asr_version')==contract.get('engine_versions',{}).get('qwen-asr'),
         'native adapter code/dependency identity differs')
    return value


def validate_producer(producer, fixed):
    need(producer.get('schema') == PRODUCER_SCHEMA and isinstance(producer.get('producer_id'),str)
         and producer['producer_id'] and sha(producer.get('bundle_sha256')), 'invalid producer/bundle identity')
    need(producer.get('purpose') in {'pilot_validation','cpu_fixture'} and
         producer.get('model_API_calls') == 0 and type(producer['model_API_calls']) is int and
         producer.get('release_training_labels') is False, 'producer purpose/API/eligibility differs')
    need(validate_controls(producer.get('fixed_controls')) == fixed, 'producer changed fixed experiment controls')
    variables = producer.get('variables')
    need(isinstance(variables,dict) and set(variables) == VARIABLE_KEYS, 'producer variable fields differ')
    need(variables['backend'] in {'transformers','vllm','cpu_fixture'} and sha(variables['code_sha256']) and
         variables['precision'] in {'float32','bfloat16','float16'}, 'invalid engine variable declaration')
    batch = variables['batching']
    need(isinstance(batch,dict) and batch.get('policy') in {'single','fixed_cohort','dynamic'} and
         type(batch.get('max_requests')) is int and batch['max_requests'] > 0,
         'batch scheduling semantics must be explicit')
    if batch['policy'] == 'single': need(batch['max_requests'] == 1, 'single policy has multiple requests')
    if 'native_asr_producer_sha256' in producer:
        need(sha(producer['native_asr_producer_sha256']), 'native producer SHA missing')
    return producer


def validate_protocol(spec):
    need(spec.get('schema') == SCHEMA and spec.get('purpose') in {'pilot_validation','cpu_fixture'}, 'wrong engine comparison schema/purpose')
    need(spec.get('mode') in {'end_to_end','fixed_text_alignment'}, 'unknown comparison mode')
    validate_controls(spec.get('fixed_controls'))
    changes = spec.get('allowed_changes')
    need(isinstance(changes,list) and len(set(changes)) == len(changes) and set(changes) <= VARIABLE_KEYS,
         'only explicit engine/code/precision/batching changes are allowed')
    for key in ('inputs','baseline','candidate'): need(key in spec, 'missing protocol field: ' + key)
    return spec


def validate_input(row):
    need(isinstance(row,dict) and isinstance(row.get('source'),dict) and
         isinstance(row.get('source_identity'),dict) and isinstance(row.get('windows'),list), 'invalid declared source input')
    source = row['source']; need(isinstance(source.get('source_id'),str) and source['source_id'], 'missing source ID')
    windows = row['windows']; seen = set()
    for w in windows:
        need(isinstance(w,dict) and all(k in w for k in WINDOW_FIELDS), 'declared window input fields missing')
        need(isinstance(w['window_id'],str) and w['window_id'] and w['window_id'] not in seen, 'duplicate/empty window ID')
        need(sha(w['input_pcm_sha256']) and type(w['model_input_frames']) is int and w['model_input_frames'] > 0 and
             w['model_sample_rate'] == 16000, 'invalid model PCM identity/clock')
        for k in ('context_start_sample','context_end_sample','owned_start_sample','owned_end_sample'):
            need(type(w[k]) is int and w[k] >= 0, 'invalid declared source clock')
        need(w['context_start_sample'] <= w['owned_start_sample'] < w['owned_end_sample'] <= w['context_end_sample'],
             'owned source interval outside context')
        if 'fixed_text' in w: need(isinstance(w['fixed_text'],str) and w['fixed_text'], 'empty fixed text')
        seen.add(w['window_id'])
    return row


def native_result_identity(native, planned, source_id, producer, native_spec):
    """独立保存vLLM原生结果，不把其终止原因当成旧transformers的EOS。"""
    need(native.get('schema') == 'annotation-asr-native-result/v1' and native.get('source_id') == source_id and
         all(native.get(k) == planned[k] for k in ('window_id','input_pcm_sha256','model_input_frames','model_sample_rate')),
         'native ASR request/input identity differs')
    identity=native.get('request_identity')
    need(isinstance(identity,dict) and identity.get('schema')=='annotation-asr-request/v1' and
         native.get('request_id')==digest(identity) and all(identity.get(k)==native.get(k) for k in
            ('source_id','window_id','input_pcm_sha256','model_input_frames','model_sample_rate','producer_sha256')),
         'native stable request identity is not bound to saved result')
    need(native.get('producer_sha256') == producer.get('native_asr_producer_sha256') and
         sha(producer.get('native_asr_producer_sha256')), 'native ASR producer differs')
    need(native.get('model_API_calls') == 0 and native.get('usable_for_training') is False and
         type(native.get('model_API_calls')) is int and native.get('usable_for_evaluation') is False,
         'native API/label release refused')
    need(type(native.get('synthetic_fixture')) is bool and
         (producer['purpose']!='cpu_fixture' or native['synthetic_fixture']), 'native synthetic marker differs')
    need(native.get('status') in {'completed','truncated','unknown','failed'}, 'invalid native ASR status')
    need(sha(native.get('cohort_sha256')), 'native cohort SHA missing or malformed')
    data = native.get('native')
    need(isinstance(data,dict) and isinstance(data.get('request_id'),str) and bool(data['request_id'].strip()),
         'native internal request ID missing or malformed')
    need(isinstance(data,dict) and type(data.get('eos_observed')) is bool and
         isinstance(data.get('generated_token_ids'),list) and all(type(x) is int and x >= 0 for x in data['generated_token_ids']),
         'native generation evidence missing')
    limit = producer['fixed_controls']['generation']['value']['max_new_tokens']
    need(data.get('max_new_tokens') == limit, 'native generator changed token cap')
    tokens=data['generated_token_ids'];eos=data.get('declared_eos_token_ids')
    need(isinstance(eos,list) and eos and all(type(x) is int and x>=0 for x in eos) and len(set(eos))==len(eos),
         'native declared EOS IDs missing')
    need(eos==native_spec['eos_token_ids'], 'native EOS declaration differs from actual adapter spec')
    from normalization import normalize_with_audit, POLICY
    need(native.get('normalization_audit')==normalize_with_audit(native.get('raw_text',''),native_spec.get('normalization_profile',POLICY)) and
         native.get('normalized_text')==native['normalization_audit']['normalized_text'], 'native normalization differs from adapter spec')
    first=next((i for i,x in enumerate(tokens) if x in eos),None)
    need(data['eos_observed']==(first is not None) and data.get('first_eos_index')==first and len(tokens)<=limit,
         'native EOS/token evidence contradicts itself')
    need(type(data.get('finished')) is bool and data.get('finish_reason') in {None,'stop','length','abort','error'},
         'native finish evidence missing')
    if native['status']=='completed':
        stop=data.get('stop_reason')
        need(data['finished'] and data['finish_reason']=='stop' and
             ((first is not None and first==len(tokens)-1) or (first is None and type(stop) is int and stop in eos)),
             'completed ASR lacks native termination evidence')
    return native
