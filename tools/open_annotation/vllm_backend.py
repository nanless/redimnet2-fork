"""保留原生生成证据的Qwen/vLLM最小离线适配。

导入此模块不导入torch、qwen_asr或vllm。CPU测试必须注入假引擎并标记为
synthetic。真实构造仅由build_real_engine(execute=True)显式触发；该低层
函数不授予机器资源、GPU质量或全量任务准入，调用者仍须先做资源组验收。
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

if __package__:
    from .asr_stage_store import ASRStageStore, canonical, digest, physical, require
    from .normalization import normalize_with_audit, POLICY, SUPPORTED_POLICIES
else:
    from asr_stage_store import ASRStageStore, canonical, digest, physical, require
    from normalization import normalize_with_audit, POLICY, SUPPORTED_POLICIES

SPEC_SCHEMA = 'annotation-vllm-asr-spec/v1'
RESULT_SCHEMA = 'annotation-asr-native-result/v1'


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _token_ids(value, label):
    require(isinstance(value, (list, tuple)) and all(type(x) is int and x >= 0 for x in value),
            'nonnegative native token IDs required: ' + label)
    return list(value)


def _evidence(value):
    """失败时也保留原生字段；非JSON值显式标记，不偷偷改成0。"""
    if value is None or type(value) in (str,int,bool): return value
    if type(value) is float:
        return value if math.isfinite(value) else {'nonfinite_native_float': repr(value)}
    if isinstance(value,(list,tuple)):return [_evidence(x) for x in value]
    if isinstance(value,dict):return {str(k):_evidence(v) for k,v in value.items()}
    return {'unsupported_native_type':type(value).__module__+'.'+type(value).__qualname__,
            'representation':repr(value)}


def _snapshot_output(output):
    items=_field(output,'outputs')
    return _evidence({'request_id':_field(output,'request_id'),'finished':_field(output,'finished'),
        'prompt_token_ids':_field(output,'prompt_token_ids'),'outputs':[
            {k:_field(item,k) for k in ('index','text','token_ids','finish_reason','stop_reason')}
            for item in items] if isinstance(items,(list,tuple)) else items})


def _file_sha(path):
    h=hashlib.sha256()
    with physical(path).open('rb') as f:
        before=os.fstat(f.fileno())
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        after=os.fstat(f.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'pinned file changed during read')
    return h.hexdigest()


def verify_local_snapshot(model):
    """绑定实际会被本地加载的整个模型目录，不能只pin config漏掉权重。"""
    root=physical(model['path']);pins=model.get('files_sha256',{})
    require(root.is_dir() and isinstance(pins,dict) and pins and 'config.json' in pins,
            'explicit local model file pins required')
    entries=[p for p in root.rglob('*') if not {'.cache','.git'} & set(p.relative_to(root).parts)]
    require(all(not p.is_symlink() for p in entries),'model snapshot symlink refused')
    files={str(p.relative_to(root)) for p in entries if p.is_file()}
    runtime={p for p in files if Path(p).suffix in {'.json','.model','.txt','.tiktoken','.safetensors','.bin','.jinja','.spm','.py'}}
    for name in [p for p in runtime if p.endswith('.index.json')]:
        value=json.loads((root/name).read_bytes());weight_map=value.get('weight_map')
        if weight_map is not None:
            require(isinstance(weight_map,dict) and weight_map and all(isinstance(v,str) and v for v in weight_map.values()),
                    'invalid model weight shard index')
            runtime.update(weight_map.values())
    require(runtime <= set(pins) and runtime <= files and
            any(Path(p).suffix in {'.safetensors','.bin'} for p in runtime),
            'model snapshot pins do not cover every actual weight/processor file')
    for rel,expected in pins.items():
        p=physical(root/rel);require(not Path(rel).is_absolute() and root in p.parents and
            _file_sha(p)==expected,'model file pin differs')
    return root


def validate_spec(spec):
    require(isinstance(spec, dict) and spec.get('schema') == SPEC_SCHEMA, 'wrong vLLM ASR spec')
    require(spec.get('purpose') in {'cpu_fixture', 'pilot_validation'}, 'real adapter is a pilot, not production admission')
    require(type(spec.get('model_API_calls_allowed')) is int and spec['model_API_calls_allowed'] == 0 and spec.get('release_training_labels') is False,
            'offline API0 and no label release required')
    engine, decode = spec['engine'], spec['decode']
    require(set(engine) == {'backend','version','dtype','tensor_parallel_size','gpu_memory_utilization','max_model_len'},
            'unknown or missing engine setting; no silently ignored execution flag')
    require(engine.get('backend') == 'vllm' and engine.get('dtype') in {'float32', 'bfloat16'} and
            type(engine.get('tensor_parallel_size')) is int and engine['tensor_parallel_size'] == 1, 'explicit supported single-GPU engine profile required')
    require(type(engine.get('gpu_memory_utilization')) in (int,float) and
            math.isfinite(engine['gpu_memory_utilization']) and 0 < engine['gpu_memory_utilization'] < 1,
            'explicit total-card GPU memory fraction required')
    require(isinstance(engine.get('version'), str) and bool(engine['version']), 'pinned vLLM version required')
    require(set(decode) == {'temperature','max_tokens','seed','n','ignore_eos'} and
            type(decode['temperature']) in (int,float) and decode['temperature'] == 0 and
            type(decode['max_tokens']) is int and 1 <= decode['max_tokens'] <= 4096 and
            type(decode['seed']) is int and type(decode['n']) is int and decode['n'] == 1 and decode['ignore_eos'] is False,
            'fixed greedy decoding with explicit generation bound required')
    require(type(engine['max_model_len']) is int and engine['max_model_len'] > decode['max_tokens'],
            'explicit model context bound must exceed output bound')
    require(type(spec.get('max_batch_rows')) is int and 1 <= spec['max_batch_rows'] <= 128,
            'finite explicit batch width required')
    require(spec.get('normalization_profile', POLICY) in SUPPORTED_POLICIES, 'unknown text normalization')
    require(spec['preprocessing']['model_sample_rate'] == 16000 and
            spec['preprocessing']['pcm_policy'] == 'float32le_mono/v1' and
            isinstance(spec['preprocessing'].get('window_policy_sha256'), str) and
            len(spec['preprocessing']['window_policy_sha256']) == 64 and
            all(x in '0123456789abcdef' for x in spec['preprocessing']['window_policy_sha256']), 'fixed preprocessing identity required')
    require(spec.get('batch_policy') == 'fixed_cohort_until_durable; immutable_published_result_reuse/v1',
            'explicit batch-sensitive recovery policy required')
    eos = _token_ids(spec['eos_token_ids'], 'declared EOS')
    require(eos and len(set(eos)) == len(eos), 'explicit unique EOS IDs required')
    languages = spec['supported_languages']
    require(isinstance(languages, list) and languages and all(isinstance(x,str) and x for x in languages)
            and len(set(languages)) == len(languages), 'fixed supported language names required')
    require(spec.get('force_language') is None or spec['force_language'] in languages, 'unsupported forced language')
    require(isinstance(spec.get('context'), str), 'explicit ASR system context required')
    require(isinstance(spec.get('model'), dict) and isinstance(spec['model'].get('path'), str)
            and Path(spec['model']['path']).is_absolute(), 'absolute model path required')
    return copy.deepcopy(spec)


class VLLMBackend:
    """每个实例只拥有自己的离线引擎队列；不接受外部混入的native请求。"""
    def __init__(self, spec, engine, store, *, parse_answer, prompt_builder, synthetic_fixture):
        self.spec = validate_spec(spec)
        require(type(synthetic_fixture) is bool, 'explicit synthetic flag required')
        require(synthetic_fixture == (spec['purpose'] == 'cpu_fixture'), 'synthetic purpose differs')
        require(isinstance(store, ASRStageStore) and store.producer_sha256 == digest(self.spec) and
                store.synthetic_fixture == synthetic_fixture, 'stage store producer differs')
        require(callable(getattr(engine, 'enqueue', None)) and callable(getattr(engine, 'wait_for_completion', None)),
                'required public enqueue/wait_for_completion API unavailable; no positional-ID fallback')
        require(callable(parse_answer) and callable(prompt_builder), 'native Qwen parser and prompt builder required')
        self.engine, self.store = engine, store
        self.parse_answer, self.prompt_builder = parse_answer, prompt_builder
        self.synthetic_fixture, self.producer_sha256 = synthetic_fixture, digest(self.spec)
        self.runtime_facts = {'synthetic_fixture': synthetic_fixture, 'real_engine_constructed': False,
                              'engine_object_injected': True,
                              'public_ID_mapping_API': 'enqueue/wait_for_completion',
                              'GPU_or_model_validation_passed': False}

    @classmethod
    def from_fake_engine(cls, spec, engine, store, *, parse_answer, prompt_builder):
        require(spec.get('purpose') == 'cpu_fixture', 'injected fake engine requires cpu_fixture purpose')
        return cls(spec, engine, store, parse_answer=parse_answer, prompt_builder=prompt_builder,
                   synthetic_fixture=True)

    def request_identity(self, row):
        for name in ('source_id','window_id','input_pcm_sha256'):
            require(isinstance(row.get(name), str) and row[name], 'missing request identity: ' + name)
        require(len(row['input_pcm_sha256']) == 64 and all(x in '0123456789abcdef' for x in row['input_pcm_sha256']),
                'input PCM SHA required')
        require(type(row.get('model_input_frames')) is int and row['model_input_frames'] > 0 and
                row.get('model_sample_rate') == 16000, 'invalid model input clock')
        return {'schema':'annotation-asr-request/v1', 'source_id':row['source_id'], 'window_id':row['window_id'],
                'input_pcm_sha256':row['input_pcm_sha256'], 'model_input_frames':row['model_input_frames'],
                'model_sample_rate':16000, 'producer_sha256':self.producer_sha256}

    def _wave(self, row):
        # 元数据缓存复用不需PCM；调用者给了数组时，即使命中缓存也校验。
        import numpy as np
        wave = row.get('audio')
        require(isinstance(wave,np.ndarray) and wave.ndim == 1 and wave.dtype == np.dtype('float32') and
                len(wave) == row['model_input_frames'] and bool(np.isfinite(wave).all()), 'finite mono float32 PCM required')
        require(hashlib.sha256(wave.astype('<f4',copy=False).tobytes()).hexdigest() == row['input_pcm_sha256'],
                'actual PCM differs from request identity')
        return wave

    def _native(self, output):
        completions = _field(output, 'outputs')
        require(isinstance(completions,(list,tuple)) and len(completions) == 1, 'exactly one native completion required')
        item = completions[0]
        require(type(_field(item,'index',0)) is int and _field(item,'index',0) == 0, 'unexpected completion index')
        text = _field(item, 'text')
        require(isinstance(text,str), 'native decoded answer missing')
        tokens = _token_ids(_field(item,'token_ids'), 'completion')
        prompt_tokens = _token_ids(_field(output,'prompt_token_ids'), 'prompt')
        finish, stop = _field(item,'finish_reason'), _field(item,'stop_reason')
        require(finish in {None,'stop','length','abort','error'}, 'unknown native finish reason')
        require(stop is None or type(stop) in (str,int), 'unsupported native stop reason type')
        require(type(_field(output,'finished')) is bool, 'native finished flag missing')
        eos = set(self.spec['eos_token_ids']); first = next((i for i,x in enumerate(tokens) if x in eos),None)
        native_id=_field(output,'request_id');require(isinstance(native_id,str) and native_id,'native request ID must be an explicit string')
        return {'request_id':native_id, 'generated_token_ids':tokens,
                'prompt_token_ids':prompt_tokens, 'native_decoded_raw_answer':text,
                'finish_reason':finish, 'stop_reason':stop, 'finished':_field(output,'finished'),
                'eos_observed':first is not None, 'first_eos_index':first,
                'declared_eos_token_ids':list(self.spec['eos_token_ids']),
                'max_new_tokens':self.spec['decode']['max_tokens']}

    def _result(self, identity, native, cohort):
        finish, tokens = native['finish_reason'], native['generated_token_ids']
        require(len(tokens) <= self.spec['decode']['max_tokens'], 'native generated tokens exceed configured bound')
        declared_stop = type(native['stop_reason']) is int and native['stop_reason'] in self.spec['eos_token_ids']
        eos_final = native['eos_observed'] and native['first_eos_index'] == len(tokens)-1
        eos_consistent = not native['eos_observed'] or (eos_final and
            sum(x in self.spec['eos_token_ids'] for x in tokens) == 1)
        termination = bool(native['finished'] and finish == 'stop' and eos_consistent and (eos_final or declared_stop))
        # finished+stop是生成器原生的结束理由；没有EOS/stop实据时保留unknown。
        status = 'completed' if termination else 'unknown'
        reasons = []
        if not eos_consistent:reasons.append('native_tokens_after_EOS_or_repeated_EOS')
        if finish == 'length' or (len(tokens) == self.spec['decode']['max_tokens'] and not termination):
            status = 'truncated';reasons.append('native_generation_limit')
        elif finish in {'abort','error'}:
            status = 'failed';reasons.append('native_generation_' + finish)
        elif not termination:
            reasons.append('native_termination_not_verified')
        parser_error = None
        try:
            language, text = self.parse_answer(native['native_decoded_raw_answer'], self.spec['force_language'])
        except Exception:
            language, text = None, ''
            status = 'failed'; reasons.append('native_parser_failed'); parser_error = traceback.format_exc()
        require(language is None or isinstance(language,str), 'parser language is not native text')
        require(isinstance(text,str), 'parser transcription must be text')
        if language not in self.spec['supported_languages']:
            reasons.append('unsupported_or_missing_language')
            if status == 'completed': status = 'unknown'
        audit = normalize_with_audit(text,self.spec.get('normalization_profile',POLICY))
        if not audit['normalized_text']:
            reasons.append('empty_transcription_not_proven_noise')
            if status == 'completed':status='unknown'
        return {'schema':RESULT_SCHEMA, 'request_id':digest(identity), **{k:identity[k] for k in
                    ('source_id','window_id','input_pcm_sha256','model_input_frames','model_sample_rate')},
                'request_identity':identity, 'producer_sha256':self.producer_sha256, 'cohort_sha256':cohort,
                'status':status,'raw_text':text,'normalized_text':audit['normalized_text'], 'language':language,
                'normalization_audit':audit,'native':native,'termination_verified':termination,'review_reasons':reasons,
                'native_parser_error':parser_error,
                'synthetic_fixture':self.synthetic_fixture,'model_API_calls':0,
                'usable_for_training':False,'usable_for_evaluation':False}

    def run_batch(self, rows, *, after_durable_commit=None):
        started=time.monotonic();require(isinstance(rows,list) and rows and len(rows)<=self.spec['max_batch_rows'],
                                          'finite nonempty batch required')
        identities=[self.request_identity(r) for r in rows];keys=[digest(v) for v in identities]
        require(len(keys)==len(set(keys)) and len({(r['source_id'],r['window_id']) for r in rows})==len(rows),
                'duplicate source/window request')
        verified_pcm=0
        for row in rows:
            if row.get('audio') is not None:
                self._wave(row);verified_pcm+=1
        by_key=dict(zip(keys,rows));new_rows=0;new_batches=0;elapsed=0.0
        with self.store.locked():
            plans=self.store.plans(identities)
            for cohort,intent in plans:
                cohort_keys=intent['request_keys'];inputs=[];native_outputs=[];committed=False
                try:
                    for key in cohort_keys:
                        wave=self._wave(by_key[key]);prompt=self.prompt_builder(self.spec['context'],self.spec['force_language'])
                        require(isinstance(prompt,str) and prompt, 'native Qwen prompt missing')
                        inputs.append({'prompt':prompt,'multi_modal_data':{'audio':[wave]}})
                    model_started=time.monotonic()
                    returned=self.engine.enqueue(inputs,sampling_params=dict(self.spec['decode']),use_tqdm=False)
                    require(isinstance(returned,(list,tuple)) and len(returned)==len(cohort_keys) and
                            all(isinstance(x,str) and x for x in returned) and len(set(returned))==len(returned),
                            'native enqueue request IDs missing or duplicated')
                    mapping=dict(zip(returned,cohort_keys))
                    outputs=self.engine.wait_for_completion(use_tqdm=False)
                    elapsed+=time.monotonic()-model_started;new_rows+=len(cohort_keys);new_batches+=1
                    require(isinstance(outputs,(list,tuple)),'native outputs must be a complete list')
                    native_outputs=[_snapshot_output(x) for x in outputs]
                    parsed=[self._native(x) for x in outputs]
                    native_ids=[x['request_id'] for x in parsed]
                    require(len(native_ids)==len(set(native_ids)) and set(native_ids)==set(mapping),
                            'native request coverage missing, duplicated or foreign')
                    by_native={x['request_id']:x for x in parsed}
                    native_for_key={mapping[x]:by_native[x] for x in mapping}
                    results=[self._result(identity,native_for_key[key],cohort)
                             for key,identity in zip(cohort_keys,intent['request_identities'])]
                    self.store.commit(cohort,results)
                    committed=True
                    if after_durable_commit is not None:after_durable_commit(cohort)
                except BaseException as exc:
                    self.store.preserve_failure(cohort,native_outputs,exc,stage_committed=committed)
                    raise
            results=[self.store.lookup(v) for v in identities]
            require(all(v is not None for v in results),'committed batch results missing')
        return {'schema':'annotation-asr-batch-run/v1','producer_sha256':self.producer_sha256,
                'results':results,'metrics':{'requested_rows':len(rows),'stage_cache_hits':len(rows)-new_rows,
                'new_model_rows':new_rows,'new_batches':new_batches,'model_elapsed_seconds':elapsed,
                'stage_wall_seconds':time.monotonic()-started,'provided_PCM_verified_rows':verified_pcm},'synthetic_fixture':self.synthetic_fixture,
                'GPU_or_model_validation_passed':False,'model_API_calls':0,'formal_admission':False}

    transcribe_requests = run_batch


class _RealQwenTransport:
    """把SamplingParams构造留在真实执行入口，CPU导入不接触vLLM。"""
    def __init__(self,llm,sampling_type):self.llm,self.sampling_type=llm,sampling_type
    def enqueue(self,inputs,sampling_params,use_tqdm=False):
        return self.llm.enqueue(inputs,sampling_params=self.sampling_type(**sampling_params),use_tqdm=use_tqdm)
    def wait_for_completion(self,use_tqdm=False):return self.llm.wait_for_completion(use_tqdm=use_tqdm)


def build_real_engine(spec, store_root, *, execute=False):
    """明确执行的低层工厂。当前只做CPU测试，实际调用还需外部资源准入。"""
    spec=validate_spec(spec)
    require(execute is True and spec['purpose']=='pilot_validation', 'explicit execute and pilot purpose required')
    require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1',
            'offline model environment must be explicitly set by resource launcher')
    root=verify_local_snapshot(spec['model'])
    require(importlib.metadata.version('vllm')==spec['engine']['version'],'vLLM dependency version differs')
    require(spec.get('qwen_asr_version') and importlib.metadata.version('qwen-asr')==spec['qwen_asr_version'],
            'Qwen dependency version differs')
    code=spec.get('source_files',{});require(code,'explicit actual Qwen/vLLM Python source pins required')
    for name,expected in code.items():
        p=physical(name);require(p.suffix=='.py' and _file_sha(p)==expected,'runtime source pin differs')
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output
    from vllm import SamplingParams, LLM
    for cls in (Qwen3ASRModel, SamplingParams, LLM, parse_asr_output):
        source = physical(inspect.getfile(cls))
        require(str(source) in code and _file_sha(source)==code[str(source)],
                'actually imported critical model/engine class is not the pinned source')
    require(callable(getattr(LLM,'enqueue',None)) and callable(getattr(LLM,'wait_for_completion',None)),
            'loaded pinned vLLM class lacks required public ID-mapping APIs; no constructor executed')
    engine=spec['engine']
    wrapper=Qwen3ASRModel.LLM(model=str(root),dtype=engine['dtype'],tensor_parallel_size=1,
        gpu_memory_utilization=engine['gpu_memory_utilization'],max_inference_batch_size=spec['max_batch_rows'],
        max_new_tokens=spec['decode']['max_tokens'],max_model_len=engine['max_model_len'],
        max_num_seqs=spec['max_batch_rows'],seed=spec['decode']['seed'],trust_remote_code=False)
    require(set(wrapper.get_supported_languages())==set(spec['supported_languages']),
            'actual Qwen supported language declaration differs')
    tokenizer=wrapper.processor.tokenizer
    eos=getattr(tokenizer,'eos_token_id',None);eos=[eos] if type(eos) is int else list(eos or [])
    require(eos==spec['eos_token_ids'],'actual tokenizer EOS IDs differ from declared EOS')
    store=ASRStageStore(store_root,digest(spec),synthetic_fixture=False)
    adapter=VLLMBackend(spec,_RealQwenTransport(wrapper.model,SamplingParams),store,
        parse_answer=lambda text,language:parse_asr_output(text,user_language=language),
        prompt_builder=lambda context,language:wrapper._build_text_prompt(context,language),synthetic_fixture=False)
    adapter.runtime_facts.update(vllm_version=engine['version'],qwen_asr_version=spec['qwen_asr_version'],
                                 source_files=code,actual_tokenizer_eos_ids=eos,
                                 actual_model_inference_completed=False,real_engine_constructed=True,
                                 engine_object_injected=False)
    return adapter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--store-root',type=Path)
    a=p.parse_args();raw=physical(a.spec).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==a.spec_sha,'spec SHA differs');spec=validate_spec(json.loads(raw))
    if a.execute:
        require(a.store_root is not None,'explicit new store root required');build_real_engine(spec,a.store_root,execute=True)
    print(json.dumps({'spec_sha256':digest(spec),'executed_constructor':a.execute,
                      'models_inferred':False,'formal_admission':False,'GPU_quality_verified':False}))


if __name__=='__main__':main()
