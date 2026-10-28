"""Offline, one-GPU ASR+FA replica; native tokens/times are never repaired here."""
from __future__ import annotations
import copy
import contextlib
import inspect
import math
import time
import traceback
import unicodedata
from pathlib import Path
import sys
from contracts import native_json, need, sha_file, verify_model_stats, verify_models, validate_execution_policy, asr_compute_profile, asr_batch_profile, asr_feature_profile, digest, model_provenance
from normalization import normalize_with_audit, POLICY
from alignment_observer import AlignmentObserver, successful_alignment, positive_integer_attribute


def normalize(text):
    return normalize_with_audit(text)["normalized_text"]


class NativeBatchRefused(RuntimeError):
    """批次、填充或请求映射不明时停止独立试验，不伪装成单条unknown。"""


def native_token_rows(input_rows, masks, generated, decoded, eos_ids, pad_id, max_tokens):
    """纯CPU核多行生成的真实token几何；原矩形和有效EOS前序列分别保留。"""
    def require(value, message):
        if not value: raise NativeBatchRefused(message)
    count=len(input_rows)
    require(count>0 and len(masks)==len(generated)==len(decoded)==count, 'native generation row coverage differs')
    width=len(input_rows[0]);require(width>0 and all(len(x)==width for x in input_rows), 'native prompt must be rectangular')
    require(type(max_tokens)is int and max_tokens>0 and isinstance(eos_ids,list)and eos_ids and
            all(type(x)is int and x>=0 for x in eos_ids), 'native EOS/token limit must be explicit')
    returned_width=len(generated[0]);require(returned_width<=max_tokens and all(len(x)==returned_width for x in generated), 'native generated token shape/limit differs')
    result=[]
    for i,(prompt,mask,ids,text)in enumerate(zip(input_rows,masks,generated,decoded)):
        require(all(type(x)is int and x>=0 for x in prompt+ids), 'native token IDs must be nonnegative integers')
        require(len(mask)==width and all(type(x)is int and x in (0,1)for x in mask)and sum(mask)>0 and sorted(mask)==mask,
                'native decoder prompt requires observed left padding without mask holes')
        require(isinstance(text,str), 'native decoded answer must be text')
        end=next((j for j,x in enumerate(ids)if x in eos_ids),None)
        effective=list(ids[:end+1]if end is not None else ids)
        post=list(ids[end+1:]if end is not None else [])
        require(not post or type(pad_id)is int and all(x==pad_id or x in eos_ids for x in post), 'unexplained tokens after row EOS')
        result.append(dict(generated_token_ids=list(ids),native_decoded_raw_answer=text,eos_observed=end is not None,
            first_eos_index=end,max_new_tokens=max_tokens,input_padded_tokens=width,input_nonpad_tokens=sum(mask),
            input_token_ids=list(prompt),input_attention_mask=list(mask),actual_generate_rows=count,actual_generate_row_index=i,
            effective_generated_token_ids=effective,effective_generated_token_count=len(effective),
            raw_returned_token_count=len(ids),post_eos_token_ids=post,eos_token_ids=list(eos_ids),padding_token_id=pad_id))
    return result


def length_batches(rows, max_batch, ratio):
    ordered = list(rows) if max_batch == 1 else sorted(rows, key=lambda r: (len(r["audio"]), r["row_id"]))
    batch = []
    for row in ordered:
        need(len(row["audio"]) > 0, "empty model input")
        if batch and (len(batch) >= max_batch or len(row["audio"]) / len(batch[0]["audio"]) > ratio):
            yield batch; batch = []
        batch.append(row)
    if batch: yield batch


def model_configurations(model):
    """包含根、thinker、audio、text子配置，避免仅修改顶层而伪称兑现。"""
    result, seen = [], set()
    def visit(name, config):
        if config is None or id(config) in seen: return
        seen.add(id(config)); result.append((name, config))
        for key in getattr(config, 'sub_configs', {}):
            visit(name + '.' + key, getattr(config, key, None))
    for name, module in model.named_modules():
        visit(name + '.config', getattr(module, 'config', None))
    return result


def apply_asr_compute(model, expected):
    """仅对显式新试验设置全部真实子配置；历史缺省绝不强改。"""
    configurations = model_configurations(model)
    need(configurations, 'ASR nested model configurations are not observable')
    for _, config in configurations:
        config._attn_implementation = expected['attention']


def loaded_compute_facts(model, expected, strict_attention=False):
    """核实际加载参数dtype和模型配置；不将配置读回冒充CUDA核追踪。"""
    dtype_counts = {}
    for parameter in model.parameters():
        if parameter.is_floating_point():
            dtype = str(parameter.dtype).removeprefix('torch.')
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + parameter.numel()
    need(set(dtype_counts) == {expected['dtype']}, 'loaded floating model parameters differ from requested dtype')
    observed = []
    for name, config in model_configurations(model):
        implementation = getattr(config, '_attn_implementation', None)
        if implementation is None: continue
        if strict_attention:
            need(implementation == expected['attention'], 'loaded attention config differs; silent fallback refused: ' + name)
        observed.append({'module': name, 'attention_config': implementation})
    need(observed, 'actual attention configuration is not observable')
    return {'requested': dict(expected), 'floating_parameter_elements_by_dtype': dtype_counts,
            'attention_configurations': observed, 'strict_attention_requested': strict_attention,
            'kernel_trace_performed': False}


class ComputeProfileRefused(RuntimeError):
    """候选计算方式未实际兑现：停止该试验，不伪装成单条原音异常。"""


class ASRAttentionTrace:
    """只跟踪本进程首个ASR调用的真实attention forward和F.sdpa调用。

    这是Python函数入口观察，不是CUDA内核性能分析；其余窗口不反复
    注册钩子，避免计时工具本身成为批量任务开销。退出总会恢复函数。
    """
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.handles, self.active, self.forwards, self.sdpa = [], [], {}, {}
        self.functions = {}; self.original = None; self.configurations = {}; self.configurations_stable = True
        self.qkv_dtypes = {}; self.qkv_dtypes_stable = True

    def __enter__(self):
        try:
            for name, module in self.model.named_modules():
                if not type(module).__name__.lower().endswith('attention'): continue
                forward = inspect.unwrap(type(module).forward)
                self.functions[name] = {'class': type(module).__module__ + '.' + type(module).__qualname__,
                    'forward': forward.__module__ + '.' + forward.__qualname__,
                    'source': inspect.getsourcefile(forward)}
                def before(mod, args, key=name):
                    self.active.append(key); self.forwards[key] = self.forwards.get(key, 0) + 1
                    implementation = getattr(getattr(mod, 'config', None), '_attn_implementation', None)
                    if key in self.configurations and self.configurations[key] != implementation:
                        self.configurations_stable = False
                    self.configurations[key] = implementation
                def after(mod, args, result, key=name):
                    if self.active and self.active[-1] == key: self.active.pop()
                self.handles.append(module.register_forward_pre_hook(before))
                self.handles.append(module.register_forward_hook(after))
            functional = self.torch.nn.functional
            self.original = functional.scaled_dot_product_attention
            def observed(*args, **kwargs):
                key = self.active[-1] if self.active else 'unattributed'
                self.sdpa[key] = self.sdpa.get(key, 0) + 1
                tensors = list(args[:3]) if len(args) >= 3 else [kwargs.get(n) for n in ('query', 'key', 'value')]
                dtypes = [str(value.dtype).removeprefix('torch.') if hasattr(value, 'dtype') else None for value in tensors]
                if key in self.qkv_dtypes and self.qkv_dtypes[key] != dtypes: self.qkv_dtypes_stable = False
                self.qkv_dtypes[key] = dtypes
                return self.original(*args, **kwargs)
            functional.scaled_dot_product_attention = observed
            return self
        except BaseException:
            self.__exit__(None, None, None); raise

    def __exit__(self, *exc):
        for handle in reversed(self.handles): handle.remove()
        self.handles.clear()
        if self.original is not None:
            self.torch.nn.functional.scaled_dot_product_attention = self.original
        self.active.clear()

    def observation(self, expected=None):
        result = {'scope': 'first_successful_ASR_transcribe_call_in_this_worker_process',
            'attention_forwards': dict(self.forwards), 'functional_sdpa_calls': dict(self.sdpa),
            'attention_module_configurations': dict(self.configurations),
            'attention_configurations_stable': self.configurations_stable,
            'unexecuted_attention_modules': sorted(set(self.functions) - set(self.forwards)),
            'functional_sdpa_qkv_dtypes': copy.deepcopy(self.qkv_dtypes),
            'functional_sdpa_qkv_dtypes_stable': self.qkv_dtypes_stable,
            'forward_functions': {k: self.functions[k] for k in self.forwards},
            'CUDA_kernel_trace_performed': False}
        if expected is not None:
            okay = bool(self.forwards) and not result['unexecuted_attention_modules'] and self.configurations_stable and all(
                implementation == expected['attention'] for implementation in self.configurations.values())
            if expected['attention'] == 'sdpa':
                okay = okay and all(self.sdpa.get(name, 0) == count for name, count in self.forwards.items())
                okay = okay and self.qkv_dtypes_stable and all(
                    self.qkv_dtypes.get(name) == [expected['dtype']] * 3 for name in self.forwards)
            else:
                okay = okay and sum(self.sdpa.values()) == 0
            if not okay: raise ComputeProfileRefused('ASR actual attention dispatch differs from explicit profile: ' + str(result))
        return result


class QwenBackend:
    name = "qwen3-asr-plus-ultra-no-head"
    def __init__(self, contract):
        validate_execution_policy(contract)
        need(contract.get("alignment", {}).get("mode") == "ultra_no_head", "production model must be explicit Ultra no-head")
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        self.torch, self.contract = torch, contract
        self.stats = {"asr_seconds": 0.0, "aligner_seconds": 0.0, "asr_rows": 0, "aligner_rows": 0,
                      "asr_batches": 0, "aligner_batches": 0, "asr_call_rows": {}, "aligner_call_rows": {}, "model_API_calls": 0,
                      "asr_generate_seconds": 0.0, "asr_generated_tokens": 0,
                      "asr_input_audio_seconds": 0.0, "aligner_input_audio_seconds": 0.0,
                      "asr_traced_calls": 0, "asr_traced_call_seconds": 0.0}
        self.model_facts = verify_models(contract, full=False)
        need(torch.cuda.device_count() == 1, "production replica must see exactly one GPU")
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        runtime = contract["runtime"]
        native_profile = asr_batch_profile(contract)
        self.native_profile = native_profile
        self.feature_profile = asr_feature_profile(contract)
        self._native_store = None
        self._expected_generate_rows = None
        self._native_generate_count = 0
        self.asr_compute = asr_compute_profile(contract)
        if self.asr_compute['dtype'] == 'bfloat16':
            need(torch.cuda.is_bf16_supported(), 'this CUDA device does not support requested bfloat16')
        started = time.monotonic()
        self.asr = Qwen3ASRModel.from_pretrained(contract["models"]["asr"]["path"], dtype=getattr(torch, self.asr_compute['dtype']),
                    device_map="cuda:0", local_files_only=True, trust_remote_code=False,
                    attn_implementation=self.asr_compute['attention'], max_inference_batch_size=(native_profile['max_requests'] if native_profile else runtime["max_batch"]),
                    max_new_tokens=runtime["max_new_tokens"])
        self.explicit_asr_compute = 'asr_compute' in runtime
        if self.explicit_asr_compute: apply_asr_compute(self.asr.model, self.asr_compute)
        self.aligner = Qwen3ForcedAligner.from_pretrained(contract["models"]["aligner"]["path"], dtype=torch.float32,
                    device_map="cuda:0", local_files_only=True, trust_remote_code=False, attn_implementation="eager")
        for model in (self.asr, self.aligner): model.model.eval()
        need(all(model.model.training is False for model in (self.asr, self.aligner)), "loaded models are not in eval mode")
        self.loaded_compute = {'asr': loaded_compute_facts(self.asr.model, self.asr_compute, self.explicit_asr_compute),
                               'aligner': loaded_compute_facts(self.aligner.model, {'dtype': 'float32', 'attention': 'eager'})}
        self.timestamp_attributes = {"timestamp_token_id": positive_integer_attribute(self.aligner.timestamp_token_id, "timestamp_token_id"),
                                     "timestamp_segment_ms": positive_integer_attribute(self.aligner.timestamp_segment_time, "timestamp_segment_time")}
        self.asr.model.generation_config.do_sample = False
        self.asr.model.generation_config.temperature = None; self.asr.model.generation_config.top_p = None
        self.native_padding = None
        if native_profile is not None:
            generation=self.asr.model.generation_config
            eos=generation.eos_token_id
            eos=[eos]if type(eos)is int else list(eos or [])
            need(eos and all(type(v)is int and v>=0 for v in eos),'native pilot requires actual explicit EOS IDs')
            original_pad=getattr(generation,'pad_token_id',None)
            # 原现场HF会临时采用首个EOS；新试验显式固定同一选择，不能审计后猜。
            generation.pad_token_id=eos[0]if original_pad is None else original_pad
            need(type(generation.pad_token_id)is int and generation.pad_token_id>=0,'native pilot pad token ID invalid')
            need(hasattr(self.asr.processor,'tokenizer'),'native pilot must observe actual tokenizer padding')
            self.asr.processor.tokenizer.padding_side='left'
            self.native_padding=dict(original_pad_token_id=original_pad,pad_token_id=generation.pad_token_id,
                eos_token_ids=eos,padding_side='left',policy='explicit_existing_pad_else_first_configured_EOS/v1')
        self._feature_processor=None
        if self.feature_profile is not None:
            from native_b1_feature_padding import B1FeatureBatchProcessor
            from qwen_asr.inference.utils import normalize_audio_input
            from transformers.feature_extraction_utils import BatchFeature
            self._feature_processor=B1FeatureBatchProcessor(self.asr.processor,torch_module=torch,
                normalize_audio=normalize_audio_input,max_requests=native_profile['max_requests'],batch_feature_type=BatchFeature,
                contract=contract,synthetic_fixture=contract['purpose']=='cpu_fixture')
            need(self._feature_processor.audio_id==self.asr.model.thinker.config.audio_token_id,
                 'official processor placeholder ID differs from actual audio model config')
            self.asr.processor=self._feature_processor
        self.stats["model_load_seconds"] = time.monotonic() - started
        self._captures = []
        original = self.asr.model.generate
        def capture_generate(*args, **kwargs):
            observed_ids = kwargs.get("input_ids")
            if native_profile is None:
                need(observed_ids is not None and observed_ids.ndim == 2 and observed_ids.shape[0] == 1,
                     "actual ASR generate must be exactly one row")
            else:
                self._native_generate_count += 1
                if not(observed_ids is not None and observed_ids.ndim==2 and
                       observed_ids.shape[0]==self._expected_generate_rows and
                       1<=self._expected_generate_rows<=native_profile['max_requests'] and self._native_generate_count==1):
                    raise NativeBatchRefused('actual native generate split/width/order cannot be mapped to one fixed cohort')
                if (kwargs.get('pad_token_id',self.asr.model.generation_config.pad_token_id)!=self.native_padding['pad_token_id'] or
                    self.asr.model.generation_config.pad_token_id!=self.native_padding['pad_token_id'] or
                    self.asr.processor.tokenizer.padding_side!='left'):
                    raise NativeBatchRefused('actual native padding configuration changed')
                if kwargs.get('max_new_tokens')!=runtime['max_new_tokens']:
                    raise NativeBatchRefused('actual native token limit differs from fixed contract')
                current_eos=kwargs.get('eos_token_id',self.asr.model.generation_config.eos_token_id)
                current_eos=[current_eos]if type(current_eos)is int else list(current_eos or [])
                if current_eos!=self.native_padding['eos_token_ids']:
                    raise NativeBatchRefused('actual native EOS configuration changed')
            need(self.asr.model.training is False and torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),
                 "actual ASR not eval/inference-only")
            preprocessing=None
            if self._feature_processor is not None:
                preprocessing=self._feature_processor.observe_generate(kwargs)
            generation_started = time.monotonic()
            try:
                result = original(*args, **kwargs)
                torch.cuda.synchronize(0)
            finally:
                self.stats['asr_generate_seconds'] += time.monotonic() - generation_started
            seq = result.sequences if hasattr(result, "sequences") else result
            input_ids = kwargs.get("input_ids"); need(input_ids is not None, "native ASR input tokens unobserved")
            if native_profile is not None:
                if not(seq.ndim==2 and seq.shape[0]==input_ids.shape[0] and seq.shape[1]>=input_ids.shape[1] and
                       seq[:,:input_ids.shape[1]].detach().cpu().tolist()==input_ids.detach().cpu().tolist()):
                    raise NativeBatchRefused('actual native output sequence prefix/row mapping differs from prompt')
            generated = seq[:, input_ids.shape[1]:]
            rows = generated.detach().cpu().tolist()
            text = self.asr.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            eos = self.asr.model.generation_config.eos_token_id
            eos = [eos] if type(eos) is int else list(eos or [])
            if native_profile is not None:
                mask=kwargs.get('attention_mask')
                if mask is None: raise NativeBatchRefused('actual native attention mask missing')
                packets=native_token_rows(input_ids.detach().cpu().tolist(),mask.detach().cpu().tolist(),rows,text,eos,
                    getattr(self.asr.model.generation_config,'pad_token_id',None),kwargs.get('max_new_tokens'))
                self.stats['asr_generated_tokens'] += sum(v['effective_generated_token_count']for v in packets)
                self.stats['asr_returned_token_slots'] = self.stats.get('asr_returned_token_slots',0)+sum(len(row)for row in rows)
                self.stats['asr_prompt_token_slots']=self.stats.get('asr_prompt_token_slots',0)+sum(v['input_padded_tokens']for v in packets)
                self.stats['asr_prompt_nonpad_tokens']=self.stats.get('asr_prompt_nonpad_tokens',0)+sum(v['input_nonpad_tokens']for v in packets)
                for packet in packets:packet.update(model_training=self.asr.model.training,
                    inference_mode=bool(torch.is_inference_mode_enabled()),grad_enabled=bool(torch.is_grad_enabled()),
                    padding_policy=copy.deepcopy(self.native_padding))
                if preprocessing is not None:
                    need(len(preprocessing)==len(packets),'actual B1 feature witness row denominator changed')
                    for packet,proof in zip(packets,preprocessing):packet['preprocessing']=proof
                self._captures.extend(packets)
                return result
            self.stats['asr_generated_tokens'] += sum(len(row) for row in rows)
            self._captures.extend({"generated_token_ids": ids, "native_decoded_raw_answer": raw,
                                   "eos_observed": any(x in eos for x in ids),
                                   "first_eos_index": next((j for j, x in enumerate(ids) if x in eos), None),
                                   "max_new_tokens": kwargs.get("max_new_tokens"),
                                   "input_padded_tokens": int(input_ids.shape[1]), "actual_generate_rows": int(input_ids.shape[0]),
                                   "model_training": self.asr.model.training, "inference_mode": bool(torch.is_inference_mode_enabled()),
                                   "grad_enabled": bool(torch.is_grad_enabled())} for ids, raw in zip(rows, text))
            return result
        self.asr.model.generate = capture_generate
        languages = self.aligner.get_supported_languages()
        self.supported_languages = {str(x).lower() for x in languages} if languages is not None else None
        self.module_provenance = self.verify_loaded_modules()
        self.compute_observation = None

    def verify_loaded_modules(self):
        expected = {str(Path(path).resolve()): pin for path, pin in self.contract["source_files"].items()
                    if "qwen_asr" in Path(path).parts and Path(path).suffix == ".py"}
        required = {"qwen_asr", "qwen_asr.inference.qwen3_asr", "qwen_asr.inference.qwen3_forced_aligner", "qwen_asr.inference.utils"}
        need(required <= set(sys.modules), "required Qwen modules were not actually imported")
        result = {}
        for name, module in list(sys.modules.items()):
            if not (name == "qwen_asr" or name.startswith("qwen_asr.")): continue
            raw = getattr(module, "__file__", None)
            if raw is None: continue
            path = str(Path(raw).resolve())
            need(path in expected and sha_file(path) == expected[path], "actually imported Qwen module is not the pinned source: " + name + " " + path)
            result[name] = {"path": path, "sha256": expected[path]}
        return result

    def _observe_call(self, stage, batch):
        policy = validate_execution_policy(self.contract)
        native_profile = asr_batch_profile(self.contract)
        if native_profile is not None:
            need(1 <= len(batch) <= (native_profile['max_requests'] if stage=='asr' else 1),
                 'native ASR width or single-window Ultra call differs')
        if policy == "single_window_independent":
            need(len(batch) == 1, "single-window contract reached a multirow " + stage + " model call")
        widths = self.stats[stage + "_call_rows"]; key = str(len(batch))
        widths[key] = widths.get(key, 0) + 1
        result = {"stage": stage, "rows": len(batch), "producer_row_ids": [r["row_id"] for r in batch],
                  "input_frames": [len(r["audio"]) for r in batch], "execution_policy": policy}
        if stage == 'asr' and 'asr_compute' in self.contract['runtime']:
            result['asr_compute'] = dict(self.asr_compute)
        return result

    def prepare_native_group(self, root, contract_sha, bundle_sha):
        """组地址由完整原来源组固定；不能用本次cache-miss另起计划。"""
        profile=asr_batch_profile(self.contract)
        need(profile is not None, 'native batch store needs explicit pilot policy')
        from native_transformers_batch import NativeTransformersBatchStore
        model_sha=digest(model_provenance(self.contract)['asr'])
        producer=digest(dict(contract_sha256=contract_sha,bundle_sha256=bundle_sha,model_sha256=model_sha,
                            execution_policy=self.contract['runtime']['execution_policy'],asr_batching=profile))
        self._native_store=NativeTransformersBatchStore(root,producer,synthetic_fixture=self.contract['purpose']=='cpu_fixture',
            contract_sha256=contract_sha,model_sha256=model_sha,max_batch_rows=profile['max_requests'])

    def transcribe(self, rows):
        profile=asr_batch_profile(self.contract)
        if profile is None:return self._transcribe_original_rows(rows)
        need(getattr(self,'_native_store',None) is not None, 'worker must prepare stable native source-group store')
        if not rows:return {}
        started=time.monotonic();prior_model_seconds=self.stats['asr_seconds']
        requests=[]
        for row in sorted(rows,key=lambda r:(len(r['audio']),r['row_id'])):
            source=row['context']['source'];window=row['window']
            requests.append(dict(row,row_id=row['row_id'],source_id=source['source_id'],window_id=window['window_id'],
                source_record_sha256=digest(source),input_pcm_sha256=window['input_pcm_sha256'],
                model_input_frames=window['model_input_frames'],model_sample_rate=window['model_sample_rate']))
        result=self._native_store.run_batch(requests,infer_batch=self._transcribe_native_cohort)
        self.stats['native_ASR_stage_cache_hits']=self.stats.get('native_ASR_stage_cache_hits',0)+self._native_store.metrics['stage_cache_hits']
        self.stats['native_ASR_batches_committed']=self.stats.get('native_ASR_batches_committed',0)+self._native_store.metrics['new_batches']
        elapsed=time.monotonic()-started
        self.stats['native_ASR_stage_wall_seconds']=self.stats.get('native_ASR_stage_wall_seconds',0)+elapsed
        self.stats['native_ASR_store_and_mapping_seconds']=self.stats.get('native_ASR_store_and_mapping_seconds',0)+max(0.,elapsed-(self.stats['asr_seconds']-prior_model_seconds))
        return result

    def _transcribe_native_cohort(self, batch):
        """复用同一ASR实现，只把这一完整cohort作为一次实际模型调用。"""
        return self._transcribe_original_rows(batch, native_cohort=True)

    def _transcribe_original_rows(self, rows, *, native_cohort=False):
        torch = self.torch; runtime = self.contract["runtime"]; outputs = {}
        batches=[rows] if native_cohort else length_batches(rows, runtime["max_batch"], runtime["length_ratio_limit"])
        for batch in batches:
            self._captures.clear(); started = time.monotonic(); ids = [r["row_id"] for r in batch]
            call = self._observe_call("asr", batch)
            trace = None
            self._expected_generate_rows=len(batch);self._native_generate_count=0
            try:
                torch.cuda.synchronize(0)
                trace = ASRAttentionTrace(self.asr.model, torch) if self.compute_observation is None else None
                with torch.inference_mode(), (trace if trace is not None else contextlib.nullcontext()):
                    if getattr(self,'_feature_processor',None)is not None:self._feature_processor.begin_cohort(batch)
                    result = self.asr.transcribe(audio=[(r["audio"], 16000) for r in batch], language=None, return_time_stamps=False)
                torch.cuda.synchronize(0)
                if trace is not None:
                    self.compute_observation = trace.observation(self.asr_compute if self.explicit_asr_compute else None)
                if native_cohort and self._native_generate_count!=1:
                    raise NativeBatchRefused('one fixed cohort must map to exactly one actual generate call')
                need(len(result) == len(batch) == len(self._captures), "ASR native row coverage differs")
                for row_id, answer, native in zip(ids, result, self._captures):
                    audit = normalize_with_audit(str(answer.text), runtime.get("normalization_profile", POLICY))
                    outputs[row_id] = {"status": "completed" if native["eos_observed"] else "truncated", "raw_text": str(answer.text),
                                       "normalized_text": audit["normalized_text"], "normalization_audit": audit, "language": answer.language,
                                       "native": native, "batch_rows": ids, "batch_size": len(batch), "inference_call": call}
            except Exception as exc:
                if native_cohort:
                    exc.native_batch_evidence=dict(expected_row_ids=ids,actual_generate_count=self._native_generate_count,
                        native_generate_captures=copy.deepcopy(self._captures),padding_policy=copy.deepcopy(getattr(self,'native_padding',None)))
                    raise
                if isinstance(exc, (ComputeProfileRefused,NativeBatchRefused)): raise
                if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc) or "device-side assert" in str(exc):
                    raise RuntimeError("fatal GPU error; preserve attempt, no silent batch reduction") from exc
                error_trace = traceback.format_exc()
                for row_id in ids: outputs[row_id] = {"status": "failed", "raw_text": "", "normalized_text": "", "language": None,
                                                       "native": {"eos_observed": False}, "normalization_audit": normalize_with_audit("",runtime.get("normalization_profile", POLICY)), "error": error_trace,
                                                       "native_generate_captures": list(self._captures), "batch_rows": ids, "inference_call": call}
            finally:
                self._expected_generate_rows=None
                elapsed = time.monotonic() - started
                self.stats["asr_seconds"] += elapsed; self.stats["asr_rows"] += len(batch); self.stats["asr_batches"] += 1
                self.stats['asr_input_audio_seconds'] += sum(len(row['audio']) / 16000 for row in batch)
                if trace is not None:
                    self.stats['asr_traced_calls'] += 1
                    self.stats['asr_traced_call_seconds'] += elapsed
        return native_json(outputs)

    def align(self, rows):
        torch = self.torch; runtime = self.contract["runtime"]; outputs = {}; eligible = []
        for row in rows:
            asr = row["asr"]; row_id = row["row_id"]
            reason = None
            if asr["status"] != "completed": reason = "asr_not_complete"
            elif not asr["normalized_text"]: reason = "empty_asr_not_proven_noise"
            elif self.supported_languages is not None and str(asr["language"]).lower() not in self.supported_languages: reason = "unsupported_alignment_language"
            if reason:
                outputs[row_id] = {"status": "not_applicable", "reason": reason, "items": [], "target_text": asr.get("normalized_text", ""),
                                   "native_timestamps_ms": [], "fixed_timestamps_ms": [], "repair_changed": False}
            else: eligible.append(row)
        for batch in length_batches(eligible, runtime["max_batch"], runtime["length_ratio_limit"]):
            need(len(batch) == 1, "Ultra no-head must execute exactly one window per call")
            row = batch[0]; row_id = row["row_id"]; started = time.monotonic(); observer = None
            call = self._observe_call("aligner", batch)
            try:
                observer = AlignmentObserver(self.aligner, torch)
                torch.cuda.synchronize(0)
                with torch.inference_mode(), observer:
                    result = self.aligner.align(audio=[(row["audio"], 16000)],
                        text=[row["asr"]["normalized_text"]], language=[row["asr"]["language"]])
                torch.cuda.synchronize(0)
                need(len(result) == 1, "Ultra no-head output row coverage differs")
                items = [{"text": str(x.text), "start_time": float(x.start_time), "end_time": float(x.end_time)} for x in result[0]]
                outputs[row_id] = successful_alignment(observer, items, row["asr"]["normalized_text"])
                outputs[row_id].update(batch_rows=[row_id], inference_call=call,
                                      actual_supported_languages=sorted(self.supported_languages) if self.supported_languages is not None else None)
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc) or "device-side assert" in str(exc):
                    raise RuntimeError("fatal GPU error; preserve attempt, no silent batch reduction") from exc
                observation = observer.observation(failed=True) if observer is not None else {"status": "observer_initialization_failed"}
                forward = observation.get("thinker_forwards", [])
                fix = observation.get("FIX_calls", [])
                outputs[row_id] = {"status": "failed", "method": "ultra_no_head", "items": [],
                    "native_timestamps_ms": forward[0].get("raw_timestamps_ms", []) if forward else [],
                    "fixed_timestamps_ms": fix[-1].get("fixed_timestamps_ms") if fix else None,
                    "repair_changed": fix[-1].get("repair_changed") if fix else None,
                    "target_text": row["asr"]["normalized_text"], "error": traceback.format_exc(),
                    "native_observation": observation, "batch_rows": [row_id], "inference_call": call,
                    "head_used": False, "extra_clipping_or_interpolation": False}
            finally:
                self.stats["aligner_seconds"] += time.monotonic() - started
                self.stats["aligner_rows"] += 1; self.stats["aligner_batches"] += 1
                self.stats['aligner_input_audio_seconds'] += len(row['audio']) / 16000

        return native_json(outputs)

    def snapshot_stats(self, include_identity=False):
        """本进程累计计时快照；不作模型全文件校验、不等待GPU或读原音。"""
        result = copy.deepcopy(self.stats) | {'asr_compute_requested': dict(self.asr_compute),
            'asr_compute_explicit': self.explicit_asr_compute,
            'first_call_trace_completed': self.compute_observation is not None,
            'trace_call_time_included_in_asr_seconds': True,
            'trace_call_seconds_are_overhead_only': False,
            'cuda_allocated_peak_bytes': self.torch.cuda.max_memory_allocated(0),
            'cuda_reserved_peak_bytes': self.torch.cuda.max_memory_reserved(0)}
        if include_identity:
            result.update(loaded_compute=copy.deepcopy(self.loaded_compute),
                          first_call_compute_observation=copy.deepcopy(self.compute_observation))
            if getattr(self,'native_profile',None) is not None:
                result['native_asr_padding']=copy.deepcopy(self.native_padding)
            if getattr(self,'_feature_processor',None)is not None:
                result['asr_preprocessing_sources_observed']=copy.deepcopy(self._feature_processor.source_observation)
        return result

    def metrics(self):
        torch = self.torch; torch.cuda.synchronize(0)
        profile=asr_batch_profile(self.contract)
        if profile is not None:
            need(self.asr.model.generation_config.pad_token_id==self.native_padding['pad_token_id'] and
                 self.asr.processor.tokenizer.padding_side=='left','native model padding state changed')
            need(set(self.stats['aligner_call_rows']) <= {'1'} and self.stats['aligner_rows']==self.stats['aligner_batches'],
                 'native ASR policy changed actual Ultra single-window calls')
            calls=self.stats['asr_call_rows']
            need(all(k.isdigit()and 1<=int(k)<=profile['max_requests']for k in calls) and
                 sum(int(k)*v for k,v in calls.items())==self.stats['asr_rows'] and sum(calls.values())==self.stats['asr_batches'],
                 'actual native ASR batch width/accounting differs')
        if validate_execution_policy(self.contract) == "single_window_independent":
            for stage in ("asr", "aligner"):
                need(set(self.stats[stage + "_call_rows"]) <= {"1"}
                     and self.stats[stage + "_rows"] == self.stats[stage + "_batches"],
                     "observed model calls violated single-window semantics: " + stage)
        verify_model_stats(self.model_facts)
        actual_modules = self.verify_loaded_modules()
        need(all(actual_modules.get(name) == value for name, value in self.module_provenance.items()), "runtime Qwen module bytes changed")
        need(all(model.model.training is False for model in (self.asr, self.aligner)), "model eval state changed")
        need(loaded_compute_facts(self.asr.model, self.asr_compute, self.explicit_asr_compute) == self.loaded_compute['asr'], 'loaded ASR compute state changed')
        need(loaded_compute_facts(self.aligner.model, {'dtype': 'float32', 'attention': 'eager'}) == self.loaded_compute['aligner'], 'loaded aligner compute state changed')
        return self.snapshot_stats(include_identity=True) | {"alignment_mode": "ultra_no_head", "timestamp_attributes": self.timestamp_attributes,
                             "actual_model_training_states": {"asr": self.asr.model.training, "aligner": self.aligner.model.training},
                             "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
                             "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(0)}


class FixtureBackend:
    """Explicit synthetic CPU fixtures; prohibited for silver/pilot contracts."""
    name = "synthetic_cpu_fixture"
    def __init__(self, contract):
        need(contract["purpose"] == "cpu_fixture", "synthetic backend cannot label real production")
        self.contract = contract
    def transcribe(self, rows):
        return {r["row_id"]: {"status": "completed", "raw_text": "fixture", "normalized_text": "fixture", "normalization_audit": normalize_with_audit("fixture"), "language": "English",
                             "native": {"eos_observed": True, "generated_token_ids": [1, 2], "native_decoded_raw_answer": "fixture"}} for r in rows}
    def align(self, rows):
        return {r["row_id"]: {"status": "completed_native_candidates", "items": [{"text": "fixture", "start_time": 0.08, "end_time": 0.16}],
                             "native_timestamps_ms": [80, 160], "fixed_timestamps_ms": [80, 160], "repair_changed": False,
                             "target_text": r["asr"]["normalized_text"]} for r in rows}
    def snapshot_stats(self): return {"model_API_calls": 0, "synthetic_fixture": True}
    def metrics(self): return self.snapshot_stats()
