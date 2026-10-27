"""转写候选配置和周期计时的CPU反例；不导入PyTorch或使用GPU。"""
import copy
import contextlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import Refused, asr_compute_profile, canonical_bytes, read_contract, sha_file
from freeze_contract import freeze
from qwen_backend import (ASRAttentionTrace, ComputeProfileRefused, QwenBackend,
                          apply_asr_compute, loaded_compute_facts)
from worker import progress_performance, timed_call
from test_maintenance_entries import external_contract


class Config:
    sub_configs = {}
    def __init__(self, attention): self._attn_implementation = attention


class Handle:
    def __init__(self, hooks, key): self.hooks, self.key = hooks, key
    def remove(self): self.hooks.pop(self.key, None)


class SyntheticAttention:
    def __init__(self, config): self.config = config; self.before = {}; self.after = {}; self.number = 0
    def register_forward_pre_hook(self, hook):
        self.number += 1; self.before[self.number] = hook; return Handle(self.before, self.number)
    def register_forward_hook(self, hook):
        self.number += 1; self.after[self.number] = hook; return Handle(self.after, self.number)
    def forward(self, functional=None, explode=False):
        for hook in list(self.before.values()): hook(self, ())
        if explode: raise ValueError('synthetic forward failure')
        tensor = types.SimpleNamespace(dtype='torch.float32')
        result = functional.scaled_dot_product_attention(tensor, tensor, tensor) if functional else None
        for hook in list(self.after.values()): hook(self, (), result)
        return result


class Model:
    def __init__(self, dtype='float32', attention='sdpa'):
        self.config = Config('eager')
        self.config.audio_config = Config(attention)
        self.config.text_config = Config(attention)
        self.config.sub_configs = {'audio_config': Config, 'text_config': Config}
        self.audio = SyntheticAttention(self.config.audio_config)
        self.text = SyntheticAttention(self.config.text_config)
        self.parameter = types.SimpleNamespace(dtype='torch.' + dtype, is_floating_point=lambda: True, numel=lambda: 7)
    def parameters(self): return iter([self.parameter])
    def named_modules(self): return iter([('', self), ('audio', self.audio), ('text', self.text)])


def fake_torch():
    return types.SimpleNamespace(nn=types.SimpleNamespace(functional=types.SimpleNamespace(scaled_dot_product_attention=lambda *a, **k: 'observed')))


class ComputeProfileTests(unittest.TestCase):
    def test_absent_profile_preserves_default_and_does_not_mutate_contract(self):
        contract = {'runtime': {}}; before = copy.deepcopy(contract)
        self.assertEqual(asr_compute_profile(contract), {'dtype': 'float32', 'attention': 'eager'})
        self.assertEqual(contract, before)

    def test_invalid_or_partial_profiles_are_refused(self):
        for value in [None, [], {}, {'dtype': 'float32'}, {'dtype': 'float16', 'attention': 'sdpa'},
                      {'dtype': 'float32', 'attention': 'flash_attention_2'},
                      {'dtype': [], 'attention': 'sdpa'}, {'dtype': 'float32', 'attention': 'sdpa', 'fallback': True}]:
            with self.subTest(value=value), self.assertRaises(Refused):
                asr_compute_profile({'runtime': {'asr_compute': value}})

    def test_explicit_profile_sets_audio_text_and_root_but_observation_alone_does_not(self):
        model = Model(); expected = {'dtype': 'float32', 'attention': 'eager'}
        legacy = loaded_compute_facts(model, expected)
        self.assertEqual(model.audio.config._attn_implementation, 'sdpa')
        self.assertFalse(legacy['strict_attention_requested'])
        with self.assertRaises(Refused): loaded_compute_facts(model, expected, True)
        apply_asr_compute(model, expected)
        facts = loaded_compute_facts(model, expected, True)
        self.assertTrue(all(row['attention_config'] == 'eager' for row in facts['attention_configurations']))

    def test_loaded_dtype_mismatch_is_refused(self):
        with self.assertRaises(Refused): loaded_compute_facts(Model('bfloat16'), {'dtype': 'float32', 'attention': 'sdpa'})

    def test_real_function_entry_trace_binds_every_observed_attention_module(self):
        model, torch = Model(), fake_torch(); original = torch.nn.functional.scaled_dot_product_attention
        with ASRAttentionTrace(model, torch) as trace:
            model.audio.forward(torch.nn.functional); model.text.forward(torch.nn.functional)
        observed = trace.observation({'dtype': 'float32', 'attention': 'sdpa'})
        self.assertEqual(observed['functional_sdpa_calls'], {'audio': 1, 'text': 1})
        self.assertEqual(observed['functional_sdpa_qkv_dtypes']['audio'], ['float32'] * 3)
        self.assertFalse(observed['CUDA_kernel_trace_performed'])
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
        self.assertFalse(model.audio.before or model.audio.after or model.text.before or model.text.after)
        with self.assertRaises(ComputeProfileRefused): trace.observation({'dtype': 'float32', 'attention': 'eager'})
        with self.assertRaises(ComputeProfileRefused): trace.observation({'dtype': 'bfloat16', 'attention': 'sdpa'})

    def test_mixed_attention_dispatch_is_not_accepted_as_all_sdpa(self):
        model, torch = Model(), fake_torch()
        with ASRAttentionTrace(model, torch) as trace:
            model.audio.forward(); model.text.forward(torch.nn.functional)
        with self.assertRaises(ComputeProfileRefused): trace.observation({'dtype': 'float32', 'attention': 'sdpa'})
        self.assertEqual(trace.observation()['attention_forwards'], {'audio': 1, 'text': 1})

    def test_partial_model_execution_cannot_prove_full_asr_compute(self):
        model, torch = Model(), fake_torch()
        with ASRAttentionTrace(model, torch) as trace: model.text.forward(torch.nn.functional)
        self.assertEqual(trace.observation()['unexecuted_attention_modules'], ['audio'])
        with self.assertRaises(ComputeProfileRefused): trace.observation({'dtype': 'float32', 'attention': 'sdpa'})

    def test_trace_restores_function_and_hooks_after_failure(self):
        model, torch = Model(), fake_torch(); original = torch.nn.functional.scaled_dot_product_attention
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            with ASRAttentionTrace(model, torch): model.audio.forward(explode=True)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
        self.assertFalse(model.audio.before or model.audio.after or model.text.before or model.text.after)

    def test_trace_restores_partially_installed_hooks_if_registration_fails(self):
        model, torch = Model(), fake_torch(); original = torch.nn.functional.scaled_dot_product_attention
        with patch.object(model.text, 'register_forward_hook', side_effect=ValueError('injected registration')):
            with self.assertRaisesRegex(ValueError, 'registration'):
                with ASRAttentionTrace(model, torch): pass
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
        self.assertFalse(model.audio.before or model.audio.after or model.text.before or model.text.after)

    def test_new_freeze_records_asr_profile_without_changing_fa_or_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); parent = root/'parent.json'; c = external_contract(); parent.write_bytes(canonical_bytes(c)); raw = parent.read_bytes()
            profile = {'dtype': 'bfloat16', 'attention': 'sdpa'}
            result = freeze(parent, sha_file(parent), root/'new', 'speed-test', asr_compute=profile)
            a = read_contract(result['contracts']['pilot_validation']['path'])
            self.assertEqual(a['runtime']['asr_compute'], profile)
            self.assertEqual((a['runtime']['dtype'], a['runtime']['attention'], a['runtime']['max_batch']), ('float32', 'eager', 1))
            self.assertEqual(parent.read_bytes(), raw)
            self.assertFalse(result['models_executed'])
            invalid = root/'invalid'
            with self.assertRaises(RuntimeError): freeze(parent, sha_file(parent), invalid, 'bad', asr_compute={})
            self.assertFalse(invalid.exists())

    def test_actual_loader_options_change_only_ASR_and_empty_execution_remains_unobserved(self):
        calls = []
        def load(path, **options):
            calls.append((path, options)); model = Model(dtype=options['dtype'])
            model.training = True
            def evaluation(): model.training = False
            model.eval = evaluation; model.generate = lambda **kwargs: None
            model.generation_config = types.SimpleNamespace()
            return types.SimpleNamespace(model=model, timestamp_token_id=151705, timestamp_segment_time=80,
                                         get_supported_languages=lambda: ['English'])
        torch = fake_torch(); torch.float32 = 'float32'; torch.bfloat16 = 'bfloat16'
        torch.set_num_threads = lambda n: None
        torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace())
        torch.cuda = types.SimpleNamespace(device_count=lambda: 1, is_bf16_supported=lambda: True,
            synchronize=lambda n: None, max_memory_allocated=lambda n: 100, max_memory_reserved=lambda n: 200)
        qwen = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=load),
                                     Qwen3ForcedAligner=types.SimpleNamespace(from_pretrained=load))
        contract = external_contract(); contract['models'] = {'asr': {'path': 'ASR'}, 'aligner': {'path': 'FA'}}
        contract['runtime']['asr_compute'] = {'dtype': 'bfloat16', 'attention': 'sdpa'}
        with patch.dict(sys.modules, {'torch': torch, 'qwen_asr': qwen}), \
             patch('qwen_backend.verify_models', return_value={}), \
             patch.object(QwenBackend, 'verify_loaded_modules', return_value={}):
            backend = QwenBackend(contract)
            self.assertEqual(calls[0][1]['dtype'], 'bfloat16')
            self.assertEqual(calls[0][1]['attn_implementation'], 'sdpa')
            self.assertEqual(calls[0][1]['max_inference_batch_size'], 1)
            self.assertEqual(calls[1][1]['dtype'], 'float32')
            self.assertEqual(calls[1][1]['attn_implementation'], 'eager')
            snapshot = backend.snapshot_stats()
            self.assertFalse(snapshot['first_call_trace_completed'])
            self.assertNotIn('loaded_compute', snapshot)
            self.assertNotIn('first_call_compute_observation', snapshot)
            with patch('qwen_backend.verify_model_stats'):
                final = backend.metrics()
            self.assertIsNone(final['first_call_compute_observation'])
            self.assertIn('loaded_compute', final)
            self.assertTrue(final['trace_call_time_included_in_asr_seconds'])

        calls.clear(); torch.cuda.is_bf16_supported = lambda: False
        with patch.dict(sys.modules, {'torch': torch, 'qwen_asr': qwen}), \
             patch('qwen_backend.verify_models', return_value={}):
            with self.assertRaisesRegex(Refused, 'bfloat16'): QwenBackend(contract)
        self.assertEqual(calls, [])

    def test_compute_dispatch_refusal_is_fatal_not_an_unknown_audio_result(self):
        backend = self.transcribe_backend()
        class BadTrace:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def observation(self, *args): raise ComputeProfileRefused('synthetic mismatch')
        with patch('qwen_backend.ASRAttentionTrace', BadTrace):
            with self.assertRaisesRegex(ComputeProfileRefused, 'mismatch'):
                backend.transcribe([{'row_id': 'synthetic', 'audio': [0.0] * 160}])
        self.assertEqual(backend.stats['asr_traced_calls'], 1)
        self.assertEqual(backend.stats['asr_rows'], 1)

    def transcribe_backend(self):
        backend = QwenBackend.__new__(QwenBackend)
        backend.contract = external_contract(); backend.contract['runtime']['asr_compute'] = {'dtype': 'float32', 'attention': 'sdpa'}
        backend.asr_compute = {'dtype': 'float32', 'attention': 'sdpa'}; backend.explicit_asr_compute = True
        backend.stats = {'asr_seconds': 0.0, 'asr_rows': 0, 'asr_batches': 0, 'asr_call_rows': {},
                         'asr_input_audio_seconds': 0.0, 'asr_traced_calls': 0, 'asr_traced_call_seconds': 0.0}
        backend.compute_observation = None; backend._captures = []
        backend.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda n: None, OutOfMemoryError=MemoryError),
                                              inference_mode=contextlib.nullcontext)
        backend.asr = types.SimpleNamespace(model=Model(), transcribe=lambda **kwargs: [])
        return backend

    def test_later_untraced_call_failure_does_not_inflate_trace_timing(self):
        backend = self.transcribe_backend(); backend.compute_observation = {'already_observed': True}
        def fail(**kwargs): raise ValueError('synthetic later source error')
        backend.asr.transcribe = fail
        result = backend.transcribe([{'row_id': 'synthetic', 'audio': [0.0] * 160}])
        self.assertEqual(result['synthetic']['status'], 'failed')
        self.assertIn('synthetic later source error', result['synthetic']['error'])
        self.assertEqual(backend.stats['asr_traced_calls'], 0)
        self.assertEqual(backend.stats['asr_traced_call_seconds'], 0)
        self.assertEqual(backend.stats['asr_rows'], 1)


class TimingTests(unittest.TestCase):
    def counters(self):
        return {'assigned_source_rows': 3, 'sidecars_committed': 2, 'sidecar_cache_hits': 1,
                'stage_cache_hits': {'asr': 1, 'aligner': 1}, 'model_windows': 2,
                'decoded_audio_seconds': 4.5, 'pipeline_seconds': {'window_read': 0.1}}

    def test_periodic_snapshot_uses_lightweight_path_and_detaches_mutable_values(self):
        backend = types.SimpleNamespace(snapshot_stats=lambda: {'asr_seconds': 2.0, 'aligner_seconds': .25,
            'asr_rows': 2, 'loaded_compute': {'dimensions': [7]}, 'first_call_compute_observation': {'functional_sdpa_calls': {'audio': 12}}},
            metrics=lambda: (_ for _ in ()).throw(AssertionError('terminal audit must not run')))
        counters = self.counters()
        with patch('worker.time.monotonic', return_value=20):
            one, previous = progress_performance(backend, counters, 'explicit-start', 10)
        self.assertEqual(one['scope'], 'this_worker_process_only')
        self.assertFalse(one['prior_attempt_elapsed_included'])
        self.assertEqual(one['since_previous_chunk']['backend']['asr_seconds'], 2.0)
        self.assertNotIn('loaded_compute', one['since_previous_chunk']['backend'])
        counters['stage_cache_hits']['asr'] = 9
        self.assertEqual(previous['counters']['stage_cache_hits']['asr'], 1)
        with patch('worker.time.monotonic', return_value=22):
            two, _ = progress_performance(backend, counters, 'explicit-start', 10, previous)
        self.assertEqual(two['since_previous_chunk']['worker_elapsed_seconds'], 2)
        self.assertEqual(two['since_previous_chunk']['backend']['asr_seconds'], 0)
        self.assertEqual(two['since_previous_chunk']['counters']['stage_cache_hits']['asr'], 8)

    def test_new_process_resets_timing_and_does_not_add_old_attempt(self):
        backend = types.SimpleNamespace(snapshot_stats=lambda: {'asr_seconds': .5})
        with patch('worker.time.monotonic', return_value=101):
            value, _ = progress_performance(backend, self.counters(), 'new-process', 100)
        self.assertEqual(value['cumulative']['worker_elapsed_seconds'], 1)
        self.assertEqual(value['since_previous_chunk']['backend']['asr_seconds'], .5)
        self.assertFalse(value['prior_attempt_elapsed_included'])

    def test_timed_exception_preserves_original_error_and_records_elapsed(self):
        counters = {}
        def fail(): raise ValueError('unchanged source failure')
        with patch('worker.time.monotonic', side_effect=[4.0, 4.25]):
            with self.assertRaisesRegex(ValueError, 'unchanged source failure'): timed_call(counters, 'source_audit', fail)
        self.assertEqual(counters['pipeline_seconds']['source_audit'], .25)
