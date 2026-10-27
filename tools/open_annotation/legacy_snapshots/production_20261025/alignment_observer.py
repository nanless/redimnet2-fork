"""Observations of Ultra no-head through the pinned Qwen API; never retimes it."""
from __future__ import annotations
import base64
import hashlib
import math
import numbers
import traceback
from contracts import native_json, need


def positive_integer_attribute(value, name):
    need(not isinstance(value, bool) and isinstance(value, numbers.Real),
         'actual ' + name + ' is not a real integer-valued number')
    need(math.isfinite(value) and value == int(value) and value > 0,
         'actual ' + name + ' must be finite, positive and integer-valued')
    return int(value)


def flatten_vector(data):
    value = data.tolist() if hasattr(data, 'tolist') else data
    need(isinstance(value, (list, tuple)), 'native timestamp vector is not a sequence')
    return native_json(list(value))


class AlignmentObserver:
    """One inference row. Hash full timestamp-slot logits, preserve every FIX call.

    Full slot tensors are inline only on software failure, avoiding millions of
    giant per-window tensor files. Raw classes/times are always kept losslessly.
    """
    def __init__(self, aligner, torch):
        self.aligner, self.torch = aligner, torch
        self.fixes, self.encoded, self.forwards = [], [], []
        self.restores, self.handles, self.slot_bytes = [], [], None
        self.token_id = positive_integer_attribute(aligner.timestamp_token_id, 'timestamp_token_id')
        self.segment_ms = positive_integer_attribute(aligner.timestamp_segment_time, 'timestamp_segment_time')
        self.attributes = {'timestamp_token_id': self.token_id, 'timestamp_segment_ms': self.segment_ms,
                           'raw_token_id_type': type(aligner.timestamp_token_id).__name__,
                           'raw_segment_ms_type': type(aligner.timestamp_segment_time).__name__}

    def __enter__(self):
        try:
            processor = self.aligner.aligner_processor
            original = processor.fix_timestamp
            def fix(*args, **kwargs):
                data = args[0] if args else kwargs.get('timestamp')
                evidence = {'native_timestamps_ms': flatten_vector(data), 'fixed_timestamps_ms': None,
                            'capture_status': 'observed_before_FIX', 'repair_changed': None}
                self.fixes.append(evidence)
                try:
                    result = original(*args, **kwargs)
                    after = flatten_vector(result)
                except BaseException:
                    evidence.update(capture_status='FIX_failed', traceback=traceback.format_exc())
                    raise
                evidence.update(fixed_timestamps_ms=after, capture_status='FIX_completed',
                                repair_changed=evidence['native_timestamps_ms'] != after)
                return result
            self.restores.append((processor, 'fix_timestamp', original))
            processor.fix_timestamp = fix
            encode_original = processor.encode_timestamp
            def encode(*args, **kwargs):
                result = encode_original(*args, **kwargs)
                self.encoded.append({'words': [str(x) for x in result[0]],
                                     'prompt': str(result[1]), 'arguments': native_json(args),
                                     'keyword_arguments': native_json(kwargs)})
                return result
            self.restores.append((processor, 'encode_timestamp', encode_original))
            processor.encode_timestamp = encode
            def forward(module, args, kwargs, result):
                evidence = {'status': 'forward_observed', 'input_tensors': {},
                            'inference_mode': bool(self.torch.is_inference_mode_enabled()),
                            'grad_enabled': bool(self.torch.is_grad_enabled()),
                            'module_training': getattr(module, 'training', None)}
                self.forwards.append(evidence)
                ids, logits = kwargs.get('input_ids'), getattr(result, 'logits', None)
                need(ids is not None and logits is not None, 'native thinker IDs/logits unobserved')
                evidence['input_ids_shape'] = list(ids.shape)
                need(ids.ndim == 2 and ids.shape[0] == 1, 'actual thinker row count differs from one')
                positions = self.torch.nonzero(ids[0] == self.token_id, as_tuple=False).flatten()
                evidence['timestamp_slot_positions'] = positions.detach().cpu().tolist()
                need(len(self.encoded) == 1 and positions.numel() == 2 * len(self.encoded[0]['words']),
                     'actual timestamp slots differ from native target words')
                for key, tensor in kwargs.items():
                    if isinstance(tensor, self.torch.Tensor):
                        a = tensor.detach().cpu().contiguous()
                        raw = a.reshape(-1).view(self.torch.uint8).numpy().tobytes()
                        evidence['input_tensors'][key] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                                                          'sha256_raw_tensor_bytes': hashlib.sha256(raw).hexdigest()}
                rows = logits[0].index_select(0, positions).detach().cpu().contiguous()
                need(rows.dtype == self.torch.float32, 'unvalidated native timestamp logits dtype')
                self.slot_bytes = rows.numpy().astype('<f4', copy=False).tobytes()
                indices = rows.argmax(-1).tolist()
                evidence.update(slot_logit_shape=list(rows.shape), slot_logit_dtype=str(rows.dtype),
                                slot_logits_sha256=hashlib.sha256(self.slot_bytes).hexdigest(),
                                all_slot_logits_finite=bool(self.torch.isfinite(rows).all()),
                                raw_argmax_class_indices=indices,
                                raw_timestamps_ms=[x * self.segment_ms for x in indices])
                need(evidence['all_slot_logits_finite'], 'nonfinite native timestamp logits')
                need(evidence['module_training'] is False and evidence['inference_mode'] and not evidence['grad_enabled'],
                     'actual thinker not eval/inference-only')
                evidence['status'] = 'completed'
            self.handles.append(self.aligner.model.thinker.register_forward_hook(forward, with_kwargs=True))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def observation(self, failed=False):
        result = {'schema': 'ultra-no-head-native-observation/v1', 'head_used': False,
                  'timestamp_attributes': self.attributes, 'target_encoding': self.encoded,
                  'thinker_forwards': self.forwards, 'FIX_calls': self.fixes,
                  'raw_slot_tensor_policy': 'hash_and_native_decisions; complete float32le bytes on software failure'}
        if failed and self.slot_bytes is not None:
            result['software_failure_full_slot_logits'] = {'encoding': 'base64_float32le',
                'shape': self.forwards[-1].get('slot_logit_shape'), 'bytes': len(self.slot_bytes),
                'sha256': hashlib.sha256(self.slot_bytes).hexdigest(),
                'base64': base64.b64encode(self.slot_bytes).decode('ascii')}
        return native_json(result)

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        for obj, name, value in reversed(self.restores):
            setattr(obj, name, value)
        self.handles.clear(); self.restores.clear()
        return False


def successful_alignment(observer, items, target_text):
    need(len(observer.encoded) == len(observer.forwards) == len(observer.fixes) == 1,
         'single-row no-head stage observation count differs')
    forward, fix = observer.forwards[0], observer.fixes[0]
    need(forward['status'] == 'completed' and fix['capture_status'] == 'FIX_completed', 'native stage incomplete')
    words = observer.encoded[0]['words']
    need([x['text'] for x in items] == words, 'final native text/order differs from encoded target occurrence')
    raw, before, fixed = forward['raw_timestamps_ms'], fix['native_timestamps_ms'], fix['fixed_timestamps_ms']
    need(raw == before and len(raw) == len(fixed) == 2 * len(items), 'raw-slot/FIX/item vector binding differs')
    counts = {}; annotated = []
    for i, item in enumerate(items):
        occurrence = counts.get(item['text'], 0); counts[item['text']] = occurrence + 1
        annotated.append(dict(item, target_word_index=i, target_occurrence_index=occurrence))
    return {'status': 'completed_native_candidates', 'method': 'ultra_no_head',
            'items': annotated, 'author_final_items': items, 'native_timestamps_ms': raw, 'fixed_timestamps_ms': fixed,
            'repair_changed': raw != fixed, 'target_text': target_text,
            'encoded_target_words': words, 'time_resolution_ms': observer.segment_ms,
            'native_observation': observer.observation(), 'head_used': False,
            'extra_clipping_or_interpolation': False}
