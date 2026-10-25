"""Reconstructible independent quality axes, never a human-truth certificate."""
from collections import Counter
import math
if __package__:
    from .source_clock import LEGACY_MAPPING
else:
    from source_clock import LEGACY_MAPPING


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def native_observation_errors(alignment):
    if alignment.get('status') != 'completed_native_candidates':
        return []
    errors = []
    def check(ok, reason):
        if not ok: errors.append(reason)
    obs = alignment.get('native_observation', {})
    if not isinstance(obs, dict): return ['ultra_native_observation_not_object']
    check(obs.get('schema') == 'ultra-no-head-native-observation/v1' and obs.get('head_used') is False,
          'ultra_native_observation_missing_or_wrong_mode')
    enc, fw, fixes = (obs.get(k, []) for k in ('target_encoding', 'thinker_forwards', 'FIX_calls'))
    if not all(isinstance(x, list) for x in (enc, fw, fixes)): return ['ultra_native_stage_captures_not_lists']
    check(len(enc) == len(fw) == len(fixes) == 1, 'ultra_single_window_stage_counts_differ')
    check(alignment.get('head_used') is False and alignment.get('extra_clipping_or_interpolation') is False,
          'ultra_author_final_was_not_preserved')
    if not (len(enc) == len(fw) == len(fixes) == 1):
        return errors
    if not all(isinstance(x[0], dict) for x in (enc, fw, fixes)): return ['ultra_native_stage_capture_not_object']
    words, items = enc[0].get('words'), alignment.get('items', [])
    if not isinstance(items, list) or not all(isinstance(r, dict) for r in items): return ['ultra_native_items_malformed']
    check(isinstance(words, list) and all(isinstance(x, str) for x in words), 'ultra_encoded_words_missing')
    if not isinstance(words, list): return errors
    check(words == [r.get('text') for r in items] and words == alignment.get('encoded_target_words'),
          'ultra_target_occurrence_order_not_preserved')
    attrs = obs.get('timestamp_attributes', {})
    if not isinstance(attrs, dict): return errors + ['ultra_timestamp_attributes_not_object']
    segment_ms = attrs.get('timestamp_segment_ms')
    check(type(segment_ms) is int and segment_ms > 0 and type(attrs.get('timestamp_token_id')) is int
          and attrs['timestamp_token_id'] > 0, 'ultra_actual_timestamp_attributes_invalid')
    f, fix = fw[0], fixes[0]
    classes, slots = f.get('raw_argmax_class_indices', []), f.get('timestamp_slot_positions', [])
    if not isinstance(classes, list) or not isinstance(slots, list): return errors + ['ultra_timestamp_classes_or_positions_not_lists']
    check(len(classes) == len(slots) == 2 * len(words) and all(type(x) is int and x >= 0 for x in classes),
          'ultra_timestamp_slots_not_bound_to_target_occurrences')
    check(all(type(x) is int and x >= 0 for x in slots) and sorted(set(slots)) == slots,
          'ultra_timestamp_positions_not_unique_ordered')
    raw = alignment.get('native_timestamps_ms'); fixed = alignment.get('fixed_timestamps_ms')
    check(f.get('status') == 'completed' and f.get('all_slot_logits_finite') is True and f.get('slot_logit_dtype') == 'torch.float32',
          'ultra_native_logits_not_finite_FP32_observation')
    check(f.get('module_training') is False and f.get('inference_mode') is True and f.get('grad_enabled') is False,
          'ultra_actual_thinker_not_eval_inference')
    shape = f.get('input_ids_shape')
    input_shape_valid = (isinstance(shape, list) and len(shape) == 2
                         and all(type(x) is int for x in shape) and shape[0] == 1 and shape[1] > 0)
    check(input_shape_valid,
          'ultra_actual_thinker_row_count_differs')
    if input_shape_valid and all(type(x) is int for x in slots):
        check(all(x < shape[1] for x in slots), 'ultra_timestamp_position_outside_input')
    logit_shape = f.get('slot_logit_shape')
    logit_shape_valid = (isinstance(logit_shape, list) and len(logit_shape) == 2
                         and all(type(x) is int for x in logit_shape)
                         and logit_shape[0] == len(slots) and logit_shape[1] > 0)
    check(logit_shape_valid, 'ultra_native_logit_shape_not_bound_to_slots')
    if logit_shape_valid and all(type(x) is int for x in classes):
        check(all(0 <= x < logit_shape[1] for x in classes), 'ultra_native_class_outside_logits')
    logit_sha = f.get('slot_logits_sha256')
    check(isinstance(logit_sha, str) and len(logit_sha) == 64
          and all(x in '0123456789abcdef' for x in logit_sha), 'ultra_native_logits_digest_missing_or_invalid')
    if type(segment_ms) is int and all(type(x) is int for x in classes):
        check(raw == [x * segment_ms for x in classes], 'ultra_raw_classes_not_bound_to_milliseconds')
    check(raw == f.get('raw_timestamps_ms') == fix.get('native_timestamps_ms'), 'ultra_raw_to_FIX_binding_differs')
    check(fix.get('capture_status') == 'FIX_completed' and fixed == fix.get('fixed_timestamps_ms'),
          'ultra_final_FIX_vector_missing_or_not_bound')
    check(alignment.get('repair_changed') == (raw != fixed) == fix.get('repair_changed'), 'ultra_repair_flag_inconsistent')
    check(alignment.get('author_final_items') == [{k: r.get(k) for k in ('text', 'start_time', 'end_time')} for r in items],
          'ultra_items_not_bound_to_unmodified_author_return')
    counts = {}
    for i, row in enumerate(items):
        text = row.get('text'); occurrence = counts.get(text, 0); counts[text] = occurrence + 1
        check(row.get('target_word_index') == i and row.get('target_occurrence_index') == occurrence,
              'ultra_encoded_repeated_occurrence_metadata_differs')
    return sorted(set(errors))


def heldout_source(source):
    """Interpret registered consumer semantics, independent of training protection.

    provider_train and augmentation are not heldout. Evaluation endpoints may
    carry the canonical provider_evaluation split, an evaluation:<protocol>
    role, or both. Supplemental dev/test use their exact registered split or
    supplemental role; an arbitrary substring is never a match.
    """
    splits = {str(x).casefold() for x in source.get('original_splits') or []}
    heldout = {'test', 'dev', 'validation', 'val', 'eval', 'evaluation',
               'calibration', 'protected_regression', 'provider_evaluation'}
    if splits & heldout:
        return True
    role_tags = {'test', 'dev', 'validation', 'val', 'eval', 'evaluation',
                 'calibration', 'regression', 'protected_regression'}
    for raw in source.get('roles') or []:
        role = str(raw).casefold()
        head, separator, tail = role.partition(':')
        if head in role_tags and (not separator or tail):
            return True
        if head == 'supplemental' and tail in heldout:
            return True
    return False


def quality_axes(record):
    source = record.get('source', {}); identity = record.get('source_identity', {})
    windows = record.get('windows', []); windows = windows if isinstance(windows, list) else []
    mappings = [w.get('time_mapping') for w in windows if isinstance(w, dict)]
    source_mapping = (mappings[0] if all(x == mappings[0] for x in mappings) else 'mixed_explicit_window_mappings') if mappings else LEGACY_MAPPING
    policy = record.get('provenance', {}).get('alignment_method')
    lang_rows, geometry_rows, repetition_rows, faithfulness_rows = [], [], [], []
    for w in windows:
        if not isinstance(w, dict): continue
        a, al = w.get('asr', {}), w.get('alignment', {})
        a = a if isinstance(a, dict) else {}; al = al if isinstance(al, dict) else {}
        normal = a.get('normalization_audit') or {}
        normal = normal if isinstance(normal, dict) else {}
        lang_rows.append({'window_id': w.get('window_id'), 'ASR_detected_language': a.get('language'),
                          'alignment_status': al.get('status'), 'alignment_reason': al.get('reason'),
                          'language_reference_status': 'ASR_prediction_not_verified_language_truth',
                          'spoken_form_flags': normal.get('reasons', [])})
        items = al.get('items') or []; items = items if isinstance(items, list) else []
        repeated = Counter(r.get('text') for r in items if isinstance(r, dict) and isinstance(r.get('text'), str))
        repetition_rows.append({'window_id': w.get('window_id'), 'native_item_count': len(items),
                                'repeated_surface_counts': {k: v for k, v in sorted(repeated.items()) if v > 1 and isinstance(k, str)},
                                'order_reference': 'encoded_target_words_and_native_index',
                                'occurrence_proof': 'verified_by_native_observation' if policy == 'ultra_no_head' and not native_observation_errors(al)
                                                    and al.get('status') == 'completed_native_candidates' else 'not_certified'})
        invalid = sum(not isinstance(r, dict) or not _finite(r.get('start_time')) or not _finite(r.get('end_time'))
                      or r['end_time'] <= r['start_time'] for r in items)
        raw = al.get('native_timestamps_ms') or []
        raw_bad = sum(not (_finite(x) and _finite(y)) or y <= x for x, y in zip(raw[::2], raw[1::2])) if isinstance(raw, list) else None
        geometry_rows.append({'window_id': w.get('window_id'), 'final_nonpositive_or_nonfinite_items': invalid,
                              'raw_nonpositive_or_nonfinite_pairs': raw_bad,
                              'repair_changed': al.get('repair_changed'), 'head_used': al.get('head_used'),
                              'author_final_time_reference': 'unaltered_native_return',
                              'complete_word_correctness_inferred_from_geometry': False})
        faithfulness_rows.append({'window_id': w.get('window_id'), 'eos_observed': a['native'].get('eos_observed') if isinstance(a.get('native'), dict) else None,
                                  'spoken_form_review_required': normal.get('spoken_form_review_required'),
                                  'raw_text_retained': isinstance(a.get('raw_text'), str),
                                  'faithfulness_status': 'not_independently_verified'})
    roles = source.get('roles') or []; splits = source.get('original_splits') or []
    protected = (source.get('protection') or {}).get('training_forbidden')
    return {'schema': 'redimnet2-annotation-quality-axes/v1',
            'text_faithfulness': {'state': 'machine_transcript_pending_independent_review', 'windows': faithfulness_rows},
            'complete_word_boundary': {'state': 'machine_boundary_pending_independent_review',
                                       'human_reference_available': False, 'geometric_validity_is_not_complete_word_truth': True},
            'native_geometry_and_repair': geometry_rows, 'language_and_spoken_form': lang_rows,
            'repetition_occurrences': repetition_rows,
            'source_and_split': {'registered_roles': roles, 'original_splits': splits, 'training_forbidden': protected,
                                 'source_split_fields_preserved': True,
                                 'heldout_split_present': heldout_source(source),
                                 'automatic_training_release': False, 'automatic_evaluation_release': False},
            'source_clock': {'decode_status': identity.get('decode_status'), 'sample_rate': identity.get('sample_rate'),
                             'mapping': source_mapping,
                             'source_time_repair_or_clipping_applied': False},
            'model_API_calls': 0, 'human_review': None}
