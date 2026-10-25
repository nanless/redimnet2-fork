"""Adversarial observation/quality tests; stdlib, no torch/model/GPU/API."""
import base64
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_observer import AlignmentObserver, positive_integer_attribute, successful_alignment
from contracts import Refused
from quality_axes import native_observation_errors
import schema
from test_quality_schema import fixture


def observation_fixture(repaired=False):
    raw = [80, 160, 240, 320]
    fixed = [80, 160, 240, 400] if repaired else raw[:]
    data = {'schema': 'ultra-no-head-native-observation/v1', 'head_used': False,
            'timestamp_attributes': {'timestamp_token_id': 151705, 'timestamp_segment_ms': 80,
                                     'raw_segment_ms_type': 'float', 'raw_token_id_type': 'int'},
            'target_encoding': [{'words': ['哈', '哈'], 'prompt': 'synthetic'}],
            'thinker_forwards': [{'status': 'completed', 'input_ids_shape': [1, 9],
                'timestamp_slot_positions': [1, 3, 5, 7], 'slot_logit_shape': [4, 20],
                'slot_logit_dtype': 'torch.float32', 'slot_logits_sha256': '8' * 64,
                'all_slot_logits_finite': True, 'raw_argmax_class_indices': [1, 2, 3, 4],
                'raw_timestamps_ms': raw, 'inference_mode': True, 'grad_enabled': False,
                'module_training': False, 'input_tensors': {}}],
            'FIX_calls': [{'native_timestamps_ms': raw, 'fixed_timestamps_ms': fixed,
                           'repair_changed': repaired, 'capture_status': 'FIX_completed'}]}
    obj = SimpleNamespace(encoded=data['target_encoding'], forwards=data['thinker_forwards'],
                          fixes=data['FIX_calls'], segment_ms=80, observation=lambda: deepcopy(data))
    items = [{'text': '哈', 'start_time': fixed[i] / 1000, 'end_time': fixed[i + 1] / 1000} for i in (0, 2)]
    return successful_alignment(obj, items, '哈哈')


class UltraObservationTests(unittest.TestCase):
    def test_actual_float80_attribute_is_accepted_without_default_guess(self):
        self.assertEqual(positive_integer_attribute(80.0, 'segment'), 80)
        for value in [True, '80', 0, -1, math.inf, math.nan, 80.5, None]:
            with self.assertRaises(Refused): positive_integer_attribute(value, 'segment')

    def test_repeated_targets_keep_separate_native_occurrences(self):
        a = observation_fixture()
        self.assertEqual([r['target_occurrence_index'] for r in a['items']], [0, 1])
        self.assertEqual([r['target_word_index'] for r in a['items']], [0, 1])
        self.assertEqual(native_observation_errors(a), [])
        a['items'][1]['target_occurrence_index'] = 0
        self.assertIn('ultra_encoded_repeated_occurrence_metadata_differs', native_observation_errors(a))

    def test_raw_classes_FIX_and_author_return_are_independently_bound(self):
        a = observation_fixture(); a['native_timestamps_ms'][0] = 0
        self.assertIn('ultra_raw_classes_not_bound_to_milliseconds', native_observation_errors(a))
        a = observation_fixture(); a['items'][0]['start_time'] = .07
        self.assertIn('ultra_items_not_bound_to_unmodified_author_return', native_observation_errors(a))
        a = observation_fixture(); a['native_observation']['FIX_calls'][0]['fixed_timestamps_ms'] = []
        self.assertIn('ultra_final_FIX_vector_missing_or_not_bound', native_observation_errors(a))

    def test_actual_training_grad_and_slot_width_cannot_be_forged_clean(self):
        a = observation_fixture(); f = a['native_observation']['thinker_forwards'][0]
        f.update(module_training=True, input_ids_shape=[2, 9], timestamp_slot_positions=[1, 1, 3, 5])
        errors = native_observation_errors(a)
        self.assertIn('ultra_actual_thinker_not_eval_inference', errors)
        self.assertIn('ultra_actual_thinker_row_count_differs', errors)
        self.assertIn('ultra_timestamp_positions_not_unique_ordered', errors)

    def test_FIX_failure_preserves_raw_and_restores_observer_hooks(self):
        class Processor:
            def fix_timestamp(self, data): raise ValueError('native FIX exploded')
            def encode_timestamp(self, text): return ['哈'], 'prompt'
        class Thinker:
            def register_forward_hook(self, fn, with_kwargs=False):
                self.callback = fn
                return SimpleNamespace(remove=lambda: setattr(self, 'removed', True))
        p = Processor(); before_fix, before_encode = p.fix_timestamp, p.encode_timestamp
        thinker = Thinker(); model = SimpleNamespace(aligner_processor=p, timestamp_token_id=151705,
            timestamp_segment_time=80.0, model=SimpleNamespace(thinker=thinker))
        obs = AlignmentObserver(model, None)
        with self.assertRaisesRegex(ValueError, 'exploded'):
            with obs:
                p.encode_timestamp('哈')
                p.fix_timestamp([80.0, 160.0])
        self.assertEqual(obs.fixes[0]['native_timestamps_ms'], [80.0, 160.0])
        self.assertEqual(obs.fixes[0]['capture_status'], 'FIX_failed')
        self.assertIn('native FIX exploded', obs.fixes[0]['traceback'])
        self.assertEqual((p.fix_timestamp, p.encode_timestamp), (before_fix, before_encode))
        self.assertTrue(thinker.removed)

    def test_software_failure_retains_full_observed_logits_bytes(self):
        obs = AlignmentObserver(SimpleNamespace(timestamp_token_id=151705, timestamp_segment_time=80.0), None)
        obs.slot_bytes = b'\x00\x00\xc0\x7f\x00\x00\x80\x3f'
        obs.forwards = [{'slot_logit_shape': [1, 2]}]
        data = obs.observation(failed=True)['software_failure_full_slot_logits']
        self.assertEqual(base64.b64decode(data['base64']), obs.slot_bytes)
        self.assertEqual(data['sha256'], hashlib.sha256(obs.slot_bytes).hexdigest())
        self.assertNotIn('software_failure_full_slot_logits', obs.observation(failed=False))

    def ultra_record(self, repaired=False):
        args = list(fixture()); args[2]['provenance']['alignment_method'] = 'ultra_no_head'
        args[4][0]['alignment'] = observation_fixture(repaired)
        from audio_windows import units_from_windows
        args[5], flags = units_from_windows(args[4], args[1]); args[-1].extend(flags)
        return schema.build_record(*args)

    def test_geometry_candidate_keeps_text_and_complete_word_axes_pending(self):
        r = self.ultra_record()
        self.assertEqual(schema.validate_record(r), [])
        self.assertEqual(r['decision']['state'], 'candidate')
        self.assertEqual(r['quality_axes']['text_faithfulness']['state'], 'machine_transcript_pending_independent_review')
        self.assertEqual(r['quality_axes']['complete_word_boundary']['state'], 'machine_boundary_pending_independent_review')
        self.assertFalse(r['decision']['usable_for_training'])
        r['quality_axes']['complete_word_boundary']['state'] = 'human_passed'
        self.assertIn('quality_axes:not_reconstructible_or_missing', schema.validate_record(r))

    def test_native_repair_stays_unknown_even_when_fixed_geometry_is_legal(self):
        r = self.ultra_record(True)
        self.assertEqual(schema.validate_record(r), [])
        self.assertEqual(r['decision']['state'], 'unknown')
        self.assertTrue(r['quality_axes']['native_geometry_and_repair'][0]['repair_changed'])

    def test_old_or_missing_actual_observation_cannot_admit_Ultra(self):
        r = self.ultra_record(); r['windows'][0]['alignment'].pop('native_observation')
        errors = schema.validate_record(r)
        self.assertTrue(any('ultra_native_observation_missing' in x for x in errors))

    def test_original_heldout_protection_is_not_erased_by_geometry(self):
        r = self.ultra_record(); args = list(fixture())
        args[0]['original_splits'] = ['dev']; args[0]['protection']['training_forbidden'] = True
        z = schema.build_record(*args)
        self.assertTrue(z['quality_axes']['source_and_split']['heldout_split_present'])
        self.assertTrue(z['quality_axes']['source_and_split']['training_forbidden'])
        self.assertFalse(z['quality_axes']['source_and_split']['automatic_training_release'])


if __name__ == '__main__': unittest.main()
