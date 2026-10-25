"""Explicit alternate inputs do not confer correctness or label eligibility."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from normalization import POLICY, APOSTROPHE_PROFILE, normalize_with_audit
import schema
from test_quality_schema import fixture


class NormalizationProfileTests(unittest.TestCase):
    def test_ascii_word_apostrophes_are_preserved_only_by_explicit_profile(self):
        for text,new,old in (("I'm here.","I'm here","I m here"),
                             ("J'espère!","J'espère","J espère"),
                             ("John's book.","John's book","John s book"),
                             ("I＇m here.","I'm here","I m here")):
            with self.subTest(text=text):
                a=normalize_with_audit(text,APOSTROPHE_PROFILE)
                self.assertEqual(a['normalized_text'],new)
                self.assertEqual(normalize_with_audit(text)['normalized_text'],old)
                self.assertEqual(a['raw_text'],text)
                self.assertTrue(a['spoken_form_review_required'])
                self.assertIn('word_internal_apostrophe_requires_review',a['reasons'])

    def test_curly_quotes_numeric_readings_and_outer_quotes_are_not_guessed(self):
        a=normalize_with_audit("'Hello' 12.5 We’re ready.",APOSTROPHE_PROFILE)
        self.assertEqual(a['normalized_text'],'Hello 12 5 We re ready')
        self.assertIn('numeric_spoken_form_unresolved',a['reasons'])
        self.assertIn('decimal_or_group_separator_requires_review',a['reasons'])
        self.assertIn('word_internal_apostrophe_requires_review',a['reasons'])
        with self.assertRaises(ValueError):normalize_with_audit('hello','unsupported')

    def test_audit_profile_must_be_bound_to_producer_runtime(self):
        args=list(fixture());asr=args[4][0]['asr'];audit=normalize_with_audit("I'm",APOSTROPHE_PROFILE)
        asr.update(raw_text="I'm",normalized_text=audit['normalized_text'],normalization_audit=audit)
        args[4][0]['alignment'].update(target_text="I'm",items=[{'text':"I'm",'start_time':.1,'end_time':.4}],
                                      native_timestamps_ms=[100,400],fixed_timestamps_ms=[100,400])
        args[5]=[dict(args[5][0],text="I'm")]
        denied=schema.build_record(*args)
        self.assertIn('window:normalization_audit_not_reconstructible',schema.validate_record(denied))
        args[2]['provenance']['runtime']={'normalization_profile':APOSTROPHE_PROFILE}
        r=schema.build_record(*args);self.assertEqual(schema.validate_record(r),[])
        self.assertEqual(r['decision']['state'],'unknown')
        self.assertIn('window_0:spoken_form_review_required',r['decision']['reasons'])
        self.assertFalse(r['decision']['usable_for_training'])
        self.assertFalse(r['decision']['usable_for_evaluation'])

    def test_default_legacy_audit_remains_byte_equivalent(self):
        for raw in ('Hello world.',"I'm ready.",'１２.５ API','中英ABC'):
            self.assertEqual(normalize_with_audit(raw),normalize_with_audit(raw,POLICY))


if __name__=='__main__':unittest.main()
