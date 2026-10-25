"""Canonical registered split semantics; no models/GPU/API."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_axes import heldout_source, quality_axes
import schema


class HeldoutSourceTests(unittest.TestCase):
    def source(self, splits=(), roles=(), forbidden=False):
        return {'original_splits': list(splits), 'roles': list(roles),
                'protection': {'training_forbidden': forbidden}}

    def test_provider_evaluation_split_and_protocol_role_independently_match(self):
        self.assertTrue(heldout_source(self.source(['provider_evaluation'])))
        for role in ['evaluation:VOICES', 'evaluation:Vox1-O', 'evaluation:CN-Celeb']:
            self.assertTrue(heldout_source(self.source(roles=[role])))

    def test_supplemental_dev_test_and_diagnostic_roles_match(self):
        for tag in ['dev', 'test', 'validation', 'val', 'eval', 'calibration', 'protected_regression']:
            self.assertTrue(heldout_source(self.source([tag])))
            self.assertTrue(heldout_source(self.source(roles=['supplemental:' + tag])))
        self.assertTrue(heldout_source(self.source(roles=['regression:old18'])))

    def test_augmentation_and_train_are_not_heldout_even_if_protected(self):
        for role in ['augmentation:music', 'augmentation:noise', 'augmentation:speech', 'train_ptn', 'supplemental:train']:
            self.assertFalse(heldout_source(self.source(['provider_train'], [role], True)))
        self.assertFalse(heldout_source(self.source(['unspecified'], [], True)))

    def test_no_substring_or_empty_protocol_guess(self):
        for role in ['evaluation:', 'evaluation_backup', 'contest', 'developer', 'supplemental:provider_train']:
            self.assertFalse(heldout_source(self.source(roles=[role])))
        self.assertFalse(heldout_source(self.source(['provider_evaluation_backup'])))



if __name__ == '__main__': unittest.main()
