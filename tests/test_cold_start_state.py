"""CPU regression checks for PTN -> LM cold starts and same-stage resumes."""
import copy
import logging
import tempfile
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'applications/speaker_rec'))
from wespeaker_lite.bin.train import _select_resume_training_state, _make_train_logger


class ColdStartStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.source = torch.nn.Linear(2, 1)
        self.source_optimizer = torch.optim.SGD(self.source.parameters(), lr=0.1, momentum=0.9)
        self.source(torch.ones(1, 2)).sum().backward()
        self.source_optimizer.step()
        self.extra = {
            'optimizer': self.source_optimizer.state_dict(),
            'scaler': {'scale': 1024.0, 'growth_factor': 2.0, 'backoff_factor': 0.5,
                       'growth_interval': 2000, '_growth_tracker': 11},
        }
        self.assertTrue(self.extra['optimizer']['state'])

    def test_model_ckpt_cold_start_keeps_training_state_fresh(self):
        before = copy.deepcopy(self.extra)
        optimizer_state, scaler_state = _select_resume_training_state(self.extra, None)
        self.assertIsNone(optimizer_state)
        self.assertIsNone(scaler_state)
        target = copy.deepcopy(self.source)
        target_optimizer = torch.optim.SGD(target.parameters(), lr=0.01, momentum=0.9)
        if optimizer_state is not None:
            target_optimizer.load_state_dict(optimizer_state)
        self.assertEqual(target_optimizer.state_dict()['state'], {})
        self.assertEqual(target_optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(self.extra['scaler'], before['scaler'])
        self.assertTrue(self.extra['optimizer']['state'])

    def test_same_stage_checkpoint_returns_actual_training_states(self):
        optimizer_state, scaler_state = _select_resume_training_state(self.extra, 'lm/models/model_1.pt')
        self.assertIs(optimizer_state, self.extra['optimizer'])
        self.assertIs(scaler_state, self.extra['scaler'])
        target = copy.deepcopy(self.source)
        target_optimizer = torch.optim.SGD(target.parameters(), lr=0.01, momentum=0.9)
        target_optimizer.load_state_dict(optimizer_state)
        self.assertEqual(len(target_optimizer.state), len(self.source_optimizer.state))
        for actual, expected in zip(target_optimizer.state.values(), self.source_optimizer.state.values()):
            self.assertTrue(torch.equal(actual['momentum_buffer'], expected['momentum_buffer']))
        self.assertEqual(scaler_state['scale'], 1024.0)
        self.assertEqual(scaler_state['_growth_tracker'], 11)

    def test_rank_zero_info_progress_is_persisted(self):
        root = logging.getLogger()
        previous_root_level = root.level
        named = logging.getLogger('Pyobj, f')
        previous_level = named.level
        previous_handlers = set(named.handlers)
        try:
            root.setLevel(logging.WARNING)
            with tempfile.TemporaryDirectory() as directory:
                logger = _make_train_logger(directory, 0)
                logger.info('start_epoch: 2; final_batch: 32')
                for handler in logger.handlers:
                    handler.flush()
                self.assertIn('start_epoch: 2; final_batch: 32',
                              (Path(directory) / 'train.log').read_text())
        finally:
            for handler in set(named.handlers) - previous_handlers:
                named.removeHandler(handler)
                handler.close()
            named.setLevel(previous_level)
            root.setLevel(previous_root_level)

    def test_legacy_bare_checkpoint_has_no_training_states(self):
        self.assertEqual(_select_resume_training_state({}, 'lm/models/model_1.pt'), (None, None))


if __name__ == '__main__':
    unittest.main()
