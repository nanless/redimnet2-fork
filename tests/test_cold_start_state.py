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



class ResumeSelectionAndClockTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        from wespeaker_lite.utils.checkpoint import select_resume_checkpoint, resume_start_epoch
        self.select = select_resume_checkpoint
        self.start = resume_start_epoch

    def checkpoint(self, name):
        path = self.root/name
        path.write_bytes(b'synthetic checkpoint-selection fixture')
        return path

    def test_latest_epoch_wins_over_newer_copy_mtime(self):
        import os
        high = self.checkpoint('model_42.pt')
        low = self.checkpoint('model_35.pt')
        os.utime(high, (1000,1000)); os.utime(low, (2000,2000))
        self.assertEqual(self.select(self.root), str(high))

    def test_pin_preserves_explicit_older_choice(self):
        self.checkpoint('model_42.pt')
        pinned = self.checkpoint('model_35.pt')
        self.assertEqual(self.select(self.root, pinned), str(pinned))

    def test_missing_pin_refused_and_empty_directory_cold(self):
        self.assertIsNone(self.select(self.root))
        with self.assertRaises(ValueError): self.select(self.root, self.root/'missing.pt')

    def test_temporary_best_and_unrelated_names_are_not_latest(self):
        for name in ['model_99.pt.tmp.123', 'best.pt', 'prefix_model_99.pt']:
            self.checkpoint(name)
        expected = self.checkpoint('model_2.pt')
        self.assertEqual(self.select(self.root), str(expected))

    def test_duplicate_numeric_epoch_names_refused(self):
        self.checkpoint('model_2.pt'); self.checkpoint('model_02.pt')
        with self.assertRaises(ValueError): self.select(self.root)

    def test_same_stage_epoch_payload_must_match_name(self):
        self.assertEqual(self.start('model_42.pt', {'epoch':42}),43)
        self.assertEqual(self.start('model_42.pt', {}),43)  # Legacy bare weight.
        with self.assertRaises(ValueError): self.start('model_42.pt', {'epoch':35})

    def test_invalid_payload_epoch_is_not_coerced(self):
        for epoch in [True,0,-1,42.0,'42']:
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                self.start('model_42.pt', {'epoch':epoch})

    def test_renamed_training_checkpoint_uses_saved_clock(self):
        self.assertEqual(self.start('pinned.pt', {'epoch':42}),43)
        self.assertEqual(self.start('released.pt', {}),1)

    def test_cross_stage_clock_and_training_state_both_cold(self):
        from wespeaker_lite.bin.train import _projection_shape_changed
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.projection=torch.nn.Linear(3,2,bias=False)
        model = Model(); state = {'projection.weight':torch.zeros(6,3)}
        self.assertTrue(_projection_shape_changed(state,model))
        self.assertEqual(_select_resume_training_state({'optimizer':{'state':'old'},'scaler':{'old':1}},
            'model_115.pt',state_dict=state,model=model),(None,None))
        self.assertEqual(self.start('model_115.pt',{'epoch':115},cross_stage=True),1)
        self.assertEqual(self.start(None,{'epoch':115}),1)
        with self.assertRaises(ValueError): self.start('model_115.pt',{'epoch':114},cross_stage=True)


class PoolingVarianceVariantTest(unittest.TestCase):
    def test_constructor_and_actual_forward_backward(self):
        from redimnet2.layers.poolings import TSTP_var
        x = torch.tensor([[[1.,2.,3.],[2.,4.,6.]]], requires_grad=True)
        pool = TSTP_var(in_dim=2)
        result = pool(x)
        self.assertEqual(pool.get_out_dim(),4)
        self.assertTrue(torch.equal(result,torch.cat((x.mean(-1),x.var(-1)),dim=1)))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class SameStageIdentityTest(unittest.TestCase):
    def setUp(self):
        import json
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); (self.root/'models').mkdir()
        self.checkpoint = self.root/'models/model_42.pt'
        self.map_path = self.root/'spk2id.json'
        self.map_path.write_text(json.dumps({'a':0,'b':1}))
        self.model = torch.nn.Linear(2,2)
        from wespeaker_lite.utils.checkpoint import validate_same_stage_resume
        self.validate = validate_same_stage_resume

    def test_exact_state_and_mapping_admitted_without_writing(self):
        before = self.map_path.read_bytes()
        self.assertEqual(self.validate(self.model.state_dict(),self.model,self.checkpoint,{'a':0,'b':1}),self.map_path)
        self.assertEqual(self.map_path.read_bytes(),before)

    def test_same_class_count_different_speakers_rejected_without_erasure(self):
        before = self.map_path.read_bytes()
        with self.assertRaisesRegex(ValueError,'speaker mapping differs'):
            self.validate(self.model.state_dict(),self.model,self.checkpoint,{'a':0,'c':1})
        self.assertEqual(self.map_path.read_bytes(),before)

    def test_missing_or_ambiguous_mapping_is_not_guessed(self):
        for raw in ['{"a":0,"b":true}', '{"a":0,"b":1,"b":1}']:
            self.map_path.write_text(raw)
            with self.assertRaises(ValueError):
                self.validate(self.model.state_dict(),self.model,self.checkpoint,{'a':0,'b':1})
        self.map_path.unlink()
        with self.assertRaises(ValueError):
            self.validate(self.model.state_dict(),self.model,self.checkpoint,{'a':0,'b':1})

    def test_explicit_mapping_path_and_full_model_keys(self):
        alternate = self.root/'external-map.json'; alternate.write_bytes(self.map_path.read_bytes())
        self.map_path.unlink()
        self.assertEqual(self.validate(self.model.state_dict(),self.model,self.checkpoint,{'a':0,'b':1},alternate),alternate)
        damaged = dict(self.model.state_dict()); damaged.pop('bias')
        with self.assertRaisesRegex(ValueError,'keys or shapes'):
            self.validate(damaged,self.model,self.checkpoint,{'a':0,'b':1},alternate)

if __name__ == '__main__':
    unittest.main()
