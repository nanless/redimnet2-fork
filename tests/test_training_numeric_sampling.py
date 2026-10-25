"""Portable sampler/config regressions; no real models, media, network or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from duration_ft.data import global_batch_indices, length_for_step
from duration_ft.config_checks import validate_numeric_hyperparameters


class PureSamplingTests(unittest.TestCase):
    def test_no_duplicate_global_batch_across_epoch_boundaries(self):
        for n in [5,7,9,17,101]:
            for step in range(100):
                r=global_batch_indices(n,step,2,2,31)
                self.assertEqual(len(r),4);self.assertEqual(len(set(r)),4)
    def test_full_epochs_no_dropped_tail_and_saved_position_replay(self):
        n=7;stream=sum([global_batch_indices(n,s,2,2,31) for s in range(21)],[])
        for i in range(0,len(stream),n):self.assertEqual(set(stream[i:i+n]),set(range(n)))
        self.assertEqual(stream[20:24],global_batch_indices(n,5,2,2,31))
    def test_too_small_or_bool_sampler_input_refused(self):
        for args in [(3,0,2,2,31),(5,True,2,2,31),(5,0,True,2,31),(5,0,2,2,True)]:
            with self.subTest(args=args),self.assertRaises(ValueError):global_batch_indices(*args)
    def test_bool_schedule_input_refused(self):
        for args in [(True,3,[1.,2.],[.5,.5]),(0,True,[1.,2.],[.5,.5]),(0,3,[True,2.],[.5,.5]),(0,3,[1.,2.],[True,.5])]:
            with self.subTest(args=args),self.assertRaises(ValueError):length_for_step(*args)


class NumericTests(unittest.TestCase):
    def valid(self):return {'seed':1,'head_seed':2,'amp':False,'optimizer':{'lr':.01,'momentum':.9,'nesterov':True,'weight_decay':.001},'head':{'scale':32.,'margin':.3,'lanbuda':.7,'t':3.,'dynamic_margin':False}}
    def test_valid(self):validate_numeric_hyperparameters(self.valid())
    def test_nan_bool_or_invalid_nesterov_rejected(self):
        for field,value in [('lr',float('nan')),('lr',True),('momentum',float('inf')),('nesterov',1),('dampening',.1)]:
            c=self.valid();c['optimizer'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate_numeric_hyperparameters(c)
        for field,value in [('scale',float('nan')),('margin',False),('t',0),('dynamic_margin',1)]:
            c=self.valid();c['head'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate_numeric_hyperparameters(c)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'existing CPU PyTorch required')
class RealEntryConfigTests(unittest.TestCase):
    def test_all_entries_call_guards_and_reject_nan_before_runtime(self):
        from duration_ft import train as d
        from word_ft import train as w, gpu_train as g
        from word_ft_v3 import train as v
        repo=Path(__file__).resolve().parents[1]
        for module,relative in [(d,'duration_ft/m1_engineering.json'),(w,'word_ft/cpu_candidate_engineering.json'),(g,'word_ft/m1_fixed6_gpu_float32.json'),(v,'word_ft_v3/m1_fixed6_w2_silver_engineering.json')]:
            c=json.loads((repo/'configs'/relative).read_text());module.validate_config(c)
            c['optimizer']['lr']=float('nan')
            with self.subTest(module=module.__name__),self.assertRaises(ValueError):module.validate_config(c)
    def test_duration_bool_count_and_unreviewed_official_refused(self):
        from duration_ft import train as d
        repo=Path(__file__).resolve().parents[1];base=json.loads((repo/'configs/duration_ft/m1_engineering.json').read_text())
        for k,v in [('batch_size',True),('total_steps',True),('official_sha256','0'*64)]:
            c=copy.deepcopy(base);c[k]=v
            with self.subTest(key=k),self.assertRaises(ValueError):d.validate_config(c)
    def test_old_word_sampler_also_unique_at_boundary(self):
        from word_ft import train,gpu_train
        for module in [train,gpu_train]:
            for step in range(40):
                with self.subTest(module=module.__name__,step=step):self.assertEqual(len(set(module.step_indices(5,step,2,2,31))),4)


class MetricsIdentityTests(unittest.TestCase):
    def test_duration_metric_code_is_part_of_source_identity(self):
        from unittest.mock import patch
        from duration_ft import evaluate as e
        repo=Path(__file__).resolve().parents[1]
        baseline=e.source_identity(repo)
        self.assertIn('duration_ft/metrics.py',baseline['files'])
        original=e.sha_file
        with patch.object(e,'sha_file',side_effect=lambda p:'0'*64 if Path(p).name=='metrics.py' else original(p)):
            self.assertNotEqual(baseline,e.source_identity(repo))
    def test_word_metric_code_is_part_of_result_and_threshold_identity(self):
        from unittest.mock import patch
        from word_ft import evaluate as e
        baseline=e.code_identity();self.assertIn('metrics.py',baseline)
        original=e.sha_file
        with patch.object(e,'sha_file',side_effect=lambda p:'0'*64 if Path(p).name=='metrics.py' else original(p)):
            self.assertNotEqual(baseline,e.code_identity())

if __name__=='__main__':unittest.main()
