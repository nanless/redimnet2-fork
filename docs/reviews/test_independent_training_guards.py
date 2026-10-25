"""对迁入的训练采样与LM准备做有限独立CPU反例，不执行模型或训练。"""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class SamplingReview(unittest.TestCase):
    def test_small_pools_unique_complete_and_out_of_order_replay(self):
        from duration_ft.data import global_batch_indices
        for pool in range(3, 34):
            for rank_batch, world in ((1, 3), (2, 2), (3, 2), (2, 4)):
                size = rank_batch * world
                if pool < size:
                    continue
                for seed in (0, 19):
                    expected = [global_batch_indices(pool, step, rank_batch, world, seed) for step in range(60)]
                    for row in expected:
                        self.assertEqual(len(set(row)), size)
                    stream = sum(expected, [])
                    for begin in range(0, len(stream) - pool + 1, pool):
                        self.assertEqual(set(stream[begin:begin + pool]), set(range(pool)))
                    steps = list(range(60))
                    random.Random(17).shuffle(steps)
                    for step in steps:
                        self.assertEqual(global_batch_indices(pool, step, rank_batch, world, seed), expected[step])


class LMReview(unittest.TestCase):
    def test_additional_bare_exception_and_synthetic_future(self):
        torch_was_loaded = 'torch' in sys.modules
        spec = importlib.util.spec_from_file_location('independent_lm_fixture', ROOT / 'tests/test_lm_continuation.py')
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        case = fixture.CrossSegmentCPU()
        case.setUp()
        try:
            positive = case.evaluate()
            self.assertTrue(positive['cpu_evidence_complete'])
            for text in (b'UnicodeDecodeError: broken raw log\n', b'PermissionError: missing evidence\n',
                         b'Exception: unexpected successor failure\n'):
                case.c['artifacts']['new_log'] = case.make(text, fixture.M.NEW + '/launch.log')
                result = case.evaluate()
                self.assertFalse(result['cpu_evidence_complete'])
                self.assertTrue(any(x['name'] == 'successor_no_new_failures' and not x['passed'] for x in result['checks']))
                self.assertFalse(result['ready_for_lm_execution'])
                self.assertFalse(result['locks_acquired'])
            self.assertEqual('torch' in sys.modules, torch_was_loaded)
        finally:
            case.tearDown()


if __name__ == '__main__':
    unittest.main()
