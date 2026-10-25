"""用实际Torch检查真实长度输入及保守取帧边界，不加载官方权重或GPU。"""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redimnet2.layers.masking import make_sample_mask, frame_mask_from_sample_mask, frame_count
from redimnet2.redimnet2 import ReDimNet2Wrap


class SampleLengthContractTests(unittest.TestCase):
    def test_invalid_lengths_are_not_silently_truncated_or_clamped(self):
        cases = [[], [0], [-1], [9], [2.7], [float('nan')], [float('inf')],
                 [True], [True, 3], torch.tensor([True]), [[2]], 2,
                 torch.tensor([1+2j])]
        for values in cases:
            with self.subTest(values=repr(values)), self.assertRaises(ValueError):
                make_sample_mask(values, 8)

    def test_valid_integer_counts_preserve_exact_prefix_mask(self):
        for counts in ([1, 8], torch.tensor([1, 8]), [1.0, 8.0]):
            actual = make_sample_mask(counts, 8)
            expected = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0.]], [[1.]*8]])
            self.assertTrue(torch.equal(actual, expected))

    def test_mixed_boolean_scalar_types_are_rejected_before_promotion(self):
        for value in (np.bool_(True), torch.tensor(True)):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                make_sample_mask([value, 3], 8)

    def test_unsigned_integer_counts_remain_supported_without_overflow(self):
        self.assertTrue(torch.equal(make_sample_mask(np.array([1, 8], dtype=np.uint64), 8),
                                    make_sample_mask([1, 8], 8)))
        with self.assertRaises(ValueError):
            make_sample_mask(np.array([2**64-1], dtype=np.uint64), 8)

    def test_invalid_input_extent_is_rejected(self):
        for maximum in (0, -2, True, 8.1):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                make_sample_mask([1], maximum)

    def test_batch_size_is_checked_before_feature_extraction(self):
        model = ReDimNet2Wrap(F=8, C=4, causal='none',
                             stages_setup=[((1, 1), 1, 1, [(3, 3)], 4)],
                             feat_type='tf', embed_dim=8).eval()
        with self.assertRaisesRegex(ValueError, '批大小'):
            model(torch.zeros(2, 16000), lengths=[16000])

    def test_conservative_frame_edge_is_not_claimed_equal_to_single_signal(self):
        # 6480采样点的单条前端包含一个右边界帧，批内严格mask将该帧排除。
        length = 6480
        sample_mask = make_sample_mask([length], 12960)
        strict = frame_mask_from_sample_mask(sample_mask, 400, 160, pad=80)
        single = frame_count(torch.tensor([length]), 400, 160, pad=80).item()
        self.assertEqual(single, 40)
        self.assertEqual(strict.sum().item(), 39)


if __name__ == '__main__':
    unittest.main()
