"""合成CPU数值取证测试；不构造模型、不请求CUDA。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('numeric_cpu',HERE/'inspect_numeric_cpu.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class NumericCPUTests(unittest.TestCase):
    def fixture(self,root,payload,name='input.pt'):
        source=root/'source'/name;source.parent.mkdir(exist_ok=True);torch.save(payload,source)
        return source,hashlib.sha256(source.read_bytes()).hexdigest()

    def test_wave_real_stride_storage_and_per_row_finite_stats(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);wave=torch.arange(8,dtype=torch.float32).reshape(1,1,8).expand(2,1,8)
            source,pin=self.fixture(root,{'wav':wave,'utterance_keys':['syntheticA','syntheticB']})
            out=root/'audit/wave.json';result=M.main(['--wav',str(source),'--wav-sha',pin,'--output',str(out)])
            row=result['wave'];self.assertEqual(row['stride'],[0,8,1]);self.assertLess(row['storage_nbytes'],row['logical_nbytes'])
            self.assertEqual(row['shape'],[2,1,8]);self.assertEqual(row['rows'][0]['mean'],3.5)
            self.assertIsNone(row['declared_sample_rate']);self.assertFalse(result['GPU_used'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),pin)

    def test_nonfinite_wave_is_reported_without_modification_or_JSON_NaN(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source,pin=self.fixture(root,{'wav':torch.tensor([[[0.,1.,float('nan'),float('inf')]]]),'utterance_keys':['synthetic']})
            out=root/'audit/wave.json';result=M.main(['--wav',str(source),'--wav-sha',pin,'--sample-rate','16000','--output',str(out)])
            row=result['wave']['rows'][0];self.assertEqual((row['nan'],row['posinf'],row['finite_count']),(1,1,2))
            self.assertEqual(row['total_samples'],4);self.assertFalse(result['wave']['waveform_normalized_clipped_resampled_or_augmented'])
            self.assertNotIn('NaN',out.read_text())

    def test_model_and_optimizer_separate_magnitudes_and_negative_BN_var(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);payload={'model':{'block.1.weight':torch.tensor([2.,3.]),'block.1.running_var':torch.tensor([.2,-.1]),
                'block.1.num_batches_tracked':torch.tensor(47)},'optimizer':{'state':{0:{'momentum_buffer':torch.tensor([1.,float('inf')])}}},
                'epoch':47,'scaler':{'scale':131072.,'_growth_tracker':1840}}
            source,pin=self.fixture(root,payload);result=M.main(['--checkpoint',str(source),'--checkpoint-sha',pin,'--output',str(root/'audit/model.json')])
            report=result['checkpoint'];self.assertEqual(report['model_tensor_count'],3);self.assertEqual(report['model_nonfinite'],[])
            self.assertEqual(report['running_var'][0]['negative_finite_count'],1)
            self.assertEqual(report['optimizer_nonfinite'],['optimizer.state.0.momentum_buffer'])
            self.assertEqual(report['norm_affine'][0]['name'],'block.1.weight');self.assertFalse(report['model_constructed'])

    def test_wrong_SHA_refused_before_checkpoint_deserialization(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source,_=self.fixture(root,{'a':torch.ones(1)})
            with patch.object(M.torch,'load',side_effect=AssertionError('must not load')) as load:
                with self.assertRaisesRegex(ValueError,'SHA'):M.main(['--checkpoint',str(source),'--checkpoint-sha','0'*64,'--output',str(root/'audit/out.json')])
                load.assert_not_called()

    def test_output_cannot_replace_or_enter_input_directory(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source,pin=self.fixture(root,{'a':torch.ones(1)})
            with self.assertRaises(ValueError):M.main(['--checkpoint',str(source),'--checkpoint-sha',pin,'--output',str(source.parent/'new.json')])
            out=root/'audit/out.json';out.parent.mkdir();out.write_text('preserve')
            with self.assertRaises(FileExistsError):M.main(['--checkpoint',str(source),'--checkpoint-sha',pin,'--output',str(out)])
            self.assertEqual(out.read_text(),'preserve')

    def test_dangling_output_symlink_cannot_create_target(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source,pin=self.fixture(root,{'a':torch.ones(1)})
            target=root/'never-create.json';link=root/'output.json';link.symlink_to(target)
            with patch.object(M,'load_bound',side_effect=AssertionError('must not read input')) as load:
                with self.assertRaises(FileExistsError):M.main(['--checkpoint',str(source),'--checkpoint-sha',pin,'--output',str(link)])
                load.assert_not_called()
            self.assertFalse(target.exists());self.assertTrue(link.is_symlink())


if __name__=='__main__':unittest.main()
