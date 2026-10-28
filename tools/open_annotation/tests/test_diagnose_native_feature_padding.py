"""只用合成PCM/数组验证特征诊断逻辑；官方安装实测另留实际证据。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diagnose_native_feature_padding as D

FUNCTION=b'''def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths
'''


class FeaturePaddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        model=self.root/'actual-model-config-only';model.mkdir();config=self.put(model/'preprocessor_config.json',dict(sampling_rate=16000,hop_length=160,n_fft=400,dither=0))
        contract=self.put(self.root/'pilot.json',dict(purpose='pilot_validation',models={'asr':{'path':str(model)}}))
        feature=self.root/'feature_extraction_whisper.py';feature.write_text('# fixed synthetic source')
        processing=self.root/'processing_qwen3_asr.py';processing.write_bytes(FUNCTION)
        self.plan=dict(schema=D.SCHEMA,contract=contract,feature_source=self.ref(feature),processing_source=self.ref(processing),config=config,output_root=str(self.root/'new-output'));self.save()
    def ref(self,p):return dict(path=str(p),sha256=D.sha(p.read_bytes()))
    def put(self,p,v):p.write_text(json.dumps(v));return self.ref(p)
    def save(self):self.pin=self.put(self.root/'plan.json',self.plan)
    def test_default_check_has_no_torch_model_or_output(self):
        with patch.dict(sys.modules,{'torch':None}):value=D.run(self.pin)
        self.assertFalse(value['executed']);self.assertFalse(Path(self.plan['output_root']).exists())
    def test_official_arithmetic_function_is_extracted_without_importing_model(self):
        fn,digest=D.output_length_function(FUNCTION)
        for frames,expected in [(100,13),(101,14),(308,40),(309,41),(2508,326),(2509,327)]:self.assertEqual(fn(frames),expected)
        self.assertEqual(len(digest),64)
    def test_source_function_with_call_or_attribute_is_refused(self):
        for raw in (b'def _get_feat_extract_output_lengths(x):\n return evil(x)\n',b'def _get_feat_extract_output_lengths(x):\n return x.shape\n'):
            with self.assertRaisesRegex(ValueError,'pure bounded'):D.output_length_function(raw)
    def test_synthetic_wave_is_deterministic_float32_not_real_audio(self):
        import numpy as np
        for n in D.LENGTHS:
            a=D.waveform(n);self.assertEqual(a.shape,(n,));self.assertEqual(a.dtype,np.dtype('float32'));self.assertTrue(np.isfinite(a).all());self.assertEqual(D.array_sha(a),D.array_sha(D.waveform(n)))
        self.assertNotEqual(D.array_sha(D.waveform(16000)),D.array_sha(D.waveform(16000,1)))
    def test_comparison_keeps_shape_change_and_common_prefix_separate(self):
        import numpy as np
        a=np.zeros((3,100),dtype=np.float32);b=np.zeros((3,101),dtype=np.float32)
        v=D.compare(a,b);self.assertFalse(v['full_valid_shape_equal']);self.assertTrue(v['common_prefix_equal']);self.assertEqual(v['frames_changed_above_tolerance'],0)
        b[0,99]=.2;v=D.compare(a,b);self.assertFalse(v['common_prefix_equal']);self.assertEqual(v['changed_frame_indices'],[99])
    def test_changed_feature_source_or_nonzero_dither_refused(self):
        p=Path(self.plan['feature_source']['path']);p.write_text('# replaced')
        with self.assertRaisesRegex(ValueError,'SHA differs'):D.validate(self.pin)
        self.plan['feature_source']=self.ref(p);config=Path(self.plan['config']['path']);self.plan['config']=self.put(config,dict(sampling_rate=16000,hop_length=160,dither=.01));self.save()
        with self.assertRaisesRegex(ValueError,'dither0'):D.validate(self.pin)
    def test_config_must_belong_to_contract_model_and_output_fresh(self):
        other=self.put(self.root/'unrelated-config.json',dict(sampling_rate=16000,hop_length=160));self.plan['config']=other;self.save()
        with self.assertRaisesRegex(ValueError,'actual ASR model'):D.validate(self.pin)
        self.plan['config']=self.ref(self.root/'actual-model-config-only/preprocessor_config.json');self.save();Path(self.plan['output_root']).mkdir()
        with self.assertRaisesRegex(ValueError,'fresh physical'):D.validate(self.pin)

if __name__=='__main__':unittest.main()
