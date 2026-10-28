"""连续上下文保真及拒绝反例；仅合成PCM、CPU、无网络。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_boundary_context_review as C
P=C.P


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        audio=self.root/'audio/q01';audio.mkdir(parents=True)
        self.pcm=np.linspace(-.7,.6,16000,dtype=np.float32);sf.write(audio/'source.wav',self.pcm,16000,subtype='FLOAT')
        self.file_sha=P.sha((audio/'source.wav').read_bytes())
        rec=dict(source=dict(source_id='synthetic'),source_identity=dict(file_sha256=self.file_sha,frames=16000,channels=1,
            sample_rate=16000,pcm_sha256=P.sha(self.pcm.astype('<f4').tobytes()),pcm_hash_policy='decoded_float32le_native_interleaved/v1'),windows=[],units=[],vad={})
        side=self.root/'sidecar.json';side.write_bytes(P.json_bytes(rec))
        self.selection=self.root/'selection.json';self.selection.write_bytes(P.json_bytes([dict(review_id='q01',local_sidecar=str(side),
            row=dict(source_id='synthetic',audio_path='source.wav',audio_recorded_file_sha256=self.file_sha,sidecar_sha256=P.sha(side.read_bytes())))]))
        self.requests=self.root/'requests.json';self.jobs=[dict(case_id='q01-context',review_id='q01',start_sample=123,end_sample=8123)]
    def run_prepare(self,jobs=None,**kwargs):
        self.requests.write_bytes(P.json_bytes(jobs or self.jobs))
        def figure(*args):Path(args[-1]).write_bytes(b'synthetic-plot-no-truth')
        with patch.object(P,'make_figure',side_effect=figure):
            return C.prepare(self.selection,P.sha(self.selection.read_bytes()),self.root/'audio',self.requests,
                P.sha(self.requests.read_bytes()),self.root/'output',**kwargs)
    def test_float_pcm_exact_continuous_gain_and_blind_reference(self):
        result=self.run_prepare();manifest=json.loads((self.root/'output/manifest.json').read_text());case=manifest['cases'][0]
        actual,sr=sf.read(case['wavpath'],dtype='float32');self.assertTrue(np.array_equal(actual,self.pcm[123:8123]));self.assertEqual(sr,16000)
        self.assertEqual(case['reference_text'],'');self.assertFalse(case['usable_for_training']);self.assertFalse(case['complete_word_boundary_verified'])
        self.assertFalse(case['gain_applied']);self.assertFalse(case['resampled']);self.assertEqual(result['API_calls'],0)
    def test_invalid_range_or_bool_refused_before_output(self):
        for a,b in [(-1,20),(4,4),(0,16001),(True,30)]:
            with self.assertRaises(ValueError):self.run_prepare([dict(self.jobs[0],start_sample=a,end_sample=b)])
            self.assertFalse((self.root/'output').exists())
    def test_duplicate_or_unselected_request_refused(self):
        with self.assertRaises(ValueError):self.run_prepare(self.jobs*2)
        with self.assertRaises(ValueError):self.run_prepare([dict(self.jobs[0],review_id='q02')])
    def test_changed_audio_refused(self):
        (self.root/'audio/q01/source.wav').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):self.run_prepare()
    def test_plot_only_does_not_create_audio_or_api_case(self):
        self.run_prepare(plots_only=True);manifest=json.loads((self.root/'output/manifest.json').read_text())
        self.assertEqual(manifest['cases'],[]);self.assertEqual(list((self.root/'output').glob('*.wav')),[])


if __name__=='__main__':unittest.main()
