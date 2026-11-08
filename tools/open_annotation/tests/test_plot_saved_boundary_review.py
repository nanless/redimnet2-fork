"""离线绘图要保留各自时钟/零长/FIX，并拒绝假原音身份。"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

P=Path(__file__).resolve().parents[1]/'plot_saved_boundary_review.py'
spec=importlib.util.spec_from_file_location('plot_saved_boundary_review',P)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class ReviewTests(unittest.TestCase):
    def fixture(self, root):
        import numpy as np
        import soundfile as sf
        sr=8000;pcm=(.2*np.sin(2*np.pi*440*np.arange(sr)/sr)).astype('float32')
        folder=root/'audio'/'q01';folder.mkdir(parents=True)
        audio=folder/'source.wav';sf.write(audio,pcm,sr,subtype='FLOAT')
        r={'source':{'source_id':'synthetic','dataset':'synthetic'},
           'source_identity':{'sample_rate':sr,'frames':sr,'channels':1,'file_sha256':M.sha(audio.read_bytes()),
                              'pcm_hash_policy':'decoded_float32le_native_interleaved/v1','pcm_sha256':M.sha(pcm.astype('<f4').tobytes())},
           'decision':{'state':'unknown','reasons':['synthetic_zero']},'errors':[],
           'vad':{'original_text':'SPEAKER synthetic 1 0.000 1.001 <NA> <NA> <NA> <NA> <NA>\n'},
           'units':[{'window_id':'w1','native_index':0,'text':'甲','start_sample':3840,'end_sample':4480}],
           'windows':[{'window_id':'w1','context_start_sample':3200,'context_end_sample':8000,
                       'owned_start_sample':0,'owned_end_sample':8000,
                       'asr':{'raw_text':'甲。','normalized_text':'甲'},
                       'alignment':{'items':[{'text':'甲'}],'native_timestamps_ms':[80,80],
                                    'fixed_timestamps_ms':[80,160]}}]}
        side=root/'sidecar.json';side.write_bytes(M.json_bytes(r))
        c={'review_id':'q01','sample_id':'synthetic','local_sidecar':str(side),
           'row':{'source_id':'synthetic','audio_path':'/never_read/source.wav',
                  'sidecar_sha256':M.sha(side.read_bytes()),'audio_recorded_file_sha256':M.sha(audio.read_bytes())}}
        selection=root/'selection.json';selection.write_bytes(M.json_bytes([c]))
        return r,c,selection

    def test_context_native_FIX_and_source_unit_independent(self):
        with tempfile.TemporaryDirectory() as t:
            r,_,_=self.fixture(Path(t));rows=M.interval_rows(r)
            native=next(x for x in rows if x['track']=='native');fixed=next(x for x in rows if x['track']=='FIX')
            self.assertEqual(native['start_seconds'],native['end_seconds'])
            self.assertAlmostEqual(native['start_seconds'],.48)
            self.assertAlmostEqual(fixed['end_seconds'],.56)
            self.assertGreater(next(x for x in rows if x['track']=='old VAD')['end_seconds'],1.)

    def test_actual_CPU_render_and_all_output_SHA(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,_,selection=self.fixture(root)
            out=root/'new';r=M.render(selection,M.sha(selection.read_bytes()),root/'audio',out)
            self.assertFalse(r['human_gold']);self.assertFalse(r['model_GPU_API_execution'])
            self.assertGreaterEqual(r['cases'][0]['figures'],4)
            self.assertIn('index.html',r['outputs_sha256'])
            self.assertTrue(r['cases'][0]['local_decode_proof']['pcm_sha256'])
            for rel,pin in r['outputs_sha256'].items():self.assertEqual(M.sha((out/rel).read_bytes()),pin)
            with self.assertRaises(ValueError):M.render(selection,M.sha(selection.read_bytes()),root/'audio',out)

    def test_audio_corruption_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,_,selection=self.fixture(root)
            (root/'audio/q01/source.wav').write_bytes(b'wrong')
            with self.assertRaises(ValueError):M.render(selection,M.sha(selection.read_bytes()),root/'audio',root/'new')
            self.assertFalse((root/'new').exists())

    def test_PCM_digest_rejected_even_if_encoded_hash_right(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);r,c,_=self.fixture(root);r['source_identity']['pcm_sha256']='0'*64
            side=Path(c['local_sidecar']);side.write_bytes(M.json_bytes(r));c['row']['sidecar_sha256']=M.sha(side.read_bytes())
            with self.assertRaisesRegex(ValueError,'PCM SHA'):M.load_case(c,root/'audio')

    def test_nonfinite_duplicate_and_unsafe_id_refused(self):
        for raw in [b'{"a":1,"a":2}',b'{"a":1e999}',b'{"a":NaN}']:
            with self.assertRaises(ValueError):M.strict_json(raw)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,c,_=self.fixture(root);c['review_id']='../escape'
            with self.assertRaises(ValueError):M.load_case(c,root/'audio')

    def test_symlink_input_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'a';p.write_bytes(b'test');q=root/'b';q.symlink_to(p)
            with self.assertRaises(ValueError):M.read_pinned(q,M.sha(b'test'),10)

if __name__=='__main__':unittest.main()
