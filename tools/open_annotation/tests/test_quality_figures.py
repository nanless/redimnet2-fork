"""CPU figure/safety regression; synthetic arrays are never real-audio evidence."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
try:
 import numpy as np
except ImportError:
 raise unittest.SkipTest('numpy is required for optional CPU figure tests')
SPEC=importlib.util.spec_from_file_location('figures',Path(__file__).resolve().parents[1]/'quality_figures.py')
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)

class FigureTests(unittest.TestCase):
    def test_minmax_overview_keeps_single_sample_transient(self):
        x=np.zeros(16000,dtype=np.float32);x[7234]=.9375
        t,lo,hi=f.envelope(x,16000,offset=2,max_columns=100)
        self.assertEqual(max(hi),.9375);self.assertEqual(min(lo),0)
        self.assertTrue(np.all((t>=2)&(t<=3)))

    def test_native_spectrogram_frequency_and_absolute_gain(self):
        sr=16000;t=np.arange(sr)/sr;x=.5*np.sin(2*np.pi*1000*t)
        tm,hz,db=f.spectrogram(x,sr,offset=4,max_columns=20)
        peak=hz[np.argmax(db.mean(axis=1))]
        self.assertLess(abs(peak-1000),40)
        self.assertLess(abs(float(db.max())-20*np.log10(.5)),.2)
        self.assertLessEqual(len(tm),20);self.assertGreaterEqual(tm.min(),4)

    def test_short_source_analysis_padding_never_mutates_input(self):
        x=np.array([.25,-.25],dtype=np.float32);before=x.tobytes()
        result=f.spectrogram(x,16000)
        self.assertEqual(x.tobytes(),before);self.assertEqual(len(result[0]),1)

    def test_raw_FIX_and_owned_intervals_remain_distinct_source_clock(self):
        r={'source_identity':{'sample_rate':8000},'units':[{'window_id':'w','start_sample':24000,'end_sample':32000,'native_index':0,'text':'a','occurrence_index':0}]}
        w={'window_id':'w','context_start_sample':16000,'alignment':{'items':[{'text':'a'}],
           'native_timestamps_ms':[1000,500],'fixed_timestamps_ms':[1000,2000]}}
        rows=f.interval_rows(r,w)
        self.assertEqual([x['layer'] for x in rows],['raw_argmax','FIX_final','owned_unit'])
        self.assertEqual((rows[0]['source_start'],rows[0]['source_end']),(3,2.5))
        self.assertFalse(rows[0]['positive']);self.assertTrue(rows[1]['positive'])

    def test_path_alias_exact_and_ambiguous_refused(self):
        self.assertEqual(str(f.map_audio('/old/a.wav',[('/old','/new')])),'/new/a.wav')
        self.assertEqual(str(f.map_audio('/older/a.wav',[('/old','/new')])),'/older/a.wav')
        with self.assertRaises(ValueError):f.map_audio('/old/nested/a.wav',[('/old','/new'),('/old/nested','/x')])

    def test_sidecar_path_traversal_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'root';root.mkdir();outside=Path(d)/'outside';outside.write_text('x');(root/'link').symlink_to(outside)
            for p in ['../outside','/outside','a/../outside','link']:
                with self.assertRaises(ValueError):f.contained(root,p)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib is absent; no renderer success claimed')
    def test_synthetic_render_smoke(self):
        sr=16000;n=sr*2;x=np.zeros((n,2),dtype=np.float32);x[4000:8000,0]=.1*np.sin(np.arange(4000)*.3)
        r={'source':{'source_id':'synthetic_CPU_fixture_not_training'},'windows':[],'units':[], 'source_identity':{'sample_rate':sr}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'figure.png';f.plot_panels(p,x,sr,r)
            self.assertGreater(p.stat().st_size,1000)
            self.assertTrue(p.read_bytes().startswith(b'\x89PNG'))

if __name__=='__main__':unittest.main()
