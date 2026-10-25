"""The unit contract must preserve continuous complete reference audio."""
import importlib.util as _deps
import unittest
if any(_deps.find_spec(p) is None for p in ['soundfile']):
    raise unittest.SkipTest('optional CPU test dependencies unavailable: '+', '.join(['soundfile']))
import copy,json,os,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from word_ft import schema as s

class WordSchemaTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.audio=self.root/'unit.wav'
        self.wave=(np.arange(4000,dtype=np.float32)%101-50)/100
        sf.write(self.audio,self.wave,16000,subtype='FLOAT')
    def tearDown(self):self.temp.cleanup()
    def unit(self,**kwargs):return s.make_unit(self.audio,'spk1','开灯。',**kwargs)
    def test_normalization(self):
        self.assertEqual(s.normalize_text('  开，灯！'),'开灯')
        self.assertEqual(s.tokenize_units("Don't stop.",'en'),["don't",'stop'])
        for text in ['数到20','hello你','１２','', '   ']:
            with self.assertRaises(ValueError):s.normalize_text(text)
    def test_whole_reference_cannot_be_cropped(self):
        with self.assertRaises(ValueError):self.unit(start_sample=100)
        r=self.unit();wave,info=s.read_unit(r,return_info=True)
        self.assertTrue(np.array_equal(wave,self.wave))
        self.assertEqual(len(wave),4000)
        self.assertFalse(info['waveform_padding']);self.assertFalse(info['vad_concatenation'])
    def test_vad_is_not_a_waveform_transform(self):
        r=self.unit(vad={'segments_source_samples':[[0,100],[3000,4000]],'source_sha256':'test'})
        self.assertTrue(np.array_equal(s.read_unit(r),self.wave))
    def test_audited_requires_actual_note(self):
        with self.assertRaises(ValueError):self.unit(evidence='human_audited_span',start_sample=100)
        r=self.unit(evidence='human_audited_span',start_sample=100,end_sample=500,provenance={'review_note':'synthetic contract fixture only'})
        self.assertTrue(np.array_equal(s.read_unit(r),self.wave[100:500]))
    def test_timed_reference_needs_provenance(self):
        with self.assertRaises(ValueError):self.unit(evidence='reference_timed_segment',end_sample=500)
        self.unit(evidence='reference_timed_segment',end_sample=500,provenance={'annotation_sha':'fixture'})
    def test_candidates_are_explicit(self):
        kw={'evidence':s.CANDIDATE_EVIDENCE,'transcript_origin':'asr_prediction'}
        with self.assertRaises(ValueError):self.unit(**kw,alignment={})
        r=self.unit(**kw,alignment={'sdk':'fixture','resolution_ms':80})
        with self.assertRaises(ValueError):s.validate_unit(r)
        s.validate_unit(r,allow_candidates=True)
        path=self.root/'candidate.json';s.write_dataset(path,[r])
        with self.assertRaises(ValueError):s.load_dataset(path)
        self.assertEqual(s.load_dataset(path,allow_candidates=True)['records'][0],r)
    def test_speaker_required(self):
        for sp in [None,'','   ']:
            with self.assertRaises(ValueError):s.make_unit(self.audio,sp,'开灯')
    def test_invalid_intervals(self):
        for a,b in [(0,0),(-1,300),(100,90),(0,4001),(True,4000)]:
            with self.assertRaises(ValueError):self.unit(start_sample=a,end_sample=b,evidence='reference_timed_segment',provenance={'id':'fixture'})
    def test_count_and_original_and_policy_bound(self):
        for field,value in [('n_units',3),('spoken_text','关灯'),('count_bin','7+'),('counting_policy','guess'),('text_original','关灯')]:
            r=self.unit();r[field]=value
            with self.assertRaises(ValueError):s.validate_unit(r)
    def test_dataset_identity_and_duplicate_and_new_file(self):
        r=self.unit();p=self.root/'units.json';s.write_dataset(p,[r],{'split':'test'})
        self.assertEqual(s.load_dataset(p)['records'][0],r)
        with self.assertRaises(FileExistsError):s.write_dataset(p,[r])
        with self.assertRaises(ValueError):s.seal_dataset([r,r])
        x=json.loads(p.read_text());x['metadata']['split']='train';p.write_text(json.dumps(x))
        with self.assertRaises(ValueError):s.load_dataset(p)
    def test_content_sha_catches_same_stat_mutation(self):
        r=self.unit();st=self.audio.stat();raw=bytearray(self.audio.read_bytes());raw[-8]^=1
        self.audio.write_bytes(raw);os.utime(self.audio,ns=(st.st_atime_ns,st.st_mtime_ns))
        self.assertEqual(self.audio.stat().st_size,r['audio_stat']['bytes'])
        with self.assertRaises(ValueError):s.validate_unit(r,verify_audio=True)
    def test_resample_preserves_exact_continuous_span(self):
        from scipy.signal import resample_poly
        p=self.root/'24k.wav';w=np.sin(np.arange(2401)*.2).astype('float32')
        sf.write(p,w,24000,subtype='FLOAT')
        r=s.make_unit(p,'spk','你好',evidence='reference_timed_segment',start_sample=11,end_sample=2400,provenance={'id':'fixture'})
        out=s.read_unit(r);expected=resample_poly(w[11:2400],2,3).astype('float32')
        self.assertEqual(len(out),1593);self.assertTrue(np.array_equal(out,expected))
    def test_protected_output_ancestors(self):
        for p in ['/root','/root/code/github_repos/redimnet2-fork/xx','/root/group-shared/voiceprint/data/xx']:
            with self.assertRaises(ValueError):s.protect_output(p)
        self.assertEqual(s.protect_output(self.root/'results'),(self.root/'results').resolve())

if __name__=='__main__':unittest.main()
