import importlib.util as _deps
import unittest
if any(_deps.find_spec(p) is None for p in ['soundfile']):
    raise unittest.SkipTest('optional CPU test dependencies unavailable: '+', '.join(['soundfile']))
import copy,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from duration_ft.data import *

def fixture(root,key='A_s/a.wav',speaker='A_s'):
    path=Path(root)/key;path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,np.sin(np.arange(96000,dtype=np.float32)*.019),16000,subtype='FLOAT')
    vad=Path(root)/'vad';p,side=annotation_paths(path,vad);p.parent.mkdir(parents=True,exist_ok=True)
    st=source_stat(path)
    headers={'source_size':st['bytes'],'source_mtime_ns':st['mtime_ns'],'source_sample_rate':16000,
             'source_channels':1,'source_frames':96000,'sample_rate':16000,'samples_16k':96000,
             'segments':2,'schema_version':2,'inference_signature':SIGNATURE}
    p.write_text(''.join(f'# {k}={v}\n' for k,v in headers.items())+
       f'SPEAKER {path.stem} 1 0.000 0.500 <NA> <NA> <NA> <NA> <NA>\n'+
       f'SPEAKER {path.stem} 1 0.750 5.250 <NA> <NA> <NA> <NA> <NA>\n')
    meta={'source_size':st['bytes'],'source_mtime_ns':st['mtime_ns'],'source_sample_rate':16000,
          'source_num_channels':1,'sample_length':96000,'segments':2,'schema_version':2,'inference_signature':SIGNATURE}
    side.write_text(json.dumps(meta))
    obj={'key':key,'spk':speaker,'wav':str(path)}
    return obj,vad,p,side

class DataTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.obj,self.vad,self.p,self.side=fixture(self.tmp.name)
    def inspect(self):return inspect_record(self.obj,self.vad)
    def error(self,reason):
        with self.assertRaises(DataError) as ctx:self.inspect()
        self.assertEqual(ctx.exception.reason,reason)
    def test_valid_and_nested_true_length(self):
        r=self.inspect();long=speech_crop_plan(r,5,42,7);short=speech_crop_plan(r,1,42,7)
        a=read_speech_crop(r,long);b=read_speech_crop(r,short)
        self.assertEqual(len(a),80000);self.assertEqual(len(b),16000)
        np.testing.assert_array_equal(a[32000:48000],b)
        self.assertEqual(long['samples'],sum(y-x for x,y in long['intervals']))
    def test_missing_rttm(self):self.p.unlink();self.error('missing_rttm')
    def test_missing_sidecar(self):self.side.unlink();self.error('missing_sidecar')
    def test_empty_annotation(self):
        raw=self.p.read_text();self.p.write_text('\n'.join(l.replace('# segments=2','# segments=0') for l in raw.splitlines() if l.startswith('#'))+'\n')
        m=json.loads(self.side.read_text());m['segments']=0;self.side.write_text(json.dumps(m));self.error('empty_vad')
    def test_nan_time(self):self.p.write_text(self.p.read_text().replace('0.000 0.500','nan 0.500'));self.error('bad_rttm_time')
    def test_bad_columns(self):self.p.write_text(self.p.read_text()+'SPEAKER x 1 1 2\n');self.error('bad_rttm_row')
    def test_stale(self):
        m=json.loads(self.side.read_text());m['source_size']+=1;self.side.write_text(json.dumps(m));self.error('stale_annotation')
    def test_signature(self):
        m=json.loads(self.side.read_text());m['inference_signature']='wrong';self.side.write_text(json.dumps(m));self.error('signature_mismatch')
    def test_out_of_bounds(self):self.p.write_text(self.p.read_text().replace('0.750 5.250','0.750 6.250'));self.error('overlap_or_out_of_bounds')
    def test_insufficient_does_not_wrap(self):
        r=self.inspect()
        with self.assertRaises(DataError):speech_crop_plan(r,6,42,7)
    def test_no_discard_sampler(self):
        stream=[]
        for step in range(4):stream+=global_batch_indices(5,step,2,2,31)
        self.assertEqual(len(stream),16)
        for k in range(0,15,5):self.assertEqual(sorted(stream[k:k+5]),list(range(5)))
    def test_sampler_epoch_cache_and_resume_tail(self):
        from duration_ft.data import _epoch_order
        _epoch_order.cache_clear()
        a=global_batch_indices(101,0,4,2,7)
        b=global_batch_indices(101,1,4,2,7)
        self.assertEqual(_epoch_order.cache_info().misses,1)
        self.assertEqual(len(a),8);self.assertEqual(len(b),8)
        tail=global_batch_indices(101,12,4,2,7)
        self.assertEqual(len(tail),8)
        self.assertEqual(tail,global_batch_indices(101,12,4,2,7))
        self.assertLessEqual(_epoch_order.cache_info().currsize,2)

    def test_step_schedule_rank_independent(self):
        a=[length_for_step(i,42,[1,2,3,5],[1/6,1/6,1/6,.5]) for i in range(100)]
        b=[length_for_step(i,42,[1,2,3,5],[1/6,1/6,1/6,.5]) for i in range(100)]
        self.assertEqual(a,b);self.assertEqual(set(a),{1,2,3,5})
    def test_dataset_tamper(self):
        r=dict(self.inspect(),split='train');obj=seal_dataset([r],{r['speaker']:0},{'split':'train'})
        p=Path(self.tmp.name)/'dataset.json';p.write_text(json.dumps(obj));load_dataset(p)
        obj['records'][0]['segments'][0][1]+=1;p.write_text(json.dumps(obj))
        with self.assertRaises(DataError):load_dataset(p)
    def test_audio_changed_rejected(self):
        r=self.inspect();self.p.write_text(self.p.read_text()+'# changed=yes\n')
        with self.assertRaises(DataError):read_speech_crop(r,speech_crop_plan(r,1,1,1))

if __name__=='__main__':unittest.main()
