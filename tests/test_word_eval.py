import importlib.util
import copy
import unittest
from word_ft.prepare import chinese_reference, count_bin, select_speaker_splits, build_count_matrix


def candidates(speakers=8):
    rows=[]
    for s in range(speakers):
        for b,text in [('2-3','我想你'),('4-6','我吃多了'),('7+','今天我们一起出去玩')]:
            for slot in range(3):
                uid=f'{s}-{b}-{slot}'
                rows.append({'unit_id':uid,'source_utterance_id':uid,'speaker_id':str(s),
                    'count_bin':b,'spoken_text':text+('啊' if slot==2 else ''), 'n_units':len(text), 'path':'unused'})
    return rows


class PreparationTests(unittest.TestCase):
    def test_reference_no_numeric_or_latin_guess(self):
        self.assertEqual(chinese_reference('我想你。'),'我想你')
        for text in ('你数到20','OK','ＡＢ','123',''):
            with self.subTest(text=text), self.assertRaises(ValueError):chinese_reference(text)

    def test_count_bins(self):
        self.assertEqual([count_bin(n) for n in [0,1,2,3,4,6,7]],[None,'1','2-3','2-3','4-6','4-6','7+'])

    def test_disjoint_and_deterministic(self):
        a,meta=select_speaker_splits(candidates(),dev_speakers=3,test_speakers=3)
        b,other=select_speaker_splits(list(reversed(candidates())),dev_speakers=3,test_speakers=3)
        self.assertEqual(a,b);self.assertEqual(meta,other)
        self.assertFalse(set(meta['splits']['dev-cal'])&set(meta['splits']['test']))

    def test_missing_single_char_is_unavailable(self):
        u,_=select_speaker_splits(candidates(),dev_speakers=3,test_speakers=3)
        matrix=build_count_matrix(u)
        self.assertEqual(len(matrix['cells']),16)
        for cell in matrix['cells']:
            group=cell['groups']['all']
            if cell['enroll_count_bin']=='1' or cell['test_count_bin']=='1':
                self.assertFalse(group['available']);self.assertEqual(group['trials'],[])
            else:
                self.assertTrue(group['available']);self.assertEqual(group['positive'],3);self.assertEqual(group['negative'],6)

    def test_no_source_overlap(self):
        u,_=select_speaker_splits(candidates(),dev_speakers=3,test_speakers=3)
        test=[x for x in u if x['split']=='test']
        a=next(x for x in test if x['count_bin']=='2-3' and x['template_slot']==0)
        b=next(x for x in test if x['speaker_id']==a['speaker_id'] and x['count_bin']=='2-3' and x['template_slot']==1)
        b['source_utterance_id']=a['source_utterance_id']
        with self.assertRaisesRegex(ValueError,'leakage'):build_count_matrix(u)

    def test_one_isolated_file_cannot_change_fixed_cohort(self):
        u,_=select_speaker_splits(candidates(),dev_speakers=3,test_speakers=3)
        x=copy.deepcopy(next(x for x in u if x['split']=='test'))
        x.update(unit_id='only-one',source_utterance_id='only-one',count_bin='1',template_slot=0,spoken_text='我',n_units=1)
        u.append(x)
        for c in build_count_matrix(u)['cells']:
            if c['enroll_count_bin']=='1' or c['test_count_bin']=='1':
                self.assertFalse(c['groups']['all']['available']);self.assertEqual(c['groups']['all']['trials'],[])

    def test_duplicate_slot(self):
        u,_=select_speaker_splits(candidates(),dev_speakers=3,test_speakers=3)
        u.append(copy.deepcopy(next(x for x in u if x['split']=='test')))
        with self.assertRaisesRegex(ValueError,'duplicate'):build_count_matrix(u)

    def test_no_fabricated_matched_class(self):
        u,_=select_speaker_splits(candidates(),dev_speakers=3,test_speakers=3)
        for x in u:x['spoken_text']=x['unit_id']
        m=build_count_matrix(u)
        for c in m['cells']:self.assertFalse(c['groups']['matched']['available'])

    def test_insufficient_speakers(self):
        with self.assertRaises(ValueError):select_speaker_splits(candidates(3),dev_speakers=2,test_speakers=2)



@unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['soundfile','torch']), 'optional CPU soundfile/torch unavailable')
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        import numpy as np
        import soundfile as sf
        from pathlib import Path
        from word_ft import schema
        from word_ft.prepare import digest
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        units=[]
        for split,sp in [('dev-cal',0),('dev-cal',1),('test',2),('test',3)]:
            for b,text in [('2-3','我想你'),('4-6','我吃多了'),('7+','今天我们一起出去玩')]:
                for slot in (0,1):
                    path=self.root/f'{sp}-{b}-{slot}.wav'
                    n=600+sp*10+slot*17+len(text)*31
                    wave=(np.sin(np.arange(n)*(.01+sp*.001+slot*.0001+len(text)*.00001))*.2).astype('float32')
                    sf.write(path,wave,16000)
                    unit=schema.make_unit(path,str(sp),text,split=split,source_utterance_id=path.stem)
                    unit['template_slot']=slot;units.append(unit)
        self.data=schema.write_dataset(self.root/'units.json',units)
        self.proto={'schema':'word-count-matrix/v1','dataset_id':self.data['dataset_id'],'dataset_file':'units.json',
            'matrices':{s:build_count_matrix(units,s) for s in ('dev-cal','test')}}
        self.proto['protocol_id']=digest(self.proto)
        import json
        (self.root/'protocol.json').write_text(json.dumps(self.proto))
        self.ckpt=self.root/'model.pt';self.ckpt.write_bytes(b'fake-model-identity')

    def tearDown(self):self.tmp.cleanup()

    def loader(self,*a,**kw):
        import torch
        from duration_ft.evaluate import sha_file
        class Model(torch.nn.Module):
            def forward(self,wave):
                self.last_samples=wave.shape[-1]
                return torch.stack((wave.mean(dim=-1),wave.std(dim=-1),torch.ones(wave.shape[0])),dim=-1)
        return Model(),{'checkpoint_sha256':sha_file(self.ckpt),'embedding_dim':3,'device':'cpu'}

    def evaluate(self,output='result',**kw):
        from unittest.mock import patch
        from word_ft.evaluate import evaluate_words
        with patch('word_ft.evaluate.load_model',side_effect=self.loader):
            return evaluate_words(self.ckpt,self.root/'units.json',self.root/'protocol.json',self.root/output,**kw)

    def test_complete_and_resume_cache(self):
        a=self.evaluate();self.assertTrue(a['complete']);self.assertEqual(a['extracted_units'],12)
        self.assertEqual(a['cache_hits'],0)
        b=self.evaluate();self.assertTrue(b['complete']);self.assertEqual(b['cache_hits'],12)
        c=next(x for x in b['cells'] if x['name']=='2-3x2-3' and x['content_condition']=='all')
        self.assertEqual(c['trial_count'],4);self.assertIsNotNone(c['metrics'])
        self.assertTrue(any(not x['available'] for x in b['cells']))

    def test_other_request_preserves_summary(self):
        self.evaluate();before=(self.root/'result/summary.json').read_bytes();self.ckpt.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'another request'):self.evaluate()
        self.assertEqual((self.root/'result/summary.json').read_bytes(),before)

    def test_device_failure_aborts_first(self):
        from unittest.mock import patch
        from duration_ft.evaluate import ModelExecutionError
        with patch('word_ft.evaluate.extract_unit',side_effect=ModelExecutionError('CUDA OOM synthetic')) as probe:
            result=self.evaluate()
        self.assertFalse(result['complete']);self.assertEqual(probe.call_count,1)
        self.assertTrue(result['errors'][0]['traceback']);self.assertTrue(all(x['metrics'] is None for x in result['cells']))

    def test_audio_failure_budget(self):
        from unittest.mock import patch
        from duration_ft.evaluate import AudioDecodeError
        with patch('word_ft.evaluate.extract_unit',side_effect=AudioDecodeError('synthetic decode')) as probe:
            result=self.evaluate(max_audio_errors=5,max_consecutive_audio_errors=2)
        self.assertFalse(result['complete']);self.assertEqual(probe.call_count,2)

    def test_protocol_tamper(self):
        import json
        from word_ft.prepare import digest
        p=self.root/'protocol.json';value=json.loads(p.read_text())
        trial=value['matrices']['test']['cells'][5]['groups']['all']['trials'][0]
        trial['enroll']=trial['verify']
        value.pop('protocol_id');value['protocol_id']=digest(value);p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'matrix differs'):self.evaluate()

    def test_source_audio_change_rejected(self):
        path=next(self.root.glob('2-*.wav'));path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'source audio'):self.evaluate()

    def test_same_source_id_leakage_rejected(self):
        from word_ft import schema
        from word_ft.evaluate import read_protocol
        import json
        d=json.loads((self.root/'units.json').read_text());test=[u for u in d['records'] if u['split']=='test']
        test[1]['source_utterance_id']=test[0]['source_utterance_id'];test[1]['unit_id']=schema.digest(schema.unit_identity(test[1]))
        d=schema.seal_dataset(d['records']);(self.root/'units.json').write_text(json.dumps(d))
        with self.assertRaises(ValueError):read_protocol(self.root/'units.json',self.root/'protocol.json')

    def test_fixed_calibration_and_test_no_refit(self):
        from word_ft.evaluate import calibrate_words
        dev=self.evaluate('dev',split='dev-cal');self.assertTrue(dev['complete'])
        cal=calibrate_words(self.root/'dev/summary.json',self.root/'units.json',self.root/'cal.json')
        self.assertEqual(cal['dev_cell'],'4-6x4-6');self.assertEqual(len(cal['dev_speakers']),2)
        result=self.evaluate(calibration=self.root/'cal.json');self.assertTrue(result['complete'])
        for cell in result['cells']:
            if cell['complete']:self.assertEqual(set(cell['fixed_threshold']),{'far_0.01','far_0.001'})
        with self.assertRaises(ValueError):calibrate_words(self.root/'result/summary.json',self.root/'units.json',self.root/'bad.json')

    def test_calibration_speaker_tamper(self):
        from word_ft.evaluate import calibrate_words
        from duration_ft.evaluate import digest
        import json
        self.evaluate('dev',split='dev-cal');cal=calibrate_words(self.root/'dev/summary.json',self.root/'units.json',self.root/'cal.json')
        cal['dev_speakers']=['2'];cal.pop('calibration_id');cal['calibration_id']=digest(cal)
        (self.root/'bad-cal.json').write_text(json.dumps(cal))
        result=self.evaluate(calibration=self.root/'bad-cal.json')
        self.assertFalse(result['complete']);self.assertIn('overlap',result['errors'][0]['message'])

    def test_candidate_default_rejected(self):
        from word_ft import schema
        from word_ft.evaluate import read_protocol
        import json
        d=json.loads((self.root/'units.json').read_text());u=d['records'][0]
        u.update(evidence='forced_alignment_candidate',boundary_review='pending',alignment={'method':'synthetic'},transcript_origin='asr_hypothesis')
        u['unit_id']=schema.digest(schema.unit_identity(u));d=schema.seal_dataset(d['records'])
        (self.root/'units.json').write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'candidate'):read_protocol(self.root/'units.json',self.root/'protocol.json')

if __name__=='__main__':unittest.main()
