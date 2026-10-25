import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from duration_ft.evaluate import (ALL_O_AXIS, COMMON_8_AXIS, EmbeddingCache,
    EvaluationRefused, audio_record, build_matrices, cache_identity, center_crop,
    extract_one, load_model, prepare_calibration, read_trials, score_cell, digest,
    evaluate_matrix, calibrate_model, protect_output)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record = dict(relative='id1/a/x.wav', path='unused', resolved='/source/x.wav',
                           frames=160001, sample_rate=16000, channels=1,
                           stat=[1, 2, 640044, 4, 5], sha256='audio-A')
        self.identity = dict(checkpoint_sha256='model-A', code={'sha256':'code-A'},
                             embedding_dim=2, device='cpu', precision='float32')

    def write_wav(self, rel, seconds=10):
        import soundfile as sf
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
        sf.write(path, .1 * np.sin(2 * np.pi * 310 * t), 16000, subtype='PCM_16')
        return path

    def trial_table(self):
        for rel, duration in [('id1/a/1.wav',10),('id1/b/2.wav',9),('id2/a/3.wav',12),('id2/b/4.wav',4)]:
            self.write_wav(rel, duration)
        protocol = self.root / 'trials.txt'
        protocol.write_text('1 id1/a/1.wav id1/b/2.wav\n0 id1/a/1.wav id2/a/3.wav\n0 id1/a/1.wav id2/b/4.wav\n')
        trials, _ = read_trials(protocol)
        records = {x:audio_record(self.root,x) for t in trials for x in [t['enroll'],t['verify']]}
        return protocol, trials, records

    def test_true_half_second_and_nested_center(self):
        small, medium, full = [center_crop(self.record,d) for d in [.5,3,'full']]
        self.assertEqual(small['actual_samples'],8000)
        self.assertLessEqual(medium['start_sample'],small['start_sample'])
        self.assertGreaterEqual(medium['end_sample'],small['end_sample'])
        self.assertEqual((full['start_sample'],full['end_sample']),(0,160001))
        self.assertNotEqual(full['actual_samples'],8*16000)

    def test_too_short_refused_not_padded(self):
        short = dict(self.record,frames=7999)
        with self.assertRaises(EvaluationRefused):center_crop(short,.5)
        with self.assertRaises(EvaluationRefused):center_crop(dict(self.record,frames=4*16000),8)

    def test_invalid_durations(self):
        for value in [0,-1,float('nan'),float('inf'),True,.00001]:
            with self.subTest(value=value),self.assertRaises(EvaluationRefused):center_crop(self.record,value)

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['soundfile']))
    def test_same_common_cohort_in_every_cell(self):
        _,trials,records=self.trial_table()
        matrices=build_matrices(trials,records)
        self.assertEqual(matrices[0]['axis'],list(ALL_O_AXIS))
        self.assertEqual(matrices[1]['axis'],list(COMMON_8_AXIS))
        self.assertEqual(matrices[0]['trial_count'],3)
        self.assertEqual(matrices[1]['trial_count'],2)
        self.assertEqual(matrices[1]['positive'],1)
        self.assertEqual(matrices[1]['negative'],1)
        for d in matrices[1]['axis']:
            for t in matrices[1]['trials']:
                for key in ('enroll','verify'):center_crop(records[t[key]],d)

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['soundfile']))
    def test_standard_counts_reject_synthetic(self):
        _,trials,records=self.trial_table()
        with self.assertRaises(EvaluationRefused):build_matrices(trials,records,strict_standard=True)

    def test_protocol_traversal_and_missing_classes(self):
        p=self.root/'bad.txt'
        for raw in ['1 /tmp/x.wav id1/a/1.wav\n0 id2/a/2.wav id3/a/3.wav\n',
                    '1 id1/../x.wav id1/a/1.wav\n0 id2/a/2.wav id3/a/3.wav\n',
                    '1 id1/a/1.wav id1/a/1.wav\n']:
            p.write_text(raw)
            with self.assertRaises(EvaluationRefused):read_trials(p)

    def test_cache_has_model_crop_audio_and_processing_identity(self):
        a=cache_identity(self.identity,center_crop(self.record,1));keys={digest(a)}
        for changed in [cache_identity(dict(self.identity,checkpoint_sha256='B'),center_crop(self.record,1)),
                        cache_identity(dict(self.identity,code={'sha256':'B'}),center_crop(self.record,1)),
                        cache_identity(self.identity,center_crop(self.record,3)),
                        cache_identity(self.identity,center_crop(dict(self.record,sha256='B'),1))]:
            self.assertNotIn(digest(changed),keys);keys.add(digest(changed))
        self.assertFalse(a['preprocess']['pad']);self.assertFalse(a['preprocess']['wrap'])

    def test_cache_resume_and_corruption_refusal(self):
        cache=EmbeddingCache(self.root/'cache');identity=cache_identity(self.identity,center_crop(self.record,1))
        self.assertIsNone(cache.get(identity));cache.put(identity,np.array([.6,.8],np.float32))
        np.testing.assert_allclose(cache.get(identity),[.6,.8])
        p=cache.path(identity);raw=json.loads(p.read_text());raw['vector']=[1.,0.];p.write_text(json.dumps(raw))
        with self.assertRaises(EvaluationRefused):cache.get(identity)

    def test_invalid_vectors_refused(self):
        cache=EmbeddingCache(self.root/'cache');identity=cache_identity(self.identity,center_crop(self.record,1))
        for vec in [[0.,0.],[float('nan'),1.],[1.,2.,3.]]:
            with self.subTest(vector=vec),self.assertRaises(EvaluationRefused):cache.put(identity,np.array(vec))

    def test_non_symmetric_scoring_uses_distinct_crops(self):
        trials=[dict(trial_id='0',label=1,enroll='x',verify='y')]
        _,scores,_=score_cell(trials,{'x':np.array([1.,0.])},{'y':np.array([0.,1.])})
        self.assertEqual(scores,[0.])
        _,reverse,_=score_cell(trials,{'x':np.array([1.,0.])},{'y':np.array([1.,0.])})
        self.assertEqual(reverse,[1.])

    def test_missing_embedding_retains_pair(self):
        rows,scores,labels=score_cell([dict(trial_id='0',label=1,enroll='x',verify='y')],{}, {})
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['status'],'missing_embedding')
        self.assertFalse(scores);self.assertFalse(labels)

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_cpu_forward_true_samples_and_resumed_cache_skips_forward(self):
        import torch
        path=self.write_wav('id1/a/1.wav',10);record=audio_record(self.root,'id1/a/1.wav')
        crop=center_crop(record,.5);calls=[]
        class Model(torch.nn.Module):
            def forward(self,x):
                calls.append(tuple(x.shape));return torch.tensor([[3.,4.]])
        cache=EmbeddingCache(self.root/'cache');vector,hit=extract_one(Model(),self.identity,record,crop,cache)
        self.assertFalse(hit);self.assertEqual(calls,[(1,8000)]);np.testing.assert_allclose(vector,[.6,.8])
        with mock.patch.object(Model,'forward',side_effect=AssertionError('should use cache')):
            _,hit=extract_one(Model(),self.identity,record,crop,cache);self.assertTrue(hit)

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_decode_short_read_refused(self):
        import torch
        self.write_wav('id1/a/1.wav',10);record=audio_record(self.root,'id1/a/1.wav')
        with mock.patch('soundfile.read',return_value=(np.ones((10,1),np.float32),16000)):
            with self.assertRaises(EvaluationRefused):extract_one(torch.nn.Identity(),self.identity,record,center_crop(record,1),EmbeddingCache(self.root/'cache'))

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_source_changed_refused(self):
        import torch
        p=self.write_wav('id1/a/1.wav',10);record=audio_record(self.root,'id1/a/1.wav')
        p.touch()
        with self.assertRaises(EvaluationRefused):extract_one(torch.nn.Identity(),self.identity,record,center_crop(record,1),EmbeddingCache(self.root/'cache'))

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['soundfile']))
    def test_prepare_speaker_disjoint_cross_video_and_negative_count(self):
        for speaker in ['id1','id2','id3','id4']:
            for video in ['a','b']:
                self.write_wav(f'{speaker}/{video}/1.wav',4)
                self.write_wav(f'{speaker}/{video}/2.wav',4)
        protocol=self.root/'o.txt';protocol.write_text('1 id1/a/1.wav id1/b/1.wav\n0 id1/a/1.wav id2/a/1.wav\n')
        info=prepare_calibration(self.root,protocol,self.root/'cal',speaker_count=2,clips_per_speaker=2)
        self.assertEqual(info['speakers'],['id3','id4']);self.assertEqual(info['clip_count'],4)
        self.assertEqual(info['positive'],2);self.assertEqual(info['negative'],4)
        self.assertFalse(info['eh_independent']);self.assertIsNone(info['thresholds'])
        for clips in info['audio'].values():self.assertEqual(len({Path(r['relative']).parts[1] for r in clips}),2)
        self.assertEqual(info,prepare_calibration(self.root,protocol,self.root/'cal',speaker_count=2,clips_per_speaker=2))

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_matrix_74_cells_and_cache_resume(self):
        import torch
        protocol,_,_=self.trial_table();calls=[]
        class Model(torch.nn.Module):
            def forward(self,x):
                calls.append(tuple(x.shape));return torch.tensor([[3.,4.]])
        out=self.root/'eval'
        with mock.patch('duration_ft.evaluate.load_model',return_value=(Model(),self.identity)):
            first=evaluate_matrix(Path('mock.pt'),protocol,self.root,out,strict_standard=False)
            self.assertTrue(first['complete']);self.assertEqual(len(first['cells']),74)
            self.assertEqual(first['unique_crops'],26)
            self.assertEqual(first['cache_hits'],0)
            second=evaluate_matrix(Path('mock.pt'),protocol,self.root,out,strict_standard=False)
            self.assertEqual(second['cache_hits'],26);self.assertEqual(len(calls),26)
            for cell in second['cells']:
                self.assertEqual(cell['expected_trials'],3 if cell['matrix']=='all-O' else 2)
                self.assertEqual(cell['fixed_threshold']['status'],'unavailable')

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_failed_forward_keeps_errors_and_nulls_biased_metrics(self):
        import torch
        protocol,_,_=self.trial_table()
        class Bad(torch.nn.Module):
            def forward(self,x):raise RuntimeError('deliberate failure')
        out=self.root/'fail'
        with mock.patch('duration_ft.evaluate.load_model',return_value=(Bad(),self.identity)):
            result=evaluate_matrix(Path('mock.pt'),protocol,self.root,out,strict_standard=False)
        self.assertFalse(result['complete']);self.assertTrue(result['extraction_errors'])
        self.assertTrue((out/'extraction_errors.json').exists())
        self.assertTrue(all(c['metrics'] is None for c in result['cells']))


    def test_protected_original_output_refused(self):
        for p in ['/root/code/github_repos/redimnet2-fork/reports/new',
                  '/root/experiments/redimnet2/b6-mix-official-20261019/new',
                  '/root/group-shared/voiceprint/data/temporary-output']:
            with self.subTest(path=p),self.assertRaises(EvaluationRefused):protect_output(Path(p))

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch']), 'optional CPU dependencies unavailable: '+', '.join(['torch']))
    def test_strict_export_loading_and_gpu_requires_explicit_budget(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self,embed_dim=2):
                super().__init__();self.linear=torch.nn.Linear(2,embed_dim)
        p=self.root/'released.pt';net=Tiny()
        torch.save({'model_config':{'embed_dim':2},'state_dict':net.state_dict()},p)
        with mock.patch('redimnet2.ReDimNet2Wrap',Tiny):
            loaded,identity=load_model(p)
            self.assertEqual(identity['embedding_dim'],2)
            for key,value in loaded.state_dict().items():self.assertTrue(torch.equal(value,net.state_dict()[key]))
            with mock.patch('torch.cuda.get_device_properties') as gpu:
                with self.assertRaises(EvaluationRefused):load_model(p,'cuda:0')
                with self.assertRaises(EvaluationRefused):load_model(p,'cuda:0',allow_gpu=True)
                gpu.assert_not_called()
            torch.save({'model_config':{'embed_dim':2},'state_dict':{}},p)
            with self.assertRaises(RuntimeError):load_model(p)
        torch.save({'model':net.state_dict(),'epoch':4},p)
        with self.assertRaises(EvaluationRefused):load_model(p)

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_cached_source_change_refused(self):
        import torch
        p=self.write_wav('id1/a/1.wav',10);record=audio_record(self.root,'id1/a/1.wav')
        crop=center_crop(record,1);cache=EmbeddingCache(self.root/'cache')
        cache.put(cache_identity(self.identity,crop),np.array([.6,.8],np.float32));p.touch()
        with self.assertRaises(EvaluationRefused):extract_one(torch.nn.Identity(),self.identity,record,crop,cache)


    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_calibrate_full_dev_then_apply_frozen_thresholds_and_reject_overlap(self):
        import torch
        protocol,_,_=self.trial_table()
        for speaker in ['id3','id4']:
            for video in ['a','b']:self.write_wav(f'{speaker}/{video}/1.wav',4)
        caldir=self.root/'cal-prepare'
        prepare_calibration(self.root,protocol,caldir,speaker_count=2,clips_per_speaker=2)
        class Model(torch.nn.Module):
            def forward(self,x):return torch.tensor([[3.,4.]])
        with mock.patch('duration_ft.evaluate.load_model',return_value=(Model(),self.identity)):
            result=calibrate_model(Path('mock.pt'),caldir/'selection.json',self.root/'cal-results')
            threshold_file=self.root/'cal-results/thresholds.json'
            self.assertEqual(set(result['thresholds']),{'far_0.01','far_0.001'})
            self.assertFalse(result['eh_independent'])
            scored=evaluate_matrix(Path('mock.pt'),protocol,self.root,self.root/'applied',strict_standard=False,calibration=threshold_file)
            self.assertIn('far_0.01',scored['cells'][0]['fixed_threshold'])
            result['speakers']=['id1'];threshold_file.write_text(json.dumps(result))
            with self.assertRaises(EvaluationRefused):
                evaluate_matrix(Path('mock.pt'),protocol,self.root,self.root/'overlap',strict_standard=False,calibration=threshold_file)

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['soundfile']))
    def test_index_failure_preserved_before_any_model_load(self):
        protocol,_,_=self.trial_table();(self.root/'id2/b/4.wav').unlink()
        with mock.patch('duration_ft.evaluate.load_model') as loader:
            with self.assertRaises(EvaluationRefused):evaluate_matrix(Path('mock.pt'),protocol,self.root,self.root/'bad-header',strict_standard=False)
            loader.assert_not_called()
        errors=json.loads((self.root/'bad-header/index_errors.json').read_text())
        self.assertEqual(errors[0]['relative'],'id2/b/4.wav')


    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_cuda_oom_fails_fast_and_preserves_complete_trace(self):
        import torch
        protocol,_,_=self.trial_table();calls=[]
        class OOM(torch.nn.Module):
            def forward(self,x):calls.append(1);raise torch.cuda.OutOfMemoryError('synthetic CUDA OOM; no GPU call')
        output=self.root/'oom'
        with mock.patch('duration_ft.evaluate.load_model',return_value=(OOM(),self.identity)):
            result=evaluate_matrix(Path('mock.pt'),protocol,self.root,output,strict_standard=False)
        self.assertEqual(len(calls),1);self.assertEqual(result['attempted_crops'],1)
        self.assertEqual(result['unattempted_crops'],25);self.assertTrue(result['aborted'])
        self.assertFalse(result['complete']);self.assertEqual(len(result['extraction_errors']),1)
        self.assertIn('OutOfMemoryError',result['extraction_errors'][0]['traceback'])
        self.assertIn('synthetic CUDA OOM',result['extraction_errors'][0]['traceback'])
        self.assertTrue(list((output/'failures').glob('*.json')))
        self.assertEqual(len(json.loads((output/'crops.json').read_text())),26)

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch', 'soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['torch', 'soundfile']))
    def test_audio_errors_bounded_and_isolated_decode_can_continue(self):
        import torch
        import soundfile as sf
        protocol,_,_=self.trial_table()
        class Model(torch.nn.Module):
            def forward(self,x):return torch.tensor([[3.,4.]])
        with mock.patch('duration_ft.evaluate.load_model',return_value=(Model(),self.identity)), mock.patch('soundfile.read',side_effect=RuntimeError('synthetic bad audio')):
            result=evaluate_matrix(Path('mock.pt'),protocol,self.root,self.root/'decode-limit',strict_standard=False,max_audio_errors=2,max_consecutive_audio_errors=3)
        self.assertTrue(result['aborted']);self.assertEqual(result['abort_reason'],'bounded_audio_error_limit')
        self.assertEqual(result['attempted_crops'],2);self.assertEqual(result['audio_error_count'],2)
        original=sf.read;count=[0]
        def once(*args,**kwargs):
            count[0]+=1
            if count[0]==1:raise RuntimeError('one bad crop')
            return original(*args,**kwargs)
        with mock.patch('duration_ft.evaluate.load_model',return_value=(Model(),self.identity)), mock.patch('soundfile.read',side_effect=once):
            result=evaluate_matrix(Path('mock.pt'),protocol,self.root,self.root/'decode-once',strict_standard=False)
        self.assertFalse(result['aborted']);self.assertEqual(result['audio_error_count'],1)
        self.assertEqual(result['attempted_crops'],26);self.assertFalse(result['complete'])
        self.assertTrue(any(cell['complete'] for cell in result['cells']))

    @unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['soundfile']), 'optional CPU dependencies unavailable: '+', '.join(['soundfile']))
    def test_model_load_failure_writes_incomplete_summary(self):
        protocol,_,_=self.trial_table();output=self.root/'load-failed'
        with mock.patch('duration_ft.evaluate.load_model',side_effect=RuntimeError('synthetic load failure')):
            result=evaluate_matrix(Path('mock.pt'),protocol,self.root,output,strict_standard=False)
        self.assertFalse(result['complete']);self.assertEqual(result['phase'],'model_load')
        self.assertEqual(result['cells'],[]);self.assertIn('synthetic load failure',result['extraction_errors'][0]['traceback'])
        self.assertTrue((output/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
