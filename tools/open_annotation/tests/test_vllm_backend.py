"""只用CPU假引擎验证原生ID、结束证据和事务恢复；不导入真实推理库。"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import vllm_backend as V
from asr_stage_store import ASRStageStore,digest


def spec():
    return {'schema':V.SPEC_SCHEMA,'purpose':'cpu_fixture','model_API_calls_allowed':0,'release_training_labels':False,
        'model':{'path':'/synthetic/model','files_sha256':{'config.json':'a'*64}},
        'engine':{'backend':'vllm','version':'synthetic-vllm','dtype':'bfloat16','tensor_parallel_size':1,
                  'gpu_memory_utilization':0.55,'max_model_len':256},
        'decode':{'temperature':0.0,'max_tokens':8,'seed':0,'n':1,'ignore_eos':False},
        'max_batch_rows':4,'normalization_profile':V.POLICY,
        'preprocessing':{'model_sample_rate':16000,'pcm_policy':'float32le_mono/v1','window_policy_sha256':'b'*64},
        'batch_policy':'fixed_cohort_until_durable; immutable_published_result_reuse/v1',
        'eos_token_ids':[99],'supported_languages':['Chinese','English'],'force_language':None,'context':''}


def rows(count=2):
    ans=[]
    for i in range(count):
        a=np.asarray([i+1,.1,.2,.3],dtype=np.float32)
        ans.append({'source_id':'source-'+str(i),'window_id':'w0','audio':a,
                    'input_pcm_sha256':hashlib.sha256(a.astype('<f4').tobytes()).hexdigest(),
                    'model_input_frames':len(a),'model_sample_rate':16000})
    return ans


class FakeEngine:
    def __init__(self,mode='normal'):self.mode=mode;self.calls=0;self.inputs=[];self.params=[]
    def enqueue(self,inputs,sampling_params,use_tqdm=False):
        self.calls+=1;self.inputs=inputs;self.params.append(copy.deepcopy(sampling_params))
        return ['native-'+str(i) for i in range(len(inputs))]
    def wait_for_completion(self,use_tqdm=False):
        outputs=[{'request_id':'native-'+str(i),'finished':True,'prompt_token_ids':[1,2],
                  'outputs':[{'index':0,'text':'Chinese|'+('你好' if i==0 else '世界'),
                              'token_ids':[10+i,99],'finish_reason':'stop','stop_reason':99}]}
                 for i in range(len(self.inputs))]
        if self.mode=='reordered':return outputs[::-1]
        if self.mode=='missing':return outputs[:-1]
        if self.mode=='duplicate':return [outputs[0]]*len(outputs)
        if self.mode=='foreign':outputs[0]['request_id']='foreign-native'
        if self.mode=='exception':raise RuntimeError('synthetic engine failure')
        for row in outputs:
            c=row['outputs'][0]
            if self.mode=='missing_finished':row.pop('finished')
            elif self.mode=='no_termination':c.update(token_ids=[10],finish_reason=None,stop_reason=None)
            elif self.mode=='stop_without_eos':c.update(token_ids=[10],stop_reason=99)
            elif self.mode=='arbitrary_stop':c.update(token_ids=[10],stop_reason='not-declared')
            elif self.mode=='after_eos':c['token_ids']=[99,10]
            elif self.mode=='repeat_eos':c['token_ids']=[99,99]
            elif self.mode=='truncated':c.update(token_ids=[10]*8,finish_reason='length',stop_reason=None)
            elif self.mode=='invalid_tokens':c['token_ids']=[float('nan')]
            elif self.mode=='unsupported_language':c['text']='Klingon|qapla'
            elif self.mode=='abort':c.update(token_ids=[],finish_reason='abort',stop_reason=None)
        return outputs


def adapter(root,engine,producer=None,parser=None):
    producer=producer or spec()
    store=ASRStageStore(root,digest(producer),synthetic_fixture=True)
    return V.VLLMBackend.from_fake_engine(producer,engine,store,
        parse_answer=parser or (lambda text,forced:tuple(text.split('|',1))),
        prompt_builder=lambda context,forced:'synthetic-audio-prompt')


class VLLMAdapterTests(unittest.TestCase):
    def test_import_and_refused_real_constructor_never_import_GPU_libraries(self):
        code="import sys;sys.path.insert(0,%r);import vllm_backend;assert not {'torch','qwen_asr','vllm'}&set(sys.modules)"%str(HERE)
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
        with self.assertRaises(ValueError):V.build_real_engine(spec(),'/synthetic/store',execute=False)

    def test_native_reordered_outputs_map_by_actual_enqueue_ID(self):
        with tempfile.TemporaryDirectory() as d:
            e=FakeEngine('reordered');a=adapter(Path(d).resolve()/'store',e);r=a.run_batch(rows())
            self.assertEqual([x['raw_text'] for x in r['results']],['你好','世界'])
            self.assertNotEqual(r['results'][0]['request_id'],r['results'][0]['native']['request_id'])
            self.assertTrue(r['results'][0]['native']['eos_observed']);self.assertTrue(r['synthetic_fixture'])
            self.assertFalse(r['formal_admission']);self.assertEqual(r['metrics']['new_model_rows'],2)
            self.assertEqual(e.params[0]['max_tokens'],8);self.assertEqual(len(e.inputs[0]['multi_modal_data']['audio']),1)

    def test_native_missing_duplicate_foreign_or_invalid_shape_preserves_attempt(self):
        for mode in ('missing','duplicate','foreign','missing_finished','invalid_tokens'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve()/'store';a=adapter(root,FakeEngine(mode))
                with self.assertRaises(ValueError):a.run_batch(rows())
                evidence=list((root/'attempts').glob('*.json'));self.assertEqual(len(evidence),1)
                v=json.loads(evidence[0].read_bytes());self.assertFalse(v['stage_committed_before_error'])
                self.assertTrue(v['native_outputs'] or mode=='missing')
                if mode=='invalid_tokens':self.assertIn('nonfinite_native_float',evidence[0].read_text())

    def test_termination_and_language_are_separate_from_guessed_EOS(self):
        for mode,status,eos,verified in [('no_termination','unknown',False,False),
            ('stop_without_eos','completed',False,True),('arbitrary_stop','unknown',False,False),
            ('after_eos','unknown',True,False),('repeat_eos','unknown',True,False),
            ('truncated','truncated',False,False),('unsupported_language','unknown',True,True),('abort','failed',False,False)]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                r=adapter(Path(d).resolve()/'s',FakeEngine(mode)).run_batch(rows(1))['results'][0]
                self.assertEqual(r['status'],status);self.assertIs(r['native']['eos_observed'],eos)
                self.assertIs(r['termination_verified'],verified);self.assertFalse(r['usable_for_training'])

    def test_parser_failure_is_preserved_per_request_and_not_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            def parser(*a):raise RuntimeError('synthetic parser failure')
            r=adapter(Path(d).resolve()/'s',FakeEngine(),parser=parser).run_batch(rows())
            self.assertEqual(len(r['results']),2);self.assertTrue(all(x['status']=='failed' for x in r['results']))
            self.assertIn('synthetic parser failure',r['results'][0]['native_parser_error'])

    def test_all_cache_restore_is_zero_engine_calls_and_accepts_metadata_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'s';a=adapter(root,FakeEngine());inp=rows();first=a.run_batch(inp)
            e=FakeEngine('exception');restored=adapter(root,e);metadata=[{k:v for k,v in x.items() if k!='audio'} for x in inp]
            second=restored.run_batch(metadata)
            self.assertEqual(first['results'],second['results']);self.assertEqual(e.calls,0)
            self.assertEqual(second['metrics']['stage_cache_hits'],2);self.assertEqual(second['metrics']['new_model_rows'],0)
            self.assertEqual(second['metrics']['provided_PCM_verified_rows'],0)

    def test_interruption_after_whole_batch_commit_reuses_without_engine(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'s';a=adapter(root,FakeEngine())
            def interrupt(_):raise KeyboardInterrupt('after durable stage before per-source consumption')
            with self.assertRaises(KeyboardInterrupt):a.run_batch(rows(),after_durable_commit=interrupt)
            evidence=json.loads(next((root/'attempts').glob('*.json')).read_bytes())
            self.assertTrue(evidence['stage_committed_before_error'])
            e=FakeEngine('exception');r=adapter(root,e).run_batch(rows());self.assertEqual(e.calls,0)
            self.assertEqual(r['metrics']['new_batches'],0)

    def test_failed_uncommitted_cohort_retry_requires_full_original_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'s';a=adapter(root,FakeEngine('missing'));inp=rows()
            with self.assertRaises(ValueError):a.run_batch(inp)
            e=FakeEngine();b=adapter(root,e)
            with self.assertRaisesRegex(ValueError,'original complete'):b.run_batch(inp[:1])
            self.assertEqual(e.calls,0);r=b.run_batch(inp);self.assertEqual(e.calls,1)
            self.assertEqual(len(r['results']),2)

    def test_mixed_published_cache_and_new_request_preserves_old_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'s';a=adapter(root,FakeEngine());inp=rows(3);first=a.run_batch(inp[:2])
            e=FakeEngine();b=adapter(root,e);second=b.run_batch([inp[0],inp[2]])
            self.assertEqual(second['results'][0],first['results'][0]);self.assertEqual(second['metrics']['new_model_rows'],1)
            self.assertNotEqual(second['results'][0]['cohort_sha256'],second['results'][1]['cohort_sha256'])

    def test_changed_decode_or_identity_does_not_relabel_old_store(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'s';a=adapter(root,FakeEngine());a.run_batch(rows())
            old=(root/'identity.json').read_bytes();changed=spec();changed['decode']['max_tokens']=7
            with self.assertRaises(ValueError):adapter(root,FakeEngine(),changed)
            self.assertEqual((root/'identity.json').read_bytes(),old)

    def test_injected_engine_is_not_a_verified_real_factory_construct(self):
        with tempfile.TemporaryDirectory() as d:
            a=adapter(Path(d).resolve()/'s',FakeEngine())
            self.assertFalse(a.runtime_facts['real_engine_constructed'])
            self.assertTrue(a.runtime_facts['engine_object_injected'])

    def test_missing_existing_database_is_never_silently_recreated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'s';a=adapter(root,FakeEngine());a.run_batch(rows());old=(root/'identity.json').read_bytes()
            (root/'stage.sqlite3').unlink()
            with self.assertRaisesRegex(ValueError,'database missing'):adapter(root,FakeEngine())
            self.assertFalse((root/'stage.sqlite3').exists());self.assertEqual((root/'identity.json').read_bytes(),old)
            with self.assertRaisesRegex(ValueError,'database missing'):a.run_batch(rows())
            self.assertFalse((root/'stage.sqlite3').exists())

    def test_stored_batch_tampering_refuses_cache_without_recompute(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'s';a=adapter(root,FakeEngine());a.run_batch(rows())
            with sqlite3.connect(root/'stage.sqlite3') as con:con.execute("UPDATE batches SET result=?",(b'{}',))
            e=FakeEngine();b=adapter(root,e)
            with self.assertRaisesRegex(ValueError,'SHA'):b.run_batch(rows())
            self.assertEqual(e.calls,0)

    def test_duplicate_source_window_and_actual_PCM_mismatch_refused_before_model(self):
        with tempfile.TemporaryDirectory() as d:
            e=FakeEngine();a=adapter(Path(d).resolve()/'s',e);inp=rows()
            with self.assertRaises(ValueError):a.run_batch([inp[0],inp[0]])
            inp[0]['audio'][0]=100
            with self.assertRaisesRegex(ValueError,'PCM'):a.run_batch(inp)
            self.assertEqual(e.calls,0)

    def test_corrupt_PCM_is_refused_even_if_matching_claimed_hash_was_cached(self):
        with tempfile.TemporaryDirectory() as d:
            e=FakeEngine();a=adapter(Path(d).resolve()/'s',e);inp=rows();a.run_batch(inp);inp[0]['audio'][0]=100
            with self.assertRaisesRegex(ValueError,'PCM'):a.run_batch(inp)
            self.assertEqual(e.calls,1)

    def test_unknown_settings_and_unsupported_force_language_refused(self):
        for mode in ('decode','engine','language','API','batch'):
            s=spec()
            if mode=='decode':s['decode']['temperature']=.5
            elif mode=='engine':s['engine']['silently_ignored_flag']=True
            elif mode=='language':s['force_language']='Klingon'
            elif mode=='API':s['model_API_calls_allowed']=1
            else:s['max_batch_rows']=0
            with self.subTest(mode=mode),self.assertRaises(ValueError):V.validate_spec(s)
        for key in ('model_API_calls_allowed','tensor_parallel_size','n','window_policy_sha256'):
            s=spec()
            if key=='model_API_calls_allowed':s[key]=False
            elif key=='tensor_parallel_size':s['engine'][key]=True
            elif key=='n':s['decode'][key]=True
            else:s['preprocessing'][key]='g'*64
            with self.subTest(strict_type=key),self.assertRaises(ValueError):V.validate_spec(s)

    def test_no_positional_request_ID_fallback_for_old_engine_API(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'enqueue'):adapter(Path(d).resolve()/'s',object())

    def test_store_parent_symlink_and_nonempty_foreign_tree_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'actual').mkdir();(root/'alias').symlink_to(root/'actual',target_is_directory=True)
            with self.assertRaises(ValueError):ASRStageStore(root/'alias/s',digest(spec()),synthetic_fixture=True)
            (root/'actual/old.json').write_text('{}')
            with self.assertRaises(ValueError):ASRStageStore(root/'actual',digest(spec()),synthetic_fixture=True)

    def test_model_snapshot_pins_all_actual_weights_not_just_config(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'config.json').write_text('{}');(root/'model.safetensors').write_bytes(b'synthetic-weights')
            pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            self.assertEqual(V.verify_local_snapshot({'path':str(root),'files_sha256':pins}),root)
            (root/'README.md').write_text('Not a runtime model input')
            self.assertEqual(V.verify_local_snapshot({'path':str(root),'files_sha256':pins}),root)
            with self.assertRaisesRegex(ValueError,'every actual'):
                V.verify_local_snapshot({'path':str(root),'files_sha256':{'config.json':pins['config.json']}})
            (root/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'pin differs'):
                V.verify_local_snapshot({'path':str(root),'files_sha256':pins})


if __name__=='__main__':unittest.main()
