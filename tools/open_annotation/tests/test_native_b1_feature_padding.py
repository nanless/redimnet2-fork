"""逐条官方B1声学前处理及真实Tensor合批反例；CPU，无模型权重。

本机缺torch时仅Tensor类明确跳过；服务器既有ASR环境以CUDA不可见执行
全部用例。合同测试始终运行，不把跳过解释成真实前处理通过。
"""
from __future__ import annotations
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import contracts as C
import native_b1_feature_padding as F
from freeze_contract import freeze,make_native_formal_contract
from test_maintenance_entries import external_contract
torch=None


def declared(width=4):
    c=external_contract();c['purpose']='cpu_fixture';c['runtime'].update(execution_policy=C.NATIVE_BATCH_POLICY,
        asr_batching=dict(policy='fixed_cohort_until_durable/v1',max_requests=width,ordering='frames_then_row_id_within_source_group/v1'),
        asr_compute={'dtype':'float32','attention':'sdpa'},asr_preprocessing=copy.deepcopy(F.DECLARATION));return c


def fixture_sources(root,base):
    """仅测试：写7个合成声明文件和假模型配置，原字典原地补全并返回bindings。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);model=root/'fake-model';model.mkdir()
    config=model/'preprocessor_config.json';config.write_text('{}\n')
    base['models']['asr']['path']=str(model);base['models']['asr']['files']['preprocessor_config.json']={'sha256':C.sha_file(config),'bytes':config.stat().st_size}
    bindings=dict(schema='annotation-official-b1-feature-sources/v1',files={},processor_config=F.file_ref(config))
    for role in F.SOURCE_ROLES:
        p=root/(role+'.py');p.write_text('# synthetic source declaration only\n');bindings['files'][role]=F.file_ref(p)
    base['source_files'].update({bindings['files'][r]['path']:bindings['files'][r]['sha256']for r in ('qwen_processor','sdk_normalizer')})
    return bindings


class FeatureContracts(unittest.TestCase):
    def test_new_policy_is_explicit16_sources_and_old_native_remains15(self):
        c=declared();self.assertEqual(C.asr_feature_profile(c),F.DECLARATION)
        self.assertEqual(len(C.engine_modules_for_contract(c)),16);self.assertIn('native_b1_feature_padding.py',C.engine_modules_for_contract(c))
        c['runtime'].pop('asr_preprocessing');self.assertIsNone(C.asr_feature_profile(c));self.assertEqual(len(C.engine_modules_for_contract(c)),15)
    def test_unknown_implicit_or_other_precision_refused(self):
        for mutation in ('old','none','extra','bf16'):
            c=declared()
            if mutation=='old':c['runtime']['execution_policy']='single_window_independent';c['runtime'].pop('asr_batching')
            elif mutation=='none':c['runtime']['asr_preprocessing']=None
            elif mutation=='extra':c['runtime']['asr_preprocessing']['silently_trim_tail']=True
            else:c['runtime']['asr_compute']['dtype']='bfloat16'
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):C.validate_execution_policy(c)
    def test_freeze_new_policy_only_pilot_then_explicit_disabled_formal_data_factory(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d).resolve();base=root/'base.json';base.write_bytes(C.canonical_bytes(external_contract()))
            value=external_contract();bindings=fixture_sources(root/'bindings',value);base.write_bytes(C.canonical_bytes(value))
            receipt=freeze(base,C.sha_file(base),root/'code','new-b1-feature',asr_compute={'dtype':'float32','attention':'sdpa'},
                asr_batching=declared()['runtime']['asr_batching'],asr_preprocessing=F.DECLARATION,asr_preprocessing_sources=bindings)
            self.assertEqual(set(receipt['contracts']),{'pilot_validation'});p=receipt['contracts']['pilot_validation'];pilot=C.read_contract(p['path'],p['sha256'])
            self.assertEqual(len(pilot['engine_source_files']),16)
            formal=make_native_formal_contract(p['path'],p['sha256'],root/'control/formal.json','new-formal-data-only')
            saved=C.read_contract(formal['new_contract']['path'],formal['new_contract']['sha256'])
            self.assertEqual(saved['runtime']['asr_preprocessing'],F.DECLARATION);self.assertFalse(formal['formal_runtime_admitted']);self.assertFalse(saved['native_formal']['default_enabled'])
    def test_length_formula_retains_b1_floor_not_batched_ceil(self):
        self.assertEqual(F.audio_placeholder_length(2508),326);self.assertEqual(F.audio_placeholder_length(2509),327)
        self.assertEqual(F.audio_placeholder_length(308),40);self.assertEqual(F.audio_placeholder_length(309),41)
    def test_profile_getter_does_not_open_hash_or_stat_seven_source_files_per_window(self):
        with tempfile.TemporaryDirectory()as d:
            c=declared();c['asr_preprocessing_sources']=fixture_sources(Path(d),c);c['purpose']='pilot_validation'
            with patch.object(Path,'open',side_effect=AssertionError('no per-window source IO')),\
                 patch.object(Path,'stat',side_effect=AssertionError('no per-window metadata IO')),\
                 patch.object(F,'file_ref',side_effect=AssertionError('no per-window source hash')):
                for _ in range(100):self.assertEqual(C.asr_feature_profile(c),F.DECLARATION)
    def test_cached_live_guard_only_stats_known_files_and_rejects_identity_changes(self):
        with tempfile.TemporaryDirectory()as d:
            c=declared();c['asr_preprocessing_sources']=fixture_sources(Path(d),c)
            def merge():pass
            def pad():pass
            extractor=SimpleNamespace(to_dict=lambda:{'sampling_rate':16000},pad=pad)
            raw=SimpleNamespace(feature_extractor=extractor,_merge_kwargs=merge)
            b=F.B1FeatureBatchProcessor.__new__(F.B1FeatureBatchProcessor);b.synthetic=False;b.contract=c;b.raw=raw
            b.batch_feature_type=list;b.normalizer=tuple;b._live_identity=tuple(id(x)for x in(type(raw),type(extractor),list,tuple,merge,pad))
            b._extractor_config={'sampling_rate':16000};refs=[*c['asr_preprocessing_sources']['files'].values(),c['asr_preprocessing_sources']['processor_config']]
            def stamp(p):
                s=Path(p).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            b._source_stats={ref['path']:stamp(ref['path'])for ref in refs};calls=[];original=Path.stat
            def counted(p,*args,**kwargs):calls.append(str(p));return original(p,*args,**kwargs)
            with patch.object(F,'file_ref',side_effect=AssertionError('no repeated source hash')),\
                 patch.object(F,'source_objects',side_effect=AssertionError('no repeated source import/inspect')),patch.object(Path,'stat',counted):
                for _ in range(20):b.verify_live_sources()
            self.assertEqual(len(calls),20*8)
            Path(refs[0]['path']).write_text('# evidence changed, never silently re-pin\n')
            with self.assertRaisesRegex(ValueError,'file identity changed'):b.verify_live_sources()


class FeatureTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch,WhisperFeatureExtractor,BatchFeature
        if os.environ.get('REDIMNET_B1_TENSOR_TESTS')!='1':raise unittest.SkipTest('real Tensor tests require explicit isolated CPU subprocess; no heavy imports during discover')
        import torch
        from transformers import WhisperFeatureExtractor
        from transformers.feature_extraction_utils import BatchFeature
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        if torch.cuda.is_initialized():raise AssertionError('CPU feature tests cannot use an initialized CUDA context')
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        self.extractor=WhisperFeatureExtractor(feature_size=128,sampling_rate=16000,n_fft=400,hop_length=160,dither=0.)
        class Raw:
            audio_token='<audio>'
            tokenizer=SimpleNamespace(padding_side='left',pad_token_id=0,convert_tokens_to_ids=lambda token:88)
            def __call__(owner,*,text,audio,**kwargs):
                if not len(text)==len(audio)==1:raise AssertionError('test raw processor must only receive one official B1 row')
                features=self.extractor(audio,sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='pt',device='cpu')
                n=F.audio_placeholder_length(int(features['attention_mask'].sum()))
                ids=torch.tensor([[10]+[88]*n+[11]],dtype=torch.int64)
                return BatchFeature(dict(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=features['input_features'],feature_attention_mask=features['attention_mask']))
            def batch_decode(owner,*a,**kw):return ['unchanged decode forwarding']
        self.raw=Raw();self.proxy=F.B1FeatureBatchProcessor(self.raw,torch_module=torch,normalize_audio=self.normalize,max_requests=8,batch_feature_type=BatchFeature,synthetic_fixture=True)
    @staticmethod
    def normalize(pair):
        x=pair[0].astype(np.float32);peak=float(np.max(np.abs(x)))
        if peak>1:x=x/peak
        return np.clip(x,-1.,1.)
    def rows(self,lengths=(16000,16128,49408,64128)):
        result=[]
        for i,length in enumerate(lengths):
            t=np.arange(length,dtype=np.float32)/16000;x=(.3*np.sin(2*np.pi*173*t)+.05*np.cos(2*np.pi*357*t)).astype(np.float32)
            result.append(dict(row_id=f'synthetic{i}:w',audio=x,window=dict(input_pcm_sha256=F.pcm_sha(x),model_input_frames=len(x),model_sample_rate=16000)))
        return result
    def call(self,rows):
        self.proxy.begin_cohort(rows)
        return self.proxy(text=['prompt']*len(rows),audio=[self.normalize((r['audio'],16000))for r in rows],return_tensors='pt',padding=True)
    def test_official_b1_values_masks_and_prompt_are_bitwise_preserved_for1_4_8(self):
        for width in (1,4,8):
            rows=self.rows(tuple(16000+128*i for i in range(width)));batch=self.call(rows).to('cpu').to(torch.float32)
            proofs=self.proxy.observe_generate(batch)
            for i,row in enumerate(rows):
                one=self.raw(text=['prompt'],audio=[row['audio']],return_tensors='pt',padding=True);fi=one['input_features'].shape[-1]
                self.assertTrue(torch.equal(batch['input_features'][i,:,:fi],one['input_features'][0]))
                self.assertEqual(proofs[i]['feature_valid_sha256'],F.feature_sha(one['input_features'][0]))
                self.assertEqual(proofs[i]['actual_generate_nonpad_prompt_token_ids'],one['input_ids'][0].tolist())
                self.assertTrue(proofs[i]['verified_from_actual_generate_tensors']);self.assertFalse(proofs[i]['human_gold'])
            self.assertFalse(torch.cuda.is_initialized())
    def test_sdk_normalized_pcm_bound_separately_and_no_source_frames_removed(self):
        rows=self.rows((16128,));rows[0]['audio']*=8;rows[0]['window']['input_pcm_sha256']=F.pcm_sha(rows[0]['audio'])
        b=self.call(rows);proof=self.proxy.observe_generate(b)[0]
        self.assertTrue(proof['SDK_normalization_changed_PCM']);self.assertEqual(proof['frames'],16128)
        self.assertNotEqual(proof['original_input_pcm_sha256'],proof['sdk_input_pcm_sha256'])
    def test_actual_generate_valid_mel_mutation_refused(self):
        b=self.call(self.rows());b['input_features'][0,0,0]+=1.
        with self.assertRaisesRegex(ValueError,'actual valid mel'):self.proxy.observe_generate(b)
    def test_actual_generate_mask_prompt_or_feature_padding_mutation_refused(self):
        for key in ('fmask','prompt','pad'):
            b=self.call(self.rows())
            if key=='fmask':b['feature_attention_mask'][0,-1]=1
            elif key=='prompt':b['input_ids'][0,-1]+=1
            else:b['input_features'][0,0,-1]=1
            with self.subTest(key=key),self.assertRaises(ValueError):self.proxy.observe_generate(b)
    def test_sdk_split_processor_or_repeated_generate_is_refused(self):
        rows=self.rows();b=self.call(rows)
        with self.assertRaisesRegex(ValueError,'split/repeated'):self.proxy(text=['prompt'],audio=[rows[0]['audio']],return_tensors='pt',padding=True)
        b=self.call(rows);self.proxy.observe_generate(b)
        with self.assertRaisesRegex(ValueError,'exactly one'):self.proxy.observe_generate(b)
    def test_changed_sdk_audio_order_original_pcm_or_unrecognized_kwargs_refused(self):
        rows=self.rows();self.proxy.begin_cohort(rows)
        with self.assertRaisesRegex(ValueError,'normalized PCM/order'):self.proxy(text=['p']*4,audio=list(reversed([r['audio']for r in rows])),return_tensors='pt',padding=True)
        rows[0]['audio'][0]+=1
        with self.assertRaisesRegex(ValueError,'original window'):self.proxy.begin_cohort(rows)
        rows=self.rows();self.proxy.begin_cohort(rows)
        with self.assertRaisesRegex(ValueError,'kwargs changed'):self.proxy(text=['p']*4,audio=[r['audio']for r in rows],return_tensors='pt',padding=True,truncation=True)


if __name__=='__main__':unittest.main()
