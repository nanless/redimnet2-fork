"""真实小WAV的准备器CPU检查，不构造任何GPU模型或读取业务原声。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import prepare_vllm_live_inputs as P
from asr_stage_store import canonical
from test_vllm_backend import spec


class PreparationFixture:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.old=self.root/'old';self.old.mkdir();self.shards=self.old/'shards';self.shards.mkdir()
        self.contract={'release_training_labels':False,'runtime':{'max_new_tokens':12,
            'window':{'core_seconds':24,'context_seconds':2},'normalization':'old explanatory description'}}
        self.rows=[]
        for dataset in sorted(P.DOMAINS):
            rows=[]
            for i in range(2):
                audio=self.old/('%s-%s.wav'%(dataset,i))
                sf.write(audio,np.zeros(16000,dtype=np.float32),16000,subtype='FLOAT')
                rows.append({'source_id':'%s-%s'%(dataset,i),'dataset':dataset,'audio_path':str(audio),
                    'channel_policy':'mean_channels_to_mono','source_interval':None,
                    'protection':{'training_forbidden':False,'reasons':[]}})
            self.rows.append(rows)
        self.index={'source_count':11521178,'domains':sorted(P.DOMAINS),'shards':[]}
        self.refresh();self.describer_calls=[]
    def save(self,path,value,lines=False):
        path.write_bytes(b''.join(canonical(x) for x in value) if lines else canonical(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def refresh(self):
        self.index['shards']=[]
        for i,rows in enumerate(self.rows):
            p=self.shards/('%02d.jsonl'%i);h=self.save(p,rows,True)
            self.index['shards'].append({'path':str(p.relative_to(self.old)),'sha256':h,'count':len(rows)})
        self.index_sha=self.save(self.old/'index.json',self.index)
        self.contract_sha=self.save(self.old/'contract.json',self.contract)
    def describe(self,contract):
        self.describer_calls.append(copy.deepcopy(contract));return spec()
    def run(self,**changes):
        kwargs={'index_path':str(self.old/'index.json'),'index_sha':self.index_sha,
                'contract_path':str(self.old/'contract.json'),'contract_sha':self.contract_sha,
                'ordinals':[0,1,2],'output':str(self.root/'prepared'),'engine_describer':self.describe}
        kwargs.update(changes);return P.prepare(**kwargs)
    def hashes(self):return {str(p.relative_to(self.old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.old.rglob('*') if p.is_file()}
    def close(self):self.tmp.cleanup()


class PrepareLiveInputsTests(unittest.TestCase):
    def setUp(self):self.f=PreparationFixture()
    def tearDown(self):self.f.close()
    def test_real_whole_second_WAVs_six_unique_outputs_do_not_touch_originals(self):
        before=self.f.hashes();r=self.f.run();self.assertEqual(self.f.hashes(),before)
        self.assertEqual(r['source_count'],6);self.assertEqual(r['domain_counts'],{d:2 for d in P.DOMAINS})
        self.assertEqual(r['PCM_seconds'],6);self.assertFalse(r['GPU_or_model_executed']);self.assertFalse(r['training_or_evaluation_released'])
        rows=json.loads(Path(r['inputs']['path']).read_bytes());self.assertEqual(len({v['source_id'] for v in rows}),6)
        for row in rows:
            wave=np.load(row['pcm_path'],allow_pickle=False)
            self.assertEqual(len(wave),16000);self.assertEqual(wave.dtype,np.dtype('float32'))
            self.assertEqual(hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),row['input_pcm_sha256'])
            self.assertEqual(row['window']['context_start_sample'],0);self.assertEqual(row['window']['context_end_sample'],16000)
    def static_description(self,config,tokenizer_config=None,**optional_files):
        """模拟可静态读取的安装布局，不import模型包。"""
        root=Path(tempfile.mkdtemp(dir=self.f.root));packages={}
        for name in ('qwen_asr','vllm'):
            packages[name]=root/name;packages[name].mkdir()
        qwen=packages['qwen_asr'];(qwen/'inference').mkdir()
        (qwen/'inference/utils.py').write_text("SUPPORTED_LANGUAGES = ['Chinese', 'English']\n")
        (qwen/'inference/qwen3_asr.py').write_text('# static fake package source\n')
        engine=packages['vllm'];(engine/'entrypoints').mkdir()
        (engine/'entrypoints/llm.py').write_text('# static fake engine source\n')
        (engine/'sampling_params.py').write_text('# static fake sampling source\n')
        model=root/'model';model.mkdir();(model/'config.json').write_text(json.dumps(config))
        if tokenizer_config is not None:(model/'tokenizer_config.json').write_text(json.dumps(tokenizer_config))
        for name,value in optional_files.items():(model/name).write_text(json.dumps(value))
        (model/'model.safetensors').write_bytes(b'not real weights')
        contract=copy.deepcopy(self.f.contract);contract['models']={'asr':{'path':str(model),'files':{
            p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in model.iterdir()}}}
        found=lambda name:SimpleNamespace(submodule_search_locations=[str(packages[name])])
        with patch.object(P.importlib.util,'find_spec',side_effect=found),patch.object(P.md,'version',return_value='static-fixture'):
            return P.local_engine_description(contract)

    def test_installed_source_description_uses_effective_profile_not_legacy_explanation(self):
        actual=self.static_description({'eos_token_id':3})
        self.assertEqual(actual['normalization_profile'],P.POLICY)
        self.assertNotEqual(actual['normalization_profile'],self.f.contract['runtime']['normalization'])
        self.assertEqual(actual['eos_token_ids'],[3]);self.assertEqual(actual['supported_languages'],['Chinese','English'])
        self.assertEqual(len(actual['source_files']),4)

    def test_tokenizer_config_decoder_resolves_EOS_without_tokenizer_JSON(self):
        for token in ('<|im_end|>',{'content':'<|im_end|>','special':True}):
            with self.subTest(token=token):
                actual=self.static_description({}, {'eos_token':token,'added_tokens_decoder':{
                    '42':{'content':'<|im_end|>','special':True},'41':{'content':'<|im_start|>'}}})
                self.assertEqual(actual['eos_token_ids'],[42])
                self.assertNotIn('tokenizer.json',actual['model']['files_sha256'])

    def test_optional_tokenizer_files_must_agree_on_actual_EOS_ID(self):
        tc={'eos_token':'<|im_end|>','added_tokens_decoder':{'42':{'content':'<|im_end|>'}}}
        matching={'tokenizer.json':{'added_tokens':[{'id':42,'content':'<|im_end|>'}]},
                  'added_tokens.json':{'<|im_end|>':42}}
        self.assertEqual(self.static_description({},tc,**matching)['eos_token_ids'],[42])
        for conflict in ({'tokenizer.json':{'added_tokens':[{'id':43,'content':'<|im_end|>'}]}},
                         {'added_tokens.json':{'<|im_end|>':43}}):
            with self.subTest(file=next(iter(conflict))):
                with self.assertRaisesRegex(ValueError,'EOS token not uniquely identified'):
                    self.static_description({},tc,**conflict)

    def test_absent_or_ambiguous_EOS_rejected_without_default_guess(self):
        for decoder in ({},{'42':{'content':'<|im_end|>'},'43':{'content':'<|im_end|>'}}):
            with self.subTest(decoder=decoder):
                with self.assertRaisesRegex(ValueError,'EOS token not uniquely identified'):
                    self.static_description({}, {'eos_token':'<|im_end|>','added_tokens_decoder':decoder})

    def test_wrong_index_or_contract_hash_fails_before_output_or_describer(self):
        for field in ('index_sha','contract_sha'):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError,'SHA differs'):self.f.run(**{field:'0'*64})
                self.assertFalse((self.f.root/'prepared').exists());self.assertEqual(self.f.describer_calls,[])
    def test_explicit_distinct_ordinals_required(self):
        for value in ([0,0,2],[0,1],[-1,1,2],[True,1,2],[0,1,9]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError,'ordinals'):self.f.run(ordinals=value)
                self.assertFalse((self.f.root/'prepared').exists())
    def test_existing_output_or_original_input_overlap_refused_without_mutation(self):
        before=self.f.hashes()
        for path in (self.f.old,self.f.old/'new-probe',self.f.root):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):self.f.run(output=str(path))
        self.assertEqual(self.f.hashes(),before)
    def test_crop_or_protected_source_rejected_and_traceback_evidence_retained(self):
        for field in ('crop','protected'):
            with self.subTest(field=field):
                f=PreparationFixture()
                try:
                    if field=='crop':f.rows[0][0]['source_interval']={'start_sample':0,'end_sample':16000}
                    else:f.rows[0][0]['protection']['training_forbidden']=True
                    f.refresh();before=f.hashes()
                    with self.assertRaisesRegex(ValueError,'two whole utterances'):f.run()
                    rejection=json.loads((f.root/'prepared/selection-rejections.json').read_text())
                    self.assertEqual(len(rejection),1);self.assertIn('traceback',rejection[0]);self.assertEqual(f.hashes(),before)
                    self.assertFalse((f.root/'prepared/preparation.json').exists())
                finally:f.close()
    def test_subsecond_or_overlong_source_not_silently_cropped(self):
        for frames in (15999,320001):
            with self.subTest(frames=frames):
                f=PreparationFixture()
                try:
                    sf.write(f.rows[0][0]['audio_path'],np.zeros(frames,dtype=np.float32),16000,subtype='FLOAT')
                    with self.assertRaisesRegex(ValueError,'two whole utterances'):f.run()
                    rejection=json.loads((f.root/'prepared/selection-rejections.json').read_text())
                    self.assertIn('1--20',rejection[0]['error'])
                finally:f.close()
    def test_duplicate_ID_retains_error_then_selects_distinct_replacement(self):
        original=copy.deepcopy(self.f.rows[0][1]);self.f.rows[0][1]['source_id']=self.f.rows[0][0]['source_id']
        self.f.rows[0].append(original);self.f.refresh();r=self.f.run()
        rows=json.loads(Path(r['inputs']['path']).read_bytes());self.assertEqual(len({v['source_id'] for v in rows}),6)
        rejected=json.loads((self.f.root/'prepared/selection-rejections.json').read_text())
        self.assertEqual(len(rejected),1);self.assertIn('source',rejected[0]['error'])

    def test_empty_source_ID_preserves_rejection_and_cannot_release_success(self):
        self.f.rows[0][0]['source_id']='';self.f.refresh()
        with self.assertRaisesRegex(ValueError,'two whole utterances'):self.f.run()
        rejected=json.loads((self.f.root/'prepared/selection-rejections.json').read_text())
        self.assertEqual(len(rejected),1);self.assertIn('source',rejected[0]['error'])
        self.assertFalse((self.f.root/'prepared/preparation.json').exists())

    def test_shard_bytes_and_declared_count_are_checked(self):
        for field in ('sha','count'):
            with self.subTest(field=field):
                f=PreparationFixture()
                try:
                    f.index['shards'][0]['sha256' if field=='sha' else 'count']='0'*64 if field=='sha' else 3
                    f.index_sha=f.save(f.old/'index.json',f.index)
                    with self.assertRaisesRegex(ValueError,'shard SHA|row count'):f.run()
                finally:f.close()


if __name__=='__main__':unittest.main()
