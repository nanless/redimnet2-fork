"""公开合成夹具；不含真实文字、诊断名单、音频或运行记录。"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import word_adapter as A


class WordAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        word = HERE.parents[1]/'word_ft/schema.py'
        cls.C = {'word':A.load_file_module(word, '_public_word_schema'),
                 'contract_sha':'a'*64,'bundle_sha':'b'*64,'sidecar_sha':'c'*64,'commit_sha':'d'*64}
        reference = {'schema':'ultra-diagnostic-protection-reference/v1','case_count':1,
                     'three_training_source_count':1,'cases':[{'case_id':'SYNTHETIC_DIAGNOSTIC',
                     'source_id':'src_'+'f'*64,'path':'/synthetic/diagnostic.wav',
                     'normalized_alias_path':'/synthetic-alias/diagnostic.wav',
                     'in_three_training_registry':True,'benchmark_diagnostic_only_not_training_release':True}]}
        cls.C.update(diagnostic_protection=reference,diagnostic_protection_sha256='e'*64,
                     diagnostic_protection_content_sha256=A.digest(reference))

    def source(self, folder):
        path=Path(folder)/'SYNTHETIC_ZERO_PCM_NOT_SPEECH.wav';sr=16000
        with wave.open(str(path),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(b'\0\0'*sr)
        identity={'file_sha256':A.file_sha(path),'pcm_sha256':hashlib.sha256(struct.pack('<f',0)*sr).hexdigest(),
                  'sample_rate':sr,'channels':1,'frames':sr}
        text=['哈','哈','甲','乙','丙','丁'];seen={};units=[]
        for i,t in enumerate(text):
            n=seen.get(t,0);seen[t]=n+1
            units.append({'text':t,'window_id':'w0','native_index':i,'occurrence_index':n,
                          'sample_rate':sr,'start_sample':i*1280,'end_sample':(i+1)*1280})
        source={'source_id':'src_'+'1'*64,'dataset':'cnceleb2','relative_path':path.name,
                'audio_path':str(path),'roles':['train_ptn'],'original_splits':['provider_train'],
                'speaker_evidence':{'provider_label':'SYNTHETIC_PROVIDER','verified_identity':False},
                'protection':{'training_forbidden':False,'reasons':[]},'synthetic_fixture_not_model_output':True}
        native=[x for i in range(6)for x in [i*80,(i+1)*80]]
        return {'source':source,'source_identity':identity,'decision':{'state':'candidate','reasons':[],
                'usable_for_training':False,'usable_for_evaluation':False,'human_review':None},'units':units,
                'windows':[{'window_id':'w0','input_pcm_sha256':identity['pcm_sha256'],
                'asr':{'raw_text':''.join(text),'normalized_text':''.join(text),'language':'Chinese'},
                'alignment':{'native_timestamps_ms':native,'fixed_timestamps_ms':native[:],
                             'repair_changed':False}}]},path

    def test_prefix_occurrences_clock_and_original_layers_preserved(self):
        with tempfile.TemporaryDirectory()as d:
            record,_=self.source(d);before=A.canonical(record);p=A.propose(record,self.C)
            rows=p['families'][0]['records_pending_media_binding'];A.validate_family(rows)
            self.assertEqual([x['n_units']for x in rows],[1,2,3,6])
            self.assertEqual([x['occurrence_index']for x in rows[1]['alignment']['native_unit_refs']],[0,1])
            self.assertEqual(A.canonical(record),before)
            self.assertTrue(all(x['human_review']is None and not x['usable_for_training']and x['source_session_id']is None for x in rows))

    def test_unknown_language_coarse_group_and_missing_units_stay_diagnostic(self):
        with tempfile.TemporaryDirectory()as d:
            r,_=self.source(d)
            variants=[]
            x=deepcopy(r);x['decision'].update(state='unknown',reasons=['native_repair']);variants.append(x)
            x=deepcopy(r);x['windows'][0]['asr']['language']='UNSUPPORTED';variants.append(x)
            x=deepcopy(r);x['units'][0]['text']='两个';variants.append(x)
            x=deepcopy(r);x['units']=x['units'][:5];variants.append(x)
            x=deepcopy(r);x['units'][2]['native_index']=8;variants.append(x)
            for x in variants:self.assertEqual(A.propose(x,self.C)['families'],[])

    def test_external_exact_protection_and_missing_or_mutated_reference(self):
        with tempfile.TemporaryDirectory()as d:
            r,_=self.source(d);case=self.C['diagnostic_protection']['cases'][0]
            for key,value in [('source_id',case['source_id']),('audio_path',case['path']),('audio_path',case['normalized_alias_path'])]:
                x=deepcopy(r);x['source'][key]=value
                self.assertEqual(A.propose(x,self.C)['families'],[])
            missing=dict(self.C);missing.pop('diagnostic_protection')
            with self.assertRaisesRegex(ValueError,'protection reference'):A.propose(r,missing)
            altered=deepcopy({k:v for k,v in self.C.items()if k!='word'});altered['word']=self.C['word']
            altered['diagnostic_protection']['cases']=[]
            with self.assertRaisesRegex(ValueError,'protection reference'):A.propose(r,altered)

    def test_exact_media_candidate_loader_and_actual_train_guard(self):
        with tempfile.TemporaryDirectory()as d:
            r,path=self.source(d);p=A.propose(r,self.C);records,refused=A.engineer_export([p],{p['source_id']:str(path)},self.C)
            self.assertFalse(refused);self.assertEqual(len(records),4)
            dataset=self.C['word'].seal_dataset(records,{'split':A.ENGINEERING_SPLIT,'synthetic_fixture':True})
            target=Path(d)/'candidate.json';target.write_text(json.dumps(dataset))
            self.C['word'].load_dataset(target,verify=False,allow_candidates=True)
            with self.assertRaisesRegex(ValueError,'candidate'):self.C['word'].load_dataset(target,verify=False)
            train=HERE.parents[1]/'word_ft_v3/train.py';tree=ast.parse(train.read_text())
            guard=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and any(isinstance(x,ast.Constant)and x.value=='only train split is allowed'for x in ast.walk(n)))
            code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),str(train),'exec')
            with self.assertRaisesRegex(ValueError,'only train split'):exec(code,{'data':dataset})

    def test_changed_media_missing_map_eligibility_and_diagnostic_rechecked(self):
        with tempfile.TemporaryDirectory()as d:
            r,path=self.source(d);p=A.propose(r,self.C)
            self.assertTrue(A.engineer_export([p],{},self.C)[1])
            for key,value in [('split','train'),('usable_for_training',True),('evidence','human_audited_span')]:
                x=deepcopy(p);x['families'][0]['records_pending_media_binding'][0][key]=value
                self.assertTrue(A.engineer_export([x],{x['source_id']:str(path)},self.C)[1])
            x=deepcopy(p);x['source_original']['audio_path']='/synthetic/diagnostic.wav'
            self.assertIn('diagnostic_only',A.engineer_export([x],{x['source_id']:str(path)},self.C)[1][0]['reason'])
            path.write_bytes(path.read_bytes()+b'changed')
            self.assertIn('encoded media SHA differs',A.engineer_export([p],{p['source_id']:str(path)},self.C)[1][0]['reason'])

    def test_explicit_export_failure_and_default_inspection_returncode(self):
        # Mock publication authenticity only; run orchestration is exercised on
        # synthetic records. Publication/fencing has independent filesystem tests.
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);record,_=self.source(d);select=root/'selection.json';select.write_text('{"records":[{}]}')
            media=root/'mapping.json';media.write_text('{"source_to_audio_path":{}}')
            context=dict(self.C,word_schema_sha='f'*64)
            for name,emit,unknown,failed in [('refused',True,False,True),('zero',True,True,True),('inspect',False,True,False)]:
                r=deepcopy(record)
                if unknown:r['decision'].update(state='unknown',reasons=['synthetic_unknown'])
                args=argparse.Namespace(selection=str(select),selection_sha=A.file_sha(select),max_sources=1,
                    output=str(root/name),emit_engineering_candidates=emit,audio_bindings=str(media),audio_bindings_sha=A.file_sha(media))
                with patch.object(A,'load_context',return_value=context),patch.object(A,'verify_publication',return_value=(r,b'{}',b'{}','a'*64,'b'*64)):
                    _,actual=A.run(args)
                self.assertIs(actual,failed)
                if emit:self.assertFalse(json.loads((root/name/'engineering_export_status.json').read_bytes())['conversion_succeeded'])

    def test_no_external_reference_refused_before_source_or_word_import(self):
        with self.assertRaisesRegex(ValueError,'explicit external diagnostic'):
            A.load_context(argparse.Namespace(producer_code=str(HERE)))

    def test_frozen_copy_can_write_sibling_output_without_invented_repository_root(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);code=root/'task/code';code.mkdir(parents=True)
            file=code/'word_adapter.py';file.write_bytes(Path(A.__file__).read_bytes())
            frozen=A.load_file_module(file,'frozen_word_adapter')
            self.assertIsNone(frozen.REPOSITORY)
            r,_=self.source(d);r['decision'].update(state='unknown',reasons=['synthetic'])
            selection=root/'selection.json';selection.write_text('{"records":[{}]}')
            args=argparse.Namespace(selection=str(selection),selection_sha=A.file_sha(selection),max_sources=1,
                    output=str(root/'task/review'),emit_engineering_candidates=False)
            context=dict(self.C,word_schema_sha='f'*64)
            with patch.object(frozen,'load_context',return_value=context),patch.object(frozen,'verify_publication',return_value=(r,b'{}',b'{}','a'*64,'b'*64)):
                _,failed=frozen.run(args)
            self.assertFalse(failed);self.assertTrue((root/'task/review/proposals.json').is_file())


if __name__=='__main__':unittest.main()
