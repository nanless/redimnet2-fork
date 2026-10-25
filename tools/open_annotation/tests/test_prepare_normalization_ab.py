"""Tiny synthetic provenance fixtures; no source audio or model is opened."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contracts as C
import freeze_contract as F
import prepare_normalization_ab as P
import schema as S
from source_clock import LEGACY_MAPPING
from normalization import POLICY,APOSTROPHE_PROFILE,normalize_with_audit
from sidecar import StageCache
from audio_windows import units_from_windows
from test_maintenance_entries import external_contract
from test_quality_schema import fixture
from test_ultra_observation import observation_fixture
from worker import _descriptor


class PrepareABTests(unittest.TestCase):
    def fixture(self,root):
        old=external_contract();old['version']='synthetic-old';old['runtime']['boundary_mapping']=LEGACY_MAPPING
        oldpath=root/'producer.json';oldpath.write_bytes(C.canonical_bytes(old));oldsha=C.sha_file(oldpath)
        fa=F.freeze(oldpath,oldsha,root/'code-A','synthetic-a',POLICY)
        fb=F.freeze(oldpath,oldsha,root/'code-B','synthetic-b',APOSTROPHE_PROFILE)
        args=list(fixture());s=args[0];s.update(dataset='cnceleb2',sidecar_relpath='cnceleb2/'+s['relative_path']+'.annotation.json',original_splits=['provider_train'])
        s['source_id']='src_'+hashlib.sha256((s['dataset']+'\0'+s['relative_path']).encode()).hexdigest()
        args[2]=_descriptor(old,oldsha,'6'*64,{})
        w=args[4][0];w.update(time_mapping=LEGACY_MAPPING,model_input_frames=32000)
        w['asr'].update(raw_text="I'm",normalized_text='I m',normalization_audit=normalize_with_audit("I'm"))
        w['asr']['native'].update(actual_generate_rows=1,model_training=False,inference_mode=True,grad_enabled=False,native_decoded_raw_answer="I'm")
        al=observation_fixture();words=['I','m'];al['target_text']='I m';al['encoded_target_words']=words[:]
        al['native_observation']['target_encoding'][0]['words']=words[:]
        for key in ('items','author_final_items'):
            for i,r in enumerate(al[key]):r['text']=words[i]
        for i,r in enumerate(al['items']):r.update(target_word_index=i,target_occurrence_index=0)
        w['alignment']=al;args[5],args[7]=units_from_windows([w],args[1]);args[6].update(job_key=C.job_key(s,oldsha),backend='qwen3-asr-plus-ultra-no-head')
        r=S.build_record(*args);S.assert_valid_record(r)
        side=root/'sidecars';side.mkdir();sp=side/'00.json';sp.write_bytes(C.canonical_bytes(r))
        entry={'source_id':s['source_id'],'job_key':args[6]['job_key'],'sidecar_relpath':s['sidecar_relpath'],
               'sidecar_sha256':C.sha_file(sp),'sidecar_bytes':sp.stat().st_size,'decision':r['decision']['state'],'commit_sha256':'9'*64}
        cp=root/'chunk.json';cp.write_bytes(C.canonical_bytes({'schema':'redimnet2-chunk-commit/v1','contract_sha256':oldsha,'source_count':1,'sources':[entry]}))
        lp=root/'samples.json';lp.write_bytes(C.canonical_bytes({'sample_count':1,'rows':[{'source_id':s['source_id'],'sidecar_sha256':C.sha_file(sp)}]}))
        return {'sample_list':lp,'sample_list_sha':C.sha_file(lp),'sidecar_root':side,'chunk_commit':cp,'chunk_commit_sha':C.sha_file(cp),
                'producer_contract':oldpath,'producer_contract_sha':oldsha,'contract_a':Path(fa['contracts']['pilot_validation']['path']),
                'contract_a_sha':fa['contracts']['pilot_validation']['sha256'],'contract_b':Path(fb['contracts']['pilot_validation']['path']),
                'contract_b_sha':fb['contracts']['pilot_validation']['sha256'],'output':root/'prepared'},r

    def test_seeds_only_new_contract_ASR_and_preserves_native_sources(self):
        with tempfile.TemporaryDirectory()as d:
            a,r=self.fixture(Path(d));before={p:C.sha_file(p)for p in Path(d).rglob('*')if p.is_file()}
            with patch.object(P,'read_contract',wraps=P.read_contract)as read:result=P.prepare(**a)
            self.assertEqual(read.call_count,2);self.assertEqual(result['source_count'],1)
            self.assertEqual(before,{p:C.sha_file(p)for p in before})
            self.assertEqual(json.loads((a['output']/'sources.jsonl').read_bytes()),r['source'])
            for name,text in (('A','I m'),('B',"I'm")):
                track=result['tracks'][name];entry=track['ASR_only_seeds'][0];value=json.loads(Path(entry['cache_path']).read_bytes())
                self.assertEqual(value['inputs']['contract_sha256'],track['contract_sha256'])
                self.assertEqual(value['output']['native'],r['windows'][0]['asr']['native'])
                self.assertEqual(value['output']['normalized_text'],text)
                self.assertFalse(value['output']['reused_ASR_evidence']['new_ASR_model_execution'])
                self.assertTrue(value['output']['normalization_audit']['spoken_form_review_required'])
                self.assertEqual(track['aligner_cache_entries'],0)
                self.assertFalse(Path(track['queue_root']).exists())
            self.assertFalse(result['training_or_evaluation_admission']);self.assertFalse(result['GPU_or_model_execution'])

    def test_wrong_SHA_foreign_job_or_truncated_ASR_refused_before_output(self):
        for mode in ('sha','job','truncated','denominator','duplicate'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as d:
                a,r=self.fixture(Path(d))
                if mode=='sha':a['sample_list_sha']='0'*64
                if mode=='job':
                    c=json.loads(a['chunk_commit'].read_bytes());c['sources'][0]['job_key']='f'*64
                    a['chunk_commit'].write_bytes(C.canonical_bytes(c));a['chunk_commit_sha']=C.sha_file(a['chunk_commit'])
                if mode=='truncated':
                    r['windows'][0]['asr']['status']='truncated';r['windows'][0]['asr']['native']['eos_observed']=False
                    # Deliberately replace even all external pins; semantic schema/
                    # observed-stage validation must still refuse this non-complete input.
                    sp=a['sidecar_root']/'00.json';sp.write_bytes(C.canonical_bytes(r));lp=json.loads(a['sample_list'].read_bytes());lp['rows'][0]['sidecar_sha256']=C.sha_file(sp)
                    a['sample_list'].write_bytes(C.canonical_bytes(lp));a['sample_list_sha']=C.sha_file(a['sample_list'])
                if mode in ('denominator','duplicate'):
                    lp=json.loads(a['sample_list'].read_bytes());lp['sample_count']=2
                    if mode=='duplicate':lp['rows']*=2
                    a['sample_list'].write_bytes(C.canonical_bytes(lp));a['sample_list_sha']=C.sha_file(a['sample_list'])
                with self.assertRaises((RuntimeError,ValueError)):P.prepare(**a)
                self.assertFalse(a['output'].exists())

    def test_contract_difference_and_silver_purpose_refused(self):
        with tempfile.TemporaryDirectory()as d:
            a,_=self.fixture(Path(d));ca=json.loads(a['contract_a'].read_bytes());cb=json.loads(a['contract_b'].read_bytes());old=json.loads(a['producer_contract'].read_bytes())
            for change in ({'purpose':'silver_annotation'},{'runtime':{**cb['runtime'],'max_new_tokens':999}}):
                bad={**deepcopy(cb),**change}
                with self.assertRaises(RuntimeError):P.comparable_contracts(ca,bad,old)

    def test_existing_or_dangling_output_is_never_appended_or_followed(self):
        for symlink in (False,True):
            with self.subTest(symlink=symlink),tempfile.TemporaryDirectory()as d:
                a,_=self.fixture(Path(d));target=Path(d)/'absent'
                if symlink:a['output'].symlink_to(target,target_is_directory=True)
                else:a['output'].mkdir();(a['output']/'preserve').write_text('unchanged')
                with self.assertRaises(RuntimeError):P.prepare(**a)
                self.assertFalse(target.exists())
                if not symlink:self.assertEqual((a['output']/'preserve').read_text(),'unchanged')


if __name__=='__main__':unittest.main()
