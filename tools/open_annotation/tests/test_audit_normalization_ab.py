"""Comparison invariants on synthetic observations, no external task evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit_normalization_ab as A
from contracts import digest
from normalization import POLICY,APOSTROPHE_PROFILE,normalize_with_audit
from source_clock import TIME_MAPPING
from test_ultra_observation import observation_fixture


def alignment(words):
    al=observation_fixture();n=len(words)
    al['items']=al['items'][:n];al['author_final_items']=al['author_final_items'][:n]
    al['encoded_target_words']=words[:];al['native_observation']['target_encoding'][0]['words']=words[:]
    al['target_text']=' '.join(words)
    for key in ('native_timestamps_ms','fixed_timestamps_ms'):al[key]=al[key][:2*n]
    f=al['native_observation']['thinker_forwards'][0]
    for key in ('raw_argmax_class_indices','raw_timestamps_ms','timestamp_slot_positions'):f[key]=f[key][:2*n]
    f['slot_logit_shape'][0]=2*n
    fix=al['native_observation']['FIX_calls'][0]
    for key in ('native_timestamps_ms','fixed_timestamps_ms'):fix[key]=fix[key][:2*n]
    for i,text in enumerate(words):
        al['items'][i].update(text=text,target_word_index=i,target_occurrence_index=0)
        al['author_final_items'][i]['text']=text
    al['inference_call']={'rows':1,'producer_row_ids':['synthetic:w0']}
    return al


def fixture():
    raw="I'm";asr={'status':'completed','raw_text':raw,'normalized_text':'I m','language':'English',
                  'native':{'generated_token_ids':[1,2],'eos_observed':True},'normalization_audit':normalize_with_audit(raw)}
    old={'window_id':'w0','context_start_sample':0,'context_end_sample':16000,'owned_start_sample':0,'owned_end_sample':16000,
         'input_pcm_sha256':'1'*64,'model_input_frames':16000,'model_sample_rate':16000,'resampling_policy':'synthetic',
         'asr':asr,'alignment':alignment(['I','m'])}
    aa,bb=deepcopy(old),deepcopy(old)
    for w,policy in ((aa,POLICY),(bb,APOSTROPHE_PROFILE)):
        w.update(time_mapping=TIME_MAPPING,asr_cache_hit=True,aligner_cache_hit=False)
        audit=normalize_with_audit(raw,policy);w['asr'].update(normalized_text=audit['normalized_text'],normalization_audit=audit,
            reused_ASR_evidence={'source_id':'synthetic','window_id':'w0','old_sidecar_sha256':'2'*64,
                'original_ASR_output_sha256':digest(asr),'original_native_sha256':digest(asr['native']),'new_ASR_model_execution':False})
    bb['alignment']=alignment(["I'm"])
    return old,aa,bb


class AuditABTests(unittest.TestCase):
    def test_raw_FIX_layers_have_separate_counts_and_denominators(self):
        g=A.geometry({'items':[{},{}],'repair_changed':True,'native_timestamps_ms':[0,80,40,120],
                      'fixed_timestamps_ms':[0,80,80,80]})
        self.assertEqual(g['raw_pairs'],2);self.assertEqual(g['raw_zero'],0)
        self.assertEqual(g['raw_adjacent_overlap'],1)
        self.assertEqual(g['FIX_zero'],1);self.assertEqual(g['FIX_adjacent_overlap'],0)
        self.assertEqual(g['repair_changed_windows'],1)

    def test_target_change_does_not_change_ASR_native_or_silently_compare_equal_units(self):
        old,aa,bb=fixture();before=deepcopy((old,aa,bb))
        r=A.compare_window(old,aa,bb,source_id='synthetic',old_sidecar_sha='2'*64)
        self.assertTrue(r['target_changed']);self.assertEqual(r['A']['geometry']['items'],2)
        self.assertEqual(r['B']['geometry']['items'],1)
        self.assertIn('word_internal_apostrophe_requires_review',r['B']['spoken_form_flags'])
        self.assertEqual((old,aa,bb),before)

    def test_native_cache_clock_PCM_or_actual_row_drift_refused(self):
        for mode in ('native','asr_hit','FA_hit','clock','PCM','source_origin','actual_rows'):
            with self.subTest(mode=mode):
                old,aa,bb=fixture()
                if mode=='native':bb['asr']['native']['generated_token_ids']=[9]
                if mode=='asr_hit':bb['asr_cache_hit']=False
                if mode=='FA_hit':bb['aligner_cache_hit']=True
                if mode=='clock':bb['time_mapping']='wrong'
                if mode=='PCM':bb['input_pcm_sha256']='3'*64
                if mode=='source_origin':bb['asr']['reused_ASR_evidence']['source_id']='foreign'
                if mode=='actual_rows':bb['alignment']['inference_call']['rows']=2
                with self.assertRaises(RuntimeError):A.compare_window(old,aa,bb,source_id='synthetic',old_sidecar_sha='2'*64)

    def test_raw_span_merge_uses_occurrences_and_never_splits_coarse_groups(self):
        old,aa,bb=fixture();r=A.compare_raw_text_spans(aa,bb)
        self.assertEqual(r['status'],'exact_raw_codepoint_span_groups')
        self.assertEqual(r['rows'][0]['A_indices'],[0,1]);self.assertTrue(r['rows'][0]['merged_A_fragments'])
        self.assertEqual((r['rows'][0]['raw_start'],r['rows'][0]['raw_end']),(0,3))
        totals=A.span_totals([r]);self.assertEqual(totals['same_raw_text_spans'],1)
        self.assertEqual(totals['merged_A_fragment_groups'],1)
        self.assertEqual(totals['maximum_absolute_FIX_endpoint_delta_ms'],160)
        self.assertFalse(totals['time_differences_are_accuracy_errors'])
        repeated={'asr':{'normalization_audit':normalize_with_audit('Bag Bag')},
                  'alignment':{'encoded_target_words':['Bag','Bag']}}
        spans=A.target_raw_spans(repeated['asr'],repeated['alignment'])
        self.assertEqual([(x['raw_start'],x['raw_end'])for x in spans],[(0,3),(4,7)])
        repeated['alignment']['encoded_target_words']=['Bag Bag']
        self.assertIsNone(A.target_raw_spans(repeated['asr'],repeated['alignment']))

    def test_report_refuses_existing_symlink_or_input_tree_before_analyzing(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);old=root/'old';old.mkdir();a=root/'a';a.mkdir();b=root/'b';b.mkdir()
            existing=root/'exists';existing.write_text('preserve');link=root/'link';link.symlink_to(root/'absent')
            for out in (existing,link,old/'report.json',a/'report.json',b/'report.json'):
                args=['--original-root',str(old),'--sample-list','unused','--preparation','unused',
                    '--preparation-sha','a'*64,'--contract-a','unused','--contract-a-sha','b'*64,
                    '--contract-b','unused','--contract-b-sha','c'*64,'--bundle-sha','d'*64,
                    '--run-a',str(a),'--run-b',str(b),'--output',str(out)]
                with self.subTest(out=out),patch.object(A,'analyze',side_effect=AssertionError('no input reads'))as read:
                    with self.assertRaises(RuntimeError):A.main(args)
                    read.assert_not_called()
            self.assertEqual(existing.read_text(),'preserve');self.assertFalse((root/'absent').exists())

    def test_one_raw_codepoint_expanding_to_multiple_words_is_not_split(self):
        audit=normalize_with_audit('\ufdfa')
        words=audit['normalized_text'].split();self.assertEqual(len(words),4)
        self.assertEqual(audit['mapping_status'],'exact_codepoint_half_open')
        observation={'asr':{'normalization_audit':audit},'alignment':{'encoded_target_words':words}}
        self.assertIsNone(A.target_raw_spans(observation['asr'],observation['alignment']))
        r=A.compare_raw_text_spans(observation,observation)
        self.assertEqual(r,{'status':'mapping_not_available_no_coarse_group_split','rows':[]})
        self.assertEqual(A.span_totals([r])['unmeasured_windows'],1)


if __name__=='__main__':unittest.main()
