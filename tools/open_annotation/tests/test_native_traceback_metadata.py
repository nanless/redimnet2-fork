"""冻结入口路径差异的严格CPU处置；不删除异常、不发布unknown、不读真实媒体。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import native_batch_admission as A


class TracebackMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.roots={label:self.root/label for label in ('baseline','candidate')};members={}
        for root in self.roots.values():
            root.mkdir()
            for name in ('audio_windows.py','contracts.py','gpu_launch.py'):
                (root/name).write_text('# synthetic CPU control fixture\n');members[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        bundle=None
        for root in self.roots.values():bundle=self.put(root/'bundle_files.json',members)
        self.matrix=dict(variants={name:dict(freeze=dict(frozen_copy=str(self.roots[label]),bundle_sha256=bundle['sha256']))for label,name in [('baseline','native-fp32-sdpa-b1'),('candidate','native-fp32-sdpa-b8')]})
        self.context,self.refs=A._traceback_roots(self.matrix,'native-fp32-sdpa-b8',bundle['sha256'])
        self.row=dict(source_id='synthetic-source',baseline_state='unknown',candidate_state='unknown',source_identity_equal=True,baseline_window_count=1,candidate_window_count=1,changed_fields=['errors','vad'],windows=[self.window()],complete_source_observations={label:self.observations(root)for label,root in self.roots.items()})
        self.evidence=self.put(self.root/'review.json',dict(synthetic_CPU_only=True,human_gold=False))
        self.candidate=self.put(self.root/'contract.json',dict(synthetic_CPU_only=True))
        self.quality=dict(schema=A.QUALITY_SCHEMA,purpose='cpu_fixture',candidate_contract=self.candidate,review_complete=True,all_changed_rows_dispositioned=True,text_quality_accepted=True,complete_word_boundary_quality_accepted=True,candidate_labels_released=False,dispositions=[dict(source_id=self.row['source_id'],window_ids=[],classification=A.TRACEBACK_CLASS,review_evidence=[self.evidence])],review_evidence=[self.evidence])
    def put(self,path,value):
        path=Path(path);path.write_bytes(A.H.P.canonical(value));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    @staticmethod
    def window():
        asr={k+'_equal':True for k in ('status','raw_text','normalized_text','language','normalization_audit','effective_generated_token_ids','eos_observed','first_eos_index','max_new_tokens','preprocessing_effective_features')}
        alignment={k+'_equal':True for k in ('status','target_text','encoded_target_words','native_timestamps_ms','fixed_timestamps_ms','repair_changed','author_final_items','items')}
        return dict(window_id='w',both_present=True,input_identity_equal=True,ASR=asr,alignment=alignment)
    @staticmethod
    def traceback(root):
        return 'Traceback (most recent call last):\n  File "'+str(root)+'/audio_windows.py", line 132, in reuse_vad\n    need(valid, "invalid old VAD interval")\n  File "'+str(root)+'/contracts.py", line 75, in need\n    raise Refused(message)\ncontracts.Refused: invalid old VAD interval\n'
    def observations(self,root):
        error=dict(stage='reuse_vad',traceback=self.traceback(root))
        return dict(errors=[copy.deepcopy(error)],units=[dict(text='完整字',start_sample=0,end_sample=160)],decision=dict(state='unknown',human_review=None,reasons=['processing_errors_preserved'],usable_for_training=False,usable_for_evaluation=False),quality_axes=dict(human_review=None,boundaries='machine_pending'),vad=dict(status='identity_mismatch_or_invalid',reused=False,segments=[],errors=[copy.deepcopy(error)],original_sha256='a'*64))
    def check(self,row=None,quality=None,context='actual'):
        row=self.row if row is None else row;quality=self.quality if quality is None else quality
        return A.check_quality(self.put(self.root/'quality.json',quality),dict(rows=[row],source_count=1),self.candidate,'cpu_fixture',traceback_context=self.context if context=='actual'else context)
    def test_precise_class_preserves_original_errors_unknown_and_all_observations(self):
        before=copy.deepcopy(self.row)
        with patch('sqlite3.connect',side_effect=AssertionError('no DB')):self.assertTrue(self.check())
        self.assertEqual(before,self.row);self.assertFalse(A._metadata_only(self.row));self.assertEqual(self.row['candidate_state'],'unknown')
    def test_original_metadata_only_is_not_broadened(self):
        q=copy.deepcopy(self.quality);q['dispositions'][0]['classification']='metadata_only_verified'
        with self.assertRaisesRegex(ValueError,'padding-only metadata'):self.check(quality=q)
    def test_no_derived_context_or_manual_wrong_root_refused(self):
        with self.assertRaisesRegex(ValueError,'actual result/matrix'):self.check(context=None)
        ctx=copy.deepcopy(self.context);ctx['candidate']['root']=str(self.root/'wrong')
        with self.assertRaisesRegex(ValueError,'replacement must occur'):self.check(context=ctx)
    def test_bundle_member_must_exist_and_hash_matches_actual_frozen_bytes(self):
        (self.roots['baseline']/'contracts.py').write_text('# changed code')
        with self.assertRaises(ValueError):A._traceback_roots(self.matrix,'native-fp32-sdpa-b8',self.refs[0]['sha256'])
    def test_distinct_actual_roots_and_selected_freeze_bundle_required(self):
        ctx=copy.deepcopy(self.context);ctx['candidate']=ctx['baseline']
        with self.assertRaisesRegex(ValueError,'distinct actual'):self.check(context=ctx)
        matrix=copy.deepcopy(self.matrix);matrix['variants']['native-fp32-sdpa-b8']['freeze']['bundle_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'root/bundle differs'):A._traceback_roots(matrix,'native-fp32-sdpa-b8',self.refs[0]['sha256'])
    def change_traces(self,row,transform):
        obs=row['complete_source_observations']['candidate']
        for error in (obs['errors'][0],obs['vad']['errors'][0]):error['traceback']=transform(error['traceback'])
    def test_line_function_body_exception_and_member_changes_are_not_metadata(self):
        transforms=[lambda s:s.replace('line 132','line 133'),lambda s:s.replace('in reuse_vad','in different'),lambda s:s.replace('need(valid','need(False'),lambda s:s.replace('contracts.Refused:','ValueError:'),lambda s:s.replace('invalid old VAD interval','different failure'),lambda s:s.replace('/audio_windows.py','/contracts.py')]
        for change in transforms:
            row=copy.deepcopy(self.row);self.change_traces(row,change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'observations differ|traceback structure'):self.check(row)
    def test_only_actual_traceback_File_frames_in_declared_python_members_are_rewritten(self):
        for change,reason in [(lambda s:s.replace('/audio_windows.py','/unlisted.py'),'outside actual'),(lambda s:s.replace('File "','Path "'),'replacement must occur'),(lambda s:s.replace(str(self.roots['candidate'])+'/',str(self.roots['candidate'])+'-extra/'),'replacement must occur')]:
            row=copy.deepcopy(self.row);self.change_traces(row,change)
            with self.assertRaisesRegex(ValueError,reason):self.check(row)
    def test_changed_units_quality_VAD_bytes_or_decision_cannot_hide(self):
        for field in ('units','quality_axes','vad','decision'):
            row=copy.deepcopy(self.row);obs=row['complete_source_observations']['candidate']
            if field=='units':obs[field][0]['end_sample']+=1
            else:obs[field]['unrelated_change']=True
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'observations differ'):self.check(row)
    def test_unknown_cannot_be_released_or_treated_as_human_gold(self):
        for key,val in [('state','candidate'),('usable_for_training',True),('usable_for_evaluation',True),('human_review',{'gold':True})]:
            row=copy.deepcopy(self.row);row['complete_source_observations']['candidate']['decision'][key]=val
            with self.assertRaisesRegex(ValueError,'must remain unreleased'):self.check(row)
    def test_extra_changed_field_identity_state_or_window_count_refused(self):
        mutations=[lambda r:r['changed_fields'].append('units'),lambda r:r.update(source_identity_equal=False),lambda r:r.update(candidate_state='candidate'),lambda r:r.update(candidate_window_count=2)]
        for change in mutations:
            row=copy.deepcopy(self.row);change(row)
            with self.assertRaises(ValueError):self.check(row)
    def test_both_window_counts_require_actual_integers_not_boolean_or_float(self):
        for key in ('baseline_window_count','candidate_window_count'):
            for value in (True,1.0):
                row=copy.deepcopy(self.row);row[key]=value
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'text/input/token/boundary'):self.check(row)
    def test_window_presence_and_input_identity_require_actual_True_not_numeric_truthiness(self):
        for key in ('both_present','input_identity_equal'):
            for value in (1,0):
                row=copy.deepcopy(self.row);row['windows'][0][key]=value;q=copy.deepcopy(self.quality);q['dispositions'][0]['window_ids']=sorted(A._changed_windows(row))
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'text/input/token/boundary'):self.check(row,quality=q)
    def test_both_sides_same_fake_traceback_header_or_other_exception_refused(self):
        for transform in (lambda text:text.replace('Traceback (most recent call last):','ordinary prose'),lambda text:text.replace('contracts.Refused: invalid old VAD interval','ValueError: unrelated failure')):
            row=copy.deepcopy(self.row)
            for observation in row['complete_source_observations'].values():
                for error in (observation['errors'][0],observation['vad']['errors'][0]):error['traceback']=transform(error['traceback'])
            with self.assertRaisesRegex(ValueError,'traceback structure'):self.check(row)
    def test_text_effective_token_feature_boundary_or_geometry_difference_refused(self):
        for group,key in [('ASR','raw_text_equal'),('ASR','effective_generated_token_ids_equal'),('ASR','preprocessing_effective_features_equal'),('alignment','items_equal')]:
            row=copy.deepcopy(self.row);row['windows'][0][group][key]=False
            # 变化窗口仍须逐项列出，不能靠覆盖计数先拒绝掩盖真正分类条件。
            q=copy.deepcopy(self.quality);q['dispositions'][0]['window_ids']=['w']
            with self.assertRaisesRegex(ValueError,'text/input/token/boundary'):self.check(row,quality=q)
        row=copy.deepcopy(self.row);row['windows'][0]['input_identity_equal']=False;q=copy.deepcopy(self.quality);q['dispositions'][0]['window_ids']=['w']
        with self.assertRaisesRegex(ValueError,'text/input/token/boundary'):self.check(row,quality=q)
    def test_first_EOS_limit_or_unknown_ASR_semantic_flags_cannot_differ_or_be_missing(self):
        for key in ('first_eos_index_equal','max_new_tokens_equal','unrecognized_semantics_equal'):
            row=copy.deepcopy(self.row);row['windows'][0]['ASR'][key]=False;q=copy.deepcopy(self.quality);q['dispositions'][0]['window_ids']=['w']
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'rectangular post-EOS'):self.check(row,quality=q)
        for key in ('first_eos_index_equal','max_new_tokens_equal'):
            row=copy.deepcopy(self.row);row['windows'][0]['ASR'].pop(key)
            with self.assertRaisesRegex(ValueError,'rectangular post-EOS'):self.check(row)
    def test_only_three_explicit_rectangular_padding_flags_may_differ(self):
        row=copy.deepcopy(self.row)
        for key in ('generated_token_ids_equal','post_eos_token_ids_equal','raw_returned_token_count_equal'):row['windows'][0]['ASR'][key]=False
        q=copy.deepcopy(self.quality);q['dispositions'][0]['window_ids']=['w'];self.check(row,quality=q)
    def test_missing_geometry_flag_and_more_than_exact_two_error_leaves_refused(self):
        row=copy.deepcopy(self.row);row['windows'][0]['alignment'].pop('items_equal')
        with self.assertRaisesRegex(ValueError,'complete unchanged alignment'):self.check(row)
        row=copy.deepcopy(self.row);row['complete_source_observations']['candidate']['errors'].append(copy.deepcopy(row['complete_source_observations']['candidate']['errors'][0]))
        with self.assertRaisesRegex(ValueError,'exact original reuse_vad'):self.check(row)
    def test_other_original_errors_and_reasons_are_preserved_not_dropped(self):
        row=copy.deepcopy(self.row)
        for observation in row['complete_source_observations'].values():
            observation['errors'].append(dict(stage='alignment',reason='invalid_native_geometry',details={'kept':True}))
            observation['decision']['reasons'].extend(['invalid_native_geometry','alignment_coverage'])
        original=copy.deepcopy(row);self.check(row);self.assertEqual(row,original)
        row['complete_source_observations']['candidate']['errors'][1]['details']['kept']=False
        with self.assertRaisesRegex(ValueError,'observations differ'):self.check(row)
        row=copy.deepcopy(original);row['complete_source_observations']['candidate']['decision']['reasons'].pop()
        with self.assertRaisesRegex(ValueError,'observations differ'):self.check(row)
    def test_only_reuse_VAD_traceback_is_normalized_other_error_paths_are_exact(self):
        row=copy.deepcopy(self.row)
        for label,observation in row['complete_source_observations'].items():
            observation['errors'].append(dict(stage='other_error',traceback=self.traceback(self.roots[label])))
        with self.assertRaisesRegex(ValueError,'observations differ'):self.check(row)
    def test_every_changed_source_still_requires_one_bound_disposition(self):
        q=copy.deepcopy(self.quality);q['dispositions'][0]['source_id']='another-source'
        with self.assertRaisesRegex(ValueError,'one explicit disposition'):self.check(quality=q)

    def test_full_admission_derives_roots_from_actual_matrix_and_rejects_forged_source_views(self):
        from test_native_formal_admission import NativeFixture
        f=NativeFixture(self);twin=f.f.root/'actual-baseline-freeze';twin.mkdir()
        (twin/'gpu_launch.py').write_bytes((f.code/'gpu_launch.py').read_bytes());(twin/'bundle_files.json').write_bytes(Path(f.bundle['path']).read_bytes())
        result=A.pinned(f.selected['pilot_result']);plan=A.pinned(result['plan']);matrix=A.pinned(plan['main_prepared'])
        matrix['variants']['native-fp32-sdpa-b1']['freeze']['frozen_copy']=str(twin)
        plan['main_prepared']=f.put('traceback-main-matrix.json',matrix);result['plan']=f.put('traceback-pilot-plan.json',plan)
        f.selected['pilot_result']=f.put('traceback-pilot-result.json',result)
        comparison=A.pinned(f.selected['comparison']);row=comparison['rows'][0]
        row.update(baseline_state='unknown',candidate_state='unknown',changed_fields=['errors','vad'])
        row['complete_source_observations']={label:self.observations(root)for label,root in [('baseline',twin),('candidate',f.code)]}
        for observation in row['complete_source_observations'].values():
            for error in (observation['errors'][0],observation['vad']['errors'][0]):error['traceback']=error['traceback'].replace('/audio_windows.py','/gpu_launch.py').replace('/contracts.py','/gpu_launch.py')
        f.selected['comparison']=f.put('forged-source-comparison.json',comparison)
        quality=copy.deepcopy(f.quality);quality['comparison']=f.selected['comparison'];quality['dispositions']=[dict(source_id=row['source_id'],window_ids=sorted(A._changed_windows(row)),classification=A.TRACEBACK_CLASS,review_evidence=quality['review_evidence'])]
        f.selected['quality_review']=f.put('precise-path-quality.json',quality);f.save()
        original=A._frozen_traceback_only;calls=[]
        def classify(actual,context):
            self.assertEqual(context['baseline']['root'],str(twin));self.assertEqual(context['candidate']['root'],str(f.code))
            value=original(actual,context);calls.append(value);return value
        with patch.object(A,'_frozen_traceback_only',side_effect=classify),self.assertRaisesRegex(ValueError,'not from sealed actual sidecars'):
            A.check_native_selected_pilot(f.pin,f.new,'cpu_fixture')
        self.assertEqual(calls,[True])

if __name__=='__main__':unittest.main()
