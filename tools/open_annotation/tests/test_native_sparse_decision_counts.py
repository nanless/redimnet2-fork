"""真实冻结Counter接口回归；不补零/剪键、不读取真实媒体或生产数据。"""
import copy
from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import native_batch_admission as A
import native_batch_readback as B
from test_native_formal_admission import NativeFixture


class SparseDecisionTests(unittest.TestCase):
    def setUp(self):self.f=NativeFixture(self)
    def test_same_actual_875_2125_sparse_pattern_has_no_failed_zero_key(self):
        records={str(i):dict(decision=dict(state='candidate'if i<875 else'unknown'))for i in range(3000)}
        counts=B._sparse_decision_counts(records)
        self.assertEqual(counts,{'candidate':875,'unknown':2125});self.assertNotIn('failed',counts)
        self.assertEqual(counts,dict(Counter(r['decision']['state']for r in records.values())))
    def test_all_three_present_states_counted_and_unknown_states_never_dropped(self):
        records={str(i):dict(decision=dict(state=state))for i,state in enumerate(['candidate','failed','unknown','failed'])}
        self.assertEqual(B._sparse_decision_counts(records),{'candidate':1,'failed':2,'unknown':1})
        for state in ('unregistered',0,False,None,[],{}):
            changed=copy.deepcopy(records);changed['0']['decision']['state']=state
            with self.subTest(state=state),self.assertRaisesRegex(ValueError,'invalid native source decision'):B._sparse_decision_counts(changed)
    def test_actual_sparse_fixture_full_admission_passes_without_any_zero_keys(self):
        value=A.check_native_selected_pilot(self.f.pin,self.f.new,'cpu_fixture');self.assertEqual(value['schema'],A.SCHEMA)
        speed=A.pinned(self.f.speed);self.assertEqual(speed['baseline_decision_counts'],{'candidate':4});self.assertEqual(speed['candidate_decision_counts'],{'candidate':4})
    def mutate_counts(self,key,counts):
        speed=A.pinned(self.f.speed);speed[key]=counts;pin=self.f.put('counter-altered-comparison.json',speed);self.f.selected['comparison']=pin
        q=copy.deepcopy(self.f.quality);q['comparison']=pin;self.f.selected['quality_review']=self.f.put('counter-bound-quality.json',q);self.f.save()
    def test_missing_positive_state_extra_zero_wrong_count_or_unknown_key_refused(self):
        invalid=[{}, {'candidate':3}, {'candidate':5}, {'candidate':4,'failed':0}, {'candidate':4,'unregistered':0}, {'candidate':4,'unregistered':1}, {'candidate':4,'unknown':0}]
        for key in ('baseline_decision_counts','candidate_decision_counts'):
            for counts in invalid:
                self.mutate_counts(key,counts)
                with self.subTest(key=key,counts=counts),self.assertRaisesRegex(ValueError,'aggregate differs.*'+key):A.check_native_selected_pilot(self.f.pin,self.f.new,'cpu_fixture')
                # 下一反例恢复另一侧原稀疏聚合，不让前一失败掩盖当前侧。
                self.f.speed=self.f.put('clean-counter-speed.json',self.f.comparison('speed'));self.f.selected['comparison']=self.f.speed
    def test_counter_values_cannot_use_float_or_boolean_to_match_integer_equality(self):
        for counts in ({'candidate':4.0},{'candidate':True},{'candidate':0},{'candidate':False}):
            self.mutate_counts('baseline_decision_counts',counts)
            with self.subTest(counts=counts),self.assertRaisesRegex(ValueError,'aggregate differs.*baseline_decision_counts'):A.check_native_selected_pilot(self.f.pin,self.f.new,'cpu_fixture')
    def test_readback_track_sparse_counts_match_real_record_Counter(self):
        track=self.f.tracks['baseline'];value=B.verify_readback_receipt(self.f.readback['baseline'],track['pin'],track['value'],self.f.base,self.f.bsha,len(self.f.sources),'cpu_fixture')
        self.assertEqual(value['track']['decision_counts'],dict(Counter(r['decision']['state']for r in value['records'].values())))
        self.assertEqual(value['track']['decision_counts'],{'candidate':4})

if __name__=='__main__':unittest.main()
