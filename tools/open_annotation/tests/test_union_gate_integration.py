"""旧发布、剩余准备、新发布、联合核对、唯一消费的CPU文件级集成。"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import cloud_gate
from union_consumer import UnionConsumer
from test_union_coverage import UnionFixture


class UnionGateIntegration(unittest.TestCase):
    def setUp(self):self.f=UnionFixture()
    def tearDown(self):self.f.close()

    def test_plan_keeps_full_denominator_and_never_grants_runtime(self):
        before=self.f.old_snapshot()
        r=cloud_gate.check_union_data(self.f.spec_path,self.f.spec_sha,'plan')
        self.assertEqual(before,self.f.old_snapshot())
        self.assertEqual(r['original_full_source_count'],6)
        self.assertEqual(r['retained_sources_not_scheduled_again'],3)
        self.assertEqual(r['remaining_source_count'],3)
        self.assertFalse(r['cloud_submission_admitted']);self.assertFalse(r['GPU_execution_admitted'])
        self.assertFalse(r['full_result_readback_verified']);self.assertTrue(r['synthetic_fixture'])
        self.assertNotIn('passed',r)  # 不冒充旧正式云任务闸门的成功格式。

    def test_complete_gate_to_unique_consumer_preserves_old_bytes(self):
        before=self.f.old_snapshot();self.f.complete()
        r=cloud_gate.check_union_data(self.f.spec_path,self.f.spec_sha,'complete',output_index=self.f.root/'consumer.sqlite')
        index=r['union_validation']['consumer_index']
        with UnionConsumer(self.f.spec_path,self.f.spec_sha,index['path'],index['sha256']) as consumer:
            selected=list(consumer.iter_selected_sources())
        self.assertEqual([v['producer_id']for v in selected],['old']*3+['new']*3)
        self.assertEqual(len({v['source_id']for v in selected}),6)
        self.assertTrue(r['full_result_readback_verified'])
        self.assertEqual(before,self.f.old_snapshot())
        self.assertTrue(all(v['record']['decision']['usable_for_training'] is False for v in selected))

    def test_partial_new_results_cannot_pass_final_gate(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            cloud_gate.check_union_data(self.f.spec_path,self.f.spec_sha,'complete')

    def test_cpu_union_cannot_satisfy_original_production_gate(self):
        with self.assertRaises((RuntimeError,KeyError,ValueError)):
            cloud_gate.verify(self.f.spec,cloud_gate.Reader(self.f.spec))

    def test_real_CLI_is_CPU_only_and_has_no_implicit_submit(self):
        done=subprocess.run([sys.executable,'-B',str(HERE/'cloud_gate.py'),
            '--spec',str(self.f.spec_path),'--spec-sha',self.f.spec_sha,'--union-mode','plan'],
            capture_output=True,text=True,check=True)
        r=json.loads(done.stdout)
        self.assertEqual(r['schema'],'annotation-union-data-readiness/v1')
        self.assertFalse(r['GPU_execution_admitted']);self.assertFalse(self.f.new_queue.exists())


if __name__=='__main__':unittest.main()
