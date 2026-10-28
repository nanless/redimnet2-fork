"""独立云两卡pilot发现的纯CPU反例；保留正式auto8和独占资源流程。"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import gpu_launch as G

class AutoTwoPilot(unittest.TestCase):
    def snapshot(self,n):
        return {'devices':[{'index':i,'uuid':'GPU-'+str(i)}for i in reversed(range(n))]}
    def test_auto2_pilot_discovers_real_UUIDs_in_index_order(self):
        self.assertEqual(G.selected_uuids(self.snapshot(2),'auto2','pilot_validation'),['GPU-0','GPU-1'])
    def test_auto2_rejects_formal_or_eight_visible_devices(self):
        for purpose,count in [('silver_annotation',2),('pilot_validation',8),('pilot_validation',1)]:
            with self.subTest(purpose=purpose,count=count),self.assertRaises(RuntimeError):
                G.selected_uuids(self.snapshot(count),'auto2',purpose)
    def test_auto8_keeps_exact_eight_requirement(self):
        self.assertEqual(len(G.selected_uuids(self.snapshot(8),'auto8','silver_annotation')),8)
        with self.assertRaises(RuntimeError):G.selected_uuids(self.snapshot(2),'auto8','silver_annotation')
    def test_explicit_development_pair_semantics_unchanged(self):
        self.assertEqual(G.selected_uuids(self.snapshot(8),'GPU-5,GPU-3','pilot_validation'),['GPU-5','GPU-3'])
        with self.assertRaises(RuntimeError):G.selected_uuids(self.snapshot(2),'GPU-1,GPU-1','pilot_validation')
    def test_formal_auto2_refused_before_model_read_or_GPU_snapshot(self):
        args=SimpleNamespace(max_runtime_seconds=60,bundle_sha='a'*64,contract='/not/read',contract_sha='b'*64,
                             shard_index='/not/read-index',index_sha='c'*64,target_uuids='auto2')
        with patch.object(G,'verify_bundle',return_value={}),patch.object(G,'sha',side_effect=lambda p:'b'*64 if p==args.contract else 'c'*64),patch.object(G,'read_contract',return_value={'purpose':'silver_annotation'}),patch.object(G,'model_facts')as models,patch.object(G.G,'capture_snapshot')as snap:
            with self.assertRaisesRegex(RuntimeError,'pilot_validation'):G.run(args)
            models.assert_not_called();snap.assert_not_called()
if __name__=='__main__':unittest.main()
