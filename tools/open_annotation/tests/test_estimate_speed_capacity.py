import copy
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from estimate_speed_capacity import estimate


def fixture():
    return {'schema':'annotation-existing-monitor-capacity/v1','observed_utc':'synthetic',
            'GPU_window_seconds':25,'utilization_window_seconds':38,
            'devices':[{'uuid':str(i),'total_mib':81920,'used_mib_latest':14579,
                        'used_mib_observed_min':14579,'used_mib_observed_max':14579,
                        'free_mib_latest':66649,'utilization_mean':38,
                        'compute_pids':[{'gpu_uuid':str(i),'used_mib':14570}]} for i in range(8)]}


class CapacityTests(unittest.TestCase):
    def test_realistic_memory_accounting_respects_unavailable_692_and_context9(self):
        r=estimate(fixture(),25000,10000000)
        self.assertEqual(r['non_instance_or_unavailable_mib'],701)
        self.assertEqual(r['static_memory_only_replicas_per_GPU'],5)
        self.assertEqual(r['guarded_memory_only_replicas_per_GPU'],4)
        self.assertFalse(r['speedup_measured']);self.assertEqual(r['extra_processes_started'],0)
        self.assertEqual(r['memory_scenarios'][3]['remaining_memory_mib_per_GPU'],22939)

    def test_missing_process_or_uuid_binding_is_not_capacity_evidence(self):
        for mode in ('empty','multiple','uuid'):
            x=fixture()
            if mode=='empty':x['devices'][0]['compute_pids']=[]
            elif mode=='multiple':x['devices'][0]['compute_pids']*=2
            else:x['devices'][0]['compute_pids'][0]['gpu_uuid']='different'
            with self.assertRaises(ValueError):estimate(x,25000,100)

    def test_observed_peak_increases_budget_and_reserve_cannot_add_capacity(self):
        x=fixture();x['devices'][0]['used_mib_observed_max']+=10000
        result=estimate(x,25000,100)
        self.assertLess(result['guarded_memory_only_replicas_per_GPU'],4)
        self.assertEqual(result['observed_single_process_mib_range'],[14570,14570])
        self.assertEqual(result['instance_memory_proxy_mib_range'],[14570,24570])
        self.assertEqual(estimate(fixture(),25000,100,reserve_mib=90000)['guarded_memory_only_replicas_per_GPU'],0)

    def test_bad_numeric_input_and_duplicate_GPU_refused(self):
        for rate in (0,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):estimate(fixture(),rate,100)
        x=fixture();x['devices'][0]['uuid']=x['devices'][1]['uuid']
        with self.assertRaises(ValueError):estimate(x,25000,100)
        for low,high in ((1.5,14579),(-1,14579),(14580,14579),(14579,81921)):
            x=fixture();x['devices'][0]['used_mib_observed_min']=low;x['devices'][0]['used_mib_observed_max']=high
            with self.assertRaises(ValueError):estimate(x,25000,100)

    def test_ETA_scenario_is_declared_assumption_not_utilization_speed(self):
        x=fixture();a=estimate(x,25000,12000000);x['devices'][0]['utilization_mean']=99
        b=estimate(x,25000,12000000)
        self.assertEqual(a['assumed_speedup_ETA'],b['assumed_speedup_ETA'])
        self.assertEqual(a['assumed_speedup_ETA'][2]['conditional_days'],10)
        self.assertIsNone(a['memory_scenarios'][1]['actual_speedup'])


if __name__=='__main__':unittest.main()
