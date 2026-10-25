import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

P=Path(__file__).resolve().parents[1]/'scripts/launch_duration_two_gpu.py'
spec=importlib.util.spec_from_file_location('duration_resources',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ResourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.snapshot={'gpus':[{'index':i,'uuid':f'u{i}','free_mib':10000,'used_mib':70000} for i in range(2)],'processes':[{'gpu_uuid':'u1','pid':'2323291','process_name':'[Not Found]'}]}
    def test_namespace_invisible_occupant_requires_shared_authorization(self):
        with self.assertRaisesRegex(RuntimeError,'explicit'):
            m.validate_resources(self.snapshot,[0,1],3072,4096,False)
    def test_invisible_unattributed_memory_is_not_empty(self):
        self.snapshot['processes']=[]
        with self.assertRaisesRegex(RuntimeError,'occupied'):
            m.validate_resources(self.snapshot,[0,1],3072,4096,False)
    def test_shared_budget_keeps_reserve(self):
        self.assertEqual(len(m.validate_resources(self.snapshot,[0,1],3072,4096,True)),2)
        self.snapshot['gpus'][1]['free_mib']=7000
        with self.assertRaisesRegex(RuntimeError,'reserve'):
            m.validate_resources(self.snapshot,[0,1],3072,4096,True)
    def test_two_unique_devices_and_reserve_required(self):
        for devices in [[0,0],[0],[0,2]]:
            with self.assertRaises(ValueError):m.validate_resources(self.snapshot,devices,3072,4096,True)
        with self.assertRaises(ValueError):m.validate_resources(self.snapshot,[0,1],3072,100,True)
    def test_nvidia_csv_names_and_invisible_pid_retained(self):
        def run(cmd,**kw):
            if '--query-compute-apps=gpu_uuid,pid,used_memory,process_name' in cmd:return 'u1, 2323291, 71916, [Not Found]\n'
            return '0, u0, A800, 81920, 55657, 26263, 0\n1, u1, A800, 81920, 71925, 9995, 0\n'
        r=m.query_resources(run)
        self.assertEqual(r['processes'][0]['process_name'],'[Not Found]')
        self.assertEqual(r['gpus'][1]['free_mib'],9995)
    def test_launcher_leaves_fresh_output_creation_to_trainer(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['config.json','dataset.json','official.pt']:(root/name).write_text('{}')
            (root/'runtime_audio/lib').mkdir(parents=True)
            (root/'runtime_audio/lib/libsndfile.so.1').write_bytes(b'fixture')
            out=root/'cpu-fixture-unused-name'
            def fake_launch(cmd,**kwargs):
                self.assertFalse(out.exists())
                self.assertEqual(kwargs['env']['DURATION_FT_GPU_AUTHORIZED'],'1')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
                return 0
            with patch.object(m,'EXPERIMENT_ROOT',root),patch.object(m,'query_resources',return_value=self.snapshot),patch.object(m.subprocess,'call',side_effect=fake_launch),patch('builtins.print'):
                result=m.main(['--config',str(root/'config.json'),'--dataset',str(root/'dataset.json'),'--official',str(root/'official.pt'),'--output',str(out),'--gpu-memory-mib','3072','--allow-shared'])
            self.assertEqual(result,0)
            self.assertEqual(len(list((root/'resource-records').glob('*-after.json'))),1)
if __name__=='__main__':unittest.main()
