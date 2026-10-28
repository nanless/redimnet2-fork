"""CPU封存控制器使用小合成脚本验证退出/超时证据；不跑真实3千条或GPU。"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_native_cpu_sealer as C


class CPUSealerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.code=self.root/'audit';self.code.mkdir();self.helper=self.root/'helper';self.helper.mkdir()
        bundle={}
        for name in ('gpu_launch.py','audit_native_transformers_batch.py','audit_speed_comparison.py'):
            p=self.code/name;p.write_text('# synthetic CPU control source\n');bundle[name]=C.sha_file(p)
        self.bundle=self.put(self.code/'bundle_files.json',bundle)
        self.track=self.put(self.root/'track-input'/'native-track.json',dict(schema='redimnet2-annotation-native-batch-track/v1',queue_root=str(self.root/'producer'/'queue'),output_root=str(self.root/'producer'/'labels'),resources_root=str(self.root/'producer'/'resources')))
        self.sealer=self.helper/'native_batch_readback.py'
        self.set_sealer("import argparse,json,os\np=argparse.ArgumentParser();p.add_argument('--track');p.add_argument('--track-sha');p.add_argument('--audit-code-root');p.add_argument('--audit-bundle-sha');p.add_argument('--output');a=p.parse_args()\nassert os.environ['CUDA_VISIBLE_DEVICES']==''\nassert os.environ['OMP_NUM_THREADS']=='1'\nassert 'PYTHONPATH'not in os.environ\nprint('synthetic CPU stdout')\nfrom pathlib import Path\nPath(a.output).write_text(json.dumps({'synthetic_CPU_only':True}))\n")
        self.plan=dict(schema=C.SCHEMA,sealer=self.filepin(self.sealer),track=self.track,audit_code_root=str(self.code),audit_bundle=self.bundle,execution_root=str(self.root/'execution'),deadline_seconds=3)
        self.save()
    def filepin(self,p):return {'path':str(p),'sha256':C.sha_file(p)}
    def put(self,path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return self.filepin(path)
    def set_sealer(self,text):self.sealer.write_text(text)
    def save(self):self.pin=self.put(self.root/'plan.json',self.plan)
    def test_default_check_does_not_create_any_output_or_process(self):
        with patch.object(C.subprocess,'Popen',side_effect=AssertionError('check may not spawn')):v=C.execute(self.pin)
        self.assertFalse(v['executed']);self.assertFalse(Path(self.plan['execution_root']).exists())
    def test_success_fixed_argv_environment_actual_RSS_and_output_SHA(self):
        with patch.dict(os.environ,PYTHONPATH='injected',CUDA_VISIBLE_DEVICES='SHOULD_BE_EMPTY'):
            v=C.execute(self.pin,allow=True)
        self.assertTrue(v['completed']);self.assertEqual(v['returncode'],0);self.assertGreater(v['peak_RSS_bytes'],0);self.assertTrue(v['child_reaped'])
        root=Path(self.plan['execution_root']);intent=json.loads((root/'intent.json').read_text());self.assertEqual(intent['argv'][0],sys.executable)
        self.assertEqual(intent['source_pins']['sealer'],self.plan['sealer']);self.assertEqual(v['sealed_output']['sha256'],C.sha_file(root/'sealed-readback.json'))
        self.assertIn('synthetic CPU stdout',(root/'stdout.log').read_text());self.assertEqual((root/'stderr.log').read_text(),'')
        meta=json.loads((root/'child-control.json').read_text());self.assertEqual(meta['CUDA_VISIBLE_DEVICES'],'')
        if hasattr(os,'sched_getaffinity'):self.assertEqual(len(meta['CPU_affinity']),1)
    def test_nonzero_exit_keeps_stdout_stderr_intent_and_receipt(self):
        self.set_sealer("import sys\nprint('before failure')\nprint('original failure',file=sys.stderr)\nraise RuntimeError('synthetic CPU error')\n");self.plan['sealer']=self.filepin(self.sealer);self.save()
        v=C.execute(self.pin,allow=True);self.assertFalse(v['completed']);self.assertEqual(v['returncode'],1)
        root=Path(self.plan['execution_root']);self.assertTrue((root/'intent.json').is_file());self.assertTrue((root/'receipt.json').is_file());self.assertIn('synthetic CPU error',(root/'stderr.log').read_text());self.assertGreater(v['peak_RSS_bytes'],0)
    def test_timeout_reaps_exact_child_and_keeps_receipt_without_output(self):
        self.set_sealer("import time\nprint('before timeout',flush=True)\ntime.sleep(30)\n");self.plan.update(sealer=self.filepin(self.sealer),deadline_seconds=.2);self.save()
        v=C.execute(self.pin,allow=True);self.assertTrue(v['timeout']);self.assertTrue(v['child_reaped']);self.assertFalse(v['completed']);self.assertLess(v['elapsed_seconds'],3);self.assertGreater(v['peak_RSS_bytes'],0)
        root=Path(self.plan['execution_root']);self.assertTrue((root/'receipt.json').is_file());self.assertIn('before timeout',(root/'stdout.log').read_text())
    def test_parent_TERM_enters_finally_and_retains_evidence(self):
        self.set_sealer("import os,signal,time\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(30)\n");self.plan['sealer']=self.filepin(self.sealer);self.save()
        v=C.execute(self.pin,allow=True);self.assertFalse(v['completed']);self.assertTrue(v['child_reaped']);self.assertIn('controller interrupted',v['exception']);self.assertGreater(v['peak_RSS_bytes'],0)
        self.assertTrue((Path(self.plan['execution_root'])/'receipt.json').is_file())
    def test_second_TERM_during_cleanup_cannot_skip_reap_or_receipt(self):
        import signal
        before={sig:signal.getsignal(sig)for sig in (signal.SIGTERM,signal.SIGINT)}
        self.set_sealer("import os,signal,time\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(30)\n");self.plan['sealer']=self.filepin(self.sealer);self.save()
        real_kill=os.kill;second=[]
        def interrupted_cleanup(pid,sig):
            if pid!=os.getpid()and sig==signal.SIGKILL:
                second.append(True);real_kill(os.getpid(),signal.SIGTERM)
            return real_kill(pid,sig)
        with patch.object(C.os,'kill',side_effect=interrupted_cleanup),patch.object(C.os,'pidfd_open',side_effect=OSError('exercise portable kill branch'),create=True):v=C.execute(self.pin,allow=True)
        self.assertTrue(second);self.assertTrue(v['child_reaped']);self.assertFalse(v['completed']);self.assertTrue((Path(self.plan['execution_root'])/'receipt.json').is_file())
        self.assertEqual({sig:signal.getsignal(sig)for sig in before},before)
    def test_spawn_exception_still_leaves_intent_and_receipt(self):
        with patch.object(C.subprocess,'Popen',side_effect=OSError('synthetic spawn failed')):v=C.execute(self.pin,allow=True)
        self.assertFalse(v['child_started']);self.assertIn('synthetic spawn failed',v['exception']);self.assertIsNone(v['peak_RSS_bytes']);self.assertTrue((Path(self.plan['execution_root'])/'receipt.json').is_file())
    def test_success_without_output_is_not_completed(self):
        self.set_sealer("print('missing output')\n");self.plan['sealer']=self.filepin(self.sealer);self.save();v=C.execute(self.pin,allow=True)
        self.assertEqual(v['returncode'],0);self.assertFalse(v['completed']);self.assertIsNone(v['sealed_output'])
    def test_existing_output_root_refused_without_overwrite(self):
        root=Path(self.plan['execution_root']);root.mkdir();(root/'old.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'fresh execution'):C.execute(self.pin,allow=True)
        self.assertEqual((root/'old.txt').read_text(),'preserve')
    def test_deadline_above_300_or_boolean_refused(self):
        for d in (301,True,0,float('inf')):
            self.plan['deadline_seconds']=d;self.save()
            with self.assertRaisesRegex(ValueError,'deadline'):C.check(self.pin)
    def test_hash_change_and_arbitrary_entry_refused(self):
        self.sealer.write_text('# changed')
        with self.assertRaisesRegex(ValueError,'SHA differs'):C.check(self.pin)
        other=self.helper/'arbitrary.py';other.write_text('# no');self.plan['sealer']=self.filepin(other);self.save()
        with self.assertRaisesRegex(ValueError,'only fixed'):C.check(self.pin)
    def test_bundle_math_change_is_not_silently_accepted(self):
        (self.code/'audit_native_transformers_batch.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'SHA differs'):C.check(self.pin)
    def test_output_inside_frozen_code_or_producer_refused(self):
        for root in (self.code/'new-output',self.root/'producer'/'labels'/'new-output'):
            root.parent.mkdir(parents=True,exist_ok=True);self.plan['execution_root']=str(root);self.save()
            with self.assertRaisesRegex(ValueError,'overlaps'):C.check(self.pin)
    def test_faulthandler_and_runpy_fixed_instead_of_shell(self):
        self.assertIn('dump_traceback_later(60,repeat=True)',C.CHILD_CODE);self.assertIn("runpy.run_path(sys.argv[0],run_name='__main__')",C.CHILD_CODE)
        self.assertNotIn('subprocess',C.CHILD_CODE)

if __name__=='__main__':unittest.main()
