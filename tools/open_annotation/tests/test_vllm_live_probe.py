"""有限真实实验助手的CPU反例；这里只注入假引擎，不运行GPU。"""
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
import vllm_live_probe as P
import vllm_backend as V
from asr_stage_store import ASRStageStore,canonical,digest
from test_vllm_backend import spec,FakeEngine


class Fixture:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.model=self.root/'model';self.model.mkdir()
        (self.model/'config.json').write_text('{}');(self.model/'model.safetensors').write_bytes(b'fake, not real weights')
        self.s=spec();self.s['max_batch_rows']=2;self.s['engine']['gpu_memory_utilization']=.1
        self.s['model']={'path':str(self.model),'files_sha256':{p.name:V._file_sha(p) for p in self.model.iterdir()}}
        self.items=[]
        for i in range(3):
            wave=np.array([i+1,.5],dtype=np.float32);p=self.root/('pcm-'+str(i)+'.npy');np.save(p,wave)
            self.items.append({'source_id':'s'+str(i),'window_id':'w0','pcm_path':str(p),'pcm_file_sha256':V._file_sha(p),
                'model_input_frames':len(wave),'model_sample_rate':16000,'input_pcm_sha256':hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()})
        self.config={'schema':P.SCHEMA,'model_API_calls_allowed':0,'release_training_labels':False,'wall_seconds':30,
                     'adapter_spec':self.save('adapter.json',self.s),'inputs':self.save('inputs.json',self.items),
                     'output_root':str(self.root/'output'),'protected_roots':[str(self.root/'old-production')],
                     'probe_source_files':{str(p):V._file_sha(p) for p in (Path(P.__file__),Path(V.__file__),HERE/'asr_stage_store.py')}}
        locks=self.root/'actual-exclusive-locks';locks.mkdir();lock=locks/'GPU-synthetic.lock';lock.touch();st=lock.stat()
        ticket={'schema':'open-annotation-resource-ticket/v1','target_uuid':'GPU-synthetic','bundle_sha256':'a'*64,
                'boot_id':'synthetic','locks':[{'uuid':'GPU-synthetic','path':str(lock),'device':st.st_dev,'inode':st.st_ino}]}
        self.grant={'schema':'annotation-idle-GPU-exclusive-admission/v1','idle_gpu_test_authorized':True,
            'shared_gpu_test_authorized':False,'driver_used_limit_mib':256,
            'existing_exclusive_ticket':self.save('existing-ticket.json',ticket),'exclusive_protocol_bundle_sha256':'a'*64,
            'task_id':'synthetic-task','hostname':socket.gethostname(),'gpu_uuid':'GPU-synthetic','headroom_mib':8192,
            'max_probe_memory_mib':10000,'allowed_existing_processes':[],
            'expires_utc':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)).isoformat(),
            'producer_sha256':digest(self.s),'inputs_sha256':self.config['inputs']['sha256']}
        self.config['probe_source_files'].update({str(HERE/n):V._file_sha(HERE/n) for n in ('gpu_launch.py','gpu_challenge.py','gpu_vendor/resources.py')})
        self.config['resource_admission']=self.save('grant.json',self.grant);self.ref=self.save('experiment.json',self.config)
        self.engines=[]
    def save(self,name,value):
        p=self.root/name;p.write_bytes(canonical(value));return {'path':str(p),'sha256':V._file_sha(p)}
    def update(self):self.ref=self.save('experiment.json',self.config)
    def snapshot(self,gpu):return {'gpu_uuid':gpu,'physical_index':3,'index_uuid_map':[{'gpu_uuid':gpu if i==3 else 'GPU-foreign-'+str(i),'physical_index':i,'pci_bus_id':'00000000:%02x:00.0'%(i+1)} for i in range(4)],'total_memory_mib':80000,'used_memory_mib':128,
        'hostname':socket.gethostname(),'observed_unix':time.time(),'processes':[]}
    def factory(self,s,root,execute):
        assert execute;engine=FakeEngine();self.engines.append(engine)
        return V.VLLMBackend.from_fake_engine(s,engine,ASRStageStore(root,digest(s),synthetic_fixture=True),
            parse_answer=lambda text,forced:tuple(text.split('|',1)),prompt_builder=lambda *_:'fixture')
    def run(self,**kw):return P.run_experiment(self.ref,factory=self.factory,snapshot=self.snapshot,pause=lambda *_:None,**kw)
    def close(self):self.tmp.cleanup()


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.f=Fixture();self.env=patch.dict(os.environ,{},clear=True);self.env.start()
        # CPU独立审阅可以并行，假GPU的协作锁也要按测试fixture隔离。
        self.locktemp=patch.object(P.tempfile,'gettempdir',return_value=str(self.f.root));self.locktemp.start()
    def tearDown(self):self.locktemp.stop();self.env.stop();self.f.close()
    def test_inspect_no_output_no_engine_import(self):
        x=P.inspect_experiment(self.f.ref);self.assertEqual(len(x[2]),3);self.assertFalse(Path(self.f.config['output_root']).exists())
        code="import sys;sys.path.insert(0,%r);import vllm_live_probe;assert not {'torch','qwen_asr','vllm'}&set(sys.modules)"%str(HERE)
        subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True)
    def test_full_stage_restore_zero_factory_even_without_valid_resource(self):
        first=self.f.run();self.assertEqual(len(self.f.engines),1);self.assertEqual(self.f.engines[0].calls,2)
        p=Path(self.f.config['output_root'])/'native-results.json';raw=p.read_bytes()
        with patch.object(P,'check_admission',side_effect=AssertionError('no GPU gate for cache read')):
            second=self.f.run()
        self.assertTrue(second['cache_only_restoration']);self.assertEqual(len(self.f.engines),1)
        self.assertEqual(sum(x['new_model_rows'] for x in second['stage_metrics']),0);self.assertEqual(p.read_bytes(),raw)
        self.assertFalse(first['quality_verified']);self.assertTrue(first['inspection']['synthetic_fixture'])
    def test_interruption_after_durable_batch_does_not_repeat_committed_rows(self):
        with self.assertRaisesRegex(RuntimeError,'interrupt'):
            self.f.run(after_durable_commit=lambda *_:(_ for _ in ()).throw(RuntimeError('interrupt')))
        self.assertEqual(self.f.engines[0].calls,1)
        result=self.f.run();self.assertEqual(self.f.engines[1].calls,1)
        self.assertEqual(sum(x['new_model_rows'] for x in result['stage_metrics']),1)
        self.assertEqual(sum(x['stage_cache_hits'] for x in result['stage_metrics']),2)
    def test_missing_unknown_or_changed_PID_refused_before_factory(self):
        def bad(gpu):
            x=self.f.snapshot(gpu);x['processes']=[{'pid':123,'starttime':'1','cmdline_sha256':'b'*64,'used_memory_mib':14000}];return x
        with self.assertRaisesRegex(ValueError,'existing compute PID'):
            P.run_experiment(self.f.ref,factory=self.f.factory,snapshot=bad,pause=lambda *_:None)
        self.assertEqual(self.f.engines,[])
    def test_declared_existing_PID_is_not_permission_from_free_memory(self):
        self.f.grant['shared_gpu_test_authorized']=True
        self.f.config['resource_admission']=self.f.save('grant.json',self.f.grant);self.f.update()
        with self.assertRaisesRegex(ValueError,'permission'):self.f.run()
        self.assertEqual(self.f.engines,[])
    def test_second_snapshot_change_refused(self):
        seen=[]
        def changed(gpu):
            row=self.f.snapshot(gpu);seen.append(1)
            if len(seen)==2:row['used_memory_mib']=78000
            return row
        with self.assertRaisesRegex(ValueError,'not idle'):
            P.run_experiment(self.f.ref,factory=self.f.factory,snapshot=changed,pause=lambda *_:None)
        self.assertEqual(self.f.engines,[])
    def test_bounds_and_output_old_tree_refused(self):
        for change in ('tokens','batch','memory','output'):
            with self.subTest(change=change):
                old=copy.deepcopy(self.f.config);s=copy.deepcopy(self.f.s)
                if change=='tokens':s['decode']['max_tokens']=513;s['engine']['max_model_len']=1024
                if change=='batch':s['max_batch_rows']=3
                if change=='memory':s['engine']['gpu_memory_utilization']=.4
                if change=='output':self.f.config['output_root']=str(self.f.root/'old-production'/'probe')
                self.f.config['adapter_spec']=self.f.save('adapter.json',s);self.f.update()
                with self.assertRaises(ValueError):P.inspect_experiment(self.f.ref)
                self.f.config=old;self.f.config['adapter_spec']=self.f.save('adapter.json',self.f.s);self.f.update()
    def test_PCM_or_source_pin_corruption_refused(self):
        p=Path(self.f.items[0]['pcm_path']);p.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'PCM file SHA'):P.inspect_experiment(self.f.ref)
    def test_existing_foreign_output_refused(self):
        p=Path(self.f.config['output_root']);p.mkdir();(p/'foreign.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'existing output'):P.inspect_experiment(self.f.ref)
        self.assertEqual((p/'foreign.txt').read_text(),'preserve')
    def test_expired_permission_refused(self):
        self.f.grant['expires_utc']=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=1)).isoformat()
        self.f.config['resource_admission']=self.f.save('grant.json',self.f.grant);self.f.update()
        with self.assertRaisesRegex(ValueError,'expired'):self.f.run()
    def test_cooperating_same_experiment_is_serialized_before_factory(self):
        config,spec,rows,out,identity,report=P.inspect_experiment(self.f.ref)
        out.mkdir();P.write_once(out/'experiment.json',identity)
        with P.cooperative_lock(out/'.bounded-probe.lock'):
            with self.assertRaisesRegex(ValueError,'another bounded probe'):self.f.run()
        self.assertEqual(self.f.engines,[])
    def test_model_PCM_verification_does_not_claim_original_media_identity(self):
        report=P.inspect_experiment(self.f.ref)[-1]
        self.assertTrue(report['input_model_PCM_verified']);self.assertFalse(report['source_audio_identity_verified'])
    def test_supervisor_interruption_only_signals_its_new_child_group(self):
        from unittest.mock import Mock
        child=Mock();child.pid=987654;child.wait.side_effect=[KeyboardInterrupt(),0]
        argv=['probe','--spec',self.f.ref['path'],'--spec-sha',self.f.ref['sha256'],'--execute']
        with patch.object(sys,'argv',argv),patch.object(P.subprocess,'Popen',return_value=child) as popen,patch.object(P.os,'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):P.main()
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(child.pid,P.signal.SIGTERM)
    def test_supervisor_timeout_only_signals_its_new_child_group(self):
        from unittest.mock import Mock
        child=Mock();child.pid=987654;child.wait.side_effect=[subprocess.TimeoutExpired('probe',30),0]
        argv=['probe','--spec',self.f.ref['path'],'--spec-sha',self.f.ref['sha256'],'--execute']
        with patch.object(sys,'argv',argv),patch.object(P.subprocess,'Popen',return_value=child),patch.object(P.os,'killpg') as kill:
            with self.assertRaises(SystemExit) as exc:P.main()
        self.assertEqual(exc.exception.code,124);kill.assert_called_once_with(child.pid,P.signal.SIGTERM)
    def test_numeric_device_comes_only_from_measured_UUID_mapping(self):
        result=self.f.run()
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'3');self.assertEqual(os.environ['CUDA_DEVICE_ORDER'],'PCI_BUS_ID')
        self.assertEqual(result['device_selection']['gpu_uuid'],'GPU-synthetic')
        self.assertEqual(result['device_selection']['physical_index'],3)
        self.assertFalse(result['device_selection']['CUDA_runtime_UUID_verified'])
    def test_missing_wrong_or_ambiguous_numeric_mapping_refused(self):
        for mode in ('missing','wrong','duplicate','boolean'):
            with self.subTest(mode=mode):
                def bad(gpu):
                    row=self.f.snapshot(gpu)
                    if mode=='missing':row.pop('physical_index')
                    if mode=='wrong':row['index_uuid_map']=[{'gpu_uuid':'GPU-foreign','physical_index':3}]
                    if mode=='duplicate':row['index_uuid_map'].append({'gpu_uuid':'GPU-foreign','physical_index':3})
                    if mode=='boolean':row['physical_index']=True
                    return row
                with self.assertRaisesRegex(ValueError,'physical index'):
                    P.run_experiment(self.f.ref,factory=self.f.factory,snapshot=bad,pause=lambda *_:None)
                self.assertEqual(self.f.engines,[])
    def test_UUID_to_numeric_index_change_between_snapshots_refused(self):
        seen=[]
        def changed(gpu):
            row=self.f.snapshot(gpu);seen.append(1)
            if len(seen)==2:
                row['physical_index']=2;row['index_uuid_map'][2]['gpu_uuid']=gpu;row['index_uuid_map'][3]['gpu_uuid']='GPU-foreign-2'
            return row
        with self.assertRaisesRegex(ValueError,'mapping changed'):
            P.run_experiment(self.f.ref,factory=self.f.factory,snapshot=changed,pause=lambda *_:None)
        self.assertEqual(self.f.engines,[])
    def test_capture_reads_numeric_index_from_nvidia_smi_without_gpu_library(self):
        from types import SimpleNamespace
        outputs=[SimpleNamespace(stdout='0, GPU-foreign, 00000000:01:00.0, 80000, 100\n1, GPU-synthetic, 00000000:02:00.0, 80000, 14000\n'),SimpleNamespace(stdout='')]
        with patch.object(P.subprocess,'run',side_effect=outputs) as run:
            value=P.capture_snapshot('GPU-synthetic')
        self.assertEqual(value['physical_index'],1)
        self.assertEqual(value['index_uuid_map'][1],{'physical_index':1,'gpu_uuid':'GPU-synthetic','pci_bus_id':'00000000:02:00.0'})
        self.assertIn('--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used',run.call_args_list[0].args[0])
    def test_noncontiguous_duplicate_or_reversed_PCI_order_refused(self):
        for mode in ('gap','reversed','duplicate','missing'):
            with self.subTest(mode=mode):
                def bad(gpu):
                    row=self.f.snapshot(gpu);mapping=row['index_uuid_map']
                    if mode=='gap':mapping[0]['physical_index']=9
                    if mode=='reversed':mapping[0]['pci_bus_id'],mapping[1]['pci_bus_id']=mapping[1]['pci_bus_id'],mapping[0]['pci_bus_id']
                    if mode=='duplicate':mapping[0]['pci_bus_id']=mapping[1]['pci_bus_id']
                    if mode=='missing':mapping[0].pop('pci_bus_id')
                    return row
                with self.assertRaisesRegex(ValueError,'physical index'):
                    P.run_experiment(self.f.ref,factory=self.f.factory,snapshot=bad,pause=lambda *_:None)
                self.assertEqual(self.f.engines,[])
    def test_known_existing_annotation_PID_is_refused_before_engine(self):
        def known(gpu):
            row=self.f.snapshot(gpu)
            row['processes']=[{'pid':3164371,'container_pid':845,'starttime':'77','cmdline_sha256':'b'*64,'used_memory_mib':14000}]
            return row
        with self.assertRaisesRegex(ValueError,'existing compute PID'):
            P.run_experiment(self.f.ref,factory=self.f.factory,snapshot=known,pause=lambda *_:None)
        self.assertEqual(self.f.engines,[])
    def test_old_shared_JSON_with_matching_known_PID_cannot_bypass_idle_rule(self):
        self.f.grant.update(schema='annotation-shared-GPU-explicit-admission/v1',shared_gpu_test_authorized=True,
            allowed_existing_processes=[{'pid':3164371,'container_pid':845,'starttime':'77','cmdline_sha256':'b'*64,'max_memory_mib':20000}])
        self.f.config['resource_admission']=self.f.save('grant.json',self.f.grant);self.f.update()
        with self.assertRaisesRegex(ValueError,'shared GPU requests are unsupported'):self.f.run()
        self.assertEqual(self.f.engines,[])
    def test_original_exclusive_lock_still_held_refused_even_when_gpu_appears_empty(self):
        ticket=P.read_ref(self.f.grant['existing_exclusive_ticket']);lock=Path(ticket['locks'][0]['path'])
        original=lock.read_bytes()
        with P.cooperative_lock(lock):
            with self.assertRaisesRegex(ValueError,'still holds original lock'):self.f.run()
        self.assertEqual(lock.read_bytes(),original);self.assertEqual(self.f.engines,[])
    def test_original_protocol_lock_is_held_through_factory_and_inference(self):
        ticket=P.read_ref(self.f.grant['existing_exclusive_ticket']);lock=Path(ticket['locks'][0]['path'])
        factory=self.f.factory
        def assert_held(s,root,execute):
            with self.assertRaisesRegex(ValueError,'another bounded probe'):
                with P.cooperative_lock(lock):pass
            return factory(s,root,execute)
        result=P.run_experiment(self.f.ref,factory=assert_held,snapshot=self.f.snapshot,pause=lambda *_:None)
        self.assertTrue(result['original_exclusive_lock']['original_protocol_lock_held_until_probe_end'])
        self.assertFalse(result['existing_compute_sharing_supported'])
    def test_missing_original_lock_ticket_or_changed_inode_refused(self):
        grant=copy.deepcopy(self.f.grant);self.f.grant.pop('existing_exclusive_ticket')
        self.f.config['resource_admission']=self.f.save('grant.json',self.f.grant);self.f.update()
        with self.assertRaisesRegex(ValueError,'actual existing exclusive lock ticket'):self.f.run()
        self.f.grant=grant;self.f.config['resource_admission']=self.f.save('grant.json',grant)
        self.f.config['output_root']=str(self.f.root/'output-inode-case');self.f.update()
        ticket=P.read_ref(grant['existing_exclusive_ticket']);lock=Path(ticket['locks'][0]['path']);lock.rename(lock.with_suffix('.preserved'));lock.touch()
        with self.assertRaisesRegex(ValueError,'lock identity changed'):self.f.run()
        self.assertEqual(self.f.engines,[])
    def test_missing_existing_protocol_source_pin_refused_before_factory(self):
        self.f.config['probe_source_files'].pop(str(HERE/'gpu_launch.py'));self.f.update()
        with self.assertRaisesRegex(ValueError,'source pins required'):self.f.run()
        self.assertEqual(self.f.engines,[])
    def test_capture_busy_row_requires_no_host_container_PID_translation(self):
        from types import SimpleNamespace
        outputs=[SimpleNamespace(stdout='0, GPU-synthetic, 00000000:01:00.0, 80000, 14000\n'),
                 SimpleNamespace(stdout='GPU-synthetic, 3164371, 14000\n')]
        with patch.object(P.subprocess,'run',side_effect=outputs):value=P.capture_snapshot('GPU-synthetic')
        self.assertEqual(value['processes'],[{'pid':3164371,'used_memory_mib':14000}])
    def _pilot_spec(self):
        self.f.s['purpose']='pilot_validation'
        self.f.s['source_files']={str(Path(V.__file__)):V._file_sha(V.__file__)}
        self.f.config['adapter_spec']=self.f.save('adapter.json',self.f.s);self.f.update()
    def test_real_pilot_incomplete_cache_refuses_before_resources_or_factory(self):
        self._pilot_spec()
        with patch.dict(os.environ,{'REDIM_PROBE_SUPERVISOR_PID':str(os.getppid())}),\
             patch.object(P,'check_admission',side_effect=AssertionError('must refuse before resource admission')),\
             patch.object(V,'build_real_engine',side_effect=AssertionError('must never construct real model')):
            with self.assertRaisesRegex(ValueError,'real GPU construction is disabled'):P.run_experiment(self.f.ref)
        self.assertEqual(self.f.engines,[])
    def test_pilot_cannot_bypass_real_disable_with_injected_factory(self):
        self._pilot_spec()
        with self.assertRaisesRegex(ValueError,'restricted to explicit CPU fixtures'):self.f.run()
        self.assertEqual(self.f.engines,[])
    def test_full_pilot_cache_restores_without_real_GPU_or_resource_access(self):
        self._pilot_spec();config,spec,rows,out,identity,report=P.inspect_experiment(self.f.ref)
        out.mkdir();P.write_once(out/'experiment.json',identity)
        # 只用本测试的假引擎制作格式有效的暂存结果；不是实际权重或GPU证据。
        stage=ASRStageStore(out/'stage',digest(spec),synthetic_fixture=False)
        adapter=V.VLLMBackend(spec,FakeEngine(),stage,parse_answer=lambda text,forced:tuple(text.split('|',1)),
                              prompt_builder=lambda *_:'fixture',synthetic_fixture=False)
        for start in range(0,len(rows),2):adapter.run_batch(rows[start:start+2])
        with patch.dict(os.environ,{'REDIM_PROBE_SUPERVISOR_PID':str(os.getppid())}),\
             patch.object(P,'check_admission',side_effect=AssertionError('cache must not touch GPU resources')),\
             patch.object(V,'build_real_engine',side_effect=AssertionError('cache must not construct real engine')):
            restored=P.run_experiment(self.f.ref)
        self.assertTrue(restored['cache_only_restoration']);self.assertFalse(restored['real_engine_execution_enabled'])
        self.assertEqual(sum(v['new_model_rows'] for v in restored['stage_metrics']),0)
    def test_existing_completed_output_cannot_be_overwritten(self):
        self.f.run();p=Path(self.f.config['output_root'])/'native-results.json';p.write_text('changed')
        with self.assertRaisesRegex(ValueError,'immutable experiment'):self.f.run()
        self.assertEqual(len(self.f.engines),1)


if __name__=='__main__':unittest.main()
