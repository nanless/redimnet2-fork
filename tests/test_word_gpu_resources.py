"""CPU-only admission red-team: no CUDA import, initialization, query or computation."""
from __future__ import annotations
import copy,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'word_ft/gpu_resources.py'
spec=importlib.util.spec_from_file_location('word_gpu_resources_under_test',SOURCE)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
U0='GPU-11111111-1111-1111-1111-111111111111'
U1='GPU-22222222-2222-2222-2222-222222222222'
U2='GPU-33333333-3333-3333-3333-333333333333'
GPU=f'0, {U0}, NVIDIA A800-SXM4-80GB, 535.129.03, 81920, 128, 81792\n1, {U1}, NVIDIA A800-SXM4-80GB, 535.129.03, 81920, 128, 81792\n'


def snapshot(now=100.):
    return dict(g.parse_snapshot(GPU,''),monotonic=now,utc='fixture-only')


class ParseAndBudgetTests(unittest.TestCase):
    def test_import_never_imports_torch(self):
        code="import importlib.util,sys; s=importlib.util.spec_from_file_location('gate',sys.argv[1]); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',code,str(SOURCE)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_valid_exact_memory_and_uuid(self):
        rows=g.check_idle(snapshot(),[0,1],24000)
        self.assertEqual([x['uuid'] for x in rows],[U0,U1])

    def test_process_even_zero_utilization_and_zero_memory_refused(self):
        for memory in ('0','[N/A]','123'):
            with self.subTest(memory=memory):
                snap=g.parse_snapshot(GPU,f'{U0}, 2323291, [Not Found], {memory}\n');snap['utilization_gpu']=0
                with self.assertRaises(g.AdmissionRefused):g.check_idle(snap,[0,1],24000)

    def test_any_selected_external_pid_refused(self):
        for uuid in (U0,U1):
            with self.subTest(uuid=uuid),self.assertRaises(g.AdmissionRefused):
                g.check_idle(g.parse_snapshot(GPU,f'{uuid}, 99999, other-user-process, 1\n'),[0,1],24000)

    def test_unselected_gpu_pid_does_not_authorize_selected_gpu(self):
        extra=f'2, {U2}, NVIDIA A800-SXM4-80GB, 535.129.03, 81920, 4096, 77824\n'
        snap=g.parse_snapshot(GPU+extra,f'{U2}, 100, someone-else, 4096\n')
        self.assertEqual(len(g.check_idle(snap,[0,1],24000)),2)

    def test_unknown_or_missing_uuid_or_driver(self):
        for raw in ('',GPU.replace(U0,'N/A'),GPU.replace(U1,U0),GPU.replace('535.129.03','unknown'),GPU.splitlines()[0]+'\n'+GPU.splitlines()[0]):
            with self.subTest(raw=raw),self.assertRaises(g.AdmissionRefused):g.parse_snapshot(raw,'')

    def test_unknown_csv_or_memory_is_not_idle(self):
        for raw in (GPU.replace('81920, 128, 81792','N/A, 128, 81792'),GPU.replace('128, 81792','-1, 81921'),GPU.replace('128, 81792','128, 82000'),GPU+'garbage\n',GPU.replace('0, '+U0,'0, '+U0+', extra')):
            with self.subTest(raw=raw),self.assertRaises(g.AdmissionRefused):g.parse_snapshot(raw,'')

    def test_process_unknown_mapping_pid_or_column_refused(self):
        for raw in (f'{U2}, 12, other, 1',f'{U0}, N/A, other, 1',f'{U0}, 0, other, 1',f'{U0}, 12, , 1',f'{U0}, 12, other'):
            with self.subTest(raw=raw),self.assertRaises(g.AdmissionRefused):g.parse_snapshot(GPU,raw)

    def test_exact_two_unique_indices(self):
        for indices in ([0],[0,0],[0,2],[True,1],[-1,0],[0,1,2]):
            with self.subTest(indices=indices),self.assertRaises(g.AdmissionRefused):g.check_idle(snapshot(),indices,24000)

    def test_budget_reserve_and_driver_limits(self):
        for budget,reserve,used in [(None,2048,256),(True,2048,256),(511,2048,256),(80000,2048,256),(24000,2047,256),(24000,2048,513),(24000,2048,-1)]:
            with self.subTest((budget,reserve,used)),self.assertRaises(g.AdmissionRefused):
                g.check_idle(snapshot(),[0,1],budget,used,reserve)
        snap=snapshot();snap['devices'][0].update(used_mib=300,free_mib=81620)
        with self.assertRaises(g.AdmissionRefused):g.check_idle(snap,[0,1],24000)

    def test_capture_runner_failure_or_stderr_refused(self):
        for ret,stderr in [(1,''),(0,'unexpected warning')]:
            runner=lambda *a,**kw:SimpleNamespace(returncode=ret,stdout=GPU,stderr=stderr)
            with self.subTest(ret=ret,stderr=stderr),self.assertRaises(g.AdmissionRefused):g.capture_snapshot(runner=runner)
        with self.assertRaises(g.AdmissionRefused):g.capture_snapshot(runner=lambda *a,**kw:(_ for _ in ()).throw(TimeoutError('fixture')))


class FreshnessTests(unittest.TestCase):
    def probe(self,transform=None,sleep_real=True):
        now=[100.];calls=[0]
        def capture():
            calls[0]+=1;snap=snapshot(now[0])
            return transform(snap,calls[0]) if transform else snap
        def sleep(seconds):
            if sleep_real:now[0]+=seconds
        return g.double_probe([0,1],24000,interval_seconds=5,capture=capture,sleep=sleep,clock=lambda:now[0])

    def test_two_fresh_stable_idle_snapshots_pass(self):self.assertEqual(len(self.probe()['snapshots']),2)
    def test_mock_sleep_without_elapsed_time_refused(self):
        with self.assertRaises(g.AdmissionRefused):self.probe(sleep_real=False)
    def test_old_cached_snapshots_five_seconds_apart_refused(self):
        def old(snap,n):snap['monotonic']-=1000;return snap
        with self.assertRaises(g.AdmissionRefused):self.probe(old)
    def test_future_or_nonfinite_snapshot_timestamp_refused(self):
        for value in (1e9,float('nan'),float('inf')):
            def stale(snap,n):snap['monotonic']=value+n*5;return snap
            with self.subTest(value=value),self.assertRaises(g.AdmissionRefused):self.probe(stale)
    def test_same_snapshot_object_or_timestamp_refused(self):
        def repeated(snap,n):snap['monotonic']=100.;return snap
        with self.assertRaises(g.AdmissionRefused):self.probe(repeated)
    def test_changed_uuid_driver_or_total_refused(self):
        for field,value in [('uuid',U2),('driver_version','535.130.01'),('name','another-device'),('total_mib',90000)]:
            def mutate(snap,n):
                if n==2:
                    snap['devices'][0][field]=value
                    if field=='total_mib':snap['devices'][0]['free_mib']=value-128
                return snap
            with self.subTest(field=field),self.assertRaises(g.AdmissionRefused):self.probe(mutate)
    def test_pid_appears_on_second_snapshot_refused(self):
        def mutate(snap,n):
            if n==2:snap['compute_processes']=[{'gpu_uuid':U1,'pid':2323291,'process_name':'unknown','used_mib':None}]
            return snap
        with self.assertRaises(g.AdmissionRefused):self.probe(mutate)
    def test_spacing_outside_bound_refused(self):
        for value in (0,4.9,31,float('nan')):
            with self.subTest(value=value),self.assertRaises(g.AdmissionRefused):g.double_probe([0,1],24000,interval_seconds=value)


class LockTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_two_uuid_locks_mutually_exclusive_and_release(self):
        with g.DeviceLocks(self.root,[U0,U1]) as held:
            self.assertEqual(len(held.evidence),2)
            with self.assertRaises(BlockingIOError):
                with g.DeviceLocks(self.root,[U1,U0]):pass
        with g.DeviceLocks(self.root,[U0,U1]):pass
    def test_single_overlap_also_refused(self):
        with g.DeviceLocks(self.root,[U0,U1]):
            with self.assertRaises(BlockingIOError):
                with g.DeviceLocks(self.root,[U1,U2]):pass
        with g.DeviceLocks(self.root,[U1,U2]):pass
    def test_symlink_lock_file_refused(self):
        outside=self.root/'outside';outside.write_text('preserve');(self.root/(U0+'.lock')).symlink_to(outside)
        with self.assertRaises(OSError):
            with g.DeviceLocks(self.root,[U0,U1]):pass
        self.assertEqual(outside.read_text(),'preserve')
    def test_invalid_uuid_lock_sets_refused(self):
        for uuids in ([U0],[U0,U0],[U0,'not-uuid']):
            with self.subTest(uuids=uuids),self.assertRaises(g.AdmissionRefused):g.DeviceLocks(self.root,uuids)
    def test_original_data_or_repo_lock_paths_refused(self):
        for path in ['/root/code/github_repos/redimnet2-fork/new-lock','/root/group-shared/voiceprint/data/new-lock','/root/experiments/redimnet2/b6-mix-official-20261019/new-lock']:
            with self.subTest(path=path),self.assertRaises(g.AdmissionRefused):g.DeviceLocks(path,[U0,U1])


class RuntimeOwnershipTests(unittest.TestCase):
    def test_own_process_group_allowed(self):
        snap=snapshot();snap['compute_processes']=[{'gpu_uuid':U0,'pid':999,'process_name':'fixture','used_mib':1}]
        with patch.object(g.os,'getpgid',return_value=42):g.allowed_process_group(snap,snap['devices'],42)
    def test_external_process_group_refused(self):
        snap=snapshot();snap['compute_processes']=[{'gpu_uuid':U1,'pid':2323291,'process_name':'[Not Found]','used_mib':None}]
        with patch.object(g.os,'getpgid',return_value=99),self.assertRaises(g.AdmissionRefused):g.allowed_process_group(snap,snap['devices'],42)
    def test_nvml_pid_missing_from_proc_is_unknown_not_idle(self):
        snap=snapshot();snap['compute_processes']=[{'gpu_uuid':U1,'pid':2323291,'process_name':'[Not Found]','used_mib':None}]
        with patch.object(g.os,'getpgid',side_effect=ProcessLookupError),self.assertRaises(g.AdmissionRefused):g.allowed_process_group(snap,snap['devices'],42)
    def test_permission_denied_ownership_refused(self):
        snap=snapshot();snap['compute_processes']=[{'gpu_uuid':U1,'pid':23,'process_name':'fixture','used_mib':1}]
        with patch.object(g.os,'getpgid',side_effect=PermissionError),self.assertRaises(g.AdmissionRefused):g.allowed_process_group(snap,snap['devices'],42)
    def test_runtime_gpu_identity_drift_refused(self):
        snap=snapshot();devices=copy.deepcopy(snap['devices']);snap['devices'][0]['uuid']=U2
        with self.assertRaises(g.AdmissionRefused):g.allowed_process_group(snap,devices,42)


class TicketTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.source={'fixture':'frozen-code'}
        self.locks=g.DeviceLocks(self.root/'locks',[U0,U1]);self.locks.__enter__()
        self.orig_read=Path.read_text;self.orig_stat=Path.stat;self.orig_resolve=Path.resolve
        self.fds={x['fd']:x for x in self.locks.evidence};self.pid=os.getpid()
        def read(p,*a,**kw):
            if str(p)=='/proc/sys/kernel/random/boot_id':return 'fixture-boot\n'
            return self.orig_read(p,*a,**kw)
        def stat(p,*a,**kw):
            if str(p).startswith(f'/proc/{self.pid}/fd/'):
                fd=int(p.name)
                if fd not in self.fds:raise FileNotFoundError(str(p))
                return os.fstat(fd)
            return self.orig_stat(p,*a,**kw)
        def resolve(p,*a,**kw):
            if str(p).startswith(f'/proc/{self.pid}/fd/'):
                fd=int(p.name)
                if fd not in self.fds:raise FileNotFoundError(str(p))
                return Path(self.fds[fd]['path'])
            return self.orig_resolve(p,*a,**kw)
        self.patches=[patch.object(g,'process_start_time',return_value=123),patch.object(g.socket,'gethostname',return_value='fixture-host'),
            patch.object(g.time,'monotonic',return_value=100.),patch.object(Path,'read_text',read),patch.object(Path,'stat',stat),patch.object(Path,'resolve',resolve)]
        for p in self.patches:p.start()
        probe={'devices':snapshot()['devices'],'snapshots':[snapshot(95),snapshot(100)],'interval_seconds':5.}
        self.ticket=g.create_ticket(probe,self.locks,24000,2048,256,self.source)
        self.path=self.root/'ticket.json';self.write()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.locks.__exit__(None,None,None);self.tmp.cleanup()
    def write(self):self.path.write_text(g.canonical(self.ticket));self.sha=g.sha_file(self.path)
    def validate(self,**kw):return g.validate_ticket(self.path,self.sha,self.source,**kw)
    def test_valid_live_ticket_and_held_fds_pass(self):self.assertEqual(self.validate()['schema'],g.TICKET_SCHEMA)
    def test_bad_expected_sha_or_source_refused(self):
        with self.assertRaises(g.AdmissionRefused):g.validate_ticket(self.path,'0'*64,self.source)
        with self.assertRaises(g.AdmissionRefused):g.validate_ticket(self.path,self.sha,{'different':'source'})
    def test_stale_future_wrong_host_boot_or_execute_refused(self):
        for field,value in [('created_monotonic',-1),('created_monotonic',101),('hostname','other'),('boot_id','other'),('execute',False),('schema','other')]:
            old=copy.deepcopy(self.ticket);self.ticket[field]=value;self.write()
            with self.subTest(field=field,value=value),self.assertRaises(g.AdmissionRefused):self.validate()
            self.ticket=old;self.write()
    def test_parent_pid_reuse_or_missing_fd_refused(self):
        self.ticket['parent_start_time']=124;self.write()
        with self.assertRaises(g.AdmissionRefused):self.validate()
        self.ticket['parent_start_time']=123;self.ticket['locks'][0]['fd']=9999;self.write()
        with self.assertRaises((g.AdmissionRefused,FileNotFoundError)):self.validate()
    def test_lock_inode_uuid_or_actual_hold_refused(self):
        for field,value in [('inode',0),('uuid',U2),('path',str(self.root/'other'))]:
            old=copy.deepcopy(self.ticket);self.ticket['locks'][0][field]=value;self.write()
            with self.subTest(field=field),self.assertRaises((g.AdmissionRefused,FileNotFoundError)):self.validate()
            self.ticket=old;self.write()
        self.locks.__exit__(None,None,None)
        with self.assertRaises((g.AdmissionRefused,OSError)):self.validate()
    def test_ticket_symlink_refused(self):
        link=self.root/'link.json';link.symlink_to(self.path)
        with self.assertRaises((g.AdmissionRefused,OSError)):g.validate_ticket(link,self.sha,self.source)
    def test_parent_must_be_current_worker_ancestor(self):
        with patch.object(g,'is_descendant',return_value=False),self.assertRaises(g.AdmissionRefused):self.validate()
    def test_ticket_old_or_busy_probe_is_refused_even_fresh_ticket(self):
        for transform in ('old','busy','missing','identity'):
            old=copy.deepcopy(self.ticket)
            if transform=='old':
                for snap in self.ticket['probe']['snapshots']:snap['monotonic']-=1000
            elif transform=='busy':self.ticket['probe']['snapshots'][1]['compute_processes']=[{'gpu_uuid':U0,'pid':2323291,'process_name':'[Not Found]','used_mib':None}]
            elif transform=='missing':self.ticket['probe']['snapshots'].pop()
            else:self.ticket['devices'][0]['uuid']=U2
            self.write()
            with self.subTest(transform=transform),self.assertRaises(g.AdmissionRefused):self.validate()
            self.ticket=old;self.write()
    def test_open_but_unlocked_fd_is_not_lock_ownership(self):
        fd=self.locks.evidence[0]['fd'];g.fcntl.flock(fd,g.fcntl.LOCK_UN)
        try:
            with self.assertRaises(g.AdmissionRefused):self.validate()
        finally:g.fcntl.flock(fd,g.fcntl.LOCK_EX|g.fcntl.LOCK_NB)
    def test_active_ticket_still_rejects_future_time(self):
        self.ticket['created_monotonic']=101.;self.write()
        with self.assertRaises(g.AdmissionRefused):self.validate(allow_running=True)

    def test_swap_between_hash_and_json_read_cannot_use_unhashed_bytes(self):
        original=self.path.read_bytes();malicious=dict(self.ticket,memory_budget_mib=79000)
        # Old two-read implementation hashes expected bytes, then consumes an independently replaced JSON.
        # New implementation must hash and parse exactly the same read_bytes result.
        original_sha=g.sha_file
        def swapped(path):
            result=original_sha(path)
            if Path(path)==self.path:self.path.write_text(g.canonical(malicious))
            return result
        with patch.object(g,'sha_file',side_effect=swapped):
            try:result=self.validate()
            except g.AdmissionRefused:pass
            else:self.assertEqual(result['memory_budget_mib'],24000)
        if self.path.read_bytes()!=original:self.path.write_bytes(original)




@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'CPU PyTorch unavailable locally; run these pure CUDA mocks in the existing server CPU environment')
class PreCudaWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from word_ft import gpu_train
        cls.gt=gpu_train
    def setUp(self):
        from contextlib import ExitStack
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.out=self.root/'new-output'
        cfg_path=Path(__file__).resolve().parents[1]/'configs/word_ft/cpu_candidate_engineering.json'
        self.config=json.loads(cfg_path.read_text());self.config['runtime_policy']=self.gt.RUNTIME_SCHEMA
        self.data=self.root/'data.json';self.data.write_text('fixture-only')
        self.official=self.root/'never-load-real-weights.pt'
        self.stack=ExitStack()
        self.stack.enter_context(patch.dict(os.environ,{'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0',
            'CUDA_VISIBLE_DEVICES':U0+','+U1,'WORD_GPU_TICKET':'','WORD_GPU_TICKET_SHA256':''},clear=False))
        self.cuda=self.stack.enter_context(patch.object(self.gt.torch,'cuda'))
        self.cuda.is_initialized.return_value=False
        for name in ('set_device','device_count','get_device_properties','set_per_process_memory_fraction','init','_lazy_init','manual_seed','manual_seed_all'):
            getattr(self.cuda,name).side_effect=AssertionError('CUDA operation forbidden in CPU test: '+name)
        self.stack.enter_context(patch.object(self.gt.dist,'is_initialized',return_value=False))
        self.init_dist=self.stack.enter_context(patch.object(self.gt.dist,'init_process_group',side_effect=AssertionError('NCCL forbidden in CPU test')))
        self.init_gpu=self.stack.enter_context(patch.object(self.gt,'initialize_cuda_runtime',side_effect=AssertionError('CUDA runtime forbidden in CPU test')))
        self.load_model=self.stack.enter_context(patch.object(self.gt,'load_official',side_effect=AssertionError('real weights forbidden in CPU test')))
        self.stack.enter_context(patch.object(self.gt,'source_identity',return_value={'fixture':'source'}))
    def tearDown(self):self.stack.close();self.tmp.cleanup()
    def run_worker(self,**kwargs):
        return self.gt.run(self.config,self.data,self.official,self.out,**kwargs)
    def fake_ticket(self):
        return {'parent_pid':os.getpid()+100000,'devices':snapshot()['devices'],'config_digest':self.gt.digest(self.config),
            'dataset_file_sha256':self.gt.file_sha(self.data),'output_root':str(self.out.resolve()),'resume_path':None,
            'stop_after':None,'memory_budget_mib':24000,'source_identity':{'fixture':'source'}}
    def assert_no_runtime(self):
        self.init_gpu.assert_not_called();self.init_dist.assert_not_called();self.load_model.assert_not_called()
        self.cuda.set_device.assert_not_called();self.cuda.get_device_properties.assert_not_called()
    def test_wrong_world_or_rank_rejected_before_any_cuda_call(self):
        for values in ({'WORLD_SIZE':'1'},{'WORLD_SIZE':'3'},{'RANK':'2','LOCAL_RANK':'2'},{'RANK':'0','LOCAL_RANK':'1'},{'WORLD_SIZE':'N/A'}):
            with self.subTest(values=values),patch.dict(os.environ,values),self.assertRaises(ValueError):self.run_worker()
        self.cuda.is_initialized.assert_not_called();self.assert_no_runtime()
    def test_cpu_checkpoint_refused_before_any_cuda_call_both_ranks(self):
        self.out.mkdir();(self.out/'.training.lock').touch();path=self.out/'train_step_2.pt';path.write_bytes(b'fixture')
        for rank in (0,1):
            with self.subTest(rank=rank),patch.dict(os.environ,{'RANK':str(rank),'LOCAL_RANK':str(rank)}),patch.object(self.gt.torch,'load',return_value={'schema':'word-training/v1','global_step':2}),self.assertRaisesRegex(ValueError,'CPU/different schema'):
                self.run_worker(resume=path)
        self.cuda.is_initialized.assert_not_called();self.assert_no_runtime()
    def test_missing_ticket_refused_without_context_or_nccl(self):
        with self.assertRaises(self.gt.G.AdmissionRefused):self.run_worker()
        self.assert_no_runtime();self.assertFalse(self.out.exists())
    def test_non_gpu_policy_or_amp_refused_before_cuda(self):
        for key,value in [('runtime_policy','CPU'),('amp',True)]:
            old=copy.deepcopy(self.config);self.config[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_worker()
            self.config=old
        self.cuda.is_initialized.assert_not_called();self.assert_no_runtime()
    def test_ticket_data_config_output_mapping_or_budget_mismatch_refused(self):
        mutations=[('config_digest','tamper'),('dataset_file_sha256','0'*64),('output_root',str(self.root/'different-output')),
            ('resume_path','foreign-checkpoint'),('stop_after',10)]
        for key,value in mutations:
            ticket=self.fake_ticket();ticket[key]=value
            with self.subTest(key=key),patch.object(self.gt.G,'validate_ticket',return_value=ticket),self.assertRaises(self.gt.G.AdmissionRefused):self.run_worker()
        with patch.object(self.gt.G,'validate_ticket',return_value=self.fake_ticket()),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':U1+','+U0}),self.assertRaises(self.gt.G.AdmissionRefused):self.run_worker()
        with patch.object(self.gt.G,'validate_ticket',return_value=self.fake_ticket()),self.assertRaises(self.gt.G.AdmissionRefused):self.run_worker(gpu_memory_mib=32000)
        self.assert_no_runtime();self.assertFalse(self.out.exists())
    def test_self_parent_ticket_is_not_worker_admission(self):
        ticket=self.fake_ticket();ticket['parent_pid']=os.getpid()
        with patch.object(self.gt.G,'validate_ticket',return_value=ticket),self.assertRaises(self.gt.G.AdmissionRefused):self.run_worker()
        self.assert_no_runtime()
    def test_existing_fresh_output_refused_before_runtime(self):
        self.out.mkdir();(self.out/'do-not-overwrite.txt').write_text('preserve')
        with patch.object(self.gt.G,'validate_ticket',return_value=self.fake_ticket()),self.assertRaises(self.gt.G.AdmissionRefused):self.run_worker()
        self.assert_no_runtime();self.assertEqual((self.out/'do-not-overwrite.txt').read_text(),'preserve')
    def test_checkpoint_cannot_reseal_changed_disk_source(self):
        pinned={'fixture':'old-source'}
        with patch.object(self.gt,'_PINNED_SOURCE_IDENTITY',copy.deepcopy(pinned)),patch.object(self.gt,'source_identity',return_value={'fixture':'new-source'}),self.assertRaisesRegex(ValueError,'source drift'):
            try:self.gt.checkpoint_payload(None,None,None,self.config,'dataset',2,2,{},[],{},'head')
            finally:self.assertEqual(self.gt._PINNED_SOURCE_IDENTITY,pinned)
        self.assert_no_runtime()
    def test_restore_refuses_changed_source_before_state_load(self):
        with patch.object(self.gt,'_PINNED_SOURCE_IDENTITY',{'fixture':'old'}),self.assertRaisesRegex(ValueError,'source identity'):
            self.gt.restore_checkpoint({'schema':self.gt.CHECKPOINT_SCHEMA,'source_identity':{'fixture':'new'}},None,None,None,self.config,'dataset',2,0,'head')
        self.assert_no_runtime()
    def test_software_runtime_drift_refused_before_cuda(self):
        self.out.mkdir();resume=self.out/'train_step_2.pt'
        data={'metadata':{'split':'train'},'records':[{'split':'train'} for _ in range(4)],'speaker_map':{'a':0,'b':1},'dataset_id':'fixture-data'}
        software={'python_version':'fixture-python','numpy_version':'fixture-numpy','scipy_version':'fixture-scipy','soundfile_version':'fixture-sf',
            'soundfile_module_sha256':'f'*64,'scipy_resample_poly_module_sha256':'e'*64}
        decoder={'fixture':'private-decoder'}
        for key in software:
            runtime={'schema':self.gt.RUNTIME_SCHEMA,'rank_device_map':g.stable_identity(snapshot()['devices']),
                'memory_budget_mib_per_rank':24000,'decoder':decoder,**software};runtime[key]='changed'
            payload={'config':self.config,'dataset_id':'fixture-data','runtime_identity':runtime}
            ticket=self.fake_ticket();ticket['resume_path']=str(resume.resolve())
            with self.subTest(key=key),patch.object(self.gt,'validate_resume_target',return_value=payload),patch.object(self.gt.G,'validate_ticket',return_value=ticket),patch.object(self.gt,'decoder_identity',return_value=decoder),patch.object(self.gt.schema,'load_dataset',return_value=data),patch.object(self.gt,'software_runtime',return_value=software),self.assertRaisesRegex(ValueError,'before CUDA'):
                self.run_worker(resume=resume)
        self.assert_no_runtime()

    def test_cpu_device_cannot_enter_gpu_implementation(self):
        with self.assertRaisesRegex(ValueError,'impersonate'):self.run_worker(device='cpu')
        self.cuda.is_initialized.assert_not_called();self.assert_no_runtime()




class LauncherMockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/launch_word_two_gpu.py'
        if not path.exists():raise unittest.SkipTest('new launcher not yet staged')
        spec=importlib.util.spec_from_file_location('word_gpu_launcher_under_test',path)
        cls.launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.launcher)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        # The launcher sees a separate synthetic protected repository, so tests
        # also work when TMPDIR is intentionally inside an ignored test area.
        self.repo_patch=patch.object(self.launcher,'REPO',self.root/'protected-repo');self.repo_patch.start()
    def tearDown(self):self.repo_patch.stop();self.tmp.cleanup()
    def args(self,execute=False):
        return SimpleNamespace(report=self.root/'report.json',probe_only=not execute,execute=execute,
            gpu_indices=[0,1],memory_mib=24000,reserve_mib=2048,driver_used_limit_mib=256,probe_seconds=5.,max_runtime_seconds=60.,
            config=self.root/'config.json',dataset=self.root/'data.json',official=self.root/'official.pt',
            output=self.root/'output',resume=None,stop_after=2)
    def test_busy_probe_refuses_without_popen_or_torchrun(self):
        L=self.launcher;busy=snapshot();busy['compute_processes']=[{'gpu_uuid':U0,'pid':2323291,'process_name':'[Not Found]','used_mib':None}]
        def refuse(*a,**kw):return L.G.check_idle(busy,[0,1],24000)
        with patch.object(L,'GLOBAL_LOCK_ROOT',self.root/'locks'),patch.object(L.G,'training_source_identity',return_value={'fixture':'source'}),patch.object(L.G,'capture_snapshot',return_value=busy),patch.object(L.G,'double_probe',side_effect=refuse),patch.object(L.subprocess,'Popen',side_effect=AssertionError('must not spawn')) as popen:
            self.assertEqual(L.launch(self.args()),2);popen.assert_not_called()
        report=json.loads((self.root/'report.json').read_text());self.assertFalse(report['gpu_hardware_validation_claimed']);self.assertIn('compute PID',report['failure_traceback'])
        self.assertFalse((self.root/'output').exists())
    def test_idle_probe_only_never_spawns(self):
        L=self.launcher;idle=snapshot();probe={'snapshots':[snapshot(95),snapshot(100)],'devices':idle['devices'],'interval_seconds':5.}
        with patch.object(L,'GLOBAL_LOCK_ROOT',self.root/'locks'),patch.object(L.G,'training_source_identity',return_value={'fixture':'source'}),patch.object(L.G,'capture_snapshot',return_value=idle),patch.object(L.G,'double_probe',return_value=probe),patch.object(L.subprocess,'Popen',side_effect=AssertionError('must not spawn')) as popen:
            self.assertEqual(L.launch(self.args()),0);popen.assert_not_called()
        self.assertFalse(json.loads((self.root/'report.json').read_text())['gpu_hardware_validation_claimed'])
    def test_child_env_fixed_uuid_library_threads_no_parent_mutation(self):
        L=self.launcher;config={'decoder':{'library_root':'/fixture/private-decoder/lib'}}
        incoming={'LD_LIBRARY_PATH':'/attacker','LD_PRELOAD':'bad.so','LD_AUDIT':'bad-audit.so','PYTHONPATH':'/attacker','PYTHONHOME':'/attacker','CUDA_VISIBLE_DEVICES':'0,1,2','OPENBLAS_NUM_THREADS':'88','OMP_NUM_THREADS':'99','NUMEXPR_NUM_THREADS':'77'}
        with patch.dict(os.environ,incoming):
            before=dict(os.environ);env=L.child_environment(self.root/'ticket.json','f'*64,snapshot()['devices'],config)
            self.assertEqual(dict(os.environ),before)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],U0+','+U1)
        self.assertEqual(env['LD_LIBRARY_PATH'],':'.join(['/fixture/private-decoder/lib']+L.FIXED_INFRA_LIBS))
        for key in ('LD_PRELOAD','LD_AUDIT','PYTHONPATH','PYTHONHOME'):self.assertNotIn(key,env)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):self.assertEqual(env[key],'1')
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
    def test_worker_command_inherits_own_torchrun_group(self):
        cmd=self.launcher.worker_command(self.args(execute=True))
        self.assertIn('--run-path',cmd);self.assertIn('--start-method=spawn',cmd);self.assertIn('--nproc-per-node=2',cmd)
        self.assertIn('--worker',cmd);self.assertIn('--stop-after',cmd);self.assertNotIn('--no-python',cmd)
    def test_cleanup_signals_only_own_group_and_rejects_foreign_group(self):
        L=self.launcher;child=unittest.mock.Mock();child.pid=987654;child.poll.return_value=None;child.wait.return_value=0
        child._word_group_owner={'pgid':child.pid,'start_time':123,'nonce':'fixture'}
        with patch.object(L,'owned_group_members',side_effect=[[child.pid+1],[]]),patch.object(L.os,'killpg') as kill:
            L.stop_own_group(child);self.assertTrue(kill.called);self.assertTrue(all(c.args[0]==child.pid for c in kill.call_args_list))
        child._word_group_owner['pgid']=888
        with patch.object(L.os,'getpgrp',return_value=333),patch.object(L.os,'killpg') as kill,self.assertRaises(L.G.AdmissionRefused):
            L.stop_own_group(child)
        kill.assert_not_called()
    def test_watchdog_external_pid_stops_only_own_child(self):
        L=self.launcher;child=unittest.mock.Mock();child.pid=987654;child.poll.return_value=None
        snap=snapshot();snap['compute_processes']=[{'gpu_uuid':U0,'pid':2323291,'process_name':'other','used_mib':1000}]
        with patch.object(L.subprocess,'Popen',return_value=child) as popen,patch.object(L.G,'process_start_time',return_value=123),patch.object(L.G,'capture_snapshot',return_value=snap),patch.object(L.G.os,'getpgid',return_value=777),patch.object(L,'stop_own_group') as stop,self.assertRaises(L.G.AdmissionRefused):
            L.supervise(['fixture-cpu-only'],{},self.root/'watch.log',snap['devices'],max_seconds=60)
        self.assertTrue(popen.call_args.kwargs['start_new_session']);stop.assert_called_once_with(child)
    def test_dead_leader_does_not_skip_owned_worker_cleanup(self):
        L=self.launcher;child=unittest.mock.Mock();child.pid=987654;child.poll.return_value=0
        child._word_group_owner={'pgid':child.pid,'start_time':123,'nonce':'fixture'}
        with patch.object(L,'owned_group_members',side_effect=[[child.pid+1],[]]),patch.object(L.os,'killpg') as kill:
            self.assertTrue(L.stop_own_group(child))
        kill.assert_called_once_with(child.pid,L.signal.SIGTERM)
    def test_normal_parent_exit_with_residual_workers_is_not_success(self):
        L=self.launcher;child=unittest.mock.Mock();child.pid=987654;child.poll.return_value=0;child.returncode=0
        with patch.object(L.subprocess,'Popen',return_value=child),patch.object(L.G,'process_start_time',return_value=123),patch.object(L,'stop_own_group',side_effect=[True,False]) as stop,self.assertRaises(L.G.AdmissionRefused):
            L.supervise(['fixture-cpu-only'],{},self.root/'residual.log',snapshot()['devices'],max_seconds=60)
        self.assertEqual(stop.call_count,2)

    def test_parent_watchdog_source_drift_stops_only_own_group(self):
        L=self.launcher;child=unittest.mock.Mock();child.pid=987654;child.poll.return_value=None
        with patch.object(L.subprocess,'Popen',return_value=child),patch.object(L.G,'process_start_time',return_value=123),patch.object(L.G,'training_source_identity',return_value={'fixture':'new'}),patch.object(L,'stop_own_group') as stop,self.assertRaises(L.G.AdmissionRefused):
            L.supervise(['fixture'],{},self.root/'source-drift.log',snapshot()['devices'],max_seconds=60,expected_source={'fixture':'old'})
        stop.assert_called_once_with(child)

    def test_cli_modes_are_explicit_and_exclusive(self):
        from contextlib import redirect_stderr
        import io
        L=self.launcher
        for argv in (['launcher'],['launcher','--probe-only','--execute'],['launcher','--probe-only']):
            with self.subTest(argv=argv),patch.object(sys,'argv',argv),redirect_stderr(io.StringIO()),patch.object(L.subprocess,'Popen') as popen,self.assertRaises(SystemExit):L.main()
            popen.assert_not_called()


if __name__=='__main__':unittest.main()
