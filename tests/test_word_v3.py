"""Independent v3 CPU red-team. Never initializes CUDA, queries GPUs or loads B6."""
from __future__ import annotations
import copy,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from word_ft_v3 import topology as P
from word_ft_v3 import resources as G
U=['GPU-'+f'{i:08x}-1111-2222-3333-{i:012x}' for i in range(1,9)]
def snapshot(t=100):
 raw='\n'.join(f'{i}, {u}, A800, 535.129.03, 81920, 128, 81792' for i,u in enumerate(U))
 return dict(G.parse_snapshot(raw,''),monotonic=t)
def ticket():
 return {'topology':P.topology(8),'devices':snapshot()['devices'],'probe':{'snapshots':[snapshot(95),snapshot(100)],'interval_seconds':5},'created_monotonic':100,'memory_budget_mib':4096,'reserve_mib':2048,'driver_used_limit_mib':256}
class TopologyTests(unittest.TestCase):
 def test_only_exact_W2_W8(self):
  for w in [1,3,4,16,True,2.0]:
   with self.subTest(w=w),self.assertRaises(ValueError):P.topology(w)
  for w in [2,8]:self.assertEqual(P.validate(P.topology(w)),w)
 def test_no_multi_node_or_backend_alias(self):
  for key,v in [('nodes',2),('backend','gloo'),('launch_mode','platform_per_rank'),('gpus_per_node',2)]:
   x=P.topology(8);x[key]=v
   with self.subTest(key=key),self.assertRaises(ValueError):P.validate(x)
 def test_exact_rank_world_local_mapping(self):
  c={'topology':P.topology(8)}
  for r in range(8):self.assertEqual(P.rank_environment(c,{'WORLD_SIZE':'8','RANK':str(r),'LOCAL_RANK':str(r)}),(8,r,r))
  for e in [{},{'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0'},{'WORLD_SIZE':'8','RANK':'8','LOCAL_RANK':'8'},{'WORLD_SIZE':'8','RANK':'1','LOCAL_RANK':'0'}]:
   with self.subTest(e=e),self.assertRaises(ValueError):P.rank_environment(c,e)
 def test_global_batch_no_duplicate_at_epoch_boundaries(self):
  for n in [16,17,19,31,33,64]:
   for step in range(128):
    z=P.step_indices(n,step,2,8,23)
    self.assertEqual(len(z),16);self.assertEqual(len(set(z)),16);self.assertTrue(all(0<=a<n for a in z))
 def test_every_full_epoch_covers_all_families_without_drop(self):
  for n in [17,19,31,33]:
   z=sum([P.step_indices(n,s,2,8,23) for s in range(n*2)],[])
   for start in range(0,len(z)-n+1,n):self.assertEqual(set(z[start:start+n]),set(range(n)))
 def test_W2K8_and_W8K2_identical_global_order(self):
  for step in range(80):self.assertEqual(P.step_indices(19,step,8,2,7),P.step_indices(19,step,2,8,7))
 def test_restart_at_saved_step_identical_sequence(self):
  allsteps=[P.step_indices(31,s,2,8,17) for s in range(40)]
  self.assertEqual(allsteps[13:],[P.step_indices(31,s,2,8,17) for s in range(13,40)])
 def test_insufficient_or_noninteger_pool_geometry(self):
  for args in [(15,0,2,8,1),(16,-1,2,8,1),(16,0,0,8,1),(16,0,2,4,1),(16,0,2,8,True)]:
   with self.subTest(args=args),self.assertRaises(ValueError):P.step_indices(*args)
class ResourceTests(unittest.TestCase):
 def test_W8_requires_all_eight_UUIDs_and_budgets(self):
  self.assertEqual(len(G.check_idle(snapshot(),list(range(8)),4096)),8)
  for ids in [list(range(7)),[0]*8,list(range(1,9))]:
   with self.subTest(ids=ids),self.assertRaises(G.AdmissionRefused):G.check_idle(snapshot(),ids,4096)
 def test_busy_zero_memory_and_unknown_PID_remain_busy(self):
  for u in U:
   s=snapshot();s['compute_processes']=[{'gpu_uuid':u,'pid':2323291,'used_mib':None,'process_name':'[Not Found]'}]
   with self.assertRaises(G.AdmissionRefused):G.check_idle(s,list(range(8)),4096)
   with patch.object(G.os,'getpgid',side_effect=ProcessLookupError),self.assertRaises(G.AdmissionRefused):G.allowed_process_group(s,s['devices'],999)
 def test_unknown_CSV_UUID_or_memory_refused(self):
  for raw in ['garbage','0, N/A, A800, 535.129.03, 81920, 0, 81920','0, '+U[0]+', A800, 535.129.03, N/A, 0, 81920']:
   with self.subTest(raw=raw),self.assertRaises(G.AdmissionRefused):G.parse_snapshot(raw,'')
 def test_W8_ticket_geometry_and_unchanged_time_gates(self):
  G.verify_ticket_probe(ticket())
  for mode in ['count','spacing','old','busy','identity','topology']:
   t=ticket()
   if mode=='count':t['devices'].pop()
   elif mode=='spacing':t['probe']['snapshots'][1]['monotonic']=96
   elif mode=='old':t['created_monotonic']=1000
   elif mode=='busy':t['probe']['snapshots'][0]['compute_processes']=[{'gpu_uuid':U[7],'pid':1}]
   elif mode=='identity':t['probe']['snapshots'][0]['devices'][7]['name']='different'
   else:t['topology']=P.topology(2)
   with self.subTest(mode=mode),self.assertRaises(G.AdmissionRefused):G.verify_ticket_probe(t)
 def test_double_probe_rejects_cached_and_fake_wait(self):
  for old,fake in [(True,False),(False,True)]:
   now=[100.]
   def capture():return snapshot(now[0]-1000 if old else now[0])
   def sleep(s):
    if not fake:now[0]+=s
   with self.subTest(old=old,fake=fake),self.assertRaises(G.AdmissionRefused):G.double_probe(list(range(8)),4096,capture=capture,sleep=sleep,clock=lambda:now[0])
 def test_eight_locks_overlap_two_card_lock_mutually_exclusive(self):
  with tempfile.TemporaryDirectory() as td:
   with G.DeviceLocks(td,U) as held:
    self.assertEqual(len(held.evidence),8)
    with self.assertRaises(BlockingIOError):
     with G.DeviceLocks(td,U[6:]):pass
   with G.DeviceLocks(td,U[6:]):pass
 def test_source_identity_binds_v3_topology_and_schema_dependencies(self):
  with tempfile.TemporaryDirectory() as td:
   r=Path(td); names=['word_ft_v3/__init__.py','word_ft_v3/train.py','word_ft_v3/resources.py','word_ft_v3/topology.py','scripts/launch_word_v3.py','word_ft/__init__.py','word_ft/schema.py','duration_ft/__init__.py','duration_ft/config_checks.py','duration_ft/train.py','duration_ft/data.py','applications/speaker_rec/wespeaker_lite/models/projections.py','redimnet2/a.py']
   for n in names:p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
   a=G.training_source_identity(r);(r/'word_ft_v3/topology.py').write_text('changed');b=G.training_source_identity(r)
   self.assertNotEqual(a,b);self.assertIn('word_ft/schema.py',a);self.assertIn('duration_ft/train.py',a)
class LauncherTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  s=importlib.util.spec_from_file_location('launcher_v3_redteam',ROOT/'scripts/launch_word_v3.py');cls.L=importlib.util.module_from_spec(s);s.loader.exec_module(cls.L)
 def test_platform_rank_environment_refused_before_probe_or_spawn(self):
  for env in [{'WORLD_SIZE':'8','RANK':'0'},{'WORLD_SIZE':'1','RANK':'1'},{'LOCAL_RANK':'0'}]:
   with self.subTest(env=env),patch.dict(os.environ,env,clear=True),patch.object(self.L.G,'capture_snapshot',side_effect=AssertionError('no query')),patch.object(self.L.subprocess,'Popen',side_effect=AssertionError('no spawn')),self.assertRaises(G.AdmissionRefused):self.L.launch(SimpleNamespace())
 def test_worker_command_exact_W8_single_parent(self):
  a=SimpleNamespace(world_size=8,config='/tmp/c',dataset='/tmp/d',official='/tmp/o',output='/tmp/x',resume=None,stop_after=2);cmd=self.L.worker_command(a)
  for flag in ['--nnodes=1','--nproc-per-node=8','--run-path','--start-method=spawn','--worker']:self.assertIn(flag,cmd)
  self.assertEqual(cmd[cmd.index('--world-size')+1],'8')
 def test_child_env_clears_platform_rank_and_external_libraries(self):
  x={'WORLD_SIZE':'8','RANK':'5','LOCAL_RANK':'5','GROUP_RANK':'1','ROLE_RANK':'1','TORCHELASTIC_RUN_ID':'old','LD_PRELOAD':'bad','LD_AUDIT':'bad','PYTHONPATH':'bad','LD_LIBRARY_PATH':'external'}
  with patch.dict(os.environ,x):
   before=dict(os.environ);e=self.L.child_environment('/tmp/t','a'*64,snapshot()['devices'],{'decoder':{'library_root':'/private/lib'}});self.assertEqual(dict(os.environ),before)
  for k in ['WORLD_SIZE','RANK','LOCAL_RANK','GROUP_RANK','ROLE_RANK','TORCHELASTIC_RUN_ID','LD_PRELOAD','LD_AUDIT','PYTHONPATH']:self.assertNotIn(k,e)
  self.assertEqual(e['CUDA_VISIBLE_DEVICES'],','.join(U));self.assertTrue(e['LD_LIBRARY_PATH'].startswith('/private/lib:'))
@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'requires existing server CPU PyTorch; CUDA still mocked')
class TrainTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from word_ft_v3 import train
  cls.T=train
 def family(self,name='f',sha='a'*64,start=0):
  return [dict(family_id=name,n_units=n,speaker_id='speaker',audio_sha256=sha,audio_realpath='/immutable/source.wav',source_utterance_id='original',start_sample=start,end_sample=start+n*100,sample_rate=16000,source_num_samples=3000,source_channels=1,language='zh',counting_policy='han',unit_texts=list('甲乙丙丁戊己')[:n]) for n in [1,2,3,6]]
 def test_disguised_same_real_anchor_is_not_new_family(self):
  a=self.family();b=self.family('renamed');
  for r in b:r['speaker_id']='changed';r['source_utterance_id']='alias'
  with self.assertRaises(ValueError):self.T.build_family_index(a+b)
 def test_different_real_anchor_same_file_can_be_new_family(self):self.assertEqual(len(self.T.build_family_index(self.family()+self.family('other',start=1000))),2)
 def test_family_text_identity_geometry_and_count_coverage_refused(self):
  for mode in ['text','speaker','end','missing','duplicate']:
   rows=self.family()
   if mode=='text':rows[0]['unit_texts']=['错']
   elif mode=='speaker':rows[0]['speaker_id']='other'
   elif mode=='end':rows[1]['end_sample']=rows[0]['end_sample']
   elif mode=='missing':rows.pop()
   else:rows.append(copy.deepcopy(rows[0]))
   with self.subTest(mode=mode),self.assertRaises(ValueError):self.T.build_family_index(rows)
 def test_matched_family_selection_same_globalbatch_world_changes(self):
  rows=sum([self.family('f'+str(i),sha=f'{i:064x}') for i in range(19)],[]);f=self.T.build_family_index(rows)
  a={'seed':3,'items_per_rank':8,'sampling':'matched_prefix_family','unit_counts':[1,2,3,6],'unit_probabilities':[.25]*4};b=dict(a,items_per_rank=2)
  for step in range(20):
   aa=self.T.select_step(rows,f,a,step,2);bb=self.T.select_step(rows,f,b,step,8);self.assertEqual(aa,bb);self.assertEqual(len({rows[i]['family_id'] for i in aa}),16)
 def test_old_world_checkpoint_rejected_before_CUDA(self):
  T=self.T
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'.training.lock').touch();c=p/'train_step_2.pt';c.write_bytes(b'not-read')
   payload={'schema':T.CHECKPOINT_SCHEMA,'global_step':2,'runtime_identity':{'schema':T.RUNTIME_SCHEMA},'source_identity':{'f':'old'},'world_size':2,'topology':P.topology(2),'micro_cursor':0}
   with patch.object(T,'source_identity',return_value={'f':'old'}),patch.object(T.torch,'load',return_value=payload),patch.object(T.torch,'cuda') as cuda,self.assertRaisesRegex(ValueError,'world-size'):
    T.validate_resume_target(p,c,8,0)
   self.assertFalse(cuda.mock_calls)
 def test_CPU_checkpoint_rejected_before_CUDA(self):
  T=self.T
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'.training.lock').touch();c=p/'train_step_2.pt';c.write_bytes(b'not-read')
   with patch.object(T.torch,'load',return_value={'schema':'word-training/v1','global_step':2}),patch.object(T.torch,'cuda') as cuda,self.assertRaisesRegex(ValueError,'CPU/different'):
    T.validate_resume_target(p,c,8,0)
   self.assertFalse(cuda.mock_calls)
 def test_pinned_source_cannot_be_resealed_on_drift(self):
  T=self.T
  with patch.object(T,'_PINNED_SOURCE_IDENTITY',{'f':'old'}),patch.object(T,'source_identity',return_value={'f':'new'}),self.assertRaisesRegex(ValueError,'source drift'):T.checkpoint_payload(None,None,None,{},'d',8,1,{},[],{},'head')
 def test_decode_nonfinite_wave_refused(self):
  import numpy as np
  for wave in [np.array([1,float('nan')],np.float32),np.ones(3,np.float32),np.ones(2,np.float64)]:
   with self.subTest(wave=str(wave)),patch.object(self.T.schema,'read_unit',return_value=wave),self.assertRaises(ValueError):self.T._read_record({'n_samples_16k':2})
 def test_nonfinite_forward_enters_collective_before_backward_or_optimizer(self):
  import numpy as np
  T=self.T;torch=T.torch;model=Mock();model.return_value=(torch.tensor([[1.]]),torch.tensor([[1.,2.]]),torch.tensor(float('nan'),requires_grad=True));opt=Mock();scaler=Mock();scaler.is_enabled.return_value=False
  def check(err):
   if err:raise RuntimeError('collective refusal:'+err)
  with patch.object(T,'collective_check',side_effect=check) as gate,self.assertRaisesRegex(RuntimeError,'collective refusal'):
   T.accumulate_step(model,Mock(),opt,scaler,[np.ones(4,np.float32)],[0],torch.device('cpu'))
  self.assertEqual(gate.call_count,1);scaler.scale.assert_not_called();scaler.step.assert_not_called()
class EightTicketTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.stack=ExitStack();self.source={'f':'fixed'};self.pid=os.getpid()
  self.held=self.stack.enter_context(G.DeviceLocks(self.root/'locks',U));self.fds={x['fd']:x for x in self.held.evidence}
  read,stat,resolve=Path.read_text,Path.stat,Path.resolve
  def rd(p,*a,**kw):return 'fixture-boot' if str(p)=='/proc/sys/kernel/random/boot_id' else read(p,*a,**kw)
  def st(p,*a,**kw):
   if str(p).startswith(f'/proc/{self.pid}/fd/'):return os.fstat(int(p.name))
   return stat(p,*a,**kw)
  def rs(p,*a,**kw):
   if str(p).startswith(f'/proc/{self.pid}/fd/'):return Path(self.fds[int(p.name)]['path'])
   return resolve(p,*a,**kw)
  for q in [patch.object(G,'process_start_time',return_value=123),patch.object(G.socket,'gethostname',return_value='host'),patch.object(G.time,'monotonic',return_value=100),patch.object(Path,'read_text',rd),patch.object(Path,'stat',st),patch.object(Path,'resolve',rs)]:self.stack.enter_context(q)
  tt=ticket();tt['probe']['devices']=tt['devices'];self.t=G.create_ticket(tt['probe'],self.held,4096,2048,256,self.source);self.path=self.root/'ticket.json'
 def tearDown(self):self.stack.close();self.tmp.cleanup()
 def check(self):
  self.path.write_text(G.canonical(self.t));return G.validate_ticket(self.path,G.sha_file(self.path),self.source)
 def test_eight_live_locks_and_ticket_pass(self):self.assertEqual(len(self.check()['locks']),8)
 def test_seven_locks_never_authorize_eight_devices(self):
  self.t['locks']=self.t['locks'][:7]
  with self.assertRaises(G.AdmissionRefused):self.check()
 def test_wrong_host_boot_age_source_and_parent_reused(self):
  for k,v in [('hostname','foreign'),('boot_id','foreign'),('created_monotonic',-1),('parent_start_time',124),('source_identity',{'new':'code'})]:
   old=copy.deepcopy(self.t);self.t[k]=v
   with self.subTest(k=k),self.assertRaises(G.AdmissionRefused):self.check()
   self.t=old
 def test_fd_inode_tamper_and_open_unlocked_not_ownership(self):
  self.t['locks'][7]['inode']=0
  with self.assertRaises(G.AdmissionRefused):self.check()
  self.t['locks']=copy.deepcopy(self.held.evidence);fd=self.held.evidence[7]['fd'];G.fcntl.flock(fd,G.fcntl.LOCK_UN)
  with self.assertRaises(G.AdmissionRefused):self.check()
 def test_models_and_original_training_output_paths_protected(self):
  for p in ['/root/group-shared/voiceprint/models/a','/shared/group-shared/voiceprint/models/a','/root/experiments/redimnet2/b6-mix-official-20261019/a']:
   with self.subTest(p=p),self.assertRaises(G.AdmissionRefused):G.protected_path(p)
if __name__=='__main__':unittest.main()
