#!/usr/bin/env python3
"""One independent PTN successor. Explicit execute, one W8 torchrun, no LM."""
import argparse,contextlib,datetime,fcntl,json,os,re,subprocess,sys,traceback,signal,time,secrets
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import strict_restore as S
PYTHON='/root/miniforge3/envs/redimnet2/bin/python'
TERMINAL_SHA='6a89de3e465417521a16414a43bf6eafd25332ff1fd6692cccb36de85570c0dd'
FAILED_TERMINAL_SHA='ae1da00db5cbe324620f301b9f4c17b4e47c9adcb86a75f1178c48569d89a373'
INFRA=['/usr/local/nccl-rdma-sharp-plugins/lib','/usr/local/nvidia/lib','/usr/local/nvidia/lib64']

def ensure_shared_alias():
 target=Path('/shared/group-shared');alias=Path('/root/group-shared')
 S.need(target.is_dir(),'shared group mount missing')
 S.need(any(line.split()[4]=='/shared/group-shared' for line in Path('/proc/self/mountinfo').read_text().splitlines()),'shared group is not the expected mounted path')
 if alias.exists() or alias.is_symlink():
  S.need(alias.is_dir() and alias.samefile(target),'group-shared alias conflicts; never overwrite existing content')
 else:alias.symlink_to(target,target_is_directory=True)
 return {'path':str(alias),'resolved':str(alias.resolve()),'target':str(target),'readonly_expected_from_platform':True}

def own_members(pgid,started,nonce):
 members=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   raw=(path/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
   if parts[0]=='Z' or int(parts[2])!=pgid:continue
   S.need(int(parts[3])==pgid and int(parts[19])>=started,'unproven process in own session')
   S.need(('PTN_RESUME_NONCE='+nonce).encode() in (path/'environ').read_bytes().split(b'\0'),'unproven nonce in own session')
   members.append(int(path.name))
  except (FileNotFoundError,ProcessLookupError):continue
 return members

def stop_own(pgid,started,nonce,child):
 if not own_members(pgid,started,nonce):return
 os.killpg(pgid,signal.SIGTERM)
 end=time.monotonic()+10
 while time.monotonic()<end:
  child.poll()
  if not own_members(pgid,started,nonce):return
  time.sleep(.1)
 if own_members(pgid,started,nonce):os.killpg(pgid,signal.SIGKILL)
 end=time.monotonic()+5
 while time.monotonic()<end:
  child.poll()
  if not own_members(pgid,started,nonce):return
  time.sleep(.1)
 S.need(not own_members(pgid,started,nonce),'own workers remained; preserve locks and failure evidence')

def cleanup_with_locks_held(pgid,started,nonce,child):
 previous_term=signal.signal(signal.SIGTERM,lambda signum,frame:None)
 try:
  try:stop_own(pgid,started,nonce,child)
  except BaseException as exc:
   print('CLEANUP_UNPROVEN_KEEPING_ALL_LOCKS '+repr(exc),flush=True)
   while True:
    active=[]
    for path in Path('/proc').iterdir():
     if not path.name.isdigit():continue
     try:
      raw=(path/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
      if parts[0]!='Z' and int(parts[2])==pgid:active.append(int(path.name))
     except (FileNotFoundError,ProcessLookupError):continue
    child.poll()
    if not active:break
    time.sleep(5)
 finally:signal.signal(signal.SIGTERM,previous_term)

def verify_terminal():
 p=S.BUNDLE/'parent_terminal_attestation.json';S.need(S.sha(p)==TERMINAL_SHA,'terminal evidence bytes differ')
 e=json.loads(p.read_text());S.need(e['task_id']=='t-20261019232824-8zqrl' and e['state']=='Terminated' and e['active_workers']==0 and e['worker_states']==['Terminated'],'original task/worker not terminal')
 age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(e['observed_utc'].replace('Z','+00:00'))).total_seconds()
 S.need(-300<=age<=86400,'terminal control-plane evidence stale/future; obtain fresh evidence without changing old run')
 failed_path=S.BUNDLE/'failed_task_terminal.json';S.need(S.sha(failed_path)==FAILED_TERMINAL_SHA,'failed predecessor terminal evidence bytes differ')
 failed=json.loads(failed_path.read_text());S.need(failed['task_id']=='t-20261024115909-mhczc' and failed['state']=='Failed' and failed['active_workers']==0 and len(failed['worker_states'])==1 and failed['worker_states'][0]['name']=='worker_0' and failed['worker_states'][0]['state']=='Failed','latest failed predecessor not terminal')
 failed_age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(failed['observed_utc'].replace('Z','+00:00'))).total_seconds()
 S.need(-300<=failed_age<=86400,'failed predecessor control-plane evidence stale/future')
 return {'task_id':e['task_id'],'state':e['state'],'active_workers':0,'observed_utc':e['observed_utc'],'attestation_sha256':TERMINAL_SHA,'latest_failed_predecessor':{'task_id':failed['task_id'],'state':failed['state'],'active_workers':failed['active_workers'],'observed_utc':failed['observed_utc'],'attestation_sha256':FAILED_TERMINAL_SHA},'direct_pid_probe_complete':False,'admission_basis':'control-plane worker terminated plus both original locks held atomically'}

@contextlib.contextmanager
def old_locks():
 handles=[]
 try:
  for rel in ['.reproduction.lock','ptn/.cloud_train.lock']:
   p=S.OLD/rel;S.need(p.is_file() and not p.is_symlink(),'original real lock missing: '+str(p))
   h=p.open('rb');handles.append(h);fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield handles
 finally:
  for h in reversed(handles):fcntl.flock(h,fcntl.LOCK_UN);h.close()

def environment():
 env=os.environ.copy()
 for key in list(env):
  if key.startswith(('SKIP_','PET_','TORCHELASTIC_')) or key in {'PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','MODEL_CKPT','CHECKPOINT','EXP_DIR','RUN_ROOT','RECIPE','NUM_EPOCHS','STAGE','SAMPLE_NUM_PER_EPOCH','BATCH_SIZE'}:env.pop(key,None)
 env.update(PYTHONPATH=str(S.SOURCE)+':'+str(S.SOURCE/'applications/speaker_rec'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',LD_LIBRARY_PATH=':'.join([str(S.DECODER.parent)]+INFRA),PTN_BUNDLE_MANIFEST_SHA=S.sha(S.BUNDLE/'bundle_files.json'))
 return env

def postflight():
 import torch,csv,math
 S.verify_bundle();S.verify_source()
 proofs=[]
 for r in range(8):
  proof=json.loads((S.RUN/'checks'/f'restore_verified_rank{r}.json').read_text());update=json.loads((S.RUN/'checks'/f'first_update_rank{r}.json').read_text())
  S.need(proof['rank']==r and proof['restored'] and update['first_optimizer_step_completed'],'missing restored/updated rank proof');proofs.append(proof)
 logs=[]
 for name in ['stdout.log','stderr.log']:
  for p in (S.RUN/'rank_logs').rglob(name):
   data=p.read_text(errors='replace');S.need(not re.search(r'read_failed|Traceback \(most recent call last\)|could NOT restore|freshly initialised|丢弃了|Non-finite training',data,re.I),'new run contains unreviewed failure: '+str(p));logs.append(str(p))
 S.need(len(logs)==16,'exactly eight rank stdout+stderr logs required')
 final=S.RUN/'ptn/models/model_120.pt';stat=S.snapshot(final);p=torch.load(final,map_location='cpu',mmap=True,weights_only=False)
 S.need(p.get('epoch')==120 and len(p['model'])==823 and len(p['optimizer']['state'])==584 and p.get('scaler'),'PTN120 final checkpoint incomplete')
 for value in list(p['model'].values())+[v for d in p['optimizer']['state'].values() for v in d.values()]:
  if torch.is_tensor(value) and value.is_floating_point():S.need(bool(torch.isfinite(value).all()),'final checkpoint nonfinite')
 S.need(stat==S.snapshot(final),'final checkpoint changed during validation')
 rows=list(csv.DictReader((S.RUN/'ptn/val_log.csv').open()));S.need([int(float(x['epoch'])) for x in rows]==list(range(36,121)),'new validation epoch lineage not exactly36..120')
 S.need(all(math.isfinite(float(x['vox1-eer-8s'])) and 0<=float(x['vox1-eer-8s'])<=100 for x in rows),'invalid final validation')
 return {'ptn_complete':True,'epochs':list(range(36,121)),'checkpoint':str(final),'checkpoint_sha256':S.sha(final),'old_checkpoint_sha256':S.CK_SHA,'rank_restore_proofs':8,'rank_first_updates':8,'lm_started':False,'lm_requires_separate_cross_segment_contract':True,'final_evaluation_complete':False}

def execute():
 S.need(int(os.environ.get('WORLD_SIZE','1'))==1 and int(os.environ.get('RANK','0'))==0 and 'LOCAL_RANK' not in os.environ,'platform parent must execute only once outside a rank group')
 S.need(not S.RUN.exists(),'new run already exists; never restart platform entry')
 S.verify_bundle()
 from predecessor_gate import wait_for_terminal,verify_unchanged
 predecessor=wait_for_terminal()
 terminal=verify_terminal();mount=ensure_shared_alias();S.verify_source()
 import yaml
 S.validate_config(yaml.safe_load((S.BUNDLE/'resume.yaml').read_text()))
 parent=S.RUN.parent;lock=parent/'.ptn-logical-chain.lock'
 admission=parent/'.ptn-resume-admission.lock'
 with contextlib.ExitStack() as stack:
  admit=stack.enter_context(admission.open('a+b'));fcntl.flock(admit,fcntl.LOCK_EX|fcntl.LOCK_NB)
  logical=stack.enter_context(lock.open('a+b'));fcntl.flock(logical,fcntl.LOCK_EX|fcntl.LOCK_NB)
  old_handles=stack.enter_context(old_locks())
  terminal=verify_terminal();verify_unchanged(predecessor)
  previous_visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
  os.environ['CUDA_VISIBLE_DEVICES']=''
  try:
   from cpu_check import run
   cpu_admission=run()
  finally:
   if previous_visibility is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
   else:os.environ['CUDA_VISIBLE_DEVICES']=previous_visibility
  S.need(S.sha(S.CK)==S.CK_SHA,'model35 changed after CPU admission')
  S.RUN.mkdir();(S.RUN/'ptn').mkdir();(S.RUN/'checks').mkdir()
  with (S.RUN/'.reproduction.lock').open('a+b') as n1,(S.RUN/'ptn/.cloud_train.lock').open('a+b') as n2:
   fcntl.flock(n1,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(n2,fcntl.LOCK_EX|fcntl.LOCK_NB)
   (S.RUN/'checks/parent_admission.json').write_text(json.dumps({'terminal':terminal,'latest_predecessor_receipt':predecessor,'old_locks_held':True,'mount':mount,'cpu_admission':cpu_admission,'logical_lock':str(lock),'checkpoint_sha256':S.CK_SHA,'source':S.verify_source(),'bundle_manifest_sha256':S.sha(S.BUNDLE/'bundle_files.json')},ensure_ascii=False,indent=2)+'\n')
   cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8','--max-restarts=0','--log-dir',str(S.RUN/'rank_logs'),'--redirects','3','--tee','3',str(S.BUNDLE/'worker.py'),'--config',str(S.BUNDLE/'resume.yaml'),'--bundle-sha',S.sha(S.BUNDLE/'bundle_files.json')]
   # While admission EX prevents a second successor, downgrade only after exclusive admission.
   # Each worker independently takes SH on these same stable files before construction.
   for h in [logical,*old_handles]:fcntl.flock(h,fcntl.LOCK_SH)
   fds=[h.fileno() for h in [admit,logical,*old_handles,n1,n2]]
   for fd in fds:os.set_inheritable(fd,True)
   print('STRICT_PARENT_ADMITTED '+json.dumps({'old_locks_held':True,'cpu_restore_verified':True,'decoder_windows_verified':True,'output':str(S.RUN),'epoch':36,'absolute_iteration':525035}),flush=True)
   nonce=secrets.token_hex(24);env=environment();env['PTN_RESUME_NONCE']=nonce
   with (S.RUN/'launch.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=S.SOURCE,env=env,pass_fds=fds,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    started=int(time.monotonic()*os.sysconf('SC_CLK_TCK'))-1
    def terminate(signum,frame):raise KeyboardInterrupt('parent signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,terminate)
    try:
     try:
      raw=Path(f'/proc/{child.pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split();started=int(parts[19])
      S.need(int(parts[2])==child.pid and int(parts[3])==child.pid,'new torchrun session not owned')
     except FileNotFoundError:pass
     for line in child.stdout:
      log.write(line);log.flush();print(line,end='',flush=True)
      S.need(not re.search(r'read_failed|Traceback \(most recent call last\)|could NOT restore|freshly initialised|丢弃了|Non-finite training',line,re.I),'new PTN failure requires separate review; abort own group')
     rc=child.wait();S.need(rc==0,'own torchrun failed: '+str(rc))
     S.need(not own_members(child.pid,started,nonce),'torchrun exited with residual workers')
    finally:
     cleanup_with_locks_held(child.pid,started,nonce,child);signal.signal(signal.SIGTERM,previous)
   result=postflight()
   with (S.RUN/'ptn_success.json').open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
   print(json.dumps(result,ensure_ascii=False,indent=2))

def main():
 p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--preflight-only',action='store_true');m.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.preflight_only:
  os.environ['CUDA_VISIBLE_DEVICES']=''
  from cpu_check import run
  with old_locks():result=run()
  result['terminal']=verify_terminal();result['original_locks_acquired_without_modification']=True
  print(json.dumps(result,ensure_ascii=False,indent=2))
 else:execute()
if __name__=='__main__':
 try:main()
 except BaseException:
  traceback.print_exc();sys.exit(2)
