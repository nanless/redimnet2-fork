"""Read-only fork worker evidence, gathered before any batch is yielded to trainer."""
import datetime,hashlib,json,multiprocessing as mp,os,secrets,time,traceback
from pathlib import Path
import torch,torch.distributed as dist
import strict_restore as S

def atomic_new(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 with tmp.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 try:os.link(tmp,path)
 finally:tmp.unlink()
def start_ticks():
 raw=Path('/proc/self/stat').read_text();return int(raw[raw.rfind(')')+2:].split()[19])
def audited_seed_worker(worker_id):
 directory=Path(os.environ['PTN_LOADER_AUDIT_DIRECTORY']);rank=int(os.environ['RANK'])
 try:
  from wespeaker_lite.utils.utils import seed_worker
  seed_worker(worker_id)  # Exact original call, prior to all read-only observations.
  from wespeaker_lite.dataset import augs
  info=torch.utils.data.get_worker_info();expected_world=int(os.environ['WORLD_SIZE']);expected_workers=int(os.environ['NUM_DATALOADER_WORKERS'])
  S.need(info is not None and info.id==worker_id and info.num_workers==expected_workers,'loader worker identity differs')
  S.need(dist.is_initialized(),'loader child lost distributed parent state')
  actual_rank=dist.get_rank();actual_world=dist.get_world_size();context=mp.get_start_method()
  source=info.dataset
  while hasattr(source,'source'):source=source.source
  S.need(type(source).__name__=='DataList' and type(source).__module__=='wespeaker_lite.dataset.dataset','unexpected innermost dataset')
  evidence={'schema':'ptn-loader-worker-audit/v1','rank':rank,'world_size':expected_world,'worker_id':worker_id,'num_workers':info.num_workers,'actual_dist_rank':actual_rank,'actual_dist_world_size':actual_world,'actual_context':context,'actual_dist_initialized':True,'sampler_inherited_rank':source.sampler.rank,'sampler_inherited_world_size':source.sampler.world_size,'sampler_partition':source.sampler.partition,'sampler_epoch':source.sampler.epoch,'legacy_aug_behavior':augs.LEGACY_AUG_BEHAVIOR,'pid':os.getpid(),'ppid':os.getppid(),'start_ticks':start_ticks(),'nonce':os.environ['PTN_LOADER_AUDIT_NONCE'],'epoch':int(os.environ['PTN_LOADER_AUDIT_EPOCH']),'bundle_sha256':os.environ['PTN_BUNDLE_MANIFEST_SHA'],'original_seed_worker_called':True,'worker_collective_called':False,'worker_GPU_operation_called':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  S.need(actual_rank==rank and actual_world==expected_world and context=='fork','loader distributed identity/context differs from parent')
  S.need(source.sampler.rank==rank and source.sampler.world_size==expected_world and source.sampler.partition is True,'inherited sampler sharding differs')
  S.need(source.sampler.epoch==int(os.environ['PTN_LOADER_AUDIT_EPOCH']),'actual sampler epoch differs')
  S.need(augs.LEGACY_AUG_BEHAVIOR is True,'legacy augmentation global was not inherited')
  atomic_new(directory/f'worker{worker_id}.json',evidence)
 except BaseException:
  try:atomic_new(directory/f'error_worker{worker_id}.json',{'rank':rank,'worker_id':worker_id,'traceback':traceback.format_exc(),'epoch':os.environ.get('PTN_LOADER_AUDIT_EPOCH')})
  except BaseException:pass
  raise

class AuditedLoader:
 def __init__(self,loader,epoch,rank,world,root):self.loader=loader;self.epoch=epoch;self.rank=rank;self.world=world;self.root=Path(root)
 def __len__(self):return len(self.loader)
 def __iter__(self):
  from wespeaker_lite.utils.utils import seed_worker
  S.need(self.loader.worker_init_fn is seed_worker and not self.loader.persistent_workers,'unexpected seed hook/persistent loader')
  n=self.loader.num_workers;directory=self.root/f'epoch{self.epoch}'/f'rank{self.rank}';directory.mkdir(parents=True,exist_ok=False);nonce=secrets.token_hex(24)
  os.environ.update(PTN_LOADER_AUDIT_DIRECTORY=str(directory),PTN_LOADER_AUDIT_NONCE=nonce,PTN_LOADER_AUDIT_EPOCH=str(self.epoch))
  self.loader.worker_init_fn=audited_seed_worker;iterator=None;error=None;proofs=[]
  try:
   try:
    iterator=iter(self.loader);deadline=time.monotonic()+180.
    while True:
     errors=list(directory.glob('error_worker*.json'));S.need(not errors,'loader child rejected its inherited state: '+str(errors))
     ready=[directory/f'worker{i}.json' for i in range(n)]
     if all(p.is_file() for p in ready):
      proofs=[]
      for i,p in enumerate(ready):
       raw=p.read_bytes();v=json.loads(raw);S.need(v['rank']==self.rank and v['world_size']==self.world and v['worker_id']==i and v['num_workers']==n and v['epoch']==self.epoch and v['nonce']==nonce and v['bundle_sha256']==os.environ['PTN_BUNDLE_MANIFEST_SHA'],'loader evidence differs')
       S.need(v['actual_dist_rank']==self.rank and v['actual_dist_world_size']==self.world and v['actual_context']=='fork' and v['legacy_aug_behavior'] is True and v['sampler_inherited_rank']==self.rank and v['sampler_inherited_world_size']==self.world and v['sampler_epoch']==self.epoch and v['ppid']==os.getpid(),'loader actual sharding/augmentation evidence differs')
       proofs.append({'worker_id':i,'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'pid':v['pid']})
      S.need(len({v['pid'] for v in proofs})==n,'duplicate loader process identity')
      break
     S.need(time.monotonic()<deadline,'timed out waiting for all local loader proofs');time.sleep(.05)
   except BaseException:error=traceback.format_exc()
   local={'nonce':nonce,'parent_rank_pid':os.getpid(),'rank':self.rank,'world_size':self.world,'epoch':self.epoch,'worker_count':len(proofs),'expected_workers':n,'error':error,'proofs':proofs}
   gathered=[None]*self.world;dist.all_gather_object(gathered,local)  # Parent ranks only; never called by loader children.
   S.need(all(x and x['rank']==i and x['world_size']==self.world and x['epoch']==self.epoch and x['worker_count']==n and x['error'] is None for i,x in enumerate(gathered)),'all-rank loader audit refused: '+json.dumps(gathered))
   gate={'schema':'ptn-loader-gate/v1','rank':self.rank,'epoch':self.epoch,'workers_per_rank':n,'world_size':self.world,'global_verified_workers':sum(x['worker_count'] for x in gathered),'ranks':gathered,'before_first_batch_yield':True}
   atomic_new(self.root/f'gate_epoch{self.epoch}_rank{self.rank}.json',gate);print('STRICT_LOADER_WORKERS_VERIFIED '+json.dumps({k:v for k,v in gate.items() if k!='ranks'}),flush=True)
   yield from iterator
  finally:
   self.loader.worker_init_fn=seed_worker
   if iterator is not None:iterator._shutdown_workers()
