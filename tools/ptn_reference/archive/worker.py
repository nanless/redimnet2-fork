#!/usr/bin/env python3
"""Single torchrun rank; gates the frozen trainer before any model forward/update."""
import argparse,datetime,functools,importlib,json,os,sys,traceback,fcntl
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import strict_restore as S

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--bundle-sha',required=True);a=p.parse_args()
 S.need(S.sha(S.BUNDLE/'bundle_files.json')==a.bundle_sha,'worker bundle manifest differs from parent pin')
 S.verify_bundle();S.verify_source()
 # Independent SH ownership survives loss of both launcher processes.
 worker_locks=[]
 for path in [S.RUN.parent/'.ptn-logical-chain.lock',S.OLD/'.reproduction.lock',S.OLD/'ptn/.cloud_train.lock']:
  S.need(path.is_file() and not path.is_symlink(),'stable worker lock missing')
  h=path.open('rb');fcntl.flock(h,fcntl.LOCK_SH|fcntl.LOCK_NB);worker_locks.append(h)
 sys.path[:0]=[str(S.SOURCE),str(S.SOURCE/'applications/speaker_rec')]
 import torch,torch.distributed as dist,yaml
 from types import SimpleNamespace
 from decoder_runtime import audit_loaded_decoder
 decoder=audit_loaded_decoder(SimpleNamespace(decoder_library=str(S.DECODER)))
 rank=int(os.environ['RANK']);S.need(int(os.environ['WORLD_SIZE'])==8 and int(os.environ['LOCAL_RANK'])==rank,'one node W8 exact ranks required')
 flags={'ready':False,'first_update':False}
 def block_forward(module,args):S.need(flags['ready'],'model forward attempted before all-rank strict restoration')
 global_hook=torch.nn.modules.module.register_module_forward_pre_hook(block_forward)
 original_step=torch.optim.SGD.step
 @functools.wraps(original_step)
 def guarded_step(optimizer,*args,**kwargs):
  S.need(flags['ready'],'optimizer update attempted before all-rank strict restoration')
  result=original_step(optimizer,*args,**kwargs)
  if not flags['first_update']:
   flags['first_update']=True
   receipt={'rank':rank,'first_optimizer_step_completed':True,'epoch':36,'absolute_iteration':flags['current_iteration'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   (S.RUN/'checks'/f'first_update_rank{rank}.json').write_text(json.dumps(receipt,indent=2)+'\n')
   print('STRICT_FIRST_UPDATE '+json.dumps(receipt),flush=True)
  return result
 torch.optim.SGD.step=guarded_step
 train=importlib.import_module('wespeaker_lite.bin.train')
 original_loop=train.training_loop
 payload=S.checkpoint()
 def strict_loop(start_epoch,configs,train_dataset,train_dataloader,epoch_iter,ddp_model,model,criterion,optimizer,scaler,scheduler,margin_scheduler,rank,world_size,gpu,logger,log_df,val_dls_dict,val_type,model_dir):
  error=None;proof=None
  try:
   S.verify_bundle();source=S.verify_source();imports=S.verify_imports()
   proof=S.verify_restored(model,optimizer,scaler,payload,start_epoch,epoch_iter,world_size,configs,S.RUN/'ptn/spk2id.json')
   scheduler.step(525035);margin_scheduler.step(525035)
   S.need(abs(scheduler.get_lr()-.72)<1e-12 and all(abs(g['lr']-.72)<1e-12 for g in optimizer.param_groups),'first restored learning rate differs')
   S.need(abs(margin_scheduler.get_margin()-.2)<1e-12,'first restored margin differs')
   proof.update(independently_held_shared_locks=[str(h.name) for h in worker_locks],rank=rank,world_size=world_size,lr=scheduler.get_lr(),margin=margin_scheduler.get_margin(),source=source,imports=imports,decoder=decoder,device=str(next(model.parameters()).device),driver_runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  except Exception:error=traceback.format_exc()
  all_results=[None]*8;dist.all_gather_object(all_results,{'rank':rank,'error':error,'restored':bool(proof and proof.get('restored'))})
  S.need(all(x and x['rank']==j and x['error'] is None and x['restored'] for j,x in enumerate(all_results)),'all-rank restoration refused: '+json.dumps(all_results))
  out=S.RUN/'checks'/f'restore_verified_rank{rank}.json'
  with out.open('x') as f:f.write(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
  dist.barrier(device_ids=[gpu]);flags['ready']=True;global_hook.remove()
  print('STRICT_RESTORE_VERIFIED '+json.dumps({'rank':rank,'epoch':36,'absolute_iteration':525035,'model_state_keys':823,'optimizer_states':584,'scale':scaler.state_dict()['scale'],'lr':.72,'margin':.2}),flush=True)
  original_scheduler_step=scheduler.step
  def tracked_scheduler_step(current_iter):
   flags['current_iteration']=current_iter
   return original_scheduler_step(current_iter)
  scheduler.step=tracked_scheduler_step
  # Keep epoch guards in the rank process, never inside the pickled dataset graph.
  from epoch_guard import guarded_run_epoch
  original_run_epoch=train.run_epoch
  train.run_epoch=functools.partial(guarded_run_epoch,original_run_epoch)
  try:return original_loop(start_epoch,configs,train_dataset,train_dataloader,epoch_iter,ddp_model,model,criterion,optimizer,scaler,scheduler,margin_scheduler,rank,world_size,gpu,logger,log_df,val_dls_dict,val_type,model_dir)
  finally:
   train.run_epoch=original_run_epoch
   torch.optim.SGD.step=original_step
 original_save=train.save_checkpoint
 def checked_save(*args,**kwargs):
  S.verify_bundle();S.verify_source()
  return original_save(*args,**kwargs)
 train.save_checkpoint=checked_save
 train.training_loop=strict_loop
 train.train(config=a.config)
 S.need(flags['first_update'],'worker completed without a real optimizer update')
 if dist.is_initialized():dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 from torch.distributed.elastic.multiprocessing.errors import record
 record(main)()
