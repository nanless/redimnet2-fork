"""Rank-local epoch gate; deliberately never attached to a Dataset instance."""
import strict_restore as S

def guarded_run_epoch(original,*args,**kwargs):
 # Frozen training_loop calls this immediately after dataset.set_epoch and before
 # executor starts DataLoader workers / consumes a batch / forwards a model.
 S.verify_bundle();S.verify_source();S.need(S.sha(S.CK)==S.CK_SHA,'parent model35 changed')
 from loader_worker_audit import AuditedLoader
 import torch.distributed as dist
 loader=args[0];epoch=args[7];rank=args[8];world=dist.get_world_size()
 S.need(world==8 and loader.num_workers==24,'formal loader audit requires W8 and24 workers')
 wrapped=AuditedLoader(loader,epoch,rank,world,S.RUN/'checks/loader_workers')
 return original(wrapped,*args[1:],**kwargs)
