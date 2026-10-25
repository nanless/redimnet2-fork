"""CPU-only real model35 strict restore and decoder checks; no training output."""
import contextlib,io,json,os,sys,time
from pathlib import Path
import strict_restore as S

def run(full_audio=True):
 S.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight requires CUDA_VISIBLE_DEVICES empty')
 S.verify_bundle();source=S.verify_source();dependencies=S.verify_dependencies()
 import torch,yaml
 torch.set_num_threads(1)
 S.need(not torch.cuda.is_initialized(),'CUDA already initialized')
 import decoder_runtime,audio_review
 from types import SimpleNamespace
 decoder=decoder_runtime.audit_loaded_decoder(SimpleNamespace(decoder_library=str(S.DECODER)))
 audio=audio_review.audit_audio_files(json.loads((S.BUNDLE/'audio_selfcheck.json').read_text())) if full_audio else None
 sys.path[:0]=[str(S.SOURCE),str(S.SOURCE/'applications/speaker_rec')]
 from wespeaker_lite.bin import train as T
 config=yaml.safe_load((S.BUNDLE/'resume.yaml').read_text());S.validate_config(config)
 before=S.snapshot(S.CK);out_before=S.RUN.exists();S.need(not out_before,'new output must not exist')
 payload=S.checkpoint();started=time.monotonic();stdout=io.StringIO()
 with contextlib.redirect_stdout(stdout):
  model=T.SpeakerModel(T.get_speaker_model(config['model'])(**config['model_args']),T.get_projection(config['projection_args']))
  model.load_state_dict(payload['model'],strict=True)
  optimizer=getattr(torch.optim,config['optimizer'])([{'params':model.backbone.parameters(),'weight_decay':config['optimizer_args']['weight_decay']},{'params':model.projection.parameters(),'weight_decay':config['loss_weight_decay_mult']*config['optimizer_args']['weight_decay']}],**config['optimizer_args'])
  scheduler=getattr(T.schedulers,config['scheduler'])(optimizer,**config['scheduler_args'])
  margin=getattr(T.schedulers,config['margin_scheduler'])(model=model,**config['margin_update'])
  keep=[{k:v for k,v in g.items() if k not in ('params','lr')} for g in optimizer.param_groups]
  optimizer.load_state_dict(payload['optimizer'])
  for group,hyper in zip(optimizer.param_groups,keep):group.update(hyper)
  scaler=torch.amp.GradScaler('cpu',enabled=True);scaler.load_state_dict(payload['scaler'])
  proof=S.verify_restored(model,optimizer,scaler,payload,36,15001,8,config,S.OLD/'ptn/spk2id.json')
  scheduler.step(525035);margin.step(525035)
  S.need(abs(scheduler.get_lr()-.72)<1e-12 and abs(margin.get_margin()-.2)<1e-12,'actual frozen scheduler clock differs')
  proof.update(actual_lr=scheduler.get_lr(),actual_margin=margin.get_margin(),actual_optimizer_lrs=[g['lr'] for g in optimizer.param_groups],actual_scaler=scaler.state_dict(),actual_imports=S.verify_imports())
 S.need(before==S.snapshot(S.CK) and not S.RUN.exists(),'old checkpoint changed/new output appeared')
 S.need(not torch.cuda.is_initialized() and not torch.distributed.is_initialized(),'CPU check initialized GPU/NCCL')
 return {'ok':True,'cpu_only':True,'dependencies':dependencies,'source':source,'decoder':decoder,'audio':audio,'strict_restoration':proof,'checkpoint_stat_before_after':before,'new_run_exists_before':out_before,'new_run_exists_after':S.RUN.exists(),'cuda_initialized':False,'torchrun_spawned':False,'model_forward_or_update':False,'actual_restore_seconds':time.monotonic()-started,'model_constructor_stdout':stdout.getvalue()}
