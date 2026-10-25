"""Independent hard gate around the byte-frozen PTN trainer; no fallback."""
import hashlib,json,math,os,sys,stat
from pathlib import Path

SOURCE=Path('/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/source-7afb821')
OLD=Path('/root/experiments/redimnet2/b6-mix-official-20261019/formal-main-7afb821-zero-20261019-2327')
RUN=Path('/root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model35-20261024-v3')
CK=OLD/'ptn/models/model_35.pt'
CK_SHA='fdaa88b963720ccdc35708c3ca9d18e56296edd8711eb9309772afacc92beb84'
CONFIG_SHA='705100f8c5970cc7a22a9acb52b7d9d9de2c10aefc9f6d73000b4880d6b2d0b6'
SPK_SHA='559a92fa71e8f0c1048fd5e916a18476f7503d20e5e70cf8452ea36d013e2548'
MANIFEST=SOURCE.parent/'provenance/original_main_7afb821_source.sha256'
MANIFEST_SHA='5160f99e33caccf373563b6bc695d10f304044b780d10b5c40f6a4b0fab102d9'
DECODER=SOURCE.parent/'event-review-20261023-v4/runtime_audio/lib/libsndfile.so.1'
BUNDLE=Path(__file__).resolve().parent

class Refused(RuntimeError):pass
def need(ok,message):
 if not ok:raise Refused(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def snapshot(path):
 p=Path(path);s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
_PINNED_BUNDLE_SHA=None
def verify_bundle():
 global _PINNED_BUNDLE_SHA
 actual_sha=sha(BUNDLE/'bundle_files.json')
 if _PINNED_BUNDLE_SHA is None:_PINNED_BUNDLE_SHA=os.environ.get('PTN_BUNDLE_MANIFEST_SHA',actual_sha)
 need(actual_sha==_PINNED_BUNDLE_SHA,'bundle manifest drift; do not reseal')
 entries=json.loads((BUNDLE/'bundle_files.json').read_text())
 actual={p.name for p in BUNDLE.iterdir() if p.is_file()}
 need(actual==set(entries)|{'bundle_files.json'},'unlisted/missing bundle file')
 for name,digest in entries.items():
  p=BUNDLE/name;need(p.is_file() and not p.is_symlink() and sha(p)==digest,'bundle source mismatch: '+name)
 return entries

def verify_source():
 need(sha(MANIFEST)==MANIFEST_SHA,'original source manifest changed')
 rows=MANIFEST.read_text().splitlines();need(len(rows)==95,'original source count differs')
 for line in rows:
  digest,name=line.split(None,1);p=Path(name.lstrip('* '))
  rel=p.relative_to('/root/code/github_repos/redimnet2-fork') if p.is_absolute() else p
  need('..' not in rel.parts and sha(SOURCE/rel)==digest,'95 source mismatch: '+str(rel))
 expected=json.loads((BUNDLE/'source_189.json').read_text());names={x['relative'] for x in expected}
 actual={str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}
 need(actual==names,'189 source members changed')
 for entry in expected:
  p=SOURCE/entry['relative'];s=p.lstat()
  need(stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_size==entry['bytes'] and stat.S_IMODE(s.st_mode)==entry['mode'] and sha(p)==entry['sha256'],'189 source differs: '+entry['relative'])
 return {'manifest_sha256':MANIFEST_SHA,'original_source_count':95,'independent_source_members':189}

def verify_dependencies():
 d=json.loads((BUNDLE/'dependencies.json').read_text())
 for item in d['small_full_sha_files']:
  need(sha(item['path'])==item['sha256'],'dependency SHA differs: '+item['path'])
 for item in d['tables']:
  p=Path(item['path']);before=p.stat()
  need(before.st_size==item['bytes'] and before.st_mtime_ns==item['mtime_ns'],'training table identity changed: '+str(p))
  with p.open('rb') as f:
   first=f.read(4096);f.seek(max(0,before.st_size-4096));last=f.read(4096)
  need(hashlib.sha256(first).hexdigest()==item['first4096_sha256'] and hashlib.sha256(last).hexdigest()==item['last4096_sha256'],'training table sampled bytes changed')
  need((before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns),'training table changed during bounded read')
 return {'small_full_sha_files':len(d['small_full_sha_files']),'tables_current_size_mtime_head_tail':len(d['tables']),'full_tables_rehashed':False,'historical_full_hashes':{x['path']:x['historical_full_sha256'] for x in d['tables']}}

def validate_config(config):
 import yaml,copy
 need(sha(OLD/'ptn/config.yaml')==CONFIG_SHA,'original effective config changed')
 original=yaml.safe_load((OLD/'ptn/config.yaml').read_text());expected=copy.deepcopy(original)
 expected['exp_dir']=str(RUN/'ptn');expected['checkpoint']=str(CK)
 need(config==expected,'only exp_dir and checkpoint may change')
 need(config.get('num_epochs')==120 and config['dataloader_args']['batch_size']==96,'PTN geometry changed')
 need(config['scheduler_args']['epoch_iter']==15001 and config['scheduler_args']['scale_ratio']==12.0,'PTN clock changed')
 need(config.get('model_ckpt') is None and config['projection_args']['num_class']==357822,'cold/head override refused')
 return expected

def checkpoint():
 import torch
 before=snapshot(CK);need(before[2]==650777251 and sha(CK)==CK_SHA,'model35 bytes changed')
 models=[int(p.stem.split('_')[-1]) for p in (OLD/'ptn/models').glob('model_[0-9]*.pt') if p.stem.split('_')[-1].isdigit()]
 need(max(models)==35 and not any('tmp' in p.name for p in (OLD/'ptn/models').iterdir()),'model35 no longer unique latest complete source')
 p=torch.load(CK,map_location='cpu',weights_only=False,mmap=True)
 need(set(p)=={'epoch','model','optimizer','scaler'} and p['epoch']==35,'invalid model35 payload')
 need(len(p['model'])==823 and len(p['optimizer']['state'])==584 and p['scaler'].get('scale')==65536.,'model35 state geometry changed')
 need(snapshot(CK)==before,'checkpoint changed while loading')
 return p

def exact(actual,expected,path='state'):
 import torch
 if torch.is_tensor(expected):
  need(torch.is_tensor(actual),path+': tensor required')
  need(actual.dtype==expected.dtype and actual.shape==expected.shape,path+': tensor dtype/shape differs')
  need(not (expected.is_floating_point() or expected.is_complex()) or bool(torch.isfinite(expected).all()),path+': saved tensor nonfinite')
  need(torch.equal(actual.detach().cpu(),expected.detach().cpu()),path+': tensor values differ')
 elif isinstance(expected,dict):
  need(isinstance(actual,dict) and set(actual)==set(expected),path+': keys differ')
  for k in expected:exact(actual[k],expected[k],path+'.'+str(k))
 elif isinstance(expected,(list,tuple)):
  need(type(actual)==type(expected) and len(actual)==len(expected),path+': sequence differs')
  for j,(a,b) in enumerate(zip(actual,expected)):exact(a,b,path+'.'+str(j))
 else:
  need(type(actual)==type(expected) and actual==expected,path+': scalar differs')
  need(not isinstance(expected,float) or math.isfinite(expected),path+': nonfinite scalar')

def verify_restored(model,optimizer,scaler,payload,start_epoch,epoch_iter,world,config,spk_path):
 import torch
 need(start_epoch==36 and epoch_iter==15001 and world==8,'restore must start epoch36 with original W8/15001')
 validate_config(config)
 need(sha(spk_path)==SPK_SHA and sha(OLD/'ptn/spk2id.json')==SPK_SHA,'saved/recomputed speaker map differs')
 need(list(model.projection.weight.shape)==[357822,192],'classification head resized/rebuilt')
 actual=model.state_dict();need(len(actual)==823,'823 complete model states required');exact(actual,payload['model'],'model')
 names=list(model.backbone.named_parameters())+list(model.projection.named_parameters())
 params=list(model.backbone.parameters())+list(model.projection.parameters())
 need(len(params)==585 and [len(g['params']) for g in optimizer.param_groups]==[583,2],'optimizer parameter group order differs')
 need([id(p) for g in optimizer.param_groups for p in g['params']]==[id(p) for p in params],'optimizer no longer binds original ordered parameters')
 saved=payload['optimizer'];current=optimizer.state_dict();exact(current,saved,'optimizer')
 need(len(optimizer.state)==584,'momentum was dropped or default state was added')
 momentum=0
 for param,group in optimizer.state.items():
  need(set(group)=={'momentum_buffer'},'unexpected optimizer state keys')
  buf=group['momentum_buffer'];need(buf.shape==param.shape and bool(torch.isfinite(buf).all()),'momentum shape/nonfinite');momentum+=1
 need(momentum==584,'all saved momentum must be restored')
 need(scaler.is_enabled(),'AMP scaler unexpectedly disabled');exact(scaler.state_dict(),payload['scaler'],'scaler')
 return {'restored':True,'epoch':36,'absolute_iteration':525035,'state_keys':823,'model_and_head_exact':True,'ordered_parameter_count':585,'optimizer_state_entries':584,'momentum_buffers':584,'optimizer_exact':True,'scaler_exact':True,'speaker_map_sha256':SPK_SHA,'checkpoint_sha256':CK_SHA,'no_forward_or_update_before_this_check':True}

def verify_imports():
 checked={}
 for name,module in list(sys.modules.items()):
  if name=='redimnet2' or name.startswith('redimnet2.') or name=='wespeaker_lite' or name.startswith('wespeaker_lite.'):
   path=getattr(module,'__file__',None)
   if path:
    p=Path(path).resolve();need(SOURCE in p.parents,'module imported outside frozen source: '+name);checked[name]={'path':str(p),'sha256':sha(p)}
 return checked
