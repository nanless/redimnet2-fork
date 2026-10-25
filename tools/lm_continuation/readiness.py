#!/usr/bin/env python3
"""Read-only CPU preparation; deliberately has no LM executor or lock acquisition."""
import argparse,csv,datetime as dt,hashlib,importlib.util,io,json,math,os,re,stat,sys
from pathlib import Path
sys.dont_write_bytecode=True
BASE='/root/experiments/redimnet2/b6-mix-official-20261019'
OLD=BASE+'/formal-main-7afb821-zero-20261019-2327';NEW=BASE+'/ptn-resume-model35-20261024-v3'
SOURCE=BASE+'/lm-continuation-20261022/source-7afb821'
OLD_TASK='t-20261019232824-8zqrl';NEW_TASK='t-20261024154424-vggvm'
CK35='fdaa88b963720ccdc35708c3ca9d18e56296edd8711eb9309772afacc92beb84'
SPK='559a92fa71e8f0c1048fd5e916a18476f7503d20e5e70cf8452ea36d013e2548'
BUNDLE='840235e7b5963ada274531a29401b9f2bd75fe92e985a8e8e560341ac48350d8'
LEGACY_SHA='aeefd70892bbac972d15c2d7f06ea9787dbc5c5c5d3165d0834d0cb2fab431df'
HELPERS={'reviewed_events.py':'1d3ec275ea961beaf98c000b824965cdcf485411281a6c4004ff024d54d84280','audio_review.py':'f5d69d59148df4f2c8693f8f67dbd432823a41c3642016b07819210e6533c507','decoder_runtime.py':'573a3511a7d633fe4b4782b88c27d643115a9eb223b5d7858e1d46b6263b611e'}
OLD_CONTRACT_SHA='3b8316daec67818354d32d1192d2d06d668de2c51b64622232f622f54aacce99'
SCHEMA='redimnet2-cross-segment-lm-preparation/v1'
LM={'model_ckpt':NEW+'/ptn/models/model_115.pt','checkpoint':None,'optimizer':'fresh','scaler':'fresh','num_epochs':1,'batch_size_per_gpu':48,'num_frms':500,'do_lm':True,'world_size':8,'ordinary_script_entry':True,'output_root':BASE+'/lm-cross-segment-20261025/lm-execution-once'}
LINEAGE={'original_task_id':OLD_TASK,'successor_task_id':NEW_TASK,'original_run':OLD,'successor_run':NEW,'original_completed_epochs':[1,35],'successor_completed_epochs':[36,120],'restore_checkpoint_sha256':CK35,'successor_bundle_sha256':BUNDLE,'source_root':SOURCE,'speaker_map_sha256':SPK}
FATAL=re.compile(r'Traceback \(most recent call last\)|\bread_failed\b|Non-finite training|\b(?:nan|[+-]?inf(?:inity)?)\b|CUDA out of memory|ChildFailedError|Watchdog caught|could NOT restore|freshly initialised|丢弃了|NCCL[^\n]{0,200}(?:error|timeout)|(?:RuntimeError|FloatingPointError|OutOfMemoryError):',re.I)
METRIC=re.compile(r'\|\s*(\d+)\|\s*(\d+)\|\s*([\d.eE+-]+)\|\s*([\d.eE+-]+)\|\s*([\d.eE+-]+)\|\s*([\d.eE+-]+)\|')
class Refused(RuntimeError):pass
def need(v,msg):
 if not v:raise Refused(msg)
def digest(data):return hashlib.sha256(data).hexdigest()
def no_duplicate_json(data):
 def pairs(items):
  d={}
  for k,v in items:
   need(k not in d,'duplicate JSON key: '+k);d[k]=v
  return d
 return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(Refused('nonfinite JSON constant')))
def stable_bytes(ref,logical=None):
 need(isinstance(ref,dict),'missing required evidence reference')
 if logical:need(ref.get('logical_path')==logical,'wrong logical source: '+str(logical))
 need(isinstance(ref.get('read_path'),str) and Path(ref['read_path']).is_absolute(),'absolute read path required')
 need(re.fullmatch('[0-9a-f]{64}',ref.get('sha256','')) is not None,'missing exact evidence SHA')
 p=Path(ref['read_path']);need(not p.is_symlink() and stat.S_ISREG(p.lstat().st_mode),'evidence must be a regular non-symlink leaf')
 with p.open('rb') as f:
  before=os.fstat(f.fileno());data=f.read();after=os.fstat(f.fileno())
 key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 need(key(before)==key(after)==key(p.stat()),'evidence changed during read')
 need(digest(data)==ref['sha256'],'evidence SHA differs')
 return data
def stable_file_digest(ref):
 need(isinstance(ref,dict) and isinstance(ref.get('read_path'),str),'missing bounded file reference')
 p=Path(ref['read_path']);need(p.is_absolute() and not p.is_symlink() and p.is_file(),'file must be real and absolute')
 need(re.fullmatch('[0-9a-f]{64}',ref.get('sha256','')) is not None,'missing file SHA')
 key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 h=hashlib.sha256()
 with p.open('rb') as f:
  before=os.fstat(f.fileno())
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  after=os.fstat(f.fileno())
 need(key(before)==key(after)==key(p.stat()),'file changed during streaming hash')
 need(h.hexdigest()==ref['sha256'],'file SHA differs')
 return h.hexdigest()
def json_evidence(ref,logical=None):
 value=no_duplicate_json(stable_bytes(ref,logical));need(isinstance(value,dict),'JSON evidence must be an object');return value
def fixed_contract(c):
 need(isinstance(c,dict),'contract must be a JSON object')
 need(isinstance(c.get('artifacts'),dict),'artifacts must be an object')
 expected={'old_validation','new_validation','new_train','new_log','legacy_contract','restore_proofs','old_terminal','new_terminal','old_writers','new_writers','old_speaker_map','new_speaker_map','checkpoint115','checkpoint120','best_checkpoint','new_success','bundle_manifest'}
 need(set(c['artifacts'])==expected,'artifact keys are missing or unknown')
 need(c.get('schema')==SCHEMA and c.get('preparation_only') is True,'wrong preparation contract')
 need(json.dumps(c.get('lineage'),sort_keys=True)==json.dumps(LINEAGE,sort_keys=True),'fixed cross-segment lineage differs')
 need(json.dumps(c.get('lm'),sort_keys=True)==json.dumps(LM,sort_keys=True),'LM must initialize only model115 and fresh optimizer/scaler in new output')
 need(set(c)=={'schema','preparation_only','lineage','lm','artifacts'},'unknown contract controls are refused')
def validation(ref,logical,start,end):
 rows=list(csv.DictReader(io.StringIO(stable_bytes(ref,logical).decode('utf-8'))))
 need([r.get('epoch') for r in rows]==[str(x) for x in range(start,end+1)],f'validation epochs must be exactly {start}..{end}')
 for r in rows:
  for key in ['lr','wd','margin','vox1-eer-8s']:need(math.isfinite(float(r[key])),'nonfinite validation metric')
  need(0<=float(r['vox1-eer-8s'])<=100,'EER outside range')
 return rows
def bundle_manifest(ref):
 data=stable_bytes(ref,BASE+'/ptn-resume-code-model35-20261024-v3/bundle_files.json');need(digest(data)==BUNDLE,'v3 bundle manifest differs from code-pinned original')
 value=no_duplicate_json(data);need(isinstance(value,dict) and bool(value),'bundle manifest must be nonempty object');return {'sha256':BUNDLE,'observed_manifest_only':'each runtime source is independently reverified by original source95/189 checks'}
def restored(refs):
 need(isinstance(refs,list) and len(refs)==8,'exactly eight restore proofs required')
 for rank,ref in enumerate(refs):
  r=json_evidence(ref,NEW+f'/checks/restore_verified_rank{rank}.json')
  expected={'rank':rank,'world_size':8,'restored':True,'epoch':36,'absolute_iteration':525035,'state_keys':823,'optimizer_state_entries':584,'model_and_head_exact':True,'optimizer_exact':True,'scaler_exact':True,'speaker_map_sha256':SPK,'checkpoint_sha256':CK35,'no_forward_or_update_before_this_check':True}
  need(all(type(r.get(k)) is type(v) and r.get(k)==v for k,v in expected.items()),'restore model35/rank/optimizer/scaler provenance differs')
 return {'ranks':8,'checkpoint_sha256':CK35,'semantics':'same-stage PTN restored model+optimizer+scaler; LM will not restore their state'}
def complete_train(ref):
 text=stable_bytes(ref,NEW+'/ptn/train.log').decode('utf-8');rows=[]
 for m in METRIC.finditer(text):
  e,b=int(m[1]),int(m[2]);need(36<=e<=120 and 1<=b<=15001,'successor train epoch/batch out of scope')
  need(all(math.isfinite(float(m[i])) for i in range(3,7)),'nonfinite train metric');rows.append((e,b))
 need(rows==sorted(set(rows)),'duplicate or reordered train progress')
 need([e for e,b in rows if b==15001]==list(range(36,121)),'successor completed epochs are not exactly36..120')
 return {'completed_epochs':85,'last_epoch':120,'last_batch':15001}
def new_logs(ref):
 raw=stable_bytes(ref,NEW+'/launch.log')
 # Reuse pinned original pure byte-level rules; never grant the old 7-event allowance to this run.
 legacy_module()
 from reviewed_events import _FATAL, _OTHER_EXCEPTION
 need(not _FATAL.search(raw) and not _OTHER_EXCEPTION.search(raw),'unreviewed bare/fatal exception in successor log')
 text=raw.decode('utf-8')
 need(not FATAL.search(text),'unreviewed failure in successor log')
 drops=re.findall(r'\bDROP(?::([^\]\s]+))?',text)
 need(all(x=='short_audio' for x in drops),'unknown DROP in successor log')
 return {'unreviewed_failures':0,'short_audio_print_events':len(drops),'fatal_rules_module_sha256':HELPERS['reviewed_events.py'],'reused_pure_rules':['_FATAL','_OTHER_EXCEPTION'],'old_seven_event_allowance_applies_to_successor':False}
def terminal(ref,task,run,now,writer=False):
 r=json_evidence(ref);need(r.get('task_id')==task and r.get('parent_run_root')==run,'terminal/writer task or run differs')
 need(r.get('source') in ['volcengine-console','volcengine-api'],'terminal/writer evidence source differs')
 stamp=r.get('observed_at');need(isinstance(stamp,str),'missing evidence UTC')
 t=dt.datetime.fromisoformat(stamp.replace('Z','+00:00'));need(t.tzinfo is not None,'timezone required')
 need(-5<=(now-t).total_seconds()<=600,'terminal/writer evidence stale or future')
 if writer:need(r.get('writers')==[] and r.get('probe_complete') is True,'writer probe incomplete or live writer remains')
 else:
  allowed=['Succeeded'] if task==NEW_TASK else ['Terminated','Succeeded','Stopped']
  need(r.get('state') in allowed and type(r.get('active_workers')) is int and r['active_workers']==0,'platform task is not in required successful/terminal state')
 return {'task_id':task,'fresh':True,'writers_empty':writer}
def legacy_module():
 p=Path(__file__).parent/'legacy_checks/legacy_preflight.py';need(digest(p.read_bytes())==LEGACY_SHA,'legacy dependency bytes changed')
 for filename,expected in HELPERS.items():
  q=p.parent/filename;need(digest(q.read_bytes())==expected,'legacy helper bytes changed: '+filename)
  mod=sys.modules.get(q.stem)
  if mod is not None:need(Path(getattr(mod,'__file__','')).resolve()==q.resolve(),'legacy helper module namespace collision')
 sys.path.insert(0,str(p.parent))
 name='_redim_legacy_lm_preflight';spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def live_legacy_runtime(c):
 oldref=c['artifacts'].get('legacy_contract');need(oldref and oldref.get('sha256')==OLD_CONTRACT_SHA,'wrong fixed old-v4 contract')
 old=json_evidence(oldref,BASE+'/lm-continuation-20261022/event-review-20261023-v4/contract.json');m=legacy_module()
 # No old single-run PTN120 check: reuse only the already reviewed source/import/7-event subchecks.
 source=m.validate_source(old,m.Policy());imports=m.verify_import_paths(old,m.Policy());events=m.audit_reviewed_events(old,m.Policy(),require_stable=True)
 need(events.get('events_review_accepted') is True and len(events.get('events',[]))==7,'old reviewed event count differs')
 return {'original_source95':len(source),'source189_verified_by_legacy':True,'imports':imports,'old_event_audit':events}
def real_checkpoint(path):
 p=Path(path);need(str(p) in [NEW+'/ptn/models/model_115.pt',NEW+'/ptn/models/model_120.pt'],'default pickle reader accepts only exact trusted live115/120 paths')
 need(p.resolve(strict=True)==p and p.is_file(),'trusted checkpoint has missing file or symlink ancestor')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 m=legacy_module();return m.checkpoint_summary(path)
def checkpoint(ref,epoch,reader):
 logical=NEW+f'/ptn/models/model_{epoch}.pt';dataref=ref
 need(isinstance(dataref,dict) and dataref.get('logical_path')==logical,'checkpoint must come from new ordinary run')
 p=Path(dataref.get('read_path',''));need(p.is_absolute() and not p.is_symlink(),'checkpoint real absolute file required')
 # Reuse old CPU model/optimizer/scaler/finite reader and stable epoch/head checker.
 m=legacy_module();info=m.verify_checkpoint(p,epoch,reader)
 need(info['sha256']==dataref.get('sha256'),'checkpoint bytes differ from frozen evidence reference')
 return {**info,'logical_path':logical}
def evaluate(c,now=None,*,_checkpoint_reader=None,_runtime_verifier=None):
 fixed_now=now;now=now or dt.datetime.now(dt.timezone.utc);checks=[];values={}
 def check(name,fn):
  try:values[name]=fn();checks.append({'name':name,'passed':True})
  except (Refused,OSError,KeyError,ValueError,TypeError,RuntimeError,ImportError) as e:checks.append({'name':name,'passed':False,'reason':str(e)})
 check('fixed_contract',lambda:fixed_contract(c))
 if not checks[-1]['passed']:return result(checks,values,now)
 a=c['artifacts']
 check('successor_bundle_manifest',lambda:bundle_manifest(a.get('bundle_manifest')))
 check('original_validation_1_35',lambda:validation(a.get('old_validation'),OLD+'/ptn/val_log.csv',1,35));check('successor_validation_36_120',lambda:validation(a.get('new_validation'),NEW+'/ptn/val_log.csv',36,120));check('eight_same_stage_model35_restores',lambda:restored(a.get('restore_proofs')));check('successor_train_complete',lambda:complete_train(a.get('new_train')));check('successor_no_new_failures',lambda:new_logs(a.get('new_log')))
 def success():
  r=json_evidence(a.get('new_success'),NEW+'/ptn_success.json')
  expected={'ptn_complete':True,'epochs':list(range(36,121)),'checkpoint':NEW+'/ptn/models/model_120.pt','old_checkpoint_sha256':CK35,'rank_restore_proofs':8,'rank_first_updates':8,'lm_started':False,'lm_requires_separate_cross_segment_contract':True,'final_evaluation_complete':False}
  need(all(type(r.get(k)) is type(v) and r.get(k)==v for k,v in expected.items()),'successor real postflight receipt incomplete or wrong lineage')
  need(isinstance(a.get('checkpoint120'),dict) and r.get('checkpoint_sha256')==a['checkpoint120'].get('sha256'),'successor postflight checkpoint120 SHA differs')
  return {'actual_ptn_success_sha256':a['new_success']['sha256'],'normal_child_exit_and_postflight_receipt':True}
 check('successor_real_ptn_success',success)
 for segment,task,run in [('old',OLD_TASK,OLD),('new',NEW_TASK,NEW)]:
  check(segment+'_terminal',lambda segment=segment,task=task,run=run:terminal(a.get(segment+'_terminal'),task,run,now));check(segment+'_writers',lambda segment=segment,task=task,run=run:terminal(a.get(segment+'_writers'),task,run,now,True))
 def spk():
  for segment,run in [('old',OLD),('new',NEW)]:need(digest(stable_bytes(a.get(segment+'_speaker_map'),run+'/ptn/spk2id.json'))==SPK,'speaker map changed')
  return {'same_sha256':SPK}
 check('same_speaker_map',spk)
 # Expensive CPU checks only after all basic and terminal evidence is actually complete.
 if all(z['passed'] for z in checks):
  check('original_source_decoder_and_exact_seven_events',lambda:(_runtime_verifier or live_legacy_runtime)(c))
  for e in [115,120]:check('checkpoint'+str(e),lambda e=e:checkpoint(a.get('checkpoint'+str(e)),e,_checkpoint_reader or real_checkpoint))
 else:checks.append({'name':'source_decoder_events_and_checkpoints115_120','passed':False,'reason':'not attempted: completion/terminal/writer/basic evidence incomplete; no torch, decoder or locks used'})
 if 'original_validation_1_35' in values and 'successor_validation_36_120' in values:
  rows=values['original_validation_1_35']+values['successor_validation_36_120'];best=min(rows,key=lambda r:float(r['vox1-eer-8s']));e=int(best['epoch']);run=OLD if e<=35 else NEW
  def best_identity():
   ref=a.get('best_checkpoint');need(isinstance(ref,dict) and ref.get('logical_path')==run+f'/ptn/models/model_{e}.pt','best checkpoint must match argmin of full lineage validation');stable_file_digest(ref);return {'epoch':e,'eer_percent':float(best['vox1-eer-8s']),'sha256':ref['sha256'],'lm_initializer_remains_epoch115':True}
  check('full_lineage_best_identity',best_identity)
 else:checks.append({'name':'full_lineage_best_identity','passed':False,'reason':'full 1..120 validation unavailable; best remains provisional'})
 if all(z['passed'] for z in checks):
  final_now=fixed_now or dt.datetime.now(dt.timezone.utc)
  for segment,task,run in [('old',OLD_TASK,OLD),('new',NEW_TASK,NEW)]:
   check(segment+'_terminal_recheck',lambda segment=segment,task=task,run=run:terminal(a.get(segment+'_terminal'),task,run,final_now));check(segment+'_writers_recheck',lambda segment=segment,task=task,run=run:terminal(a.get(segment+'_writers'),task,run,final_now,True))
 return result(checks,values,now)
def result(checks,values,now):
 return {'schema':'redimnet2-cross-segment-lm-cpu-readiness/v1','checked_utc':now.isoformat(),'cpu_evidence_complete':all(x['passed'] for x in checks),'ready_for_lm_execution':False,'lm_started':False,'locks_acquired':False,'checks':checks,'results':values,'mandatory_future_execution_gate':['fresh live checks under all original + successor + logical-chain locks','atomic unique new output claim','eight-rank cold optimizer/scaler actual evidence and LM1 postflight'],'meaning':'CPU preparation/readback only. This tool has no execute option; no output directory or training process is created.'}
def ref(path,logical):
 p=Path(path);return {'read_path':str(p.resolve()),'logical_path':logical,'sha256':digest(p.read_bytes())}
def build_from_references(path):
    artifacts=no_duplicate_json(Path(path).read_bytes())
    c={'schema':SCHEMA,'preparation_only':True,'lineage':LINEAGE.copy(),
       'lm':LM.copy(),'artifacts':artifacts}
    fixed_contract(c)
    return c
def write_new(path,value):
 p=Path(path);need(p.is_absolute(),'result output must be absolute')
 # Only own package descendants, checked before mkdir/open; symlink ancestors cannot escape.
 home=Path(__file__).resolve().parent;target=p.resolve(strict=False)
 need(target!=home and home in target.parents,'contract/result output must stay in this new preparation package')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True);b=sub.add_parser('build');b.add_argument('--references',required=True);b.add_argument('--output',required=True);v=sub.add_parser('check');v.add_argument('--contract',required=True);v.add_argument('--output',required=True);args=ap.parse_args()
 if args.command=='build':write_new(args.output,build_from_references(args.references));return
 c=no_duplicate_json(Path(args.contract).read_bytes());r=evaluate(c);r['contract_sha256']=digest(Path(args.contract).read_bytes());write_new(args.output,r);print(json.dumps({'cpu_evidence_complete':r['cpu_evidence_complete'],'ready_for_lm_execution':False,'failed_checks':[x['name'] for x in r['checks'] if not x['passed']]},ensure_ascii=False));raise SystemExit(0 if r['cpu_evidence_complete'] else 2)
if __name__=='__main__':main()
