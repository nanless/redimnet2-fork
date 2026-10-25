import copy,csv,datetime as dt,hashlib,importlib.util,io,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
path=Path(__file__).resolve().parents[1]/'tools/lm_continuation/readiness.py';spec=importlib.util.spec_from_file_location('cross_readiness_tested',path);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
NOW=dt.datetime(2026,10,25,4,0,tzinfo=dt.timezone.utc)
class CrossSegmentCPU(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.n=0
  self.spk=b'same small speaker-map fixture\n';self.patch=mock.patch.object(M,'SPK',M.digest(self.spk));self.patch.start();self.bundle=b'{"fixture":true}';self.bundle_patch=mock.patch.object(M,'BUNDLE',M.digest(self.bundle));self.bundle_patch.start()
  self.c={'schema':M.SCHEMA,'preparation_only':True,'lineage':copy.deepcopy(M.LINEAGE),'lm':copy.deepcopy(M.LM),'artifacts':{}};a=self.c['artifacts'];a['bundle_manifest']=self.make(self.bundle,M.BASE+'/ptn-resume-code-model35-20261024-v3/bundle_files.json');a['legacy_contract']=None
  for key,run,start,end in [('old_validation',M.OLD,1,35),('new_validation',M.NEW,36,120)]:
   stream=io.StringIO();w=csv.writer(stream);w.writerow(['epoch','lr','wd','margin','vox1-eer-8s'])
   for e in range(start,end+1):w.writerow([e,.72,.00002,.2,1 if e==27 else 2])
   a[key]=self.make(stream.getvalue().encode(),run+'/ptn/val_log.csv')
  a['new_train']=self.make(''.join(f'| {e}|15001|.72|.2|5.7|26|\n' for e in range(36,121)).encode(),M.NEW+'/ptn/train.log');a['new_log']=self.make(b'[dataset][DROP:short_audio] shorter than 1s\n',M.NEW+'/launch.log')
  a['restore_proofs']=[]
  for r in range(8):
   j={'rank':r,'world_size':8,'restored':True,'epoch':36,'absolute_iteration':525035,'state_keys':823,'optimizer_state_entries':584,'model_and_head_exact':True,'optimizer_exact':True,'scaler_exact':True,'speaker_map_sha256':M.SPK,'checkpoint_sha256':M.CK35,'no_forward_or_update_before_this_check':True}
   a['restore_proofs'].append(self.make_json(j,M.NEW+f'/checks/restore_verified_rank{r}.json'))
  for segment,task,run in [('old',M.OLD_TASK,M.OLD),('new',M.NEW_TASK,M.NEW)]:
   common={'task_id':task,'parent_run_root':run,'source':'volcengine-console','observed_at':NOW.isoformat()}
   a[segment+'_terminal']=self.make_json({**common,'state':'Terminated' if segment=='old' else 'Succeeded','active_workers':0},run+'/terminal-external.json')
   a[segment+'_writers']=self.make_json({**common,'writers':[],'probe_complete':True},run+'/writers-external.json')
   a[segment+'_speaker_map']=self.make(self.spk,run+'/ptn/spk2id.json')
  for e in [115,120]:a['checkpoint'+str(e)]=self.make_json({'epoch':e,'head_shape':[357822,192]},M.NEW+f'/ptn/models/model_{e}.pt')
  a['best_checkpoint']=self.make(b'best27 fixture',M.OLD+'/ptn/models/model_27.pt')
  a['new_success']=self.make_json({'ptn_complete':True,'epochs':list(range(36,121)),'checkpoint':M.NEW+'/ptn/models/model_120.pt','checkpoint_sha256':a['checkpoint120']['sha256'],'old_checkpoint_sha256':M.CK35,'rank_restore_proofs':8,'rank_first_updates':8,'lm_started':False,'lm_requires_separate_cross_segment_contract':True,'final_evaluation_complete':False},M.NEW+'/ptn_success.json')
 def tearDown(self):self.patch.stop();self.bundle_patch.stop();self.temp.cleanup()
 def make(self,data,logical):
  self.n+=1;p=self.root/str(self.n);p.write_bytes(data);return {'read_path':str(p),'logical_path':logical,'sha256':M.digest(data)}
 def make_json(self,j,logical):return self.make(json.dumps(j).encode(),logical)
 def alter(self,key,fn,index=None):
  a=self.c['artifacts'];r=a[key] if index is None else a[key][index];j=json.loads(Path(r['read_path']).read_bytes());fn(j);new=self.make_json(j,r['logical_path'])
  if index is None:a[key]=new
  else:a[key][index]=new
 def evaluate(self):
  return M.evaluate(self.c,NOW,_checkpoint_reader=lambda p:json.loads(Path(p).read_bytes()),_runtime_verifier=lambda c:{'fixture_only':True,'production_runtime_not_executed':True})
 def refuses(self,name=None):
  r=self.evaluate();self.assertFalse(r['cpu_evidence_complete']);self.assertFalse(r['ready_for_lm_execution'])
  if name:self.assertTrue(any(x['name']==name and not x['passed'] for x in r['checks']),r)
 def test_complete_fixture_only_cpu_not_execution(self):
  r=self.evaluate();self.assertTrue(r['cpu_evidence_complete'],r);self.assertFalse(r['ready_for_lm_execution']);self.assertFalse(r['lm_started']);self.assertFalse(r['locks_acquired']);self.assertEqual(r['results']['full_lineage_best_identity']['epoch'],27)
 def test_wrong_task_fixed_policy(self):self.c['lineage']['successor_task_id']='other';self.refuses('fixed_contract')
 def test_boolean_or_float_policy_rejected(self):self.c['lm']['world_size']=8.0;self.refuses('fixed_contract')
 def test_lm_checkpoint_instead_of_model_only(self):self.c['lm']['checkpoint']=self.c['lm']['model_ckpt'];self.refuses('fixed_contract')
 def test_lm_optimizer_not_fresh(self):self.c['lm']['optimizer']='restore';self.refuses('fixed_contract')
 def test_missing_validation_epoch(self):
  a=self.c['artifacts'];r=a['new_validation'];data=Path(r['read_path']).read_bytes().replace(b'100,0.72',b'101,0.72');a['new_validation']=self.make(data,r['logical_path']);self.refuses('successor_validation_36_120')
 def test_only_current43_refuses(self):
  a=self.c['artifacts'];r=a['new_validation'];lines=Path(r['read_path']).read_bytes().splitlines(True);a['new_validation']=self.make(b''.join(lines[:9]),r['logical_path']);self.refuses('successor_validation_36_120')
 def test_missing_120_checkpoint(self):self.c['artifacts']['checkpoint120']=None;self.refuses('successor_real_ptn_success')
 def test_old_run115_is_refused(self):self.c['artifacts']['checkpoint115']['logical_path']=M.OLD+'/ptn/models/model_115.pt';self.refuses('checkpoint115')
 def test_wrong_checkpoint_epoch(self):self.alter('checkpoint115',lambda x:x.update(epoch=114));self.refuses('checkpoint115')
 def test_corrupt_raw_bytes(self):Path(self.c['artifacts']['new_train']['read_path']).write_bytes(b'corrupt');self.refuses('successor_train_complete')
 def test_duplicate_train_completion(self):
  a=self.c['artifacts'];r=a['new_train'];a['new_train']=self.make(Path(r['read_path']).read_bytes()+b'|120|15001|.72|.2|5.7|26|\n',r['logical_path']);self.refuses('successor_train_complete')
 def test_new_flac_or_unknown_error(self):
  a=self.c['artifacts'];a['new_log']=self.make(b'read_failed fresh-event',M.NEW+'/launch.log');self.refuses('successor_no_new_failures')
 def test_bare_nonfinite(self):
  a=self.c['artifacts'];a['new_log']=self.make(b'gradient NaN',M.NEW+'/launch.log');self.refuses('successor_no_new_failures')
 def test_unknown_drop(self):
  a=self.c['artifacts'];a['new_log']=self.make(b'[DROP:decode_error]',M.NEW+'/launch.log');self.refuses('successor_no_new_failures')
 def test_succeeded_summary_alone_not_enough(self):self.c['artifacts']['new_log']=None;self.refuses('successor_no_new_failures')
 def test_new_terminal_failed(self):self.alter('new_terminal',lambda x:x.update(state='Failed'));self.refuses('new_terminal')
 def test_stale_terminal(self):self.alter('old_terminal',lambda x:x.update(observed_at=(NOW-dt.timedelta(seconds=601)).isoformat()));self.refuses('old_terminal')
 def test_future_terminal(self):self.alter('new_terminal',lambda x:x.update(observed_at=(NOW+dt.timedelta(seconds=6)).isoformat()));self.refuses('new_terminal')
 def test_writer_live(self):self.alter('new_writers',lambda x:x.update(writers=[{'pid':3}]));self.refuses('new_writers')
 def test_wrong_writer_task(self):self.alter('old_writers',lambda x:x.update(task_id=M.NEW_TASK));self.refuses('old_writers')
 def test_incomplete_probe(self):self.alter('new_writers',lambda x:x.update(probe_complete=False));self.refuses('new_writers')
 def test_restore_rank_duplicate(self):self.alter('restore_proofs',lambda x:x.update(rank=0),index=7);self.refuses('eight_same_stage_model35_restores')
 def test_ptn_optimizer_restore_missing(self):self.alter('restore_proofs',lambda x:x.update(optimizer_exact=False),index=2);self.refuses('eight_same_stage_model35_restores')
 def test_speaker_map_changed(self):self.c['artifacts']['new_speaker_map']=self.make(b'new ids',M.NEW+'/ptn/spk2id.json');self.refuses('same_speaker_map')
 def test_best_not_from_argmin(self):self.c['artifacts']['best_checkpoint']['logical_path']=M.NEW+'/ptn/models/model_41.pt';self.refuses('full_lineage_best_identity')
 def test_no_model_or_runtime_read_when_43_incomplete(self):
  self.c['artifacts']['new_validation']=None
  with mock.patch.object(M,'real_checkpoint',side_effect=AssertionError('must not import torch')),mock.patch.object(M,'live_legacy_runtime',side_effect=AssertionError('must not import decoder')):r=M.evaluate(self.c,NOW)
  self.assertFalse(r['cpu_evidence_complete'])
 def test_original_dependency_sha_pin(self):self.assertTrue(hasattr(M.legacy_module(),'verify_checkpoint'));self.assertTrue(hasattr(M.legacy_module(),'audit_reviewed_events'))
 def test_duplicate_json_key_refused(self):
  with self.assertRaises(M.Refused):M.no_duplicate_json(b'{"x":1,"x":2}')
 def test_bad_top_level_type_structured_refusal(self):
  r=M.evaluate([],NOW);self.assertFalse(r['cpu_evidence_complete']);self.assertEqual(r['checks'][0]['name'],'fixed_contract')
 def test_bad_artifacts_type_structured_refusal(self):self.c['artifacts']=[];self.refuses('fixed_contract')
 def test_unknown_artifact_controls_refused(self):self.c['artifacts']['skip_terminal']=True;self.refuses('fixed_contract')
 def test_bundle_manifest_missing(self):self.c['artifacts']['bundle_manifest']=None;self.refuses('successor_bundle_manifest')
 def test_bundle_manifest_corrupt_even_rehashed(self):self.c['artifacts']['bundle_manifest']=self.make(b'{"wrong":true}',M.BASE+'/ptn-resume-code-model35-20261024-v3/bundle_files.json');self.refuses('successor_bundle_manifest')
 def test_default_pickle_reader_rejects_arbitrary_path_before_legacy(self):
  with mock.patch.object(M,'legacy_module',side_effect=AssertionError('must refuse before loading helpers/torch')):
   with self.assertRaises(M.Refused):M.real_checkpoint(self.root/'fake.pt')
 def test_json_evidence_array_refused(self):self.c['artifacts']['new_terminal']=self.make(b'[]',M.NEW+'/terminal-external.json');self.refuses('new_terminal')
 def test_old_entry_failure_phrases_preserved(self):
  for phrase in ['could NOT restore','freshly initialised','丢弃了','NCCL runtime error']:
   with self.subTest(phrase=phrase):self.c['artifacts']['new_log']=self.make(phrase.encode(),M.NEW+'/launch.log');self.refuses('successor_no_new_failures')
 def test_output_protected_tree_refused_before_mkdir(self):
  with mock.patch.object(Path,'mkdir',side_effect=AssertionError('must reject before mkdir')):
   for target in [M.OLD+'/bad.json',M.NEW+'/bad.json',M.SOURCE+'/bad.json',M.LM['output_root']+'/bad.json','/private/tmp/outside-lm-prep.json']:
    with self.subTest(target=target):
     with self.assertRaises(M.Refused):M.write_new(target,{'not_created':True})
 def test_output_relative_refused(self):
  with self.assertRaises(M.Refused):M.write_new('relative.json',{})
 def test_output_symlink_ancestor_escape_refused(self):
  # Fixture changes only __file__ to a temp module location; no source/package files are written.
  own=self.root/'own';outside=self.root/'outside';own.mkdir();outside.mkdir();(own/'escape').symlink_to(outside,target_is_directory=True)
  with mock.patch.object(M,'__file__',str(own/'module.py')):
   with self.assertRaises(M.Refused):M.write_new(str(own/'escape/result.json'),{})
  self.assertFalse((outside/'result.json').exists())
 def test_own_output_original_bytes_not_overwritten(self):
  own=self.root/'package';own.mkdir();target=own/'new.json'
  with mock.patch.object(M,'__file__',str(own/'module.py')):
   M.write_new(str(target),{'first':1});before=target.read_bytes()
   with self.assertRaises(FileExistsError):M.write_new(str(target),{'second':2})
  self.assertEqual(target.read_bytes(),before)
 def test_bare_exceptions_all_refused_without_traceback_or_drop(self):
  for name in ['LibsndfileError','OSError','ValueError','TypeError']:
   with self.subTest(name=name):self.c['artifacts']['new_log']=self.make((name+': bare new unreviewed failure\n').encode(),M.NEW+'/launch.log');self.refuses('successor_no_new_failures')
 def test_nonexception_warning_not_mislabeled(self):
  self.c['artifacts']['new_log']=self.make(b'UserWarning: bounded harmless warning\n',M.NEW+'/launch.log');r=self.evaluate();self.assertTrue(r['cpu_evidence_complete'],r);self.assertFalse(r['ready_for_lm_execution'])
 def test_generic_regression_windows_retain_critical_legal_starts(self):
  M.legacy_module()
  import audio_review
  header={'frames':379321,'samplerate':16000,'channels':1}
  windows=audio_review._production_windows({'sample_key':'synthetic-public-fixture'},header)
  ptn={w['start_frame'] for w in windows if w['stage']=='ptn'}
  lm={w['start_frame'] for w in windows if w['stage']=='lm'}
  self.assertIn(24593,ptn);self.assertIn(319487,ptn);self.assertNotIn(319487,lm)
 def test_generic_windows_do_not_depend_on_private_sample_identity(self):
  M.legacy_module()
  import audio_review
  h={'frames':379321,'samplerate':16000,'channels':1}
  self.assertEqual(audio_review._production_windows({'sample_key':'synthetic-a'},h),audio_review._production_windows({'sample_key':'synthetic-b'},h))
 def test_builder_takes_explicit_portable_references(self):
  p=self.root/'refs.json';p.write_text(json.dumps(self.c['artifacts']))
  built=M.build_from_references(p)
  self.assertEqual(built['artifacts'],self.c['artifacts'])
if __name__=='__main__':unittest.main(verbosity=2)
