"""CPU-only parameterized admission gate for NEW Ultra W2 evidence.
No GPU, submission, model execution or historical-v1/v5 approval shortcut.
"""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,math,sys
import collections
EXPECTED_DOMAINS={'voxceleb2':1092009,'voxblink2':9904382,'cnceleb2':524787}
EXPECTED_SOURCE_COUNT=sum(EXPECTED_DOMAINS.values())
SOURCE_SCOPE={'datasets':['cnceleb2','voxblink2','voxceleb2'],'allowed_roles':['train_lm','train_ptn'],'forbid_heldout_or_protected':True,'policy':'three_actual_training_domains_only/v1'}
BAD_QUALITY_SHA='67272c306cc88163217d34f09ea0fd0bc96b6b1142c36929abe70e99fdd1886a'
def validation_targets(spec):
    values=spec['validation_UUIDs']
    need(type(values)is list and len(values)==2 and len(set(values))==2
         and all(isinstance(x,str) and __import__('re').fullmatch(r'GPU-[0-9a-fA-F-]{36}',x) for x in values),'two explicit validation UUIDs required')
    return set(values)

def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def issha(value):return isinstance(value,str) and len(value)==64 and set(value)<={*'0123456789abcdef'}
def utc(value):
    t=datetime.datetime.fromisoformat(value.replace('Z','+00:00'));need(t.tzinfo is not None,'UTC-aware evidence time required');return t
class Reader:
    def __init__(self,spec):self.map=spec.get('evidence_path_map',{});self.seen={}
    def path(self,value):
        text=str(value)
        for source,target in sorted(self.map.items(),key=lambda x:len(x[0]),reverse=True):
            if text==source or text.startswith(source.rstrip('/')+'/'):text=target.rstrip('/')+text[len(source.rstrip('/')):];break
        return Path(text)
    def read(self,path,pin=None):
        path=self.path(path);need(not path.is_symlink(),'evidence symlink refused: '+str(path));before=path.stat();raw=path.read_bytes();after=path.stat()
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'evidence changed while read')
        h=sha(raw)
        if pin is not None:need(issha(pin) and h==pin,'evidence SHA differs: '+str(path))
        self.seen[str(path)]=h;return raw
    def load(self,path,pin=None):return json.loads(self.read(path,pin),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON '+x)))
    def ref(self,ref):return self.load(ref['path'],ref['sha256'])
def contracts_and_bundle(spec,r):
    code=r.path(spec['code_root']);bundle=r.ref(spec['bundle']);need(type(bundle)is dict and bundle,'bundle members required')
    for rel,pin in bundle.items():
        p=PurePosixPath(rel);need(not p.is_absolute() and '..' not in p.parts,'unsafe bundle member');r.read(code/rel,pin)
    pilot=r.ref(spec['pilot_contract']);silver=r.ref(spec['silver_contract'])
    need(pilot['purpose']=='pilot_validation' and silver['purpose']=='silver_annotation' and pilot['version']!=silver['version'],'pilot/silver purpose/version must differ explicitly')
    need({k:v for k,v in pilot.items() if k not in {'purpose','version'}}=={k:v for k,v in silver.items() if k not in {'purpose','version'}},'pilot/silver semantic fields differ')
    for c in [pilot,silver]:
        need(c['schema']=='redimnet2-production-contract/v2' and c['model_API_calls_allowed']==0 and c['release_training_labels'] is False,'formal contract purpose/API/release invalid')
        need(c['alignment']['mode']=='ultra_no_head' and c['alignment']['head_used'] is False and c['alignment']['revision']=='765dc1d714e4c80196782a5df5bfbcb3e1416736','unvalidated alignment backend')
        need(c['runtime']['max_batch']==1 and c['runtime']['execution_policy']=='single_window_independent' and c['runtime']['dtype']=='float32' and c['runtime']['attention']=='eager' and c['runtime']['TF32'] is False,'unvalidated runtime semantics')
        need(c['quality_policy']['source_split_policy']=='canonical_provider_evaluation_and_registered_heldout_roles/v2','heldout source split fix not declared')
        need(c['engine_source_files'].get('quality_axes.py')!=BAD_QUALITY_SHA,'known-bad v1 quality axes refused')
        for name,pin in c['engine_source_files'].items():need(bundle.get(name)==pin,'contract engine module not same frozen bundle: '+name)
    return pilot,silver
def idle_probe(probe,uuids):
    snapshots=probe['snapshots'];need(len(snapshots)==2 and probe['interval_seconds']>=5,'actual resource double snapshot missing')
    need(set(x['uuid'] for x in probe['devices'])==set(uuids),'return device UUIDs differ')
    need(snapshots[1]['monotonic']-snapshots[0]['monotonic']>=5,'resource snapshots not5seconds apart')
    for snapshot in snapshots:
        need(not [p for p in snapshot['compute_processes'] if p['gpu_uuid'] in uuids],'resource not idle: existing/unknown compute PID')
        for item in snapshot.get('raw',[]):need(item['returncode']==0,'resource query failed')
def validate_challenge(challenge,identity):
    nonce=identity['nonce'];uuid=identity['uuid'];pid=identity['container_pid']
    need(challenge['complete'] is True and challenge['known_container_worker_pid']==pid and challenge['UUIDs']==[uuid] and challenge['random_allocation_rounds_per_GPU']==3 and challenge['all_free_phases_returned_to_baseline'] is True,'complete 7-phase context proof missing')
    phases=challenge['phases'];need(len(phases)==7 and [p['seq'] for p in phases]==list(range(7)),'7phase sequence differs')
    baseline=phases[0]['expected_memory_mib'][uuid]
    for i,p in enumerate(phases):
        req=p['request'];resp=p['response'];expected_action='baseline' if i==0 else ('allocate' if i%2 else 'free')
        need(req['nonce']==resp['nonce']==nonce and req['seq']==resp['seq']==i and resp['worker_pid']==pid and resp['request']==req and req['action']==expected_action and resp['actual_cuda_uuids']==[uuid] and resp['minor_fds_verified'] is True,'phase request/response mismatch')
        need(p['opaque_nvml_pid']==challenge['opaque_nvml_pid'] and p['stable_samples']>=2 and len(p['accepted_fresh_snapshots'])>=2,'fresh stable NVML proof absent')
        expected=baseline+(req['bytes']//1048576 if i%2 else 0);need(p['expected_memory_mib'][uuid]==expected and abs(p['NVML_memory_mib'][uuid]-expected)<=1,'actual allocation/free memory pattern differs')
        samples=p['accepted_fresh_snapshots'];need(samples[-1]['monotonic']>samples[0]['monotonic']>=resp['observed_monotonic'],'phase observations stale/repeated')
        for sample in samples:
            rows=[row for row in sample['compute_processes'] if row['gpu_uuid']==uuid];need(len(rows)==1 and rows[0]['pid']==challenge['opaque_nvml_pid'] and abs(rows[0]['used_mib']-expected)<=1,'actual NVML row differs')
def workers(stage,parent,spec,r):
    resource=r.path(stage['resources']);workers=parent['workers'];need(parent['worker_count']==2 and len(workers)==2 and set(parent['UUIDs'])==validation_targets(spec),'actual W2 topology differs')
    need({x['worker_id'] for x in workers}=={'worker-0','worker-1'} and {x['uuid'] for x in workers}==validation_targets(spec),'worker mapping differs')
    actual_aligner_rows=0
    for entry in workers:
        folder=resource/entry['worker_id'];identity=r.load(folder/'worker_identity.json');ticket=r.load(folder/'admission.json');ready=r.load(folder/'terminal_ready.json');ack=r.load(folder/'terminal_ack.json');summary=r.load(folder/'worker_summary.json');att=r.load(folder/'resource_attestation.json');challenge=r.load(folder/'context_challenge/attestation.json')
        nonce=identity['nonce'];uuid=entry['uuid'];pid=identity['container_pid']
        need(identity['uuid']==uuid and identity['worker_id']==entry['worker_id'] and identity['PGID']==identity['SID']==pid,'worker PID/session identity differs')
        for value in [ticket,ready,att]:
            need(value['nonce']==nonce and value['worker_id']==entry['worker_id'] and value['bundle_sha256']==spec['bundle']['sha256'] and value['contract_sha256']==spec['pilot_contract']['sha256'] and value['index_sha256']==stage['index']['sha256'],'worker nonce/bundle/contract/index mismatch')
        ticketsha=r.seen[str(folder/'admission.json')];readysha=r.seen[str(folder/'terminal_ready.json')];summarysha=r.seen[str(folder/'worker_summary.json')]
        need(ready['worker_pid']==pid and ready['worker_start_ticks']==identity['started'] and ready['ticket_sha256']==att['ticket_sha']==ticketsha,'ready/resource ticket binding differs')
        need(r.path(ready['summary_path'])==folder/'worker_summary.json' and ready['summary_sha256']==summarysha==entry['terminal']['summary_sha256'] and summary==entry['terminal']['summary'],'actual worker summary differs from terminal')
        need(ready['resource_attestation_sha256']==r.seen[str(folder/'resource_attestation.json')] and ack=={'nonce':nonce,'ready_sha256':readysha} and entry['terminal']['ready_sha256']==readysha,'ready/ACK chain differs')
        need(att['known_container_worker_pid']==pid and att['UUIDs']==[uuid] and r.path(att['challenge_attestation_path'])==folder/'context_challenge/attestation.json' and att['challenge_attestation_sha']==r.seen[str(folder/'context_challenge/attestation.json')],'resource/challenge identity differs')
        validate_challenge(challenge,identity)
        need(summary['worker_id']==entry['worker_id'] and summary['device_uuid']==uuid and summary['contract_sha256']==spec['pilot_contract']['sha256'] and summary['index_sha256']==stage['index']['sha256'] and summary['synthetic_fixture'] is False,'worker summary is not real bound execution')
        if stage is spec['vertical']:need(summary['sidecar_cache_hits']==0,'fresh vertical attempt reused whole sidecars')
        metrics=summary['backend_metrics'];need(summary['model_API_calls']==metrics['model_API_calls']==0 and metrics['alignment_mode']=='ultra_no_head' and metrics['actual_model_training_states']=={'asr':False,'aligner':False},'model API/eval/backend differs')
        actual_aligner_rows+=metrics['aligner_rows']
        need(metrics['asr_rows']>0,'one GPU did no actual ASR')
        for name in ['asr','aligner']:
            hist=metrics[name+'_call_rows'];need(set(hist)<={'1'} and sum(hist.values())==metrics[name+'_batches']==metrics[name+'_rows'],'actual inference batch is not1')
    need(actual_aligner_rows>0,'stage has no actual Ultra native alignment inference')
    drain=r.load(resource/'terminal_drain_complete.json');need(drain['all_attested_contexts_absent'] is True,'contexts not drained');idle_probe(r.load(resource/'resource_return_double_probe.json'),validation_targets(spec))
def stage_result(stage,spec,r,count):
    parent=r.ref(stage['parent_result']);need(r.path(stage['parent_result']['path'])==r.path(stage['resources'])/'parent_result.json','parent result resource path differs')
    need(parent['bundle_sha256']==spec['bundle']['sha256'] and parent['contract_sha256']==spec['pilot_contract']['sha256'] and parent['index_sha256']==stage['index']['sha256'],'W2 result identity differs')
    need(utc(parent['started_utc'])>=utc(spec['not_before_utc']) and utc(parent['finished_utc'])>=utc(parent['started_utc']),'stale W2 run')
    need(parent['resource_returned'] is True and parent['model_API_calls']==0 and parent['full_sidecar_readback'] is True,'run not complete/returned/fullread')
    value=parent['verification'];need(value['passed'] is True and value['full_sidecar_readback_verified'] is True and value['completed_source_count']==value['sidecars_individually_reopened']==count and value['index_sha256']==stage['index']['sha256'] and value['contract_sha256']==spec['pilot_contract']['sha256'],'actual full sidecar count/binding differs')
    need(value['error_count']==0 and value['decision_counts']['failed']==0,'software/readback failures cannot pass formal admission')
    index=r.ref(stage['index']);need(index['source_count']==count and sum(x['count'] for x in index['shards'])==count,'actual stage index denominator')
    ids=set();domains=collections.Counter()
    for shard in index['shards']:
        rel=PurePosixPath(shard['path']);need(not rel.is_absolute() and '..' not in rel.parts,'unsafe stage shard path')
        raw=r.read(Path(stage['index']['path']).parent/shard['path'],shard['sha256']);rows=[json.loads(line) for line in raw.splitlines()];need(len(rows)==shard['count'],'actual stage shard count differs')
        for row in rows:
            need(row['dataset'] in EXPECTED_DOMAINS and bool(set(row['roles'])&{'train_ptn','train_lm'}) and not set(row['roles'])-{'train_ptn','train_lm'} and row['protection']['training_forbidden'] is False and not row.get('split_conflicts'),'stage source outside actual three-training-domain scope')
            need(row['source_id']=='src_'+sha((row['dataset']+'\0'+row['relative_path']).encode()) and row['sidecar_relpath']==row['dataset']+'/'+row['relative_path']+'.annotation.json','stage source ID/mirror identity differs')
            need(row['source_id'] not in ids,'duplicate stage source ID');ids.add(row['source_id']);domains[row['dataset']]+=1
    need(len(ids)==count and dict(domains)==index['domain_counts'] and set(domains)==set(EXPECTED_DOMAINS),'actual three-domain stage union differs')
    workers(stage,parent,spec,r);return parent
def verify(spec,r):
    need(spec['schema']=='ultra-eight-three-train-spec/v3','wrong admission spec');pilot,silver=contracts_and_bundle(spec,r)
    need(spec['source_scope']==SOURCE_SCOPE,'latest explicit three-training-domain scope absent or widened')
    full=r.ref(spec['full_index']);need(full['source_count']==EXPECTED_SOURCE_COUNT and full['domains']==EXPECTED_DOMAINS and full['role_union_counts']=={'train_ptn':EXPECTED_SOURCE_COUNT,'train_lm':5021010},'new full three-training-domain denominator/union differs')
    need(full['RIR_or_MUSAN_or_Child_or_evaluation_in_this_job'] is False and full['automatic_training_release'] is False,'new job contains outside training scope or grants training release')
    need(len({s['id'] for s in full['shards']})==len(full['shards']) and len({s['path'] for s in full['shards']})==len(full['shards']) and sum(s['count'] for s in full['shards'])==EXPECTED_SOURCE_COUNT,'duplicate/invalid full-index shard partition')
    need(all(not PurePosixPath(s['path']).is_absolute() and '..' not in PurePosixPath(s['path']).parts and issha(s['sha256']) and type(s['count'])is int and s['count']>0 for s in full['shards']),'unsafe or invalid full-index descriptors')
    filter_summary=r.ref(spec['full_filter_summary']);r.read(spec['full_filter_source']['path'],spec['full_filter_source']['sha256'])
    need(filter_summary['passed'] is True and filter_summary['source_count']==EXPECTED_SOURCE_COUNT and filter_summary['dataset_counts']==EXPECTED_DOMAINS and filter_summary['role_counts']==full['role_union_counts'] and filter_summary['index_sha256']==spec['full_index']['sha256'] and filter_summary['script_sha256']==spec['full_filter_source']['sha256']==full['script_sha256'],'full filter actual output/source/SHA accounting differs')
    need(filter_summary['old_files_modified'] is False and filter_summary['audio_stat_or_decode_performed'] is False and filter_summary['GPU_or_model_executed'] is False,'full filter touched excluded media/runtime')
    audit=r.ref(spec['full_filter_audit']);r.read(spec['full_filter_audit_source']['path'],spec['full_filter_audit_source']['sha256'])
    need(audit['passed'] is True and audit['index_sha256']==spec['full_index']['sha256'] and audit['source_count_validated']==EXPECTED_SOURCE_COUNT and audit['output_shards_verified']==len(full['shards']) and audit['dataset_counts']==EXPECTED_DOMAINS and audit['role_counts']==full['role_union_counts'] and audit['all_shard_file_SHA_and_source_ID_digest_reopened'] is True and audit['script_sha256']==spec['full_filter_audit_source']['sha256'],'actual full new-shard/source-role readback incomplete or wrong input')
    need(audit['GPU_or_model_execution'] is False and audit['old_files_modified'] is False and audit['audio_or_VAD_media_stat_or_decode'] is False,'full registry audit touched excluded runtime/media')
    r.read(Path(spec['full_filter_audit']['path']).parent/'shard_readback.jsonl',audit['shard_readback_ledger_sha256'])
    protection=r.ref(spec['diagnostic_protection_reference']);need(protection['schema']=='ultra-diagnostic-protection-reference/v1' and protection['case_count']==len(protection['cases'])==34 and protection['three_training_source_count']==12 and protection['all_original_audio_VAD_splits_preserved'] is True and all(c['benchmark_diagnostic_only_not_training_release'] is True for c in protection['cases']),'diagnostic protection reference missing or widened')
    r.read(spec['supervisor']['path'],spec['supervisor']['sha256']);start=r.ref(spec['sequence_start']);end=r.ref(spec['sequence_complete']);need(start['supervisor_sha256']==spec['supervisor']['sha256'] and start['bundle_sha256']==spec['bundle']['sha256'] and start['contract_sha256']==spec['pilot_contract']['sha256'],'actual sequence source/bundle/contract differs')
    need(end['passed'] is True and end['baseline_source_count']==spec['main']['source_count'] and end['actual_recovery_source_count']==spec['recovery']['source_count'] and end['native_equivalence'] is True and end['API_calls']==0,'new real sequence not complete')
    need(spec['main']['source_count']>=3000 and spec['vertical']['source_count']>=30 and spec['recovery']['source_count']>=512,'insufficient latest three-training-source acceptance cohort')
    parents={name:stage_result(spec[name],spec,r,spec[name]['source_count']) for name in ['vertical','main','recovery']}
    main=spec['main'];cache_hits=sum(w['terminal']['summary']['sidecar_cache_hits'] for w in parents['main']['workers'])
    if 'baseline_receipt' in main:
        reuse=r.ref(main['baseline_receipt']);need(reuse['schema']=='verified-three-training-source-reuse/v1' and reuse['source_count']==main['source_count'] and reuse['contract_sha256']==spec['pilot_contract']['sha256'] and reuse['index_sha256']==main['index']['sha256'],'baseline reuse source/index/contract differs')
        need(reuse['reused_sources']==len(reuse['copied'])==cache_hits and reuse['fresh_sources_remaining']+reuse['reused_sources']==main['source_count'] and reuse['other_domain_results_stage_cache_or_queue_copied'] is False and reuse['new_inference_claimed_for_reuse'] is False and reuse['GPU_used'] is False and reuse['PTN_modified'] is False,'baseline fresh/reused accounting or isolation differs')
        idx=r.ref(main['index']);source_rows={}
        for shard in idx['shards']:
            for line in r.read(Path(main['index']['path']).parent/shard['path'],shard['sha256']).splitlines():
                row=json.loads(line);source_rows[row['source_id']]=row
        seen=set()
        for packet in reuse['copied']:
            sid=packet['source_id'];need(sid not in seen and sid in source_rows,'reused source outside new main scope or duplicate');seen.add(sid);row=source_rows[sid]
            need(packet['dataset']==row['dataset'] and packet['new_inference_claimed'] is False and r.path(packet['new_sidecar'])==r.path(main['run_root'])/'labels'/pilot['version']/row['sidecar_relpath'],'reused source path/scope/freshness differs')
            oldraw=r.read(packet['old_sidecar'],packet['sidecar_sha256']);newraw=r.read(packet['new_sidecar'],packet['sidecar_sha256']);need(oldraw==newraw,'reused sidecar original bytes changed');body=json.loads(newraw)
            need(body['source']==row and body['contract']['sha256']==spec['pilot_contract']['sha256'] and body['contract']['provenance']['code_sha256']==spec['bundle']['sha256'],'reused source record or producer contract/code differs')
            identity=body['source_identity'];need(identity['file_sha256']==packet['fresh_encoded_source_sha256'] and identity['stat_before']==identity['stat_after']==packet['source_stat_verified'],'reused source fresh encoded SHA/stat differs')
            oldcommit=r.read(packet['old_sidecar']+'.commit.json');newcommit=r.read(packet['new_sidecar']+'.commit.json');need(oldcommit==newcommit,'reused original commit bytes changed');commit=json.loads(newcommit);rowsha=sha(canonical(row));job=sha(canonical({'source_id':sid,'source_record_sha256':rowsha,'contract_sha256':spec['pilot_contract']['sha256']}))
            need(commit['source_id']==sid and commit['source_record_sha256']==rowsha and commit['contract_sha256']==spec['pilot_contract']['sha256'] and commit['sidecar_sha256']==packet['sidecar_sha256'] and commit['bytes']==len(newraw) and commit['job_key']==job,'reused source commit identity differs')
    else:need(cache_hits==0,'whole-sidecar cache hits need explicit verified baseline reuse receipt')
    need(start['vertical_parent_result_sha256']==spec['vertical']['parent_result']['sha256'] and start['baseline_source_count']==spec['main']['source_count'] and start['recovery_source_count']==spec['recovery']['source_count'] and start['API_calls']==0 and start['PTN_modified'] is False and end['PTN_modified'] is False,'sequence vertical/scope identity differs')
    recovery=spec['recovery'];home=r.path(recovery['run_root']);preserve=r.load(home/'resume-byte-preservation.json');need(type(preserve['before_source_count'])is int and 0<preserve['before_source_count']<recovery['source_count'] and preserve['after_source_count']==recovery['source_count'] and preserve['existing_sha_bytes_mtime_unchanged'] is True and preserve['missing_recovery_verified'] is True,'actual missing-source recovery not established')
    before=r.load(home/'intentional-interruption-after.json');need(before['GPU_returned'] is True and len(before['sidecars'])==preserve['before_source_count'],'interrupted resource/source count differs');idle_probe(before['resource_return_double_probe'],validation_targets(spec))
    intent=r.load(home/'intentional-interruption-before.json');owned=r.load(home/'supervisor-start-attempt1.json');exited=r.load(home/'supervisor-exit-attempt1.json')
    need(exited['returncode']!=0 and exited['intentional_TERM_sent'] is True and owned['intentional_interruption'] is True,'real intentional interruption not established')
    need(intent['parent_pid']==owned['container_pid']==owned['PGID']==owned['SID'] and intent['parent_start_ticks']==owned['start_ticks'] and intent['argv']==owned['argv'] and owned['pidfd_inode']>0,'interrupted parent exact PID/start/session/argv identity differs')
    need(intent['sidecars'] and set(intent['sidecars'])<=set(before['sidecars']) and intent['completed_chunk_ids'],'no actual committed partial progress before TERM')
    drain=r.load(home/'interruption-drain-complete.json');need(drain['drain']['all_attested_contexts_absent'] is True and len(drain['proofs'])==2,'interrupted contexts not drained')
    interrupted_uuids=set()
    for proof in drain['proofs']:
        ch=r.load(proof['path'],proof['sha256']);identity=proof['known_identity'];observed=proof['observed_container_process'];interrupted_uuids.add(identity['uuid']);folder=home/'resources-attempt1'/identity['worker_id']
        need(r.path(proof['path'])==folder/'context_challenge/attestation.json' and r.load(folder/'worker_identity.json')==identity,'interruption evidence not from actual attempt1 worker')
        ticket=r.load(folder/'admission.json');att=r.load(folder/'resource_attestation.json')
        for item in [ticket,att]:need(item['nonce']==identity['nonce'] and item['worker_id']==identity['worker_id'] and item['bundle_sha256']==spec['bundle']['sha256'] and item['contract_sha256']==spec['pilot_contract']['sha256'] and item['index_sha256']==recovery['index']['sha256'],'interruption worker contract/nonce/index differs')
        need(att['ticket_sha']==r.seen[str(folder/'admission.json')] and att['challenge_attestation_sha']==proof['sha256'] and att['known_container_worker_pid']==identity['container_pid'] and att['UUIDs']==[identity['uuid']],'interruption challenge/ticket resource chain differs')
        validate_challenge(ch,identity)
        need(not observed['exists'] or observed['start_ticks']!=identity['started'] or observed['state']=='Z','interrupted worker still alive')
    need(interrupted_uuids==validation_targets(spec),'interrupted GPU identity union differs')
    # Independently verify every previously committed SHA/mtime/bytes against current files.
    for relative,expected in before['sidecars'].items():
        path=home/'labels'/relative;raw=r.read(path,expected['sha256']);s=path.stat();need(s.st_size==expected['bytes'] and s.st_mtime_ns==expected['mtime_ns'],'old committed sidecar bytes/mtime changed')
    final=r.load(home/'resume-final-sidecar-snapshot.json');need(len(final['sidecars'])==recovery['source_count'] and final['old_completed_chunk_markers_unchanged'] is True,'actual final recovery snapshot incomplete')
    need(all(final['sidecars'].get(k)==v for k,v in before['sidecars'].items()),'old sidecar SHA/bytes/mtime not preserved in final snapshot')
    for relative,expected in final['sidecars'].items():
        need(not PurePosixPath(relative).is_absolute() and '..' not in PurePosixPath(relative).parts,'unsafe final sidecar snapshot path');path=home/'labels'/relative;r.read(path,expected['sha256']);stat=path.stat();need(stat.st_size==expected['bytes'] and stat.st_mtime_ns==expected['mtime_ns'],'actual final sidecar differs from remote snapshot')
    completed=final['completed_chunk_markers'];recovery_index=r.ref(recovery['index']);need(set(completed)=={s['id']+'.json' for s in recovery_index['shards']},'actual final completed marker union differs')
    need(all(completed.get(k)==v for k,v in intent['completed_chunk_markers'].items()),'old completed marker bytes changed')
    for name,pin in completed.items():r.read(home/'queue/completed'/name,pin)
    seed=r.load(home/'partial-stage-cache-seed.json');need(len(set(seed['source_ids']))==8 and seed['aligner_packets_copied']==seed['sidecars_copied']==0 and seed['new_model_execution_claimed_for_copies'] is False and seed['contract_sha256']==spec['pilot_contract']['sha256'] and seed['index_sha256']==recovery['index']['sha256'],'ASR-only8 source seed contract differs')
    need(all(x not in before['sidecars'] for x in seed['pending_sidecar_relpaths']) and len(seed['pending_sidecar_relpaths'])==8,'late partial seeds already completed before restart')
    missing=r.load(home/'partial-FA-missing-before-restart.json');need(missing['count']==len(set(missing['paths']))>0 and set(missing['paths'])<={x['missing_FA_cache_path'] for x in seed['packets']},'specific missing FA evidence differs')
    specific=r.load(home/'specific-partial-stage-consumption.json');need(specific['fresh_missing_FA_windows']>0 and specific['verified'] and all(x['ASR_actual_cache_hit'] is True for x in specific['verified']),'specific copied ASR/missing FA not executed')
    need({(x['source_id'],x['window_id']) for x in specific['verified']}=={(x['source_id'],x['window_id']) for x in seed['packets']} and {x['source_id'] for x in seed['packets']}==set(seed['source_ids']),'specific seed source/window coverage differs')
    for packet in seed['packets']:r.read(packet['destination'],packet['sha256'])
    comparison=r.ref(recovery['native_comparison']);need(comparison['source_count']==recovery['source_count'] and comparison['matched_windows']>0 and comparison['strict_source_and_model_observations_equal'] is True and comparison['every_changed_source']==[],'raw/native/quality recovery comparison failed')
    review=r.ref(spec['independent_review']);need(review['schema']=='ultra-independent-three-train-review/v1' and review['passed'] is True and review['bundle_sha256']==spec['bundle']['sha256'] and review['pilot_contract_sha256']==spec['pilot_contract']['sha256'] and review['main_index_sha256']==spec['main']['index']['sha256'] and review['recovery_index_sha256']==recovery['index']['sha256'],'C review stale or wrong version')
    scope=review['scope'];need(scope['actual_sidecars_reopened']==spec['main']['source_count'] and scope['recovery_source_count']==recovery['source_count'] and scope['three_training_scope_verified'] is True and scope['diagnostic34_exclusion_verified'] is True and scope['no_software_failures'] is True and scope['source_clock_and_original_vs_repaired_reviewed'] is True and scope['silver_generation_admitted'] is True and scope['human_gold_claimed'] is False,'independent actual quality review gate missing')
    need(scope['baseline_source_count']==main['source_count'] and scope['fresh_source_count']==main['source_count']-cache_hits and scope['reused_sidecar_count']==cache_hits,'C baseline/fresh/reused accounting differs')
    need(utc(review['recorded_utc'])>=max(utc(p['finished_utc']) for p in parents.values()),'C review predates final W2 evidence')
    for path,pin in review['evidence_file_sha256'].items():r.read(path,pin)
    need(review['evidence_file_sha256'],'C review has no concrete evidence SHA')
    review_refs={str(r.path(p)):h for p,h in review['evidence_file_sha256'].items()}
    for stage in ['main','recovery']:
        ref=spec[stage]['parent_result'];need(review_refs.get(str(r.path(ref['path'])))==ref['sha256'],'C review does not bind actual '+stage+' result')
    need(review_refs.get(str(r.path(recovery['native_comparison']['path'])))==recovery['native_comparison']['sha256'],'C review does not bind raw/native recovery comparison')
    if 'baseline_receipt' in main:need(review_refs.get(str(r.path(main['baseline_receipt']['path'])))==main['baseline_receipt']['sha256'],'C review does not bind actual baseline reuse receipt')
    need(review_refs.get(str(home/'resume-final-sidecar-snapshot.json'))==r.seen[str(home/'resume-final-sidecar-snapshot.json')],'C review does not bind actual final recovery snapshot')
    return {'schema':'ultra-eight-three-train-result/v2','passed':True,'bundle_sha256':spec['bundle']['sha256'],'pilot_contract_sha256':spec['pilot_contract']['sha256'],'silver_contract_sha256':spec['silver_contract']['sha256'],'full_index_sha256':spec['full_index']['sha256'],'review_sha256':spec['independent_review']['sha256'],'evidence_file_sha256':r.seen,'human_gold_claimed':False,'model_API_calls':0,'GPU_execution_or_submission':False}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha',required=True);p.add_argument('--live',action='store_true',help='Production: read actual server paths; refuse inspection remapping');a=p.parse_args();raw=a.spec.read_bytes();need(sha(raw)==a.spec_sha,'admission spec changed');spec=json.loads(raw)
    if a.live:need(not spec.get('evidence_path_map'),'live admission cannot use a local inspection path map')
    result=verify(spec,Reader(spec));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
