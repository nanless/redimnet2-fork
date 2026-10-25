"""Bounded foreground validation; only this supervisor's exact parent is interrupted.

Run with the existing qwen3-asr interpreter. Each GPU launcher independently
performs fresh idle double probes and UUID locks. No persistent scheduler.
"""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time,traceback
# The maintained supervisor has no default server, UUID or frozen experiment.
HERE=Path(__file__).resolve().parent
from gpu_common import need,sha,write,process_start,open_pidfd,wait_attested_context_drain,DrainState,verify_bundle
from gpu_vendor import resources as G

def configure(path,pin):
    global BASE,CODE,ROOT,BUNDLE,B1,P2,P3,UUIDS,LOCK_ROOT,VERTICAL_SHA,CONFIG_SHA
    path=Path(path);need(sha(path)==pin,'supervisor config bytes differ')
    cfg=json.loads(path.read_text());need(cfg['schema']=='annotation-three-training-validation/v1','wrong supervisor config')
    BASE=Path(cfg['task_root']).resolve();CODE=Path(cfg['code_root']).resolve();ROOT=Path(cfg['output_root']).resolve()
    need(CODE==HERE and BASE.is_dir() and BASE in ROOT.parents and BASE in CODE.parents and not (ROOT==CODE or CODE in ROOT.parents or ROOT in CODE.parents),'isolated frozen code/validation roots required')
    BUNDLE=cfg['bundle_sha256'];verify_bundle(CODE,BUNDLE)
    B1=(cfg['pilot_contract_name'],cfg['pilot_contract_sha256'])
    need(Path(B1[0]).name==B1[0] and len(B1[1])==64,'unsafe pilot contract identity')
    need(sha(CODE/'contracts'/B1[0])==B1[1],'pilot contract SHA differs')
    P2=(Path(cfg['recovery_index']['path']),cfg['recovery_index']['sha256'])
    P3=(Path(cfg['main_index']['path']),cfg['main_index']['sha256'])
    for item,count in [(P2,512),(P3,3000)]:
        need(sha(item[0])==item[1] and json.loads(item[0].read_text())['source_count']==count,'explicit validation index count/SHA differs')
    values=cfg['validation_UUIDs'];need(len(values)==2 and len(set(values))==2,'two explicit UUIDs required')
    UUIDS=','.join(values);LOCK_ROOT=Path(cfg['device_lock_root']).resolve();VERTICAL_SHA=cfg['vertical_index_sha256'];CONFIG_SHA=pin
    need(BASE not in LOCK_ROOT.parents and LOCK_ROOT!=BASE,'shared device locks must remain outside per-task root')
    return cfg

def args(name,attempt,contract,index):
    home=ROOT/name
    return [sys.executable,'-B',str(CODE/'gpu_launch.py'),'--execute','--bundle-sha',BUNDLE,
        '--contract',str(CODE/'contracts'/contract[0]),'--contract-sha',contract[1],
        '--shard-index',str(index[0]),'--index-sha',index[1],
        '--queue-root',str(home/'queue'),'--output-root',str(home/'labels'),
        '--output',str(home/('resources-attempt'+str(attempt))),
        '--run-root',str(BASE),'--device-lock-root',str(LOCK_ROOT),'--target-uuids',UUIDS,'--memory-budget-mib','60000','--max-runtime-seconds','14400','--full-verify']

def snapshot_labels(home):
    root=home/'labels'
    return {str(p.relative_to(root)):{'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns,'bytes':p.stat().st_size}
            for p in root.rglob('*.annotation.json')}

def launch(name,attempt,contract,index,interrupt=False):
    home=ROOT/name;home.mkdir(parents=True,exist_ok=True)
    command=args(name,attempt,contract,index);out=home/('parent-attempt'+str(attempt)+'.log')
    need(not out.exists(),'old attempt log exists; inspect before resuming')
    with out.open('x') as stream:
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=None;fd=None;interrupted=False;started=time.monotonic()
        try:
            ticks=process_start(proc.pid);fd=open_pidfd(proc.pid)
            write(home/('supervisor-start-attempt'+str(attempt)+'.json'),{
                'argv':command,'container_pid':proc.pid,'start_ticks':ticks,'supervisor_pid':os.getpid(),
                'supervisor_start_ticks':process_start(os.getpid()),'pidfd_inode':os.fstat(fd).st_ino,
                'PGID':os.getpgid(proc.pid),'SID':os.getsid(proc.pid),'intentional_interruption':interrupt})
            print(json.dumps({'phase':'started','name':name,'attempt':attempt,'parent_pid':proc.pid}),flush=True)
            while proc.poll() is None:
                need(time.monotonic()-started<15000,'supervisor bounded deadline exceeded')
                commits=list((home/'queue/completed').glob('*.json'))
                if interrupt and commits:
                    need(process_start(proc.pid)==ticks and os.getpgid(proc.pid)==os.getsid(proc.pid)==proc.pid,
                         'owned parent identity differs before intentional TERM')
                    actual=Path(f'/proc/{proc.pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
                    need(actual==[x.encode() for x in command],'owned parent argv differs before TERM')
                    # The live Popen object plus kernel pidfd/start/argv identify this one parent.
                    need(hasattr(signal,'pidfd_send_signal'),'verified pidfd signal API required')
                    before=snapshot_labels(home)
                    seed_path=home/'partial-stage-cache-seed.json'
                    if seed_path.exists():
                        seed=json.loads(seed_path.read_text())
                        need(all(x not in before for x in seed['pending_sidecar_relpaths']), 'seeded partial stages were already fully published before TERM')
                    need(0<len(before)<512,'P2 must actually leave missing sources before TERM')
                    write(home/'intentional-interruption-before.json',{
                        'parent_pid':proc.pid,'parent_start_ticks':ticks,'argv':command,
                        'completed_chunk_ids':[p.stem for p in commits], 'completed_chunk_markers':{p.name:sha(p) for p in commits}, 'sidecars':before,
                        'scope':'TERM own foreground GPU launcher only; launcher cleans its attested sessions; no other process signalled'})
                    signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
                    interrupted=True;break
                time.sleep(.5)
            rc=proc.wait(timeout=60 if interrupted else 1)
        except BaseException:
            # An unexpected supervisor error must not launch the next GPU job.
            # Ask only its exact still-live parent to run its own guarded cleanup.
            if proc.poll() is None:
                actual_start=process_start(proc.pid)
                need((ticks is None or actual_start==ticks) and os.getpgid(proc.pid)==os.getsid(proc.pid)==proc.pid
                     and int(Path(f'/proc/{proc.pid}/status').read_text().split('PPid:')[1].split()[0])==os.getpid(),
                     'cannot identify owned parent for supervisor abort')
                actual=Path(f'/proc/{proc.pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
                need(actual==[x.encode() for x in command], 'owned parent argv changed on abort')
                if fd is None:
                    # This unreaped, still-live Popen child has exact PPID/argv/session proof.
                    proc.terminate()
                else:signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
                proc.wait(timeout=60)
            raise
        finally:
            if fd is not None:os.close(fd)
        print(json.dumps({'phase':'finished','name':name,'attempt':attempt,'rc':rc,'interrupted':interrupted}),flush=True)
        write(home/('supervisor-exit-attempt'+str(attempt)+'.json'),{'returncode':rc,'intentional_TERM_sent':interrupted})
        if interrupt:
            need(interrupted and rc!=0,'interruption was not actually exercised')
            # Caller may restart only after the fresh launcher again sees both devices idle.
            drains=[];proofs=[]
            for w in sorted((home/('resources-attempt'+str(attempt))).glob('worker-*')):
                identity=json.loads((w/'worker_identity.json').read_text())
                proof_path=w/'context_challenge/attestation.json';proof=json.loads(proof_path.read_text())
                need(proof['complete'] and proof['known_container_worker_pid']==identity['container_pid']
                     and proof['UUIDs']==[identity['uuid']] and len(proof['phases'])==7,
                     'interrupted worker context proof differs')
                need(all(p['request']['nonce']==identity['nonce'] for p in proof['phases']), 'context nonce differs')
                try:
                    rawstat=Path(f"/proc/{identity['container_pid']}/stat").read_text()
                    fields=rawstat[rawstat.rfind(')')+2:].split()
                    observed_worker={'exists':True,'start_ticks':int(fields[19]),'state':fields[0]}
                except FileNotFoundError:observed_worker={'exists':False}
                need(not observed_worker['exists'] or observed_worker['start_ticks']!=identity['started']
                     or observed_worker['state']=='Z', 'interrupted owned worker still alive')
                drains.append(DrainState([identity['uuid']],proof['opaque_nvml_pid']))
                proofs.append({'path':str(proof_path),'sha256':sha(proof_path),'known_identity':identity,'observed_container_process':observed_worker})
            need(set(u for d in drains for u in d.uuids)==set(UUIDS.split(',')), 'interrupted proof UUID set differs')
            def retain(snapshot):
                with (home/'interruption-drain-samples.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
            settled=wait_attested_context_drain(drains,G.capture_snapshot,on_snapshot=retain)
            write(home/'interruption-drain-complete.json',{'drain':settled,'proofs':proofs})
            snap=G.capture_snapshot()
            targets=UUIDS.split(',')
            indices=[next(d['index'] for d in snap['devices'] if d['uuid']==u) for u in targets]
            probe=G.double_probe(indices,60000,interval_seconds=5.,reserve_mib=8192)
            need([d['uuid'] for d in probe['devices']]==targets,'P2 resource return UUID differs')
            write(home/'intentional-interruption-after.json',{'sidecars':snapshot_labels(home),'GPU_returned':True,'resource_return_double_probe':probe})
        else:
            need(rc==0,'GPU validation failed: '+name)
            result=json.loads((home/('resources-attempt'+str(attempt))/'parent_result.json').read_text())
            need(result['verification']['passed'] and result['verification']['full_sidecar_readback_verified'],'full pilot readback missing')
    return home


def seed_partial_asr(cold, target):
    """Only restore real ASR packets for eight late, FA-eligible pending sources."""
    from contracts import read_contract,digest,atomic_new,model_provenance
    from sidecar import SidecarStore,StageCache
    from worker import _stage_inputs
    contract=read_contract(CODE/'contracts'/B1[0],B1[1]);models=model_provenance(contract)
    need(sha(P2[0])==P2[1], 'recovery index bytes changed before ASR seeding')
    index=json.loads(P2[0].read_text());sources=[]
    for shard in index['shards']:
        p=P2[0].parent/shard['path'];need(sha(p)==shard['sha256'],'cold source shard changed')
        sources.extend(json.loads(x) for x in p.read_text().splitlines())
    old=cold/'labels'/contract['version'];old_cache=StageCache(old/'.stage_cache')
    selected=[]
    for src in reversed(sources[-64:]):
        record=json.loads((old/src['sidecar_relpath']).read_text())
        if record['windows'] and all(w['asr']['status']=='completed' and w['asr'].get('normalized_text')
                and w['alignment']['status']=='completed_native_candidates' for w in record['windows']):
            available = True
            for window in record['windows']:
                packet_inputs = _stage_inputs('asr', window, None, contract, B1[1], models)
                if old_cache.get('asr', packet_inputs) != window['asr']:
                    available = False
                    break
            if available:
                selected.append((src,record))
        if len(selected)==8:break
    need(len(selected)==8,'eight late real ASR+FA producers unavailable; do not fake partial-stage test')
    target.mkdir(parents=True,exist_ok=False)
    store=SidecarStore(target/'labels',contract,B1[1]);cache=StageCache(store.root/'.stage_cache')
    packets=[]
    for src,record in selected:
        for w in record['windows']:
            inputs=_stage_inputs('asr',w,None,contract,B1[1],models)
            key=digest(inputs);origin=old_cache.path('asr',key)
            output=old_cache.get('asr',inputs)
            need(output==w['asr'],'producer sidecar and real ASR cache differ')
            call=output['inference_call']
            need(call['rows']==1 and call['execution_policy']=='single_window_independent','producer was not a real single-row call')
            destination=cache.path('asr',key);atomic_new(destination,origin.read_bytes())
            need(sha(destination)==sha(origin) and cache.get('asr',inputs)==output,'copied packet readback differs')
            fa_inputs=_stage_inputs('aligner',w,None,contract,B1[1],models)
            packets.append({'original':str(origin),'destination':str(destination),'sha256':sha(origin),
                'source_id':src['source_id'],'sidecar_relpath':src['sidecar_relpath'],'window_id':w['window_id'],
                'input_pcm_sha256':w['input_pcm_sha256'],'producer_row_ids':call['producer_row_ids'],
                'missing_FA_cache_path':str(cache.path('aligner',digest(fa_inputs)))})
    need(not (store.root/'.stage_cache/aligner').exists() and not list(store.root.rglob('*.annotation.json')),'seed must not copy FA or completed sidecars')
    write(target/'partial-stage-cache-seed.json',{'schema':'real-ASR-only-partial-cache-restore/v1','contract_sha256':B1[1],
        'index_sha256':P2[1],'producer_cold_run':str(cold),'source_ids':sorted(src['source_id'] for src,r in selected),
        'pending_sidecar_relpaths':[contract['version']+'/'+src['sidecar_relpath'] for src,r in selected],
        'packets':packets,'new_model_execution_claimed_for_copies':False,'aligner_packets_copied':0,'sidecars_copied':0})
    return len({p['destination'] for p in packets})


# Source of inherited exact-parent interrupt/drain/ASR-stage seeding logic:
# ReDimNet2全量开源标注_20261024/全量实施_20261024/资源与执行/两卡最终单窗验收_v5.py
# SHA256 3c49355171a8378d372d7f11450a10ded5485a63c82aeed1d5a1f5e3602cfa0c


def compare_recovery_subset(cold, resumed):
    """Strict source/model comparison; cache and execution telemetry stay separate."""
    from contracts import read_contract, digest
    contract = read_contract(CODE/'contracts'/B1[0], B1[1])
    need(sha(P2[0]) == P2[1], 'recovery index bytes changed before comparison')
    tree = contract['version']; index = json.loads(P2[0].read_text())
    sources = []
    for shard in index['shards']:
        path = P2[0].parent/shard['path']
        need(sha(path) == shard['sha256'], 'recovery input shard changed')
        sources.extend(json.loads(x) for x in path.read_text().splitlines())
    need(len(sources) == len({s['source_id'] for s in sources}) == 512, 'recovery source set differs')
    differences = []; cache_hits = {'asr': 0, 'aligner': 0}; matched_windows = 0
    for source in sources:
        records = []
        for run in (cold, resumed):
            path = run/'labels'/tree/source['sidecar_relpath']
            record = json.loads(path.read_text())
            commit = json.loads(path.with_name(path.name+'.commit.json').read_text())
            need(record['source'] == source and record['contract']['sha256'] == B1[1], 'source/contract differs')
            need(commit['sidecar_sha256'] == sha(path) and commit['source_record_sha256'] == digest(source)
                 and commit['contract_sha256'] == B1[1], 'sidecar commit binding differs')
            records.append(record)
        left, right = records
        changed = []
        for field in ('source', 'source_identity', 'vad', 'units', 'decision', 'quality_axes', 'errors'):
            if left.get(field) != right.get(field): changed.append(field)
        windows = []
        for record in records:
            windows.append([{k: v for k, v in w.items() if k not in ('asr_cache_hit', 'aligner_cache_hit')}
                            for w in record['windows']])
        if windows[0] != windows[1]: changed.append('complete_native_windows_except_cache_telemetry')
        matched_windows += len(left['windows'])
        for w in right['windows']:
            cache_hits['asr'] += w.get('asr_cache_hit') is True
            cache_hits['aligner'] += w.get('aligner_cache_hit') is True
        if changed:
            differences.append({'source_id': source['source_id'], 'changed_fields': changed,
                                'cold': left, 'resumed': right})
    report = {'schema': 'ultra-actual-cold-partial-recovery-comparison/v1',
        'source_count': len(sources), 'matched_windows': matched_windows, 'cache_hits': cache_hits,
        'strict_source_and_model_observations_equal': not differences,
        'operational_execution_telemetry_not_compared_as_model_truth': True,
        'every_changed_source': differences, 'human_boundary_accuracy_certified': False}
    write(resumed/'baseline3000-vs-recovery512-native-comparison.json', report)
    need(not differences, 'cold versus recovery raw/model/quality differs; retain and investigate')
    return report


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True);p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);a=p.parse_args()
    configure(a.config,a.config_sha)
    need(hasattr(signal, 'pidfd_send_signal'), 'pidfd signal API required')
    need(not (ROOT/'ultra-sequence-start.json').exists(), 'sequence already started; inspect, do not duplicate')
    vertical = ROOT/'vertical30/resources-attempt1/parent_result.json'
    result = json.loads(vertical.read_text())
    need(result['bundle_sha256'] == BUNDLE and result['contract_sha256'] == B1[1]
         and result['index_sha256'] == VERTICAL_SHA
         and result['resource_returned'] is True and result['verification']['completed_source_count'] == 30
         and result['verification']['passed'] and result['verification']['full_sidecar_readback_verified'],
         'actual compact vertical30 readback not passed')
    ROOT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt('owned supervisor terminated')))
    write(ROOT/'ultra-sequence-start.json', {'supervisor_sha256': sha(__file__), 'supervisor_config_sha256':CONFIG_SHA,
        'pid': os.getpid(), 'start_ticks': process_start(os.getpid()), 'bundle_sha256': BUNDLE,
        'contract_sha256': B1[1], 'baseline_source_count': 3000, 'recovery_source_count': 512,
        'vertical_parent_result_sha256': sha(vertical), 'API_calls': 0, 'PTN_modified': False})
    cold = ROOT/'P1-training3000-baseline'
    cold_result = cold/'resources-attempt1/parent_result.json'
    if cold.exists():
        completed = json.loads(cold_result.read_text())
        need(completed['bundle_sha256'] == BUNDLE and completed['contract_sha256'] == B1[1]
             and completed['index_sha256'] == P3[1] and completed['resource_returned'] is True
             and completed['verification']['passed']
             and completed['verification']['full_sidecar_readback_verified']
             and completed['verification']['completed_source_count'] == 3000,
             'existing baseline3000 is not complete and verified; do not duplicate or resume blindly')
        write(ROOT/'existing-baseline3000-accepted.json', {'parent_result_path': str(cold_result),
            'sha256': sha(cold_result), 'source_count': 3000, 'new_model_execution_claimed_here': False})
    else:
        cold = launch('P1-training3000-baseline', 1, B1, P3)
    target = ROOT/'P2-training512-partial-resume'
    seed_count = seed_partial_asr(cold, target)
    home = launch(target.name, 1, B1, P2, interrupt=True)
    before = json.loads((home/'intentional-interruption-after.json').read_text())['sidecars']
    seed = json.loads((home/'partial-stage-cache-seed.json').read_text())
    need(all(x not in before for x in seed['pending_sidecar_relpaths']), 'partial sources already completed before restart')
    missing_fa = {p['missing_FA_cache_path'] for p in seed['packets'] if not Path(p['missing_FA_cache_path']).exists()}
    need(missing_fa, 'no actual FA stage left missing')
    write(home/'partial-FA-missing-before-restart.json', {'paths': sorted(missing_fa), 'count': len(missing_fa)})
    launch(target.name, 2, B1, P2)
    after = snapshot_labels(home)
    need(len(after) == 512 and len(after) > len(before), 'restart did not actually fill missing sources')
    need(all(after.get(k) == v for k, v in before.items()), 'completed sidecar SHA/mtime/bytes changed')
    before_markers = json.loads((home/'intentional-interruption-before.json').read_text())['completed_chunk_markers']
    after_markers = {p.name: sha(p) for p in (home/'queue/completed').glob('*.json')}
    need(all(after_markers.get(k) == v for k, v in before_markers.items()), 'previous completed chunk marker changed')
    write(home/'resume-final-sidecar-snapshot.json', {'sidecars': after,
        'completed_chunk_markers': after_markers,
        'old_completed_chunk_markers_unchanged': True,
        'source': 'actual remote stat/bytes/SHA after GPU recovery; not archive extraction metadata'})
    from contracts import read_contract
    contract = read_contract(CODE/'contracts'/B1[0], B1[1]); fresh_fa = 0; verified = []
    summaries = [json.loads(p.read_text()) for p in (home/'resources-attempt2').glob('worker-*/worker_summary.json')]
    need(sum(x['stage_cache_hits']['asr'] for x in summaries) >= seed_count, 'seeded ASR cache not consumed')
    for packet in seed['packets']:
        need(sha(packet['destination']) == packet['sha256'], 'seeded packet changed')
        data = json.loads(Path(packet['destination']).read_text())
        record = json.loads((home/'labels'/contract['version']/packet['sidecar_relpath']).read_text())
        window = next(w for w in record['windows'] if w['window_id'] == packet['window_id'])
        need(window['asr_cache_hit'] is True and window['asr'] == data['output']
             and window['input_pcm_sha256'] == packet['input_pcm_sha256'], 'specific ASR seed was not consumed')
        if packet['missing_FA_cache_path'] in missing_fa and not window['aligner_cache_hit']:
            need(window['alignment']['status'] == 'completed_native_candidates', 'missing FA did not complete')
            fresh_fa += 1
        verified.append({'source_id': packet['source_id'], 'window_id': packet['window_id'],
                         'ASR_actual_cache_hit': True, 'FA_actual_cache_hit': window['aligner_cache_hit']})
    need(fresh_fa > 0, 'missing FA not actually reexecuted')
    write(home/'specific-partial-stage-consumption.json', {'verified': verified, 'fresh_missing_FA_windows': fresh_fa})
    write(home/'resume-byte-preservation.json', {'before_source_count': len(before), 'after_source_count': len(after),
        'existing_sha_bytes_mtime_unchanged': True, 'missing_recovery_verified': True,
        'real_partial_ASR_packets': seed_count, 'no_copied_packet_claimed_as_new_inference': True})
    comparison = compare_recovery_subset(cold, home)
    write(ROOT/'ultra-sequence-complete.json', {'passed': True, 'baseline_source_count': 3000,
        'actual_recovery_source_count': 512, 'native_equivalence': comparison['strict_source_and_model_observations_equal'],
        'API_calls': 0, 'PTN_modified': False, 'scope': 'same-host W2 inference/recovery; independent text/boundary review and cloud submission separate',
        'human_boundary_accuracy_certified': False, 'finished_epoch': time.time()})
    print('COMPLETE_ULTRA_W2_COUNT_BASED_SEQUENCE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        error = {'traceback': traceback.format_exc(), 'pid': os.getpid(),
                 'scope': 'preserve all outputs; no automatic retry or cloud submission'}
        if 'ROOT' in globals() and ROOT.is_dir() and not (ROOT/'ultra-sequence-failure.json').exists():
            write(ROOT/'ultra-sequence-failure.json', error)
        print(json.dumps(error), flush=True)
        raise
