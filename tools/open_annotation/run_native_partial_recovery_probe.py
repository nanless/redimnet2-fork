"""固定30来源的真实整批ASR中途恢复探针；默认只检查，不执行GPU。

只在全新独立两卡任务使用。正常冷跑后，在另一新树观察首个原子整批
结果，让原SerialExecutor精确TERM本次父进程并等待原cleanup；新一次
UUID锁和NVML双空闲证明后，才以同合同/队列/结果目录启动attempt2。
不改冻结计算源码，不注入worker hook，不操作现有标注或PTN，不建数据库。
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import signal
import stat
import threading
import time
import traceback
from contracts import atomic_new,canonical_bytes,digest,model_provenance,need,safe_join
from run_bounded_speed_pilot import SerialExecutor,physical,disjoint,pin,file_identity,verify_parent
from run_native_batch_pilot import load_matrix

SCHEMA='annotation-native-mid-cohort-probe/v1'
ANCHOR='src_0010c3bdaebf5268747c4f2707a36c3234d97791cd05f8fc95e17263e6b48852'
SHARD_ID='mid-cohort-fixed30'
CONTROL_DEPENDENCIES=('contracts.py','run_bounded_speed_pilot.py','run_native_batch_pilot.py','audit_native_transformers_batch.py',
    'gpu_common.py','readonly_progress.py','gpu_vendor/resources.py','gpu_vendor/topology.py','gpu_vendor/__init__.py')


def planned_requests(records,sources):
    rows=[]
    for source in sources:
        record=records[source['source_id']]
        need(record['source']==source and record['source_identity'].get('decode_status')=='decoded_complete',
             'first source group must have verified decoded originals')
        for w in record['windows']:
            need(w.get('preprocessing_status')=='completed','first source group contains a failed input')
            rows.append(dict(row_id=source['source_id']+':'+w['window_id'],source_id=source['source_id'],window_id=w['window_id'],
                source_record_sha256=digest(source),input_pcm_sha256=w['input_pcm_sha256'],
                model_input_frames=w['model_input_frames'],model_sample_rate=w['model_sample_rate']))
    need(len({r['row_id']for r in rows})==len(rows)and len(rows)<=256,'finite unique window group required')
    return sorted(rows,key=lambda r:(r['model_input_frames'],r['row_id']))


def compile_plan(plan_ref,output):
    plan=pin(plan_ref);need(plan.get('schema')==SCHEMA and plan.get('purpose')in {'production','cpu_fixture'},'wrong recovery probe schema/purpose')
    need(plan.get('controller_sha256')==file_identity(Path(__file__).resolve())['sha256'],'probe controller source SHA differs from fixed plan')
    need(plan.get('wall_budget_seconds')==1800,'probe has exact maximum1800 second budget')
    profile=plan.get('profile');need(profile in {'native-fp32-sdpa-b4','native-fp32-sdpa-b8'},'only actual native batch4/8 probe allowed')
    matrix=load_matrix(plan['candidate_prepared'],30,cpu_fixture=plan['purpose']=='cpu_fixture');v=matrix['variants'][profile]
    need(v['contract']['runtime'].get('source_batch_rows',8)==8,'probe requires actual source group width8')
    need(v['contract']['purpose']=='pilot_validation','probe cannot operate formal production contracts')
    from audit_native_transformers_batch import load_track
    observed=load_track(v['track_ref']['path'],v['track_ref']['sha256'])
    need(set(observed['records'])==set(matrix['rows'])and len(matrix['rows'])==30,'fixed30 observed sources differ')
    anchor=plan.get('anchor_source_id')
    need(anchor in matrix['rows']and (plan['purpose']=='cpu_fixture'or anchor==ANCHOR),'explicit verified long anchor required')
    ar=observed['records'][anchor]
    if plan['purpose']=='production':
        need(ar['source_identity']['frames']==401408 and ar['source_identity']['sample_rate']==16000 and len(ar['windows'])==2,
             'anchor must retain original25.088 second/two-window identity')
    eligible=[]
    for sid in sorted(matrix['rows']):
        if sid==anchor:continue
        rec=observed['records'][sid]
        if rec['source_identity'].get('decode_status')=='decoded_complete'and rec.get('windows')and all(w.get('preprocessing_status')=='completed'for w in rec['windows']):eligible.append(sid)
    need(len(eligible)>=7,'not enough verified inputs for first8 source group')
    first_ids=[anchor,*eligible[:7]];ordered_ids=first_ids+sorted(set(matrix['rows'])-set(first_ids))
    sources=[matrix['rows'][sid]for sid in ordered_ids];requests=planned_requests(observed['records'],sources[:8])
    batch=v['contract']['runtime']['asr_batching']['max_requests'];need(len(requests)>batch and len(requests)>=9,'first group must contain9+windows and an uncommitted later cohort')
    output=physical(output);need(not output.exists()and output.parent.is_dir(),'new probe root required')
    here=Path(__file__).resolve().parent
    repo=next((p for p in (here,*here.parents)if(p/'AGENTS.md').is_file()and(p/'redimnet2').is_dir()),here)
    protected=plan.get('protected_roots');need(isinstance(protected,list)and 1<=len(protected)<=16,'explicit production/PTN protected roots required')
    inputs=[physical(plan_ref['path']),physical(plan['candidate_prepared']['path']).parent,physical(v['track']['index']['path']).parent,
            physical(v['fields']['--device-lock-root']),repo,*[physical(p)for p in protected]]
    need(all(disjoint(output,p)for p in inputs),'probe overlaps code, evidence, inputs, locks or existing tasks')
    need(all(v['members'].get(p)==file_identity(here/p)['sha256']for p in CONTROL_DEPENDENCIES),
         'probe lifecycle/auditor/resource helpers must match original frozen sources')
    return dict(plan=plan,plan_ref=plan_ref,variant=v,rows=matrix['rows'],sources=sources,requests=requests,output=output,batch=batch,
                observed_track_ref=v['track_ref'],observed_source_sha256=digest(observed['records']))


def variant_for(v,root,index,resource_name,*,uuids=None):
    flags=dict(v['fields']);track=dict(v['track'],index=index,queue_root=str(root/'queue'),output_root=str(root/'labels'),resources_root=str(root/resource_name))
    flags.update({'--shard-index':index['path'],'--index-sha':index['sha256'],'--queue-root':track['queue_root'],
        '--output-root':track['output_root'],'--output':track['resources_root'],'--run-root':str(root),'--max-runtime-seconds':'600'})
    if uuids is not None:flags['--target-uuids']=','.join(uuids)
    argv=list(v['argv'][:3])
    for key,value in flags.items():argv.extend([key]if value is True else[key,value])
    return dict(v,argv=argv,fields=flags,track=track)


def verify_controls(c):
    need(pin(c['plan_ref'])==c['plan'],'fixed probe plan changed between phases')
    here=Path(__file__).resolve().parent
    need(c['plan']['controller_sha256']==file_identity(Path(__file__).resolve())['sha256']and
         all(c['variant']['members'].get(p)==file_identity(here/p)['sha256']for p in CONTROL_DEPENDENCIES),
         'probe controller/lifecycle source changed between phases')
    if c['plan']['purpose']=='production':
        pin(c['plan']['candidate_prepared']);pin(c['variant']['track_ref'])


def expected_plan(c,requests,*,fixture=False):
    v=c['variant'];contract=v['contract'];model_sha=digest(model_provenance(contract)['asr'])
    producer=digest(dict(contract_sha256=v['track']['contract']['sha256'],bundle_sha256=v['fields']['--bundle-sha'],model_sha256=model_sha,
                         execution_policy=contract['runtime']['execution_policy'],asr_batching=contract['runtime']['asr_batching']))
    identity=dict(schema='annotation-native-transformers-file-store/v1',producer_sha256=producer,
        contract_sha256=v['track']['contract']['sha256'],model_sha256=model_sha,synthetic_fixture=fixture,
        policy='asr_fixed_cohort_fa_single/v1',max_batch_rows=c['batch'],alignment=dict(dtype='float32',attention='eager',max_batch_rows=1),
        database_created=False,model_API_calls=0,release_training_labels=False)
    cohorts=[]
    for start in range(0,len(requests),c['batch']):
        intent=dict(schema='annotation-native-transformers-cohort/v1',store_identity_sha256=digest(identity),
                    requests=requests[start:start+c['batch']],original_plan_requests_sha256=digest(requests))
        cohorts.append(dict(cohort_sha256=digest(intent),intent=intent))
    return dict(schema='annotation-native-transformers-plan/v1',store_identity=identity,requests=requests,cohorts=cohorts)


def group_path(c,v):
    return Path(v['track']['output_root'])/v['contract']['version']/'.native_torch_cohorts'/SHARD_ID/digest(c['sources'][:8])


def checked_packet(root,plan,cohort):
    path=root/'batches'/(cohort['cohort_sha256']+'.json')
    if not path.exists():return None
    physical(path);published=path.lstat();need(stat.S_ISREG(published.st_mode),'atomic packet must be a physical regular file')
    # atomic_new先link目标、fsync目录、最后unlink临时名；nlink2仍在发布过程中。
    # 中断要等到nlink1，不能自己制造首包孤立tmp再把探针误判失败。
    if published.st_nlink==2:return None
    need(published.st_nlink==1,'atomic packet has unexpected external hardlinks')
    value=pin(dict(path=str(path),sha256=file_identity(path)['sha256']),64*1024**2);content=value.get('content',{})
    need(value.get('schema')=='annotation-native-transformers-atomic-result/v1'and value.get('content_sha256')==digest(content)
         and content.get('cohort')==cohort and content.get('plan_sha256')==digest(plan)
         and content.get('store_identity')==plan['store_identity'],'observed atomic packet identity differs')
    rows=cohort['intent']['requests'];outputs=content.get('outputs')
    need(isinstance(outputs,dict)and set(outputs)=={r['row_id']for r in rows},'observed whole cohort incomplete')
    for i,r in enumerate(rows):
        n=outputs[r['row_id']].get('native',{})
        need(type(n.get('actual_generate_rows'))is int and n['actual_generate_rows']==len(rows)
             and type(n.get('actual_generate_row_index'))is int and n['actual_generate_row_index']==i,'observed actual row width/map differs')
    return dict(path=str(path),**file_identity(path))


def watch_first_packet(c,v,cancel,outer_cancel,stop,deadline,evidence,state):
    """只轮询这个新probe的首组固定计划；不遍历其它来源或任务。"""
    try:
        root=group_path(c,v);expected=expected_plan(c,c['requests'],fixture=c['plan']['purpose']=='cpu_fixture')
        while not stop.is_set():
            if outer_cancel.is_set()or time.monotonic()>=deadline:cancel.set();return
            p=root/'plan.json'
            if p.exists():
                plan=pin(dict(path=str(p),sha256=file_identity(p)['sha256']),64*1024**2)
                need(plan==expected,'first group plan changed from fixed source/PCM/order')
                first=checked_packet(root,plan,plan['cohorts'][0])
                if first is not None:
                    state['trigger']=dict(first_packet=first,plan=dict(path=str(p),sha256=digest(plan)),observed_monotonic=time.monotonic())
                    atomic_new(evidence/'first-packet-trigger.json',state['trigger']);cancel.set();return
            stop.wait(.02)
    except BaseException:
        state['error']=traceback.format_exc();atomic_new(evidence/'watcher-failure.json',dict(traceback=state['error']));cancel.set()


def snapshot_partial(c,v):
    root=group_path(c,v);plan=pin(dict(path=str(root/'plan.json'),sha256=file_identity(root/'plan.json')['sha256']))
    need(plan==expected_plan(c,c['requests'],fixture=c['plan']['purpose']=='cpu_fixture'),'interrupted plan changed')
    committed={};uncommitted=[]
    for cohort in plan['cohorts']:
        packet=checked_packet(root,plan,cohort)
        if packet is None:uncommitted.append(cohort['cohort_sha256'])
        else:committed[cohort['cohort_sha256']]=packet
    tree=Path(v['track']['output_root'])/v['contract']['version'];published={}
    for source in c['sources']:
        p=safe_join(tree,source['sidecar_relpath']);marker=Path(str(p)+'.commit.json')
        need(p.exists()==marker.exists(),'orphan sidecar/commit retained; no recovery may start')
        if p.exists():published[str(p)]=file_identity(p);published[str(marker)]=file_identity(marker)
    if tree.exists():
        count=0
        for path in tree.rglob('*'):
            count+=1;need(count<=4096,'new finite probe tree unexpectedly large')
            need(not path.is_symlink(),'symlink in interrupted tree retained/refused')
            need(not(path.is_file()and(path.name.endswith('.tmp')or path.name.startswith('.tmp'))),'orphan partial file retained/refused')
    return dict(plan=dict(path=str(root/'plan.json'),**file_identity(root/'plan.json')),committed_packets=committed,
                uncommitted_cohorts=uncommitted,published_sidecars=published,all_sources_published=len(published)==60)


def prove_release(v,uuids):
    """只在原自有父已经退出后拿同UUID锁，再读真实NVML两次；不发送进程信号。"""
    from gpu_vendor import resources as G
    initial=G.capture_snapshot();devices=[d for d in initial['devices']if d['uuid']in uuids]
    need(len(devices)==2 and {d['uuid']for d in devices}==set(uuids)and all('A800'in d['name']and d['total_mib']>=80000 for d in devices),
         'same two actual A800 UUIDs required after interrupted parent exit')
    devices=sorted(devices,key=lambda d:uuids.index(d['uuid']))
    with G.DeviceLocks(v['fields']['--device-lock-root'],uuids)as locks:
        probe=G.double_probe([d['index']for d in devices],60000,interval_seconds=5.,reserve_mib=8192)
        need([d['uuid']for d in probe['devices']]==uuids,'post-interruption UUID order changed')
        return dict(probe=probe,locks=copy.deepcopy(locks.evidence),only_after_owned_parent_exit=True,no_process_signalled=True)


def verify_preserved(before):
    for entry in before['committed_packets'].values():
        need(file_identity(entry['path'])=={k:v for k,v in entry.items()if k!='path'},'committed packet bytes/inode/mtime changed on resume')
    need(file_identity(before['plan']['path'])=={k:v for k,v in before['plan'].items()if k!='path'},'fixed plan changed on resume')
    for path,identity in before['published_sidecars'].items():need(file_identity(path)==identity,'published sidecar/commit changed on resume')


def normalized_results(track):
    """跨独立冷跑/恢复树，仅规范化已由auditor核过的packet根及缓存旗标。"""
    result=copy.deepcopy(track);tree=Path(track['spec']['output_root'])/track['contract']['version']
    for record in result['records'].values():
        for window in record.get('windows',[]):
            ref=window.get('asr',{}).get('durable_native_batch')
            if ref is not None:
                p=physical(ref['packet_path']);need(tree in p.parents,'audited native packet path escaped its own tree')
                ref['packet_path']=str(p.relative_to(tree))
    return result


def run(plan_ref,output,execute=False,*,_fixture_executor=None,_fixture_release=None):
    c=compile_plan(plan_ref,output)
    if not execute:return dict(checked=True,GPU_executed=False,source_count=30,first_group_sources=8,first_group_windows=len(c['requests']),
                               first_group_source_ids=[s['source_id']for s in c['sources'][:8]],database_created=False,partial_recovery_verified=False)
    fixture=c['plan']['purpose']=='cpu_fixture'
    need(fixture or(_fixture_executor is None and _fixture_release is None),'fake executor/resource proof forbidden for production')
    need(not fixture or(_fixture_executor is not None and _fixture_release is not None),'fixture cannot launch real GPU parent')
    need(fixture or platform.system()=='Linux','real probe requires Linux pidfd supervision')
    need(int(os.environ.get('WORLD_SIZE','1'))==1 and int(os.environ.get('RANK','0'))==0 and 'LOCAL_RANK'not in os.environ,'probe controller runs once outside DDP')
    out=c['output'];out.mkdir(mode=0o700);deadline=time.monotonic()+1800;outer_cancel=threading.Event();phase_cancel=threading.Event()
    handlers={s:signal.signal(s,lambda a,b:(outer_cancel.set(),phase_cancel.set()))for s in (signal.SIGTERM,signal.SIGINT)}
    executor_factory=_fixture_executor or SerialExecutor;result=dict(schema=SCHEMA+'/result',plan=plan_ref,purpose=c['plan']['purpose'],
        GPU_executed=not fixture,database_created=False,partial_recovery_verified=False,human_accuracy_certified=False,steps=[])
    try:
        atomic_new(out/'effective-plan.json',c['plan']);inputs=out/'inputs';inputs.mkdir()
        raw=b''.join(canonical_bytes(s)for s in c['sources']);(inputs/'sources.jsonl').write_bytes(raw)
        index_value=dict(schema='redimnet2-shard-index/v1',source_count=30,shards=[dict(id=SHARD_ID,path='sources.jsonl',count=30,sha256=__import__('hashlib').sha256(raw).hexdigest())])
        index=dict(path=str(inputs/'index.json'),sha256=atomic_new(inputs/'index.json',index_value))
        def stage(root,name,uuids=None):
            root.mkdir(exist_ok=True);v=variant_for(c['variant'],root,index,name,uuids=uuids)
            track_path=out/(root.name+'-'+name+'-track.json');v['track_ref']=dict(path=str(track_path),sha256=atomic_new(track_path,v['track']));return v
        def audit(v):
            from audit_native_transformers_batch import load_track
            verify_parent(v);return load_track(v['track_ref']['path'],v['track_ref']['sha256'])
        verify_controls(c)
        cold=stage(out/'cold','resources-attempt1');executor_factory(out,deadline,outer_cancel).command(cold['argv'],'cold',cold['code']);cold_track=audit(cold)
        parent=pin(dict(path=str(Path(cold['track']['resources_root'])/'parent_result.json'),sha256=cold_track['parent_result_sha256']))
        uuids=parent['UUIDs'];need(len(uuids)==len(set(uuids))==2,'cold run must attest exactly two physical UUIDs')
        result['steps'].append(dict(name='cold',parent_sha256=file_identity(Path(cold['track']['resources_root'])/'parent_result.json')['sha256']))
        need(not outer_cancel.is_set(),'probe cancelled before interruption stage')
        verify_controls(c)
        fault=stage(out/'interrupted','resources-attempt1',uuids);state={};stop=threading.Event()
        watcher=threading.Thread(target=watch_first_packet,args=(c,fault,phase_cancel,outer_cancel,stop,deadline,out,state),daemon=True)
        watcher.start();failure=None
        try:executor_factory(out,deadline,phase_cancel).command(fault['argv'],'interrupted',fault['code'])
        except BaseException:failure=traceback.format_exc()
        finally:stop.set();watcher.join(10)
        need(not watcher.is_alive(),'probe watcher did not stop')
        atomic_new(out/'interrupted-command-result.json',dict(traceback=failure,watcher=state,original_parent_command_returned=True))
        need(failure is not None and state.get('trigger')and not state.get('error')and phase_cancel.is_set(),'expected own controlled mid-cohort interruption not observed')
        need(not outer_cancel.is_set(),'external cancellation forbids recovery')
        need((out/'interrupted-TERM.json').is_file(),'own parent TERM receipt missing; do not assume it was stopped')
        before=snapshot_partial(c,fault);atomic_new(out/'interrupted-durable-snapshot.json',before)
        need(before['committed_packets']and before['uncommitted_cohorts']and not before['all_sources_published'],
             'inconclusive: no genuine committed+uncommitted cohort split; do not claim partial recovery')
        # 单片已完成会污染attempt2终态分母，不能把全命中说成中途恢复。
        need(not(Path(fault['track']['queue_root'])/'completed'/(SHARD_ID+'.json')).exists(),'inconclusive: interrupted shard already complete')
        release=(_fixture_release or prove_release)(fault,uuids);atomic_new(out/'interrupted-resource-return.json',release)
        need(not outer_cancel.is_set()and time.monotonic()<deadline,'probe cancelled/budget exhausted after cleanup')
        verify_controls(c)
        resume=stage(out/'interrupted','resources-attempt2',uuids);executor_factory(out,deadline,outer_cancel).command(resume['argv'],'resumed',resume['code']);resumed=audit(resume)
        verify_preserved(before)
        from audit_native_transformers_batch import analyze
        comparison=analyze(normalized_results(cold_track),normalized_results(resumed),mode='recovery');atomic_new(out/'cold-versus-resumed-native.json',comparison)
        need(comparison['strict_recovery_native_results_equal'],'cold versus resumed native results differ; retain all observations')
        committed_rows={r['row_id']for cohort in expected_plan(c,c['requests'],fixture=fixture)['cohorts']if cohort['cohort_sha256']in before['committed_packets']for r in cohort['intent']['requests']}
        observed_rows={}
        for sid,record in resumed['records'].items():
            for w in record['windows']:
                ref=w.get('asr',{}).get('durable_native_batch')
                if ref is not None:observed_rows[sid+':'+w['window_id']]=ref
        need(committed_rows and all(observed_rows[r]['cache_hit']is True for r in committed_rows),'already durable ASR rows were recomputed or omitted')
        first_plan=expected_plan(c,c['requests'],fixture=fixture)
        for cohort in first_plan['cohorts']:
            if cohort['cohort_sha256']not in before['uncommitted_cohorts']:continue
            for row in cohort['intent']['requests']:
                ref=observed_rows[row['row_id']]
                need(ref['cohort_sha256']==cohort['cohort_sha256']and ref['cache_hit']is False,
                     'uncommitted original cohort was omitted, regrouped or falsely reused')
        totals=resumed['performance']['totals'];need(totals['native_ASR_stage_cache_hits']==len(committed_rows),
             'actual worker cache-hit accounting differs from previously committed rows')
        result.update(partial_recovery_verified=True,committed_ASR_rows_reused=len(committed_rows),all_previous_files_unchanged=True,
                      cold_result_comparison_path=str(out/'cold-versus-resumed-native.json'),formal_admission=False)
        atomic_new(out/'result.json',result);return result
    except BaseException:
        atomic_new(out/'failure.json',dict(traceback=traceback.format_exc(),steps=result['steps'],partial_recovery_verified=False,
            no_old_task_modified=True,no_orphan_deleted=True));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    print(json.dumps(run(dict(path=str(a.plan),sha256=a.plan_sha),a.output,a.execute),ensure_ascii=False))
