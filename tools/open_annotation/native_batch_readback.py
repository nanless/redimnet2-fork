"""固定有限native读回收据，重核整批/计划/行映射；不重开全部侧车或原音。"""
from __future__ import annotations
import copy
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
if __package__:
    from . import bounded_speed_handoff as H
    from . import resume_existing_annotation as R
else:
    import bounded_speed_handoff as H
    import resume_existing_annotation as R
need=H.need


SEALED_SCHEMA='annotation-native-finite-admission-readback/v2'


def _sparse_decision_counts(records):
    """沿冻结load_track的Counter表达，只出现真实状态；不补零、不丢非法状态。"""
    need(isinstance(records,dict)and records and all(isinstance(r,dict)and isinstance(r.get('decision'),dict)for r in records.values()),
         'complete native decision records required')
    states=[record['decision'].get('state')for record in records.values()]
    need(all(type(state)is str and state in {'candidate','unknown','failed'}for state in states),'invalid native source decision state')
    return dict(Counter(states))


def seal_admission_readback(track):
    """有限auditor已实际读回之后，仅用内存封完整观察；不再开侧车或媒体。"""
    if __package__:from . import audit_native_transformers_batch as A
    else:import audit_native_transformers_batch as A
    value=A.native_readback_receipt(track)
    need(0<value['source_count']<=4096,'only finite native pilot observations may be sealed')
    for source_ref in value['sources']:
        record=track['records'][source_ref['source_id']]
        # 包含schema及未来未知顶层字段；原侧车canonical字节要能完整重建。
        view=copy.deepcopy(record)
        view['windows']=[]
        for window in record.get('windows',[]):
            saved=copy.deepcopy(window)
            if window.get('asr',{}).get('durable_native_batch')is not None:saved.pop('asr')
            view['windows'].append(saved)
        source_ref['record_view']=view;source_ref['record_view_sha256']=H.P.digest(view)
        source_ref['source_identity_sha256']=H.P.digest(record['source_identity'])
        source_ref['source_input_identity']={key:copy.deepcopy(record['source_identity'].get(key))for key in A.A.SOURCE_INPUTS}
        source_ref['complete_source_observations_sha256']=H.P.digest({key:record.get(key)for key in ('errors','units','decision','quality_axes','vad')})
    if isinstance(track.get('receipts'),dict):
        receipts=[dict(id=sid,path=str(Path(track['spec']['queue_root'])/'completed'/(sid+'.json')),sha256=item['sha256'])for sid,item in sorted(track['receipts'].items())]
        value['additional_small_control_reads_for_seal']=0
    else:
        # 冻结auditor当前不返回receipts；只从已核父summary补取小控制引用，不重开侧车。
        parent,_=H.read_pin(value['parent_result']);receipts=[];seen=set()
        for worker in parent.get('workers',[]):
            for ref in worker['terminal']['summary']['chunk_receipt_refs']:
                sid=ref['id'];need(sid not in seen,'duplicate native parent chunk reference');seen.add(sid)
                path=str(Path(track['spec']['queue_root'])/'completed'/(sid+'.json'))
                if 'path'in ref:need(ref['path']==path,'native parent chunk path differs')
                receipts.append(dict(id=sid,path=path,sha256=ref['sha256']))
        value['additional_small_control_reads_for_seal']=1
    value['publication_receipts']=sorted(receipts,key=lambda v:v['id'])
    value['schema']=SEALED_SCHEMA;value['comparison_observations_bound_to_original_sidecars']=True
    value['finite_sidecar_readback_already_completed']=True;value['formal_runtime_admitted']=False
    need(len(H.P.canonical(value))<=256*1024**2,'sealed finite native observations exceed 256MiB; never truncate')
    return value


def verify_readback_receipt(pin,track_pin,track,contract,bundle_sha,count,purpose):
    if __package__:from . import audit_native_transformers_batch as A
    else:import audit_native_transformers_batch as A
    proof,_=H.read_pin(pin,256*1024**2)
    need(proof.get('schema')==SEALED_SCHEMA and proof.get('comparison_observations_bound_to_original_sidecars')is True and proof.get('purpose')==contract['purpose']and
        proof.get('track_spec')==track_pin and proof.get('track_spec_sha256')==track_pin['sha256']and proof.get('contract_sha256')==track['contract']['sha256']and
        proof.get('index_sha256')==track['index']['sha256']and proof.get('bundle_sha256')==bundle_sha and proof.get('source_count')==count and
        proof.get('full_sidecar_commit_readback_verified')is True and proof.get('native_whole_batch_readback_verified')is True and proof.get('database_created')is False and
        proof.get('model_API_calls')==0 and proof.get('training_labels_released')is False,'sealed native v2 observations required; old unbound readback v1 cannot admit production')
    need(proof.get('parent_result',{}).get('path')==str(Path(track['resources_root'])/'parent_result.json'),'native readback parent path differs')
    R.pinned_file(proof['parent_result'])
    index,_=H.read_pin(track['index']);need(index.get('source_count')==count and 0<count<=4096 and isinstance(index.get('shards'),list)and
        0<len(index['shards'])<=count and sum(s['count']for s in index['shards'])==count,'finite native readback input denominator differs')
    originals={};pins=[proof['parent_result'],track['index']]
    for shard in index['shards']:
        rows=H.P.read_shard(Path(track['index']['path']).parent,shard)
        for source in rows:
            need(source['source_id']not in originals,'duplicate finite source input');originals[source['source_id']]=source
        pins.append(dict(path=str(H.P.child(Path(track['index']['path']).parent,shard['path'])),sha256=shard['sha256']))
    sources=proof.get('sources');need(isinstance(sources,list)and len(sources)==count and {s.get('source_id')for s in sources}==set(originals),'native readback omitted/duplicated source rows')
    tree=Path(track['output_root'])/contract['version'];packets=proof.get('native_packets');need(isinstance(packets,list)and proof.get('native_packet_count')==len(packets),'complete native packet reference set required')
    packet_by_key={};content_by_key={};cache={}
    for item in packets:
        pp=item['packet'];key=(pp['path'],pp['sha256']);need(key not in packet_by_key,'duplicated native packet evidence')
        packet,_=H.read_pin(pp,64*1024**2);content=packet.get('content',{});need(packet.get('schema')=='annotation-native-transformers-atomic-result/v1'and
            packet.get('content_sha256')==H.P.digest(content)and item.get('cohort_sha256')==content.get('cohort',{}).get('cohort_sha256')and
            item.get('store_identity_sha256')==H.P.digest(content.get('store_identity'))and item.get('requests')==content.get('cohort',{}).get('intent',{}).get('requests')and
            item.get('plan',{}).get('path')==str(Path(pp['path']).parent.parent/'plan.json')and item['plan']['sha256']==content.get('plan_sha256'),
            'native whole packet/plan/cohort row reference differs')
        R.pinned_file(item['plan']);pins.extend([pp,item['plan']]);packet_by_key[key]=item;content_by_key[key]=content
    seen_packets=set();source_map={};row_map={};records={}
    for source_ref in sources:
        sid=source_ref['source_id'];source=originals[sid];need(source_ref.get('source_record_sha256')==H.P.digest(source),'native readback source record changed')
        sidecar=H.P.child(tree,source['sidecar_relpath']);need(source_ref.get('sidecar',{}).get('path')==str(sidecar)and
            source_ref.get('commit',{}).get('path')==str(sidecar)+'.commit.json'and H.is_sha(source_ref['sidecar'].get('sha256'))and
            H.is_sha(source_ref['commit'].get('sha256'))and H.is_sha(source_ref.get('errors_sha256')),'native original sidecar/commit refs missing')
        view=copy.deepcopy(source_ref.get('record_view'))
        need(isinstance(view,dict)and H.P.digest(view)==source_ref.get('record_view_sha256')and view.get('source')==source and
            H.P.digest(view.get('source_identity'))==source_ref.get('source_identity_sha256')and
            {key:view['source_identity'].get(key)for key in A.A.SOURCE_INPUTS}==source_ref.get('source_input_identity')and
            H.P.digest({key:view.get(key)for key in ('errors','units','decision','quality_axes','vad')})==source_ref.get('complete_source_observations_sha256'),
            'sealed native source identity/word/quality/VAD observations differ')
        records[sid]=view
        windows=source_ref.get('windows');need(isinstance(windows,list)and len({w.get('window_id')for w in windows})==len(windows),'native window map duplicated')
        source_map[sid]=source_ref
        actual_windows={w['window_id']:w for w in view.get('windows',[])}
        need(len(actual_windows)==len(windows)and set(actual_windows)=={w['window_id']for w in windows},'sealed native window geometry denominator differs')
        for w in windows:
            request=w.get('request');ref=w.get('durable_native_batch');wid=w.get('window_id');row_id=sid+':'+str(wid)
            need(row_id not in row_map,'native row map duplicated')
            if request is None:
                need(actual_windows[wid].get('asr',{}).get('status')=='failed','native failed window observations missing')
                need(ref is None and w.get('preprocessing_status')!='completed','completed native window cannot omit whole-packet reference')
                row_map[row_id]=dict(request=None,ref=None);continue
            need(isinstance(ref,dict)and set(ref)=={'plan_sha256','cohort_sha256','packet_path','packet_sha256','position','cache_hit'}and
                request.get('row_id')==row_id and request.get('source_id')==sid and request.get('window_id')==wid and
                request.get('source_record_sha256')==H.P.digest(source),'native source/window request map differs')
            key=(ref['packet_path'],ref['packet_sha256']);need(key in packet_by_key,'window references undeclared native packet')
            item=packet_by_key[key];need(item['plan']['sha256']==ref['plan_sha256']and item['cohort_sha256']==ref['cohort_sha256'],'native row plan/cohort mismatch')
            content=content_by_key[key];output=content['outputs'].get(row_id);need(output is not None,'whole packet omitted required native output row')
            asr=copy.deepcopy(output);asr['durable_native_batch']=ref
            need('asr'not in actual_windows[wid],'sealed native successful ASR must be taken from actual whole packet')
            actual_windows[wid]['asr']=asr
            need(all(actual_windows[wid].get(key)==request[key]for key in ('input_pcm_sha256','model_input_frames','model_sample_rate')),
                 'native source-clock geometry does not bind actual PCM packet request')
            window=dict(window_id=wid,input_pcm_sha256=request['input_pcm_sha256'],model_input_frames=request['model_input_frames'],model_sample_rate=request['model_sample_rate'],asr=asr)
            # 原auditor逐包核完整cohort：恢复子集允许包内有512范围之外的旧成员。
            A.verify_native_window(dict(source=source),window,contract,track['contract']['sha256'],bundle_sha,tree,cache)
            seen_packets.add(key);row_map[row_id]=dict(request=request,ref=ref,output=output)
    need(seen_packets==set(packet_by_key),'unreferenced native packet evidence cannot silently expand input scope')
    # SidecarStore.publish固定canonical_bytes；重建后直接认证原侧车及commit字节，不信自签view摘要。
    for sid,record in records.items():
        ref=source_map[sid];raw=H.P.canonical(record)
        need(H.P.byte_sha(raw)==ref['sidecar']['sha256'],'reconstructed native geometry/source observations do not match original committed sidecar bytes')
        marker=dict(schema='redimnet2-sidecar-commit/v1',source_id=sid,job_key=H.P.old_job_key(originals[sid],track['contract']['sha256']),
            contract_sha256=track['contract']['sha256'],source_record_sha256=H.P.digest(originals[sid]),sidecar_sha256=ref['sidecar']['sha256'],bytes=len(raw))
        need(H.P.digest(marker)==ref['commit']['sha256'],'reconstructed native sidecar commit bytes differ')
    need(H.P.digest(records)==proof.get('records_sha256'),'native complete original record aggregate digest differs')
    publication=proof.get('publication_receipts');need(isinstance(publication,list)and {v.get('id')for v in publication}=={v['id']for v in index['shards']}and len(publication)==len(index['shards']),'native actual publication receipt set incomplete')
    expected={};assigned=set()
    for ref in publication:
        path=str(Path(track['queue_root'])/'completed'/(ref['id']+'.json'));need(ref['path']==path,'native publication receipt path differs')
        rpin={k:ref[k]for k in ('path','sha256')};chunk,_=H.read_pin(rpin);pins.append(rpin)
        need(chunk.get('contract_sha256')==track['contract']['sha256']and chunk.get('index_sha256')==track['index']['sha256']and
            chunk.get('shard_id')==ref['id']and len(chunk.get('sources',[]))==chunk.get('source_count'),'native actual chunk identity differs')
        fresh=set()
        for item in chunk['sources']:
            sid=item['source_id'];need(sid in source_map and sid not in assigned,'native actual publication duplicated/foreign source');assigned.add(sid)
            need(item.get('sidecar_sha256')==source_map[sid]['sidecar']['sha256']and item.get('commit_sha256')==source_map[sid]['commit']['sha256']and
                item.get('job_key')==H.P.old_job_key(originals[sid],track['contract']['sha256']),'native source ledger is not original committed sidecar')
            if records[sid]['execution'].get('attempt_id')==chunk['attempt_id']:fresh.add(sid)
        expected[ref['id']]=dict(sha256=ref['sha256'],value=chunk,fresh_source_ids=fresh)
    need(assigned==set(originals),'native actual publication source union incomplete')
    parent,_=H.read_pin(proof['parent_result']);actual_perf=A.summarize_workers(parent,Path(track['resources_root']),expected,contract)
    actual_perf['native_whole_batch_readback_verified']=True
    need(actual_perf==proof.get('performance'),'native performance/dispatch/rows must match actual parent and worker terminal summaries')
    for worker in parent['workers']:
        wid=worker['worker_id'];base=Path(track['resources_root'])/wid
        for name in ('worker_summary.json','terminal_ready.json','terminal_ack.json'):
            path=base/name;pins.append(dict(path=str(path),sha256=R.file_sha(path)))
        if purpose=='production':
            for name in ('resource_attestation.json','context_challenge/attestation.json'):
                path=base/name;pins.append(dict(path=str(path),sha256=R.file_sha(path)))
    return dict(pins=pins,sources=source_map,rows=row_map,proof=proof,records=records,performance=actual_perf,
                track=dict(contract=contract,contract_sha256=track['contract']['sha256'],spec_sha256=track_pin['sha256'],records=records,performance=actual_perf,
                           decision_counts=_sparse_decision_counts(records)))


def verify_comparison_maps(comparison,proofs):
    """从封存的真侧车view及真实终态指标重算整份有限比较，不信摘要布尔。"""
    if __package__:from . import audit_native_transformers_batch as A
    else:import audit_native_transformers_batch as A
    mode=comparison.get('mode');need(mode in {'speed','recovery'},'explicit native comparison mode required')
    actual=A.analyze(proofs['baseline']['track'],proofs['candidate']['track'],mode=mode)
    need(comparison.get('rows')==actual['rows'],'native complete transcript/boundary/source-clock/units/VAD comparison not from sealed actual sidecars')
    for key in ('counts','source_count','baseline_total_sources','candidate_total_sources','decoded_audio_identity_comparable_sources',
                'maximum_same_target_endpoint_delta_ms','baseline_decision_counts','candidate_decision_counts',
                'baseline_contract_sha256','candidate_contract_sha256','baseline_spec_sha256','candidate_spec_sha256'):
        # 仅decision聚合要求字节级JSON类型相等；不把零键删掉，也不接受True/浮点充当计数。
        equal=H.P.canonical(comparison.get(key))==H.P.canonical(actual[key])if key in ('baseline_decision_counts','candidate_decision_counts')else comparison.get(key)==actual[key]
        need(equal,'native complete comparison aggregate differs from actual sidecars: '+key)
    pa,pb=(proofs[name]['performance']for name in ('baseline','candidate'))
    need(comparison['baseline_performance']==pa and comparison['candidate_performance']==pb,'native compared throughput not from actual parent summaries')
    if mode=='speed':
        need(comparison.get('cold_end_to_end_speed_ratio')==pa['parent_wall_seconds']/pb['parent_wall_seconds'],'native speed ratio does not equal actual parent elapsed times')
        if proofs['baseline']['track']['contract']['purpose']!='cpu_fixture':need(comparison.get('cold_speed_ratio_valid')is actual['cold_speed_ratio_valid'],'native cold-speed claim contradicts actual model/packet evidence')
    else:need(comparison.get('strict_recovery_native_results_equal')is actual['strict_recovery_native_results_equal']and
              comparison.get('strict_native_results_equal')is actual['strict_native_results_equal'],'native complete recovery claim differs from actual original sidecars')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--track',required=True);p.add_argument('--track-sha',required=True)
    p.add_argument('--audit-code-root',required=True);p.add_argument('--audit-bundle-sha',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    code=R.physical(a.audit_code_root);R.frozen_bundle(dict(code_root=str(code),bundle=dict(path=str(code/'bundle_files.json'),sha256=a.audit_bundle_sha)))
    sys.path.insert(0,str(code))
    import audit_native_transformers_batch as auditor
    need(Path(auditor.__file__).resolve()==code/'audit_native_transformers_batch.py','seal must use actual frozen native auditor; do not mix working source')
    track=auditor.load_track(a.track,a.track_sha)
    if track['contract']['purpose']!='cpu_fixture':need(auditor.native_readback_receipt(track)['bundle_sha256']==a.audit_bundle_sha,'seal auditor must be exact actual pilot executable bundle')
    value=seal_admission_readback(track)
    output=R.physical(a.output);need(output.parent.is_dir()and not output.exists(),'fresh explicit sealed readback output required')
    need(all(output!=Path(track['spec'][k])and Path(track['spec'][k])not in output.parents for k in ('queue_root','output_root','resources_root')),'seal output must be outside actual producer trees')
    H.P.write_new(output,value)
    print(json.dumps(dict(output=str(output),source_count=value['source_count'],finite_sidecar_readback_once=True,full_library_audit=False,GPU_executed=False,formal_runtime_admitted=False),ensure_ascii=False))

if __name__=='__main__':main()
