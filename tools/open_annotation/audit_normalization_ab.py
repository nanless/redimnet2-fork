"""Read <=16 completed diagnostic cases; no queue constructors/media/models.

Copied remote paths are resolved only by explicit local run roots and their
fixed relative protocol layout. Original JSON bytes are not rewritten.
This compares recorded machine observations, never human boundary accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from contracts import canonical_bytes, digest, job_key, model_provenance, need, safe_join, sha_file
from normalization import POLICY, APOSTROPHE_PROFILE, normalize_with_audit
from prepare_normalization_ab import read_bound, comparable_contracts
from quality_axes import native_observation_errors
from readonly_progress import inspect_progress
from schema import assert_valid_record
from source_clock import TIME_MAPPING


def read(path):
    path=Path(path)
    return read_bound(path,sha_file(path),maximum=16*1024**2)[0]


def geometry(alignment):
    """Counts retain raw, FIX and final-return layers and their denominators."""
    out={'items':len(alignment.get('items',[])), 'repair_changed_windows':int(alignment.get('repair_changed') is True)}
    for label,key in (('raw','native_timestamps_ms'),('FIX','fixed_timestamps_ms')):
        v=alignment.get(key,[]);need(isinstance(v,list) and len(v)%2==0,'incomplete timestamp pairs')
        pairs=list(zip(v[::2],v[1::2]));out[label+'_pairs']=len(pairs)
        out[label+'_zero']=sum(b==a for a,b in pairs)
        out[label+'_negative']=sum(b<a for a,b in pairs)
        out[label+'_adjacent_overlap']=sum(pairs[i][0]<pairs[i-1][1] for i in range(1,len(pairs)))
    return out


def target_raw_spans(asr,alignment):
    """Exact whitespace-token case only; never split a coarse native group."""
    audit=asr['normalization_audit'];text=audit['normalized_text']
    tokens=list(re.finditer(r'\S+',text));mapping=audit.get('normalized_character_map')
    if (audit.get('mapping_status')!='exact_codepoint_half_open' or not isinstance(mapping,list)
            or [t.group() for t in tokens]!=alignment['encoded_target_words']):
        return None
    rows=[]
    for i,t in enumerate(tokens):
        covered=mapping[t.start():t.end()]
        rows.append({'native_index':i,'text':t.group(),'raw_start':min(x['raw_start']for x in covered),
                     'raw_end':max(x['raw_end']for x in covered)})
    # One compatibility codepoint can expand to several whitespace words.
    # Its raw interval cannot distinguish their occurrences or boundaries.
    if any(r['raw_end']<=r['raw_start'] for r in rows) or any(
            left['raw_end']>right['raw_start'] for left,right in zip(rows,rows[1:])):
        return None
    return rows


def compare_raw_text_spans(a,b):
    """Compare whole preserved raw-text spans, allowing an explicit A→B merge.

    These are boundary differences, not errors. No time is interpolated across
    subwords, and unsupported/coarse tokenization is left unmeasured.
    """
    aa=target_raw_spans(a['asr'],a['alignment']);bb=target_raw_spans(b['asr'],b['alignment'])
    if aa is None or bb is None:return {'status':'mapping_not_available_no_coarse_group_split','rows':[]}
    result=[]
    for target in bb:
        group=[x for x in aa if target['raw_start']<=x['raw_start'] and x['raw_end']<=target['raw_end']]
        if not group or group[0]['raw_start']!=target['raw_start'] or group[-1]['raw_end']!=target['raw_end']:
            return {'status':'raw_span_group_unresolved','rows':[]}
        first,last=group[0]['native_index'],group[-1]['native_index'];j=target['native_index']
        row={'raw_start':target['raw_start'],'raw_end':target['raw_end'],'B_text':target['text'],
             'A_indices':[x['native_index']for x in group],'B_index':j,'merged_A_fragments':len(group)>1}
        for label,key in (('raw','native_timestamps_ms'),('FIX','fixed_timestamps_ms')):
            va=a['alignment'][key];vb=b['alignment'][key]
            row[label+'_A_span_ms']=[va[2*first],va[2*last+1]]
            row[label+'_B_span_ms']=[vb[2*j],vb[2*j+1]]
            row[label+'_start_delta_ms']=vb[2*j]-va[2*first]
            row[label+'_end_delta_ms']=vb[2*j+1]-va[2*last+1]
        result.append(row)
    return {'status':'exact_raw_codepoint_span_groups','rows':result}


def span_totals(comparisons):
    out=Counter();maximum=0
    for comparison in comparisons:
        if comparison['status']!='exact_raw_codepoint_span_groups':
            out['unmeasured_windows']+=1;continue
        for row in comparison['rows']:
            out['same_raw_text_spans']+=1
            out['merged_A_fragment_groups']+=int(row['merged_A_fragments'])
            out['FIX_span_changed']+=int(row['FIX_start_delta_ms']!=0 or row['FIX_end_delta_ms']!=0)
            maximum=max(maximum,abs(row['FIX_start_delta_ms']),abs(row['FIX_end_delta_ms']))
            for label in ('raw','FIX'):
                for side in 'AB':
                    a,b=row[label+'_'+side+'_span_ms']
                    out[label+'_'+side+'_zero']+=int(a==b);out[label+'_'+side+'_negative']+=int(b<a)
    return dict(out,maximum_absolute_FIX_endpoint_delta_ms=maximum,
                time_differences_are_accuracy_errors=False)


def compare_window(old,a,b,*,source_id,old_sidecar_sha):
    """Refuse target/model provenance drift instead of silently pooling it."""
    need(old['window_id']==a['window_id']==b['window_id'],'window IDs differ')
    for field in ('context_start_sample','context_end_sample','owned_start_sample','owned_end_sample',
                  'input_pcm_sha256','model_input_frames','model_sample_rate','resampling_policy'):
        need(old.get(field)==a.get(field)==b.get(field),'window/audio clock differs: '+field)
    need(a['time_mapping']==b['time_mapping']==TIME_MAPPING,'tracks do not share rational clock')
    observations=[]
    for w,policy in ((a,POLICY),(b,APOSTROPHE_PROFILE)):
        asr,al=w['asr'],w['alignment'];origin=asr.get('reused_ASR_evidence',{})
        need(w.get('asr_cache_hit') is True and w.get('aligner_cache_hit') is False,'not ASR-only reuse plus fresh alignment')
        need(asr.get('status')=='completed' and old['asr'].get('status')=='completed','ASR incomplete')
        for key in ('raw_text','native','language'):
            need(asr[key]==old['asr'][key],'ASR was not held fixed: '+key)
        need(origin.get('source_id')==source_id and origin.get('window_id')==w['window_id']
             and origin.get('old_sidecar_sha256')==old_sidecar_sha
             and origin.get('original_ASR_output_sha256')==digest(old['asr'])
             and origin.get('original_native_sha256')==digest(old['asr']['native'])
             and origin.get('new_ASR_model_execution') is False,'ASR seed origin differs')
        audit=normalize_with_audit(old['asr']['raw_text'],policy)
        need(asr['normalization_audit']==audit and asr['normalized_text']==audit['normalized_text']
             and al['target_text']==audit['normalized_text'],'explicit text condition differs')
        need(al.get('status')=='completed_native_candidates' and not native_observation_errors(al),'native alignment incomplete/inconsistent')
        need(al.get('inference_call',{}).get('rows')==1 and al['inference_call'].get('producer_row_ids')==[source_id+':'+w['window_id']],
             'fresh actual alignment row differs')
        observations.append({'target_text':al['target_text'],'encoded_words':al['encoded_target_words'],
            'native_timestamps_ms':al['native_timestamps_ms'],'fixed_timestamps_ms':al['fixed_timestamps_ms'],
            'author_final_items':al['author_final_items'],'geometry':geometry(al),
            'spoken_form_flags':audit['reasons'],'language':asr['language']})
    aa,bb=observations
    return {'window_id':a['window_id'],'ASR_raw_text':old['asr']['raw_text'],
        'ASR_native_sha256':digest(old['asr']['native']),'input_pcm_sha256':a['input_pcm_sha256'],
        'model_input_frames':a['model_input_frames'],'target_changed':aa['target_text']!=bb['target_text'],
        'A':aa,'B':bb,'raw_equal':aa['native_timestamps_ms']==bb['native_timestamps_ms'],
        'FIX_equal':aa['fixed_timestamps_ms']==bb['fixed_timestamps_ms'],
        'author_items_equal':aa['author_final_items']==bb['author_final_items'],
        'same_raw_text_span_comparison':compare_raw_text_spans(a,b),
        'original_to_A':{'raw_equal':old['alignment']['native_timestamps_ms']==aa['native_timestamps_ms'],
                         'FIX_equal':old['alignment']['fixed_timestamps_ms']==aa['fixed_timestamps_ms'],
                         'author_items_equal':old['alignment']['author_final_items']==aa['author_final_items']}}


def resource_summary(run,track,contract_sha,bundle_sha,index,index_sha,expected_publications,expected_windows):
    root=Path(run)/('resources-'+track+'-attempt1');parent=read(root/'parent_result.json')
    need(parent.get('resource_returned') is True and parent.get('full_sidecar_readback') is True
         and parent.get('contract_sha256')==contract_sha and parent.get('bundle_sha256')==bundle_sha
         and parent.get('index_sha256')==index_sha and parent.get('model_API_calls')==0,'parent result identity/completion differs')
    need(parent['worker_count']==2 and len(parent['workers'])==2 and len(set(parent['UUIDs']))==2,'two workers/UUIDs required')
    workers=[];assigned=set();aggregate=Counter();all_ids=set()
    for part in parent['workers']:
        wid=part['worker_id'];need(wid not in assigned and wid in {'worker-0','worker-1'},'worker ID repeated/unknown');assigned.add(wid)
        directory=root/wid;summary=read(directory/'worker_summary.json');ready=read(directory/'terminal_ready.json');ack=read(directory/'terminal_ack.json')
        att=read(directory/'resource_attestation.json');challenge=read(directory/'context_challenge/attestation.json')
        need(summary==part['terminal']['summary'] and sha_file(directory/'worker_summary.json')==part['terminal']['summary_sha256']
             ==ready['summary_sha256'] and sha_file(directory/'terminal_ready.json')==part['terminal']['ready_sha256']
             ==ack['ready_sha256'],'terminal summary/ACK digest differs')
        need(att['worker_id']==wid==ready['worker_id'] and att['nonce']==ready['nonce']==ack['nonce']
             and att['UUIDs']==[part['uuid']] and att['contract_sha256']==ready['contract_sha256']==contract_sha
             and att['bundle_sha256']==ready['bundle_sha256']==bundle_sha
             and att['index_sha256']==ready['index_sha256']==index_sha
             and sha_file(directory/'resource_attestation.json')==ready['resource_attestation_sha256']
             and sha_file(directory/'context_challenge/attestation.json')==att['challenge_attestation_sha'],'resource chain differs')
        need(challenge['complete'] is True and challenge['all_free_phases_returned_to_baseline'] is True
             and len(challenge['phases'])==7 and challenge['UUIDs']==att['UUIDs']
             and challenge['known_container_worker_pid']==att['known_container_worker_pid']==ready['worker_pid'], 'historical seven-phase attestation differs')
        metrics=summary['backend_metrics'];n=summary['model_windows']
        need(summary['contract_sha256']==contract_sha and summary['index_sha256']==index_sha
             and summary['device_uuid']==part['uuid'] and summary['model_API_calls']==metrics['model_API_calls']==0
             and summary['sidecar_cache_hits']==0 and summary['stage_cache_hits']=={'asr':n,'aligner':0}
             and metrics['asr_rows']==metrics['asr_batches']==0
             and metrics['aligner_rows']==metrics['aligner_batches']==n
             and metrics['aligner_call_rows']=={'1':n},'actual cache/model call accounting differs')
        ids=set()
        for ref in summary['chunk_receipt_refs']:
            path=safe_join(Path(run)/'prepared'/('queue-'+track),'completed/'+ref['id']+'.json')
            chunk,_=read_bound(path,ref['sha256']);need(chunk['contract_sha256']==contract_sha and chunk['index_sha256']==index_sha,'worker chunk identity differs')
            rows=chunk['sources'];need(len(rows)==chunk['source_count']==ref['source_count'],'chunk source denominator differs')
            current={r['source_id'] for r in rows};need(len(current)==len(rows) and not ids&current and not all_ids&current,'source ownership overlap')
            need(all(r==expected_publications.get(r['source_id']) for r in rows),'chunk receipt does not bind reopened source publication')
            ids.update(current)
        all_ids.update(ids);need(len(ids)==summary['sidecars_committed']==summary['assigned_source_rows'],'worker source denominator differs')
        aggregate.update({k:metrics[k] for k in ('aligner_rows','aligner_seconds','asr_rows','asr_seconds')})
        workers.append({'worker_id':wid,'uuid':part['uuid'],'assigned_sources':len(ids),'backend_metrics':metrics,
                        'worker_elapsed_seconds':summary['elapsed_seconds'],'decoded_audio_seconds':summary['decoded_audio_seconds'],
                        'native_calls_verified':True,'historical_resource_digest_chain_verified':True})
    need(all_ids==set(expected_publications) and aggregate['aligner_rows']==expected_windows,'source/window union differs')
    progress=inspect_progress(index,index_sha,Path(run)/'prepared'/('queue-'+track),contract_sha)
    need(progress['committed_source_lower_bound']==len(expected_publications),'queue completion denominator differs')
    returned=read(root/'resource_return_double_probe.json')
    need(len(returned['snapshots'])==2 and returned['interval_seconds']>=5,'resource return double probe missing')
    for snap in returned['snapshots']:
        need(not snap['compute_processes'] and {d['uuid'] for d in snap['devices']}==set(parent['UUIDs'])
             and all(d['used_mib']<=256 for d in snap['devices']),'historical resource return was not idle')
    utilization={u:[] for u in parent['UUIDs']}
    for line in (root/'GPU_utilization.jsonl').read_text().splitlines():
        sample=json.loads(line);need(sample['returncode']==0,'GPU utilization sampling failed')
        for u,v in sample['utilization_percent'].items():
            need(u in utilization and type(v)is int and 0<=v<=100,'invalid GPU utilization sample');utilization[u].append(v)
    return {'parent_sha256':sha_file(root/'parent_result.json'),'decision_counts':progress['decision_counts'],
        'parent_elapsed_seconds':parent['parent_elapsed_seconds'],'gpu_lease_wall_seconds':parent['gpu_lease_wall_seconds'],
        'model_guard_seconds':parent['model_guard_seconds'],'aggregate':dict(aggregate),'workers':workers,
        'GPU_utilization_percent':{u:{'samples':len(v),'min':min(v),'max':max(v),'mean':sum(v)/len(v)}for u,v in utilization.items()},
        'resource_returned_in_saved_double_probe':True,'current_GPU_availability_checked':False,'model_API_calls':0}


def analyze(*,original_root,sample_list,preparation,preparation_sha,contract_a,contract_a_sha,
            contract_b,contract_b_sha,bundle_sha,run_a,run_b):
    prep,_=read_bound(preparation,preparation_sha);listing,_=read_bound(sample_list,prep['input_pins']['sample_list'])
    rows=listing['rows'];need(1<=len(rows)<=16 and len(rows)==listing['sample_count']==prep['source_count'],'diagnostic denominator differs')
    old_contract,_=read_bound(Path(original_root)/'contract.json',prep['input_pins']['producer_contract'])
    ca,_=read_bound(contract_a,contract_a_sha);cb,_=read_bound(contract_b,contract_b_sha);comparable_contracts(ca,cb,old_contract)
    index=Path(preparation).parent/'index.json';need(sha_file(index)==prep['index_sha256'],'fixed diagnostic index differs')
    results=[];totals={'A':Counter(),'B':Counter()};decisions={'A':Counter(),'B':Counter()};window_count=0
    publications={'A':{},'B':{}}
    for i,row in enumerate(rows):
        old,old_raw=read_bound(safe_join(Path(original_root)/'sidecars',row.get('sidecar_relative_path','%02d.json'%i)),row['sidecar_sha256'])
        assert_valid_record(old);sid=old['source']['source_id'];need(sid==row['source_id'],'old source identity differs')
        actual=[]
        for track,c,pin,run in (('A',ca,contract_a_sha,run_a),('B',cb,contract_b_sha,run_b)):
            path=safe_join(Path(run)/'prepared'/('labels-'+track)/c['version'],old['source']['sidecar_relpath']);r=read(path);assert_valid_record(r)
            commit=read(path.with_name(path.name+'.commit.json'))
            need(commit=={'schema':'redimnet2-sidecar-commit/v1','source_id':sid,'job_key':job_key(old['source'],pin),
                'contract_sha256':pin,'source_record_sha256':digest(old['source']),'sidecar_sha256':sha_file(path),'bytes':path.stat().st_size},'new sidecar commit identity differs')
            publications[track][sid]={'source_id':sid,'job_key':job_key(old['source'],pin),
                'sidecar_relpath':old['source']['sidecar_relpath'],'sidecar_sha256':sha_file(path),
                'sidecar_bytes':path.stat().st_size,'commit_sha256':sha_file(path.with_name(path.name+'.commit.json')),
                'decision':r['decision']['state']}
            need(r['source']==old['source'] and r['contract']['sha256']==pin and r['contract']['version']==c['version']
                 and r['provenance']['code_sha256']==bundle_sha and r['provenance']['models']==model_provenance(c)
                 and r['provenance']['runtime']==c['runtime'],'sidecar source/contract/model provenance differs')
            for key in ('file_sha256','pcm_sha256','sample_rate','channels','frames','bytes'):
                need(r['source_identity'][key]==old['source_identity'][key],'source content/clock differs: '+key)
            need(r['decision']['usable_for_training'] is False and r['decision']['usable_for_evaluation'] is False
                 and r['decision']['human_review'] is None,'machine eligibility was promoted')
            decisions[track][r['decision']['state']]+=1;actual.append(r)
        aa,bb=actual;need(len(old['windows'])==len(aa['windows'])==len(bb['windows']),'window coverage differs')
        windows=[]
        for wo,wa,wb in zip(old['windows'],aa['windows'],bb['windows']):
            need(wa['asr']['reused_ASR_evidence']['old_contract_sha256']==wb['asr']['reused_ASR_evidence']['old_contract_sha256']
                 ==old['contract']['sha256'],'seed producer contract differs')
            item=compare_window(wo,wa,wb,source_id=sid,old_sidecar_sha=row['sidecar_sha256']);windows.append(item);window_count+=1
            for track in 'AB':totals[track].update(item[track]['geometry'])
        results.append({'ordinal':i,'source_id':sid,'dataset':old['source']['dataset'],
            'audio_seconds':old['source_identity']['frames']/old['source_identity']['sample_rate'],
            'A_decision':aa['decision'],'B_decision':bb['decision'],'A_errors':aa['errors'],'B_errors':bb['errors'],
            'A_units':aa['units'],'B_units':bb['units'],'windows':windows,
            'target_changed':any(w['target_changed']for w in windows),
            'A_source_clock':aa['quality_axes']['source_clock'],'B_source_clock':bb['quality_axes']['source_clock']})
    need(len({r['source_id']for r in results})==len(rows),'repeated source IDs')
    resources={t:resource_summary(run,t,pin,bundle_sha,index,prep['index_sha256'],publications[t],window_count)
               for t,pin,run in (('A',contract_a_sha,run_a),('B',contract_b_sha,run_b))}
    decisions={t:{k:decisions[t][k] for k in ('candidate','unknown','failed')}for t in 'AB'}
    for t in 'AB':need(resources[t]['decision_counts']==decisions[t],'resource/sidecar states differ')
    return {'schema':'annotation-normalization-AB-audit/v1','passed_for_recorded_machine_comparison':True,
        'source_count':len(rows),'window_count':window_count,'audio_seconds':sum(r['audio_seconds']for r in results),
        'target_changed_source_count':sum(r['target_changed']for r in results),'geometry_totals':{t:dict(v)for t,v in totals.items()},
        'decision_counts':{t:dict(v)for t,v in decisions.items()},'resources':resources,'rows':results,
        'same_raw_text_span_totals':span_totals(w['same_raw_text_span_comparison']for r in results for w in r['windows']),
        'source_clock_mapping':TIME_MAPPING,'analyzer_model_GPU_API_execution':False,'human_gold':False,
        'label_eligibility_released':False,'raw_logits_tensors_recomputed':False,
        'limits':['Fixed biased prefix; no CN-Celeb2 case or random accuracy estimate.',
                  'Fewer target fragments change denominators; geometric counts are not acoustic accuracy.',
                  'Preserved ASR is held fixed, not an independent transcript truth.',
                  'Historical resource records checked, not current GPU availability or fresh model reexecution.',
                  'Aligner-only diagnostic with ASR cache; not full ASR+FA production throughput or ETA.']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original-root','sample-list','preparation','contract-a','contract-b','run-a','run-b','output'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('preparation-sha','contract-a-sha','contract-b-sha','bundle-sha'):p.add_argument('--'+name,required=True)
    args=vars(p.parse_args(argv));out=args.pop('output')
    need(not out.exists() and not out.is_symlink() and out.parent.is_dir(),'report must be fresh in existing directory')
    target=out.resolve()
    for key in ('original_root','run_a','run_b'):
        root=args[key].resolve();need(target!=root and root not in target.parents,'report overlaps input tree')
    result=analyze(**args);result['analyzer_sha256']=sha_file(__file__)
    with out.open('xb')as stream:stream.write(canonical_bytes(result))
    print(json.dumps({'source_count':result['source_count'],'target_changed_source_count':result['target_changed_source_count'],
        'geometry_totals':result['geometry_totals'],'decision_counts':result['decision_counts'],'report':str(out)},ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
