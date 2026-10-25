"""Prepare <=16 diagnostic sources and ASR-only seeds; no media/model execution.

Every input is explicitly SHA-bound. The two fresh pilot contracts must differ
only by version and the explicit normalization profile. Existing publication,
queue and cache trees are never reused or changed. Future workers must decode
the source and match its model PCM cache key before consuming any seed.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from contracts import (need, sha_file, canonical_bytes, digest, job_key, atomic_new,
                       read_contract, model_provenance, safe_join)
from normalization import POLICY, APOSTROPHE_PROFILE, normalize_with_audit
from schema import assert_valid_record
from sidecar import SidecarStore, StageCache
from source_clock import TIME_MAPPING
from worker import _stage_inputs

DOMAINS = {'voxceleb2', 'voxblink2', 'cnceleb2'}


def read_bound(path, pin, maximum=8*1024**2):
    path = Path(path)
    need(not path.is_symlink() and path.is_file(), 'physical regular evidence required')
    before = path.stat(); need(before.st_size <= maximum, 'evidence exceeds bounded size')
    raw = path.read_bytes(); after = path.stat()
    snap = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(snap(before) == snap(after) and hashlib.sha256(raw).hexdigest() == pin, 'evidence changed or SHA differs')
    return json.loads(raw), raw


def comparable_contracts(a, b, old):
    need(a['purpose'] == b['purpose'] == 'pilot_validation', 'A/B is diagnostic pilot only')
    need(a['version'] != b['version'], 'A/B need separate versions')
    need(a['runtime']['normalization_profile'] == POLICY and b['runtime']['normalization_profile'] == APOSTROPHE_PROFILE,
         'A must be legacy text policy; B explicit ASCII word-apostrophe profile')
    left,right = deepcopy(a),deepcopy(b)
    for c in (left,right):
        c.pop('version');c['runtime'].pop('normalization_profile')
    need(left == right, 'A/B differ beyond explicit normalization profile/version')
    for c in (a,b):
        need(c['runtime']['boundary_mapping'] == TIME_MAPPING, 'both tracks need identical rational source clock')
        need(c['models'] == old['models'] and c['source_files'] == old['source_files'] and c['decoder'] == old['decoder'],
             'ASR/model/decoder source identities differ from captured producer')
        actual,previous = deepcopy(c['runtime']),deepcopy(old['runtime'])
        for value in (actual,previous):
            for key in ('normalization','normalization_profile','boundary_mapping'):value.pop(key,None)
        need(actual == previous and c['alignment'] == old['alignment'], 'inference/window semantics differ beyond diagnostic policies')


def prepare(*, sample_list, sample_list_sha, sidecar_root, chunk_commit, chunk_commit_sha,
            producer_contract, producer_contract_sha, contract_a, contract_a_sha,
            contract_b, contract_b_sha, output):
    listing,_ = read_bound(sample_list,sample_list_sha,1024**2)
    rows = listing.get('rows'); need(isinstance(rows,list) and 1 <= len(rows) <= 16, 'fixed sample list must contain 1..16 sources')
    need(listing.get('sample_count') == len(rows), 'sample-list denominator differs')
    need(len({r['source_id'] for r in rows}) == len(rows), 'duplicate diagnostic sources')
    receipt,_ = read_bound(chunk_commit,chunk_commit_sha)
    old,_ = read_bound(producer_contract,producer_contract_sha)
    need(old.get('purpose') in {'silver_annotation','pilot_validation'} and old.get('model_API_calls_allowed')==0
         and old.get('release_training_labels') is False, 'unreleased offline producer contract required')
    a=read_contract(contract_a,contract_a_sha); b=read_contract(contract_b,contract_b_sha)
    comparable_contracts(a,b,old)
    need(a.get('parent_contract_sha256')==b.get('parent_contract_sha256')==producer_contract_sha,
         'both diagnostic contracts must explicitly descend from captured producer')
    need(receipt['schema']=='redimnet2-chunk-commit/v1' and receipt['contract_sha256']==producer_contract_sha,
         'chunk belongs to another producer')
    committed={r['source_id']:r for r in receipt['sources']}
    need(len(committed)==receipt['source_count']==len(receipt['sources']), 'chunk source denominator/uniqueness differs')
    root=Path(sidecar_root).absolute(); need(root.is_dir() and not root.is_symlink(),'physical sidecar root required')
    records=[];comparisons=[]
    for i,row in enumerate(rows):
        path=safe_join(root,row.get('sidecar_relative_path','%02d.json'%i))
        record,raw=read_bound(path,row['sidecar_sha256']);assert_valid_record(record)
        source=record['source'];sid=source['source_id'];published=committed.get(sid)
        need(sid==row['source_id'] and source['dataset'] in DOMAINS, 'wrong source or outside three training domains')
        need(source['original_splits']==['provider_train'] and source['roles'] and set(source['roles'])<={'train_ptn','train_lm'},
             'not actual registered training source')
        need(sid=='src_'+hashlib.sha256((source['dataset']+'\0'+source['relative_path']).encode()).hexdigest(), 'source path identity differs')
        need(record['contract']['sha256']==producer_contract_sha and record['contract']['version']==old['version']
             and record['contract']['purpose']==old['purpose'], 'embedded producer version/contract differs')
        need(record['provenance']==record['contract']['provenance'] and record['provenance']['models']==model_provenance(old)
             and record['provenance']['runtime']==old['runtime'] and record['provenance']['alignment_identity']==old['alignment'],
             'captured model/runtime provenance differs')
        need(published is not None and published['sidecar_sha256']==row['sidecar_sha256']
             and published['sidecar_bytes']==len(raw) and published['job_key']==job_key(source,producer_contract_sha)
             ==record['execution']['job_key'], 'source/sidecar/chunk job binding differs')
        need(published['sidecar_relpath']==source['sidecar_relpath'] and published['decision']==record['decision']['state']
             and record['execution'].get('backend')=='qwen3-asr-plus-ultra-no-head', 'producer backend/path/decision differs')
        need(record['decision']['human_review'] is None and record['decision']['usable_for_training'] is False
             and record['decision']['usable_for_evaluation'] is False, 'not an unreleased machine observation')
        for w in record['windows']:
            asr=w['asr'];native=asr.get('native',{});ids=native.get('generated_token_ids')
            need(asr.get('status')=='completed' and native.get('eos_observed') is True and isinstance(ids,list)
                 and ids and all(type(x)is int and x>=0 for x in ids) and native.get('actual_generate_rows')==1
                 and native.get('model_training') is False and native.get('inference_mode') is True
                 and native.get('grad_enabled') is False and isinstance(native.get('native_decoded_raw_answer'),str),
                 'only complete observed single-row ASR can seed diagnostic alignment')
            need(asr['normalization_audit']==normalize_with_audit(asr['raw_text'])
                 and asr['normalized_text']==asr['normalization_audit']['normalized_text'], 'old ASR normalization is not exact legacy profile')
            need(type(w.get('model_input_frames'))is int and w['model_input_frames']>0,'model input length missing')
            comparisons.append({'source_id':sid,'window_id':w['window_id'],'old_sidecar_sha256':row['sidecar_sha256'],
                'ASR_raw_sha256':digest(asr['raw_text']),'ASR_native_sha256':digest(native),
                'A':normalize_with_audit(asr['raw_text'],POLICY),'B':normalize_with_audit(asr['raw_text'],APOSTROPHE_PROFILE)})
        records.append((row,record))
    out=Path(output).absolute();need(not out.exists() and not out.is_symlink(),'output must be fresh, not a queue/cache to append')
    resolved=out.resolve()
    protected=[root.resolve(),Path(__file__).resolve().parent,Path(chunk_commit).resolve(),Path(producer_contract).resolve(),
               Path(contract_a).resolve(),Path(contract_b).resolve(),Path(sample_list).resolve()]
    need(all(resolved!=p and p not in resolved.parents and resolved not in p.parents for p in protected), 'output overlaps code/input evidence')
    need(out.parent.is_dir(),'output parent must exist');out.mkdir()
    source_raw=b''.join(canonical_bytes(r['source'])for _,r in records)
    shard=out/'sources.jsonl';atomic_new(shard,source_raw)
    chunks=[]
    for start in range(0,len(records),8):
        group=records[start:start+8];rel='shards/diagnostic-%06d.jsonl'%(start//8)
        path=safe_join(out,rel);atomic_new(path,b''.join(canonical_bytes(r['source'])for _,r in group))
        chunks.append({'id':'diagnostic-%06d'%(start//8),'path':rel,'count':len(group),'sha256':sha_file(path),
                       'source_ids_sha256':digest(sorted(r['source']['source_id']for _,r in group))})
    index={'schema':'redimnet2-shard-index/v1','source_count':len(records),'purpose':'fixed_diagnostic_normalization_AB_not_training_admission',
           'shards':chunks}
    index_pin=atomic_new(out/'index.json',index);tracks={}
    for name,c,pin in (('A',a,contract_a_sha),('B',b,contract_b_sha)):
        labels=out/('labels-'+name);store=SidecarStore(labels,c,pin);cache=StageCache(store.root/'.stage_cache');seeds=[]
        for row,r in records:
            for w in r['windows']:
                original=w['asr'];result=deepcopy(original);audit=normalize_with_audit(original['raw_text'],c['runtime']['normalization_profile'])
                result.update(normalized_text=audit['normalized_text'],normalization_audit=audit,
                    reused_ASR_evidence={'schema':'diagnostic-ASR-reuse/v1','old_contract_sha256':producer_contract_sha,
                     'old_sidecar_sha256':row['sidecar_sha256'],'source_id':r['source']['source_id'],'window_id':w['window_id'],
                     'original_ASR_output_sha256':digest(original),'original_native_sha256':digest(original['native']),
                     'normalization_rederived_from_unchanged_raw':True,'new_ASR_model_execution':False})
                inputs=_stage_inputs('asr',w,None,c,pin,model_provenance(c))
                winner=cache.put('asr',inputs,result);need(winner==result and winner['native']==original['native'],'seed changed native observation')
                cp=cache.path('asr',digest(inputs));seeds.append({'source_id':r['source']['source_id'],'window_id':w['window_id'],
                    'cache_path':str(cp),'cache_sha256':sha_file(cp),'inputs_sha256':digest(inputs),'output_sha256':digest(result)})
        tracks[name]={'contract_path':str(Path(contract_a if name=='A'else contract_b).resolve()),'contract_sha256':pin,
                      'version':c['version'],'output_root':str(labels),'queue_root':str(out/('queue-'+name)),
                      'resources_must_be_new':True,'ASR_only_seeds':seeds,'aligner_cache_entries':0}
    proof={'schema':'diagnostic-normalization-AB-preparation/v1','index_path':str(out/'index.json'),'index_sha256':index_pin,
           'source_count':len(records),'window_count':len(comparisons),'source_records_unchanged':True,
           'input_pins':{'sample_list':sample_list_sha,'chunk_commit':chunk_commit_sha,'producer_contract':producer_contract_sha},
           'tracks':tracks,'targets':comparisons,'media_read_or_modified':False,'GPU_or_model_execution':False,'API_calls':0,
           'new_ASR_model_execution':False,'training_or_evaluation_admission':False,'human_gold':False,
           'followup':'Worker must freshly decode and match identical model PCM cache keys; inspect each ASR cache hit and original-native digest after the actual GPU alignment. A seed receipt alone does not prove consumption.'}
    atomic_new(out/'preparation.json',proof);return proof


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sample-list','sidecar-root','chunk-commit','producer-contract','contract-a','contract-b','output'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('sample-list','chunk-commit','producer-contract','contract-a','contract-b'):p.add_argument('--'+name+'-sha',required=True)
    a=vars(p.parse_args(argv));result=prepare(**a);print(json.dumps({'source_count':result['source_count'],'window_count':result['window_count'],
        'index_path':result['index_path'],'index_sha256':result['index_sha256'],'GPU_or_model_execution':False},ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
