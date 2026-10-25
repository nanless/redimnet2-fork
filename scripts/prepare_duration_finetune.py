#!/usr/bin/env python3
"""Bounded engineering selection, never rewrite original audio/RTTM/tables."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from duration_ft.data import inspect_record,verify_record,seal_dataset,DataError,file_sha,validate_output_path

def prepare(manifest,rttm_root,output,train_speakers=24,heldout_speakers=4,utterances=8,
            max_lines=2000000,max_checked_per_corpus=20000,candidate_speakers_per_corpus=96):
    limits=(train_speakers,heldout_speakers,utterances,max_lines,max_checked_per_corpus,candidate_speakers_per_corpus)
    if any(type(value) is not int or value < 1 for value in limits):
        raise ValueError('all selection limits must be positive integers, not booleans')
    if candidate_speakers_per_corpus<train_speakers+heldout_speakers:
        raise ValueError('candidate speaker bound smaller than required speakers')
    root=validate_output_path(output,Path(__file__).resolve().parents[1]);root.mkdir(parents=True,exist_ok=False)
    pools={c:defaultdict(list) for c in 'ABC'};candidates={c:set() for c in 'ABC'};counts={c:Counter() for c in 'ABC'}
    excluded=Counter();examples=[];seen=set();scanned=0;complete=False
    with open(manifest) as stream:
        for scanned,line in enumerate(stream,1):
            if scanned>max_lines:scanned-=1;break
            try:obj=json.loads(line)
            except Exception as exc:
                excluded['bad_json']+=1
                if len(examples)<100:examples.append({'line':scanned,'reason':'bad_json','detail':str(exc)})
                continue
            if not isinstance(obj,dict) or any(not isinstance(obj.get(key),str) or not obj[key] for key in ('key','spk','wav')):
                excluded['bad_record']+=1
                if len(examples)<100:examples.append({'line':scanned,'reason':'bad_record','detail':'nonempty string key/spk/wav required'})
                continue
            corpus=obj.get('key','').split('_',1)[0]
            if corpus not in pools:excluded['unknown_corpus']+=1;continue
            counts[corpus]['lines_seen']+=1
            if counts[corpus]['checked']>=max_checked_per_corpus:
                counts[corpus]['not_checked_bound']+=1;continue
            speaker=obj.get('spk','')
            if not speaker.startswith(corpus+'_'):
                excluded['speaker_prefix_mismatch']+=1;continue
            if speaker not in candidates[corpus]:
                if len(candidates[corpus])>=candidate_speakers_per_corpus:
                    counts[corpus]['not_checked_speaker_bound']+=1;continue
                candidates[corpus].add(speaker)
            if len(pools[corpus].get(speaker,[]))>=utterances:continue
            if obj['key'] in seen:excluded['duplicate_key']+=1;continue
            seen.add(obj['key']);counts[corpus]['checked']+=1
            try:
                record=inspect_record(obj,rttm_root)
                verify_record(record)
            except (DataError,OSError) as exc:
                reason=exc.reason if isinstance(exc,DataError) else 'io_error'
                excluded[corpus+':'+reason]+=1
                if len(examples)<100:examples.append({'line':scanned,'key':obj.get('key'),'reason':reason,'detail':str(exc)})
                continue
            pools[corpus][speaker].append(record);counts[corpus]['accepted_candidates']+=1
            sufficient={c:sum(len(v)>=utterances for v in pools[c].values()) for c in pools}
            if all(v>=train_speakers+heldout_speakers for v in sufficient.values()):complete=True;break
    selection={'kind':'bounded-first-eligible-engineering-only','manifest':str(Path(manifest).resolve()),
               'manifest_stat':{'bytes':Path(manifest).stat().st_size,'mtime_ns':Path(manifest).stat().st_mtime_ns},
               'scan_max_lines':max_lines,'checked_max_per_corpus':max_checked_per_corpus,
               'candidate_speakers_max_per_corpus':candidate_speakers_per_corpus,'candidate_speakers':{c:sorted(v) for c,v in candidates.items()},
               'scanned_lines':scanned,'counts_by_corpus':{c:dict(v) for c,v in counts.items()},
               'excluded':dict(excluded),'excluded_examples':examples,'requested_train_speakers_per_corpus':train_speakers,
               'requested_heldout_speakers_per_corpus':heldout_speakers,'utterances_per_speaker':utterances,
               'selection_complete':complete,'bias':'bounded first eligible speakers/utterances; not a population representative or formal quality dataset'}
    (root/'selection_report.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    if not complete:raise ValueError('bounded selection incomplete; see selection_report.json, increase explicit bounds or supply an engineering subset table')
    train=[];held=[]
    for corpus in 'ABC':
        # Encounter order is frozen; no optimizing this choice by evaluation quality.
        speakers=[s for s,v in pools[corpus].items() if len(v)>=utterances][:train_speakers+heldout_speakers]
        for i,speaker in enumerate(speakers):
            split='train' if i<train_speakers else 'heldout'
            target=train if split=='train' else held
            for record in pools[corpus][speaker][:utterances]:
                record=dict(record,split=split);target.append(record)
    for name,records in [('dataset',train),('heldout',held)]:
        speaker_map={s:i for i,s in enumerate(sorted({r['speaker'] for r in records}))}
        dataset=seal_dataset(records,speaker_map,dict(selection,split='train' if name=='dataset' else 'heldout'))
        (root/(name+'.json')).write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+'\n')
    return {'train_records':len(train),'heldout_records':len(held),'selection':selection}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--rttm-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-speakers',type=int,default=24);p.add_argument('--heldout-speakers',type=int,default=4)
    p.add_argument('--utterances',type=int,default=8);p.add_argument('--max-lines',type=int,default=2000000)
    p.add_argument('--max-checked-per-corpus',type=int,default=20000)
    p.add_argument('--candidate-speakers-per-corpus',type=int,default=96)
    args=p.parse_args();print(json.dumps(prepare(**vars(args)),ensure_ascii=False))
