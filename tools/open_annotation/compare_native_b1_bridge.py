"""只读最多64来源，核旧FP32/SDPA单窗与新版native B1的数学结果桥接。

不加载模型、不读取音频、不计算跨机器速度比、不改变任何标注资格。
可经SSH标准输入执行本仓库这份原字节；结果只输出到调用者标准输出。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from collections import Counter


def need(value,message):
    if not value:raise ValueError(message)


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def json_value(raw):
    def pairs(items):
        value={}
        for key,item in items:need(key not in value,'duplicate JSON field');value[key]=item
        return value
    def constant(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def read(path,sha=None,*,maximum=32*1024**2):
    path=Path(path);need(path.is_absolute()and '..'not in path.parts and all(not p.is_symlink()for p in (path,*path.parents)),
                           'absolute physical evidence path required')
    before=path.lstat();need(stat.S_ISREG(before.st_mode)and before.st_size<=maximum,'bounded regular evidence required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    stamp=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    try:
        actual=os.fstat(fd);need(stamp(before)==stamp(actual),'evidence changed before read')
        with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(maximum+1)
        need(len(raw)==actual.st_size and stamp(actual)==stamp(os.fstat(fd))==stamp(path.stat()),'evidence changed during read')
    finally:os.close(fd)
    digest=hashlib.sha256(raw).hexdigest();need(sha is None or digest==sha,'evidence SHA differs')
    return raw,digest


def pin(ref):
    need(isinstance(ref,dict)and set(ref)=={'path','sha256'}and isinstance(ref['sha256'],str)and len(ref['sha256'])==64 and
         all(c in '0123456789abcdef'for c in ref['sha256']),'explicit evidence pin required')
    raw,digest=read(ref['path'],ref['sha256']);return json_value(raw),digest


def child(root,relative):
    p=Path(relative);need(not p.is_absolute()and p.parts and '..'not in p.parts,'safe child evidence path required')
    return Path(root)/p


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def completion_slots(raw,index,contract_sha,index_sha):
    """只解固定试验清单的完成槽，不读取语料库或任何媒体。"""
    need(len(raw)==1024*len(index['shards']),'finite completion map shape differs')
    result={}
    for i,shard in enumerate(index['shards']):
        block=raw[i*1024:(i+1)*1024];size=struct.unpack('>I',block[8:12])[0]
        need(block[:8]==b'RDMAP001'and 0<size<=980,'torn completion slot')
        body=block[44:44+size];need(hashlib.sha256(body).digest()==block[12:44],'completion checksum differs')
        slot=json_value(body)
        for key,expected in dict(shard_id=shard['id'],shard_sha256=shard['sha256'],source_count=shard['count'],
                                 contract_sha256=contract_sha,index_sha256=index_sha).items():
            need(slot.get(key)==expected,'completion identity differs: '+key)
        need(shard['id']not in result,'duplicate bridge shard');result[shard['id']]=slot
    return result


def parent_publications(parent,resources,queue,contract_sha,index_sha):
    """父终态→worker摘要/ready/ACK→具体片段收据SHA，拒绝只看成功布尔值。"""
    chunks={};workers=parent.get('workers',[])
    need(len(workers)==parent.get('worker_count')and workers,'parent worker denominator differs')
    worker_ids=set()
    for part in workers:
        wid=part['worker_id'];need(isinstance(wid,str)and wid not in worker_ids,'duplicate worker')
        worker_ids.add(wid);root=child(resources,wid);terminal=part['terminal']
        summary,summary_sha=pin(dict(path=str(root/'worker_summary.json'),sha256=terminal['summary_sha256']))
        need(summary==terminal['summary']and summary.get('worker_id')==wid,'parent embedded summary differs')
        ready,ready_sha=pin(dict(path=str(root/'terminal_ready.json'),sha256=terminal['ready_sha256']))
        ack_raw,_=read(root/'terminal_ack.json');ack=json_value(ack_raw)
        need(ready.get('summary_path')==str(root/'worker_summary.json')and ready.get('summary_sha256')==summary_sha
             and ack.get('ready_sha256')==ready_sha and ready.get('nonce')==ack.get('nonce'),'worker publication ACK differs')
        for value in (summary,ready):
            need(value.get('contract_sha256')==contract_sha and value.get('index_sha256')==index_sha,'worker input pin differs')
        need(ready.get('bundle_sha256')==parent['bundle_sha256'],'worker producer bundle differs')
        for ref in summary.get('chunk_receipt_refs',[]):
            name=ref['id'];need(name not in chunks,'duplicate worker chunk ownership')
            need(ref.get('path')==str(child(queue,'completed/'+name+'.json')),'worker chunk path differs')
            chunks[name]=ref
    return chunks


def effective_tokens(window):
    asr=window.get('asr',{});native=asr.get('native',{});ids=native.get('generated_token_ids')
    if asr.get('status')not in {'completed','truncated'}:
        return None
    need(isinstance(ids,list)and all(type(v)is int and v>=0 for v in ids),'saved generation IDs missing')
    first=native.get('first_eos_index');observed=native.get('eos_observed')
    need(type(observed)is bool,'saved EOS observation must be boolean')
    if observed:
        need(type(first)is int and 0<=first<len(ids),'saved EOS index invalid');effective=ids[:first+1]
    else:
        need(first is None,'unobserved EOS must not have index');effective=ids
    if 'padding_policy'in native:
        eos=native.get('eos_token_ids');pad=native.get('padding_token_id')
        need(isinstance(eos,list)and eos and all(type(v)is int and v>=0 for v in eos)
             and type(pad)is int and pad>=0,'native actual EOS/pad missing')
        actual=next((i for i,t in enumerate(ids)if t in eos),None)
        need(actual==first and observed==(actual is not None),'native actual EOS contradicts saved index')
        tail=[]if actual is None else ids[actual+1:]
        need(all(t==pad or t in eos for t in tail)and native.get('post_eos_token_ids')==tail,'native post-EOS suffix differs')
        need(native.get('effective_generated_token_count')==len(effective)and native.get('raw_returned_token_count')==len(ids),
             'native raw/effective denominator differs')
    if 'effective_generated_token_ids'in native:need(native['effective_generated_token_ids']==effective,'native effective IDs contradict raw generation')
    return effective


def load(ref,profile,*,native,parent_sha):
    prepared,_=pin(ref)
    need(prepared.get('schema')==('annotation-native-batch-prepared/v1'if native else'annotation-bounded-speed-prepared/v1'),
         'wrong bridge preparation schema')
    count=prepared.get('source_count');need(type(count)is int and 0<count<=64,'bridge is limited to64 fixed sources')
    variant=prepared['variants'][profile];track,_=pin(variant['track']);contract,_=pin(track['contract']);index,_=pin(track['index'])
    need(index['source_count']==count and track['index']==prepared['index'],'bridge input denominator differs')
    runtime=contract['runtime']
    need(contract.get('purpose')=='pilot_validation'and runtime.get('asr_compute')=={'dtype':'float32','attention':'sdpa'}and
         runtime.get('max_batch')==1,'bridge requires actual offline FP32/SDPA B1 producers')
    if native:
        need(runtime.get('execution_policy')=='asr_fixed_cohort_fa_single/v1'and runtime.get('asr_batching')==dict(
            policy='fixed_cohort_until_durable/v1',max_requests=1,ordering='frames_then_row_id_within_source_group/v1'),
            'native bridge endpoint must be explicit B1')
    else:need(runtime.get('execution_policy')=='single_window_independent','baseline must retain old single-window semantics')
    parent,parent_sha=pin(dict(path=str(Path(track['resources_root'])/'parent_result.json'),sha256=parent_sha))
    need(parent.get('contract_sha256')==track['contract']['sha256']and parent.get('index_sha256')==track['index']['sha256']and
         parent.get('bundle_sha256')==variant['freeze']['bundle_sha256']and parent.get('resource_returned')is True and
         parent.get('full_sidecar_readback')is True and parent.get('verification',{}).get('passed')is True and
         parent['verification'].get('database_created')is False,'bridge requires upstream complete finite readback/resource return')
    queue=Path(track['queue_root']);completion_raw,completion_sha=read(queue/'completion_map.bin')
    slots=completion_slots(completion_raw,index,track['contract']['sha256'],track['index']['sha256'])
    publications=parent_publications(parent,Path(track['resources_root']),queue,track['contract']['sha256'],track['index']['sha256'])
    need(set(publications)==set(slots),'parent published chunk union differs')
    tree=Path(track['output_root'])/contract['version'];records={};refs={};chunk_pins={}
    expected_models={name:dict(path=value['path'],files_sha256=digest(value['files']),files=value['files'])for name,value in contract['models'].items()}
    for shard in index['shards']:
        slot=slots[shard['id']];publication=publications[shard['id']]
        need(publication['sha256']==slot['commit_sha256']and publication['source_count']==shard['count'],'parent/completion chunk pin differs')
        chunk_path=child(queue,'completed/'+shard['id']+'.json');chunk,chunk_sha=pin(dict(path=str(chunk_path),sha256=slot['commit_sha256']))
        for key,expected in dict(schema='redimnet2-chunk-commit/v1',shard_id=shard['id'],shard_sha256=shard['sha256'],
                source_count=shard['count'],contract_sha256=track['contract']['sha256'],index_sha256=track['index']['sha256']).items():
            need(chunk.get(key)==expected,'published chunk identity differs: '+key)
        receipts={r['source_id']:r for r in chunk['sources']}
        need(len(receipts)==len(chunk['sources'])==shard['count']and digest(sorted(receipts))==slot['source_ids_sha256'],'published source union differs')
        chunk_pins[shard['id']]=dict(path=str(chunk_path),sha256=chunk_sha);decisions=Counter()
        raw,_=read(child(Path(track['index']['path']).parent,shard['path']),shard['sha256'])
        rows=[json_value(line)for line in raw.splitlines()if line]
        need(len(rows)==shard['count'],'bridge shard count differs')
        for source in rows:
            sid=source['source_id'];need(sid not in records,'duplicate bridge source')
            receipt=receipts[sid];file=child(tree,source['sidecar_relpath']);raw,sha=read(file,receipt['sidecar_sha256']);record=json_value(raw)
            marker_raw,msha=read(str(file)+'.commit.json',receipt['commit_sha256']);marker=json_value(marker_raw)
            source_sha=hashlib.sha256(canonical(source)).hexdigest()
            job=digest(dict(source_id=sid,source_record_sha256=source_sha,contract_sha256=track['contract']['sha256']))
            need(record['execution']['job_key']==receipt['job_key']==job and receipt['sidecar_bytes']==len(raw)
                 and receipt['decision']==record['decision']['state'],'published source job/bytes/decision differs')
            need(record['provenance'].get('runtime')==runtime and record['provenance'].get('models')==expected_models
                 and record['provenance'].get('code_sha256')==parent['bundle_sha256'],'source runtime/model/producer differs')
            need(record['source']==source and record['contract']['sha256']==track['contract']['sha256']and
                 marker==dict(schema='redimnet2-sidecar-commit/v1',source_id=sid,job_key=record['execution']['job_key'],
                     contract_sha256=track['contract']['sha256'],source_record_sha256=source_sha,sidecar_sha256=sha,bytes=len(raw)),
                 'bridge source/contract/commit differs')
            records[sid]=record;refs[sid]=dict(sidecar=dict(path=str(file),sha256=sha),commit=dict(path=str(file)+'.commit.json',sha256=msha))
            decisions[record['decision']['state']]+=1
        need(set(receipts)=={s['source_id']for s in rows}and slot['decision_counts']=={k:decisions[k]for k in ('candidate','unknown','failed')},'published decisions differ')
    need(len(records)==count,'bridge complete source union differs')
    return dict(contract=contract,contract_ref=track['contract'],index_ref=track['index'],track_ref=variant['track'],records=records,refs=refs,
                parent_sha256=parent_sha,publication_chain=dict(parent_sha256=parent_sha,completion_sha256=completion_sha,chunks=chunk_pins))


def compare(old,new):
    for key in ('models','source_files','alignment','decoder','quality_policy'):
        need(old['contract'].get(key)==new['contract'].get(key),'bridge model/control differs: '+key)
    runtimes=[]
    for track in (old,new):runtimes.append({k:v for k,v in track['contract']['runtime'].items()if k not in {'execution_policy','asr_batching'}})
    need(runtimes[0]==runtimes[1],'bridge changed clock/window/normalization/generation/precision')
    need(set(old['records'])==set(new['records']),'bridge source sets differ')
    counts={k:0 for k in ('sources','windows','source_identity_changed','input_changed','text_changed','language_changed',
        'status_changed','effective_tokens_changed','raw_tokens_changed','native_time_array_changed','fixed_time_array_changed','comparable_geometry_windows')}
    rows=[]
    source_fields=('decode_status','file_sha256','pcm_sha256','pcm_hash_policy','sample_rate','channels','frames')
    window_fields=('window_id','input_pcm_sha256','model_input_frames','model_sample_rate','context_start_sample',
        'context_end_sample','owned_start_sample','owned_end_sample','resampling_policy','time_mapping')
    for sid in sorted(old['records']):
        a,b=(track['records'][sid]for track in (old,new));need(a['source']==b['source'],'bridge source record changed')
        counts['sources']+=1;source_equal=all(a['source_identity'].get(k)==b['source_identity'].get(k)for k in source_fields)
        counts['source_identity_changed']+=not source_equal
        wa={w['window_id']:w for w in a.get('windows',[])};wb={w['window_id']:w for w in b.get('windows',[])}
        need(len(wa)==len(a.get('windows',[]))and len(wb)==len(b.get('windows',[]))and set(wa)==set(wb),'bridge window union differs')
        row=dict(source_id=sid,dataset=a['source']['dataset'],source_identity_equal=source_equal,
                 metadata_changed=[k for k in ('errors','vad','units','decision','quality_axes')if a.get(k)!=b.get(k)],windows=[])
        for wid in sorted(wa):
            left,right=wa[wid],wb[wid];aa,bb=left['asr'],right['asr'];al,bl=left.get('alignment',{}),right.get('alignment',{})
            equal=dict(input=all(left.get(k)==right.get(k)for k in window_fields),
                text=all(aa.get(k)==bb.get(k)for k in ('raw_text','normalized_text','normalization_audit')),
                language=aa.get('language')==bb.get('language'),status=aa.get('status')==bb.get('status'),
                effective_tokens=effective_tokens(left)==effective_tokens(right),
                raw_tokens=aa.get('native',{}).get('generated_token_ids')==bb.get('native',{}).get('generated_token_ids'),
                native_time_array=al.get('native_timestamps_ms')==bl.get('native_timestamps_ms'),
                fixed_time_array=al.get('fixed_timestamps_ms')==bl.get('fixed_timestamps_ms'))
            same_target=(bool(al.get('target_text'))and al.get('target_text')==bl.get('target_text')and
                [v.get('text')for v in al.get('items',[])]==[v.get('text')for v in bl.get('items',[])])
            delta={}
            for key in ('native_timestamps_ms','fixed_timestamps_ms'):
                x,y=al.get(key),bl.get(key)
                if same_target and isinstance(x,list)and isinstance(y,list)and x and len(x)==len(y)and all(type(z)in (int,float)and math.isfinite(z)for z in x+y):
                    delta[key]=max(abs(v-w)for v,w in zip(x,y))
            counts['windows']+=1;counts['comparable_geometry_windows']+=bool(delta)
            for k,value in equal.items():counts[{'input':'input_changed','text':'text_changed','language':'language_changed','status':'status_changed',
                'effective_tokens':'effective_tokens_changed','raw_tokens':'raw_tokens_changed','native_time_array':'native_time_array_changed','fixed_time_array':'fixed_time_array_changed'}[k]]+=not value
            row['windows'].append(dict(window_id=wid,equal=equal,same_target_geometry_comparable=bool(delta),maximum_endpoint_delta_ms=delta,
                observations={name:dict(asr=w['asr'],alignment=w.get('alignment',{}),input={k:w.get(k)for k in window_fields})for name,w in [('old_single',left),('native_b1',right)]}))
        rows.append(row)
    return dict(schema='annotation-native-b1-mathematical-bridge/v1',source_count=len(rows),counts=counts,rows=rows,
        baseline=dict(contract=old['contract_ref'],index=old['index_ref'],track=old['track_ref'],parent_sha256=old['parent_sha256'],sources=old['refs'],publication_chain=old.get('publication_chain')),
        candidate=dict(contract=new['contract_ref'],index=new['index_ref'],track=new['track_ref'],parent_sha256=new['parent_sha256'],sources=new['refs'],publication_chain=new.get('publication_chain')),
        comparison_is_same_controlled_model_outputs_not_same_software_or_hardware=True,speedup_ratio=None,
        GPU_executed_by_this_tool=False,original_audio_read=False,database_created=False,human_accuracy_certified=False,formal_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for prefix in ('old','native'):
        p.add_argument('--'+prefix+'-prepared',required=True);p.add_argument('--'+prefix+'-sha',required=True);p.add_argument('--'+prefix+'-parent-sha',required=True)
    a=p.parse_args();old=load(dict(path=a.old_prepared,sha256=a.old_sha),'asr-fp32-sdpa',native=False,parent_sha=a.old_parent_sha)
    new=load(dict(path=a.native_prepared,sha256=a.native_sha),'native-fp32-sdpa-b1',native=True,parent_sha=a.native_parent_sha)
    print(json.dumps(compare(old,new),ensure_ascii=False,sort_keys=True,indent=2))


if __name__=='__main__':main()
