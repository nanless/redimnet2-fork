"""仅核已发布薄索引、报告、新队列身份和完成槽；不重验来源闭包或授运行资格。"""
import sys
sys.dont_write_bytecode=True
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct

from readonly_progress import read_small, strict_json

STATES=('candidate','unknown','failed')
SLOT_KEYS={'shard_id','shard_sha256','source_count','source_ids_sha256','commit_sha256',
    'index_sha256','contract_sha256','decision_counts','retained_source_count',
    'current_producer_source_count','retained_producer_contract_sha256',
    'current_producer_decision_counts','retained_decision_counts'}


def need(ok,message):
    if not ok:raise ValueError(message)


def physical(path):
    path=Path(path)
    need(path.is_absolute()and '..'not in path.parts and all(not p.is_symlink()for p in(path,*path.parents)),'physical absolute metadata path required')
    return path


def sha(value):return isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)is not None


def pinned(pin):
    need(isinstance(pin,dict)and set(pin)=={'path','sha256'}and sha(pin['sha256']),'explicit metadata pin required')
    path=physical(pin['path']);need(path.suffix=='.json'and not str(path).endswith(('.annotation.json','.commit.json')),'only explicit index/report metadata JSON is allowed')
    value,raw=read_small(path,64*1024**2)
    need(hashlib.sha256(raw).hexdigest()==pin['sha256'],'metadata pin SHA differs')
    return value


def metadata_inputs(index_pin,report_pin):
    """仅解析两份固定JSON；不是BoundedShardReader，不打开任何来源或旧输出。"""
    index=pinned(index_pin);report=pinned(report_pin);shards=index.get('shards')
    need(index.get('schema')=='annotation-bounded-source-index/v1'and isinstance(shards,list)and shards,'explicit nonempty thin metadata required')
    need(type(index.get('source_count'))is int and type(index.get('shard_count'))is int and index['shard_count']==len(shards),'thin denominator type/count differs')
    need(len({s['id']for s in shards})==len(shards)and all(type(s.get('count'))is int and s['count']>0 and sha(s.get('sha256'))for s in shards),'thin descriptor count/uniqueness differs')
    need(sum(s['count']for s in shards)==index['source_count'],'thin source sum differs')
    parts=[s for s in shards if s.get('storage')=='local_partial'];unclaimed=[s for s in shards if s.get('storage')=='parent_unclaimed']
    need(len(parts)<=8 and shards==parts+unclaimed,'partial/unclaimed ordering differs')
    cursor=index.get('parent_cursor');need(type(cursor)is int and cursor>=0,'parent cursor missing')
    need(all(type(s.get('parent_ordinal'))is int for s in unclaimed)and
         [s.get('parent_ordinal')for s in unclaimed]==list(range(cursor,cursor+len(unclaimed))),'unclaimed ordinals reordered or missing')
    need(all(type(index.get(k))is int and index[k]>=0 for k in ('previous_retained_count','parent_preserved_source_count','global_source_count'))and
         index['previous_retained_count']+index['parent_preserved_source_count']+index['source_count']==index['global_source_count'],'global metadata count differs')
    need(report.get('status')=='terminal_thin_plan_not_gpu_admission'and report.get('new_index')==index_pin and
         report.get('old_producer_modified')is False and report.get('database_created')is False and report.get('training_labels_released')is False,'published bounded report binding differs')
    need(all(type(report.get(k))is int and report[k]>=0 for k in ('exact_remaining_source_count','global_source_count','previous_retained_count','retained_complete_source_count','retained_partial_source_count'))and
         report.get('exact_remaining_source_count')==index['source_count']and
         report.get('global_source_count')==index['global_source_count']and report.get('previous_retained_count')==index['previous_retained_count']and
         type(report.get('retained_complete_source_count'))is int and type(report.get('retained_partial_source_count'))is int and
         report['retained_complete_source_count']+report['retained_partial_source_count']==index['parent_preserved_source_count'],'published report accounting differs')
    return index


def inspect(index_pin,report_pin,queue_root,contract_sha):
    index=metadata_inputs(index_pin,report_pin);root=physical(queue_root);need(str(root)==index['new_queue_root']and sha(contract_sha),'explicit new queue/contract required')
    identity,identity_raw=read_small(root/'identity.json',65536)
    expected=dict(schema='redimnet2-bounded-work-queue/v1',index_sha256=index_pin['sha256'],contract_sha256=contract_sha,
                  old_contract_sha256=index['old_contract']['sha256'],source_count=index['source_count'],shard_count=len(index['shards']))
    need(json.dumps(identity,sort_keys=True)==json.dumps(expected,sort_keys=True),'new queue identity differs')
    broker,broker_raw=read_small(root/'broker.json',1024**2);shards=index['shards'];by_id={s['id']:i for i,s in enumerate(shards)}
    need(isinstance(broker,dict)and set(broker)=={'next','leases','completed_chunks'},'broker fields differ')
    cursor=broker['next'];leases=broker['leases'];completed=broker['completed_chunks']
    need(type(cursor)is int and 0<=cursor<=len(shards)and isinstance(leases,dict)and len(leases)<=8 and type(completed)is int and completed==cursor-len(leases),'broker cursor/count differs')
    need(all(sid in by_id and by_id[sid]<cursor and isinstance(v,dict)for sid,v in leases.items()),'claimed shard outside cursor')
    counts={k:0 for k in STATES};current=dict(counts);retained=dict(counts);total=0;slots=0;slots_hash=hashlib.sha256()
    storage={k:dict(chunks=0,sources=0,current_producer_sources=0,retained_sources=0)for k in ('local_partial','parent_unclaimed')}
    path=physical(root/'completion_map.bin');initial=path.lstat();need(stat.S_ISREG(initial.st_mode),'completion map must be regular')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode)and(before.st_dev,before.st_ino,before.st_size)==(initial.st_dev,initial.st_ino,len(shards)*1024),'completion map shape differs')
        for ordinal in range(cursor):
            shard=shards[ordinal]
            if shard['id']in leases:continue
            data=os.pread(fd,1024,ordinal*1024);need(len(data)==1024 and data[:8]==b'RDMAP001','torn completion slot')
            size=struct.unpack('>I',data[8:12])[0];need(0<size<=980,'slot payload size differs');raw=data[44:44+size]
            need(hashlib.sha256(raw).digest()==data[12:44]and data[44+size:]==bytes(980-size),'slot checksum/padding differs')
            value=strict_json(raw);need(set(value)==SLOT_KEYS,'slot fields differ')
            need(value['shard_id']==shard['id']and value['shard_sha256']==shard['sha256']and value['index_sha256']==index_pin['sha256']and
                 value['contract_sha256']==contract_sha and value['retained_producer_contract_sha256']==index['old_contract']['sha256']and
                 type(value['source_count'])is int and value['source_count']==shard['count']and sha(value['commit_sha256'])and sha(value['source_ids_sha256']),'slot ordinal/producer/source identity differs')
            for key in ('decision_counts','current_producer_decision_counts','retained_decision_counts'):
                d=value[key];need(isinstance(d,dict)and set(d)==set(STATES)and all(type(n)is int and n>=0 for n in d.values()),'slot decision counts differ')
            c=value['current_producer_decision_counts'];r=value['retained_decision_counts'];all_counts=value['decision_counts']
            need(type(value['current_producer_source_count'])is int and type(value['retained_source_count'])is int and
                 sum(c.values())==value['current_producer_source_count']and sum(r.values())==value['retained_source_count']and
                 sum(c.values())+sum(r.values())==shard['count']and all(all_counts[k]==c[k]+r[k]for k in STATES),'slot producer accounting differs')
            for key in STATES:counts[key]+=all_counts[key];current[key]+=c[key];retained[key]+=r[key]
            group=storage[shard['storage']];group['chunks']+=1;group['sources']+=shard['count'];group['current_producer_sources']+=sum(c.values());group['retained_sources']+=sum(r.values())
            total+=shard['count'];slots+=1;slots_hash.update(data)
        after=os.fstat(fd);entry=path.lstat();need((before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size)==(entry.st_dev,entry.st_ino,entry.st_size)and stat.S_ISREG(entry.st_mode),'completion map replaced/resized')
    finally:os.close(fd)
    need(slots==completed,'completed slot count differs')
    preserved=index['previous_retained_count']+index['parent_preserved_source_count']
    return dict(schema='redimnet2-bounded-metadata-progress/v1',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),index=index_pin,report=report_pin,
        contract_sha256=contract_sha,identity_sha256=hashlib.sha256(identity_raw).hexdigest(),broker_snapshot_sha256=hashlib.sha256(broker_raw).hexdigest(),broker_snapshot=broker,
        source_count=index['source_count'],verified_completed_chunks=slots,committed_source_lower_bound=total,decision_counts=counts,
        current_producer_source_count=sum(current.values()),current_producer_decision_counts=current,previous_producer_retained_in_this_queue=sum(retained.values()),
        previous_producer_decision_counts=retained,completed_by_storage=storage,preserved_before_new_queue=preserved,global_committed_source_lower_bound=preserved+total,
        remaining_source_count=index['source_count']-total,completed_slots_sha256=slots_hash.hexdigest(),completion_map_bytes_read=slots*1024,
        source_JSONL_reads=0,sidecar_or_commit_reads=0,queue_writes=0,locks_acquired=0,database_created=False,GPU_executed=False,
        source_closed_scope_revalidated=False,human_quality_verified=False,runtime_admitted=False,
        meaning='Pinned published metadata and one broker snapshot; active claims excluded. Publication ownership is not current-attempt model calls or human correctness.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True);p.add_argument('--index-sha',required=True);p.add_argument('--report',required=True);p.add_argument('--report-sha',required=True);p.add_argument('--queue-root',required=True);p.add_argument('--contract-sha',required=True);a=p.parse_args()
    print(json.dumps(inspect(dict(path=a.index,sha256=a.index_sha),dict(path=a.report,sha256=a.report_sha),a.queue_root,a.contract_sha),ensure_ascii=False))
