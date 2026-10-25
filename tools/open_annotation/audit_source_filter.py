"""CPU full byte-SHA and source-role read-back of NEW three-training registry.
Only new registered JSONL is read; original audio/VAD/models/PTN are untouched.
"""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,json,os,time,traceback
from source_filter import DOMAINS,classify,canonical,safe_relative,need,atomic,digest
def one(job):
    ordinal,shard,root,rate=job;started=time.monotonic();path=Path(root)/safe_relative(shard['path']);before=path.stat();raw=path.read_bytes();after=path.stat();need((before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev),'new shard changed while audited');need(digest(raw)==shard['sha256'],'actual new shard SHA differs')
    ds=collections.Counter();roles=collections.Counter();ids=[];lines=raw.splitlines();need(len(lines)==shard['count'],'actual new source count differs')
    for line in lines:
        row=json.loads(line);yes,reason=classify(row);need(yes,'outside three-training source in new index');ds[row['dataset']]+=1;roles.update(row['roles']);ids.append(row['source_id'])
    need(len(set(ids))==len(ids) and digest(canonical(sorted(ids)))==shard['source_ids_sha256'],'new source-ID receipt differs')
    delay=len(raw)/rate-(time.monotonic()-started)
    if delay>0:time.sleep(delay)
    return {'ordinal':ordinal,'id':shard['id'],'path':shard['path'],'sha256':digest(raw),'source_ids_sha256':shard['source_ids_sha256'],'count':len(lines),'bytes':len(raw),'dataset_counts':dict(ds),'role_counts':dict(roles)}
def main(a):
    need(1<=a.workers<=4 and 0<a.total_mib_per_second<=24,'bounded <=4 CPU/24MiB/s required');os.nice(10);started=time.monotonic();deadline=started+a.deadline_seconds
    while not a.index.exists():
        need(time.monotonic()<deadline,'new filter index not ready before audit deadline');time.sleep(20)
    raw=a.index.read_bytes();idx=json.loads(raw);ish=digest(raw);need(idx['domains']==DOMAINS and idx['source_count']==sum(DOMAINS.values()),'new three-training scope mismatch');need(not a.output.exists(),'new audit output exists; inspect not overwrite');a.output.mkdir(parents=True)
    total=collections.Counter();ds=collections.Counter();roles=collections.Counter();last=time.monotonic();begin=last
    try:
        with (a.output/'shard_readback.jsonl').open('xb') as ledger,concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as executor:
            for start in range(0,len(idx['shards']),a.workers):
                need(time.monotonic()<deadline,'bounded audit deadline exceeded')
                jobs=[(i,idx['shards'][i],str(a.index.parent),a.total_mib_per_second*1048576/a.workers) for i in range(start,min(start+a.workers,len(idx['shards'])))]
                for result in executor.map(one,jobs,chunksize=1):
                    ledger.write(canonical(result));total['source_count']+=result['count'];total['bytes']+=result['bytes'];total['shards']+=1;ds.update(result['dataset_counts']);roles.update(result['role_counts'])
                if time.monotonic()-last>=30:print(json.dumps({'audit_completed_shards':total['shards'],'total_shards':len(idx['shards']),'actual_source_rows_reopened':total['source_count'],'actual_new_shard_bytes_rehashed':total['bytes'],'elapsed_seconds':time.monotonic()-begin}),flush=True);ledger.flush();last=time.monotonic()
        need(ds==DOMAINS and total['source_count']==sum(DOMAINS.values()) and total['shards']==len(idx['shards']) and roles==idx['role_union_counts'],'new full audit union differs')
        ledger_sha=digest((a.output/'shard_readback.jsonl').read_bytes());result={'schema':'redimnet2-three-train-full-readback/v1','passed':True,'index_sha256':ish,'source_count_validated':total['source_count'],'output_shards_verified':total['shards'],'actual_new_shard_bytes_rehashed':total['bytes'],'dataset_counts':dict(ds),'role_counts':dict(roles),'all_shard_file_SHA_and_source_ID_digest_reopened':True,'global_ID_set_rechecked_independently':False,'global_ID_uniqueness_basis':'unchanged-source subset of globally deduplicated frozen registry; pinned producer filtering proof, plus actual per-shard IDs SHA match','shard_readback_ledger_sha256':ledger_sha,'script_sha256':digest(Path(__file__).read_bytes()),'elapsed_seconds':time.monotonic()-begin,'GPU_or_model_execution':False,'old_files_modified':False,'audio_or_VAD_media_stat_or_decode':False};atomic(a.output/'summary.json',canonical(result));print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException:
        atomic(a.output/'failure.json',canonical({'traceback':traceback.format_exc(),'index_sha256':ish,'partial_counts':dict(total),'no_model_GPU_or_audio_IO':True}));raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--total-mib-per-second',type=float,default=24);p.add_argument('--deadline-seconds',type=int,default=10800);main(p.parse_args())
