"""LATEST USER SCOPE. CPU-only immutable 3-training-domain source index filter.
Reads registered JSONL only (no audio stat/decode/model/GPU); preserves source-line
bytes, PTN+LM role union. New output only; resumable per-input-shard receipts.
"""
from pathlib import Path,PurePosixPath
import argparse,collections,concurrent.futures,hashlib,json,os,time,traceback
DOMAINS={'voxceleb2':1092009,'voxblink2':9904382,'cnceleb2':524787}
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def source_id(row):return 'src_'+digest((row['dataset']+'\0'+row['relative_path']).encode())
def classify(row):
    if row['dataset'] not in DOMAINS:return False,'outside_requested_three_training_domains'
    need(row['original_splits']==['provider_train'],'three-domain source is not canonical provider_train')
    roles=set(row['roles'])
    need(bool(roles&{'train_ptn','train_lm'}),'three-domain source missing actual PTN/LM role')
    need(not roles-{'train_ptn','train_lm'},'three-domain row has evaluation/non-training role conflict')
    need(row['protection']['training_forbidden'] is False and not row.get('split_conflicts'),'three-domain source has protection/split conflict; preserve and investigate')
    need(row['schema']=='redimnet2-source/v1' and row['source_id']==source_id(row),'source identity differs')
    need(row['sidecar_relpath']==row['dataset']+'/'+row['relative_path']+'.annotation.json','sidecar path identity differs')
    return True,'actual_training_source'
def safe_relative(text):
    p=PurePosixPath(text);need(not p.is_absolute() and '..' not in p.parts and text,'unsafe shard path');return p
def atomic(path,data):
    temp=path.with_name(path.name+'.partial');need(not path.exists(),'immutable output collision '+str(path))
    with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()
def filter_one(job):
    ordinal,source,index_root,out_root,rate,script_sha=job;started=time.monotonic();out=Path(out_root);done=out/'work'/f'{ordinal:05d}.json';stem=f'three-train-{ordinal:05d}';target=out/'shards'/(stem+'.jsonl')
    if done.exists():
        result=json.loads(done.read_bytes());need(result['source_shard']==source and result['script_sha256']==script_sha,'prior filter receipt input/code mismatch')
        if result['selected_count']:need(digest(target.read_bytes())==result['shard']['sha256'],'prior filtered shard changed')
        return result
    try:
        path=Path(index_root)/safe_relative(source['path']);before=path.stat();raw=path.read_bytes();after=path.stat()
        need((before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev),'source shard changed while read')
        need(digest(raw)==source['sha256'],'input shard SHA differs '+source['id'])
        lines=raw.splitlines(keepends=True);need(len(lines)==source['count'] and all(x.endswith(b'\n') and x.strip() for x in lines),'input source count/newline invalid')
        selected=[];datasets=collections.Counter();role_counts=collections.Counter();excluded=collections.Counter();source_ids=[];seen=set()
        for line in lines:
            row=json.loads(line);need(row['source_id'] not in seen,'input chunk duplicate source ID');seen.add(row['source_id']);yes,reason=classify(row)
            if yes:selected.append(line);datasets[row['dataset']]+=1;role_counts.update(row['roles']);source_ids.append(row['source_id'])
            else:excluded[row['dataset']]+=1
        payload=b''.join(selected);shard=None
        if selected:
            shard={'id':stem,'path':'shards/'+stem+'.jsonl','sha256':digest(payload),'count':len(selected),'source_ids_sha256':digest(canonical(sorted(source_ids)))}
            if target.exists():need(target.read_bytes()==payload,'uncommitted pre-existing output differs')
            else:atomic(target,payload)
        result={'source_shard':source,'script_sha256':script_sha,'selected_count':len(selected),'excluded_count':len(lines)-len(selected),'dataset_counts':dict(datasets),'role_counts':dict(role_counts),'excluded_dataset_counts':dict(excluded),'input_bytes':len(raw),'output_bytes':len(payload),'shard':shard}
        delay=(len(raw)+len(payload))/rate-(time.monotonic()-started)
        if delay>0:time.sleep(delay)
        atomic(done,canonical(result));return result
    except BaseException:
        error=out/'errors'/f'{ordinal:05d}.json'
        if not error.exists():atomic(error,canonical({'input_ordinal':ordinal,'source_shard':source,'traceback':traceback.format_exc(),'script_sha256':script_sha}))
        raise
def build(a):
    need(1<=a.workers<=4 and 0<a.total_mib_per_second<=30,'bounded <=4 CPU and <=30MiB/s combined read+write required')
    need(len(a.index_sha)==64 and all(c in '0123456789abcdef' for c in a.index_sha),'explicit parent registry SHA required')
    raw=a.index.read_bytes();need(digest(raw)==a.index_sha,'input parent index SHA differs');parent=json.loads(raw)
    need(parent['source_count']==11740953 and len(parent['shards'])==45983,'parent source denominator differs')
    need(a.output.is_absolute() and a.output!=a.index.parent and a.index.parent not in a.output.parents,'output must be a new independent root, not inside old frozen registry')
    owned=a.run_root.resolve();need(owned.is_dir() and owned in a.output.resolve().parents and '..' not in a.output.parts,'filter output must remain inside explicit independent task root')
    os.nice(10);script_sha=digest(Path(__file__).read_bytes());job={'schema':'redimnet2-three-train-filter-job/v1','parent_index':str(a.index),'parent_index_sha256':a.index_sha,'script_sha256':script_sha,'domains':DOMAINS,'roles':['train_ptn','train_lm'],'source_rows_unchanged':True,'model_or_audio_read':False,'requested_scope':'latest user correction; excludes all Child/evaluation/augmentation/business/diagnostics datasets'}
    if a.output.exists():need(json.loads((a.output/'filter-job.json').read_bytes())==job,'existing output does not match exact same filter job')
    else:
        a.output.mkdir(parents=True);atomic(a.output/'filter-job.json',canonical(job))
    for name in ['shards','work','errors']:(a.output/name).mkdir(exist_ok=True)
    need(not (a.output/'index.json').exists(),'final index already exists; audit instead of rewriting')
    totals=collections.Counter();datasets=collections.Counter();roles=collections.Counter();excluded=collections.Counter();shards=[];start=time.monotonic();deadline=start+a.deadline_seconds;last_print=start
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as executor:
        # At most four queued/in-flight chunks; a deadline does not drain a huge
        # pre-submitted map after refusal. Each batch is yielded in input order.
        for batch_start in range(0,len(parent['shards']),a.workers):
            need(time.monotonic()<deadline,'bounded CPU filter deadline exceeded; keep receipts, resume same job')
            batch=[(i,parent['shards'][i],str(a.index.parent),str(a.output),a.total_mib_per_second*1048576/a.workers,script_sha) for i in range(batch_start,min(batch_start+a.workers,len(parent['shards'])))]
            for offset,result in enumerate(executor.map(filter_one,batch,chunksize=1)):
                ordinal=batch_start+offset
                totals.update({k:result[k] for k in ['selected_count','excluded_count','input_bytes','output_bytes']});datasets.update(result['dataset_counts']);roles.update(result['role_counts']);excluded.update(result['excluded_dataset_counts'])
                if result['shard']:shards.append(result['shard'])
                if (ordinal+1)%256==0 or time.monotonic()-last_print>=30:
                    print(json.dumps({'completed_input_shards':ordinal+1,'total_input_shards':len(parent['shards']),'selected':totals['selected_count'],'excluded':totals['excluded_count'],'combined_IO_bytes':totals['input_bytes']+totals['output_bytes'],'elapsed_seconds':time.monotonic()-start}),flush=True);last_print=time.monotonic()
    need(datasets==DOMAINS and totals['selected_count']==11521178 and totals['selected_count']+totals['excluded_count']==parent['source_count'],'exact three-domain/partition denominator differs')
    need(roles['train_ptn']==11521178 and roles['train_lm']==5021010 and set(roles)=={'train_ptn','train_lm'},'PTN/LM merged role accounting differs')
    index={'schema':'redimnet2-shard-index/v1','source_schema':'redimnet2-source/v1','version':'ultra-three-training-only-full-20261025-v1','source_count':totals['selected_count'],'shards':shards,'scope':'ONLY actual model-used VoxCeleb2/VoxBlink2/CN-Celeb2 train_ptn/train_lm sources, latest explicit user correction','domains':dict(datasets),'role_union_counts':dict(roles),'LM_new_independent_sources':0,'parent_registry_lineage_only':{'path':str(a.index),'sha256':a.index_sha,'source_count':parent['source_count']},'excluded_dataset_counts':dict(excluded),'source_record_policy':'each selected original raw JSONL source line byte-identical; no audio/VAD/split/source metadata changes','source_identity':'dataset+relative path SHA; NOT global byte/PCM/speaker dedup','global_source_ID_uniqueness':'subset of SHA-pinned globally deduplicated parent registry, unchanged IDs; per-input-chunk duplicates independently refused','API_calls':0,'automatic_training_release':False,'human_review':None,'RIR_or_MUSAN_or_Child_or_evaluation_in_this_job':False,'script_sha256':script_sha}
    atomic(a.output/'index.json',canonical(index));summary=job|{'passed':True,'source_count':totals['selected_count'],'shard_count':len(shards),'index_sha256':digest((a.output/'index.json').read_bytes()),'dataset_counts':dict(datasets),'role_counts':dict(roles),'excluded_dataset_counts':dict(excluded),'totals':dict(totals),'elapsed_seconds':time.monotonic()-start,'GPU_or_model_executed':False,'audio_stat_or_decode_performed':False,'old_files_modified':False}
    atomic(a.output/'summary.json',canonical(summary));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--index-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--total-mib-per-second',type=float,default=24);p.add_argument('--deadline-seconds',type=int,default=7200);build(p.parse_args())
