"""只读校正已完成portable准备的角色聚合，另写同目录v2元数据。

顺序核现有JSONL，不读媒体，不修改v1索引/计划/分片、队列、侧车或旧合同。
当前索引role计数与parent全量计数分别保存；不授GPU或训练资格。
"""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from urllib.parse import quote

if __package__:
    from . import prepare_speed_continuation as P
    from . import portable_source_identity as V
    from . import union_coverage as U
else:
    import prepare_speed_continuation as P
    import portable_source_identity as V
    import union_coverage as U

ROLES=('train_ptn','train_lm')


def pin(path):return {'path':str(path),'sha256':P.file_sha(path)}

def read_pin(value):return U.ref({key:value[key]for key in ('path','sha256')})

def filename(value):
    P.need(isinstance(value,str) and Path(value).name==value and value.endswith('.json') and value not in {'.','..'},
           'new metadata filename must be one JSON filename in the original preparation directory')
    return value


def role_counts(value,count):
    P.need(isinstance(value,dict) and set(value)==set(ROLES) and
           all(type(v)is int and 0<=v<=count for v in value.values()) and value['train_ptn']==count,
           'original full role aggregate must have one PTN membership per original source')
    return value


def finalize(plan_path,plan_sha,*,index_name='index-roles-v2.json',plan_name='coverage-plan-roles-v2.json',
             receipt_name='remaining-role-finalize-v2.json',progress=None):
    plan_path=P.physical(plan_path);plan_pin={'path':str(plan_path),'sha256':plan_sha};plan=read_pin(plan_pin)
    P.need(plan.get('schema')=='annotation-speed-continuation-plan/v1' and plan.get('status')=='final_prepared_not_started' and
           plan.get('source_identity_mode')==V.MODE and plan.get('new_inference_count')==0 and plan.get('formal_runtime_admitted')is False and
           plan.get('models_executed')is False and plan.get('stage_cache_relabelled')is False and
           plan.get('old_queue_mutated')is False and plan.get('old_results_rewritten')is False,
           'completed portable v1 preparation required; no empty/provisional successor')
    parent=plan_path.parent;names=[filename(v)for v in (index_name,plan_name,receipt_name)]
    outputs=[P.child(parent,v)for v in names]
    P.need(len(set(outputs))==3 and all(not p.exists()for p in outputs),'all v2 metadata files must be new and distinct')
    original_pin=plan['remaining_index'];oldpath=P.physical(original_pin['path'])
    P.need(oldpath.parent==parent and oldpath!=plan_path,'v1 remaining index must share plan directory')
    index=read_pin(original_pin);full_pin=plan['old_index'];full=read_pin(full_pin)
    oldc=read_pin(plan['old_contract']);queue=P.physical(plan['old_queue']);tree=P.physical(plan['old_sidecar_tree'])
    P.need(index.get('parent_full_index')==full_pin and index.get('source_count')==plan['remaining_source_count']>0 and
           plan['retained_source_count']+index['source_count']==full['source_count'], 'full/retained/remaining index count differs')
    full_roles=role_counts(full.get('role_union_counts'),full['source_count'])
    policy_ref=plan['portable_source_spec_ref']
    policy=V.load_portable_spec(policy_ref['path'],policy_ref['sha256'],contract_pin={k:plan['old_contract'][k]for k in ('path','sha256')},
        index_pin=full_pin,queue=queue,tree=tree)
    if policy['purpose']=='production':
        P.need(full['source_count']==P.EXPECTED_SOURCE_COUNT and full.get('domains')==P.EXPECTED_DOMAINS,
               'original formal three-domain denominator differs')
    inventory_pin=plan['portable_sidecar_inventory_ref'];inventory_path=P.physical(inventory_pin['path'])
    P.need(P.file_sha(inventory_path)==inventory_pin['sha256'],'inventory SHA differs')
    inventory_stamp=inventory_path.stat()
    db=sqlite3.connect('file:'+quote(str(inventory_path),safe='/')+'?mode=ro',uri=True)
    started=time.monotonic();last=[0.]
    if progress is None and policy['purpose']=='production':
        progress=lambda event:print(json.dumps(event,ensure_ascii=False),file=sys.stderr,flush=True)
    def emit(stage,force=False,**counts):
        now=time.monotonic()
        if progress is not None and (force or now-last[0]>=30):
            progress({'schema':'annotation-role-finalize-progress/v1','stage':stage,'elapsed_seconds':now-started,**counts});last[0]=now
    try:
        metadata={k:json.loads(v)for k,v in db.execute('SELECT key,value FROM metadata')}
        P.need(metadata==plan['portable_sidecar_inventory_summary'] and metadata.get('schema')=='annotation-portable-sidecar-inventory/v1' and
               metadata.get('portable_spec_sha256')==policy_ref['sha256'] and metadata.get('old_contract')==policy['old_contract'] and
               metadata.get('full_index')==full_pin and metadata.get('old_sidecar_tree')==str(tree) and
               metadata.get('all_inventory_paths_matched_original_sources')is True and metadata.get('original_sources_checked')==full['source_count'],
               'inventory metadata/source scope differs')
        P.need(db.execute('SELECT COUNT(*) FROM paths').fetchone()[0]==plan['retained_source_count'] and
               db.execute("SELECT COUNT(*) FROM paths WHERE sidecar!=1 OR commit_marker!=1 OR seen!=1 OR source_id IS NULL OR kind NOT IN ('completed','partial','unclaimed')").fetchone()[0]==0,
               'inventory retained count or source mapping invalid')
        # 内容账本只覆盖partial/unclaimed；重开其原JSON证据，不读取编码媒体。
        emit('validating_saved_proofs',True,remaining_rows_checked=0)
        with tempfile.TemporaryDirectory(prefix='.role-finalize-ledger-',dir=parent)as scratch:
            proofs=sqlite3.connect(str(Path(scratch)/'ledger.sqlite'));proofs.execute('CREATE TABLE entries(sid TEXT PRIMARY KEY,value TEXT)')
            try:
                ledger_ref=plan['portable_source_ledger_ref'];h=hashlib.sha256();count=0
                fd=U.open_readonly(P.physical(ledger_ref['path']))
                try:
                    before=os.fstat(fd)
                    with os.fdopen(fd,'rb',closefd=False)as stream:
                        while True:
                            line=stream.readline(65537)
                            if not line:break
                            P.need(line.strip() and len(line)<=65536,'portable ledger row blank or oversized');h.update(line)
                            entry=U.strict_json(line)
                            try:proofs.execute('INSERT INTO entries VALUES(?,?)',(entry['source_id'],line.decode('utf-8')))
                            except sqlite3.IntegrityError as exc:raise ValueError('duplicate portable ledger source')from exc
                            count+=1
                    after=os.fstat(fd);P.need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'portable ledger changed')
                finally:os.close(fd)
                P.need(count==ledger_ref['count'] and h.hexdigest()==ledger_ref['sha256'],'portable ledger SHA/count differs')
                def check_saved(expected):
                    record,raw=P.read_json(P.physical(expected['sidecar_path']),32*1024**2)
                    source=record['source'];producer={'sidecar_tree':str(tree),'contract':policy['old_contract']}
                    _,rs,cs=U.read_result(source,producer,expected['sidecar_sha256'],expected['commit_sha256'],contract=oldc)
                    P.need(expected['source_id']==source['source_id'] and expected['source_record_sha256']==P.digest(source) and
                           expected['sidecar_path']==str(P.child(tree,source['sidecar_relpath'])) and
                           expected['commit_path']==str(P.child(tree,source['sidecar_relpath']+'.commit.json')),
                           'retained saved source reference differs')
                    row=proofs.execute('SELECT value FROM entries WHERE sid=?',(source['source_id'],)).fetchone()
                    P.need(row is not None,'portable ledger omits saved source');entry=U.strict_json(row[0])
                    P.need(entry['sidecar_sha256']==rs and entry['commit_sha256']==cs,'portable ledger original byte binding differs')
                    V.validate_ledger_entry(entry,source,record,policy['old_contract']['sha256'],policy,spec_sha=policy_ref['sha256'])
                    match=db.execute('SELECT source_id,kind FROM paths WHERE relative=?',(source['sidecar_relpath'],)).fetchone()
                    P.need(match is not None and match[0]==source['source_id'] and match[1]in {'partial','unclaimed'},'saved source inventory binding differs')
                    proofs.execute('DELETE FROM entries WHERE sid=?',(source['source_id'],))
                for expected in plan['retained_partial_sources']:check_saved(expected)
                unclaimed=plan['retained_unclaimed_sources_ref'];P.need(P.file_sha(P.physical(unclaimed['path']))==unclaimed['sha256'],'unclaimed ledger SHA differs')
                n=0
                with P.physical(unclaimed['path']).open('rb')as stream:
                    for line in stream:
                        P.need(len(line)<=65536 and line.strip(),'unclaimed row invalid');check_saved(U.strict_json(line));n+=1
                P.need(n==unclaimed['count']==plan['retained_unclaimed_source_count'] and
                       proofs.execute('SELECT COUNT(*) FROM entries').fetchone()[0]==0,'portable/unclaimed evidence coverage differs')
            finally:proofs.close()
        parents={v['id']:v for v in full['shards']};items={v['descriptor']['id']:v for v in plan['remaining_shards']}
        P.need(len(parents)==len(full['shards']) and len(items)==len(plan['remaining_shards'])==len(index['shards']),
               'duplicate parent/remaining shard descriptor')
        measured=collections.Counter();domains=collections.Counter();origins=set();total=0
        with P.old_parent_lock(queue)as current:
            P.need(all(current[k]==plan['old_parent_lock'][k]for k in ('path','device','inode')),'original preparation lock identity changed')
            P.need(P.file_sha(queue/'broker.json')==plan['old_progress_snapshot']['broker_snapshot_sha256'],'old broker changed')
            emit('counting_remaining_metadata',True,remaining_rows_checked=0,total_remaining=index['source_count'])
            for _,descriptor,rows in U.shards(index,oldpath):
                if 'source_ids_sha256'in descriptor:
                    P.need(descriptor['source_ids_sha256']==P.digest(sorted(s['source_id']for s in rows)),
                           'remaining source ID digest differs')
                item=items.pop(descriptor['id']);declared=item['descriptor']
                P.need(all(declared[k]==descriptor[k]for k in ('id','count','sha256')),'v1 plan/index shard binding differs')
                origin=item.get('origin_shard_id',descriptor['id']);P.need(origin in parents and origin not in origins,'remaining origin missing or duplicated');origins.add(origin)
                original=parents[origin]
                if item['mode']=='copy_original_shard':
                    P.need(descriptor['sha256']==original['sha256'] and descriptor['count']==original['count'],'whole source shard differs from parent')
                    allowed=None
                else:
                    P.need(item['mode']in {'new_partial_shard','new_unclaimed_shard'},'unknown filtered shard kind')
                    allowed={s['source_id']:s for s in P.read_shard(Path(full_pin['path']).parent,original)}
                for source in rows:
                    if allowed is not None:P.need(allowed.get(source['source_id'])==source,'filtered source changed from original parent')
                    P.need(db.execute('SELECT 1 FROM paths WHERE relative=?',(source['sidecar_relpath'],)).fetchone()is None,
                           'remaining source overlaps retained inventory')
                    measured.update(P.source_roles(source));domains[source['dataset']]+=1;total+=1
                emit('counting_remaining_metadata',remaining_rows_checked=total,total_remaining=index['source_count'])
            P.need(not items and total==index['source_count'] and dict(domains)==index['domains'],'remaining metadata coverage/count differs')
            counts={role:measured[role]for role in ROLES};retained={role:full_roles[role]-counts[role]for role in ROLES}
            P.need(counts['train_ptn']==total and all(0<=v<=total for v in counts.values()) and
                   retained['train_ptn']==plan['retained_source_count'] and all(0<=v<=plan['retained_source_count']for v in retained.values()),
                   'remaining/retained role subtraction invalid')
            P.need(P.file_sha(queue/'broker.json')==plan['old_progress_snapshot']['broker_snapshot_sha256'],'old broker changed while finalizing')
        now=inventory_path.stat();P.need((inventory_stamp.st_dev,inventory_stamp.st_ino,inventory_stamp.st_size,inventory_stamp.st_mtime_ns,inventory_stamp.st_ctime_ns)==
            (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns),'inventory changed during finalization')
        P.need(P.file_sha(plan_path)==plan_sha and P.file_sha(oldpath)==original_pin['sha256'],'v1 metadata changed during finalization')
        corrected=copy.deepcopy(index);corrected.update(role_union_counts=counts,role_union_counts_scope='remaining_index_sources',
            parent_role_union_counts=full_roles,role_union_counts_parent_index=full_pin,
            coverage_plan_file=names[1],metadata_derived_from_index=original_pin)
        new_index_pin={'path':str(outputs[0]),'sha256':P.write_new(outputs[0],corrected)}
        new_plan=copy.deepcopy(plan);new_plan['remaining_index']=new_index_pin
        new_plan['metadata_correction']={'schema':'annotation-remaining-role-correction/v1','source_plan':plan_pin,
            'source_index':original_pin,'tool_sha256':P.file_sha(Path(__file__).resolve()),'remaining_role_union_counts':counts,
            'retained_role_union_counts':retained,'source_shard_bytes_changed':False,'models_executed':False}
        new_plan_pin={'path':str(outputs[1]),'sha256':P.write_new(outputs[1],new_plan)}
        receipt={'schema':'annotation-remaining-role-finalize/v1','old_plan':plan_pin,'old_remaining_index':original_pin,
            'new_plan':new_plan_pin,'new_remaining_index':new_index_pin,'parent_full_index':full_pin,
            'parent_role_union_counts':full_roles,'remaining_role_union_counts':counts,'retained_role_union_counts':retained,
            'remaining_sources_checked':total,'portable_proofs_reopened':count,'inventory_ref':inventory_pin,
            'source_shard_bytes_changed':False,'v1_bytes_rewritten':False,'source_media_read':False,'models_executed':False,
            'formal_runtime_admitted':False,'GPU_executed':False,
            'unchanged_parent_lineage_fields':['version','script_sha256','excluded_dataset_counts','parent_registry_lineage_only','source_record_policy','scope','global_source_ID_uniqueness'],
            'inherited_parent_metadata_scope':'retained_verbatim_lineage_not_new_filter_execution_or_byte_equivalence_to_parent_shards',
            'inherited_parent_metadata':{key:index[key]for key in ('version','script_sha256','excluded_dataset_counts','parent_registry_lineage_only','source_record_policy','scope','global_source_ID_uniqueness')if key in index}}

        P.write_new(outputs[2],receipt);emit('completed',True,remaining_rows_checked=total)
        return receipt
    finally:db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--index-name',default='index-roles-v2.json');parser.add_argument('--plan-name',default='coverage-plan-roles-v2.json')
    parser.add_argument('--receipt-name',default='remaining-role-finalize-v2.json');a=parser.parse_args()
    print(json.dumps(finalize(a.plan,a.plan_sha,index_name=a.index_name,plan_name=a.plan_name,receipt_name=a.receipt_name),ensure_ascii=False))


if __name__=='__main__':main()
