"""After GPU exit: same-parent-lock full readback plus orphan/symlink inventory.

No models/CUDA. Expected membership uses temporary local SQLite (no WAL), and
large audit manifests stream as gzip JSONL. Never deletes or repairs artifacts.
"""
from contextlib import ExitStack
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import traceback

DEFAULT_PRODUCTION=Path(__file__).resolve().parent
_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--production-root',type=Path,default=DEFAULT_PRODUCTION)
_pre_args,_unknown=_pre.parse_known_args()
sys.path.insert(0,str(_pre_args.production_root))
from contracts import atomic_new,canonical_bytes,need,read_contract,safe_join,sha_file,utcnow
from report import verify_run
from work_queue import ProductionRunLock,read_index

TEMP_PATTERN=re.compile(r'^\.(?P<target>.+)\.[0-9a-f]{24}\.tmp$')


def stat_record(path):
    s=os.stat(path,follow_symlinks=False)
    return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,
        'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'mode':s.st_mode}


def audit_run(contract,index,queue_root,output_root,report_root,scratch_root='/tmp'):
    cp,ip,queue,output,reports=map(Path,(contract,index,queue_root,output_root,report_root))
    need(not reports.exists(),'audit evidence directory must be fresh')
    csha,isha=sha_file(cp),sha_file(ip);c=read_contract(cp,csha);idx=read_index(ip,isha)
    tree=output/c['version'];scratch=Path(scratch_root)
    need(not tree.is_symlink() and not output.is_symlink(),'output root symlink refused')
    need(tree.resolve()!=reports.resolve() and tree.resolve() not in reports.resolve().parents,
        'audit reports must be outside the versioned output tree')
    need(scratch.is_dir() and tree.resolve() not in scratch.resolve().parents,
        'local scratch must exist outside output tree')
    reports.mkdir(parents=True)
    errors=[];error_count=0;counts={'expected_files':0,'expected_files_seen':0,'directories_seen':0,
        'known_atomic_temporary_files':0,'unexpected_files':0,'unexpected_directories':0,
        'symlinks':0,'special_files':0,'stage_cache_subtrees_excluded':0}
    def error(record):
        nonlocal error_count
        error_count+=1
        if len(errors)<1000:errors.append(record)
    started=utcnow()
    try:
        with ProductionRunLock(queue):
            full=verify_run(cp,ip,queue,output,full=True)
            atomic_new(reports/'full_report.json',full)
            if not full['passed']:error({'stage':'full_report','error':'per-source full verification failed'})
            with tempfile.TemporaryDirectory(prefix='redimnet2-final-audit-',dir=scratch) as temp, ExitStack() as stack:
                db=sqlite3.connect(str(Path(temp)/'expected.sqlite'))
                stack.callback(db.close)
                db.execute('PRAGMA journal_mode=DELETE')
                db.execute('CREATE TABLE source_ids(source_id TEXT PRIMARY KEY)')
                db.execute('CREATE TABLE expected(path TEXT PRIMARY KEY,kind TEXT,source_id TEXT,sha TEXT,seen INTEGER DEFAULT 0)')
                db.execute('CREATE TABLE directories(path TEXT PRIMARY KEY)')
                sinks={};manifest_counts={k:0 for k in ('expected','inventory','temporaries')}
                for name in manifest_counts:
                    raw=stack.enter_context((reports/(name+'.jsonl.gz')).open('xb'))
                    sinks[name]=stack.enter_context(gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1))
                def emit(name,record):sinks[name].write(canonical_bytes(record));manifest_counts[name]+=1
                def expected(path,kind,source_id,digest):
                    db.execute('INSERT INTO expected(path,kind,source_id,sha) VALUES(?,?,?,?)',(path,kind,source_id,digest))
                    for parent in Path(path).parents:
                        if str(parent)!='.':db.execute('INSERT OR IGNORE INTO directories VALUES(?)',(parent.as_posix(),))
                expected('identity.json','root_identity',None,sha_file(tree/'identity.json'))
                sources=0
                for shard in idx['shards']:
                    p=safe_join(ip.parent,shard['path']);need(sha_file(p)==shard['sha256'],'source shard changed during final audit')
                    receipt=json.loads((queue/'completed'/(shard['id']+'.json')).read_text())
                    by_id={r['source_id']:r for r in receipt['sources']};rows=0
                    with p.open() as stream:
                        for line in stream:
                            source=json.loads(line);sid=source['source_id'];db.execute('INSERT INTO source_ids VALUES(?)',(sid,))
                            rel=source['sidecar_relpath'];safe_join(tree,rel);r=by_id[sid]
                            expected(rel,'sidecar',sid,r['sidecar_sha256'])
                            expected(rel+'.commit.json','commit',sid,r['commit_sha256'])
                            rows+=1;sources+=1
                    need(rows==shard['count'],'source shard denominator changed')
                    db.commit()
                need(sources==idx['source_count'],'source denominator differs')
                counts['expected_files']=db.execute('SELECT COUNT(*) FROM expected').fetchone()[0]
                for row in db.execute('SELECT path,kind,source_id,sha FROM expected ORDER BY path'):
                    emit('expected',dict(zip(('path','kind','source_id','sha256'),row)))
                # Each directory is scanned exactly once. Never follow symlinks.
                pending=[tree]
                while pending:
                    directory=pending.pop()
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            path=Path(entry.path);rel=path.relative_to(tree).as_posix()
                            record={'path':rel,'stat':stat_record(path)}
                            if entry.is_symlink():
                                counts['symlinks']+=1;record.update(kind='rejected_symlink',target=os.readlink(path))
                                error(record)
                            elif entry.is_dir(follow_symlinks=False):
                                counts['directories_seen']+=1
                                if rel=='.stage_cache':
                                    counts['stage_cache_subtrees_excluded']+=1
                                    record.update(kind='excluded_stage_cache',contents_verified=False)
                                else:
                                    known=db.execute('SELECT 1 FROM directories WHERE path=?',(rel,)).fetchone()
                                    record['kind']='expected_directory' if known else 'unexpected_directory'
                                    if not known:counts['unexpected_directories']+=1;error(record)
                                    pending.append(path)
                            elif entry.is_file(follow_symlinks=False):
                                known=db.execute('SELECT kind,source_id,sha FROM expected WHERE path=?',(rel,)).fetchone()
                                if known:
                                    record.update(kind=known[0],source_id=known[1],sha256_verified_by_full_report=known[2])
                                    db.execute('UPDATE expected SET seen=1 WHERE path=?',(rel,))
                                    counts['expected_files_seen']+=1
                                else:
                                    match=TEMP_PATTERN.fullmatch(path.name);target=None
                                    if match:
                                        target=path.with_name(match.group('target')).relative_to(tree).as_posix()
                                        known_target=db.execute('SELECT kind FROM expected WHERE path=?',(target,)).fetchone()
                                    else:known_target=None
                                    record['sha256']=sha_file(path)
                                    record['stat_after_hash']=stat_record(path)
                                    if record['stat_after_hash']!=record['stat']:error({'path':rel,'error':'uncommitted artifact changed during inventory'})
                                    if known_target:
                                        counts['known_atomic_temporary_files']+=1
                                        record.update(kind='retained_known_atomic_temporary',target_path=target,
                                            target_kind=known_target[0],published=False,deleted=False)
                                        emit('temporaries',record)
                                    else:
                                        counts['unexpected_files']+=1
                                        record['kind']='orphan_commit' if rel.endswith('.commit.json') else 'unexpected_file'
                                        error(record)
                            else:
                                counts['special_files']+=1;record['kind']='rejected_special_file';error(record)
                            emit('inventory',record)
                    db.commit()
                missing=db.execute('SELECT COUNT(*) FROM expected WHERE seen=0').fetchone()[0]
                if missing:
                    for row in db.execute('SELECT path,kind FROM expected WHERE seen=0 LIMIT 1000'):
                        error({'path':row[0],'kind':row[1],'error':'expected file missing in tree inventory'})
                    # Preserve actual count even when examples were truncated.
                    if missing>1000:error_count+=missing-1000
                manifests={name:{'path':str(reports/(name+'.jsonl.gz')),'entries':count}
                           for name,count in manifest_counts.items()}
            # gzip files are finalized before hashing, scratch DB already closed.
            for name,value in manifests.items():value.update(sha256=sha_file(value['path']),bytes=Path(value['path']).stat().st_size)
            result={'schema':'redimnet2-full-CPU-final-audit/v1','passed':error_count==0,
                'started_utc':started,'finished_utc':utcnow(),'tool_sha256':sha_file(__file__),
                'contract_path':str(cp),'contract_sha256':csha,'index_path':str(ip),'index_sha256':isha,
                'queue_root':str(queue),'versioned_output_tree':str(tree),
                'production_runtime_root':str(Path(sys.modules['contracts'].__file__).resolve().parent),
                'scratch_root':str(scratch),'scratch_local_filesystem_verified_by_tool':False,
                'execution_python':sys.executable,'execution_python_version':sys.version,
                'parent_lock_path':str(queue/'.parent.lock'),'same_parent_lock_held_for_entire_audit':True,
                'full_report':{'path':str(reports/'full_report.json'),'sha256':sha_file(reports/'full_report.json'),
                    'passed':full['passed']},'source_count':sources,'counts':counts,'manifests':manifests,
                'error_count':error_count,'errors':errors,'error_examples_truncated':error_count>len(errors),
                'known_atomic_temps_retained_without_admission':True,'deletion_or_repair_performed':False,
                'atomic_temp_classification_is_name_and_expected_target_protocol_not_authorship':True,
                'stage_cache_contents_explicitly_excluded':True,'stage_cache_integrity_claimed':False,
                'expected_file_bytes_reopened_by_full_report':True,
                'inventory_expected_file_SHA_from_full_report_bound_chunk_receipts':True,
                'expected_files_hashed_again_during_tree_inventory':False,
                'all_expected_and_unexpected_tree_entries_in_inventory_manifest':True,
                'input_audio_rehashed_in_this_audit':False,'human_word_boundary_accuracy_validated':False,
                'GPU_or_model_execution':False,'model_API_calls':0}
            atomic_new(reports/'final_audit.json',result)
            return result
    except BaseException:
        if not (reports/'audit_failure.json').exists():atomic_new(reports/'audit_failure.json',{
            'passed':False,'traceback':traceback.format_exc(),'started_utc':started,'finished_utc':utcnow(),
            'contract_sha256':csha,'index_sha256':isha,'deletion_or_repair_performed':False})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production-root',type=Path,default=_pre_args.production_root)
    for name in ('contract','shard-index','queue-root','output-root','report-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--scratch-root',default='/tmp')
    args=p.parse_args();result=audit_run(args.contract,args.shard_index,args.queue_root,args.output_root,args.report_root,args.scratch_root)
    print(json.dumps({'passed':result['passed'],'report':str(Path(args.report_root)/'final_audit.json'),
        'counts':result['counts'],'error_count':result['error_count']},ensure_ascii=False))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
