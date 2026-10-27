"""有界比较旧完成片的串行、线程和spawn进程纯读；不是准入或GPU测速。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import union_coverage as U
from prepare_speed_continuation import read_shard
from contracts import digest

SCHEMA='annotation-retained-read-benchmark/v1'
MODES={'serial1':('thread',1),'thread2':('thread',2),'thread4':('thread',4),
       'process2':('process',2),'process4':('process',4)}
MAX_CHUNKS=8
MAX_SOURCES=2048


def benchmark(plan_path,plan_sha,positions=(0,),modes=('serial1',),*,purpose='production',max_sources=1024):
    """固定同一小批完成片；每种模式只建一次pool，不创建/claim任何生产队列。"""
    U.need(purpose in {'production','cpu_fixture'},'explicit production/cpu_fixture purpose required')
    U.need(isinstance(positions,(list,tuple))and 0<len(positions)<=MAX_CHUNKS and
        all(type(i)is int and i>=0 for i in positions)and len(set(positions))==len(positions),'bounded distinct retained positions required')
    U.need(isinstance(modes,(list,tuple))and 0<len(modes)<=8 and all(type(m)is str and m in MODES for m in modes),'unknown or too many benchmark modes')
    U.need(type(max_sources)is int and 1<=max_sources<=MAX_SOURCES,'max_sources must be integer in 1..2048')
    plan=U.ref({'path':str(plan_path),'sha256':plan_sha})
    U.need(plan.get('schema')=='annotation-speed-continuation-plan/v1'and
        plan.get('status')=='final_prepared_not_started','fixed final preparation plan required')
    index=U.ref(plan['old_index']);retained=plan['retained_complete_shards']
    U.need(all(i<len(retained)for i in positions),'retained position outside plan')
    selected=[retained[i]for i in positions]
    U.need(sum(s['count']for s in selected)<=max_sources,'selected source count exceeds explicit benchmark limit')
    for name in modes:
        backend,workers=MODES[name]
        U.retained_read_backend({'retained_read_backend':backend,'retained_read_workers':workers,'purpose':purpose})
    old={'id':'readonly-benchmark-old','contract':{k:plan['old_contract'][k]for k in ('path','sha256')},
         'queue_root':plan['old_queue'],'sidecar_tree':plan['old_sidecar_tree']}
    contract,queue,_=U.producer(old,index_pin=plan['old_index'])
    descriptor_lookup={s['id']:(ordinal,s)for ordinal,s in enumerate(index['shards'])}
    U.need(len(descriptor_lookup)==len(index['shards']),'duplicate parent shard IDs')
    chunks=[]
    for item in selected:
        U.need(item['id']in descriptor_lookup,'retained shard missing from parent index')
        ordinal,descriptor=descriptor_lookup[item['id']]
        U.need(all(item.get(k)==v for k,v in descriptor.items()),'retained descriptor differs from parent')
        U.need(item.get('chunk_receipt_path')==str(queue/'completed'/(descriptor['id']+'.json')),'retained chunk receipt path differs')
        rows=read_shard(Path(plan['old_index']['path']).parent,descriptor)
        U.need(len(rows)==descriptor['count']and len({s['source_id']for s in rows})==len(rows),'source count/unique identity differs')
        U.need(digest(sorted(s['source_id']for s in rows))==item['source_ids_sha256'],'source-ID digest differs')
        chunks.append((item,ordinal,descriptor,rows))
    broker=queue/'broker.json';broker_before=U.sha_file(broker)
    U.need(broker_before==plan['old_progress_snapshot']['broker_snapshot_sha256'],'old broker differs from prepared plan')
    reports=[];expected=None
    for attempt,name in enumerate(modes):
        U.need(U.sha_file(broker)==broker_before,'old broker changed before benchmark mode')
        backend,workers=MODES[name];stats={'max_submitted_batch':0,'old_completed_source_reads':0}
        proofs=[];child_before=resource.getrusage(resource.RUSAGE_CHILDREN);cpu_before=time.process_time();wall=time.perf_counter()
        with U.retained_read_pool(workers,backend,producer_value=old,contract=contract,purpose=purpose)as executor:
            for item,ordinal,descriptor,rows in chunks:
                for source,rs,cs,decision in U._complete_chunk(plan['old_index']['sha256'],descriptor,ordinal,queue,old,rows,
                        item['chunk_receipt_sha256'],item['completion_slot_sha256'],contract=contract,
                        read_executor=executor,read_stats=stats,read_backend=backend):
                    proofs.append({'source_id':source['source_id'],'source_record_sha256':digest(source),
                        'sidecar_sha256':rs,'commit_sha256':cs,'decision':decision})
        elapsed=time.perf_counter()-wall;cpu=time.process_time()-cpu_before;child_after=resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu=(child_after.ru_utime-child_before.ru_utime)+(child_after.ru_stime-child_before.ru_stime)
        U.need(U.sha_file(broker)==broker_before,'old broker changed during benchmark mode')
        proof_sha=digest(proofs)
        if expected is None:expected=proof_sha
        U.need(proof_sha==expected,'ordered source proofs differ across execution modes')
        reports.append({'attempt_index':attempt,'mode':name,'backend':backend,'workers':workers,
            'effective_executor':'serial'if workers==1 else backend+'_pool','wall_seconds':elapsed,
            'main_process_cpu_seconds':cpu,'reaped_children_cpu_seconds':child_cpu,
            'cpu_measurement_scope':'process_time plus RUSAGE_CHILDREN delta after shutdown; includes startup; not per-job CPU',
            'source_count':len(proofs),'ordered_proofs_sha256':proof_sha,'ordered_proofs':proofs,
            'max_submitted_batch':stats['max_submitted_batch'],'completed_source_reads':stats['old_completed_source_reads']})
    U.need(U.sha_file(broker)==broker_before,'old broker changed after benchmark')
    U.need(U.sha_file(plan_path)==plan_sha,'plan changed during benchmark')
    U.need(U.sha_file(plan['old_index']['path'])==plan['old_index']['sha256'],'parent index changed during benchmark')
    U.need(U.sha_file(old['contract']['path'])==old['contract']['sha256'],'producer contract changed during benchmark')
    return {'schema':SCHEMA,'purpose':purpose,'plan':{'path':str(plan_path),'sha256':plan_sha},
        'old_index':plan['old_index'],'old_contract':old['contract'],'retained_positions':list(positions),
        'selected_shards':[s[0]['id']for s in chunks],'source_limit':max_sources,'modes':reports,
        'all_ordered_proofs_identical':True,'broker_sha256_before':broker_before,'broker_sha256_after':U.sha_file(broker),
        'old_parent_lock_acquired':False,'scope':'bounded_readonly_benchmark_not_union_final_or_execution_lease',
        'process_lifecycle_platform':(('linux_prctl_in_initializer'if sys.platform=='linux'else'unsupported_nonlinux_cpu_fixture')
            if any(MODES[m][0]=='process'for m in modes)else'not_used'),
        'models_executed':False,'audio_read':False,'queue_created':False,'GPU_runtime_verified':False,
        'formal_runtime_admitted':False,'validated_GPU_speedup':None,
        'limitations':['Read-only file proof comparison; no writer exclusion or platform authentication.',
            'Actual wall times include current filesystem cache/order/startup; no automatic speedup claim.',
            'Non-Linux CPU fixture does not verify Linux parent-death behavior.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--positions',default='0',help='retained_complete_shards positions; at most 8 and source limit applies')
    p.add_argument('--modes',default='serial1',help='comma-separated serial1/thread2/thread4/process2/process4; repeat for cache-order checks')
    p.add_argument('--max-sources',type=int,default=1024);p.add_argument('--purpose',choices=('production','cpu_fixture'),default='production')
    a=p.parse_args()
    try:positions=[int(x)for x in a.positions.split(',')]
    except ValueError:p.error('positions must be comma-separated nonnegative integers')
    print(json.dumps(benchmark(a.plan,a.plan_sha,positions,a.modes.split(','),purpose=a.purpose,max_sources=a.max_sources),ensure_ascii=False))


if __name__=='__main__':main()
