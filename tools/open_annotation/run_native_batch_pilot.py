"""独立原生ASR批处理验证：同批1/4/8、完整读回和512缓存恢复。

旧单窗driver和比较器保持原义。共享的SerialExecutor只负责本次自有
GPU父进程的精确PID/start/pidfd清理；各轨始终先排空再进入下一轨。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from contracts import atomic_new, need, safe_join, asr_batch_profile
from prepare_speed_validation import check_index
from run_bounded_speed_pilot import (physical, disjoint, pin, file_identity, argv_fields,
                                     SerialExecutor, verify_parent, snapshot_recovery)
SCHEMA='annotation-native-batch-pilot/v1'
PROFILES=tuple('native-fp32-sdpa-b'+str(n)for n in (1,4,8))

def load_matrix(ref,count,*,cpu_fixture=False):
    matrix=pin(ref);root=physical(ref['path']).parent
    need(matrix.get('schema')=='annotation-native-batch-prepared/v1' and matrix.get('source_count')==count,
         'wrong finite prepared denominator/schema')
    need(matrix.get('execution_order')==list(PROFILES) and set(matrix.get('variants',{}))==set(PROFILES),
         'exact native batch1/4/8 order required')
    need(matrix.get('GPU_executed')is False and matrix.get('commands_executed')is False
         and matrix.get('database_created')is False,'preparation is not execution proof')
    need(matrix.get('worker_count') == 2,'independent native pilot requires exactly two workers')
    index_rows=check_index(matrix['index'],count)
    result={'matrix':matrix,'root':root,'rows':index_rows,'variants':{}}
    for name in PROFILES:
        v=matrix['variants'][name];code=physical(v['freeze']['frozen_copy']);track_path=physical(v['track']['path'])
        need(code==root/name/'code' and track_path==root/name/'native-track.json','prepared directory layout differs')
        bundle_ref={'path':str(code/'bundle_files.json'),'sha256':v['freeze']['bundle_sha256']}
        members=pin(bundle_ref)
        need(isinstance(members,dict) and all(k in members for k in ('gpu_launch.py','report.py','worker.py','audit_native_transformers_batch.py')),
             'frozen executable bundle incomplete')
        for rel,digest in members.items():
            path=physical(safe_join(code,rel));need(path.is_file() and file_identity(path)['sha256']==digest,'frozen code member changed')
        if not cpu_fixture:
            for module in ('gpu_launch.py','report.py','audit_native_transformers_batch.py'):
                need(members[module]==file_identity(HERE/module)['sha256'],'driver and frozen pilot control source differ: '+module)
        track=pin(v['track']);contract=pin(track['contract'])
        need(track.get('schema')=='redimnet2-annotation-native-batch-track/v1' and track['index']==matrix['index']
             and track['contract']==v['freeze']['contracts']['pilot_validation'],'track inputs differ')
        need(contract.get('purpose')=='pilot_validation' and contract.get('model_API_calls_allowed')==0
             and contract.get('release_training_labels')is False,'only offline pilot contracts permitted')
        need(physical(track['contract']['path']).parent==code/'contracts','contract outside frozen code')
        need(contract['runtime'].get('asr_compute')=={'dtype':'float32','attention':'sdpa'},
             'native pilot must hold ASR precision and attention fixed')
        batch=asr_batch_profile(contract)
        need(batch is not None and batch['max_requests']==int(name.rsplit('b',1)[1]),
             'explicit native ASR cohort policy/width differs')
        need(track.get('execution_policy')=='asr_fixed_cohort_fa_single/v1', 'native track policy differs')
        run=root/name/'run'
        need(track['queue_root']==str(run/'queue') and track['output_root']==str(run/'labels')
             and track['resources_root']==str(run/'resources-attempt1'),'finite run roots differ')
        argv=v['argv'];fields=argv_fields(argv)
        need(physical(argv[0]).is_file() and argv[1]=='-B' and physical(argv[2])==code/'gpu_launch.py',
             'only frozen gpu_launch argv permitted')
        expected={'--execute':True,'--bundle-sha':bundle_ref['sha256'],'--contract':track['contract']['path'],
                  '--contract-sha':track['contract']['sha256'],'--shard-index':matrix['index']['path'],
                  '--index-sha':matrix['index']['sha256'],'--queue-root':track['queue_root'],
                  '--output-root':track['output_root'],'--output':track['resources_root'],
                  '--run-root':str(root/name),'--full-verify':True}
        need(all(fields.get(k)==x for k,x in expected.items()) and set(fields)==set(expected)|{
            '--target-uuids','--device-lock-root','--max-runtime-seconds'},'frozen launcher flags differ')
        cap=1800 if count==30 else 7200
        need(fields['--max-runtime-seconds'].isdigit() and 60<=int(fields['--max-runtime-seconds'])<=cap,
             'phase runtime exceeds bounded pilot budget')
        need(fields['--target-uuids']=='auto2' or len(fields['--target-uuids'].split(','))==2,
             'unsupported pilot card selection')
        need(matrix['worker_count']==2 and fields['--target-uuids']!='auto8','prepared worker count/card selection differs')
        physical(fields['--device-lock-root'])
        result['variants'][name]={'argv':list(argv),'fields':fields,'track':track,'track_ref':v['track'],
                                  'contract':contract,'code':code,'members':members}
    first=result['variants'][PROFILES[0]]
    # 在任何模型执行前拒绝混入不同模型、窗口、Ultra或生成规则；
    # 不能花完GPU时间才发现这不是受控的批宽对照。
    from audit_native_transformers_batch import compare_contracts
    for v in result['variants'].values():
        compare_contracts(first['contract'],v['contract'],'speed')
        for key in ('--target-uuids','--device-lock-root'):
            need(v['fields'][key]==first['fields'][key],'different device group across finite profiles')
        need(v['argv'][0]==first['argv'][0],'interpreter differs across profiles')
        need(v['members']==first['members'],'executable source bundle differs across profiles')
    return result


def compile_plan(plan_ref,output):
    plan=pin(plan_ref)
    need(plan.get('schema')==SCHEMA and plan.get('purpose') in ('production','cpu_fixture'),'wrong bounded pilot schema/purpose')
    need(plan.get('model_API_calls_allowed')==0 and plan.get('production_switch_allowed')is False,
         'pilot cannot release production or API use')
    budget=plan.get('wall_budget_seconds',21000)
    need(type(budget)is int and 900<=budget<=21600,'overall pilot budget must be 900..21600 seconds')
    matrices={key:load_matrix(plan[key+'_prepared'],count,cpu_fixture=plan['purpose']=='cpu_fixture')for key,count in [('screen',30),('main',3000)]}
    screen,main=matrices['screen'],matrices['main']
    need(all(main['rows'].get(k)==r for k,r in screen['rows'].items()),'screen must be exact main source subset')
    for name in PROFILES:
        a,b=(matrices[k]['variants'][name]for k in ('screen','main'))
        need(a['contract']==b['contract'] and a['members']==b['members'],'screen/main software or model contract differs')
        for key in ('--target-uuids','--device-lock-root'):
            need(a['fields'][key]==b['fields'][key],'screen/main resource group differs')
        need(a['argv'][0]==b['argv'][0],'screen/main interpreter differs')
    output=physical(output)
    need(not output.exists() and output.parent.is_dir() and all(disjoint(output,m['root'])for m in matrices.values()),
         'new external evidence output required')
    repo=next((p for p in (HERE,*HERE.parents)
               if (p/'AGENTS.md').is_file() and (p/'redimnet2').is_dir()),None)
    control_root=repo if repo is not None else HERE
    # 计划常放在/P/plan.json，而输出为/P/evidence；不能把共同祖先/P
    # 错当仓库。保护计划文件本身、实际源码和有限输入目录即可。
    input_roots=[physical(plan_ref['path']),control_root]
    for m in matrices.values():
        input_roots.append(physical(m['matrix']['index']['path']).parent)
        input_roots.extend(physical(v['fields']['--device-lock-root'])for v in m['variants'].values())
    if plan.get('recovery_index')is not None:
        input_roots.append(physical(plan['recovery_index']['path']).parent)
    need(all(disjoint(output,p)for p in input_roots), 'evidence output overlaps control code, plan, inputs or locks')
    need(disjoint(screen['root'],main['root']),'screen/main trees must be independent')
    protected=plan.get('protected_roots')
    need(isinstance(protected,list) and 1<=len(protected)<=16,'explicit current production/PTN roots required')
    need(all(disjoint(output,physical(q)) and all(disjoint(m['root'],physical(q))for m in matrices.values())for q in protected),
         'pilot evidence/compute tree overlaps protected production or PTN')
    recovery=None
    if plan.get('recovery_index')is not None:
        index=pin(plan['recovery_index']);count=index.get('source_count')
        need(type(count)is int and count in (500,512),'recovery must explicitly contain 500 or 512 sources')
        rows=check_index(plan['recovery_index'],count)
        need(all(main['rows'].get(k)==r for k,r in rows.items()),'recovery source records differ from main')
        profiles=plan.get('recovery_profiles',list(PROFILES))
        need(isinstance(profiles,list) and profiles and all(isinstance(x,str)for x in profiles)
             and len(profiles)==len(set(profiles)) and set(profiles)<=set(PROFILES),'unknown recovery profiles')
        recovery={'index':plan['recovery_index'],'rows':rows,'profiles':profiles,'count':count}
    return {'plan':plan,'plan_ref':plan_ref,'budget':budget,'matrices':matrices,'recovery':recovery,'output':output}


def audited_track(v):
    from audit_native_transformers_batch import load_track
    return load_track(v['track_ref']['path'],v['track_ref']['sha256'])


def save_native_readback(output,name,track):
    """保存刚完成的有限完整读回，不为准入反复重开3000侧车。"""
    from audit_native_transformers_batch import native_readback_receipt
    return atomic_new(output/(name+'-native-readback.json'),native_readback_receipt(track))


def run(plan_ref,output,execute=False,*,_fixture_executor=None):
    compiled=compile_plan(plan_ref,output)
    if not execute:return dict(checked=True,GPU_executed=False,database_created=False,source_counts=[30,3000],recovery_count=compiled['recovery']['count']if compiled['recovery']else 0)
    fixture=compiled['plan']['purpose']=='cpu_fixture'
    need(_fixture_executor is None or fixture,'custom executors prohibited for real pilot')
    need(not fixture or _fixture_executor is not None,'cpu_fixture execution requires fake executor; real commands refused')
    need(fixture or platform.system()=='Linux','real pilot requires Linux pidfd resources')
    need(int(os.environ.get('WORLD_SIZE','1'))==1 and int(os.environ.get('RANK','0'))==0 and 'LOCAL_RANK'not in os.environ,
         'driver must run once outside DDP ranks')
    out=compiled['output'];out.mkdir(mode=0o700)
    cancel=threading.Event();old_handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):old_handlers[sig]=signal.signal(sig,lambda s,f:cancel.set())
    deadline=time.monotonic()+compiled['budget'];executor=_fixture_executor or SerialExecutor(out,deadline,cancel)
    result={'schema':'annotation-native-batch-pilot-result/v1','plan':plan_ref,'purpose':compiled['plan']['purpose'],
            'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'steps':[],'database_created':False,'production_switch_allowed':False,'human_accuracy_certified':False,
            'GPU_runtime_verified':False,'all_finite_audits_completed':False,
            'native_ASR_batch_pilot_only':True}
    try:
        atomic_new(out/'effective-plan.json',compiled['plan'])
        from audit_native_transformers_batch import analyze
        for stage in ('screen','main'):
            matrix=compiled['matrices'][stage];tracks={}
            for name in PROFILES:
                # 每条命令前重新核固定准备和全部冻结源，拒执行中被改写。
                need(pin(plan_ref)==compiled['plan'],'outer pilot plan changed before execution')
                matrix=load_matrix(compiled['plan'][stage+'_prepared'],30 if stage=='screen' else 3000,cpu_fixture=fixture)
                v=matrix['variants'][name];executor.command(v['argv'],stage+'-'+name,v['code'])
                need(not cancel.is_set(),'pilot cancelled after current parent cleanup')
                returned=verify_parent(v);tracks[name]=audited_track(v)
                save_native_readback(out,stage+'-'+name,tracks[name])
                result['steps'].append(dict(stage=stage,profile=name,**returned,source_count=len(tracks[name]['records'])))
                atomic_new(out/(stage+'-'+name+'-verified.json'),result['steps'][-1])
            for name in PROFILES[1:]:
                comparison=analyze(tracks[PROFILES[0]],tracks[name],mode='speed')
                atomic_new(out/(stage+'-compare-'+name+'.json'),comparison)
                need(comparison['all_planned_sources_accounted']and comparison['cold_speed_ratio_valid'],
                     'finite comparison incomplete or true cold measurement unavailable')
        recovery=compiled['recovery']
        if recovery:
            for name in recovery['profiles']:
                need(pin(plan_ref)==compiled['plan'],'outer pilot plan changed before recovery')
                main=load_matrix(compiled['plan']['main_prepared'],3000,cpu_fixture=fixture);v=main['variants'][name]
                before=snapshot_recovery(v,recovery['rows']);root=main['root']/name/('recovery'+str(recovery['count']))
                # 不复制或改写主结果；新队列只复用同合同已发布完整侧车。
                track=dict(v['track'],index=recovery['index'],queue_root=str(root/'queue'),resources_root=str(root/'resources-attempt1'))
                track_path=out/('recovery-'+name+'-track.json');track_sha=atomic_new(track_path,track)
                flags=dict(v['fields']);flags.update({'--shard-index':recovery['index']['path'],'--index-sha':recovery['index']['sha256'],
                    '--queue-root':track['queue_root'],'--output':track['resources_root'],'--max-runtime-seconds':'1800'})
                argv=list(v['argv'][:3])
                for k,value in flags.items():argv.extend([k]if value is True else[k,value])
                rv=dict(v,track=track,track_ref={'path':str(track_path),'sha256':track_sha},fields=flags)
                executor.command(argv,'recovery-'+name,v['code']);need(not cancel.is_set(),'pilot cancelled after recovery parent cleanup');returned=verify_parent(rv);resumed=audited_track(rv)
                save_native_readback(out,'recovery-'+name,resumed)
                baseline=audited_track(v);comparison=analyze(baseline,resumed,mode='recovery')
                atomic_new(out/('recovery-'+name+'-comparison.json'),comparison)
                need(snapshot_recovery(v,recovery['rows'])==before,'published finite results changed during recovery')
                totals=resumed['performance']['totals']
                need(comparison['strict_recovery_native_results_equal']and totals['sidecars_committed']==0
                     and totals['sidecar_cache_hits']==recovery['count']and totals['asr_rows']==totals['aligner_rows']==0,
                     'recovery reran model or changed native result')
                result['steps'].append(dict(stage='recovery',profile=name,sources=recovery['count'],new_ASR_calls=0,new_FA_calls=0,**returned))
        need(not cancel.is_set(),'pilot cancelled before final result')
        result['all_finite_audits_completed']=True
        atomic_new(out/'result.json',result);return result
    except BaseException:
        atomic_new(out/'failure.json',dict(traceback=traceback.format_exc(),completed_steps=result['steps'],no_next_step_started=True))
        raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();print(json.dumps(run({'path':str(a.plan),'sha256':a.plan_sha},a.output,a.execute),ensure_ascii=False))

if __name__=='__main__':main()
