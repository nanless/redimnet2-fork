"""Render a reviewable W8 launcher; does not deploy, submit or run GPU.
An unexecuted launcher is not W2 admission. Runtime gate re-reads all pinned proofs.
"""
from pathlib import Path
import argparse,ast,json,shlex
from cloud_gate import Reader,contracts_and_bundle,issha,need,sha,EXPECTED_SOURCE_COUNT,SOURCE_SCOPE,verify
def child(path,parent):
    p=Path(path);return p.is_absolute() and '..' not in p.parts and Path(parent) in p.parents
def validate_spec(spec):
    need(spec['schema']=='ultra-eight-three-train-spec/v3','wrong spec schema')
    BASE=spec['cloud']['task_root']
    need(Path(BASE).is_absolute() and BASE not in {'/','/root'},'independent absolute task root required')
    c=spec['cloud'];run=c['run_root'];code=spec['code_root'];control=c['control_root']
    need(c['task_root']==BASE and all(child(x,BASE) for x in [run,code,control]),'cloud paths escape independent Ultra root')
    need(all(x!=y and not child(x,y) and not child(y,x) for x,y in [(run,code),(run,control),(control,code)]),'run/control/code overlap')
    need(not child(run,BASE+'/validation-v1') and not child(run,BASE+'/validation-v2'),'formal outputs overlap validation')
    need(c['priority']==6 and c['platform_workers']==1 and c['gpus_per_worker']==8 and c['max_runtime_seconds']==2592000,'P6/one platform worker/eight replicas/720h protection required')
    need(Path(c['python']).is_absolute(),'explicit absolute runtime Python required')
    need(Path(c['device_lock_root']).is_absolute() and c['device_lock_root']!=c['task_root'] and not child(c['device_lock_root'],c['task_root']) and not child(c['task_root'],c['device_lock_root']),'shared locks must stay outside per-task root')
    for key in ['bundle','pilot_contract','silver_contract','full_index','full_filter_summary','full_filter_source','full_filter_audit','full_filter_audit_source','diagnostic_protection_reference','supervisor','sequence_start','sequence_complete','independent_review']:
        need(Path(spec[key]['path']).is_absolute() and issha(spec[key]['sha256']),'concrete absolute path/SHA required: '+key)
    need(spec['source_scope']==SOURCE_SCOPE,'latest actual three-training-domain source scope required')
    for name in ['vertical','main','recovery']:
        stage=spec[name];need(child(stage['run_root'],BASE) and child(stage['resources'],stage['run_root']) and issha(stage['parent_result']['sha256']) and issha(stage['index']['sha256']),'concrete new W2 stage paths/SHA required')
        need(run!=stage['run_root'] and not child(run,stage['run_root']) and not child(stage['run_root'],run),'formal run overlaps W2 evidence')
    profiles={m['target']:m for m in c['mounts']};need(set(profiles)=={'/root','/shared/group-shared'} and len(c['mounts'])==2,'wrong persistent mounts')
    need(profiles['/root']['mode']=='rw' and profiles['/shared/group-shared']['mode']=='ro' and all(m['fstype']=='gpfs' and m['source'] for m in c['mounts']),'invalid persistent mount semantics')
    reader=Reader(spec);contracts_and_bundle(spec,reader)
    full=reader.ref(spec['full_index']);need(full['source_count']==EXPECTED_SOURCE_COUNT and full['domains']=={'voxceleb2':1092009,'voxblink2':9904382,'cnceleb2':524787} and full['RIR_or_MUSAN_or_Child_or_evaluation_in_this_job'] is False,'formal source index is not the new three-training-only scope')
    need(b'run-root' in reader.read(Path(code)/'gpu_launch.py') and b'device-lock-root' in reader.read(Path(code)/'gpu_launch.py'),'engine must accept explicit task and shared lock roots')
    return reader
def render(spec,spec_sha,inspection_map=None):
    inspection=dict(spec)
    if inspection_map is not None:inspection['evidence_path_map']=inspection_map
    validate_spec(inspection);BASE=spec['cloud']['task_root'];c=spec['cloud'];q=shlex.quote;control=c['control_root'];run=c['run_root'];py=c['python'];code=spec['code_root']
    command=[py,'-B',code+'/gpu_launch.py','--execute','--bundle-sha',spec['bundle']['sha256'],'--contract',spec['silver_contract']['path'],'--contract-sha',spec['silver_contract']['sha256'],'--shard-index',spec['full_index']['path'],'--index-sha',spec['full_index']['sha256'],'--queue-root',run+'/queue','--output-root',run+'/labels','--output',run+'/resources-attempt1','--run-root',BASE,'--device-lock-root',c['device_lock_root'],'--target-uuids','auto8','--memory-budget-mib','60000','--max-runtime-seconds','2592000']
    # All copied controls are independently SHA checked before execution.
    here=Path(__file__).resolve().parent
    pins={name:sha((here/name).read_bytes()) for name in ['cloud_gate.py','verify_cloud_mounts.py']}
    lines=['#!/bin/bash','set -euo pipefail','umask 077','unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_AUDIT','export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1','export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1','export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2',
       '# Preserve inherited RANK/WORLD_SIZE: frozen engine rejects duplicate/distributed parents.',
       'cd '+q(control)]
    for name,pin in pins.items():lines.append("printf '%s\\n' "+q(pin+'  '+name)+' | sha256sum --check --status')
    lines+=["printf '%s\\n' "+q(spec_sha+'  admission-spec.json')+' | sha256sum --check --status',
       q(py)+' -B '+q(control+'/cloud_gate.py')+' --live --spec '+q(control+'/admission-spec.json')+' --spec-sha '+q(spec_sha),
       q(py)+' -B '+q(control+'/verify_cloud_mounts.py')+' --spec '+q(control+'/admission-spec.json')+' --spec-sha '+q(spec_sha),
       '# Same semantic formal job has one persistent parent lock, independent of RUN spelling.',
       'mkdir -p '+q(BASE+'/.formal-parent-locks'),
       'exec 9> '+q(BASE+'/.formal-parent-locks/'+spec['silver_contract']['sha256']+'-'+spec['full_index']['sha256']+'.lock'),
       'flock -n 9',
       '# Persistent semantic job claim survives parent failure; fresh duplicate runs refuse.',
       '(set -o noclobber; printf \'%s\\n\' '+q(json.dumps({'schema':'ultra-three-train-semantic-job-claim/v1','run_root':run,'spec_sha256':spec_sha,'contract_sha256':spec['silver_contract']['sha256'],'index_sha256':spec['full_index']['sha256'],'status':'fresh_job_claimed_inspect_actual_task_before_any_retry'},sort_keys=True,separators=(',',':')))+' > '+q(BASE+'/.formal-parent-locks/'+spec['silver_contract']['sha256']+'-'+spec['full_index']['sha256']+'.claim.json')+')',
       '# mkdir without -p is the fresh-run guard; inspect/query any existing run before retry.',
       'mkdir '+q(run),
       'exec '+' '.join(q(x) for x in command)]
    return '\n'.join(lines)+'\n',pins
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--inspection-map',type=Path,help='Local inspection only; never included in deployed spec');p.add_argument('--verify-now',action='store_true');a=p.parse_args();raw=a.spec.read_bytes();need(sha(raw)==a.spec_sha,'spec SHA differs');spec=json.loads(raw);need(not spec.get('evidence_path_map'),'deployed spec must read actual server evidence paths');inspection_map=json.loads(a.inspection_map.read_text()) if a.inspection_map else None;script,pins=render(spec,a.spec_sha,inspection_map)
    inspection=dict(spec)
    if inspection_map:inspection['evidence_path_map']=inspection_map
    if a.verify_now:
        verify(inspection,Reader(inspection))
    a.output_dir.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
    for name in pins:(a.output_dir/name).write_bytes((here/name).read_bytes())
    (a.output_dir/'admission-spec.json').write_bytes(raw);(a.output_dir/'launch-ultra-w8.sh').write_text(script)
    plan={'schema':'ultra-eight-three-train-platform-preparation/v2','status':'prepared_only_not_submitted','W2_gate_executed_here':a.verify_now,'runtime_W2_gate_mandatory':True,'spec_sha256':a.spec_sha,'priority':6,'platform_workers':1,'gpus_per_worker':8,'independent_inference_replicas':8,'training_DDP_used':False,'protective_max_runtime_seconds':2592000,'protective_max_runtime_is_ETA':False,'task_root':spec['cloud']['task_root'],'run_root':spec['cloud']['run_root'],'mounts':spec['cloud']['mounts'],'full_source_count':EXPECTED_SOURCE_COUNT,'full_shard_count':len(Reader(inspection).ref(spec['full_index'])['shards']),'full_index_sha256':spec['full_index']['sha256'],'source_scope':SOURCE_SCOPE,'outside_scope':['Child','evaluation','MUSAN','RIR','business','diagnostics datasets'],'diagnostic_original_train_sources':'kept for silver annotation only with separate protection reference; no training release','formal_API_calls':0,'human_gold_claimed':False,'platform_submission_required_readback':['actual priority6','one platform worker eight A800','image/queue/zone exact values','user-root RW and group-root RO mounts','single parent command','unique task/run lookup before retry after unknown submit outcome'],'resume_policy':'No implicit rerun: inspect actual task/parent/queue identity; derive explicit audited resume command if interrupted','files_sha256':{name:sha(pth.read_bytes()) for name,pth in ((p.name,p) for p in a.output_dir.iterdir())}}
    (a.output_dir/'platform-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(json.dumps(plan,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
