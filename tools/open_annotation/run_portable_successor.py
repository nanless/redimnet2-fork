"""用固定的内容/联合准备证明启动旧计算方式的新八卡标注生产者。

默认只核元数据。旧生产者永久保留；运行时分别取得当前节点的旧语义锁和
旧队列锁，不把准备节点的device号当当前租约，也不重跑跨节点严格union。
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import traceback
from urllib.parse import quote

if __package__:
    from .resume_existing_annotation import (physical, ref, pinned_file, file_sha, write_new,
        existing_lock, frozen_bundle, frozen_roots, ref_unpinned, legacy_media_path)
    from .readonly_progress import inspect_progress
    from .freeze_legacy_successor import BUNDLE_SHA256, ENGINE_MODULES, LEGACY_RUNTIME
    from .verify_portable_successor_mounts import validate_output_namespaces,attestation_field
    from ..task_names import validate_task_name
else:
    from resume_existing_annotation import (physical, ref, pinned_file, file_sha, write_new,
        existing_lock, frozen_bundle, frozen_roots, ref_unpinned, legacy_media_path)
    from readonly_progress import inspect_progress
    from freeze_legacy_successor import BUNDLE_SHA256, ENGINE_MODULES, LEGACY_RUNTIME
    from verify_portable_successor_mounts import validate_output_namespaces,attestation_field
    if not (Path(__file__).resolve().parent/'task_names.py').is_file():
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from task_names import validate_task_name

SCHEMA='annotation-portable-successor-run/v1'
SCHEMA_V2='annotation-portable-successor-run/v2'
ROLE_FINALIZER_SHA256='1af4f2f93590b6594785a2cb493ec8a2ddfa057e6269661cd7406a13ecba5fd2'
COUNT=11521178
DOMAINS={'voxceleb2':1092009,'voxblink2':9904382,'cnceleb2':524787}


def need(value,message):
    if not value:raise ValueError(message)


def evidence_file(value):
    need(isinstance(value,dict) and set(value)in ({'path','sha256'},{'path','sha256','count'}),
         'explicit evidence file ref required')
    if 'count'in value:need(type(value['count'])is int and value['count']>=0,'evidence count differs')
    return pinned_file({k:value[k]for k in ('path','sha256')})


def profile(old,new):
    need(all(isinstance(c.get('version'),str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',c['version'])
             and c['version']not in {'.','..'} for c in (old,new)), 'producer version must be a safe single directory component')
    need(old.get('schema')==new.get('schema')=='redimnet2-production-contract/v2' and
         old.get('version')!=new.get('version'),'distinct valid producer versions required')
    need({k:v for k,v in old.items()if k!='version'}=={k:v for k,v in new.items()if k!='version'},
         'only producer version may differ; old computation/quality/source rules are immutable')


def terminal(value,old,*,fresh=False):
    need(value.get('schema')=='annotation-writer-terminal-observation/v1' and
         value.get('task_id')==old['old_task_id'] and value.get('queue_root')==old['queue_root'] and
         value.get('state')in {'Failed','Stopped','Succeeded','Completed'} and
         type(value.get('active_workers'))is int and value['active_workers']==0,
         'bound original platform terminal observation required')
    workers=value.get('workers');need(isinstance(workers,list)and len(workers)==1 and
        workers[0].get('name')==old['old_task_id']+'-worker-0'and
        workers[0].get('state')in {'Failed','Stopped','Succeeded','Completed'},'old worker terminal identity differs')
    observed=datetime.datetime.fromisoformat(value['observed_utc'])
    need(observed.tzinfo is not None,'terminal timezone required')
    age=(datetime.datetime.now(datetime.timezone.utc)-observed).total_seconds()
    need(age>=0 and (not fresh or age<=3600),'terminal observation future/stale')


def platform_observation(pin,old):
    value=ref(pin)
    need(value.get('schema')=='annotation-portable-platform-observation/v1' and
         value.get('old_task_id')==old['old_task_id'] and value.get('competing_successor_tasks')==[] and
         value.get('remote_proc_enumeration_performed')is False and
         value.get('evidence_basis')=='platform_terminal_and_current_shared_locks',
         'explicit fresh platform/absence observation required; no invented cross-node process enumeration')
    terminal(value['original_terminal'],old,fresh=True)
    need(value['observed_utc']==value['original_terminal']['observed_utc'],'platform observation times differ')
    for prior in value.get('prior_attempts',[]):
        need(isinstance(prior.get('task_id'),str)and prior.get('state')in {'Failed','Stopped','Succeeded','Completed'}and
             type(prior.get('active_workers'))is int and prior['active_workers']==0,
             'prior successor still active/unobserved')
    return value


def check_database(pin,union,attestation):
    """仅核固定计划数据库的元数据，不把plan数据库当完整标签终验。"""
    p=pinned_file(pin)
    db=sqlite3.connect('file:'+quote(str(p))+'?mode=ro&immutable=1',uri=True)
    try:
        meta={k:json.loads(v)for k,v in db.execute('SELECT key,value FROM metadata')}
        need(meta.get('spec')==union and meta.get('spec_sha256')==attestation['spec_sha256'],
             'fixed plan database spec binding differs')
        expected={k:v for k,v in attestation.items()if k!='consumer_index'}
        need(meta.get('validation')==expected,'fixed plan database validation differs')
        columns={row[1]:row for row in db.execute('PRAGMA table_info(sources)')}
        need(set(columns)=={'sid','source_sha','shard','row_number','relative','selected','record_sha','commit_sha','decision'} and
             columns['sid'][5]==1 and sum(bool(row[5])for row in columns.values())==1,
             'plan database lacks the real unique-source table')
        oldid=union['old_producer']['id'];newid=union['new_producer']['id']
        # 两个合法生产者用一次全表聚合核数，避免千万行GROUP BY建立临时排序表。
        # 两组要实际存在且按字节精确匹配；NULL、未知生产者和额外行仍拒绝。
        total,oldcount,newcount=db.execute('''SELECT COUNT(*),
            COALESCE(SUM(CASE WHEN selected COLLATE BINARY=? THEN 1 ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN selected COLLATE BINARY=? THEN 1 ELSE 0 END),0) FROM sources''',
            (oldid,newid)).fetchone()
        need(oldcount>0 and newcount>0 and oldcount==attestation['retained_source_count'] and
             newcount==attestation['remaining_source_count'] and
             total==oldcount+newcount,
             'plan database source/producer partition differs from full attestation')
        malformed=db.execute('''SELECT COUNT(*) FROM sources WHERE
            typeof(source_sha)!='text' OR typeof(sid)!='text' OR typeof(selected)!='text'
            OR source_sha IS NULL OR length(source_sha)!=64 OR source_sha GLOB '*[^0-9a-f]*'
            OR sid IS NULL OR length(sid)!=68 OR substr(sid,1,4)!='src_'
            OR substr(sid,5) GLOB '*[^0-9a-f]*'
            OR (selected=? AND (record_sha IS NULL OR commit_sha IS NULL
                OR typeof(record_sha)!='text' OR typeof(commit_sha)!='text' OR typeof(decision)!='text'
                OR length(record_sha)!=64 OR length(commit_sha)!=64
                OR record_sha GLOB '*[^0-9a-f]*' OR commit_sha GLOB '*[^0-9a-f]*'
                OR decision IS NULL OR decision NOT IN ('candidate','unknown','failed')))
            OR (selected=? AND (record_sha IS NOT NULL OR commit_sha IS NOT NULL OR decision IS NOT NULL))''',
            (oldid,newid)).fetchone()[0]
        need(malformed==0,'plan database has invalid retained proofs or falsely completed new sources')
    finally:db.close()
    need(file_sha(p)==pin['sha256'],'plan database changed during metadata read')


def verify_python(old):
    target=Path(old['python']).resolve(strict=True)
    need(target==pinned_file(old['python_binary']) and target.parent==Path(old['python']).parent,
         'original interpreter bytes or symlink changed')
    return target


def check_preparation(spec,old,oldindex,union,plan,remaining):
    """复用已完整完成的准备证明，不重建union库或重读百万旧侧车。"""
    receipt=ref(spec['preparation_attestation'])
    need(receipt.get('schema')=='annotation-remaining-role-finalize/v1'and
         receipt.get('new_plan')==union['continuation_plan']and
         receipt.get('new_remaining_index')==union['new_producer']['index']and
         receipt.get('parent_full_index')==old['full_index']and
         receipt.get('inventory_ref')==union['portable_sidecar_inventory_ref'],
         'completed preparation receipt refs differ')
    for key in ('source_shard_bytes_changed','v1_bytes_rewritten','source_media_read','models_executed','formal_runtime_admitted','GPU_executed'):
        need(receipt.get(key)is False,'preparation receipt may not rewrite/reinfer/admit GPU: '+key)
    oldplan=ref(receipt['old_plan']);oldremaining=ref(receipt['old_remaining_index'])
    need(oldplan.get('schema')=='annotation-speed-continuation-plan/v1'and
         oldplan.get('status')=='final_prepared_not_started'and
         oldplan.get('old_index')==old['full_index']and
         {k:oldplan.get('old_contract',{}).get(k)for k in ('path','sha256')}==old['contract']and
         oldplan.get('old_queue')==old['queue_root']and oldplan.get('remaining_index')==receipt['old_remaining_index'],
         'original completed preparation lineage differs')
    for key in ('formal_runtime_admitted','models_executed','old_queue_mutated','old_results_rewritten','stage_cache_relabelled'):
        need(oldplan.get(key)is False,'original preparation was not immutable/offline: '+key)
    need(type(oldplan.get('new_inference_count'))is int and oldplan['new_inference_count']==0 and
         oldplan.get('unclaimed_sidecars_scanned')is True,'final original preparation must have covered unclaimed sources')
    total=oldindex['source_count'];left=plan['remaining_source_count'];kept=plan['retained_source_count']
    need(all(type(v)is int and v>=0 for v in (total,left,kept))and left>0 and kept+left==total and
         type(receipt.get('remaining_sources_checked'))is int and receipt['remaining_sources_checked']==left==remaining['source_count'],
         'completed preparation source counts differ')
    roles=('train_ptn','train_lm');groups={}
    for key,bound in (('parent_role_union_counts',total),('remaining_role_union_counts',left),('retained_role_union_counts',kept)):
        group=receipt.get(key)
        need(isinstance(group,dict)and set(group)==set(roles)and
             all(type(v)is int and 0<=v<=bound for v in group.values())and group['train_ptn']==bound,
             'completed preparation role counts differ: '+key);groups[key]=group
    parent_roles=groups['parent_role_union_counts'];left_roles=groups['remaining_role_union_counts'];kept_roles=groups['retained_role_union_counts']
    need(parent_roles==oldindex.get('role_union_counts')and
         all(left_roles[k]+kept_roles[k]==parent_roles[k]for k in roles), 'prepared role counts do not conserve original scope')
    correction={'schema':'annotation-remaining-role-correction/v1','source_plan':receipt['old_plan'],
        'source_index':receipt['old_remaining_index'],'tool_sha256':ROLE_FINALIZER_SHA256,
        'remaining_role_union_counts':left_roles,'retained_role_union_counts':kept_roles,
        'source_shard_bytes_changed':False,'models_executed':False}
    expected_plan={**oldplan,'remaining_index':union['new_producer']['index'],'metadata_correction':correction}
    need(json.dumps(plan,sort_keys=True,allow_nan=False)==json.dumps(expected_plan,sort_keys=True,allow_nan=False),
         'prepared v2 plan differs beyond fixed role metadata correction')
    expected_remaining={**oldremaining,'role_union_counts':left_roles,'role_union_counts_scope':'remaining_index_sources',
        'parent_role_union_counts':parent_roles,'role_union_counts_parent_index':old['full_index'],
        'coverage_plan_file':Path(union['continuation_plan']['path']).name,'metadata_derived_from_index':receipt['old_remaining_index']}
    need(json.dumps(remaining,sort_keys=True,allow_nan=False)==json.dumps(expected_remaining,sort_keys=True,allow_nan=False),
         'prepared remaining index changed beyond role metadata; source shards must stay fixed')
    inventory=oldplan.get('portable_sidecar_inventory_summary',{})
    need(oldplan.get('portable_sidecar_inventory_ref')==receipt['inventory_ref']and
         inventory.get('schema')=='annotation-portable-sidecar-inventory/v1'and
         inventory.get('portable_spec_sha256')==union['portable_source_spec_ref']['sha256']and
         inventory.get('old_contract')==old['contract']and inventory.get('full_index')==old['full_index']and
         inventory.get('old_sidecar_tree')==oldplan.get('old_sidecar_tree')==union['old_producer']['sidecar_tree']and
         inventory.get('all_inventory_paths_matched_original_sources')is True and
         inventory.get('original_sources_checked')==total and
         inventory.get('sidecar_files')==inventory.get('commit_files')==kept and
         inventory.get('matched_completed_sources',0)+inventory.get('matched_partial_sources',0)+inventory.get('matched_unclaimed_sources',0)==kept,
         'completed preparation inventory did not exactly cover retained sources')
    ledger=oldplan['portable_source_ledger_ref'];partials=oldplan.get('retained_partial_sources');unclaimed=oldplan.get('retained_unclaimed_source_count')
    need(isinstance(partials,list)and type(unclaimed)is int and unclaimed>=0 and type(ledger.get('count'))is int and
         type(receipt.get('portable_proofs_reopened'))is int and
         receipt['portable_proofs_reopened']==ledger['count']==len(partials)+unclaimed,
         'prepared partial/unclaimed content proof count differs')
    completed=oldplan['old_progress_snapshot']['committed_source_lower_bound']
    need(type(completed)is int and completed+len(partials)+unclaimed==kept and
         inventory.get('matched_completed_sources')==completed and
         inventory.get('matched_partial_sources')==len(partials)and inventory.get('matched_unclaimed_sources')==unclaimed and
         inventory.get('missing_source_lookups_without_PFS_stat')==left,
         'prepared completed/partial/unclaimed partition differs')
    need(oldplan.get('portable_source_spec_ref')==union['portable_source_spec_ref']and
         ledger==union['portable_source_ledger_ref'],'prepared portable proof refs differ')
    unclaimed_ref=oldplan['retained_unclaimed_sources_ref']
    need(type(unclaimed_ref.get('count'))is int and unclaimed_ref['count']==unclaimed,'prepared unclaimed ledger count differs')
    evidence_file(unclaimed_ref)
    dependencies=[receipt['old_plan'],receipt['old_remaining_index'],{k:unclaimed_ref[k]for k in ('path','sha256')}]
    if spec['purpose']=='production':
        tool={'path':str(Path(__file__).resolve().with_name('finalize_remaining_roles.py')),'sha256':ROLE_FINALIZER_SHA256}
        pinned_file(tool);dependencies.append(tool)
    return {'source_count':total,'domains':oldindex['domains'],'retained_source_count':kept,'remaining_source_count':left,
        'prepared_receipt':receipt,'full_old_sidecar_reaudit_performed':False,'union_database_used':False},dependencies


def inputs(spec):
    """纯读全部控制输入；固定摘要也是运行中再次复核的输入集合。"""
    need(spec.get('schema')in {SCHEMA,SCHEMA_V2}and spec.get('purpose')in {'production','cpu_fixture'},'wrong successor schema/purpose')
    proof_field=attestation_field(spec)
    validate_task_name(spec['task_name'])
    need(type(spec.get('model_API_calls_allowed'))is int and spec['model_API_calls_allowed']==0 and
         spec.get('release_training_labels')is False,'offline API0/no automatic release required')
    old=ref(spec['old_control']);need(old.get('protocol')=='legacy_frozen_v2','explicit original v2 control required')
    base=physical(old['task_root']);oldrun=physical(old['run_root']);oldq=physical(old['queue_root'])
    need(oldrun.parent==base and oldq==oldrun/'queue','original run/queue binding differs')
    oldc=ref(old['contract']);oldindex=ref(old['full_index']);admission=ref(old['admission'])
    need(admission.get('schema')=='ultra-eight-three-train-spec/v2' and
         admission.get('silver_contract')==old['contract'] and admission.get('full_index')==old['full_index'] and
         admission.get('bundle')==old['bundle'] and admission.get('code_root')==old['code_root'],
         'original historical admission binding differs')
    need(admission['cloud']['task_root']==str(base) and admission['cloud']['run_root']==str(oldrun) and
         admission['cloud']['python']==old['python'] and admission['cloud']['priority']==6 and
         admission['cloud']['platform_workers']==1 and admission['cloud']['gpus_per_worker']==8,
         'original platform profile differs')
    claim=ref(old['original_claim']);key=old['contract']['sha256']+'-'+old['full_index']['sha256']
    semantic=physical(old['semantic_lock'])
    need(semantic==base/'.formal-parent-locks'/(key+'.lock') and
         old['original_claim']['path']==str(semantic.with_name(key+'.claim.json')) and
         claim.get('schema')=='ultra-three-train-semantic-job-claim/v1' and
         claim.get('run_root')==str(oldrun) and claim.get('spec_sha256')==old['admission']['sha256'] and
         claim.get('contract_sha256')==old['contract']['sha256'] and claim.get('index_sha256')==old['full_index']['sha256'],
         'original semantic job claim differs')
    lineage=physical(spec['lineage_claim'])
    need(lineage==semantic.with_name(key+'.portable-successor-v1.claim.json') and not lineage.exists(),
         'permanent portable successor claim exists or wrong path; inspect prior task, never retry blindly')
    frozen_bundle(old);frozen_roots(Path(old['code_root'])/'gpu_launch.py',base,physical(old['device_lock_root']))
    legacy_media_path(old,None)
    terminal(ref(old['terminal_observation']),old)
    absence=ref(old['writer_absence']);need(absence.get('task_id')==old['old_task_id'] and
        absence.get('run_root')==str(oldrun) and absence.get('other_successor_tasks')==[] and
        absence.get('platform_instance_state')in {'Failed','Stopped','Succeeded','Completed'},
        'original absence/instance terminal chain differs')
    platform_observation(spec['current_platform_observation'],old)
    pinned_file(old['original_launcher']);python=Path(old['python']);verify_python(old)
    checks=[]
    for item,name,prefix in zip(old['checks'],('ultra_w8_gate_v2.py','verify_ultra_cloud_mounts_v2.py'),(['--live'],[])):
        p=pinned_file(item['script']);expected=prefix+['--spec',old['admission']['path'],'--spec-sha',old['admission']['sha256']]
        need(p.name==name and item['args']==expected,'original historical gate/mount command differs')
        checks.append([str(python),'-B',str(p),*expected])
    need(len(checks)==len(old['checks'])==2,'exactly two original profile/mount checks required')
    freeze=ref(spec['freeze_receipt']);need(freeze.get('schema')=='annotation-legacy-successor-freeze/v1' and
        freeze['base_contract']==old['contract'] and freeze['bundle']['sha256']==old['bundle']['sha256'] and
        freeze['semantic_engine_bytes_unchanged']is True and freeze['all_contract_fields_except_version_unchanged']is True,
        'new freeze lineage differs')
    newc=ref(freeze['new_contract']);profile(oldc,newc)
    code=physical(freeze['code_root']);need(base in code.parents and oldrun not in code.parents and
         physical(old['code_root'])not in code.parents and code!=physical(old['code_root']),'new copied code must be independent under original ROOT')
    proxy={'code_root':str(code),'bundle':freeze['bundle']};frozen_bundle(proxy)
    frozen_roots(code/'gpu_launch.py',base,physical(old['device_lock_root']))
    need(freeze['new_version']==newc['version'] and freeze['bundle']['path']==str(code/'bundle_files.json'),
         'copied code/new version binding differs')
    union=ref(spec['union_spec']);plan=ref(union['continuation_plan']);remaining=ref(union['new_producer']['index'])
    historical=plan['terminal_observation']
    need(historical.get('task_id')==old['old_task_id'],'historical preparation terminal task differs')
    terminal(ref({k:historical[k]for k in ('path','sha256')}),old)
    need(union.get('schema')=='annotation-union-coverage/v1' and union.get('purpose')==spec['purpose'] and
         union['old_producer']['contract']==old['contract'] and union['full_index']==old['full_index'] and
         union['old_producer']['queue_root']==str(oldq) and
         union['old_producer']['sidecar_tree']==str(physical(old['output_root'])/oldc['version']) and
         union['new_producer']['contract']==freeze['new_contract'], 'union producer/full index binding differs')
    need(union.get('source_identity_mode')=='portable_encoded_sha256/v1' and
         plan.get('source_identity_mode')==union['source_identity_mode'] and
         plan.get('status')=='final_prepared_not_started' and plan.get('old_queue_mutated')is False and
         plan.get('old_results_rewritten')is False and plan.get('stage_cache_relabelled')is False,
         'explicit portable preparation with preserved old producer required')
    for name in ('portable_source_spec_ref','portable_source_ledger_ref'):
        need(union.get(name)==plan.get(name),'portable evidence binding differs: '+name)
        evidence_file(union[name])
    need(union.get('portable_sidecar_inventory_ref')==plan.get('portable_sidecar_inventory_ref') and
         union.get('portable_sidecar_inventory_ref')is not None,'fixed portable directory inventory required')
    evidence_file(union['portable_sidecar_inventory_ref'])
    dependencies=[]
    if proof_field=='preparation_attestation':data,dependencies=check_preparation(spec,old,oldindex,union,plan,remaining)
    else:
        data=ref(spec['union_attestation'])
        need(data.get('schema')=='annotation-union-validation/v1' and data.get('spec_sha256')==spec['union_spec']['sha256'] and
         data.get('purpose')==spec['purpose'] and data.get('mode')=='plan' and data.get('exact_union_and_disjoint')is True and
         data.get('formal_runtime_admitted')is False and data.get('GPU_executed')is False and
         data.get('portable_source_ledger_verified')is True and data.get('old_retained_sidecars_reopened')is True and
             data.get('source_identity_mode')==union['source_identity_mode'], 'actual production union plan attestation incomplete')
        check_database(data['consumer_index'],union,data);dependencies.append(data['consumer_index'])
    need(data.get('source_count')==oldindex['source_count']==union['expected_full_source_count'] and
         data.get('domains')==oldindex['domains'] and
         data.get('retained_source_count')==plan['retained_source_count'] and
         data.get('remaining_source_count')==plan['remaining_source_count'] and
         data['retained_source_count']+data['remaining_source_count']==data['source_count'], 'fixed full/retained/remaining denominator differs')
    if spec['purpose']=='production':
        need(data['source_count']==COUNT and data['domains']==DOMAINS and oldc.get('purpose')=='silver_annotation' and
             old['bundle']['sha256']==BUNDLE_SHA256 and oldc.get('runtime')==LEGACY_RUNTIME and
             set(oldc.get('engine_source_files',{}))==set(ENGINE_MODULES),'original production scope/engine profile differs')
    need(plan['remaining_index']==union['new_producer']['index'] and remaining['source_count']==data['remaining_source_count']>0 and
         remaining.get('parent_full_index')==old['full_index'],'remaining index parent/count differs')
    run=physical(spec['new_run_root']);queue=run/'queue';labels=run/'labels';resources=run/'resources-attempt1'
    need(base in run.parents and run!=oldrun and oldrun not in run.parents and run not in oldrun.parents and
         code!=run and code not in run.parents and run not in code.parents and not run.exists(),
         'fresh independent successor run required under original ROOT')
    need(union['new_producer']['queue_root']==str(queue) and
         union['new_producer']['sidecar_tree']==str(labels/newc['version']), 'new producer output namespace differs')
    evidence=physical(spec['evidence_root'])
    validate_output_namespaces(old,freeze,union,run,evidence,controller_root=Path(__file__).resolve().parent)
    need(not evidence.exists() and evidence.parent.is_dir() and all(evidence!=p and p not in evidence.parents and
         evidence not in p.parents for p in (run,oldrun,code,physical(old['code_root']))), 'fresh disjoint execution evidence required')
    for queue_field,name in (('old_broker','broker.json'),('old_completion_map','completion_map.bin'),('old_queue_identity','identity.json')):
        need(pinned_file(spec[queue_field])==oldq/name,'old queue snapshot pin/path differs: '+name)
    need(spec['old_broker']['sha256']==plan['old_progress_snapshot']['broker_snapshot_sha256'],'old broker differs from data plan')
    progress=inspect_progress(old['full_index']['path'],old['full_index']['sha256'],oldq,old['contract']['sha256'])
    need(progress['broker_snapshot_sha256']==spec['old_broker']['sha256'] and
         progress['committed_source_lower_bound']==plan['old_progress_snapshot']['committed_source_lower_bound'],
         'original queue advanced since fixed data proof')
    historic_lock=plan['old_parent_lock'];current=(oldq/'.parent.lock').stat()
    need(historic_lock.get('path')==str(oldq/'.parent.lock') and historic_lock.get('inode')==current.st_ino and
         historic_lock.get('existing_lock_opened_readonly')is True and historic_lock.get('lock_created')is False,
         'historical preparation lock path/inode differs; device observations are retained separately')
    argv=[str(python),'-B',str(code/'gpu_launch.py'),'--execute','--bundle-sha',freeze['bundle']['sha256'],
          '--contract',freeze['new_contract']['path'],'--contract-sha',freeze['new_contract']['sha256'],
          '--shard-index',union['new_producer']['index']['path'],'--index-sha',union['new_producer']['index']['sha256'],
          '--queue-root',str(queue),'--output-root',str(labels),'--output',str(resources),'--target-uuids','auto8',
          '--memory-budget-mib','60000','--max-runtime-seconds','2592000']
    mount_checker=Path(__file__).resolve().with_name('verify_portable_successor_mounts.py')
    mount_pin={'path':str(mount_checker),'sha256':file_sha(mount_checker)}
    return {'old':old,'freeze':freeze,'union':union,'plan':plan,'data':data,'checks':[checks[0]],
        'attestation_field':proof_field,'data_dependency_refs':dependencies,
        'check_sources':[old['checks'][0]['script'],mount_pin],'historical_mount_check':checks[1],
        'portable_mount_checker':mount_pin,'gpu_command':argv,
        'current_old_parent_lock':{'path':str(oldq/'.parent.lock'),'device':current.st_dev,'inode':current.st_ino},
        'historical_preparation_lock':historic_lock,'runtime_mount_gate_still_required':True,
        'new_run_root':str(run),'new_resources':str(resources),'GPU_started':False,'new_GPU_acceptance_inherited':False}


def recheck(spec,checked,semantic_fd,queue_fd,*,owned_claim=None,run_identity=None):
    """门后/启动前重核旧与新可执行件和数据证明，当前锁用当前FD身份核验。"""
    old=checked['old'];frozen_bundle(old)
    validate_output_namespaces(old,checked['freeze'],checked['union'],checked['new_run_root'],spec['evidence_root'],
                               controller_root=Path(__file__).resolve().parent)
    pinned_file(checked['portable_mount_checker'])
    legacy_media_path(old,None);platform_observation(spec['current_platform_observation'],old)
    frozen_bundle({'code_root':checked['freeze']['code_root'],'bundle':checked['freeze']['bundle']})
    pinned_file(old['original_launcher']);verify_python(old)
    for check in old['checks']:pinned_file(check['script'])
    for key in ('old_control','freeze_receipt','union_spec',checked['attestation_field'],'old_broker','old_completion_map','old_queue_identity',
                'current_platform_observation'):pinned_file(spec[key])
    for pin in (old['contract'],old['full_index'],old['original_claim'],old['admission'],checked['freeze']['new_contract'],
                checked['union']['continuation_plan'],checked['union']['new_producer']['index'],*checked['data_dependency_refs']):pinned_file(pin)
    for key in ('portable_source_spec_ref','portable_source_ledger_ref','portable_sidecar_inventory_ref'):evidence_file(checked['union'][key])
    for name,fd in ((old['semantic_lock'],semantic_fd),(str(Path(old['queue_root'])/'.parent.lock'),queue_fd)):
        pathstat=physical(name).stat();held=os.fstat(fd)
        need((pathstat.st_dev,pathstat.st_ino)==(held.st_dev,held.st_ino),'held current lock inode replaced')
    held=os.fstat(queue_fd);initial=checked['current_old_parent_lock']
    need((held.st_dev,held.st_ino)==(initial['device'],initial['inode']) and
         held.st_ino==checked['historical_preparation_lock']['inode'],
         'old parent lock was replaced between inspection and gate completion')
    if owned_claim is None:
        need(not Path(spec['lineage_claim']).exists(),'another portable successor was claimed')
        need(not Path(checked['new_run_root']).exists(),'new run appeared before launch')
    else:
        need(ref_unpinned(spec['lineage_claim'])==owned_claim,'own permanent claim changed before GPU launch')
        s=physical(checked['new_run_root']).stat()
        need((s.st_dev,s.st_ino)==run_identity,'new run inode changed before GPU launch')


def real_runner(argv,log,fds):
    """两个旧锁FD交给原GPU父进程，等其既有精确清理协议退出。"""
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(name,None)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    with log.open('xb')as stream:
        child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,pass_fds=tuple(fds))
        previous={}
        def terminate(signum,frame):
            if child.poll()is None:child.send_signal(signal.SIGTERM)
        try:
            for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,terminate)
            write_new(log.with_name(log.name+'.process.json'),{'pid':child.pid,'argv':argv,'synthetic_fixture':False,
                'inherited_protected_lock_FDs':list(fds),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            return child.wait()
        finally:
            if child.poll()is None:
                child.send_signal(signal.SIGTERM);child.wait()
            for sig,handler in previous.items():signal.signal(sig,handler)


def run(path,pin,*,execute=False,runner=None):
    spec=ref({'path':str(path),'sha256':pin})
    need(runner is None or spec.get('purpose')=='cpu_fixture','synthetic runner may only execute CPU fixtures')
    need(not execute or spec.get('purpose')=='production' or runner is not None,'CPU fixture cannot call real commands')
    checked=inputs(spec)
    checked['checks'].append([checked['old']['python'],'-B',checked['portable_mount_checker']['path'],
                              '--spec',str(path),'--spec-sha',pin])
    if not execute:return {'schema':'annotation-portable-successor-check/v1','passed':True,
        'data_scope':('fixed production union plan, not complete labels'if checked['attestation_field']=='union_attestation'else
            'fixed completed remaining preparation, not complete labels'),
        'union_database_used':checked['attestation_field']=='union_attestation',
        'current_old_parent_lock':checked['current_old_parent_lock'],
        'historical_preparation_lock':checked['historical_preparation_lock'],'GPU_started':False,'runtime_admitted':False}
    evidence=physical(spec['evidence_root']);evidence.mkdir(mode=0o700)
    write_new(evidence/'prepared.json',{'spec_sha256':pin,'spec':spec,'check':checked})
    invoked=runner or real_runner;launched=False;claimed=False
    old=checked['old']
    try:
        with existing_lock(old['semantic_lock'])as semantic_fd:
            for ordinal,argv in enumerate(checked['checks']):
                pinned_file(checked['check_sources'][ordinal])
                verify_python(old)
                rc=invoked(argv,evidence/('old-profile-check-%d.log'%ordinal),(semantic_fd,))
                need(type(rc)is int and rc==0,'historical profile/mount gate failed')
            with existing_lock(Path(old['queue_root'])/'.parent.lock')as queue_fd:
                pinned_file({'path':str(path),'sha256':pin})
                recheck(spec,checked,semantic_fd,queue_fd)
                claim={'schema':'annotation-portable-successor-claim/v1','spec_sha256':pin,'task_name':spec['task_name'],
                    'old_task_id':old['old_task_id'],'old_contract':old['contract'],'old_full_index':old['full_index'],
                    'new_contract':checked['freeze']['new_contract'],'new_index':checked['union']['new_producer']['index'],
                    'new_run_root':checked['new_run_root'],'status':'claimed_no_implicit_retry','synthetic_fixture':spec['purpose']=='cpu_fixture'}
                write_new(spec['lineage_claim'],claim);claimed=True;write_new(evidence/'claimed.json',claim)
                Path(checked['new_run_root']).mkdir(mode=0o700)
                ownrun=Path(checked['new_run_root']).stat()
                pinned_file({'path':str(path),'sha256':pin})
                recheck(spec,checked,semantic_fd,queue_fd,owned_claim=claim,run_identity=(ownrun.st_dev,ownrun.st_ino))
                write_new(evidence/'GPU-start-intent.json',{'argv':checked['gpu_command'],'GPU_started_not_proven_by_intent':True})
                launched=True
                rc=invoked(checked['gpu_command'],evidence/'GPU-parent.log',(semantic_fd,queue_fd))
                write_new(evidence/'GPU-exit.json',{'returncode':rc,'resource_return_not_proven_by_exit_code':True})
                need(type(rc)is int and rc==0,'original GPU parent failed; keep evidence and do not retry blindly')
                p=Path(checked['new_resources'])/'parent_result.json';result=ref_unpinned(p)
                need(result.get('schema')=='open-annotation-independent-gpu-run/v1' and result.get('worker_count')==8 and
                     result.get('bundle_sha256')==checked['freeze']['bundle']['sha256'] and
                     result.get('contract_sha256')==checked['freeze']['new_contract']['sha256'] and
                     result.get('index_sha256')==checked['union']['new_producer']['index']['sha256'] and
                     result.get('resource_returned')is True and result.get('model_API_calls')==0 and
                     result.get('verification',{}).get('passed')is True,'original GPU complete/resource-return receipt missing or differs')
                write_new(evidence/'completed.json',{'schema':'annotation-portable-successor-completed/v1',
                    'parent_result':{'path':str(p),'sha256':file_sha(p)},'union_complete_validation_still_required':True,
                    'old_results_rewritten':False,'training_labels_released':False})
                return {'completed':True,'union_complete_validation_still_required':True}
    except BaseException:
        write_new(evidence/'failure.json',{'traceback':traceback.format_exc(),'launch_attempted':launched,
            'permanent_claim_written':claimed,'automatic_retry_allowed':False})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.spec,a.spec_sha,execute=a.execute),ensure_ascii=False))


if __name__=='__main__':main()
