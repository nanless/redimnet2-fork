"""核新后继的实际持久挂载和全新run，不执行旧fresh-run检查器。

旧W2质量门和旧挂载脚本原字节保留。本门读取同一新后继spec/SHA，
只核原RW/RO挂载、明确共享别名及实际新输入/模型路径，不启动GPU。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

if __package__:
    from .resume_existing_annotation import physical,ref,pinned_file,legacy_media_path
else:
    from resume_existing_annotation import physical,ref,pinned_file,legacy_media_path

SCHEMA='annotation-portable-successor-run/v1'
SCHEMA_V2='annotation-portable-successor-run/v2'
ROOT='/root/experiments/redimnet2/open-annotation-ultra-20261025'


def need(value,message):
    if not value:raise ValueError(message)


def attestation_field(spec):
    """v1仍要求完整union证明；v2只接受已完成的准备/角色定稿证明。"""
    need(spec.get('schema')in {SCHEMA,SCHEMA_V2},'wrong successor schema')
    field='union_attestation'if spec['schema']==SCHEMA else 'preparation_attestation'
    other='preparation_attestation'if field=='union_attestation'else 'union_attestation'
    need(field in spec and other not in spec,'exact versioned attestation field required')
    return field


def validate_row(expected,value):
    rows=value.get('filesystems',[]);need(len(rows)==1,'ambiguous findmnt result')
    row=rows[0];options=set(row['options'].split(','));mode=expected['mode']
    need(row['target']==expected['target'] and row['source']==expected['source'] and
         row['fstype']==expected['fstype']=='gpfs' and mode in options and
         ({'ro','rw'}-{mode}).isdisjoint(options),'persistent mount source/target/type/mode differs')
    return row


def validate_output_namespaces(old,freeze,union,newrun,evidence,*,controller_root):
    """保护专用源码/索引/声明目录；不把放一个JSON的通用base全部禁用。"""
    base=physical(old['task_root']);run=physical(newrun);ev=physical(evidence)
    if old.get('purpose')=='production':
        need(Path('/root')in ev.parents,'production execution evidence must stay in persistent user /root mount')
    roots=[physical(old['code_root']),physical(old['full_index']['path']).parent,
           physical(old['admission']['path']).parent,physical(union['new_producer']['index']['path']).parent,
           physical(freeze['code_root']),physical(freeze['code_root']).parent,physical(controller_root),legacy_media_path(old,None)]
    if 'original_launcher'in old:roots.append(physical(old['original_launcher']['path']).parent)
    if 'semantic_lock'in old:roots.append(physical(old['semantic_lock']).parent)
    # CPU格式样本/旧元数据可能把单个JSON放在base或其通用父目录。
    # 这些通用目录不是专用输入树；固定叶子文件仍由SHA及fresh输出规则保护。
    roots=list(dict.fromkeys(p for p in roots if p!=base and p not in base.parents))
    def disjoint(a,b):return a!=b and a not in b.parents and b not in a.parents
    need(all(disjoint(run,p)for p in roots),'new run overlaps frozen/control/index/semantic namespace')
    oldrun=physical(old['run_root'])
    need(disjoint(run,oldrun),'new run overlaps old producer')
    need(all(disjoint(ev,p)for p in roots+[oldrun,run]),
         'execution evidence overlaps frozen/input/semantic/output namespace')
    return [str(p)for p in roots]


def verify(spec_path,spec_sha,*,runner=subprocess.run):
    spec_pin={'path':str(spec_path),'sha256':spec_sha};spec=ref(spec_pin)
    field=attestation_field(spec)
    need(spec.get('purpose')in {'production','cpu_fixture'} and
         type(spec.get('model_API_calls_allowed'))is int and spec['model_API_calls_allowed']==0 and
         spec.get('release_training_labels')is False,'fixed offline successor spec required')
    old=ref(spec['old_control']);admission=ref(old['admission']);cloud=admission['cloud']
    base=physical(old['task_root']);oldrun=physical(old['run_root']);newrun=physical(spec['new_run_root'])
    if spec['purpose']=='production':need(str(base)==ROOT,'original independent Ultra ROOT required')
    need(cloud['task_root']==str(base) and cloud['run_root']==str(oldrun) and
         admission.get('silver_contract')==old['contract'] and admission.get('full_index')==old['full_index'],
         'original cloud mount profile/admission identity differs')
    need(base.is_dir() and base in newrun.parents and newrun!=oldrun and oldrun not in newrun.parents and
         newrun not in oldrun.parents and not newrun.exists(),'fresh independent new successor run required')
    profiles={m['target']:m for m in cloud['mounts']}
    need(set(profiles)=={'/root','/shared/group-shared'} and len(cloud['mounts'])==2,
         'exact original user/shared persistent mounts required')
    need(profiles['/root']['mode']=='rw' and profiles['/shared/group-shared']['mode']=='ro' and
         all(m.get('fstype')=='gpfs' and isinstance(m.get('source'),str)and m['source']for m in profiles.values()),
         'user RW/shared RO GPFS profiles required')
    shared=legacy_media_path(old,None);alias=Path(old['legacy_source_alias']['alias_path'])
    alias_before=alias.lstat();alias_text=str(alias.readlink());shared_before=shared.stat()
    proofs={}
    def query(path,expected):
        key=str(path)
        if key in proofs:
            need(proofs[key]['expected']==expected,'same path assigned conflicting mount profile');return
        result=runner(['findmnt','-J','-T',key,'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],capture_output=True,text=True,timeout=5,check=True)
        need(not result.stderr.strip(),'findmnt returned stderr')
        proofs[key]={'expected':expected,'actual':validate_row(expected,json.loads(result.stdout))}
    for target,expected in profiles.items():query(Path(target),expected)
    def user_path(path):
        p=physical(path);need(p.exists(),'required user-side path missing: '+str(p))
        if spec['purpose']=='production':need(Path('/root')in p.parents or p==Path('/root'),'user control path outside original /root mount')
        query(p,profiles['/root']);return p
    ancestor=newrun.parent
    while not ancestor.exists():ancestor=ancestor.parent
    need(ancestor==base or base in ancestor.parents,'new run parent escapes original ROOT')
    user_path(base);user_path(ancestor);user_path(oldrun)
    evidence_ancestor=physical(spec['evidence_root'])
    while not evidence_ancestor.exists():evidence_ancestor=evidence_ancestor.parent
    user_path(evidence_ancestor)
    user_path(old['queue_root']);user_path(old['output_root']);user_path(Path(__file__).resolve().parent)
    freeze=ref(spec['freeze_receipt']);need(freeze.get('base_contract')==old['contract'],'freeze old producer differs')
    code=user_path(freeze['code_root']);oldcode=user_path(old['code_root'])
    need(base in code.parents and code!=oldcode and code!=newrun and code not in newrun.parents and newrun not in code.parents,
         'new code/run namespaces overlap')
    union=ref(spec['union_spec']);need(union.get('full_index')==old['full_index'] and
         union['new_producer']['contract']==freeze['new_contract'],'actual full/new model input refs differ')
    protected=validate_output_namespaces(old,freeze,union,newrun,spec['evidence_root'],controller_root=Path(__file__).resolve().parent)
    oldc=ref(old['contract']);newc=ref(freeze['new_contract'])
    need(newc.get('models')==oldc.get('models') and set(oldc.get('models',{}))=={'asr','aligner'},'original two-model source paths required')
    actual_refs=[spec_pin,spec['old_control'],old['admission'],old['contract'],old['full_index'],spec['freeze_receipt'],
                freeze['new_contract'],freeze['bundle'],spec['union_spec'],union['continuation_plan'],union['new_producer']['index'],spec[field],
                spec['current_platform_observation'],old['original_claim'],old['terminal_observation'],old['writer_absence']]
    for key in ('portable_source_spec_ref','portable_source_ledger_ref','portable_sidecar_inventory_ref'):
        actual_refs.append({k:union[key][k]for k in ('path','sha256')})
    for item in actual_refs:user_path(pinned_file(item))
    data=ref(spec[field])
    if field=='union_attestation':user_path(pinned_file(data['consumer_index']))
    else:
        need(data.get('schema')=='annotation-remaining-role-finalize/v1'and
             data.get('new_plan')==union['continuation_plan']and data.get('new_remaining_index')==union['new_producer']['index'],
             'completed preparation mount refs differ')
        for key in ('old_plan','old_remaining_index'):user_path(pinned_file(data[key]))
    checks=old['checks'];need(len(checks)==2 and Path(checks[1]['script']['path']).name=='verify_ultra_cloud_mounts_v2.py',
         'original fresh mount checker lineage missing')
    historical_mount=pinned_file(checks[1]['script']);user_path(historical_mount)
    # 旧run存在是恢复的正常状态；仅保留旧脚本SHA，从不执行它或替换旧admission。
    model_files=0
    for model in oldc['models'].values():
        directory=legacy_media_path(old,model['path']);need(directory.is_dir() and (directory==shared or shared in directory.parents),
            'model must resolve inside original shared RO mount')
        query(directory,profiles['/shared/group-shared'])
        for relative,pin in model['files'].items():
            rel=Path(relative);need(not rel.is_absolute() and rel.parts and '..'not in rel.parts,'unsafe model asset path')
            p=physical(directory/rel);need(p.is_file() and p.stat().st_size==pin['bytes'],'model asset path/bytes differs')
            query(p,profiles['/shared/group-shared']);model_files+=1
    need(legacy_media_path(old,None)==shared and alias.readlink()==Path(alias_text) and
         (alias.lstat().st_dev,alias.lstat().st_ino)==(alias_before.st_dev,alias_before.st_ino) and
         (shared.stat().st_dev,shared.stat().st_ino)==(shared_before.st_dev,shared_before.st_ino),'shared mount alias changed during validation')
    need(not newrun.exists() and not newrun.is_symlink(),'new successor run appeared during mount validation')
    pinned_file(spec_pin)
    storage=os.statvfs(base)
    return {'schema':'annotation-portable-successor-mounts/v1','passed':True,'spec':spec_pin,
        'original_mount_profile_admission':old['admission'],'historical_fresh_mount_checker':checks[1]['script'],
        'historical_fresh_mount_checker_executed':False,'old_run_exists':oldrun.exists(),'old_run_required_fresh':False,
        'new_run_root':str(newrun),'new_run_created':False,'execution_evidence_path':spec['evidence_root'],
        'execution_evidence_mount_checked_at':str(evidence_ancestor),'protected_output_namespaces':protected,'mounts':proofs,
        'shared_alias':{'path':str(alias),'link_target':alias_text,'resolved_target':str(shared),
            'device':alias_before.st_dev,'inode':alias_before.st_ino,'target_device':shared_before.st_dev,'target_inode':shared_before.st_ino},
        'model_asset_paths_and_bytes_checked':model_files,'model_content_SHA_rechecked':False,
        'available_bytes':storage.f_bavail*storage.f_frsize,'available_inodes':storage.f_favail,'quota_known':False,
        'GPU_or_model_execution':False,'runtime_GPU_admission_granted':False,'synthetic_fixture':spec['purpose']=='cpu_fixture'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True)
    args=parser.parse_args();print(json.dumps(verify(args.spec,args.spec_sha),ensure_ascii=False))


if __name__=='__main__':main()
