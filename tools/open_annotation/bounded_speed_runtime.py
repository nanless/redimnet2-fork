"""有界新生产者的显式运行准入；只读小控制文件，不扫描全库或建数据库。

该模块把原始语义锁、原始队列父锁和最近旧生产者的锁一起保留到
GPU父子进程生命周期结束。试验、质量复核和当前平台终态缺一不可；
CPU格式检查不能生成试验通过证明，也不会停止或重启任何旧任务。
"""
from __future__ import annotations
import contextlib
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

if __package__:
    from . import bounded_speed_handoff as H
    from . import bounded_speed_queue as Q
    from . import resume_existing_annotation as R
    from .verify_portable_successor_mounts import validate_row
else:
    import bounded_speed_handoff as H
    import bounded_speed_queue as Q
    import resume_existing_annotation as R
    from verify_portable_successor_mounts import validate_row

SCHEMA='annotation-bounded-speed-runtime/v1'
PLATFORM_SCHEMA='annotation-bounded-speed-platform-observation/v1'
ADMISSION_SCHEMA='annotation-asr-speed-selected-pilot-admission/v1'
QUALITY_SCHEMA='annotation-asr-speed-quality-review/v1'
need=H.need


def pinned(pin):
    return H.read_pin(pin)[0]


def index_kind(path, expected_sha):
    """仅识别显式schema；绝不凭可选参数把旧v1变成新薄索引。"""
    # 仅分派沿用旧v1读取语义；新薄索引随后由BoundedShardReader独立核物理路径。
    if __package__:from .readonly_progress import read_small
    else:from readonly_progress import read_small
    value,raw=read_small(Path(path),64*1024**2)
    need(H.is_sha(expected_sha)and H.P.byte_sha(raw)==expected_sha,'index routing SHA differs')
    kind=value.get('schema')
    need(kind in {'redimnet2-shard-index/v1',H.INDEX_SCHEMA},'unsupported source index schema')
    return kind


def runtime_options(args):
    path=getattr(args,'bounded_runtime_spec',None);sha=getattr(args,'bounded_runtime_sha',None)
    need(bool(path)==bool(sha),'bounded runtime spec/SHA must be paired')
    bounded=index_kind(args.shard_index,args.index_sha)==H.INDEX_SCHEMA
    need(bounded==bool(path),'thin index requires explicit bounded runtime; v1 refuses bounded options')
    return {'path':str(path),'sha256':sha}if bounded else None


def same_profile(a,b,*,allow_control_changes=False):
    """只允许ASR数值方式与已声明引擎源码不同；模型/定位/质量条件不变。"""
    old_pins=a.get('engine_source_files');new_pins=b.get('engine_source_files')
    need(isinstance(old_pins,dict)and isinstance(new_pins,dict)and set(old_pins)==set(new_pins),'selected semantic module set differs')
    permitted={'worker.py','report.py'}if allow_control_changes else set()
    need(all(old_pins[k]==new_pins[k]for k in old_pins if k not in permitted),'selected pilot semantic computation source differs')
    for name in ('models','alignment','decoder','source_files','quality_policy'):
        need(a.get(name)==b.get(name),'selected pilot changes protected model/quality field: '+name)
    left=dict(a['runtime']);right=dict(b['runtime']);left.pop('asr_compute',None);right.pop('asr_compute',None)
    need(left==right,'selected pilot changes Ultra/window/normalization policy')
    need(a.get('model_API_calls_allowed')==b.get('model_API_calls_allowed')==0 and
         a.get('release_training_labels')is b.get('release_training_labels')is False,'offline/no-release policy differs')


def check_selected_pilot(pin,new_contract,purpose):
    """只核已发生的有限GPU原件及逐差异质量复核，不从CPU结果推断速度。"""
    selected=pinned(pin)
    need(selected.get('schema')==ADMISSION_SCHEMA and selected.get('purpose')==purpose,'explicit selected-pilot admission required')
    candidate=pinned(selected['candidate_contract']);comparison=pinned(selected['comparison'])
    recovery=pinned(selected['recovery_comparison']);result=pinned(selected['pilot_result']);quality=pinned(selected['quality_review'])
    need(candidate.get('purpose')==('cpu_fixture'if purpose=='cpu_fixture'else'pilot_validation'),'real pilot contract required')
    same_profile(candidate,new_contract,allow_control_changes=True)
    need(candidate['runtime'].get('asr_compute',{'dtype':'float32','attention':'eager'})==
         new_contract['runtime'].get('asr_compute',{'dtype':'float32','attention':'eager'})==selected.get('selected_asr_compute'),
         'selected ASR compute profile differs')
    need(comparison.get('schema')=='redimnet2-annotation-speed-comparison/v1' and comparison.get('mode')=='speed' and
         comparison.get('candidate_contract_sha256')==selected['candidate_contract']['sha256'] and
         comparison.get('cold_speed_ratio_valid')is True and comparison.get('all_planned_sources_accounted')is True and
         comparison.get('full_sidecar_commit_readback_verified')is True and
         type(comparison.get('cold_end_to_end_speed_ratio'))in (int,float)and math.isfinite(comparison['cold_end_to_end_speed_ratio'])and comparison['cold_end_to_end_speed_ratio']>1 and
         comparison.get('baseline_performance',{}).get('actual_compute_execution_verified')is True and
         comparison.get('candidate_performance',{}).get('actual_compute_execution_verified')is True,
         'actual cold speed/complete finite comparison required')
    count=comparison.get('source_count');need(type(count)is int and count>0 and (purpose!='production'or count>=3000),'representative finite GPU denominator too small')
    need(recovery.get('schema')=='redimnet2-annotation-speed-comparison/v1' and recovery.get('mode')=='recovery' and
         recovery.get('candidate_contract_sha256')==selected['candidate_contract']['sha256'] and
         recovery.get('strict_recovery_native_results_equal')is True and recovery.get('all_planned_sources_accounted')is True and
         recovery.get('full_sidecar_commit_readback_verified')is True,'actual selected-profile recovery evidence required')
    rc=recovery.get('source_count');need(type(rc)is int and rc>0 and (purpose!='production'or rc>=512),'selected-profile recovery denominator too small')
    totals=recovery.get('candidate_performance',{}).get('totals',{})
    need(totals.get('sidecars_committed')==0 and totals.get('sidecar_cache_hits')==rc and
         totals.get('asr_rows')==totals.get('aligner_rows')==0,'recovery must reuse published bytes without model calls')
    need(result.get('schema')=='annotation-bounded-speed-pilot-result/v1' and result.get('purpose')==purpose and
         result.get('all_finite_audits_completed')is True and result.get('database_created')is False,'finite pilot not completely verified')
    # 比较结果、完整父进程归还和主/恢复track必须属于同一次固定试验。
    plan=pinned(result['plan']);main=pinned(selected['main_track']);baseline=pinned(selected['baseline_track']);resume=pinned(selected['recovery_track'])
    baseline_contract=pinned(baseline['contract']);same_profile(candidate,baseline_contract)
    need(baseline_contract['runtime'].get('asr_compute',{'dtype':'float32','attention':'eager'})=={'dtype':'float32','attention':'eager'},'measured baseline must be historical FP32/eager')
    need(main.get('schema')==baseline.get('schema')==resume.get('schema')=='redimnet2-annotation-speed-track/v1' and
         main['contract']==resume['contract']==selected['candidate_contract'] and main['index']==baseline['index'] and
         comparison.get('candidate_spec_sha256')==selected['main_track']['sha256'] and
         comparison.get('baseline_spec_sha256')==selected['baseline_track']['sha256'] and
         comparison.get('baseline_contract_sha256')==baseline['contract']['sha256'] and
         recovery.get('baseline_spec_sha256')==selected['main_track']['sha256'] and
         recovery.get('candidate_spec_sha256')==selected['recovery_track']['sha256'], 'selected comparison/main/recovery track chain differs')
    matrix=pinned(plan['main_prepared']);variants=matrix.get('variants',{})
    profile=selected.get('selected_profile_name');steps=result.get('steps',[])
    need(isinstance(profile,str)and profile in variants and variants[profile].get('track')==selected['main_track'] and
         variants.get('asr-fp32-eager',{}).get('track')==selected['baseline_track'], 'pilot result plan does not bind selected tracks')
    proof_pins=[]
    for name,track in (('baseline',baseline),('main',main),('recovery',resume)):
        proofs=selected['resource_evidence'][name];root=Path(track['resources_root'])
        expected={'parent_result':'parent_result.json','resource_return':'resource_return_double_probe.json','terminal_drain':'terminal_drain_complete.json'}
        need(set(proofs)==set(expected),'complete resource evidence pin set required')
        for key,filename in expected.items():need(proofs[key]['path']==str(root/filename),'selected resource evidence path differs')
        parent=pinned(proofs['parent_result']);returned=pinned(proofs['resource_return']);drain=pinned(proofs['terminal_drain']);proof_pins.extend(proofs.values())
        need(parent.get('schema')=='open-annotation-independent-gpu-run/v1'and parent.get('contract_sha256')==track['contract']['sha256']and
             parent.get('index_sha256')==track['index']['sha256']and parent.get('resource_returned')is True and
             parent.get('full_sidecar_readback')is True and parent.get('worker_count')==2 and
             parent.get('bundle_sha256')==matrix['variants']['asr-fp32-eager'if name=='baseline'else profile]['freeze']['bundle_sha256'] and
             parent.get('verification',{}).get('passed')is True and parent['verification'].get('database_created')is False and
             parent['verification'].get('source_disjointness_storage')=='bounded_pilot_memory_set', 'actual two-GPU parent verification/resource return missing')
        need(returned.get('interval_seconds',0)>=5 and len(returned.get('snapshots',[]))==2 and
             all(not snap['compute_processes']and all(d['used_mib']<=256 for d in snap['devices'])for snap in returned['snapshots'])and
             drain.get('all_attested_contexts_absent')is True,'actual selected CUDA drain/idle snapshots missing')
        if name!='baseline':
            stage='main'if name=='main'else'recovery'
            need(any(st.get('stage')==stage and st.get('profile')==profile and st.get('parent_result_sha256')==proofs['parent_result']['sha256']for st in steps), 'result step not bound to selected parent evidence')
    need(isinstance(profile,str) and any(s.get('stage')=='main'and s.get('profile')==profile and s.get('source_count')==count and s.get('resource_returned')is True for s in steps) and
         any(s.get('stage')=='recovery'and s.get('profile')==profile and s.get('sources')==rc and
             s.get('new_ASR_calls')==s.get('new_FA_calls')==0 and s.get('resource_returned')is True for s in steps),
         'selected pilot main/recovery resource-return chain incomplete')
    need(quality.get('schema')==QUALITY_SCHEMA and quality.get('purpose')==purpose and
         quality.get('comparison')==selected['comparison'] and quality.get('candidate_contract')==selected['candidate_contract'] and
         quality.get('review_complete')is True and quality.get('all_changed_rows_dispositioned')is True and
         quality.get('text_quality_accepted')is True and quality.get('complete_word_boundary_quality_accepted')is True and
         quality.get('candidate_labels_released')is False and quality.get('review_evidence'),
         'explicit text/word-boundary difference review is required; timing is not accuracy')
    changed={name:{'pilot':candidate['engine_source_files'][name],'successor':new_contract['engine_source_files'][name]}
             for name in candidate['engine_source_files'] if candidate['engine_source_files'][name]!=new_contract['engine_source_files'][name]}
    integration_pins=[]
    if changed:
        review=pinned(selected['integration_review'])
        need(review.get('schema')=='annotation-bounded-integration-review/v1'and review.get('purpose')==purpose and
             review.get('candidate_contract')==selected['candidate_contract']and
             pinned(review['new_contract'])==new_contract and review.get('changed_semantic_control_modules')==changed and
             set(changed)<={'worker.py','report.py'}and review.get('independent_review_complete')is True and
             review.get('CPU_tests_passed')is True and review.get('review_evidence'), 'new transport/control source requires explicit independent review')
        integration_pins=[selected['integration_review'],review['new_contract'],*review['review_evidence']]
        for item in integration_pins:R.pinned_file(item)
    # 完整审阅原件可以由人工听审、图和已授权API组成；只固定引用，不重读音频。
    evidence=quality['review_evidence'];need(isinstance(evidence,list)and evidence,'quality evidence refs required')
    for item in evidence:R.pinned_file(item)
    selected['verified_evidence_pins']=[selected[key]for key in ('candidate_contract','comparison','recovery_comparison','pilot_result','quality_review','main_track','baseline_track','recovery_track')]+[result['plan'],plan['main_prepared'],baseline['contract']]+proof_pins+evidence+integration_pins
    return selected


def check_platform(pin,index,*,fresh=True):
    value=pinned(pin)
    need(value.get('schema')==PLATFORM_SCHEMA and value.get('old_task_id')==index['old_task_id'] and
         value.get('old_queue_root')==index['old_queue_root'] and value.get('competing_successor_tasks')==[] and
         value.get('remote_proc_enumeration_performed')is False and
         value.get('evidence_basis')=='platform_terminal_and_current_shared_locks','current platform absence binding differs')
    observation=value.get('terminal_observation');need(isinstance(observation,dict),'terminal pin missing from platform observation')
    if fresh:H.P.terminal_observation(Path(observation['path']),observation['sha256'],index['old_task_id'],Path(index['old_queue_root']))
    else:pinned(observation)
    at=datetime.datetime.fromisoformat(value['observed_utc']);need(at.tzinfo is not None,'platform timezone required')
    age=(datetime.datetime.now(datetime.timezone.utc)-at).total_seconds()
    need(age>=0 and (not fresh or age<=3600),'current platform observation stale/future')
    term=pinned(observation);need(term['observed_utc']==value['observed_utc'],'platform/terminal observation times differ')
    return observation


def compile_runtime(runtime_pin,*,fresh=True):
    """固定薄索引、前一真实run和原始祖先；不运行旧union或任何数据库查询。"""
    spec=pinned(runtime_pin);purpose=spec.get('purpose')
    need(spec.get('schema')==SCHEMA and purpose in {'production','cpu_fixture'},'explicit bounded runtime schema/purpose required')
    need(type(spec.get('model_API_calls_allowed'))is int and spec['model_API_calls_allowed']==0 and
         spec.get('release_training_labels')is False,'offline API0/no automatic label release required')
    need(isinstance(spec.get('task_name'),str)and spec['task_name'].startswith('zhounan_'),'task name must start zhounan_')
    reader=H.BoundedShardReader(spec['index']['path'],spec['index']['sha256']);index=reader.index
    need(index['purpose']==purpose,'runtime/index purpose differs')
    prior=pinned(spec['previous_runtime_spec'])
    need(prior.get('schema')in {'annotation-portable-successor-run/v1','annotation-portable-successor-run/v2'} and prior.get('purpose')==purpose,'fixed previous portable run spec required')
    old=pinned(prior['old_control']);freeze=pinned(prior['freeze_receipt']);union=pinned(prior['union_spec'])
    need(old.get('protocol')=='legacy_frozen_v2','original legacy v2 control required')
    need(freeze.get('new_contract')==index['old_contract'] and union['new_producer']['contract']==index['old_contract'] and
         union['new_producer']['index']==index['parent_index'] and prior['new_run_root']==str(Path(index['old_queue_root']).parent) and
         union['new_producer']['queue_root']==index['old_queue_root'] and union['new_producer']['sidecar_tree']==index['old_sidecar_tree'],
         'current old producer/index/run not bound to previous runtime chain')
    original=pinned(old['contract']);admission=pinned(old['admission']);claim=pinned(old['original_claim'])
    current=pinned(index['old_contract'])
    need(freeze.get('schema')=='annotation-legacy-successor-freeze/v1'and freeze.get('base_contract')==old['contract']and
         freeze.get('semantic_engine_bytes_unchanged')is True and freeze.get('all_contract_fields_except_version_unchanged')is True and
         {k:v for k,v in original.items()if k!='version'}=={k:v for k,v in current.items()if k!='version'},'previous producer computation lineage differs')
    if purpose=='production':
        if __package__:from .freeze_legacy_successor import BUNDLE_SHA256,ENGINE_MODULES
        else:from freeze_legacy_successor import BUNDLE_SHA256,ENGINE_MODULES
        need(old['bundle']['sha256']==BUNDLE_SHA256 and set(original.get('engine_source_files',{}))==set(ENGINE_MODULES), 'original production frozen identity differs')
    need(index['old_sidecar_tree']==str(Path(prior['new_run_root'])/'labels'/current['version']), 'previous output tree not in original declared run')
    base=R.physical(old['task_root']);oldrun=R.physical(old['run_root']);oldq=R.physical(old['queue_root'])
    need(oldrun.parent==base and oldq==oldrun/'queue','original ancestor run/queue binding differs')
    need(admission.get('schema')=='ultra-eight-three-train-spec/v2' and admission.get('silver_contract')==old['contract'] and
         admission.get('full_index')==old['full_index'] and admission.get('bundle')==old['bundle'] and admission.get('code_root')==old['code_root'],
         'original quality admission identity differs')
    cloud=admission['cloud'];need(cloud['task_root']==str(base)and cloud['run_root']==str(oldrun)and cloud['python']==old['python']and
         cloud['priority']==6 and cloud['platform_workers']==1 and cloud['gpus_per_worker']==8,'original platform profile differs')
    key=old['contract']['sha256']+'-'+old['full_index']['sha256'];semantic=base/'.formal-parent-locks'/(key+'.lock')
    need(old['semantic_lock']==str(semantic)and old['original_claim']['path']==str(semantic.with_name(key+'.claim.json'))and
         claim.get('schema')=='ultra-three-train-semantic-job-claim/v1'and claim.get('run_root')==str(oldrun)and
         claim.get('spec_sha256')==old['admission']['sha256']and claim.get('contract_sha256')==old['contract']['sha256']and
         claim.get('index_sha256')==old['full_index']['sha256'],'original semantic claim differs')
    lineage=R.physical(prior['lineage_claim']);lineage_value=R.ref_unpinned(lineage)
    need(lineage==semantic.with_name(key+'.portable-successor-v1.claim.json')and
         lineage_value.get('schema')=='annotation-portable-successor-claim/v1'and
         lineage_value.get('spec_sha256')==spec['previous_runtime_spec']['sha256']and
         lineage_value.get('new_contract')==index['old_contract']and lineage_value.get('new_index')==index['parent_index']and
         lineage_value.get('new_run_root')==prior['new_run_root'],'previous runtime permanent claim differs')
    checks=[];need(len(old['checks'])==2,'exact original quality/mount commands required')
    for item,name,prefix in zip(old['checks'],('ultra_w8_gate_v2.py','verify_ultra_cloud_mounts_v2.py'),(['--live'],[])):
        script=R.pinned_file(item['script']);expected=prefix+['--spec',old['admission']['path'],'--spec-sha',old['admission']['sha256']]
        need(script.name==name and item['args']==expected,'original quality/mount command changed')
        checks.append([old['python'],'-B',str(script),*expected])
    new=pinned(spec['new_contract']);need(new.get('schema')=='redimnet2-production-contract/v2'and
        new.get('purpose')==('cpu_fixture'if purpose=='cpu_fixture'else'silver_annotation')and new.get('version')!=index['old_contract_version'],
        'distinct new producer contract required')
    selected=check_selected_pilot(spec['selected_pilot_admission'],new,purpose)
    run=R.physical(spec['new_run_root']);code=R.physical(spec['new_code_root'])
    need(base in run.parents and base in code.parents and run!=code and run not in code.parents and code not in run.parents,'new runtime/code namespace differs')
    need(index['new_queue_root']==str(run/'queue')and index['new_sidecar_root']==str(run/'labels')and
         spec['device_lock_root']==old['device_lock_root'],'new run/queue/labels/shared UUID locks differ')
    protected=[oldrun,Path(index['old_queue_root']).parent,R.physical(old['code_root']),Path(index['parent_index']['path']).parent,Path(spec['index']['path']).parent,semantic.parent]
    need(all(run!=p and run not in p.parents and p not in run.parents for p in protected),'new run overlaps old/control/input namespace')
    need(spec['new_bundle']['path']==str(code/'bundle_files.json'),'new bundle location differs')
    terminal=check_platform(spec['current_platform_observation'],index,fresh=fresh)
    # 所有运行时再核输入都是小控制JSON或冻结源码；旧大证明只保留历史SHA引用。
    pins=[runtime_pin,spec['index'],spec['previous_runtime_spec'],prior['old_control'],prior['freeze_receipt'],prior['union_spec'],
          old['contract'],old['full_index'],old['admission'],old['original_claim'],index['old_contract'],index['parent_index'],
          spec['new_contract'],spec['new_bundle'],spec['selected_pilot_admission'],spec['current_platform_observation'],terminal,*selected['verified_evidence_pins']]
    return dict(spec=spec,index=index,reader=reader,old=old,admission=admission,new_contract=new,selected=selected,
        previous=prior,original_contract=original,ancestor_paths=[str(semantic),str(oldq/'.parent.lock')],
        original_commands=checks,terminal_pin=terminal,pins=pins,lineage_path=str(lineage),lineage_value=lineage_value,
        unique_claim_path=str(semantic.with_name('bounded-speed-'+H.P.digest(dict(old_contract=index['old_contract'],parent_index=index['parent_index'],old_queue_root=index['old_queue_root']))+'.claim.json')),
        unique_claim_core=dict(schema='annotation-bounded-speed-successor-claim/v1',old_contract=index['old_contract'],
            parent_index=index['parent_index'],old_queue_root=index['old_queue_root'],index=spec['index'],
            new_contract=spec['new_contract'],new_bundle=spec['new_bundle'],new_code_root=spec['new_code_root'],
            new_run_root=spec['new_run_root'],selected_pilot_admission=spec['selected_pilot_admission'],
            status='claimed_no_implicit_new_producer',synthetic_fixture=purpose=='cpu_fixture'))


def verify_mounts(compiled,*,runner=subprocess.run):
    """同原RW/RO挂载、同共享别名和模型路径；不执行要求旧run不存在的脚本。"""
    old=compiled['old'];spec=compiled['spec'];cloud=compiled['admission']['cloud'];profiles={m['target']:m for m in cloud['mounts']}
    need(len(cloud['mounts'])==2 and set(profiles)=={'/root','/shared/group-shared'}and
         profiles['/root']['mode']=='rw'and profiles['/shared/group-shared']['mode']=='ro','exact original mount profile required')
    shared=R.legacy_media_path(old,None);proofs={}
    def query(path,expected):
        path=R.physical(path);key=str(path)
        if key in proofs:return
        result=runner(['findmnt','-J','-T',key,'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],capture_output=True,text=True,timeout=5,check=True)
        need(not result.stderr.strip(),'findmnt returned stderr');proofs[key]=validate_row(expected,json.loads(result.stdout))
    for target,expected in profiles.items():query(Path(target),expected)
    user=[Path(spec['new_run_root']),Path(spec['new_code_root']),Path(old['task_root']),Path(old['run_root']),Path(compiled['index']['old_queue_root'])]
    for pin in compiled['pins']:user.append(R.pinned_file(pin))
    for path in user:
        need(path.exists(),'required runtime/control path missing')
        if spec['purpose']=='production':need(path==Path('/root')or Path('/root')in path.parents,'runtime control outside persistent /root')
        query(path,profiles['/root'])
    for model in compiled['new_contract']['models'].values():
        directory=R.legacy_media_path(old,model['path']);need(directory==shared or shared in directory.parents,'model outside original shared RO mount')
        query(directory,profiles['/shared/group-shared'])
        for rel,pin in model['files'].items():
            p=directory/rel;need(not Path(rel).is_absolute()and '..'not in Path(rel).parts and p.is_file()and p.stat().st_size==pin['bytes'],'model asset location/bytes differs')
            query(p,profiles['/shared/group-shared'])
    return dict(schema='annotation-bounded-speed-mounts/v1',passed=True,mounts=proofs,historical_mount_checker=old['checks'][1]['script'],historical_fresh_mount_checker_executed=False)


class RuntimeLease:
    """父进程开已有祖先锁；子进程只接收同一FD，绝不另造替代锁。"""
    def __init__(self,runtime_pin):
        self.pin=runtime_pin;self.compiled=compile_runtime(runtime_pin);self.handles=[];self.current=None;self.held=False;self.unique_claim_pin=None

    def _identity(self,entry,fd):
        path=R.physical(entry['path']);a=os.fstat(fd);b=path.stat()
        need(stat.S_ISREG(a.st_mode)and(a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)==(entry['device'],entry['inode']),'ancestor inherited lock identity differs')
        if self.compiled['spec']['purpose']=='production':
            info=Path('/proc/self/fdinfo')/str(fd)
            need(info.is_file()and any('FLOCK'in line and 'WRITE'in line for line in info.read_text().splitlines()if line.startswith('lock:')),'ancestor lock ownership not visible')

    def __enter__(self):
        try:
            for path in self.compiled['ancestor_paths']:
                fd=os.open(R.physical(path),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0));s=os.fstat(fd)
                self.handles.append(dict(path=path,fd=fd,device=s.st_dev,inode=s.st_ino))
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);self._identity(self.handles[-1],fd)
            self.current=Q.ExistingProducerLease(self.compiled['spec']['index']['path'],self.compiled['spec']['index']['sha256'],self.compiled['terminal_pin']).__enter__()
            self.held=True
            path=Path(self.compiled['unique_claim_path']);core=self.compiled['unique_claim_core']
            if path.exists():
                actual=R.ref_unpinned(path)
                need({k:v for k,v in actual.items()if k not in {'first_runtime_spec','created_utc'}}==core,
                     'another bounded producer was permanently claimed; inspect prior results, do not rerun')
            else:H.P.write_new(path,dict(core,first_runtime_spec=self.pin,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            self.unique_claim_pin={'path':str(path),'sha256':R.file_sha(path)}
            self.verify();return self
        except BaseException:self.__exit__(None,None,None);raise

    @classmethod
    def adopt(cls,runtime_pin,evidence):
        obj=cls(runtime_pin)
        need(isinstance(evidence,dict)and evidence.get('runtime_spec')==runtime_pin,'inherited runtime ticket binding differs')
        entries=evidence.get('ancestor_locks');need(isinstance(entries,list)and len(entries)==2 and
            {e['path']for e in entries}==set(obj.compiled['ancestor_paths'])and len({e['fd']for e in entries})==2,'ancestor inherited lock set differs')
        try:
            for entry in entries:
                fd=entry['fd'];need(type(fd)is int and fd>2,'inherited ancestor FD required');obj._identity(entry,fd)
                obj.handles.append(dict(entry,fd=os.dup(fd)))
            spec=obj.compiled['spec'];obj.current=Q.ExistingProducerLease.adopt_inherited(spec['index']['path'],spec['index']['sha256'],obj.compiled['terminal_pin'],evidence['current_old_locks'])
            claim_pin=evidence.get('unique_producer_claim')
            need(isinstance(claim_pin,dict)and claim_pin.get('path')==obj.compiled['unique_claim_path'],'inherited unique producer claim missing')
            actual=pinned(claim_pin)
            need({k:v for k,v in actual.items()if k not in {'first_runtime_spec','created_utc'}}==obj.compiled['unique_claim_core'],'inherited unique producer claim differs')
            obj.unique_claim_pin=claim_pin
            obj.held=True;obj.verify();return obj
        except BaseException:obj.__exit__(None,None,None);raise

    def assert_held(self):
        need(self.held and self.current is not None,'runtime lineage lease is not held')
        for entry in self.handles:self._identity(entry,entry['fd'])
        self.current.assert_held()

    def verify(self):
        self.assert_held()
        for pin in self.compiled['pins']:R.pinned_file(pin)
        need(R.ref_unpinned(self.compiled['lineage_path'])==self.compiled['lineage_value'],'previous permanent claim changed')
        for check in self.compiled['old']['checks']:R.pinned_file(check['script'])
        need(self.unique_claim_pin is not None,'unique producer claim not admitted')
        R.pinned_file(self.unique_claim_pin)
        self.current.verify_frozen_state()

    @property
    def pass_fds(self):
        self.assert_held();return tuple(h['fd']for h in self.handles)+self.current.pass_fds

    @property
    def ticket_evidence(self):
        self.assert_held();return dict(runtime_spec=self.pin,ancestor_locks=[dict(h)for h in self.handles],current_old_locks=self.current.fd_evidence,unique_producer_claim=self.unique_claim_pin)

    def gates(self,output,*,runner=subprocess.run):
        """仅GPU父调用；原W2原字节原参数，挂载核同原profile，不读百万侧车。"""
        self.verify();c=self.compiled;old=c['old'];R.frozen_bundle(old)
        R.frozen_roots(Path(old['code_root'])/'gpu_launch.py',Path(old['task_root']),Path(old['device_lock_root']))
        # Python可执行原件必须同前一run固定的二进制，而不是PATH另找解释器。
        R.pinned_file(old['python_binary']);need(Path(old['python']).resolve(strict=True)==Path(old['python_binary']['path'])==Path(sys.executable).resolve(strict=True),'original/actual interpreter differs')
        command=c['original_commands'][0];result=runner(command,capture_output=True,text=True,timeout=600,check=False)
        output=R.physical(output);output.mkdir(parents=True,exist_ok=True)
        H.P.write_new(output/'bounded-original-W2.json',dict(argv=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        need(result.returncode==0,'original W2 quality gate failed')
        mounts=verify_mounts(c,runner=runner);H.P.write_new(output/'bounded-mounts.json',mounts)
        R.frozen_bundle(dict(code_root=c['spec']['new_code_root'],bundle=c['spec']['new_bundle']))
        self.verify();return mounts

    def queue(self,args,contract):
        self.verify();spec=self.compiled['spec']
        need(spec['index']=={'path':str(args.shard_index),'sha256':args.index_sha}and
             spec['new_contract']=={'path':str(args.contract),'sha256':args.contract_sha},'runtime contract/index arguments differ')
        need(Path(args.queue_root)==Path(self.compiled['index']['new_queue_root'])and Path(args.output_root)==Path(self.compiled['index']['new_sidecar_root']), 'runtime output arguments differ')
        need(contract==self.compiled['new_contract'],'runtime new contract differs')
        return Q.BoundedWorkQueue(args.shard_index,args.index_sha,args.queue_root,args.contract_sha,Path(args.output_root)/contract['version'],self.current)

    def __exit__(self,*unused):
        self.held=False
        if self.current is not None:self.current.__exit__(None,None,None);self.current=None
        for entry in reversed(self.handles):os.close(entry['fd'])
        self.handles=[]
