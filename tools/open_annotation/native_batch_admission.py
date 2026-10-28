"""原生合批的独立正式准入v2；CPU格式或旧单窗证据均不授运行资格。

只读已经完成的有限试验、完整批次/行映射收据及质量审阅；不创建模型、
不读取原音、不扫描正式全库、不建数据库。正式合同与选定pilot仅允许
purpose/version及默认关闭声明不同，模型数学与工程执行源必须完全相同。
"""
from __future__ import annotations
import math
import argparse
import json
import shutil
from pathlib import Path

if __package__:
    from . import bounded_speed_handoff as H
    from . import resume_existing_annotation as R
else:
    import bounded_speed_handoff as H
    import resume_existing_annotation as R

SCHEMA='annotation-native-selected-pilot-admission/v2'
QUALITY_SCHEMA='annotation-native-batch-quality-review/v2'
RUNTIME_SCHEMA='annotation-bounded-native-runtime/v2'
NATIVE_POLICY='asr_fixed_cohort_fa_single/v1'
TRACK_SCHEMA='redimnet2-annotation-native-batch-track/v1'
COMPARE_SCHEMA='redimnet2-annotation-native-batch-comparison/v1'
need=H.need


def pinned(pin):return H.read_pin(pin,256*1024**2)[0]



def feature_source_identity(contract):
    """新显式逐源特征协议固定七源及配置；旧无声明合同不隐式升级。"""
    runtime=contract.get('runtime',{})
    if 'asr_preprocessing'not in runtime:
        need('asr_preprocessing_sources'not in contract,'preprocessing sources need explicit feature protocol')
        return None
    from contracts import asr_feature_profile
    need(asr_feature_profile(contract)=={'policy':'official_b1_features_then_pad/v1'},'explicit official B1 feature policy required')
    value=contract.get('asr_preprocessing_sources')
    roles={'whisper_feature_extractor','sequence_feature_extractor','batch_feature','processor_mixin','audio_utils','qwen_processor','sdk_normalizer'}
    need(isinstance(value,dict)and set(value)=={'schema','files','processor_config'}and value['schema']=='annotation-official-b1-feature-sources/v1'and
         isinstance(value['files'],dict)and set(value['files'])==roles,'explicit seven preprocessing sources and processor config required')
    for ref in [*value['files'].values(),value['processor_config']]:
        need(isinstance(ref,dict)and set(ref)=={'path','sha256'}and H.is_sha(ref['sha256']),'explicit preprocessing source/config SHA pin required')
        R.physical(ref['path'])
    need(all(Path(ref['path']).suffix=='.py'for ref in value['files'].values())and Path(value['processor_config']['path']).name=='preprocessor_config.json','actual Python sources and processor configuration required')
    return value


def formal_equivalent(pilot,new,purpose):
    from contracts import asr_batch_profile,NATIVE_FORMAL_DECLARATION
    need(pilot.get('purpose')==('cpu_fixture'if purpose=='cpu_fixture'else'pilot_validation'),'actual native pilot required')
    need(feature_source_identity(pilot)==feature_source_identity(new),'native formal preprocessing sources differ')
    profile=asr_batch_profile(pilot);need(profile is not None and profile['max_requests']in (4,8),'selected native width must be explicitly 4 or 8')
    expected={k:v for k,v in pilot.items()if k not in {'purpose','version','native_formal'}}
    actual={k:v for k,v in new.items()if k not in {'purpose','version','native_formal'}}
    need(expected==actual,'native production and pilot math/runtime/source fields must be exactly identical')
    if purpose=='production':
        from contracts import engine_modules_for_contract
        need(set(new.get('engine_source_files',{}))==set(engine_modules_for_contract(new)) and
             all(H.is_sha(v)for v in new['engine_source_files'].values()),'native exact declared math source SHA set required')
        need(new.get('purpose')=='silver_annotation'and new.get('native_formal')==NATIVE_FORMAL_DECLARATION and
             new['native_formal'].get('default_enabled')is False,'default-disabled native formal contract structure required')
    else:need(new.get('purpose')=='cpu_fixture'and 'native_formal'not in new,'CPU fixture cannot carry formal declaration')
    need(new.get('version')!=pilot.get('version'),'native formal producer must use distinct version')
    return profile


def compare_native_profiles(baseline,candidate):
    from contracts import asr_batch_profile
    a=asr_batch_profile(baseline);b=asr_batch_profile(candidate)
    need(a is not None and b is not None and a['max_requests']==1 and b['max_requests']in (4,8),'native comparison must be explicit B1 to B4/8')
    need(feature_source_identity(baseline)==feature_source_identity(candidate),'native baseline/candidate preprocessing sources differ')
    for key in ('models','source_files','alignment','decoder','quality_policy','engine_source_files','asr_preprocessing_sources'):
        need(baseline.get(key)==candidate.get(key),'native baseline/candidate producer identity differs: '+key)
    left={k:v for k,v in baseline['runtime'].items()if k!='asr_batching'};right={k:v for k,v in candidate['runtime'].items()if k!='asr_batching'}
    need(left==right and left.get('execution_policy')==NATIVE_POLICY and left.get('asr_compute')=={'dtype':'float32','attention':'sdpa'}and
         {k:v for k,v in a.items()if k!='max_requests'}=={k:v for k,v in b.items()if k!='max_requests'},'only native batch width may differ')


def _changed_windows(row):
    return {w['window_id']for w in row.get('windows',[])if not w.get('both_present')or not w.get('input_identity_equal')or
            any(value is False for group in ('ASR','alignment')for key,value in w.get(group,{}).items()if key.endswith('_equal'))}


def _metadata_only(row):
    if row.get('changed_fields'):return False
    for w in row.get('windows',[]):
        if not w.get('both_present')or not w.get('input_identity_equal'):return False
        expected=('status_equal','raw_text_equal','normalized_text_equal','language_equal','normalization_audit_equal','effective_generated_token_ids_equal','eos_observed_equal')
        if any(w.get('ASR',{}).get(k)is not True for k in expected):return False
        # 新显式预处理协议的有效mel/mask/prompt差异属于计算输入变化，不能当元数据。
        if any(value is not True for key,value in w.get('ASR',{}).items()if key.startswith('preprocessing_')and key.endswith('_equal')):return False
        if any(v is not True for k,v in w.get('alignment',{}).items()if k.endswith('_equal')):return False
    return True


def check_quality(pin,comparison,candidate_pin,purpose):
    q=pinned(pin);need(q.get('schema')==QUALITY_SCHEMA and q.get('purpose')==purpose and q.get('candidate_contract')==candidate_pin and
        q.get('review_complete')is True and q.get('all_changed_rows_dispositioned')is True and q.get('text_quality_accepted')is True and
        q.get('complete_word_boundary_quality_accepted')is True and q.get('candidate_labels_released')is False,'native text/boundary review incomplete')
    rows=comparison.get('rows');need(isinstance(rows,list)and len(rows)==comparison['source_count']and len({r.get('source_id')for r in rows})==len(rows),'complete finite native comparison rows required')
    for row in rows:
        for window in row.get('windows',[]):
            for group in ('ASR','alignment'):
                need(all(type(value)is bool for key,value in window.get(group,{}).items()if key.endswith('_equal')),
                     'native comparison equality flags must be actual booleans; numeric false cannot hide review')
    changed={r['source_id']:r for r in rows if r.get('changed_fields')or _changed_windows(r)}
    dispositions=q.get('dispositions');need(isinstance(dispositions,list)and len(dispositions)==len(changed)and
        {v.get('source_id')for v in dispositions}==set(changed),'every changed native source needs one explicit disposition')
    refs=[]
    for d in dispositions:
        row=changed[d['source_id']];need(set(d.get('window_ids',[]))==_changed_windows(row)and len(d['window_ids'])==len(set(d['window_ids'])),'native changed-window review coverage differs')
        cls=d.get('classification');need(cls in {'metadata_only_verified','text_and_boundary_reviewed'},'explicit native difference class required')
        if cls=='metadata_only_verified':need(_metadata_only(row),'text or boundary differences cannot be called padding-only metadata')
        else:need(d.get('text_checked')is True and d.get('leading_trailing_neighbor_checked')is True,'native text/cut-head/cut-tail/neighbor review required')
        evidence=d.get('review_evidence');need(isinstance(evidence,list)and evidence,'actual per-difference review evidence required');refs.extend(evidence)
    evidence=q.get('review_evidence');need(isinstance(evidence,list)and evidence,'actual native quality evidence required');refs.extend(evidence)
    for ref in refs:R.pinned_file(ref)
    return refs


CONTROL_REVIEW_SCHEMA='annotation-native-control-source-review/v2'
REVIEWABLE_CONTROL={'native_batch_admission.py','native_batch_readback.py','native_batch_queue_handoff.py','bounded_speed_runtime.py'}


def check_formal_bundle(selected,pilot_bundle_pin,new_contract,purpose):
    """仅独立审阅的非ENGINE控制修正可换正式包；声明的数学源一字节也不能变。"""
    old=pinned(pilot_bundle_pin);formal_pin=selected.get('formal_execution_bundle',pilot_bundle_pin);formal=pinned(formal_pin)
    need(isinstance(old,dict)and isinstance(formal,dict),'actual native bundle member mappings required')
    for name,digest in new_contract.get('engine_source_files',{}).items():
        need(old.get(name)==formal.get(name)==digest,'native formal bundle changed actual tested math ENGINE source: '+name)
    changed={name:{'pilot':old.get(name),'formal':formal.get(name)}for name in set(old)|set(formal)if old.get(name)!=formal.get(name)}
    refs=[pilot_bundle_pin,formal_pin]
    if changed:
        need(set(changed)<=REVIEWABLE_CONTROL and all(v['formal']is not None for v in changed.values()),'unreviewable native formal source change/deletion')
        review_pin=selected.get('control_source_review');need(isinstance(review_pin,dict),'native formal control change requires explicit review pin')
        review=pinned(review_pin)
        need(review.get('schema')==CONTROL_REVIEW_SCHEMA and review.get('purpose')==purpose and review.get('pilot_bundle')==pilot_bundle_pin and
            review.get('formal_bundle')==formal_pin and review.get('changed_control_sources')==changed and
            review.get('math_engine_source_files')==new_contract.get('engine_source_files')and review.get('GPU_math_source_unchanged')is True and
            review.get('independent_review_complete')is True and review.get('CPU_tests_passed')is True and review.get('review_evidence'),
            'native formal control review does not bind exact source differences')
        refs.extend([review_pin,*review['review_evidence']])
    # 复制到B内的正式code可改路径，文件内容和声明成员仍逐个核。
    R.frozen_bundle(dict(code_root=str(Path(formal_pin['path']).parent),bundle=formal_pin))
    for ref in refs:R.pinned_file(ref)
    return formal_pin,refs


def check_native_selected_pilot(pin,new_contract,purpose):
    """严格新schema；缺实际3000/512或完整packet/readback时永不授正式。"""
    selected=pinned(pin);need(selected.get('schema')==SCHEMA and selected.get('purpose')==purpose,'explicit native selected-admission v2 required; old single-window evidence refused')
    candidate=pinned(selected['candidate_contract']);profile=formal_equivalent(candidate,new_contract,purpose)
    baseline=pinned(selected['baseline_track']);main=pinned(selected['main_track']);resume=pinned(selected['recovery_track']);basec=pinned(baseline['contract']);compare_native_profiles(basec,candidate)
    need(all(t.get('schema')==TRACK_SCHEMA and t.get('execution_policy')==NATIVE_POLICY for t in (baseline,main,resume))and
        main['contract']==resume['contract']==selected['candidate_contract']and baseline['index']==main['index'],'native track schema/source/producer differs')
    name='native-fp32-sdpa-b'+str(profile['max_requests']);need(selected.get('selected_profile_name')==name and selected.get('selected_asr_batching')==profile,'selected native batch profile differs')
    speed=pinned(selected['comparison']);recovery=pinned(selected['recovery_comparison']);result=pinned(selected['pilot_result'])
    for value,mode in ((speed,'speed'),(recovery,'recovery')):
        need(value.get('schema')==COMPARE_SCHEMA and value.get('mode')==mode and value.get('candidate_contract_sha256')==selected['candidate_contract']['sha256']and
            value.get('all_planned_sources_accounted')is True and value.get('full_sidecar_commit_readback_verified')is True and
            value.get('native_whole_batch_readback_verified')is True and value.get('database_created')is False,'native complete sidecar/whole-packet comparison required')
    n=speed.get('source_count');rc=recovery.get('source_count');need(type(n)is int and 0<n<=4096 and type(rc)is int and 0<rc<=n and
        (purpose!='production'or(n>=3000 and rc==512)),'native formal requires real 3000 and exactly512 recovery')
    ratio=speed.get('cold_end_to_end_speed_ratio');need(type(ratio)in (int,float)and math.isfinite(ratio)and ratio>1 and speed.get('cold_speed_ratio_valid')is True and
        all(speed.get(k,{}).get('actual_compute_execution_verified')is True and speed[k].get('native_whole_batch_readback_verified')is True and
            speed[k].get('all_sources_recomputed_cold')is True and speed[k].get('totals',{}).get('asr_rows',0)>0 for k in ('baseline_performance','candidate_performance')),'actual native cold end-to-end speed must exceed one')
    need(recovery.get('strict_recovery_native_results_equal')is True and recovery.get('strict_native_results_equal')is True,'native recovery must preserve complete original results')
    totals=recovery.get('candidate_performance',{}).get('totals',{});need(totals.get('sidecars_committed')==0 and totals.get('sidecar_cache_hits')==rc and
        totals.get('asr_rows')==totals.get('aligner_rows')==0,'native recovery may not repeat ASR/FA or sidecar publication')
    need(speed.get('baseline_spec_sha256')==selected['baseline_track']['sha256']and speed.get('candidate_spec_sha256')==selected['main_track']['sha256']and
        speed.get('baseline_contract_sha256')==baseline['contract']['sha256']and recovery.get('baseline_spec_sha256')==selected['main_track']['sha256']and
        recovery.get('candidate_spec_sha256')==selected['recovery_track']['sha256'],'native comparison track SHA chain differs')
    need(result.get('schema')=='annotation-native-batch-pilot-result/v1'and result.get('purpose')==purpose and
        result.get('all_finite_audits_completed')is True and result.get('database_created')is False,'finished actual native pilot result required')
    plan=pinned(result['plan']);matrix=pinned(plan['main_prepared']);need(plan.get('schema')=='annotation-native-batch-pilot/v1'and
        matrix.get('schema')=='annotation-native-batch-prepared/v1'and matrix.get('source_count')==n and
        matrix['variants'][name]['track']==selected['main_track']and matrix['variants']['native-fp32-sdpa-b1']['track']==selected['baseline_track'],'native result plan/matrix not bound to selected tracks')
    refs=[selected[k]for k in ('candidate_contract','baseline_track','main_track','recovery_track','comparison','recovery_comparison','pilot_result','quality_review')]+[baseline['contract'],result['plan'],plan['main_prepared']]
    bundles={};steps=result.get('steps',[])
    for role,track in (('baseline',baseline),('main',main),('recovery',resume)):
        ev=selected['resource_evidence'][role];root=Path(track['resources_root']);need(set(ev)=={'parent_result','resource_return','terminal_drain'},'complete native resource pins required')
        for key,filename in (('parent_result','parent_result.json'),('resource_return','resource_return_double_probe.json'),('terminal_drain','terminal_drain_complete.json')):
            need(ev[key]['path']==str(root/filename),'native resource path differs')
        parent=pinned(ev['parent_result']);idle=pinned(ev['resource_return']);drain=pinned(ev['terminal_drain']);refs.extend(ev.values())
        variant='native-fp32-sdpa-b1'if role=='baseline'else name;bundle=matrix['variants'][variant]['freeze']['bundle_sha256'];bundles[role]=bundle
        need(parent.get('schema')=='open-annotation-independent-gpu-run/v1'and parent.get('contract_sha256')==track['contract']['sha256']and
            parent.get('index_sha256')==track['index']['sha256']and parent.get('bundle_sha256')==bundle and parent.get('worker_count')==2 and
            parent.get('resource_returned')is True and parent.get('full_sidecar_readback')is True and parent.get('verification',{}).get('passed')is True and
            parent['verification'].get('database_created')is False and parent['verification'].get('source_disjointness_storage')=='bounded_pilot_memory_set','actual native two-GPU terminal/readback evidence incomplete')
        need(idle.get('interval_seconds',0)>=5 and len(idle.get('snapshots',[]))==2 and
            all(not snap['compute_processes']and all(d['used_mib']<=256 for d in snap['devices'])for snap in idle['snapshots'])and drain.get('all_attested_contexts_absent')is True,'actual native CUDA drain/idle missing')
        stage='recovery'if role=='recovery'else'main';pname='native-fp32-sdpa-b1'if role=='baseline'else name
        need(any(st.get('stage')==stage and st.get('profile')==pname and st.get('parent_result_sha256')==ev['parent_result']['sha256']and st.get('resource_returned')is True and
            (st.get('sources')==rc and st.get('new_ASR_calls')==st.get('new_FA_calls')==0 if role=='recovery'else st.get('source_count')==n)for st in steps),'native result step does not bind actual selected resources')
    need(bundles['baseline']==bundles['main']==bundles['recovery'],'native trial executable bundles must be identical')
    chosen=matrix['variants'][name]['freeze']
    pilot_bundle_pin=dict(path=str(Path(chosen['frozen_copy'])/'bundle_files.json'),sha256=bundles['main'])
    formal_bundle,extra=check_formal_bundle(selected,pilot_bundle_pin,new_contract,purpose);refs.extend(extra)
    need(pinned(selected['quality_review']).get('comparison')==selected['comparison'],'native quality review does not bind actual comparison')
    refs.extend(check_quality(selected['quality_review'],speed,selected['candidate_contract'],purpose))
    # 收据由有限native auditor已核的原件生成；逐包/plan/行映射本身仍要核，不信布尔口号。
    if __package__:from .native_batch_readback import verify_readback_receipt,verify_comparison_maps
    else:from native_batch_readback import verify_readback_receipt,verify_comparison_maps
    readbacks={}
    for role,track,nrows in (('baseline',baseline,n),('main',main,n),('recovery',resume,rc)):
        proof=selected['native_readbacks'][role];refs.append(proof)
        value=verify_readback_receipt(proof,selected[role+'_track'],track,pinned(track['contract']),bundles[role],nrows,purpose);readbacks[role]=value;refs.extend(value['pins'])
        expected=speed['baseline_performance'if role=='baseline'else'candidate_performance']if role!='recovery'else recovery['candidate_performance']
        need(value['proof']['performance']==expected and value['proof']['parent_result']==selected['resource_evidence'][role]['parent_result'],'native readback metrics/parent not bound to actual comparison')
    verify_comparison_maps(speed,{'baseline':readbacks['baseline'],'candidate':readbacks['main']})
    verify_comparison_maps(recovery,{'baseline':readbacks['main'],'candidate':readbacks['recovery']})
    # 恢复512可引用原main整包，但其源/侧车/窗口原件必须正是main中的同一成员。
    for sid,value in readbacks['recovery']['sources'].items():need(readbacks['main']['sources'].get(sid)==value,'native recovery source/sidecar/packet changed from main')
    need(any(len(item['requests'])==profile['max_requests']for item in readbacks['main']['proof']['native_packets']),
         'selected native maximum batch width was not actually exercised')
    selected['verified_evidence_pins']=refs
    selected['verified_native_bundle_sha256']=formal_bundle['sha256']
    selected['verified_pilot_bundle_sha256']=bundles['main']
    return selected


def prepare_formal_code(pilot_code,pilot_bundle_pin,candidate_pin,output,*,overlay_pin=None):
    """只复制实际已冻候选包，最多替换四CPUhelper；不从工作目录重freeze。"""
    from contracts import asr_batch_profile,engine_modules_for_contract,relative_path
    code=R.physical(pilot_code);need(pilot_bundle_pin['path']==str(code/'bundle_files.json'),'actual native pilot bundle path differs')
    old=pinned(pilot_bundle_pin);R.frozen_bundle(dict(code_root=str(code),bundle=pilot_bundle_pin));candidate=pinned(candidate_pin)
    need(candidate.get('purpose')=='pilot_validation'and asr_batch_profile(candidate)is not None and asr_batch_profile(candidate)['max_requests']in (4,8),'actual selected native B4/8 pilot contract required for code preparation')
    relative_path(candidate.get('version'));need('/'not in candidate['version'],'native candidate version must be one safe component')
    math=candidate.get('engine_source_files');need(isinstance(math,dict)and set(math)==set(engine_modules_for_contract(candidate))and
        all(old.get(name)==digest for name,digest in math.items()),'actual declared pilot math source pins differ from frozen bundle')
    output=R.physical(output);need(not any((p/'AGENTS.md').is_file()and(p/'redimnet2').is_dir()for p in output.parents),'formal code copy must stay outside maintenance repository')
    need(output.parent.is_dir()and not output.exists()and output!=code and code not in output.parents and output not in code.parents,'fresh independent formal code output required')
    overlay={}
    if overlay_pin is not None:
        spec=pinned(overlay_pin);need(spec.get('schema')=='annotation-native-cpu-helper-overlay/v2'and isinstance(spec.get('files'),dict)and
            set(spec['files'])<=REVIEWABLE_CONTROL,'only four explicitly pinned CPU helper replacements allowed')
        overlay=spec['files']
        for ref in overlay.values():R.pinned_file(ref)
    output.mkdir(mode=0o700)
    for name,digest in old.items():
        relative=Path(name);need(not relative.is_absolute()and '..'not in relative.parts and relative.parts,'unsafe native pilot bundle member')
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code/relative,target)
    for name,ref in overlay.items():
        shutil.copyfile(ref['path'],output/name)
        need(R.file_sha(output/name)==ref['sha256'],'CPU helper changed while copying pinned overlay')
        R.pinned_file(ref)
    current={name:R.file_sha(output/name)for name in old}
    for name in overlay:
        if name not in current:current[name]=R.file_sha(output/name)
    need(all(current.get(name)==digest for name,digest in math.items()),'formal preparation changed tested native math source')
    changes={name:{'pilot':old.get(name),'formal':current.get(name)}for name in set(old)|set(current)if old.get(name)!=current.get(name)}
    need(set(changes)<=REVIEWABLE_CONTROL,'formal preparation included unreviewable source change')
    bundle_path=output/'bundle_files.json';H.P.write_new(bundle_path,current);formal_pin=dict(path=str(bundle_path),sha256=R.file_sha(bundle_path))
    R.frozen_bundle(dict(code_root=str(output),bundle=formal_pin));R.frozen_bundle(dict(code_root=str(code),bundle=pilot_bundle_pin))
    # 只复制实际候选pilot合同原字节；新silver由已复制包内的独立数据工厂另行生成。
    contract_path=output/'contracts'/(candidate['version']+'.json');contract_path.parent.mkdir();shutil.copyfile(R.pinned_file(candidate_pin),contract_path)
    need(R.file_sha(contract_path)==candidate_pin['sha256'],'copied native pilot contract differs')
    receipt=dict(schema='annotation-native-formal-code-preparation/v2',pilot_bundle=pilot_bundle_pin,formal_bundle=formal_pin,candidate_contract=candidate_pin,
        copied_pilot_contract=dict(path=str(contract_path),sha256=candidate_pin['sha256']),math_engine_source_files=math,changed_control_sources=changes,
        overlay=overlay_pin,math_source_unchanged=True,independent_review_complete=False,formal_runtime_admitted=False,GPU_executed=False,old_production_modified=False)
    H.P.write_new(output/'native-formal-code-preparation.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare-code',))
    p.add_argument('--pilot-code',required=True);p.add_argument('--pilot-bundle-sha',required=True);p.add_argument('--candidate-contract',required=True);p.add_argument('--candidate-sha',required=True)
    p.add_argument('--output',required=True);p.add_argument('--overlay');p.add_argument('--overlay-sha');a=p.parse_args()
    need(bool(a.overlay)==bool(a.overlay_sha),'CPU helper overlay path/SHA must be paired')
    value=prepare_formal_code(a.pilot_code,dict(path=str(Path(a.pilot_code)/'bundle_files.json'),sha256=a.pilot_bundle_sha),
        dict(path=a.candidate_contract,sha256=a.candidate_sha),a.output,overlay_pin=dict(path=a.overlay,sha256=a.overlay_sha)if a.overlay else None)
    print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
