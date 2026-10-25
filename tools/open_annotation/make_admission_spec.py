"""LATEST SCOPE: build actual parameterized 30/3000/512 three-train W8 spec.
CPU-only. No platform submission, GPU execution or historical-mixed gate shortcut.
"""
from pathlib import Path
import argparse,json
from cloud_gate import need,sha,verify,Reader,SOURCE_SCOPE
from make_cloud_launcher import validate_spec
def reference(path):
    path=Path(path);need(path.is_absolute() and not path.is_symlink(),'absolute physical evidence path required');return {'path':str(path),'sha256':sha(path.read_bytes())}
def build(a):
    root=a.validation_root;code=a.code_root;data=a.data_root;BASE=str(a.task_root)
    spec={'schema':'ultra-eight-three-train-spec/v3','validation_UUIDs':a.validation_uuids.split(','),'source_scope':SOURCE_SCOPE,'not_before_utc':a.not_before_utc,'code_root':str(code),'bundle':reference(code/'bundle_files.json'),'pilot_contract':reference(a.pilot_contract),'silver_contract':reference(a.silver_contract),'full_index':reference(a.full_index),'full_filter_summary':reference(a.full_filter_summary),'full_filter_source':reference(a.full_filter_source),'full_filter_audit':reference(a.full_filter_audit),'full_filter_audit_source':reference(a.full_filter_audit_source),'diagnostic_protection_reference':reference(a.diagnostic_protection_reference),'supervisor':reference(a.supervisor),'sequence_start':reference(root/'ultra-sequence-start.json'),'sequence_complete':reference(root/'ultra-sequence-complete.json'),'independent_review':reference(a.independent_review),'cloud':{'task_root':BASE,'run_root':str(a.run_root),'control_root':str(a.control_root),'python':str(a.python),'device_lock_root':str(a.device_lock_root),'priority':6,'platform_workers':1,'gpus_per_worker':8,'max_runtime_seconds':2592000,'mounts':json.loads(a.mount_profile.read_text())['mounts']}}
    for key,run,index,attempt,count in [('vertical',a.vertical_run,a.vertical_index,1,a.vertical_count),('main',a.main_run,a.main_index,1,a.main_count),('recovery',a.recovery_run,a.recovery_index,2,a.recovery_count)]:
        home=root/run;resources=home/f'resources-attempt{attempt}';spec[key]={'run_root':str(home),'resources':str(resources),'parent_result':reference(resources/'parent_result.json'),'index':reference(data/index),'source_count':count}
    spec['recovery']['native_comparison']=reference(a.native_comparison)
    if a.baseline_receipt:spec['main']['baseline_receipt']=reference(a.baseline_receipt)
    validate_spec(spec);result=verify(spec,Reader(spec));return spec,result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['task-root','python','device-lock-root','code-root','validation-root','data-root','pilot-contract','silver-contract','full-index','full-filter-summary','full-filter-source','full-filter-audit','full-filter-audit-source','diagnostic-protection-reference','supervisor','independent-review','run-root','control-root','mount-profile','native-comparison','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--baseline-receipt',type=Path)
    for key,default in [('vertical-run','vertical30'),('main-run','P1-training3000-baseline'),('recovery-run','P2-training512-partial-resume'),('vertical-index','vertical30/index.json'),('main-index','representative3000/index-clock-complete-v2.json'),('recovery-index','recovery512/index.json')]:p.add_argument('--'+key,default=default)
    for key,default in [('vertical-count',30),('main-count',3000),('recovery-count',512)]:p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--validation-uuids',required=True)
    p.add_argument('--not-before-utc',required=True);a=p.parse_args();spec,result=build(a);raw=(json.dumps(spec,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'schema':'ultra-three-train-admission-spec-build/v2','spec_path':str(a.output),'spec_sha256':sha(raw),'actual_evidence_gate_passed':result['passed'],'GPU_execution_or_submission':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
