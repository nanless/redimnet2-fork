"""只复制已核冻结控制树并准备中途恢复计划；finish仅执行默认CPU检查。

此工具不重新冻结数学源码，不修改来源树，不带GPU执行开关。控制器须
独立上传至新控制目录且摘要等于已审版本；所有来源文件及九依赖复核后
才生成供外层人工读取的未来入口命令，命令不会在本工具中执行。
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess

APPROVED_CONTROLLER_SHA='231649b4ac0652824ed3f23d82ae639f82d81e696a1a1a969c26c5e1481a6c90'
CONTROLLER='run_native_partial_recovery_probe.py'
DEPENDENCIES=('contracts.py','run_bounded_speed_pilot.py','run_native_batch_pilot.py','audit_native_transformers_batch.py',
    'gpu_common.py','readonly_progress.py','gpu_vendor/resources.py','gpu_vendor/topology.py','gpu_vendor/__init__.py')
ANCHOR='src_0010c3bdaebf5268747c4f2707a36c3234d97791cd05f8fc95e17263e6b48852'


def need(ok,message):
    if not ok:raise ValueError(message)


def physical(value):
    p=Path(value);need(p.is_absolute()and '..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents)),'physical absolute path required');return p


def disjoint(a,b):return a!=b and a not in b.parents and b not in a.parents


def sha(path):
    p=physical(path);st=p.lstat();need(stat.S_ISREG(st.st_mode)and st.st_size<=8*1024**2,'bounded source/control file required')
    raw=p.read_bytes();after=p.stat();need((st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'control file changed during read')
    return hashlib.sha256(raw).hexdigest()


def pin(ref):
    p=physical(ref['path']);need(sha(p)==ref['sha256'],'control input SHA differs');return json.loads(p.read_bytes())


def write(path,value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with path.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def sources(root):
    result={}
    for p in root.rglob('*'):
        need(not p.is_symlink(),'source code tree contains symlink')
        if p.is_file():
            need(len(result)<512,'source code tree unexpectedly large');result[str(p.relative_to(root))]=sha(p)
    need(result and CONTROLLER not in result,'source build must be original pre-probe immutable tree');return result


def stage(source,prepared_ref,control,output,protected,operation_sha):
    source,control,output=map(physical,(source,control,output));protected=[physical(p)for p in protected]
    need(source.is_dir()and not control.exists()and not output.exists()and control.parent.is_dir()and output.parent.is_dir(),'fresh independent roots required')
    need(len(protected)==len(set(protected))==4,'four distinct existing-task protected roots required')
    need(all(disjoint(control,p)and disjoint(output,p)for p in [source,*protected])and disjoint(control,output),'control/output overlaps existing tasks or code')
    matrix=pin(prepared_ref);need(matrix.get('schema')=='annotation-native-batch-prepared/v1'and matrix.get('source_count')==30,'fixed native screen30 required')
    v=matrix['variants']['native-fp32-sdpa-b4'];code=physical(v['freeze']['frozen_copy'])
    bundle_ref=dict(path=str(code/'bundle_files.json'),sha256=v['freeze']['bundle_sha256']);bundle=pin(bundle_ref)
    source_files=sources(source)
    for name in DEPENDENCIES:need(source_files.get(name)==bundle.get(name)==sha(code/name),'nine dependency mismatch: '+name)
    shutil.copytree(source,control);control.chmod(0o700)
    for name,value in source_files.items():need(sha(control/name)==value==sha(source/name),'copy/source bytes changed: '+name)
    plan=dict(schema='annotation-native-mid-cohort-probe/v1',purpose='production',candidate_prepared=prepared_ref,
        profile='native-fp32-sdpa-b4',anchor_source_id=ANCHOR,protected_roots=[str(p)for p in protected],
        wall_budget_seconds=1800,controller_sha256=APPROVED_CONTROLLER_SHA)
    plan_ref=write(control/'probe-plan-v1.json',plan)
    receipt=dict(schema='native-partial-recovery-control-stage/v1',source_root=str(source),source_files=source_files,
        source_file_count=len(source_files),control_root=str(control),output_root=str(output),prepared=prepared_ref,
        candidate_bundle=bundle_ref,dependencies={p:source_files[p]for p in DEPENDENCIES},plan=plan_ref,
        expected_controller_sha256=APPROVED_CONTROLLER_SHA,python=v['argv'][0],operation_source_sha256=operation_sha,
        GPU_executed=False,controller_uploaded=False,source_tree_modified=False,refreeze_performed=False)
    write(control/'control-stage-receipt-v1.json',receipt);return receipt


def finish(control,*,runner=subprocess.run):
    control=physical(control);stage_path=control/'control-stage-receipt-v1.json';s=json.loads(stage_path.read_bytes())
    need(s['control_root']==str(control)and s['expected_controller_sha256']==APPROVED_CONTROLLER_SHA,'stage identity differs')
    for name,value in s['source_files'].items():need(sha(control/name)==value==sha(Path(s['source_root'])/name),'copied/original source changed')
    bundle=pin(s['candidate_bundle']);code=Path(s['candidate_bundle']['path']).parent
    for name in DEPENDENCIES:need(s['dependencies'][name]==sha(control/name)==bundle[name]==sha(code/name),'frozen/current dependency differs')
    controller=control/CONTROLLER;need(sha(controller)==APPROVED_CONTROLLER_SHA,'approved controller not uploaded or changed')
    tree=ast.parse(controller.read_bytes());decl=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CONTROL_DEPENDENCIES'for t in n.targets))
    need(ast.literal_eval(decl.value)==DEPENDENCIES,'approved controller dependency declaration differs')
    pin(s['plan']);plan=pin(s['plan']);need(plan['candidate_prepared']==s['prepared']and plan['profile']=='native-fp32-sdpa-b4','fixed plan differs')
    output=physical(s['output_root']);need(not output.exists(),'probe GPU/result root must remain absent before CPU check')
    argv=[s['python'],'-B',str(controller),'--plan',s['plan']['path'],'--plan-sha',s['plan']['sha256'],'--output',str(output)]
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    result=runner(argv,cwd=control,env=env,capture_output=True,text=True,timeout=180,check=False)
    stdout_ref=write(control/'cpu-default-check-stdout-v1.json',dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode,argv=argv))
    need(result.returncode==0,'default CPU check failed; original stdout/stderr retained')
    check=json.loads(result.stdout);need(check.get('checked')is True and check.get('GPU_executed')is False
        and check.get('source_count')==30 and check.get('first_group_windows',0)>=9 and check.get('partial_recovery_verified')is False,
        'default check did not prove fixed30 CPU-only preparation')
    need(not output.exists(),'default CPU check unexpectedly created result root')
    entry='cd '+shlex.quote(str(control))+' && exec '+shlex.join(argv+['--execute'])
    receipt=dict(schema='native-partial-recovery-control-ready/v1',stage=dict(path=str(stage_path),sha256=sha(stage_path)),
        plan=s['plan'],controller=dict(path=str(controller),sha256=sha(controller)),source_files=s['source_files'],
        dependencies=s['dependencies'],candidate_bundle=s['candidate_bundle'],stdout=stdout_ref,CPU_check=check,
        entry_command_for_review_only=entry,GPU_executed=False,platform_submitted=False,output_root_still_absent=True)
    write(control/'control-ready-receipt-v1.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('stage','finish'),required=True)
    parser.add_argument('--control',type=Path,required=True);parser.add_argument('--source',type=Path);parser.add_argument('--prepared',type=Path)
    parser.add_argument('--prepared-sha');parser.add_argument('--output',type=Path);parser.add_argument('--protected-root',action='append',default=[])
    parser.add_argument('--operation-source-sha');a=parser.parse_args()
    value=finish(a.control)if a.phase=='finish'else stage(a.source,dict(path=str(a.prepared),sha256=a.prepared_sha),a.control,a.output,a.protected_root,a.operation_source_sha)
    print(json.dumps(value,ensure_ascii=False))
