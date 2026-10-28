"""新八卡实例的有限等待与有界材料交接；不停止旧标注、不操作PTN。

先固定真实试验选型和新软件，再让已分配的新实例生成一次私有请求。
操作者核新实例后才受控停止旧标注，发布真实终态；本工具只检查最多
八个旧在途片并产生薄索引。等待、发布和exec都不等于GPU已经通过。
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True

if __package__:
    from . import queue_handoff as F
    from . import bounded_speed_handoff as H
    from . import bounded_speed_runtime as M
    from . import resume_existing_annotation as R
else:
    import queue_handoff as F
    import bounded_speed_handoff as H
    import bounded_speed_runtime as M
    import resume_existing_annotation as R

SCHEMA='annotation-bounded-speed-queue-blueprint/v1'
REQUEST='annotation-bounded-speed-queue-request/v1'
REPLY='annotation-bounded-speed-queue-reply/v1'
BASE_KEYS={'schema','purpose','task_name','previous_runtime_spec','new_contract','selected_pilot_admission',
           'new_run_root','new_code_root','new_bundle','device_lock_root','model_API_calls_allowed','release_training_labels'}
BLUEPRINT_KEYS={'schema','purpose','task_name','runtime_base','handoff_spec','final_handoff_root','handoff_root',
                'python_binary','wait_seconds','required_prior_attempt_ids','gpu_runtime_seconds','memory_budget_mib','outer_controller'}
SHARED_SOURCE={'bounded_speed_handoff.py','bounded_speed_queue.py','bounded_speed_runtime.py','queue_handoff.py',
    'prepare_speed_continuation.py','resume_existing_annotation.py','readonly_progress.py','portable_source_identity.py',
    'verify_portable_successor_mounts.py','contracts.py','schema.py','normalization.py','quality_axes.py','source_clock.py',
    'freeze_legacy_successor.py','gpu_vendor/__init__.py','gpu_vendor/resources.py','gpu_vendor/topology.py'}
REQUIRED_SOURCE={'gpu_launch.py','gpu_bootstrap.py','worker.py',*SHARED_SOURCE}
OUTER_NAMES=SHARED_SOURCE|{'bounded_speed_queue_handoff.py','build-receipt.json'}
need=H.need

GPU_SOURCE_COMMIT='cf7e8599eedda5a8a61a57cc6c30e1b4c764a2aa'
# 本v1运行入口只接已独立审阅的单窗生产包；未验native源码不能混入。
CF7_REQUIRED_SHA = {'alignment_observer.py': '2e99b4438fa8d96fd4e2f31922b039f34423740b6ec3687da6e7c5e77e01f5e6',
 'audio_windows.py': 'f737326d4830e79682da9331eb988213f671801f1b52743d3cda0f23a04b7d1a',
 'bounded_speed_handoff.py': 'fa1a97cf9ee7c94a9d02dbf776b8a16a405151042a8747ded53638a4c81e1bb4',
 'bounded_speed_queue.py': 'da634b014b3112987f802968793e3d6ca42663bb0bee2f93965a67832bcd2950',
 'bounded_speed_runtime.py': '1b85c92e71bea58e925b0debb9553313af402f8f6631d3e5d6cd78c3ab984c53',
 'contracts.py': '0d5bd38b20a2035d9e732fbcfb3c99269d5e83d1305b6d211f97a357b6be33c7',
 'freeze_legacy_successor.py': 'aabcc4b6aec77caee48d96160b1cbfa71d594ae60fe8c7f49e841e7057204670',
 'gpu_bootstrap.py': '2ce2b05b111f160c6f0b357f67c21caf360bf535dc8902e498837beaa842023d',
 'gpu_challenge.py': '2ed974d601a33ebc6714b904837c0dc6183970020a7becfd1371e16332452e44',
 'gpu_common.py': '9055ffc0284d9149b40a72a8a561ee322dd128c7424aa3de1b6b9130aa459a57',
 'gpu_launch.py': '89d3ad6966cc25a599f751bdda76313df0e43b448eec72035ef1e1bd0c9ccd05',
 'gpu_vendor/__init__.py': '71072f6dfa31ef3b329debecaf293c31c9549160700ca8b5f5f4169c6dcb7d5c',
 'gpu_vendor/resources.py': '12da19e4d9a06d1c1e9c9c5d50d15f94a6b6d4a442ec8ab56871b31e3bcedf3f',
 'gpu_vendor/topology.py': '278382dd8f349a136570d5426128f596fe6e7c2341771d8a96795a368bc7961a',
 'manifest_builder.py': '252eaa67fb18b52bbbbde7fff453f7187975a86f240cf2136efa80223b260233',
 'normalization.py': 'a79d35711018d035c0716b523641771e742ca61b3166f02f8b959bbdb7f20044',
 'portable_source_identity.py': 'd13c3fc8ed1f021592a3ede010d5208fa9d5b9496599a0370b124ce07d48c76f',
 'prepare_speed_continuation.py': '90673e55bd8d0e021eea545a0922acbf4bafd53aa5cda8f15aa4e86a6c9af423',
 'quality_axes.py': '3a1e5968f305fa57e093523699e1fd5de8eb4469999e266c7a63030857e3274e',
 'queue_handoff.py': 'ea9596e197ab101526936ae88166858baa2b5f97fd8227a339566066dde1470f',
 'qwen_backend.py': '36b0785c40f02399c7747b8635779b662b8eedc6074c553ff2ed7c91e4745d2b',
 'readonly_progress.py': '0ec8ef8dfb8e060530dcbfe6aa56f5410280f3e6cd7ed3eaae6c2ddf25baa735',
 'report.py': '208563741e6956e5afb843b34873fda882c9a4428a1db0fa837356cb82d4a879',
 'resume_existing_annotation.py': '56488a3cd73007f4b4439de33066d3cbea62197d1c92afa4c53e0a0d10d77ca4',
 'schema.py': 'b4eb1b082b3324676ebb649928c1adedcb21da4e3d5c46881038951c76d077a8',
 'sidecar.py': '09a9771b101adad0f95b4a4c4bb37bc3b5f86fe8a25c564cba2270a17407cf6c',
 'source_clock.py': 'f89560130f998863442c6a301c79cf520f62d602432b4055bb7fd3d2b872c11c',
 'source_registry.py': '96a5516a6530a1187620be5520131a838c49afbcb4066efe875ecd3e1e140d0d',
 'verify_portable_successor_mounts.py': '1ac3a72bddb317f5c12a58dddaf84faa354b7f04e49608ef4fa4981690f0a7bd',
 'work_queue.py': '928df1d96b2bae65a4436fbffe53dd8b86b1bb6f0872625f50462459b45495d1',
 'worker.py': '4a5a4cddf00478c5d36d9fe23031612fcd14a6754bab9a508b8ab1f910dbdfe5'}


def verify_dedicated_gpu_source(members):
    need(all(members.get(name)==digest for name,digest in CF7_REQUIRED_SHA.items()),
         'production GPU source differs from dedicated cf7e859 single-window revision')


def build_controller(gpu_code,bundle_pin,output):
    """从已冻结cf7e859专属GPU包取共同helper，新wrapper单独发布，不改GPU包。"""
    gpu_code=F.physical(gpu_code);need(bundle_pin['path']==str(gpu_code/'bundle_files.json'),'GPU source bundle path differs')
    members=F.ref(bundle_pin);need(REQUIRED_SOURCE<=set(members),'fixed GPU source bundle incomplete')
    R.frozen_bundle(dict(code_root=str(gpu_code),bundle=bundle_pin))
    output=F.physical(output);source=Path(__file__).resolve()
    need(not output.exists()and output.parent.is_dir()and F._disjoint(output,gpu_code)and F._disjoint(output,source.parent),'fresh independent outer controller output required')
    output.mkdir(mode=0o700)
    for name in sorted(SHARED_SOURCE):
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(gpu_code/name,target)
    shutil.copyfile(source,output/'bounded_speed_queue_handoff.py')
    original=F.file_pin(source)
    receipt=dict(schema='annotation-bounded-speed-outer-build/v1',GPU_bundle=bundle_pin,wrapper_source=original,
                 shared_helpers_byte_identical=True,GPU_source_modified=False,GPU_executed=False)
    (output/'build-receipt.json').write_bytes(F.canonical(receipt))
    mapping={name:hashlib.sha256(F.read_file(output/name)).hexdigest()for name in sorted(OUTER_NAMES)}
    manifest=output/'controller-files.json';manifest.write_bytes(F.canonical(mapping))
    controller=dict(root=str(output),manifest=F.file_pin(manifest))
    verify_outer(controller,members,production=False)
    R.frozen_bundle(dict(code_root=str(gpu_code),bundle=bundle_pin))
    return controller


def verify_outer(controller,gpu_members,*,production):
    need(isinstance(controller,dict)and set(controller)=={'root','manifest'},'explicit independent outer controller required')
    root=F.physical(controller['root']);need(controller['manifest']['path']==str(root/'controller-files.json'),'outer manifest path differs')
    files=F.ref(controller['manifest']);need(isinstance(files,dict)and set(files)==OUTER_NAMES,'exact independent outer source set required')
    entries=list(root.rglob('*'));need(all(not p.is_symlink()for p in entries),'outer linked import member refused')
    actual={str(p.relative_to(root))for p in entries if not p.is_dir()}
    directories={str(p.relative_to(root))for p in entries if p.is_dir()}
    expected_dirs={str(parent)for name in files for parent in Path(name).parents if str(parent)!='.'}
    need(actual==set(files)|{'controller-files.json'}and directories==expected_dirs,'outer controller has undeclared import members')
    for name,digest in files.items():
        need(isinstance(digest,str)and F.HEX.fullmatch(digest)and hashlib.sha256(F.read_file(root/name)).hexdigest()==digest,'outer source bytes changed: '+name)
        if name in SHARED_SOURCE:need(gpu_members.get(name)==digest,'outer shared helper differs from frozen GPU code: '+name)
    if production:
        need(Path(__file__).resolve().parent==root,'wrapper must execute inside independent fixed outer controller')
        for module,name in ((F,'queue_handoff.py'),(H,'bounded_speed_handoff.py'),(M,'bounded_speed_runtime.py'),(R,'resume_existing_annotation.py')):
            need(Path(module.__file__).resolve()==root/name,'outer imported helper origin differs: '+name)
    return root


def check(pin):
    """只读固定控制和有限试验证据；旧writer可以仍在运行，不核旧全库。"""
    b=F.ref(pin);need(set(b)==BLUEPRINT_KEYS and b.get('schema')==SCHEMA and b.get('purpose')in {'production','cpu_fixture'},'exact bounded queue blueprint required')
    base=b['runtime_base'];need(isinstance(base,dict)and set(base)==BASE_KEYS and base.get('schema')==M.SCHEMA and
         base.get('purpose')==b['purpose']and base.get('task_name')==b['task_name'],'fixed runtime base differs')
    F._task_name(b['task_name']);need(type(base['model_API_calls_allowed'])is int and base['model_API_calls_allowed']==0 and base['release_training_labels']is False,'offline/no label release required')
    wait=b['wait_seconds'];need(type(wait)is int and 1<=wait<=1800,'waiting must be explicitly bounded to 1800 seconds')
    need(type(b['gpu_runtime_seconds'])is int and 1<=b['gpu_runtime_seconds']<=720*3600 and
         type(b['memory_budget_mib'])is int and 8192<=b['memory_budget_mib']<=60000,'fixed GPU duration/memory budget invalid')
    priors=b['required_prior_attempt_ids'];need(isinstance(priors,list)and len(priors)==len(set(priors))and all(isinstance(t,str)and F.TASK.fullmatch(t)for t in priors),'unique actual prior attempt IDs required')
    prepared=F.ref(b['handoff_spec']);H.validate_spec(prepared);need(prepared['purpose']==b['purpose'],'handoff purpose differs')
    prior=F.ref(base['previous_runtime_spec']);need(prior.get('schema')in {'annotation-portable-successor-run/v1','annotation-portable-successor-run/v2'}and prior.get('purpose')==b['purpose'],'actual previous portable run required')
    old=F.ref(prior['old_control']);freeze=F.ref(prior['freeze_receipt']);union=F.ref(prior['union_spec']);producer=union['new_producer']
    need(prepared['old_contract']==freeze['new_contract']==producer['contract']and prepared['old_index']==producer['index']and
         prepared['old_queue_root']==producer['queue_root']==str(Path(prior['new_run_root'])/'queue')and
         prepared['old_sidecar_tree']==producer['sidecar_tree'],'handoff/current old producer chain differs')
    parent=F.ref(prepared['old_index']);need(parent.get('source_count',-1)+prepared['previous_retained_count']==prepared['global_source_count'],'frozen global denominator differs')
    new=F.ref(base['new_contract']);need(new.get('purpose')==('cpu_fixture'if b['purpose']=='cpu_fixture'else'silver_annotation'),'new explicit producer purpose differs')
    M.check_selected_pilot(base['selected_pilot_admission'],new,b['purpose'])
    code=F.physical(base['new_code_root']);need(base['new_bundle']['path']==str(code/'bundle_files.json'),'fixed new bundle path differs')
    members=F.ref(base['new_bundle']);need(isinstance(members,dict)and REQUIRED_SOURCE<=set(members),'bounded source bundle incomplete')
    R.frozen_bundle(dict(code_root=str(code),bundle=base['new_bundle']))
    # 新GPU包保持cf7e859专属版本不改；外层只加入等待脚本，共同helper须逐字节相同。
    if b['purpose']=='production':
        verify_dedicated_gpu_source(members)
    outer=verify_outer(b['outer_controller'],members,production=b['purpose']=='production')
    need(F._disjoint(outer,code),'outer controller cannot modify or overlap frozen GPU source')
    python=R.pinned_file(b['python_binary']);need(python.name.startswith('python')and python.is_absolute(),'fixed Python binary required')
    if b['purpose']=='production':
        need(Path(old['python']).resolve(strict=True)==python==Path(sys.executable).resolve(strict=True)and b['python_binary']==old['python_binary'],'actual/original interpreter differs')
    run=F.physical(base['new_run_root']);final=F.physical(b['final_handoff_root']);handoff=F.physical(b['handoff_root']);root=F.physical(old['task_root'])
    need(root in run.parents and root in final.parents and root in handoff.parents and base['device_lock_root']==old['device_lock_root'],'new namespaces/original UUID lock root differ')
    need(prepared['new_queue_root']==str(run/'queue')and prepared['new_sidecar_root']==str(run/'labels'),'new queue/labels namespace differs')
    protected=[F.physical(old['run_root']),F.physical(prior['new_run_root']),code,Path(prepared['old_index']['path']).parent,
        Path(prepared['old_contract']['path']).parent,Path(b['handoff_spec']['path']).parent,Path(base['new_contract']['path']).parent,
        F.physical(old['semantic_lock']).parent,outer]
    protected=[p for p in protected if p!=root and p not in root.parents]
    need(all(F._disjoint(a,p)for a in (run,final,handoff)for p in protected),'new output overlaps old/frozen/input/control namespace')
    need(all(F._disjoint(a,c)for i,a in enumerate((run,final,handoff))for c in (run,final,handoff)[i+1:]),'new output namespaces overlap')
    return dict(blueprint=b,pin=pin,base=base,handoff=prepared,old=old,new_contract=new,code=code,root=root,
                handoff_root=handoff,final_root=final,run=run,outer=outer)


def validate_resource_snapshot(snapshot,b):
    """新实例只读NVML证明可见八张空闲A800；不导入torch或创建CUDA。"""
    devices=snapshot.get('devices');need(isinstance(devices,list)and len(devices)==8 and
         len({d.get('uuid')for d in devices})==8 and all(isinstance(d.get('uuid'),str)and d['uuid'].startswith('GPU-')and
         'A800'in d.get('name','')and d.get('total_mib',0)>=80000 and d.get('used_mib',999999)<=256 and
         d.get('free_mib',0)>=b['memory_budget_mib']+8192 for d in devices)and snapshot.get('compute_processes')==[],
         'actual new worker must expose eight idle A800; unknown/existing compute PID refuses request')
    return snapshot


def request_value(c,pin,now):
    value=F.ref(pin,private=True);directory=F.private_dir(c['handoff_root']/'request-v1');b=c['blueprint']
    need(Path(pin['path'])==directory/'request.json'and set(value)=={'schema','blueprint','identity','nonce','created_utc','wait_seconds','reply_path','resource_snapshot'}and
         value.get('schema')==REQUEST and value['blueprint']==c['pin']and value['wait_seconds']==b['wait_seconds']and
         value['reply_path']==str(directory/'reply.json')and isinstance(value['nonce'],str)and F.HEX.fullmatch(value['nonce']),'request fixed identity/path/nonce differs')
    F._identity(value['identity']);F._age(value['created_utc'],now,b['wait_seconds'])
    need(value['resource_snapshot']['path']==str(directory/'resource-snapshot.json'),'resource snapshot outside actual request')
    resource=F.ref(value['resource_snapshot'],private=True)
    need(resource.get('identity')==value['identity']and resource.get('GPU_context_created')is False,'resource snapshot identity differs')
    validate_resource_snapshot(resource['snapshot'],b)
    need(value['identity']['task_id']not in {c['handoff']['old_task_id'],*b['required_prior_attempt_ids']},'waiting task must be a distinct successor')
    return value


def observation(c,request,pin,now):
    value=F.ref(pin);spec=c['handoff'];b=c['blueprint']
    need(value.get('schema')==M.PLATFORM_SCHEMA and value.get('old_task_id')==spec['old_task_id']and
         value.get('old_queue_root')==spec['old_queue_root']and value.get('competing_successor_tasks')==[]and
         value.get('remote_proc_enumeration_performed')is False and value.get('evidence_basis')=='platform_terminal_and_current_shared_locks',
         'actual old terminal/platform absence observation required')
    observed=F._age(value['observed_utc'],now,3600)
    need(observed>=datetime.datetime.fromisoformat(request['created_utc']).timestamp(),'platform observation must follow actual new cloud request')
    terminal=F.ref(value['terminal_observation']);workers=terminal.get('workers')
    need(terminal.get('schema')=='annotation-writer-terminal-observation/v1'and terminal.get('task_id')==spec['old_task_id']and
         terminal.get('queue_root')==spec['old_queue_root']and terminal.get('state')in F.TERMINAL and
         type(terminal.get('active_workers'))is int and terminal['active_workers']==0 and terminal.get('observed_utc')==value['observed_utc']and
         isinstance(workers,list)and len(workers)==1 and isinstance(workers[0],dict)and
         workers[0].get('name')==spec['old_task_id']+'-worker-0'and workers[0].get('state')in F.TERMINAL,'actual old task/worker must be terminal')
    priors=value.get('prior_attempts');need(isinstance(priors,list),'explicit prior terminal attempt list required')
    ids=[]
    for p in priors:
        need(isinstance(p,dict)and F.TASK.fullmatch(p.get('task_id',''))and p.get('state')in F.TERMINAL and
             type(p.get('active_workers'))is int and p['active_workers']==0,'prior successor active or unverified');ids.append(p['task_id'])
    need(len(ids)==len(set(ids))and set(b['required_prior_attempt_ids'])<=set(ids),'required prior terminal attempt missing/duplicate')
    own=value.get('current_successor_task');identity=request['identity']
    need(isinstance(own,dict)and own.get('task_id')==identity['task_id']and own.get('task_name')==b['task_name']and
         own.get('state')=='Running'and type(own.get('active_workers'))is int and own['active_workers']==1 and
         own.get('workers')==[{'name':identity['hostname'],'state':'Running'}],'actual new allocated task/worker Running observation required')
    return value


def _expected_runtime(c,index,obs):
    return dict(c['base'],index=index,current_platform_observation=obs)


def publish(blueprint_pin,request_pin,observation_pin,*,clock=None):
    """仅真实终态后生成≤8片材料；成功reply不代表平台操作或GPU准入。"""
    c=check(blueprint_pin);b=c['blueprint'];need(clock is None or b['purpose']=='cpu_fixture','production cannot inject clock')
    now=clock or time.time;request=request_value(c,request_pin,now());observation(c,request,observation_pin,now())
    directory=Path(request_pin['path']).parent;need(not Path(request['reply_path']).exists(),'reply exists; preserve publication, no implicit retry')
    need(not c['final_root'].exists()and not c['run'].exists(),'fresh final preparation and new run required; inspect existing result')
    publication=F.mkdir_new(directory/'publication-v1')
    try:
        value=H.prepare(Path(b['handoff_spec']['path']),b['handoff_spec']['sha256'],c['final_root'],execute_final=True,
            terminal_pin=F.ref(observation_pin)['terminal_observation'])
        need(value.get('status')=='terminal_thin_plan_not_gpu_admission'and value.get('old_producer_modified')is False and value.get('database_created')is False,
             'bounded terminal preparation incomplete')
        need(value['exact_remaining_source_count']>0,'old producer already completed every source; no successor GPU task needed')
        index=value['new_index'];runtime=_expected_runtime(c,index,observation_pin)
        runtime_pin=F.write_new(publication/'runtime-spec.json',runtime)
        # 完整新运行检查只会读最多8个已领取父片；不取得GPU，也不建立全库。
        compiled=M.compile_runtime(runtime_pin)
        need(compiled['index']['old_contract']==c['handoff']['old_contract']and compiled['index']['parent_index']==c['handoff']['old_index'],'compiled final old producer differs')
        request_value(c,request_pin,now());observation(c,request,observation_pin,now());check(blueprint_pin)
        reply=dict(schema=REPLY,blueprint=blueprint_pin,request=request_pin,identity=request['identity'],request_nonce=request['nonce'],
            runtime_spec=runtime_pin,observation=observation_pin,prepared_report=F.file_pin(c['final_root']/'report.json'),
            published_utc=F._utc(now()),platform_authentication_performed=False,GPU_admission_granted=False)
        result=F.write_new(Path(request['reply_path']),reply)
        return dict(schema='annotation-bounded-speed-queue-publication/v1',reply=result,runtime_spec=runtime_pin,
                    database_created=False,full_library_audit=False,GPU_started=False,old_task_stopped_by_this_tool=False)
    except BaseException:
        failure(publication,'publisher-failure.json');raise


def reply_value(c,request_pin,request,identity,now):
    need(identity==request['identity'],'actual waiting process identity changed');request_value(c,request_pin,now)
    reply_pin=F.file_pin(request['reply_path'],private=True);reply=F.ref(reply_pin,private=True)
    need(set(reply)=={'schema','blueprint','request','identity','request_nonce','runtime_spec','observation','prepared_report','published_utc','platform_authentication_performed','GPU_admission_granted'}and
         reply.get('schema')==REPLY and reply['blueprint']==c['pin']and reply['request']==request_pin and
         reply['identity']==identity and reply['request_nonce']==request['nonce']and reply['platform_authentication_performed']is False and reply['GPU_admission_granted']is False,
         'reply identity/request/nonce/blueprint differs')
    published=F._age(reply['published_utc'],now,c['blueprint']['wait_seconds']);need(published>=datetime.datetime.fromisoformat(request['created_utc']).timestamp(),'reply predates actual request')
    publication=F.private_dir(Path(request_pin['path']).parent/'publication-v1');pin=reply['runtime_spec']
    need(Path(pin['path'])==publication/'runtime-spec.json','runtime outside fixed private publication namespace')
    runtime=F.ref(pin,private=True);need(set(runtime)==BASE_KEYS|{'index','current_platform_observation'}and
         F.canonical(runtime)==F.canonical(_expected_runtime(c,runtime['index'],reply['observation'])),'runtime changed beyond final index/current observation')
    need(runtime['index']['path']==str(c['final_root']/'index.json')and reply['prepared_report']['path']==str(c['final_root']/'report.json'),'final index/report namespace differs')
    report=F.ref(reply['prepared_report']);need(report.get('new_index')==runtime['index']and report.get('spec_sha256')==c['blueprint']['handoff_spec']['sha256']and
         report.get('status')=='terminal_thin_plan_not_gpu_admission'and report.get('old_producer_modified')is False and report.get('database_created')is False,'final handoff report does not bind runtime')
    observation(c,request,reply['observation'],now);M.compile_runtime(pin)
    return pin,reply_pin


def failure(directory,name):
    try:F.write_new(Path(directory)/name,dict(schema='annotation-bounded-speed-queue-failure/v1',traceback=traceback.format_exc(),
            observed_utc=F._utc(time.time()),GPU_context_created_by_outer=False,automatic_retry_allowed=False))
    except BaseException:print(traceback.format_exc(),file=sys.stderr,flush=True)


def wait(blueprint_pin,*,execute=False,runner=None,clock=None,monotonic=None,sleeper=None,identity_provider=None,resource_provider=None):
    c=check(blueprint_pin);b=c['blueprint'];injected=any(x is not None for x in (runner,clock,monotonic,sleeper,identity_provider,resource_provider))
    need(not injected or b['purpose']=='cpu_fixture','production cannot inject process/clock/exec hooks')
    if not execute:return dict(schema='annotation-bounded-speed-queue-check/v1',passed=True,request_created=False,GPU_started=False,old_task_stopped=False)
    need(b['purpose']=='production'or runner is not None,'CPU fixtures cannot invoke real launcher')
    need(int(os.environ.get('WORLD_SIZE','1'))==1 and int(os.environ.get('RANK','0'))==0 and 'LOCAL_RANK'not in os.environ,
         'waiter must execute once outside platform/DDP rank groups')
    now=clock or time.time;mono=monotonic or time.monotonic;sleep=sleeper or time.sleep;identify=identity_provider or F.actual_identity
    identity=identify();F._identity(identity);need(identity['task_id']not in {c['handoff']['old_task_id'],*b['required_prior_attempt_ids']},'new task must be independent')
    root=None;directory=None;handlers={}
    def interrupted(signum,frame):raise RuntimeError('bounded handoff interrupted before GPU: signal '+str(signum))
    try:
        for signum in (signal.SIGINT,signal.SIGTERM):handlers[signum]=signal.signal(signum,interrupted)
        root=F.mkdir_new(c['handoff_root']);directory=F.mkdir_new(root/'request-v1')
        if resource_provider is None:
            if __package__:from .gpu_vendor.resources import capture_snapshot
            else:from gpu_vendor.resources import capture_snapshot
            resource_provider=capture_snapshot
        snapshot=validate_resource_snapshot(resource_provider(),b)
        resource_pin=F.write_new(directory/'resource-snapshot.json',dict(identity=identity,snapshot=snapshot,GPU_context_created=False,observed_utc=F._utc(now())))
        request=dict(schema=REQUEST,blueprint=blueprint_pin,identity=identity,nonce=secrets.token_hex(32),created_utc=F._utc(now()),wait_seconds=b['wait_seconds'],reply_path=str(directory/'reply.json'),resource_snapshot=resource_pin)
        request_pin=F.write_new(directory/'request.json',request)
        print(json.dumps(dict(schema='annotation-bounded-speed-queue-awaiting/v1',request=request_pin,cloud_instance_already_allocated=True,
            possible_GPU_billing=True,GPU_context_created_by_outer=False),ensure_ascii=False),flush=True)
        started=mono()
        while True:
            need(0<=mono()-started<b['wait_seconds'],'bounded handoff timed out before GPU')
            need(identify()==identity,'actual waiting process identity changed');request_value(c,request_pin,now())
            if not F._reply_pending(request['reply_path']):break
            sleep(min(2.,b['wait_seconds']-(mono()-started)))
        c=check(blueprint_pin);runtime_pin,reply_pin=reply_value(c,request_pin,request,identify(),now())
        need(mono()-started<b['wait_seconds'],'handoff timed out during final CPU checks')
        # 已通过fresh终态、父范围与完整字候选质量门，才为原GPU父入口建立独立run。
        need(not c['run'].exists()and not c['run'].is_symlink(),'new run already exists; inspect prior attempt')
        c['run'].mkdir(mode=0o700);F._fsync_dir(c['run'].parent)
        runtime=F.ref(runtime_pin,private=True);old=c['old'];code=c['code'];resources=c['run']/'resources-attempt1'
        argv=[str(R.pinned_file(b['python_binary'])),'-B',str(code/'gpu_launch.py'),'--execute','--bundle-sha',c['base']['new_bundle']['sha256'],
            '--contract',runtime['new_contract']['path'],'--contract-sha',runtime['new_contract']['sha256'],'--shard-index',runtime['index']['path'],
            '--index-sha',runtime['index']['sha256'],'--queue-root',str(c['run']/'queue'),'--output-root',str(c['run']/'labels'),
            '--output',str(resources),'--target-uuids','auto8','--run-root',str(c['run']),'--device-lock-root',old['device_lock_root'],
            '--memory-budget-mib',str(b['memory_budget_mib']),'--max-runtime-seconds',str(b['gpu_runtime_seconds']),
            '--bounded-runtime-spec',runtime_pin['path'],'--bounded-runtime-sha',runtime_pin['sha256']]
        F.write_new(directory/'exec-intent.json',dict(argv=argv,request=request_pin,reply=reply_pin,runtime_spec=runtime_pin,GPU_started_not_proven_by_intent=True))
        # 保存intent后再核一遍，不把CPU检查描述当作FD/GPU运行许可。
        need(identify()==identity and mono()-started<b['wait_seconds'],'identity/deadline changed at exec boundary')
        again=check(blueprint_pin);lastspec,lastreply=reply_value(again,request_pin,request,identify(),now())
        need(lastspec==runtime_pin and lastreply==reply_pin,'publication changed at exec boundary')
        need(identify()==identity and 0<=mono()-started<b['wait_seconds'],'identity/deadline changed after final material verification')
        env=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT','BASH_ENV','ENV','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1')
        if runner is not None:return runner(argv,env)
        os.execve(argv[0],argv,env)
        raise RuntimeError('GPU parent exec unexpectedly returned')
    except BaseException:
        if directory is not None:failure(directory,'waiter-failure.json')
        elif root is not None:failure(root,'waiter-failure.json')
        raise
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('check','wait','publish','build-controller'),nargs='?',default='check')
    p.add_argument('--blueprint');p.add_argument('--blueprint-sha');p.add_argument('--execute',action='store_true')
    p.add_argument('--gpu-code');p.add_argument('--gpu-bundle-sha');p.add_argument('--controller-output')
    p.add_argument('--request');p.add_argument('--request-sha');p.add_argument('--observation');p.add_argument('--observation-sha');a=p.parse_args()
    if a.action=='build-controller':
        need(a.gpu_code and a.gpu_bundle_sha and a.controller_output,'fixed GPU source and fresh outer output required')
        value=build_controller(a.gpu_code,dict(path=str(Path(a.gpu_code)/'bundle_files.json'),sha256=a.gpu_bundle_sha),a.controller_output)
        print(json.dumps(value,ensure_ascii=False,indent=2));return
    need(a.blueprint and a.blueprint_sha,'explicit blueprint/SHA required')
    pin=dict(path=a.blueprint,sha256=a.blueprint_sha)
    if a.action=='publish':
        need(a.request and a.request_sha and a.observation and a.observation_sha,'explicit actual request/observation pins required')
        value=publish(pin,dict(path=a.request,sha256=a.request_sha),dict(path=a.observation,sha256=a.observation_sha))
    else:value=wait(pin,execute=a.action=='wait'and a.execute)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
