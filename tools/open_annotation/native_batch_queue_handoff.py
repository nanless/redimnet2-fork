"""原生合批v2新八卡有限等待与材料交接；旧单窗入口不扩受理范围。

先固定真实试验选型和新软件，再让已分配的新实例生成一次私有请求。
先核唯一正式新任务；碎片时依授权停精确旧标注，随后核新实例与新鲜终态。
本工具只检查最多
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

SCHEMA='annotation-native-batch-queue-blueprint/v2'
REQUEST='annotation-native-batch-queue-request/v2'
REPLY='annotation-native-batch-queue-reply/v2'
BASE_KEYS={'schema','purpose','task_name','previous_runtime_spec','new_contract','selected_pilot_admission',
           'new_run_root','new_code_root','new_bundle','device_lock_root','model_API_calls_allowed','release_training_labels'}
BLUEPRINT_KEYS={'schema','purpose','task_name','runtime_base','handoff_spec','final_handoff_root','handoff_root',
                'python_binary','wait_seconds','required_prior_attempt_ids','gpu_runtime_seconds','memory_budget_mib','outer_controller'}
SHARED_SOURCE={'bounded_speed_handoff.py','bounded_speed_queue.py','bounded_speed_runtime.py','queue_handoff.py',
    'prepare_speed_continuation.py','resume_existing_annotation.py','readonly_progress.py','portable_source_identity.py',
    'verify_portable_successor_mounts.py','contracts.py','schema.py','normalization.py','quality_axes.py','source_clock.py',
    'freeze_legacy_successor.py','native_transformers_batch.py','native_batch_admission.py','native_batch_readback.py',
    'audit_native_transformers_batch.py','audit_speed_comparison.py','work_queue.py','gpu_vendor/__init__.py','gpu_vendor/resources.py','gpu_vendor/topology.py'}
REQUIRED_SOURCE={'gpu_launch.py','gpu_bootstrap.py','worker.py',*SHARED_SOURCE}
OUTER_NAMES=SHARED_SOURCE|{'native_batch_queue_handoff.py','build-receipt.json'}
need=H.need
FEATURE_CORE='native_b1_feature_padding.py'
FEATURE_POLICY='official_b1_features_then_pad/v1'

# 本v2仅接真实native B1/4/8试验和完整批次/映射证明，不消费旧单窗准入。




def feature_dependency(contract,gpu_members):
    """先只核声明与GPU数学SHA，不在外层模块尚缺失时递归import核心。"""
    runtime=contract.get('runtime',{})
    if 'asr_preprocessing'not in runtime:
        need('asr_preprocessing_sources'not in contract,'feature sources need explicit preprocessing policy')
        return None
    need(runtime['asr_preprocessing']=={'policy':FEATURE_POLICY},'explicit B1 feature outer dependency required')
    from contracts import ENGINE_MODULES,asr_batch_profile
    need(asr_batch_profile(contract)is not None and runtime.get('asr_compute')=={'dtype':'float32','attention':'sdpa'},'actual native FP32 feature contract required')
    expected=set(ENGINE_MODULES)|{'native_transformers_batch.py',FEATURE_CORE};math=contract.get('engine_source_files')
    need(isinstance(math,dict)and set(math)==expected and len(math)==16 and all(F.HEX.fullmatch(v)if isinstance(v,str)else False for v in math.values())and
         all(gpu_members.get(name)==digest for name,digest in math.items()),'explicit actual GPU ENGINE16 source SHA set required')
    sources=contract.get('asr_preprocessing_sources');roles={'whisper_feature_extractor','sequence_feature_extractor','batch_feature','processor_mixin','audio_utils','qwen_processor','sdk_normalizer'}
    need(isinstance(sources,dict)and set(sources)=={'schema','files','processor_config'}and sources['schema']=='annotation-official-b1-feature-sources/v1'and
         isinstance(sources['files'],dict)and set(sources['files'])==roles,'explicit seven feature sources/config declaration required')
    for ref in [*sources['files'].values(),sources['processor_config']]:
        need(isinstance(ref,dict)and set(ref)=={'path','sha256'}and isinstance(ref['sha256'],str)and F.HEX.fullmatch(ref['sha256']),'explicit feature source pin required')
        F.physical(ref['path'])
    return dict(schema='annotation-native-b1-feature-outer-dependency/v1',policy=FEATURE_POLICY,core_member=FEATURE_CORE,
                core_sha256=math[FEATURE_CORE],math_engine_source_files=math,preprocessing_sources=sources)


def build_controller(gpu_code,bundle_pin,output,*,feature_contract_pin=None):
    """从已真实验收的native专属GPU包取共同helper，新wrapper单独发布，不改GPU包。"""
    gpu_code=F.physical(gpu_code);need(bundle_pin['path']==str(gpu_code/'bundle_files.json'),'GPU source bundle path differs')
    members=F.ref(bundle_pin);need(REQUIRED_SOURCE<=set(members),'fixed GPU source bundle incomplete')
    R.frozen_bundle(dict(code_root=str(gpu_code),bundle=bundle_pin))
    feature_contract=None;dependency=None
    if feature_contract_pin is not None:
        feature_contract=F.ref(feature_contract_pin);dependency=feature_dependency(feature_contract,members)
        need(dependency is not None,'feature-contract option refuses old15/implicit feature contract')
    shared=SHARED_SOURCE|({FEATURE_CORE}if dependency is not None else set())
    output=F.physical(output);source=Path(__file__).resolve()
    need(not output.exists()and output.parent.is_dir()and F._disjoint(output,gpu_code)and F._disjoint(output,source.parent),'fresh independent outer controller output required')
    output.mkdir(mode=0o700)
    for name in sorted(shared):
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(gpu_code/name,target)
    shutil.copyfile(source,output/'native_batch_queue_handoff.py')
    original=F.file_pin(source)
    receipt=dict(schema='annotation-native-batch-outer-build/v2',GPU_bundle=bundle_pin,wrapper_source=original,
                 shared_helpers_byte_identical=True,GPU_source_modified=False,GPU_executed=False)
    if dependency is not None:receipt.update(feature_contract=feature_contract_pin,feature_dependency=dependency)
    (output/'build-receipt.json').write_bytes(F.canonical(receipt))
    mapping={name:hashlib.sha256(F.read_file(output/name)).hexdigest()for name in sorted(shared|{'native_batch_queue_handoff.py','build-receipt.json'})}
    manifest=output/'controller-files.json';manifest.write_bytes(F.canonical(mapping))
    controller=dict(root=str(output),manifest=F.file_pin(manifest))
    verify_outer(controller,members,production=False,contract=feature_contract,gpu_bundle_pin=bundle_pin)
    R.frozen_bundle(dict(code_root=str(gpu_code),bundle=bundle_pin))
    return controller


def verify_outer(controller,gpu_members,*,production,contract=None,gpu_bundle_pin=None):
    need(isinstance(controller,dict)and set(controller)=={'root','manifest'},'explicit independent outer controller required')
    root=F.physical(controller['root']);need(controller['manifest']['path']==str(root/'controller-files.json'),'outer manifest path differs')
    dependency=feature_dependency(contract,gpu_members)if contract is not None else None
    shared=SHARED_SOURCE|({FEATURE_CORE}if dependency is not None else set());expected=shared|{'native_batch_queue_handoff.py','build-receipt.json'}
    files=F.ref(controller['manifest']);need(isinstance(files,dict)and set(files)==expected,'exact independent outer source set required; explicit feature contract requires core16 closure')
    entries=list(root.rglob('*'));need(all(not p.is_symlink()for p in entries),'outer linked import member refused')
    actual={str(p.relative_to(root))for p in entries if not p.is_dir()}
    directories={str(p.relative_to(root))for p in entries if p.is_dir()}
    expected_dirs={str(parent)for name in files for parent in Path(name).parents if str(parent)!='.'}
    need(actual==set(files)|{'controller-files.json'}and directories==expected_dirs,'outer controller has undeclared import members')
    for name,digest in files.items():
        need(isinstance(digest,str)and F.HEX.fullmatch(digest)and hashlib.sha256(F.read_file(root/name)).hexdigest()==digest,'outer source bytes changed: '+name)
        if name in shared:need(gpu_members.get(name)==digest,'outer shared helper differs from frozen GPU code: '+name)
    receipt=F.ref(F.file_pin(root/'build-receipt.json'))
    need(isinstance(receipt.get('GPU_bundle'),dict)and F.ref(receipt['GPU_bundle'])==gpu_members,'outer build receipt does not bind actual GPU bundle')
    if gpu_bundle_pin is not None:need(receipt['GPU_bundle']==gpu_bundle_pin,'outer build actual GPU bundle pin differs')
    if dependency is not None:
        need(receipt.get('feature_dependency')==dependency and isinstance(receipt.get('feature_contract'),dict),'actual feature outer build receipt differs')
        declared=F.ref(receipt['feature_contract']);need(feature_dependency(declared,gpu_members)==dependency,'feature build contract changed or differs from runtime contract')
    else:need('feature_dependency'not in receipt and 'feature_contract'not in receipt,'old15 outer cannot silently include feature dependency')
    if production:
        need(Path(__file__).resolve().parent==root,'wrapper must execute inside independent fixed outer controller')
        for module,name in ((F,'queue_handoff.py'),(H,'bounded_speed_handoff.py'),(M,'bounded_speed_runtime.py'),(R,'resume_existing_annotation.py')):
            need(Path(module.__file__).resolve()==root/name,'outer imported helper origin differs: '+name)
    return root


def check(pin,*,selection_scope=None):
    """只读固定控制和有限试验证据；旧writer可以仍在运行，不核旧全库。"""
    b=F.ref(pin);need(set(b)==BLUEPRINT_KEYS and b.get('schema')==SCHEMA and b.get('purpose')in {'production','cpu_fixture'},'exact bounded queue blueprint required')
    base=b['runtime_base'];need(isinstance(base,dict)and set(base)in (BASE_KEYS,BASE_KEYS|{'proof_transport'})and base.get('schema')==M.NATIVE_SCHEMA and
         base.get('purpose')==b['purpose']and base.get('task_name')==b['task_name'],'fixed runtime base differs')
    if 'proof_transport'in base:
        need(callable(getattr(M,'proof_transport_profile',None)),'explicit transport requires reviewed runtime helper')
        M.proof_transport_profile(base)
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
    code=F.physical(base['new_code_root']);need(base['new_bundle']['path']==str(code/'bundle_files.json'),'fixed new bundle path differs')
    members=F.ref(base['new_bundle']);need(isinstance(members,dict)and REQUIRED_SOURCE<=set(members),'bounded source bundle incomplete')
    R.frozen_bundle(dict(code_root=str(code),bundle=base['new_bundle']))
    # 先核外层完整导入闭包，再进入可能import新核心的选型证明。
    outer=verify_outer(b['outer_controller'],members,production=b['purpose']=='production',contract=new,gpu_bundle_pin=base['new_bundle'])
    if __package__:from .native_batch_admission import check_native_selected_pilot
    else:from native_batch_admission import check_native_selected_pilot
    selected=(check_native_selected_pilot(base['selected_pilot_admission'],new,b['purpose'])if selection_scope is None else
              check_native_selected_pilot(base['selected_pilot_admission'],new,b['purpose'],selection_scope=selection_scope))
    need(selected['verified_native_bundle_sha256']==base['new_bundle']['sha256'],
         'native formal executable bundle must equal actual selected native pilot')
    need(F._disjoint(outer,code),'outer controller cannot modify or overlap frozen GPU source')
    python=R.pinned_file(b['python_binary']);need(python.name.startswith('python')and python.is_absolute(),'fixed Python binary required')
    if b['purpose']=='production':
        need(Path(old['python']).resolve(strict=True)==python==Path(sys.executable).resolve(strict=True)and b['python_binary']==old['python_binary'],'actual/original interpreter differs')
    run=F.physical(base['new_run_root']);final=F.physical(b['final_handoff_root']);handoff=F.physical(b['handoff_root']);root=F.physical(old['task_root'])
    # 2026-11-09 用户规定：大数据落 /root/group-shared/voiceprint，不再限定在旧任务根内。
    # 因此把"必须位于 old.task_root 之下"放宽为"位于 old.task_root 或 blueprint/base 声明的 storage_root 之下"，
    # 其余判据（device_lock 相等、互不重叠、受保护命名空间）一律不变。
    def _allowed(where,base_,blue_):
        roots=[root]
        for src in (blue_.get('storage_root'),base_.get('storage_root')):
            if src:
                cand=F.physical(src)
                if cand not in roots: roots.append(cand)
        return any(where==r or r in where.parents for r in roots)
    need(_allowed(run,base,b) and _allowed(final,base,b) and _allowed(handoff,base,b) and base['device_lock_root']==old['device_lock_root'],'new namespaces/original UUID lock root differ')
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


def _check_with_scope(pin,scope):
    return check(pin)if scope is None else check(pin,selection_scope=scope)


def _compile_with_scope(pin,scope):
    return M.compile_runtime(pin)if scope is None else M.compile_runtime(pin,selection_scope=scope)


def _run_in_selection_scope(enabled,operation):
    need(type(enabled)is bool,'same-CLI selection reuse must be explicit boolean opt-in')
    if not enabled:return operation(None)
    if __package__:from .native_batch_admission import NativeSelectionScope
    else:from native_batch_admission import NativeSelectionScope
    with NativeSelectionScope(enabled=True)as scope:result=operation(scope)
    if isinstance(result,dict):result['selection_reuse_scope']=scope.facts()
    return result


def publish(blueprint_pin,request_pin,observation_pin,*,clock=None,reuse_verified_selection=False):
    return _run_in_selection_scope(reuse_verified_selection,lambda scope:_publish(blueprint_pin,request_pin,observation_pin,clock=clock,selection_scope=scope))


def _publish(blueprint_pin,request_pin,observation_pin,*,clock=None,selection_scope=None):
    """仅真实终态后生成≤8片材料；成功reply不代表平台操作或GPU准入。"""
    c=_check_with_scope(blueprint_pin,selection_scope);b=c['blueprint'];need(clock is None or b['purpose']=='cpu_fixture','production cannot inject clock')
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
        compiled=_compile_with_scope(runtime_pin,selection_scope)
        need(compiled['index']['old_contract']==c['handoff']['old_contract']and compiled['index']['parent_index']==c['handoff']['old_index'],'compiled final old producer differs')
        request_value(c,request_pin,now());observation(c,request,observation_pin,now());_check_with_scope(blueprint_pin,selection_scope)
        reply=dict(schema=REPLY,blueprint=blueprint_pin,request=request_pin,identity=request['identity'],request_nonce=request['nonce'],
            runtime_spec=runtime_pin,observation=observation_pin,prepared_report=F.file_pin(c['final_root']/'report.json'),
            published_utc=F._utc(now()),platform_authentication_performed=False,GPU_admission_granted=False)
        result=F.write_new(Path(request['reply_path']),reply)
        return dict(schema='annotation-native-batch-queue-publication/v2',reply=result,runtime_spec=runtime_pin,
                    database_created=False,full_library_audit=False,GPU_started=False,old_task_stopped_by_this_tool=False)
    except BaseException:
        failure(publication,'publisher-failure.json');raise


def reply_value(c,request_pin,request,identity,now,*,selection_scope=None):
    need(identity==request['identity'],'actual waiting process identity changed');request_value(c,request_pin,now)
    reply_pin=F.file_pin(request['reply_path'],private=True);reply=F.ref(reply_pin,private=True)
    need(set(reply)=={'schema','blueprint','request','identity','request_nonce','runtime_spec','observation','prepared_report','published_utc','platform_authentication_performed','GPU_admission_granted'}and
         reply.get('schema')==REPLY and reply['blueprint']==c['pin']and reply['request']==request_pin and
         reply['identity']==identity and reply['request_nonce']==request['nonce']and reply['platform_authentication_performed']is False and reply['GPU_admission_granted']is False,
         'reply identity/request/nonce/blueprint differs')
    published=F._age(reply['published_utc'],now,c['blueprint']['wait_seconds']);need(published>=datetime.datetime.fromisoformat(request['created_utc']).timestamp(),'reply predates actual request')
    publication=F.private_dir(Path(request_pin['path']).parent/'publication-v1');pin=reply['runtime_spec']
    need(Path(pin['path'])==publication/'runtime-spec.json','runtime outside fixed private publication namespace')
    runtime=F.ref(pin,private=True);need(set(runtime)==set(c['base'])|{'index','current_platform_observation'}and
         F.canonical(runtime)==F.canonical(_expected_runtime(c,runtime['index'],reply['observation'])),'runtime changed beyond final index/current observation')
    need(runtime['index']['path']==str(c['final_root']/'index.json')and reply['prepared_report']['path']==str(c['final_root']/'report.json'),'final index/report namespace differs')
    report=F.ref(reply['prepared_report']);need(report.get('new_index')==runtime['index']and report.get('spec_sha256')==c['blueprint']['handoff_spec']['sha256']and
         report.get('status')=='terminal_thin_plan_not_gpu_admission'and report.get('old_producer_modified')is False and report.get('database_created')is False,'final handoff report does not bind runtime')
    observation(c,request,reply['observation'],now);_compile_with_scope(pin,selection_scope)
    return pin,reply_pin


def failure(directory,name):
    try:F.write_new(Path(directory)/name,dict(schema='annotation-native-batch-queue-failure/v2',traceback=traceback.format_exc(),
            observed_utc=F._utc(time.time()),GPU_context_created_by_outer=False,automatic_retry_allowed=False))
    except BaseException:print(traceback.format_exc(),file=sys.stderr,flush=True)


def wait(blueprint_pin,*,execute=False,runner=None,clock=None,monotonic=None,sleeper=None,identity_provider=None,resource_provider=None,reuse_verified_selection=False):
    return _run_in_selection_scope(reuse_verified_selection,lambda scope:_wait(blueprint_pin,execute=execute,runner=runner,clock=clock,monotonic=monotonic,sleeper=sleeper,identity_provider=identity_provider,resource_provider=resource_provider,selection_scope=scope))


def _wait(blueprint_pin,*,execute=False,runner=None,clock=None,monotonic=None,sleeper=None,identity_provider=None,resource_provider=None,selection_scope=None):
    c=_check_with_scope(blueprint_pin,selection_scope);b=c['blueprint'];injected=any(x is not None for x in (runner,clock,monotonic,sleeper,identity_provider,resource_provider))
    need(not injected or b['purpose']=='cpu_fixture','production cannot inject process/clock/exec hooks')
    if not execute:return dict(schema='annotation-native-batch-queue-check/v2',passed=True,request_created=False,GPU_started=False,old_task_stopped=False)
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
        print(json.dumps(dict(schema='annotation-native-batch-queue-awaiting/v2',request=request_pin,cloud_instance_already_allocated=True,
            possible_GPU_billing=True,GPU_context_created_by_outer=False),ensure_ascii=False),flush=True)
        started=mono()
        while True:
            need(0<=mono()-started<b['wait_seconds'],'bounded handoff timed out before GPU')
            need(identify()==identity,'actual waiting process identity changed');request_value(c,request_pin,now())
            if not F._reply_pending(request['reply_path']):break
            sleep(min(2.,b['wait_seconds']-(mono()-started)))
        c=_check_with_scope(blueprint_pin,selection_scope);runtime_pin,reply_pin=reply_value(c,request_pin,request,identify(),now(),selection_scope=selection_scope)
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
        F.write_new(directory/'exec-intent.json',dict(argv=argv,request=request_pin,reply=reply_pin,runtime_spec=runtime_pin,GPU_started_not_proven_by_intent=True,selection_reuse_scope=None if selection_scope is None else selection_scope.facts()))
        # 保存intent后再核一遍，不把CPU检查描述当作FD/GPU运行许可。
        need(identify()==identity and mono()-started<b['wait_seconds'],'identity/deadline changed at exec boundary')
        again=_check_with_scope(blueprint_pin,selection_scope);lastspec,lastreply=reply_value(again,request_pin,request,identify(),now(),selection_scope=selection_scope)
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
    p.add_argument('--feature-contract');p.add_argument('--feature-contract-sha')
    p.add_argument('--reuse-verified-selection',action='store_true',help='仅同一CLI实际全核后复用解析结果；每次全部输入实际核SHA，默认关闭')
    p.add_argument('--request');p.add_argument('--request-sha');p.add_argument('--observation');p.add_argument('--observation-sha');a=p.parse_args()
    if a.action=='build-controller':
        need(not a.reuse_verified_selection,'build-controller does not select pilot evidence; no reuse option here')
        need(a.gpu_code and a.gpu_bundle_sha and a.controller_output,'fixed GPU source and fresh outer output required')
        need((a.feature_contract is None)==(a.feature_contract_sha is None),'feature contract path/SHA must be paired')
        feature=None if a.feature_contract is None else dict(path=a.feature_contract,sha256=a.feature_contract_sha)
        value=build_controller(a.gpu_code,dict(path=str(Path(a.gpu_code)/'bundle_files.json'),sha256=a.gpu_bundle_sha),a.controller_output,feature_contract_pin=feature)
        print(json.dumps(value,ensure_ascii=False,indent=2));return
    need(a.feature_contract is None and a.feature_contract_sha is None,'feature contract option is only for build-controller')
    need(a.blueprint and a.blueprint_sha,'explicit blueprint/SHA required')
    pin=dict(path=a.blueprint,sha256=a.blueprint_sha)
    if a.action=='publish':
        need(a.request and a.request_sha and a.observation and a.observation_sha,'explicit actual request/observation pins required')
        value=publish(pin,dict(path=a.request,sha256=a.request_sha),dict(path=a.observation,sha256=a.observation_sha),reuse_verified_selection=a.reuse_verified_selection)
    else:value=wait(pin,execute=a.action=='wait'and a.execute,reuse_verified_selection=a.reuse_verified_selection)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
