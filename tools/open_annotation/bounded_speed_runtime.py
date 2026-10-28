"""有界新生产者的显式运行准入；核控制文件与有限证明，不扫描全库或建数据库。

该模块把原始语义锁、原始队列父锁和最近旧生产者的锁一起保留到
GPU父子进程生命周期结束。试验、质量复核和当前平台终态缺一不可；
CPU格式检查不能生成试验通过证明，也不会停止或重启任何旧任务。
"""
from __future__ import annotations
import contextlib
import datetime
import fcntl
import hashlib
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
    from .readonly_progress import strict_json
else:
    import bounded_speed_handoff as H
    import bounded_speed_queue as Q
    import resume_existing_annotation as R
    from verify_portable_successor_mounts import validate_row
    from readonly_progress import strict_json

SCHEMA='annotation-bounded-speed-runtime/v1'
NATIVE_SCHEMA='annotation-bounded-native-runtime/v2'
NATIVE_ADMISSION_SCHEMA='annotation-native-selected-pilot-admission/v2'
PLATFORM_SCHEMA='annotation-bounded-speed-platform-observation/v1'
ADMISSION_SCHEMA='annotation-asr-speed-selected-pilot-admission/v1'
QUALITY_SCHEMA='annotation-asr-speed-quality-review/v1'
PROOF_TRANSPORT={'policy':'parent_sealed_memfd/v1'}
SEALED_PROOF_SCHEMA='annotation-native-parent-selected-proof/v1'
MEMFD_NAME='redimnet-selected-proof'
MAX_PROOF_BYTES=32*1024**2
MAX_PROOF_FILES=100000
_PROOF_CAPABILITY=object()
need=H.need


def pinned(pin):
    return H.read_pin(pin)[0]


def proof_transport_profile(spec):
    """缺省完全沿用旧验证；普通JSON只能请求传输，不能授复用资格。"""
    if 'proof_transport'not in spec:return None
    need(spec.get('schema')==NATIVE_SCHEMA and spec['proof_transport']==PROOF_TRANSPORT
         and isinstance(spec['proof_transport'],dict), 'explicit native sealed memfd proof transport required')
    return dict(PROOF_TRANSPORT)


def _linux_memfd_required():
    need(sys.platform.startswith('linux')and hasattr(os,'memfd_create')and hasattr(os,'pread')
         and all(hasattr(fcntl,k)for k in ('F_ADD_SEALS','F_GET_SEALS','F_SEAL_WRITE','F_SEAL_GROW','F_SEAL_SHRINK','F_SEAL_SEAL')),
         'sealed parent proof requires actual Linux memfd/seals/pread')


def _required_seals():
    return fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL


def _assert_memfd_name(fd):
    need(os.readlink('/proc/self/fd/%d'%fd).lstrip('/')=='memfd:'+MEMFD_NAME+' (deleted)',
         'proof FD is not the explicit anonymous parent memfd')


def _file_stamp(path):
    s=os.lstat(path)
    need(stat.S_ISREG(s.st_mode),'proof input must remain a regular nonlinked file')
    return [s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def _directory_stamp(path):
    s=os.lstat(path);need(stat.S_ISDIR(s.st_mode),'proof ancestor must remain a nonlinked directory')
    # 无关文件的创建不使历史证明失效；目录本身被换掉则必须拒绝。
    return [s.st_dev,s.st_ino,s.st_mode]


def _unique_pins(pins):
    unique={}
    for pin in pins:
        need(isinstance(pin,dict)and set(pin)=={'path','sha256'}and H.is_sha(pin['sha256']), 'explicit proof input pin required')
        p=Path(pin['path']);need(p.is_absolute()and '..'not in p.parts,'canonical absolute proof input required')
        need(str(p)not in unique or unique[str(p)]==pin['sha256'],'one proof path has contradictory SHA pins')
        unique[str(p)]=pin['sha256']
    need(0<len(unique)<=MAX_PROOF_FILES,'bounded proof input denominator exceeded')
    return [dict(path=path,sha256=sha)for path,sha in sorted(unique.items())]


def _snapshot_verified_pins(pins):
    """复用首次verify的SHA扫描；before/hash/after把指纹与已验字节绑定。"""
    pins=_unique_pins(pins);directories=sorted({str(p)for pin in pins for p in Path(pin['path']).parents})
    before_dirs={p:_directory_stamp(p)for p in directories};files=[]
    for pin in pins:
        before=_file_stamp(pin['path']);R.pinned_file(pin);after=_file_stamp(pin['path'])
        need(before==after,'proof input changed around verified SHA read')
        files.append(dict(pin=pin,stamp=after))
    need(all(_directory_stamp(p)==stamp for p,stamp in before_dirs.items()),'proof ancestor changed during verified snapshot')
    result=dict(files=files,directories=[dict(path=p,stamp=stamp)for p,stamp in before_dirs.items()])
    _verify_snapshot(result);return result


def _verify_snapshot(snapshot):
    files=snapshot.get('files');directories=snapshot.get('directories')
    need(isinstance(files,list)and 0<len(files)<=MAX_PROOF_FILES and isinstance(directories,list), 'complete sealed input snapshot required')
    pins=_unique_pins([v['pin']for v in files])
    need(len(pins)==len(files)and pins==[v['pin']for v in files], 'sealed snapshot input set/order differs')
    expected=sorted({str(p)for pin in pins for p in Path(pin['path']).parents})
    need([v.get('path')for v in directories]==expected,'sealed snapshot ancestor coverage incomplete')
    for entry in directories:need(_directory_stamp(entry['path'])==entry['stamp'],'sealed proof ancestor identity changed')
    for entry in files:
        _verify_snapshot_file(entry)
    for entry in directories:need(_directory_stamp(entry['path'])==entry['stamp'],'sealed proof ancestor identity changed during SHA read')


def _verify_snapshot_file(entry):
    """祖先由调用者前后各核一次；每个文件仍实际读完整SHA，并核FD与路径。"""
    need(hasattr(os,'O_NOFOLLOW')and hasattr(os,'O_NONBLOCK'),'safe nonfollowing proof FD flags required')
    pin=entry['pin'];expected=entry['stamp'];fd=None
    def stamp(s):return [s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    need(_file_stamp(pin['path'])==expected,'sealed proof input identity/content timestamp changed')
    try:
        # 不跟随末级软链接；FIFO等替换也不能在fstat拒绝前把父进程挂住。
        fd=os.open(pin['path'],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode)and stamp(before)==expected,'sealed proof input identity/content timestamp changed')
        actual=hashlib.sha256();size=0
        while True:
            block=os.read(fd,4*1024**2)
            if not block:break
            size+=len(block);actual.update(block)
        need(stamp(os.fstat(fd))==expected and _file_stamp(pin['path'])==expected,
             'sealed proof input changed during SHA read')
        # Linux同tick全部stat值可能碰撞；指纹从不替代实际内容摘要。
        need(size==expected[3]and actual.hexdigest()==pin['sha256'],'pinned file differs: '+pin['path'])
    finally:
        if fd is not None:os.close(fd)


def _boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _process_start(pid):
    # legacy25／feature26 CPU外层不带GPU资源源；仅真GPU父子传输分支才加载。
    if __package__:from .gpu_common import process_start
    else:from gpu_common import process_start
    return process_start(pid)


def _transport_input_pins(compiled):
    # 与同CLI scope共用明确的完整输入角色枚举：含所有bundle成员和SDK七源。
    # 这里只取小bundle映射／源码pin，不重新解析comparison/readback或读权重。
    if __package__:from .native_batch_admission import _selection_inputs
    else:from native_batch_admission import _selection_inputs
    return _unique_pins([*compiled['pins'],*_selection_inputs(compiled['selected'],compiled['new_contract'],
        compiled['spec']['purpose'],compiled['spec']['selected_pilot_admission'])])


class _SealedSelectedProof:
    """只能由完整父检查或真实继承memfd认证创建的内部能力。"""
    def __init__(self,fd,payload,ref,role,token):
        need(token is _PROOF_CAPABILITY,'sealed proof is not a runtime capability')
        self.fd,self.payload,self.ref,self.role,self._capability=fd,payload,ref,role,token

    def assert_identity(self):
        _linux_memfd_required();_assert_memfd_name(self.fd);s=os.fstat(self.fd)
        need(stat.S_ISREG(s.st_mode)and [s.st_dev,s.st_ino,s.st_size]==[self.ref[k]for k in ('device','inode','bytes')]
             and fcntl.fcntl(self.fd,fcntl.F_GET_SEALS)&_required_seals()==_required_seals(), 'proof FD is not the original immutable sealed memfd')
        issuer=self.payload['issuer'];pid=os.getpid()if self.role=='parent'else os.getppid()
        need(type(issuer.get('pid'))is int and issuer['pid']==pid and _process_start(pid)==issuer['start_ticks']
             and _boot_id()==issuer['boot_id'],'sealed proof actual parent PID/start/boot differs')
        if self.role=='child':
            held=os.stat('/proc/%d/fd/%d'%(pid,self.ref['parent_fd']))
            need((held.st_dev,held.st_ino)==(s.st_dev,s.st_ino),'sealed proof is not still held by the actual parent')

    def verify_files(self):
        self.assert_identity();_verify_snapshot(self.payload['snapshot'])

    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None

    def export(self):
        need(self.role=='parent','child cannot issue another parent proof')
        self.assert_identity();return dict(self.ref,fd=self.fd)


def _new_parent_proof(lease):
    _linux_memfd_required();c=lease.compiled
    pins=[*_transport_input_pins(c),lease.unique_claim_pin,*[item['script']for item in c['old']['checks']]]
    # 原lineage为语义比较；同时把它的实际字节纳入本次不可变输入快照。
    pins.append(dict(path=c['lineage_path'],sha256=R.file_sha(c['lineage_path'])))
    snapshot=_snapshot_verified_pins(pins)
    payload=dict(schema=SEALED_PROOF_SCHEMA,transport=PROOF_TRANSPORT,runtime_spec=lease.pin,
        selected_admission=c['spec']['selected_pilot_admission'],new_contract=c['spec']['new_contract'],new_bundle=c['spec']['new_bundle'],
        unique_producer_claim=lease.unique_claim_pin,issuer=dict(pid=os.getpid(),start_ticks=_process_start(os.getpid()),boot_id=_boot_id()),
        selected=c['selected'],synthetic_fixture=c['spec']['purpose']=='cpu_fixture',snapshot=snapshot)
    raw=H.P.canonical(payload);need(len(raw)<=MAX_PROOF_BYTES,'sealed selected proof exceeds bounded memfd size')
    fd=os.memfd_create(MEMFD_NAME,os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    try:
        os.fchmod(fd,0o600);offset=0
        while offset<len(raw):
            written=os.write(fd,raw[offset:]);need(written>0,'memfd proof write made no progress');offset+=written
        fcntl.fcntl(fd,fcntl.F_ADD_SEALS,_required_seals());s=os.fstat(fd)
        ref=dict(parent_fd=fd,device=s.st_dev,inode=s.st_ino,bytes=s.st_size,sha256=hashlib.sha256(raw).hexdigest())
        proof=_SealedSelectedProof(fd,payload,ref,'parent',_PROOF_CAPABILITY);proof.verify_files();return proof
    except BaseException:os.close(fd);raise


def _adopt_parent_proof(runtime_pin,evidence):
    _linux_memfd_required();ref=evidence.get('selected_proof')
    need(isinstance(ref,dict)and set(ref)=={'fd','parent_fd','device','inode','bytes','sha256'}
         and all(type(ref[k])is int for k in ('fd','parent_fd','device','inode','bytes'))
         and ref['fd']>2 and ref['parent_fd']>2 and 0<ref['bytes']<=MAX_PROOF_BYTES and H.is_sha(ref['sha256']),
         'explicit inherited sealed selected-proof FD required; ordinary JSON is not proof')
    fd=os.dup(ref['fd'])
    try:
        _assert_memfd_name(fd)
        s=os.fstat(fd);need(stat.S_ISREG(s.st_mode)and [s.st_dev,s.st_ino,s.st_size]==[ref[k]for k in ('device','inode','bytes')]
             and fcntl.fcntl(fd,fcntl.F_GET_SEALS)&_required_seals()==_required_seals(),'inherited selected proof is not a fully sealed memfd')
        raw=os.pread(fd,MAX_PROOF_BYTES+1,0)
        need(len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'],'inherited sealed proof byte identity differs')
        payload=strict_json(raw);need(payload.get('schema')==SEALED_PROOF_SCHEMA and payload.get('transport')==PROOF_TRANSPORT
             and payload.get('runtime_spec')==runtime_pin and payload.get('unique_producer_claim')==evidence.get('unique_producer_claim'),
             'sealed proof runtime/producer claim differs')
        proof=_SealedSelectedProof(fd,payload,{k:v for k,v in ref.items()if k!='fd'},'child',_PROOF_CAPABILITY)
        proof.verify_files();return proof
    except BaseException:os.close(fd);raise


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


def guard_native_formal_route(contract,runtime_pin,*,lease=None,require_lease=False):
    """正式合批不借一般索引或旧单窗准入；此检查本身不授GPU资格。"""
    runtime=contract.get('runtime',{})
    native=runtime.get('execution_policy')=='asr_fixed_cohort_fa_single/v1'
    formal=native and contract.get('purpose')=='silver_annotation'
    if not formal:return False
    from contracts import asr_batch_profile
    need(asr_batch_profile(contract)is not None,'explicit native formal contract structure required')
    need(runtime_pin is not None,'native silver requires authenticated bounded native runtime v2; ordinary v1 index is forbidden')
    spec=pinned(runtime_pin)
    need(spec.get('schema')==NATIVE_SCHEMA and spec.get('purpose')=='production','native silver cannot borrow single-window runtime/admission')
    if require_lease:
        need(lease is not None and lease.pin==runtime_pin and lease.compiled['spec'].get('schema')==NATIVE_SCHEMA and
             lease.compiled.get('new_contract')==contract and lease.compiled.get('selected',{}).get('schema')==NATIVE_ADMISSION_SCHEMA,
             'native silver worker requires inherited verified native v2 lineage lease')
        lease.assert_held()
    return True


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


def compile_runtime(runtime_pin,*,fresh=True,selection_scope=None,_selected_proof=None):
    """固定薄索引、前一真实run和原始祖先；不运行旧union或任何数据库查询。"""
    spec=pinned(runtime_pin);purpose=spec.get('purpose')
    need(spec.get('schema')in {SCHEMA,NATIVE_SCHEMA} and purpose in {'production','cpu_fixture'},'explicit bounded runtime schema/purpose required')
    transport=proof_transport_profile(spec)
    if _selected_proof is not None:
        need(isinstance(_selected_proof,_SealedSelectedProof)and _selected_proof._capability is _PROOF_CAPABILITY
             and _selected_proof.role=='child'and transport is not None and selection_scope is None,
             'selected reuse requires authenticated inherited memfd capability')
        _selected_proof.verify_files()
        payload=_selected_proof.payload
        need(payload['runtime_spec']==runtime_pin and payload['selected_admission']==spec['selected_pilot_admission']
             and payload['new_contract']==spec['new_contract']and payload['new_bundle']==spec['new_bundle'],
             'sealed selected facts do not bind this runtime/contract/bundle')
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
    if spec['schema']==NATIVE_SCHEMA:
        if __package__:from .native_batch_admission import check_native_selected_pilot
        else:from native_batch_admission import check_native_selected_pilot
        if _selected_proof is not None:
            # 只在真parent仍持有封印FD时采用其已验结果，磁盘布尔/普通JSON不可进入。
            selected=json.loads(json.dumps(_selected_proof.payload['selected']))
            need(selected.get('schema')==NATIVE_ADMISSION_SCHEMA and selected.get('purpose')==purpose,
                 'sealed selected fact schema/purpose differs')
            original_selected=pinned(spec['selected_pilot_admission'])
            derived={'verified_evidence_pins','verified_native_bundle_sha256','verified_pilot_bundle_sha256'}
            need(set(selected)==set(original_selected)|derived and
                 {k:v for k,v in selected.items()if k not in derived}==original_selected,
                 'sealed selected facts differ from actual pinned admission JSON')
        elif selection_scope is not None:
            selected=check_native_selected_pilot(spec['selected_pilot_admission'],new,purpose,selection_scope=selection_scope)
        else:selected=check_native_selected_pilot(spec['selected_pilot_admission'],new,purpose)
        need(selected['verified_native_bundle_sha256']==spec['new_bundle']['sha256'],'native formal GPU executable bundle must equal actual selected pilot bundle')
    else:
        need(new.get('runtime',{}).get('execution_policy')!='asr_fixed_cohort_fa_single/v1',
             'old runtime v1 refuses native contract; use explicit native v2')
        selected=check_selected_pilot(spec['selected_pilot_admission'],new,purpose)
    run=R.physical(spec['new_run_root']);code=R.physical(spec['new_code_root'])
    need(base in run.parents and base in code.parents and run!=code and run not in code.parents and code not in run.parents,'new runtime/code namespace differs')
    need(index['new_queue_root']==str(run/'queue')and index['new_sidecar_root']==str(run/'labels')and
         spec['device_lock_root']==old['device_lock_root'],'new run/queue/labels/shared UUID locks differ')
    protected=[oldrun,Path(index['old_queue_root']).parent,R.physical(old['code_root']),Path(index['parent_index']['path']).parent,Path(spec['index']['path']).parent,semantic.parent]
    need(all(run!=p and run not in p.parents and p not in run.parents for p in protected),'new run overlaps old/control/input namespace')
    need(spec['new_bundle']['path']==str(code/'bundle_files.json'),'new bundle location differs')
    terminal=check_platform(spec['current_platform_observation'],index,fresh=fresh)
    # 当前选型含有限3000/512大证明与原批次，仍实际核SHA；
    # 祖先全库证明只保留已核历史引用，不重新遍历全库。
    pins=[runtime_pin,spec['index'],spec['previous_runtime_spec'],prior['old_control'],prior['freeze_receipt'],prior['union_spec'],
          old['contract'],old['full_index'],old['admission'],old['original_claim'],index['old_contract'],index['parent_index'],
          spec['new_contract'],spec['new_bundle'],spec['selected_pilot_admission'],spec['current_platform_observation'],terminal,*selected['verified_evidence_pins']]
    compiled=dict(spec=spec,index=index,reader=reader,old=old,admission=admission,new_contract=new,selected=selected,
        previous=prior,original_contract=original,ancestor_paths=[str(semantic),str(oldq/'.parent.lock')],
        original_commands=checks,terminal_pin=terminal,pins=pins,lineage_path=str(lineage),lineage_value=lineage_value,
        unique_claim_path=str(semantic.with_name('bounded-speed-'+H.P.digest(dict(old_contract=index['old_contract'],parent_index=index['parent_index'],old_queue_root=index['old_queue_root']))+'.claim.json')),
        unique_claim_core=dict(schema='annotation-bounded-speed-successor-claim/v1',old_contract=index['old_contract'],
            parent_index=index['parent_index'],old_queue_root=index['old_queue_root'],index=spec['index'],
            new_contract=spec['new_contract'],new_bundle=spec['new_bundle'],new_code_root=spec['new_code_root'],
            new_run_root=spec['new_run_root'],selected_pilot_admission=spec['selected_pilot_admission'],
            status='claimed_no_implicit_new_producer',synthetic_fixture=purpose=='cpu_fixture'))
    if _selected_proof is not None:_assert_compiled_proof_binding(_selected_proof,runtime_pin,compiled)
    return compiled


def _assert_compiled_proof_binding(proof,runtime_pin,compiled):
    """完整finite输入集合不可被一个较小的、看似自洽的快照代替。"""
    p=proof.payload;c=compiled
    need(p['runtime_spec']==runtime_pin and p['new_bundle']==c['spec']['new_bundle']and
         p['new_contract']==c['spec']['new_contract']and p['selected_admission']==c['spec']['selected_pilot_admission']and
         p['unique_producer_claim']['path']==c['unique_claim_path']and p['selected']==c['selected'],
         'sealed selected/runtime/permanent producer binding differs')
    need(p.get('synthetic_fixture')is(c['spec']['purpose']=='cpu_fixture'), 'CPU sealed fixture cannot admit real production')
    indexed={v['pin']['path']:v['pin']for v in p['snapshot']['files']}
    need(c['lineage_path']in indexed,'sealed proof omitted original lineage claim')
    expected=_unique_pins([*_transport_input_pins(c),p['unique_producer_claim'],indexed[c['lineage_path']],
                          *[item['script']for item in c['old']['checks']]])
    need(expected==[v['pin']for v in p['snapshot']['files']],'sealed proof omitted or added verified finite inputs')
    if proof.role=='child'and c['spec']['purpose']=='production':
        # GPU bootstrap已核真实父票据；这里再把memfd发行者绑定同一固定GPU入口。
        # 一个自己写普通JSON或-c脚本的父不能借其PPID自称通过完整selected验证。
        pid=p['issuer']['pid'];raw=Path('/proc/%d/cmdline'%pid).read_bytes()
        argv=[v.decode()for v in raw.rstrip(b'\0').split(b'\0')]
        need(len(argv)>=3 and argv[1]=='-B'and argv[2]==str(Path(c['spec']['new_code_root'])/'gpu_launch.py'),
             'sealed proof issuer is not the fixed GPU parent entry')
        binary=c['old'].get('python_binary');need(isinstance(binary,dict),'actual parent interpreter pin missing')
        R.pinned_file(binary)
        need(os.readlink('/proc/%d/exe'%pid)==binary['path'],'sealed proof issuer interpreter differs from original pinned Python')
        for key,expected_value in (('--bundle-sha',c['spec']['new_bundle']['sha256']),
            ('--contract',c['spec']['new_contract']['path']),('--contract-sha',c['spec']['new_contract']['sha256']),
            ('--shard-index',c['spec']['index']['path']),('--index-sha',c['spec']['index']['sha256']),
            ('--bounded-runtime-spec',runtime_pin['path']),('--bounded-runtime-sha',runtime_pin['sha256'])):
            need(argv.count(key)==1 and argv.index(key)+1<len(argv)and argv[argv.index(key)+1]==expected_value,
                 'sealed proof issuer actual command/runtime arguments differ')


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
        self._initialize(runtime_pin)

    def _initialize(self,runtime_pin,proof=None):
        self.pin=runtime_pin;self.proof=proof
        self.compiled=compile_runtime(runtime_pin,_selected_proof=proof)
        self.handles=[];self.current=None;self.held=False;self.unique_claim_pin=None
        self.transport=proof_transport_profile(self.compiled['spec'])

    def _small_pins(self):
        # 不按文件体积猜“小”：仅保留原控制链的明确关键JSON与脚本。
        # 大finite证明仍逐文件核实际SHA，memfd仅省JSON／映射重构。
        facts=self.compiled['selected'];selected={p['path']for p in facts['verified_evidence_pins']}
        common=[p for p in self.compiled['pins']if p['path']not in selected]
        for key in ('selected_pilot_admission','new_contract','new_bundle','index','current_platform_observation'):
            common.append(self.compiled['spec'][key])
        common.extend([self.pin,self.unique_claim_pin,*[v['script']for v in self.compiled['old']['checks']]])
        for key in ('candidate_contract','baseline_track','main_track','recovery_track','pilot_result','quality_review',
                    'control_source_review','formal_execution_bundle'):
            if key in facts:common.append(facts[key])
        for value in facts.get('resource_evidence',{}).values():common.extend(value.values())
        # 原完整check已核这些角色；再次读小result/plan只取固定的plan/matrix pin。
        result=pinned(facts['pilot_result']);common.append(result['plan'])
        plan=pinned(result['plan']);common.append(plan['main_prepared'])
        if __package__:from . import native_batch_admission as native_admission
        else:import native_batch_admission as native_admission
        common.append(dict(path=str(Path(native_admission.__file__).resolve()),sha256=R.file_sha(native_admission.__file__)))
        return _unique_pins(common)

    def _identity(self,entry,fd):
        path=R.physical(entry['path']);a=os.fstat(fd);b=path.stat()
        need(stat.S_ISREG(a.st_mode)and(a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)==(entry['device'],entry['inode']),'ancestor inherited lock identity differs')
        if self.compiled['spec']['purpose']=='production':
            info=Path('/proc/self/fdinfo')/str(fd)
            need(info.is_file()and any('FLOCK'in line and 'WRITE'in line for line in info.read_text().splitlines()if line.startswith('lock:')),'ancestor lock ownership not visible')

    def __enter__(self):
        try:
            if self.transport is not None:_linux_memfd_required()
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
        need(isinstance(evidence,dict)and evidence.get('runtime_spec')==runtime_pin,'inherited runtime ticket binding differs')
        declaration=proof_transport_profile(pinned(runtime_pin));proof=None
        if declaration is not None:
            proof=_adopt_parent_proof(runtime_pin,evidence)
            try:
                obj=cls.__new__(cls);obj._initialize(runtime_pin,proof)
            except BaseException:proof.close();raise
        else:
            need('selected_proof'not in evidence,'undeclared sealed proof transport cannot change legacy runtime')
            obj=cls(runtime_pin)
        try:
            entries=evidence.get('ancestor_locks');need(isinstance(entries,list)and len(entries)==2 and
                {e['path']for e in entries}==set(obj.compiled['ancestor_paths'])and len({e['fd']for e in entries})==2,'ancestor inherited lock set differs')
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
        if self.transport is not None:
            if self.proof is None:self.proof=_new_parent_proof(self)
            _assert_compiled_proof_binding(self.proof,self.pin,self.compiled)
            self.proof.verify_files()
            for pin in self._small_pins():R.pinned_file(pin)
        else:
            for pin in self.compiled['pins']:R.pinned_file(pin)
        need(R.ref_unpinned(self.compiled['lineage_path'])==self.compiled['lineage_value'],'previous permanent claim changed')
        for check in self.compiled['old']['checks']:R.pinned_file(check['script'])
        need(self.unique_claim_pin is not None,'unique producer claim not admitted')
        R.pinned_file(self.unique_claim_pin)
        self.current.verify_frozen_state()

    @property
    def pass_fds(self):
        self.assert_held();fds=tuple(h['fd']for h in self.handles)+self.current.pass_fds
        if self.transport is not None:
            need(self.proof is not None,'parent selected proof was not sealed after full validation')
            self.proof.assert_identity();fds+=(self.proof.fd,)
        return fds

    @property
    def ticket_evidence(self):
        self.assert_held();value=dict(runtime_spec=self.pin,ancestor_locks=[dict(h)for h in self.handles],current_old_locks=self.current.fd_evidence,unique_producer_claim=self.unique_claim_pin)
        if self.transport is not None:
            need(self.proof is not None,'parent selected proof was not sealed after full validation')
            value['selected_proof']=self.proof.export()
        return value

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
        if self.proof is not None:self.proof.close();self.proof=None
