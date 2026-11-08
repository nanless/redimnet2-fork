"""把当前维护源码复制为独立冻结副本，并从显式外部合同生成新身份。"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def helpers(code):
    spec = importlib.util.spec_from_file_location('_annotation_freeze_contract', code/'contracts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_names_source(code):
    """冻结副本优先用自身已固定helper；仓库仅允许已知tools布局。"""
    local=code/'task_names.py'
    if local.exists()or local.is_symlink():
        if not local.is_file()or local.is_symlink():raise ValueError('frozen task naming helper must be a regular file')
        return local
    parent=code.parent/'task_names.py'
    if (code.name!='open_annotation'or code.parent.name!='tools'or not parent.is_file()or parent.is_symlink()):
        raise ValueError('explicit maintained task naming helper missing; do not search arbitrary parents')
    return parent


def freeze(base_path, base_sha, output, version_prefix, normalization_profile=None, asr_compute=None, asr_batching=None, asr_preprocessing=None, asr_preprocessing_sources=None):
    h = helpers(HERE)
    naming_source=task_names_source(HERE)
    base_raw = Path(base_path).read_bytes()
    h.need(hashlib.sha256(base_raw).hexdigest() == base_sha, 'external base contract SHA differs')
    c = json.loads(base_raw)
    h.need(c['purpose'] in {'pilot_validation', 'silver_annotation'}, 'real offline contract required')
    h.need(c.get('model_API_calls_allowed') == 0 and c.get('release_training_labels') is False,
           'automatic API or label release is forbidden')
    if asr_compute is not None:
        c['runtime']['asr_compute'] = copy.deepcopy(asr_compute)
    if asr_batching is not None:
        c['purpose'] = 'pilot_validation'
        c['runtime']['execution_policy'] = h.NATIVE_BATCH_POLICY
        c['runtime']['asr_batching'] = copy.deepcopy(asr_batching)
    if asr_preprocessing is not None:
        c['runtime']['asr_preprocessing']=copy.deepcopy(asr_preprocessing)
    if asr_preprocessing_sources is not None:
        c['asr_preprocessing_sources']=copy.deepcopy(asr_preprocessing_sources)
    if c['runtime'].get('execution_policy')==h.NATIVE_BATCH_POLICY:
        # 通用冻结器只生成合批试验；正式声明只能走下面显式数据工厂。
        c['purpose']='pilot_validation';c.pop('native_formal',None)
    h.validate_execution_policy(c)
    if h.asr_feature_profile(c)is not None:
        from native_b1_feature_padding import validate_source_bindings
        validate_source_bindings(c)
    h.asr_compute_profile(c)
    h.relative_path(version_prefix)
    h.need('/' not in version_prefix, 'version prefix must be one directory name')
    output = Path(output).resolve()
    repository = next((p for p in HERE.parents if (p/'AGENTS.md').is_file() and (p/'redimnet2').is_dir()), None)
    h.need(output != HERE and HERE not in output.parents and not output.exists()
           and (repository is None or (output != repository and repository not in output.parents)),
           'fresh frozen copy must be outside source repository')
    output.mkdir(parents=True)
    for path in HERE.rglob('*'):
        rel = path.relative_to(HERE)
        if path.is_file() and not path.is_symlink() and 'tests' not in rel.parts and '__pycache__' not in rel.parts:
            if path.suffix == '.py' or path.name == 'sidecar.schema.json':
                target = output/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    # 共享表单helper也是新软件包成员；它不属于模型语义ENGINE_MODULES。
    naming_target=output/'task_names.py'
    if naming_target.exists():h.need(h.sha_file(naming_target)==h.sha_file(naming_source),'copied task naming helper differs')
    else:shutil.copyfile(naming_source,naming_target)
    c['parent_contract_sha256'] = base_sha
    from source_clock import TIME_MAPPING
    c['runtime']['boundary_mapping'] = TIME_MAPPING
    from normalization import POLICY, SUPPORTED_POLICIES
    profile = normalization_profile if normalization_profile is not None else c['runtime'].get('normalization_profile', POLICY)
    h.need(profile in SUPPORTED_POLICIES, 'unknown explicit normalization profile')
    c['runtime']['normalization_profile'] = profile
    c['runtime']['normalization'] = 'Explicit normalization_profile; preserve raw and review flags; no invented spoken forms'
    # New source pins are intentional; historical contract bytes are never rewritten.
    c['engine_source_files'] = {name: h.sha_file(output/name) for name in h.engine_modules_for_contract(c)}
    identity = h.digest({key: c[key] for key in ('decoder','models','source_files','runtime',
                                                'alignment','engine_source_files','quality_policy')})[:16]
    records = {}
    purposes = ('pilot_validation',) if h.asr_batch_profile(c) is not None else ('pilot_validation', 'silver_annotation')
    for purpose in purposes:
        current = copy.deepcopy(c)
        current.update(purpose=purpose, version=version_prefix+'-'+purpose+'-'+identity)
        path = output/'contracts'/(current['version']+'.json')
        pin = h.atomic_new(path, current)
        h.read_contract(path, pin)
        records[purpose] = {'path': str(path), 'sha256': pin}
    bundle = {str(path.relative_to(output)):h.sha_file(path) for path in sorted(output.rglob('*'))
              if path.is_file() and 'contracts' not in path.relative_to(output).parts}
    bundle_sha = h.atomic_new(output/'bundle_files.json', bundle)
    receipt = {'schema':'annotation-maintenance-freeze/v1','parent_contract_sha256':base_sha,
               'bundle_sha256':bundle_sha,'contracts':records,'frozen_copy':str(output),
               'GPU_used':False,'models_executed':False,'historical_acceptance_inherited':False,
               'native_ASR_batch_pilot_only':h.asr_batch_profile(c) is not None}
    h.atomic_new(output/'freeze_receipt.json', receipt)
    return receipt


def make_native_formal_contract(pilot_path, pilot_sha, output, version):
    """只生成默认关闭的正式合同数据，不授GPU/运行/标签资格，不修改模型语义。"""
    h=helpers(HERE)
    source=Path(pilot_path)
    h.need(source.is_absolute()and '..'not in source.parts and all(not p.is_symlink()for p in (source,*source.parents))and
           source.is_file()and source.stat().st_size<=64*1024**2,'bounded physical pilot contract input required')
    pilot=h.read_contract(pilot_path,pilot_sha)
    h.need(pilot['purpose']=='pilot_validation'and h.asr_batch_profile(pilot)is not None,
           'explicit native pilot contract input required; quality evidence is checked by runtime admission')
    h.relative_path(version);h.need('/'not in version and version!=pilot['version'],'distinct single-component formal version required')
    target=Path(output);h.need(target.is_absolute()and '..'not in target.parts and not target.exists()and
        all(not p.is_symlink()for p in (target,*target.parents)),'fresh physical formal contract output required')
    code=HERE.resolve();repository=next((p for p in code.parents if (p/'AGENTS.md').is_file()and(p/'redimnet2').is_dir()),None)
    protected=repository if repository is not None else code
    h.need(target!=protected and protected not in target.parents and target not in protected.parents,
           'formal contract data must be outside source repository/deployed code')
    value=copy.deepcopy(pilot);value.update(purpose='silver_annotation',version=version,
        native_formal=copy.deepcopy(h.NATIVE_FORMAL_DECLARATION))
    sha=h.atomic_new(target,value);h.read_contract(target,sha)
    return dict(schema='annotation-native-formal-contract-preparation/v1',pilot_contract=dict(path=str(Path(pilot_path).absolute()),sha256=pilot_sha),
        new_contract=dict(path=str(target),sha256=sha),default_enabled=False,formal_runtime_admitted=False,
        GPU_executed=False,training_labels_released=False,math_source_and_policy_unchanged=True,
        changed_top_level_fields=['purpose','version','native_formal'])


def main():
    from normalization import SUPPORTED_POLICIES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-contract', type=Path, required=True)
    p.add_argument('--base-contract-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--version-prefix', required=True)
    p.add_argument('--normalization-profile', choices=SUPPORTED_POLICIES)
    p.add_argument('--asr-dtype', choices=('float32', 'bfloat16'))
    p.add_argument('--asr-attention', choices=('eager', 'sdpa'))
    p.add_argument('--native-asr-batch-size', type=int, choices=(1,4,8),
                   help='explicit pilot-only fixed ASR cohort; Ultra remains batch1, no silver contract generated')
    p.add_argument('--asr-preprocessing',choices=('official_b1_features_then_pad/v1',),
                   help='explicit new16-source protocol: each official B1 feature/mask/prompt is preserved before batching')
    p.add_argument('--asr-preprocessing-sources',type=Path,help='explicit seven source/config binding JSON for new B1 feature protocol')
    p.add_argument('--asr-preprocessing-sources-sha',help='SHA256 of the explicit preprocessing source binding JSON')
    a = p.parse_args()
    if (a.asr_dtype is None) != (a.asr_attention is None):
        p.error('--asr-dtype and --asr-attention must be supplied together')
    compute = None if a.asr_dtype is None else {'dtype': a.asr_dtype, 'attention': a.asr_attention}
    batching = None if a.native_asr_batch_size is None else dict(policy='fixed_cohort_until_durable/v1',
        max_requests=a.native_asr_batch_size,ordering='frames_then_row_id_within_source_group/v1')
    preprocessing=None if a.asr_preprocessing is None else {'policy':a.asr_preprocessing}
    if(a.asr_preprocessing_sources is None)!=(a.asr_preprocessing_sources_sha is None):p.error('source file and SHA must be supplied together')
    sources=None
    if a.asr_preprocessing_sources is not None:
        raw=a.asr_preprocessing_sources.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=a.asr_preprocessing_sources_sha:p.error('preprocessing source binding SHA differs')
        sources=json.loads(raw)
    print(json.dumps(freeze(a.base_contract,a.base_contract_sha,a.output,a.version_prefix,a.normalization_profile,compute,batching,preprocessing,sources), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
