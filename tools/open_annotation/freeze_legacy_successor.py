"""原样复制旧生产源码，只改新合同version；不继承新任务GPU成功。

默认读取仓库内已固定的2026-10-25生产副本。冻结只核源码和合同元数据，
不导入torch/Qwen/vLLM，不读取模型权重或原音，不写旧producer。
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import types
import json
import os
from pathlib import Path
import re

SNAPSHOT=Path(__file__).resolve().parent/'legacy_snapshots'/'production_20261025'
BUNDLE_SHA256='54cd49b77ce3c82bbd61bc4ca3cc9389016e908e976ef2574fb953b3249554b0'
ENGINE_MODULES=('contracts.py','audio_windows.py','qwen_backend.py','worker.py','schema.py',
                'sidecar.py','work_queue.py','report.py','source_registry.py','manifest_builder.py',
                'normalization.py','alignment_observer.py','quality_axes.py')
LEGACY_RUNTIME={'GPU_error_policy':'abort_own_attempt_no_silent_batch_reduction','TF32':False,
    'attention':'eager','boundary_mapping':'source_offset_floor_start_ceil_end_v1','dtype':'float32',
    'execution_policy':'single_window_independent','length_ratio_limit':1.0,'max_batch':1,'max_new_tokens':512,
    'normalization':'NFKC punctuation-to-space collapse-whitespace/v1; preserve raw; no invented spoken forms',
    'ownership':'source-clock midpoint-owned continuous-context/v1','resampling':'scipy_resample_poly_gcd_float32/v1',
    'source_batch_rows':8,'window':{'context_seconds':2.0,'core_seconds':24.0}}


def need(value,message):
    if not value:raise ValueError(message)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def sha(raw):return hashlib.sha256(raw).hexdigest()


def physical(value):
    p=Path(value)
    need(p.is_absolute() and '..' not in p.parts and all(not x.is_symlink() for x in (p,*p.parents)),
         'physical absolute path without symlink/traversal required')
    return p


def read_bytes(path):
    p=physical(path)
    with p.open('rb')as stream:
        before=os.fstat(stream.fileno());raw=stream.read();after=os.fstat(stream.fileno())
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and len(raw)==before.st_size,
         'source changed while reading')
    return raw


def strict_json(raw):
    def pairs(rows):
        result={}
        for k,v in rows:need(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def verify_snapshot(snapshot=SNAPSHOT):
    """固定整个原bundle，AST核13模块，不执行推理源码。"""
    root=physical(snapshot);bundle_raw=read_bytes(root/'bundle_files.json')
    need(sha(bundle_raw)==BUNDLE_SHA256,'original deployed bundle SHA differs')
    files=strict_json(bundle_raw);need(isinstance(files,dict) and len(files)==23,'original 23 bundle members required')
    data={}
    for name,pin in files.items():
        rel=Path(name)
        need(not rel.is_absolute() and str(rel)==name and '..'not in rel.parts and '.git'not in rel.parts and
             re.fullmatch('[0-9a-f]{64}',pin),'unsafe bundle member or pin')
        raw=read_bytes(root/rel);need(sha(raw)==pin,'original bundle member differs: '+name);data[name]=raw
    actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}
    need(actual==set(files)|{'bundle_files.json'},'snapshot has missing/extra files')
    need(all(not p.is_symlink()for p in root.rglob('*')),'snapshot descendant symlink refused')
    tree=ast.parse(data['contracts.py']);values=[ast.literal_eval(n.value)for n in tree.body
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ENGINE_MODULES'for t in n.targets)]
    need(values==[ENGINE_MODULES] and set(ENGINE_MODULES)<=set(files),'original 13 semantic engine list differs')
    return root,files,data,bundle_raw


def _read_base(path,pin,root,files):
    raw=read_bytes(path);need(isinstance(pin,str) and re.fullmatch('[0-9a-f]{64}',pin)and sha(raw)==pin,'base contract SHA differs')
    base=strict_json(raw)
    need(base.get('schema')=='redimnet2-production-contract/v2' and base.get('purpose')=='silver_annotation' and
         type(base.get('model_API_calls_allowed'))is int and base['model_API_calls_allowed']==0 and
         base.get('release_training_labels')is False,'original silver/API0/no-release contract required; CPU fixture is not production')
    need(base.get('engine_source_files')=={n:files[n]for n in ENGINE_MODULES},'base 13 engine pins differ from deployed snapshot')
    need(canonical(base.get('runtime'))==canonical(LEGACY_RUNTIME),'old FP32/eager/single-window/512/clock profile changed')
    source_pins=base.get('source_files')
    need(isinstance(source_pins,dict)and source_pins and all(isinstance(p,str)and Path(p).is_absolute()and
         isinstance(h,str)and re.fullmatch('[0-9a-f]{64}',h)for p,h in source_pins.items()),'external Qwen source identities missing')
    # 只载入已逐字核实的旧stdlib合同校验器，不导入当前14模块版本。
    module=types.ModuleType('_redimnet2_frozen_legacy_contracts');module.__file__=str(root/'contracts.py')
    # 显式compile/exec避免SourceFileLoader给冻结原目录写__pycache__。
    contract_source=read_bytes(root/'contracts.py');need(sha(contract_source)==files['contracts.py'],'old loader source changed')
    exec(compile(contract_source,module.__file__,'exec'),vars(module))
    validated=module.read_contract(path,pin)
    need(validated==base,'old contract loader and strict JSON disagree')
    for m in base['models'].values():
        for item in m['files'].values():need(isinstance(item['sha256'],str)and re.fullmatch('[0-9a-f]{64}',item['sha256']),
                                            'model file SHA malformed')
    return base


def validate_output(output,snapshot,base_path):
    out=physical(output)
    need(out not in {Path('/'),Path('/root')} and out.parent.is_dir()and not out.exists(),'fresh external output with existing parent required')
    need(out!=snapshot and snapshot not in out.parents and out not in snapshot.parents and out!=base_path,
         'output overlaps fixed inputs')
    need('.git'not in out.parts and all(not (p/'.git').exists() for p in (out,*out.parents)),
         'production freeze output must be outside Git/repository trees')
    return out


def _write_new(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(str(path.parent),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def freeze(base_contract_path,base_sha,output,version,*,snapshot=SNAPSHOT):
    """只改version并写新的独立包；失败保留新目录，不覆盖旧输入。"""
    base_path=physical(base_contract_path);root,files,data,bundle_raw=verify_snapshot(snapshot)
    base=_read_base(base_path,base_sha,root,files)
    need(isinstance(version,str)and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',version)and
         version not in {'.','..'}and version!=base['version'],'safe distinct single-component version required')
    out=validate_output(output,root,base_path);new=copy.deepcopy(base);new['version']=version
    need({k:v for k,v in new.items()if k!='version'}=={k:v for k,v in base.items()if k!='version'},'only version may change')
    new_raw=canonical(new);new_sha=sha(new_raw)
    out.mkdir(mode=0o700)
    for name,raw in data.items():_write_new(out/'code'/name,raw)
    _write_new(out/'code/bundle_files.json',bundle_raw)
    for name,pin in files.items():need(sha(read_bytes(out/'code'/name))==pin,'copied source readback differs')
    need(sha(read_bytes(out/'code/bundle_files.json'))==BUNDLE_SHA256,'copied bundle changed')
    contract_path=out/'production-contract.json';_write_new(contract_path,new_raw)
    need(strict_json(read_bytes(contract_path))==new,'new contract readback differs')
    # lineage只能写这里，不能给新合同增删parent字段或重贴旧结果。
    receipt={'schema':'annotation-legacy-successor-freeze/v1','base_contract':{'path':str(base_path),'sha256':base_sha},
        'new_contract':{'path':str(contract_path),'sha256':new_sha},'new_version':version,
        'code_root':str(out/'code'),'bundle':{'path':str(out/'code/bundle_files.json'),'sha256':BUNDLE_SHA256},
        'snapshot_root':str(root),'bundle_members_copied':23,'semantic_engine_modules':list(ENGINE_MODULES),
        'semantic_engine_bytes_unchanged':True,'all_contract_fields_except_version_unchanged':True,
        'model_weight_files_rehashed':False,'external_runtime_sources_reopened':False,
        'historical_W2_evidence_verified_by_this_tool':False,'new_GPU_execution_verified':False,
        'new_task_submitted':False,'runtime_admission_granted':False,'model_API_calls':0,
        'training_or_evaluation_eligibility_released':False,
        'retained_complete_old_sidecars_must_be_excluded_from_remaining':True,
        'old_stage_cache_preserved_not_rekeyed':True,'unfinished_stage_work_may_be_recomputed':True}
    _write_new(out/'freeze-receipt.json',canonical(receipt))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base-contract','base-sha','output','version'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();print(json.dumps(freeze(args.base_contract,args.base_sha,args.output,args.version),ensure_ascii=False))


if __name__=='__main__':main()
