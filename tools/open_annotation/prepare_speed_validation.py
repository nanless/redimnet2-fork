"""准备三个单因素转写版本和两卡命令；不执行命令、不读取模型或原声。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

# 作为脚本和 -m 入口均从明确同目录找维护工具，避免cwd改变依赖来源。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_contract as freezer
from contracts import atomic_new, need, safe_join
from source_registry import source_id

VARIANTS = {'legacy': None,
            'asr-fp32-sdpa': {'dtype': 'float32', 'attention': 'sdpa'},
            'asr-bf16-sdpa': {'dtype': 'bfloat16', 'attention': 'sdpa'}}
DOMAINS = {'voxceleb2', 'voxblink2', 'cnceleb2'}


def pinned(ref):
    need(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'explicit path/SHA required')
    p = Path(ref['path'])
    need(p.is_absolute() and all(not x.is_symlink() for x in (p, *p.parents)), 'physical absolute input required')
    raw = p.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'input SHA differs')
    return p, json.loads(raw)


def check_index(ref, count):
    path, value = pinned(ref)
    need(value['schema'] == 'redimnet2-shard-index/v1' and value['source_count'] == count, 'validation index count/schema differs')
    sources, datasets = {}, set()
    for shard in value['shards']:
        p = safe_join(path.parent, shard['path'])
        raw = p.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == shard['sha256'], 'validation shard SHA differs')
        rows = [json.loads(x) for x in raw.splitlines()]
        need(len(rows) == shard['count'], 'validation shard count differs')
        for row in rows:
            need(row['dataset'] in DOMAINS and set(row['roles']) and
                 set(row['roles']) <= {'train_ptn', 'train_lm'}, 'not an actual three-training-domain source')
            need(row['source_id'] not in sources, 'duplicate validation source')
            need(row['source_id'] == source_id(row['dataset'], row['relative_path']), 'source ID/path identity differs')
            sources[row['source_id']] = row
            datasets.add(row['dataset'])
    need(len(sources) == count and datasets == DOMAINS, 'source denominator or three-domain coverage differs')
    return sources


def prepare(spec_path, spec_sha, output):
    _, spec = pinned({'path': str(spec_path), 'sha256': spec_sha})
    need(spec['schema'] == 'annotation-speed-matrix-preparation/v1', 'wrong preparation schema')
    _, base = pinned(spec['base_contract'])
    need('asr_compute' not in base['runtime'], 'legacy baseline must have no explicit ASR override')
    indices = spec['indices']
    main = check_index(indices['main'], 3000)
    recovery = check_index(indices['recovery'], 512)
    check_index(indices['screen'], 30)
    need(all(main.get(k) == row for k, row in recovery.items()), 'recovery is not the exact source subset of main')
    uuids = spec['target_uuids']
    import re
    need(isinstance(uuids, list) and len(uuids) == len(set(uuids)) == 2 and
         all(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', x) for x in uuids), 'two explicit physical UUIDs required')
    output = Path(output).absolute()
    repo = Path(__file__).resolve().parents[2]
    need(not output.exists() and all(not x.is_symlink() for x in (output, *output.parents)) and
         output != repo and repo not in output.parents, 'fresh output outside repository required')
    protected = [Path(spec['base_contract']['path']).parent]
    if protected[0].name == 'contracts': protected.append(protected[0].parent)
    protected.extend(Path(v['path']).parent for v in indices.values())
    need(all(output != p and p not in output.parents for p in protected),
         'output overlaps historical contract/code or index tree')
    locks = Path(spec['device_lock_root'])
    need(locks.is_absolute() and locks != output and output not in locks.parents and locks not in output.parents,
         'device locks must be outside task tree')
    need(all(not p.is_symlink() for p in (locks, *locks.parents)), 'device-lock path symlink refused')
    need(Path(spec['python']).is_absolute(), 'explicit absolute target interpreter required')
    records = {}
    for name, profile in VARIANTS.items():
        root = output / name
        frozen = freezer.freeze(spec['base_contract']['path'], spec['base_contract']['sha256'],
                                root / 'code', 'speed-' + name, asr_compute=profile)
        pilot = frozen['contracts']['pilot_validation']
        cfg = {'schema': 'annotation-three-training-validation/v1', 'task_root': str(root),
               'code_root': frozen['frozen_copy'], 'output_root': str(root / 'validation'),
               'bundle_sha256': frozen['bundle_sha256'], 'pilot_contract_name': Path(pilot['path']).name,
               'pilot_contract_sha256': pilot['sha256'], 'main_index': indices['main'],
               'recovery_index': indices['recovery'], 'vertical_index_sha256': indices['screen']['sha256'],
               'validation_UUIDs': uuids, 'device_lock_root': str(locks)}
        cfg_path = root / 'validation-config.json'
        cfg_sha = atomic_new(cfg_path, cfg)
        stage = root / 'validation' / 'vertical30'
        command = [spec['python'], '-B', str(root / 'code' / 'gpu_launch.py'), '--execute',
                   '--bundle-sha', frozen['bundle_sha256'], '--contract', pilot['path'],
                   '--contract-sha', pilot['sha256'], '--shard-index', indices['screen']['path'],
                   '--index-sha', indices['screen']['sha256'], '--queue-root', str(stage / 'queue'),
                   '--output-root', str(stage / 'labels'), '--output', str(stage / 'resources-attempt1'),
                   '--target-uuids', ','.join(uuids), '--run-root', str(root),
                   '--device-lock-root', str(locks), '--max-runtime-seconds', '900', '--full-verify']
        supervisor = [spec['python'], '-B', str(root / 'code' / 'validate_two_gpu.py'), '--execute',
                      '--config', str(cfg_path), '--config-sha', cfg_sha]
        records[name] = {'freeze': frozen, 'config': {'path': str(cfg_path), 'sha256': cfg_sha},
                         'screen_argv': command, 'main_and_recovery_argv': supervisor}
    receipt = {'schema': 'annotation-speed-matrix-prepared/v1', 'input_spec_sha256': spec_sha,
               'variants': records, 'GPU_executed': False, 'commands_executed': False,
               'cloud_admission': False, 'automatic_variant_selection': False,
               'notes': ['先依次screen三组，再只执行选中候选及必要基线的3000/512。',
                         '不得把此文件当资源空闲或质量通过证明；每次实际启动仍走资源守卫。']}
    atomic_new(output / 'prepared.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.spec, a.spec_sha, a.output)
    print(json.dumps({'output': str(a.output), 'commands_executed': False,
                      'screen_commands': {k: shlex.join(v['screen_argv']) for k, v in result['variants'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
