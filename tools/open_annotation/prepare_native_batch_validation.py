"""准备同一有限三训练域的原生转写批次1/4/8对照，不执行GPU。

与旧单窗提速协议分开；转写固定FP32/SDPA、Ultra保持单窗FP32/eager。
每轨独立新合同/队列/整批文件缓存，不修改既有生产或创建数据库。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import atomic_new, need
from prepare_speed_validation import pinned, check_index, DOMAINS
import freeze_contract as freezer

SCHEMA = 'annotation-native-batch-preparation/v1'
VARIANTS = {'native-fp32-sdpa-b' + str(n): n for n in (1, 4, 8)}
DEFAULT_PROFILES = list(VARIANTS)
COMPUTE = {'dtype': 'float32', 'attention': 'sdpa'}
MAX_SOURCES = 4096
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def physical(path):
    """不创建输入目录；所有祖先和末端均拒绝软链。"""
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and all(not q.is_symlink() for q in (p, *p.parents)),
         'physical absolute path required: ' + str(path))
    return p


def disjoint(a, b):
    return a != b and a not in b.parents and b not in a.parents


def prepare(spec_path, spec_sha, output):
    _, spec = pinned({'path': str(spec_path), 'sha256': spec_sha})
    need(spec.get('schema') == SCHEMA, 'wrong bounded preparation schema')
    count = spec.get('source_count')
    need(type(count) is int and 3 <= count <= MAX_SOURCES, 'finite source count must be 3..4096')
    _, base = pinned(spec['base_contract'])
    need('asr_batching' not in base['runtime'], 'native comparison requires an explicit nonbatch parent')
    need(base.get('purpose') in {'pilot_validation', 'silver_annotation'}, 'real offline parent contract required')
    need(base.get('model_API_calls_allowed') == 0 and base.get('release_training_labels') is False,
         'model API or automatic label release refused')
    preprocessing = spec.get('asr_preprocessing')
    preprocessing_sources = spec.get('asr_preprocessing_sources')
    if preprocessing is not None:
        need(preprocessing == {'policy': 'official_b1_features_then_pad/v1'},
             'explicit supported per-source official B1 feature policy required')
        need(isinstance(preprocessing_sources, dict),
             'explicit official B1 preprocessing source pins required')
    else:
        need(preprocessing_sources is None, 'preprocessing source pins require explicit feature policy')
    # 先核片数及声明数量，避免标错总数的小索引偷偷列出全库分片。
    index_path, finite_index = pinned(spec['index'])
    shards = finite_index.get('shards')
    need(isinstance(shards, list) and 1 <= len(shards) <= count and
         all(isinstance(x, dict) and type(x.get('count')) is int and 1 <= x['count'] <= count for x in shards) and
         sum(x['count'] for x in shards) == count, 'finite shard metadata exceeds selected denominator')
    # 只重新打开这份有限索引及其有限分片；没有全量清单或旧侧车遍历。
    sources = check_index(spec['index'], count)
    targets = spec.get('target_uuids')
    if targets == 'auto2':
        gpu_arg, card_count = targets, 2
    else:
        need(isinstance(targets, list) and all(isinstance(x, str) for x in targets) and
             len(targets) == len(set(targets)) == 2 and all(UUID.fullmatch(x) for x in targets),
             'two explicit UUIDs or auto2 independent pilot required')
        gpu_arg, card_count = ','.join(targets), 2
    profiles = spec.get('profiles', list(DEFAULT_PROFILES))
    need(isinstance(profiles, list) and all(isinstance(x, str) for x in profiles) and
         len(profiles) == len(set(profiles)) and
         profiles == DEFAULT_PROFILES,
         'exact native batch1 baseline then batch4 and batch8 required')
    seconds = spec.get('max_runtime_seconds', 1800)
    need(type(seconds) is int and 60 <= seconds <= 7200, 'bounded runtime must be 60..7200 seconds')
    python = physical(spec['python'])
    output = physical(output)
    code_root = Path(__file__).resolve().parent
    # 仓库入口与部署副本的目录深度不同。只有同时存在仓库说明和模型源码
    # 的实际祖先才是仓库；部署时不能把任意两级祖先误当成受保护仓库。
    repo = next((p for p in (code_root, *code_root.parents)
                 if (p / 'AGENTS.md').is_file() and (p / 'redimnet2').is_dir()), None)
    source_root = repo if repo is not None else code_root
    need(not output.exists() and disjoint(output, source_root),
         'fresh output outside repository or deployed code required')
    named_roots = spec.get('protected_roots')
    need(isinstance(named_roots, list) and 1 <= len(named_roots) <= 16,
         'explicit current production/PTN protected roots required')
    protected = [physical(spec['base_contract']['path']).parent, physical(spec['index']['path']).parent]
    protected.extend(physical(q) for q in named_roots)
    if protected[0].name == 'contracts':
        protected.append(protected[0].parent)
    need(all(disjoint(output, q) for q in protected), 'output overlaps historical code or finite inputs')
    locks = physical(spec['device_lock_root'])
    need(disjoint(output, locks), 'device locks must be outside task tree')
    # freeze沿相同字节复制当前维护源码；所有轨采用同一新源钟/文字规则。
    variants = {}
    for name in profiles:
        root = output / name
        extra = {'asr_preprocessing': preprocessing, 'asr_preprocessing_sources': preprocessing_sources} if preprocessing is not None else {}
        frozen = freezer.freeze(spec['base_contract']['path'], spec['base_contract']['sha256'],
                                root / 'code', 'zhounan-speed-' + name, asr_compute=COMPUTE,
                                asr_batching={'policy':'fixed_cohort_until_durable/v1', 'max_requests':VARIANTS[name],
                                              'ordering':'frames_then_row_id_within_source_group/v1'}, **extra)
        contract = frozen['contracts']['pilot_validation']
        run = root / 'run'
        argv = [str(python), '-B', str(root / 'code/gpu_launch.py'), '--execute',
                '--bundle-sha', frozen['bundle_sha256'], '--contract', contract['path'],
                '--contract-sha', contract['sha256'], '--shard-index', spec['index']['path'],
                '--index-sha', spec['index']['sha256'], '--queue-root', str(run / 'queue'),
                '--output-root', str(run / 'labels'), '--output', str(run / 'resources-attempt1'),
                '--target-uuids', gpu_arg, '--run-root', str(root),
                '--device-lock-root', str(locks), '--max-runtime-seconds', str(seconds), '--full-verify']
        track = {'schema': 'redimnet2-annotation-native-batch-track/v1',
                 'execution_policy': 'asr_fixed_cohort_fa_single/v1', 'contract': contract,
                 'index': spec['index'], 'queue_root': str(run / 'queue'),
                 'output_root': str(run / 'labels'), 'resources_root': str(run / 'resources-attempt1')}
        track_path = root / 'native-track.json'
        track_sha = atomic_new(track_path, track)
        variants[name] = {'freeze': frozen, 'track': {'path': str(track_path), 'sha256': track_sha},
                          'argv': argv, 'shell_command': shlex.join(argv)}
    comparisons = {}
    baseline_name = profiles[0]
    baseline = variants[baseline_name]
    for name, candidate in variants.items():
        if name == baseline_name:
            continue
        report = output / ('compare-' + baseline_name + '-vs-' + name + '.json')
        argv = [str(python), '-B', str(Path(baseline['freeze']['frozen_copy']) / 'audit_native_transformers_batch.py'),
                '--baseline-spec', baseline['track']['path'], '--baseline-spec-sha', baseline['track']['sha256'],
                '--candidate-spec', candidate['track']['path'], '--candidate-spec-sha', candidate['track']['sha256'],
                '--mode', 'speed', '--output', str(report)]
        comparisons[name] = {'argv': argv, 'shell_command': shlex.join(argv), 'output': str(report)}
    receipt = {'schema': 'annotation-native-batch-prepared/v1', 'spec_sha256': spec_sha,
               'source_count': len(sources), 'domains': sorted({s['dataset'] for s in sources.values()}),
               'index': spec['index'], 'worker_count': card_count, 'variants': variants, 'comparisons': comparisons,
               'GPU_executed': False, 'commands_executed': False, 'database_created': False,
               'existing_sidecars_reopened': 0, 'cloud_admission': False, 'automatic_variant_selection': False,
               'comparison_baseline': 'explicit native FP32/SDPA batch1 with same maintained code as batch4/8; separate from old single-window GPU evidence',
               'execution_order': profiles, 'must_run_sequentially_on_idle_exclusive_devices': True,
               'notes': ['先用30—数百条筛选，选中候选再用同接口做代表性完整流程。',
                         '当前生产仍占卡时不得执行；不能并行启动三轨或在旧父守卫旁构造模型。',
                         '每轨使用独立queue/labels/cache，真实冷速不能由缓存命中代替。',
                         '只允许显式ASR固定批宽变化；Ultra仍单窗，不改变数值精度或质量规则。',
                         '较快或机器几何通过不证明人工完整字准确，交接资格由独立审阅决定。']}
    if preprocessing is not None:
        receipt['asr_preprocessing'] = preprocessing
        receipt['asr_preprocessing_sources'] = preprocessing_sources
        receipt['notes'].append('各来源先沿官方B1提声学特征，再只在特征级补齐；须核实际有效mel、mask和去左padding提示词，PCM相同不够。')
    atomic_new(output / 'prepared.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.spec, a.spec_sha, a.output)
    print(json.dumps({'output': str(a.output), 'sources': result['source_count'],
                      'commands_executed': False, 'variants': {k: v['shell_command'] for k, v in result['variants'].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
