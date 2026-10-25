#!/usr/bin/env python3
"""分进程运行本仓库的CPU回归检查，保留每组原始输出和退出码。

需要安装requirements.txt中的对应CPU依赖。没有依赖、用例跳过、超时或失败都
不会被写成完整验收通过。GPU、官方权重质量、真实全库和人工边界验收不在
本脚本范围；双rank合成恢复用例只使用本机回环地址，不连接服务器。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    ('duration', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_duration*.py']),
    ('complete_word', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_word*.py']),
    ('numeric_sampling', ['tests/test_training_numeric_sampling.py']),
    ('LM_preparation', ['tests/test_lm_continuation.py']),
    ('PTN_reference', ['-m', 'unittest', 'discover', '-s', 'tools/ptn_reference/tests', '-p', 'test_*.py']),
    ('cold_state', ['tests/test_cold_start_state.py']),
    ('embedding_export', ['scripts/test_checkpoint_export.py']),
    ('VAD_geometry', ['tests/test_vad_segment_boundary.py']),
    ('schedule_launch', ['tests/test_schedule_and_launch_guards.py']),
    ('sample_length_mask', ['tests/test_mask_input_contract.py']),
    ('annotation', ['-m', 'unittest', 'discover', '-s', 'tools/open_annotation/tests', '-p', 'test_*.py']),
    ('independent_reviews', ['-m', 'unittest', 'discover', '-s', 'docs/reviews', '-p', 'test_*.py']),
    ('historical_metrics', ['docs/reviews/check_report_tables.py']),
]
UNITTEST_GROUPS = {name for name, _ in GROUPS} - {'VAD_geometry', 'historical_metrics'}


def execute(name, arguments, directory, timeout, environment, require_unittest=False):
    """超时时只收束本次新建的进程组，不按名字查杀任何其他进程。"""
    stdout = directory / (name + '.stdout.txt')
    stderr = directory / (name + '.stderr.txt')
    start = time.monotonic()
    timed_out = False
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.Popen([sys.executable, *arguments], cwd=ROOT,
                                   env=environment, stdout=out, stderr=err,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    # unittest默认结果写stderr。普通stdout可能恰含“Ran ...”或“skipped=0”，
    # 不能让它覆盖真实跳过/计数。仅解析独立成行的标准结果摘要。
    text = stderr.read_bytes().decode('utf-8', errors='replace')
    counts = re.findall(r'^Ran (\d+) tests? in [0-9.]+s\s*$', text, re.MULTILINE)
    statuses = re.findall(r'^(OK(?: \([^\n]*\))?|FAILED \([^\n]*\))\s*$', text, re.MULTILINE)
    count = sum(int(value) for value in counts) if counts else None
    skips = sum(int(value) for status in statuses for value in re.findall(r'\bskipped=(\d+)\b', status))
    summary_valid = bool(counts and statuses and count > 0 and len(counts) == len(statuses))
    suites_ok = summary_valid and all(status.startswith('OK') for status in statuses)
    if count is not None or require_unittest:
        accepted = suites_ok
    else:
        accepted = True  # 非unittest的VAD断言、历史CSV及静态检查以真实退出码为准。
    return {'name': name, 'arguments': arguments, 'exit_code': code,
            'timeout': timed_out, 'elapsed_seconds': time.monotonic()-start,
            'reported_unittest_count': count,
            'unittest_summary_valid': summary_valid,
            'reported_skipped': skips, 'passed_without_skip': code == 0 and not timed_out and skips == 0 and accepted,
            'stdout': stdout.name, 'stderr': stderr.name,
            'stdout_sha256': hashlib.sha256(stdout.read_bytes()).hexdigest(),
            'stderr_sha256': hashlib.sha256(stderr.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='本次不存在的新结果目录，建议.audit-local下')
    parser.add_argument('--timeout-per-group', type=int, default=300)
    args = parser.parse_args(argv)
    if args.timeout_per_group <= 0:
        parser.error('每组超时必须为正整数秒')
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'WANDB_MODE': 'offline', 'TOKENIZERS_PARALLELISM': 'false',
                        'GLOO_SOCKET_IFNAME': 'lo0' if sys.platform == 'darwin' else 'lo'})
    # 直接执行tests中的文件时，Python只把tests目录放在sys.path[0]；
    # 明确指定本仓库，防止找不到维护模块或误导入其他已安装的同名包。
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT/'applications/speaker_rec')])
    results = []
    for name, command in GROUPS:
        result = execute(name, command, args.output, args.timeout_per_group, environment,
                         require_unittest=name in UNITTEST_GROUPS)
        results.append(result)
        print(json.dumps({'group': name, 'exit_code': result['exit_code'],
                          'tests': result['reported_unittest_count'],
                          'skipped': result['reported_skipped']}, ensure_ascii=False), flush=True)
    result = execute('repository_static', ['scripts/check_repository.py', '--output', str(args.output.resolve()/'static.json')],
                     args.output, args.timeout_per_group, environment)
    results.append(result)
    summary = {'schema': 'repository-CPU-check-run/v1',
               'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'groups': results, 'all_selected_checks_passed_without_skip': all(x['passed_without_skip'] for x in results),
               'reported_unittest_total': sum(x['reported_unittest_count'] or 0 for x in results),
               'scope': 'portable CPU synthetic/code/document checks; not GPU/model-quality/full-corpus/human-boundary acceptance'}
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': summary['all_selected_checks_passed_without_skip'],
                      'reported_unittests': summary['reported_unittest_total']}, ensure_ascii=False))
    return 0 if summary['all_selected_checks_passed_without_skip'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
