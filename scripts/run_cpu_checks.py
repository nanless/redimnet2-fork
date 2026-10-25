#!/usr/bin/env python3
"""分进程运行本仓库的CPU回归检查，保留每组原始输出和退出码。

需要安装requirements.txt中的对应CPU依赖。没有依赖、用例跳过、预期失败、超时或失败都
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
    ('PTN_numeric_diagnostics', ['-m', 'unittest', 'discover', '-s', 'tools/ptn_diagnostics/tests', '-p', 'test_*.py']),
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


def own_group_exists(process):
    """只查询Popen创建时已核实的新session/进程组，不枚举系统进程。"""
    group = process.pid
    proof = getattr(process, '_cpu_check_group_owner', None)
    if group <= 1 or group == os.getpgrp() or proof != {'session': group, 'process_group': group}:
        raise ValueError('缺少本次新session/进程组所有权证明')
    if process.poll() is None:
        try:
            if os.getsid(group) != group or os.getpgid(group) != group:
                raise RuntimeError('本次父进程session/进程组身份改变')
        except ProcessLookupError:
            pass
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_own_session(process, grace_seconds=5):
    """父进程退出后仍收束原session中的子进程；绝不按程序名查杀。"""
    actions = []; probe_errors = []
    def result(complete, present, error=None):
        value = {'complete': complete, 'actions': actions, 'remaining_group_present': present,
                 'process_group': process.pid, 'probe_errors': probe_errors}
        if error is not None: value['error'] = error
        return value
    def observe(phase):
        try:
            return own_group_exists(process)
        except PermissionError as exc:
            # Darwin在成功KILL后的组销毁期间可能短暂返回EPERM。它是未知，
            # 不是组消失；只在原宽限内重读，直到新鲜ESRCH才能确认结束。
            probe_errors.append({'phase': phase, 'operation': 'identity_and_signal0_probe',
                                 'signal': 0, 'process_group': process.pid,
                                 'monotonic': time.monotonic(), 'errno': exc.errno, 'error': str(exc)})
            return None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + grace_seconds
        phase = 'before_' + signal.Signals(sig).name
        state = observe(phase)
        while state is None and time.monotonic() < deadline:
            time.sleep(.05)
            state = observe(phase)
        if state is None:
            return result(False, None, '权限探测在宽限内未恢复，组状态未知')
        if state is False:
            process.poll()
            return result(True, False)
        action = {'signal': int(sig), 'process_group': process.pid,
                  'phase': 'send_' + signal.Signals(sig).name,
                  'ownership': 'Popen start_new_session identity verified at creation'}
        actions.append(action)
        try:
            os.killpg(process.pid, sig)
            action['result'] = 'sent'
        except ProcessLookupError:
            action['result'] = 'already_absent_before_signal'
        except PermissionError as exc:
            action.update(result='denied', errno=exc.errno, error=str(exc))
            return result(False, None, '发送终止信号被拒绝，不能报告清理成功')
        while time.monotonic() < deadline:
            process.poll()
            state = observe('after_' + signal.Signals(sig).name)
            if state is False:
                return result(True, False)
            time.sleep(.1)
    return result(False, state, '本次进程组在有限清理期限后仍存在或状态未知')


def execute(name, arguments, directory, timeout, environment, require_unittest=False):
    """超时时只收束本次新建的进程组，不按名字查杀任何其他进程。"""
    stdout = directory / (name + '.stdout.txt')
    stderr = directory / (name + '.stderr.txt')
    start = time.monotonic()
    timed_out = False
    cleanup = None
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.Popen([sys.executable, *arguments], cwd=ROOT,
                                   env=environment, stdout=out, stderr=err,
                                   start_new_session=True)
        try:
            process._cpu_check_group_owner = {'session': os.getsid(process.pid),
                                               'process_group': os.getpgid(process.pid)}
        except ProcessLookupError:
            process._cpu_check_group_owner = None
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                cleanup = terminate_own_session(process)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                cleanup = {'complete': False, 'error': str(exc)}
            code = process.poll()
    # unittest默认结果写stderr。普通stdout可能恰含“Ran ...”或“skipped=0”，
    # 不能让它覆盖真实跳过/计数。仅解析独立成行的标准结果摘要。
    text = stderr.read_bytes().decode('utf-8', errors='replace')
    counts = re.findall(r'^Ran (\d+) tests? in [0-9.]+s\s*$', text, re.MULTILINE)
    statuses = re.findall(r'^(OK(?: \([^\n]*\))?|FAILED \([^\n]*\))\s*$', text, re.MULTILINE)
    count = sum(int(value) for value in counts) if counts else None
    skips = sum(int(value) for status in statuses for value in re.findall(r'\bskipped=(\d+)\b', status))
    expected_failures = sum(int(value) for status in statuses
                            for value in re.findall(r'\bexpected failures=(\d+)\b', status))
    summary_valid = bool(counts and statuses and count > 0 and len(counts) == len(statuses))
    suites_ok = summary_valid and expected_failures == 0 and all(status.startswith('OK') for status in statuses)
    if count is not None or require_unittest:
        accepted = suites_ok
    else:
        accepted = True  # 非unittest的VAD断言、历史CSV及静态检查以真实退出码为准。
    return {'name': name, 'arguments': arguments, 'exit_code': code,
            'timeout': timed_out, 'elapsed_seconds': time.monotonic()-start,
            'timeout_cleanup': cleanup,
            'reported_unittest_count': count,
            'unittest_summary_valid': summary_valid,
            'reported_skipped': skips, 'passed_without_skip': code == 0 and not timed_out and skips == 0 and accepted,
            'reported_expected_failures': expected_failures,
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
                          'skipped': result['reported_skipped'],
                          'expected_failures': result['reported_expected_failures']}, ensure_ascii=False), flush=True)
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
