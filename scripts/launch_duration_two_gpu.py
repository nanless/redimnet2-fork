#!/usr/bin/env python3
"""Launch this experiment on two GPUs after a bounded, read-only resource check."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

EXPERIMENT_ROOT = Path('/root/experiments/redimnet2/duration-finetune-20261023')


def query_resources(run=subprocess.check_output):
    gpu_text = run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    fields = ['index', 'uuid', 'name', 'total_mib', 'used_mib', 'free_mib', 'utilization_percent']
    gpus = []
    for row in csv.reader(gpu_text.splitlines()):
        if len(row) != len(fields):
            raise ValueError('Unexpected nvidia-smi GPU row')
        record = dict(zip(fields, [x.strip() for x in row]))
        for key in ['index', 'total_mib', 'used_mib', 'free_mib', 'utilization_percent']:
            record[key] = int(record[key])
        gpus.append(record)
    process_text = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory,process_name', '--format=csv,noheader,nounits'], text=True)
    processes = []
    for row in csv.reader(process_text.splitlines()):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError('Unexpected nvidia-smi process row')
        processes.append(dict(zip(['gpu_uuid', 'pid', 'used_memory_mib', 'process_name'], [x.strip() for x in row])))
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'gpus': gpus, 'processes': processes}


def validate_resources(snapshot, devices, budget_mib, reserve_mib, allow_shared):
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError('Exactly two distinct GPU indices are required')
    if budget_mib <= 0 or reserve_mib < 2048:
        raise ValueError('Positive allocation budget and at least 2048 MiB reserve required')
    by_index = {x['index']: x for x in snapshot['gpus']}
    selected = []
    for index in devices:
        if index not in by_index:
            raise ValueError(f'GPU index unavailable: {index}')
        gpu = by_index[index]
        if gpu['free_mib'] < budget_mib + reserve_mib:
            raise RuntimeError(f"GPU {index}: {gpu['free_mib']} MiB free < {budget_mib} allocation + {reserve_mib} reserve")
        occupants = [p for p in snapshot['processes'] if p['gpu_uuid'] == gpu['uuid']]
        # Memory that cannot be assigned a visible PID is still occupied.
        if (occupants or gpu['used_mib'] > 1024) and not allow_shared:
            raise RuntimeError(f'GPU {index} already occupied; shared use requires explicit --allow-shared')
        selected.append(gpu)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--gpu-memory-mib', type=int, required=True)
    parser.add_argument('--reserve-mib', type=int, default=4096)
    parser.add_argument('--allow-shared', action='store_true', help='Use only after existing-task coexistence has been authorized')
    parser.add_argument('--dry-run', action='store_true', help='Record/check resources and print the launch command; no CUDA initialization')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args(argv)
    devices = [int(x) for x in args.devices.split(',')]
    out = args.output.resolve()
    if not out.is_relative_to(EXPERIMENT_ROOT.resolve()) or out == EXPERIMENT_ROOT.resolve():
        parser.error('Output must be a new subdirectory of the isolated duration experiment')
    for p in [args.config, args.dataset, args.official]:
        if not p.is_file():
            parser.error(f'Missing input: {p}')
    snapshot = query_resources()
    # Dry-run does not override occupied-GPU or free-memory checks.
    validate_resources(snapshot, devices, args.gpu_memory_mib, args.reserve_mib, args.allow_shared)
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2', str(root/'scripts/train_duration_finetune.py'), '--config', str(args.config.resolve()), '--dataset', str(args.dataset.resolve()), '--official', str(args.official.resolve()), '--output', str(out), '--device', 'cuda', '--gpu-memory-mib', str(args.gpu_memory_mib), '--cpu-threads', str(args.cpu_threads)]
    if args.resume:
        cmd += ['--resume', str(args.resume.resolve())]
    if args.stop_after is not None:
        cmd += ['--stop-after', str(args.stop_after)]
    report = {'before': snapshot, 'command': cmd, 'per_process_allocator_mib': args.gpu_memory_mib, 'additional_free_reserve_mib': args.reserve_mib, 'shared_use_explicit': args.allow_shared, 'dry_run': args.dry_run, 'limitations': 'Allocator budget does not reserve GPUs or cap compute utilization; other tasks can change free memory after this check.'}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not args.resume and out.exists():
        raise FileExistsError('Fresh training output must not already exist')
    records = EXPERIMENT_ROOT/'resource-records'
    records.mkdir(parents=True, exist_ok=True)
    evidence = records/f'{out.name}-{dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")}.json'
    evidence.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lib_dir = EXPERIMENT_ROOT/'runtime_audio/lib'
    if not (lib_dir/'libsndfile.so.1').is_file():
        raise FileNotFoundError('Independent audio decoder missing')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, devices)), DURATION_FT_GPU_AUTHORIZED='1', OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH'] = str(lib_dir) + (':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    code = None
    try:
        # Keep stdout/stderr visible in the caller; never terminate unrelated PIDs.
        code = subprocess.call(cmd, cwd=root, env=env)
        return code
    finally:
        after = {'returncode': code}
        try:
            after['resources'] = query_resources()
        except Exception as exc:
            after['query_error'] = repr(exc)
        after_path = evidence.with_name(evidence.stem+'-after.json')
        after_path.write_text(json.dumps(after, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    raise SystemExit(main())
