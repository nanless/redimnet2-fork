"""Explicit W2/W8 independent annotation workers with physical-device leases."""
import argparse
import contextlib
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from gpu_common import (need, sha, write, verify_bundle, model_facts, verify_model_stats,
                        open_pidfd, process_start, own_members, cleanup, DrainState, wait_attested_context_drain)
from gpu_challenge import ParentChallenge, minor_map
from gpu_vendor import resources as G

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/experiments/redimnet2/open-annotation-ultra-20261025')
GLOBAL_LOCKS = Path('/root/experiments/redimnet2/.word-gpu-device-locks')


def worker_ready(state, args):
    out = state['out']
    ready_path = out / 'terminal_ready.json'
    value = json.loads(ready_path.read_text())
    for key, expected in {
        'schema': 'open-annotation-terminal-ready/v1', 'nonce': state['nonce'],
        'worker_id': state['worker_id'], 'worker_pid': state['proc'].pid,
        'worker_start_ticks': state['started'], 'ticket_sha256': state['ticket_sha'],
        'contract_sha256': args.contract_sha, 'index_sha256': args.index_sha,
        'bundle_sha256': args.bundle_sha,
    }.items():
        need(value[key] == expected, 'terminal ready differs: ' + key)
    summary_path = out / 'worker_summary.json'
    need(value['summary_path'] == str(summary_path) and value['summary_sha256'] == sha(summary_path),
         'worker summary SHA/path differs')
    summary = json.loads(summary_path.read_text())
    need(summary['worker_id'] == state['worker_id'] and summary['contract_sha256'] == args.contract_sha
         and summary['index_sha256'] == args.index_sha and summary['model_API_calls'] == 0,
         'worker summary provenance differs')
    need(value['resource_attestation_sha256'] == sha(out / 'resource_attestation.json'),
         'resource attestation changed')
    for ref in summary.get('chunk_receipt_refs', []):
        p = Path(ref['path']).resolve()
        need(Path(args.queue_root).resolve() in p.parents and p.is_file()
             and sha(p) == ref['sha256'], 'owned chunk receipt differs')
    ready_sha = sha(ready_path)
    write(out / 'terminal_ack.json', {'nonce': state['nonce'], 'ready_sha256': ready_sha})
    return {'ready_sha256': ready_sha, 'summary_sha256': value['summary_sha256'],
            'summary': summary, 'ack_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def ensure_alias():
    target = Path('/shared/group-shared')
    alias = Path('/root/group-shared')
    need(target.is_dir(), 'shared group mount is missing')
    if alias.exists() or alias.is_symlink():
        need(alias.is_dir() and alias.samefile(target), 'existing shared alias conflicts')
    else:
        alias.symlink_to(target, target_is_directory=True)


def run(args):
    need(int(os.environ.get('WORLD_SIZE', '1')) == 1 and int(os.environ.get('RANK', '0')) == 0
         and 'LOCAL_RANK' not in os.environ, 'parent must execute once, not inside platform rank group')
    need(1 <= args.max_runtime_seconds <= 720 * 3600, 'invalid bounded run time')
    files = verify_bundle(HERE, args.bundle_sha)
    need(sha(args.contract) == args.contract_sha and sha(args.shard_index) == args.index_sha,
         'contract/index pin differs')
    contract = json.loads(Path(args.contract).read_text())
    output = Path(args.output).resolve()
    need(ROOT.resolve() in output.parents and not output.exists(), 'fresh isolated resource output required')
    for name in ('queue_root', 'output_root'):
        path = Path(getattr(args, name)).resolve()
        need(ROOT.resolve() in path.parents and path != HERE, 'unsafe queue/label output')
    ensure_alias()
    t0 = time.monotonic()
    # Pure CPU reads occur before holding GPUs. No torch import in this parent.
    facts = model_facts(contract, full_hash=True)
    model_guard_seconds = time.monotonic() - t0
    current = G.capture_snapshot()
    if args.target_uuids == 'auto8':
        need(len(current['devices']) == 8, 'cloud parent must see exactly eight physical GPUs')
        uuids = [d['uuid'] for d in sorted(current['devices'], key=lambda d: d['index'])]
    else:
        uuids = args.target_uuids.split(',')
        need(len(uuids) == 2 and len(set(uuids)) == 2, 'two exact development UUIDs required')
    selected = []
    for uuid in uuids:
        matched = [d for d in current['devices'] if d['uuid'] == uuid]
        need(len(matched) == 1 and 'A800' in matched[0]['name'] and matched[0]['total_mib'] >= 80000,
             'physical A800 UUID/capacity differs')
        selected.append(matched[0])
    indices = [d['index'] for d in selected]
    queue_root = Path(args.queue_root)
    queue_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(queue_root / '.parent.lock'), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    states = []
    result = {'schema': 'open-annotation-independent-gpu-run/v1', 'hostname': socket.gethostname(),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'bundle_sha256': args.bundle_sha, 'contract_sha256': args.contract_sha,
              'index_sha256': args.index_sha, 'worker_count': len(uuids), 'UUIDs': uuids,
              'model_guard_seconds': model_guard_seconds, 'model_API_calls': 0,
              'training_DDP_used': False, 'human_accuracy_certified': False}
    with os.fdopen(fd, 'a+b') as parent_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from work_queue import prepare_queue
        prepare_queue(args.shard_index, args.index_sha, args.queue_root, args.contract_sha, args.output_root)
        with G.DeviceLocks(GLOBAL_LOCKS, uuids) as locks:
            lease_started = time.monotonic()
            probe = G.double_probe(indices, args.memory_budget_mib, interval_seconds=5., reserve_mib=8192)
            need([d['uuid'] for d in probe['devices']] == uuids, 'device order changed')
            verify_bundle(HERE, args.bundle_sha)
            verify_model_stats(facts)
            output.mkdir(parents=True)
            write(output / 'preflight.json', result | {'probe': probe, 'bundle_members': files})
            minors = minor_map(uuids)
            previous = signal.signal(signal.SIGTERM,
                                     lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt('parent terminated')))
            try:
                for i, uuid in enumerate(uuids):
                    worker_id = 'worker-' + str(i)
                    out = output / worker_id
                    out.mkdir()
                    nonce = secrets.token_hex(24)
                    ticket = {'schema': 'open-annotation-resource-ticket/v1', 'parent_pid': os.getpid(),
                              'parent_start_time': process_start(os.getpid()),
                              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                              'created_monotonic': time.monotonic(), 'locks': locks.evidence,
                              'devices': probe['devices'], 'model_facts': facts, 'worker_id': worker_id,
                              'target_uuid': uuid, 'minor_map': minors, 'memory_budget_mib': args.memory_budget_mib,
                              'bundle_sha256': args.bundle_sha, 'contract_sha256': args.contract_sha,
                              'index_sha256': args.index_sha, 'nonce': nonce}
                    ticket_path = out / 'admission.json'
                    write(ticket_path, ticket)
                    env = os.environ.copy()
                    for key in ['PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_AUDIT', 'RANK',
                                'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE']:
                        env.pop(key, None)
                    env.update(CUDA_VISIBLE_DEVICES=uuid, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               NUMEXPR_NUM_THREADS='2', PILOT_NONCE=nonce)
                    command = [sys.executable, '-B', str(HERE / 'gpu_bootstrap.py')]
                    for key, value in [('ticket', ticket_path), ('ticket-sha', sha(ticket_path)),
                                       ('bundle-sha', args.bundle_sha), ('contract', args.contract),
                                       ('contract-sha', args.contract_sha), ('shard-index', args.shard_index),
                                       ('index-sha', args.index_sha), ('queue-root', args.queue_root),
                                       ('output-root', args.output_root), ('worker-id', worker_id),
                                       ('worker-output', out), ('resource-attestation', out / 'resource_attestation.json')]:
                        command += ['--' + key, str(value)]
                    log = (out / 'worker.log').open('x')
                    pass_fds = [parent_lock.fileno()] + [e['fd'] for e in locks.evidence]
                    proc = subprocess.Popen(command, env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                            pass_fds=pass_fds, start_new_session=True)
                    state = {'worker_id': worker_id, 'out': out, 'nonce': nonce, 'proc': proc,
                             'ticket_sha': sha(ticket_path), 'log': log,
                             'started': int(time.monotonic() * os.sysconf('SC_CLK_TCK')) - 1,
                             'pidfd': None, 'phase': 'challenge', 'read_offset': 0, 'uuid': uuid}
                    states.append(state)
                    state['started'] = process_start(proc.pid)
                    state['pidfd'] = open_pidfd(proc.pid)
                    need(os.getpgid(proc.pid) == os.getsid(proc.pid) == proc.pid, 'worker session not isolated')
                    write(out / 'worker_identity.json', {k: state[k] for k in ('worker_id', 'nonce', 'started', 'uuid')}
                          | {'container_pid': proc.pid, 'PGID': proc.pid, 'SID': proc.pid,
                             'pidfd_inode': os.fstat(state['pidfd']).st_ino})
                    state['challenge'] = ParentChallenge(out / 'context_challenge', nonce, [uuid], proc.pid)
                started = time.monotonic()
                while True:
                    need(time.monotonic() - started < args.max_runtime_seconds, 'bounded GPU phase timed out')
                    snapshot = G.capture_snapshot()
                    need(G.stable_identity(G.selected_devices(snapshot, indices)) == G.stable_identity(probe['devices']),
                         'GPU physical identity/driver changed')
                    with (output / 'GPU_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + '\n')
                    exited = 0
                    for state in states:
                        proc = state['proc']
                        rc = proc.poll()
                        if state['phase'] in ('draining', 'exited'):
                            state['drain'].observe(snapshot)
                            need(time.monotonic() - state['drain_started'] < 30
                                 or state['uuid'] in state['drain'].disappeared,
                                 'CUDA context did not disappear after worker drain')
                        if rc is not None:
                            need(state['phase'] in ('draining', 'exited') and rc == 0,
                                 'worker exited before ACK or failed: ' + state['worker_id'])
                            need(not own_members(proc, state['started'], state['nonce']), 'owned residual worker remains')
                            state['phase'] = 'exited'
                            exited += 1
                            continue
                        need(process_start(proc.pid) == state['started']
                             and os.getpgid(proc.pid) == os.getsid(proc.pid) == proc.pid, 'worker identity drift')
                        challenge = state['challenge']
                        if state['phase'] == 'challenge':
                            challenge.advance(snapshot)
                            if challenge.complete:
                                state['phase'] = 'models'
                        elif state['phase'] == 'models':
                            challenge.monitor(snapshot)
                            row = [x for x in snapshot['compute_processes'] if x['gpu_uuid'] == state['uuid']]
                            need(len(row) == 1 and type(row[0]['used_mib']) is int
                                 and row[0]['used_mib'] <= args.memory_budget_mib + 8192, 'worker GPU budget/identity differs')
                            if (state['out'] / 'terminal_ready.json').is_file():
                                state['terminal'] = worker_ready(state, args)
                                state['phase'] = 'draining'
                                state['drain_started'] = time.monotonic()
                                state['drain'] = DrainState([state['uuid']], challenge.opaque_pid)
                        elif state['phase'] == 'draining':
                            need(time.monotonic() - state['drain_started'] < 30, 'worker terminal drain timed out')
                        with (state['out'] / 'worker.log').open() as stream:
                            stream.seek(state['read_offset'])
                            delta = stream.read()
                            state['read_offset'] = stream.tell()
                        if delta:
                            print('[' + state['worker_id'] + '] ' + delta, end='', flush=True)
                    if exited == len(states):
                        break
                    # Utilization is descriptive telemetry, never an idle test.
                    # Keep extra queries outside the strict allocation challenge.
                    if all(s['phase'] != 'challenge' for s in states):
                        cmd = ['nvidia-smi', '--query-gpu=uuid,utilization.gpu', '--format=csv,noheader,nounits']
                        try:
                            usage = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
                            values = {row[0].strip(): row[1].strip() for row in csv.reader(usage.stdout.splitlines())
                                      if len(row) == 2}
                            telemetry = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                         'monotonic': time.monotonic(), 'command': cmd, 'returncode': usage.returncode,
                                         'raw_stdout': usage.stdout, 'raw_stderr': usage.stderr,
                                         'utilization_percent': {u: int(values[u]) if values.get(u, '').isdigit() else None for u in uuids}}
                        except Exception:
                            telemetry = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                         'error': traceback.format_exc(), 'utilization_percent': {u: None for u in uuids}}
                        with (output / 'GPU_utilization.jsonl').open('a') as stream:
                            stream.write(json.dumps(telemetry, ensure_ascii=False, allow_nan=False) + '\n')
                    time.sleep(1)
                verify_bundle(HERE, args.bundle_sha)
                verify_model_stats(facts)
                def retain_terminal_sample(snapshot):
                    with (output / 'terminal_drain_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + '\n')
                drain = wait_attested_context_drain([s['drain'] for s in states], G.capture_snapshot,
                                                     on_snapshot=retain_terminal_sample)
                write(output / 'terminal_drain_complete.json', drain)
                post = G.double_probe(indices, args.memory_budget_mib, interval_seconds=5., reserve_mib=8192)
                need(G.stable_identity(post['devices']) == G.stable_identity(probe['devices']),
                     'physical device identity changed before resource return')
                write(output / 'resource_return_double_probe.json', post)
                result['workers'] = [{'worker_id': s['worker_id'], 'uuid': s['uuid'], 'terminal': s['terminal']}
                                     for s in states]
                result['resource_returned'] = True
                result['gpu_lease_wall_seconds'] = time.monotonic() - lease_started
            except BaseException:
                write(output / 'failure.json', {'traceback': traceback.format_exc(), 'own_worker_pids': [s['proc'].pid for s in states]})
                raise
            finally:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                for state in states:
                    cleanup(state['proc'], state['started'], state['nonce'])
                    if state['pidfd'] is not None:
                        os.close(state['pidfd'])
                    state['log'].close()
                signal.signal(signal.SIGTERM, previous)
        # CUDA has been drained and physical GPU leases returned before CPU verification.
        from report import verify_run
        verification = verify_run(args.contract, args.shard_index, args.queue_root, args.output_root,
                                  full=args.full_verify)
        result.update(verification=verification, full_sidecar_readback=args.full_verify,
                      parent_elapsed_seconds=time.monotonic() - t0,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        need(verification.get('passed', verification.get('pass', False)), 'output verification failed')
        write(output / 'parent_result.json', result)
        print(json.dumps({'parent_finished': True, 'output': str(output), 'worker_count': len(states),
                          'verification': verification}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true', required=True)
    for name in ('bundle-sha', 'contract', 'contract-sha', 'shard-index', 'index-sha',
                 'queue-root', 'output-root', 'output', 'target-uuids'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--memory-budget-mib', type=int, default=60000)
    p.add_argument('--max-runtime-seconds', type=int, default=7200)
    p.add_argument('--full-verify', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(2)
