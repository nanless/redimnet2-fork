"""One explicitly admitted GPU; retain its context until parent terminal ACK."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
from gpu_common import need, sha, write, process_start, verify_bundle, verify_model_stats
from gpu_challenge import worker_challenge, normalized_uuid


_BOUNDED_LEASE = None  # 持有到CUDA进程退出，异常也不提前释放旧writer锁。


def main():
    global _BOUNDED_LEASE
    p = argparse.ArgumentParser()
    for name in ('ticket', 'ticket-sha', 'bundle-sha', 'contract', 'contract-sha',
                 'shard-index', 'index-sha', 'queue-root', 'output-root', 'worker-id',
                 'worker-output', 'resource-attestation'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--bounded-runtime-spec'); p.add_argument('--bounded-runtime-sha')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    verify_bundle(here, a.bundle_sha)
    need(sha(a.ticket) == a.ticket_sha and sha(a.contract) == a.contract_sha
         and sha(a.shard_index) == a.index_sha, 'ticket/contract/index pin differs')
    t = json.loads(Path(a.ticket).read_text())
    need(t['schema'] == 'open-annotation-resource-ticket/v1', 'wrong ticket schema')
    need(os.getppid() == t['parent_pid'] and process_start(os.getppid()) == t['parent_start_time'],
         'actual parent PID/start differs')
    need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == t['boot_id'], 'boot differs')
    need(0 <= time.monotonic() - t['created_monotonic'] <= 900, 'ticket stale/future')
    need(t['worker_id'] == a.worker_id and t['bundle_sha256'] == a.bundle_sha
         and t['contract_sha256'] == a.contract_sha and t['index_sha256'] == a.index_sha,
         'ticket worker/source identity differs')
    nonce = os.environ['PILOT_NONCE']
    need(nonce == t['nonce'] and os.environ.get('CUDA_VISIBLE_DEVICES') == t['target_uuid'],
         'nonce/UUID visibility differs')
    for lock in t['locks']:
        local = Path(lock['path'])
        st = local.stat()
        held = Path(f"/proc/{t['parent_pid']}/fd/{lock['fd']}").stat()
        need((st.st_dev, st.st_ino) == (lock['device'], lock['inode'])
             == (held.st_dev, held.st_ino), 'parent device lock identity differs')
        with local.open('rb') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
                raise RuntimeError('parent UUID lock not held')
    from bounded_speed_runtime import runtime_options, RuntimeLease, guard_native_formal_route
    bounded_pin = runtime_options(a)
    need(('bounded_lineage' in t) == (bounded_pin is not None), 'bounded ticket/schema routing differs')
    if bounded_pin is not None:
        _BOUNDED_LEASE = RuntimeLease.adopt(bounded_pin, t['bounded_lineage'])
        need(_BOUNDED_LEASE.compiled['spec']['new_contract'] == {'path':a.contract,'sha256':a.contract_sha},
             'bounded ticket new producer differs')
    c = json.loads(Path(a.contract).read_text())
    guard_native_formal_route(c, bounded_pin, lease=_BOUNDED_LEASE, require_lease=True)
    verify_model_stats(t['model_facts'])
    for name, digest in c.get('source_files', {}).items():
        need(sha(name) == digest, 'inference source changed before CUDA')
    print('RESOURCE_GUARDS_PASSED_BEFORE_TORCH ' + a.worker_id, flush=True)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need(torch.cuda.device_count() == 1, 'worker must see exactly one GPU')
    prop = torch.cuda.get_device_properties(0)
    need(normalized_uuid(prop.uuid) == t['target_uuid'], 'actual CUDA UUID differs')
    torch.cuda.set_per_process_memory_fraction(t['memory_budget_mib'] * 1024 ** 2 / prop.total_memory, 0)
    out = Path(a.worker_output)
    worker_challenge(out / 'context_challenge', nonce, [t['target_uuid']], torch, t['minor_map'])
    proof = out / 'context_challenge' / 'attestation.json'
    need(proof.is_file(), 'native context attestation absent')
    att = {'schema': 'open-annotation-resource-attestation/v1', 'nonce': nonce,
           'worker_id': a.worker_id, 'UUIDs': [t['target_uuid']],
           'known_container_worker_pid': os.getpid(), 'ticket_sha': a.ticket_sha,
           'challenge_attestation_path': str(proof), 'challenge_attestation_sha': sha(proof),
           'bundle_sha256': a.bundle_sha, 'contract_sha256': a.contract_sha,
           'index_sha256': a.index_sha, 'PID_translation_claimed': False}
    write(a.resource_attestation, att)
    print('SINGLE_GPU_CONTEXT_ATTESTED_BEFORE_MODELS ' + a.worker_id, flush=True)
    from worker import main as model_main
    argv = []
    for name in ('contract', 'contract_sha', 'shard_index', 'index_sha', 'queue_root',
                 'output_root', 'worker_id', 'resource_attestation', 'ticket',
                 'ticket_sha', 'worker_output'):
        argv.extend(['--' + name.replace('_', '-'), getattr(a, name)])
    if bounded_pin is not None:
        argv += ['--bounded-runtime-spec',a.bounded_runtime_spec,'--bounded-runtime-sha',a.bounded_runtime_sha]
    summary = model_main(argv, bounded_lease=_BOUNDED_LEASE)
    summary_path = out / 'worker_summary.json'
    need(summary_path.is_file(), 'worker summary missing')
    stored = json.loads(summary_path.read_text())
    need(summary == stored and stored['worker_id'] == a.worker_id
         and stored['contract_sha256'] == a.contract_sha and stored['index_sha256'] == a.index_sha,
         'terminal summary identity differs')
    need(stored['model_API_calls'] == 0, 'formal worker used a model API')
    verify_model_stats(t['model_facts'])
    verify_bundle(here, a.bundle_sha)
    if _BOUNDED_LEASE is not None: _BOUNDED_LEASE.verify()
    ready = {'schema': 'open-annotation-terminal-ready/v1', 'nonce': nonce,
             'worker_id': a.worker_id, 'worker_pid': os.getpid(),
             'worker_start_ticks': process_start(os.getpid()), 'ticket_sha256': a.ticket_sha,
             'contract_sha256': a.contract_sha, 'index_sha256': a.index_sha,
             'bundle_sha256': a.bundle_sha, 'summary_path': str(summary_path),
             'summary_sha256': sha(summary_path), 'resource_attestation_sha256': sha(a.resource_attestation)}
    write(out / 'terminal_ready.json', ready)
    deadline = time.monotonic() + 300
    ack = out / 'terminal_ack.json'
    while not ack.is_file():
        need(time.monotonic() < deadline, 'parent terminal ACK missing; preserve context evidence')
        time.sleep(.1)
    value = json.loads(ack.read_text())
    need(value['nonce'] == nonce and value['ready_sha256'] == sha(out / 'terminal_ready.json'),
         'terminal ACK identity differs')
    print('WORKER_FINISHED_AFTER_TERMINAL_ACK ' + a.worker_id, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(2)
