"""CPU protocol tests. Synthetic NVML events never count as GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_common import Refused, DrainState, write, sha, verify_bundle
import gpu_common
from gpu_challenge import ParentChallenge, atomic
from gpu_launch import worker_ready
from gpu_vendor import resources as G

UUID = 'GPU-00000000-0000-4000-8000-000000000001'


class ExecutionProtocolTests(unittest.TestCase):
    def test_control_json_becomes_visible_only_after_complete_write_and_never_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / 'ready.json'
            link = os.link
            observed = []
            def publish(tmp, destination, **kwargs):
                self.assertFalse(final.exists())
                self.assertEqual(json.loads(Path(tmp).read_text()), {'nonce': 'test', 'ids': [1, 2]})
                observed.append(True)
                return link(tmp, destination, **kwargs)
            with patch.object(gpu_common.os, 'link', side_effect=publish):
                write(final, {'nonce': 'test', 'ids': [1, 2]})
            self.assertEqual(observed, [True])
            old = final.read_bytes()
            with self.assertRaises(FileExistsError):
                write(final, {'nonce': 'different'})
            self.assertEqual(final.read_bytes(), old)
            self.assertTrue(list(Path(directory).glob('*.tmp')))

    def test_single_gpu_three_random_rounds_all_seven_fresh_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            c = ParentChallenge(Path(directory) / 'challenge', 'nonce', [UUID], os.getpid())
            self.assertEqual(len(c.todo), 7)
            sizes = []
            while not c.complete:
                req = dict(c.current)
                value = {'schema': req['schema'], 'seq': req['seq'], 'nonce': 'nonce',
                         'worker_pid': os.getpid(), 'request': req, 'actual_cuda_uuids': [UUID],
                         'minor_fds_verified': True, 'observed_monotonic': time.monotonic()}
                atomic(c.root / f"response_{req['seq']}.json", value)
                used = 512 + (req['bytes'] // 1024 ** 2 if req['action'] == 'allocate' else 0)
                if req['action'] == 'allocate':
                    sizes.append(req['bytes'])
                for unused in range(2):
                    c.advance({'monotonic': time.monotonic(), 'compute_processes':
                               [{'gpu_uuid': UUID, 'pid': 98765, 'used_mib': used}]})
            proof = json.loads((c.root / 'attestation.json').read_text())
            self.assertEqual(len(proof['phases']), 7)
            self.assertEqual(len(sizes), 3)
            self.assertTrue(all(64 * 1024 ** 2 <= n <= 384 * 1024 ** 2
                                and n % (16 * 1024 ** 2) == 0 for n in sizes))
            self.assertFalse(proof['PID_translation_claimed'])

    def test_unknown_extra_compute_row_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            c = ParentChallenge(Path(directory) / 'c', 'n', [UUID], 10)
            with self.assertRaises(Refused):
                c.rows({'compute_processes': [{'gpu_uuid': UUID, 'pid': 1, 'used_mib': 2},
                                             {'gpu_uuid': UUID, 'pid': 2, 'used_mib': 3}]})

    def test_drain_can_shrink_but_never_reappear_or_change_identity(self):
        d = DrainState([UUID], 42)
        d.observe({'compute_processes': [{'gpu_uuid': UUID, 'pid': 42, 'used_mib': 500}]})
        d.observe({'compute_processes': []})
        with self.assertRaises(Refused):
            d.observe({'compute_processes': [{'gpu_uuid': UUID, 'pid': 42, 'used_mib': 100}]})
        with self.assertRaises(Refused):
            DrainState([UUID], 42).observe({'compute_processes':
                                           [{'gpu_uuid': UUID, 'pid': 43, 'used_mib': 500}]})

    def test_bundle_checks_relative_members_and_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'code.py').write_text('constant=1\n')
            write(root / 'bundle_files.json', {'code.py': sha(root / 'code.py')})
            expected = sha(root / 'bundle_files.json')
            self.assertEqual(len(verify_bundle(root, expected)), 1)
            (root / 'code.py').write_text('constant=2\n')
            with self.assertRaises(Refused):
                verify_bundle(root, expected)

    def test_terminal_ready_binds_complete_summary_and_only_then_acks(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summary = {'worker_id': 'worker-0', 'contract_sha256': 'c', 'index_sha256': 'i',
                       'model_API_calls': 0, 'chunk_receipt_refs': []}
            write(out / 'worker_summary.json', summary)
            write(out / 'resource_attestation.json', {'synthetic_CPU_fixture': True})
            ready = {'schema': 'open-annotation-terminal-ready/v1', 'nonce': 'n',
                     'worker_id': 'worker-0', 'worker_pid': 123, 'worker_start_ticks': 45,
                     'ticket_sha256': 't', 'contract_sha256': 'c', 'index_sha256': 'i',
                     'bundle_sha256': 'b', 'summary_path': str(out / 'worker_summary.json'),
                     'summary_sha256': sha(out / 'worker_summary.json'),
                     'resource_attestation_sha256': sha(out / 'resource_attestation.json')}
            write(out / 'terminal_ready.json', ready)
            state = {'out': out, 'worker_id': 'worker-0', 'nonce': 'n',
                     'proc': types.SimpleNamespace(pid=123), 'started': 45, 'ticket_sha': 't'}
            args = types.SimpleNamespace(contract_sha='c', index_sha='i', bundle_sha='b', queue_root=directory)
            result = worker_ready(state, args)
            ack = json.loads((out / 'terminal_ack.json').read_text())
            self.assertEqual(ack['ready_sha256'], result['ready_sha256'])

    def test_changed_summary_does_not_receive_terminal_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write(out / 'worker_summary.json', {'model_API_calls': 0})
            ready = {'schema': 'open-annotation-terminal-ready/v1', 'nonce': 'n',
                     'worker_id': 'worker-0', 'worker_pid': 123, 'worker_start_ticks': 45,
                     'ticket_sha256': 't', 'contract_sha256': 'c', 'index_sha256': 'i',
                     'bundle_sha256': 'b', 'summary_path': str(out / 'worker_summary.json'),
                     'summary_sha256': '0' * 64}
            write(out / 'terminal_ready.json', ready)
            args = types.SimpleNamespace(contract_sha='c', index_sha='i', bundle_sha='b', queue_root=directory)
            state = {'out': out, 'worker_id': 'worker-0', 'nonce': 'n',
                     'proc': types.SimpleNamespace(pid=123), 'started': 45, 'ticket_sha': 't'}
            with self.assertRaises(Refused):
                worker_ready(state, args)
            self.assertFalse((out / 'terminal_ack.json').exists())

    def terminal_drain_fixture(self, rows, timeout=30., stale=False):
        stamp = [100.]
        def clock(): return stamp[0]
        def sleep(seconds): stamp[0] += seconds
        iterator = iter(rows)
        seen = []
        def capture():
            processes = next(iterator)
            return {'monotonic': stamp[0] - (10 if stale else 0), 'compute_processes': processes}
        states = [DrainState([UUID], 42), DrainState(['second'], 43)]
        value = gpu_common.wait_attested_context_drain(states, capture, timeout=timeout,
                    interval=.5, clock=clock, sleep=sleep, on_snapshot=seen.append)
        return value, seen

    def test_terminal_wait_records_delayed_attested_teardown_before_separate_admission(self):
        row = {'gpu_uuid': UUID, 'pid': 42, 'used_mib': 13328}
        value, seen = self.terminal_drain_fixture([[row], [row], []])
        self.assertEqual(len(seen), 3)
        self.assertEqual(value['elapsed_seconds'], 1.)
        self.assertTrue(value['all_attested_contexts_absent'])
        self.assertFalse(value['idle_admission_proved'])

    def test_terminal_wait_rejects_foreign_or_reappearing_row(self):
        with self.assertRaises(Refused):
            self.terminal_drain_fixture([[{'gpu_uuid': UUID, 'pid': 99, 'used_mib': 0}]])
        other = {'gpu_uuid': 'second', 'pid': 43, 'used_mib': 1}
        row = {'gpu_uuid': UUID, 'pid': 42, 'used_mib': 1}
        with self.assertRaises(Refused):
            self.terminal_drain_fixture([[other], [other, row]])

    def test_terminal_wait_is_bounded_and_requires_fresh_snapshot(self):
        row = {'gpu_uuid': UUID, 'pid': 42, 'used_mib': 1}
        with self.assertRaises(Refused): self.terminal_drain_fixture([[row]] * 4, timeout=1.)
        with self.assertRaises(Refused): self.terminal_drain_fixture([[]], stale=True)

    def test_terminal_wait_rejects_empty_bindings_and_slow_final_capture(self):
        with self.assertRaises(Refused):
            gpu_common.wait_attested_context_drain([DrainState([], 1)], lambda: None)
        now = [100.]
        def slow_capture():
            now[0] += 40.
            return {'monotonic': now[0], 'compute_processes': []}
        with self.assertRaises(Refused):
            gpu_common.wait_attested_context_drain([DrainState([UUID], 1)], slow_capture,
                                                   clock=lambda: now[0])

    def test_zero_utilization_busy_compute_pid_still_refused(self):
        inventory = G.parse_snapshot('0, '+UUID+', NVIDIA A800, 535.129.03, 81920, 2, 81918\n'
                                    '1, GPU-00000000-0000-4000-8000-000000000002, NVIDIA A800, 535.129.03, 81920, 2, 81918\n',
                                    UUID+', 123, python, 0\n')
        with self.assertRaises(G.AdmissionRefused):
            G.check_idle(inventory, [0, 1], 60000, reserve_mib=8192)


if __name__ == '__main__':
    unittest.main()
