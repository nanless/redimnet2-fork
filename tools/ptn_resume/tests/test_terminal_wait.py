"""终态等待的纯CPU运行层检查；不启动进程、不导入Torch、不伪造平台授权。"""
import builtins
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.ptn_resume import contracts as C

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'


def load(name, modules):
    spec = importlib.util.spec_from_file_location('terminal_wait_probe_' + name, RUNTIME / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(value)
    return value


class Clock:
    def __init__(self, update=None):
        self.elapsed = 0.
        self.update = update
    def monotonic(self):
        return self.elapsed
    def sleep(self, seconds):
        self.elapsed += seconds
        if self.update is not None:
            update, self.update = self.update, None
            update()


class TerminalWaitRuntimeTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            control = root / 'control'
            control.mkdir(mode=0o700)
            now = dt.datetime.now(dt.timezone.utc)
            details = {
                'original': ('t-20261019232824-8zqrl', C.ORIGINAL, 'Terminated'),
                'predecessor': ('t-20261024154424-vggvm', C.PREDECESSOR, 'Failed'),
            }
            def publish(key, when, **changes):
                task, run, state = details[key]
                value = {'schema': 'ptn-predecessor-terminal/v1', 'task_id': task,
                         'run_root': run, 'state': state, 'active_workers': 0,
                         'worker_states': [{'name': 'worker_0', 'state': state}],
                         'observed_utc': when.isoformat()}
                value.update(changes)
                final = control / (key + '-terminal.json')
                temporary = final.with_suffix('.new')
                with temporary.open('x') as stream:
                    json.dump(value, stream)
                temporary.chmod(0o600)
                os.replace(temporary, final)
                return {'path': str(final), 'sha256': hashlib.sha256(final.read_bytes()).hexdigest()}
            old = {key: publish(key, now - dt.timedelta(hours=2)) for key in details}
            contract = root / 'static-contract.json'
            contract.write_text(json.dumps({'terminals': old}))
            before = contract.read_bytes()
            strict = load('strict_restore', {'ptn_contracts': C, 'state_gate': types.ModuleType('state_gate')})
            strict.CONTROL = control
            strict.RUN = root / 'not-created-run'
            gate = load('predecessor_gate', {'strict_restore': strict})
            yield root, now, publish, contract, before, strict, gate
            self.assertEqual(contract.read_bytes(), before)
            self.assertFalse(strict.RUN.exists())

    @contextlib.contextmanager
    def guarded_wait(self, gate, clock):
        original = builtins.__import__
        def import_without_torch(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('terminal wait imported Torch')
            return original(name, *args, **kwargs)
        with patch.object(gate.time, 'monotonic', side_effect=clock.monotonic), \
             patch.object(gate.time, 'sleep', side_effect=clock.sleep), \
             patch.object(builtins, '__import__', side_effect=import_without_torch), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            yield output

    def test_stale_control_exhausts_bounded_wait_without_torch_or_run_creation(self):
        with self.fixture() as (_, _, _, _, _, strict, gate):
            clock = Clock()
            with self.guarded_wait(gate, clock) as output:
                with self.assertRaisesRegex(ValueError, 'wait exhausted; no CUDA launched'):
                    gate.wait_for_terminal(timeout=2)
            self.assertEqual(clock.elapsed, 2.)
            self.assertIn('terminal evidence stale/future', output.getvalue())
            self.assertFalse(strict.RUN.exists())

    def test_fresh_atomic_control_arrives_after_queue_delay_without_rewriting_contract(self):
        with self.fixture() as (_, now, publish, contract, before, strict, gate):
            refs = {}
            def update():
                refs.update({key: publish(key, now) for key in ('original', 'predecessor')})
            clock = Clock(update)
            with self.guarded_wait(gate, clock):
                result = gate.wait_for_terminal(timeout=2)
                verified = gate.verify_unchanged(result)
            self.assertEqual(clock.elapsed, 1.)
            self.assertEqual(result['refs'], refs)
            self.assertEqual(verified['predecessor']['reference'], refs['predecessor'])
            self.assertEqual(contract.read_bytes(), before)
            self.assertFalse(strict.RUN.exists())

    def test_future_wrong_task_or_boolean_active_workers_never_pass_runtime_wait(self):
        cases = ('future', 'wrong_task', 'bool_active')
        for case in cases:
            with self.subTest(case=case), self.fixture() as (_, now, publish, _, _, _, gate):
                publish('original', now)
                changes = {'task_id': 'different-task'} if case == 'wrong_task' else (
                    {'active_workers': False} if case == 'bool_active' else {})
                publish('predecessor', now + dt.timedelta(minutes=2) if case == 'future' else now, **changes)
                clock = Clock()
                with self.guarded_wait(gate, clock):
                    with self.assertRaisesRegex(ValueError, 'wait exhausted'):
                        gate.wait_for_terminal(timeout=2)

    def test_bad_or_changed_concrete_refs_remain_rejected(self):
        with self.fixture() as (_, now, publish, _, _, strict, gate):
            refs = {key: publish(key, now) for key in ('original', 'predecessor')}
            result = {'refs': refs, 'evidence': {}}
            with self.assertRaisesRegex(ValueError, 'reference bytes differ'):
                C.verify_fresh_terminals({}, evidence={**refs, 'predecessor': {
                    **refs['predecessor'], 'sha256': '0' * 64}})
            for bad in (True, {'original': refs['original']}, {'original': {}, 'predecessor': refs['predecessor']}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    C.verify_fresh_terminals({}, evidence=bad)
            publish('predecessor', now + dt.timedelta(seconds=1))
            with self.assertRaisesRegex(ValueError, 'bytes changed under admission'):
                gate.verify_unchanged(result)
            self.assertFalse(strict.RUN.exists())


if __name__ == '__main__':
    unittest.main()
