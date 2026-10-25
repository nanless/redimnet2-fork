"""用真实调度类和实际入口保护段，检查末步学习率及空环境覆盖拒绝。

只需Python标准库与Bash，不启动模型、训练、服务器或GPU。合成优化器只提供
调度器公开需要的param_groups接口；没有替换被测调度器和启动保护逻辑。
"""
import importlib.util
from pathlib import Path
import math
import os
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('audited_schedulers', ROOT / 'applications/speaker_rec/wespeaker_lite/utils/schedulers.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class Optimizer:
    def __init__(self):
        self.param_groups = [{'lr': 0.0}, {'lr': 0.0}]


class SchedulerTests(unittest.TestCase):
    def make(self, mode='legacy', staircase=False, opt_in=True):
        return S.ExponentialDecayPlateauWithWarmup(
            Optimizer(), num_epochs=10, epoch_iter=4, initial_lr=.06,
            final_lr=.00006, warm_up_epoch=2, plateau_epochs=1,
            decay_epochs=2, scale_ratio=12, staircase=staircase,
            decay_to_final_lr=opt_in, lr_scaling=mode)

    def test_opt_in_final_step_reaches_declared_lr(self):
        for mode in ('legacy', 'base_lr'):
            for staircase in (False, True):
                with self.subTest(mode=mode, staircase=staircase):
                    self.assertAlmostEqual(self.make(mode, staircase).step_return_lr(39), .00006, places=14)

    def test_legacy_default_formula_is_preserved(self):
        for staircase in (False, True):
            schedule = self.make(staircase=staircase, opt_in=False)
            for step in (0, 7, 8, 11, 12, 20, 39):
                coeff = 12 if step >= 8 else 1 + 11 * step / 8
                offset = step - 12
                expected = .06 if step < 12 else .06 * (.5 ** (offset // 8) if staircase else math.exp(offset / 8 * math.log(.5)))
                self.assertAlmostEqual(schedule.step_return_lr(step), expected * coeff, places=14)

    def test_both_optimizer_groups_receive_same_actual_lr(self):
        schedule = self.make()
        schedule.step(39)
        rates = [x['lr'] for x in schedule.optimizer.param_groups]
        self.assertEqual(rates[0], rates[1])
        self.assertAlmostEqual(rates[0], .00006, places=14)


class EnvironmentGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'applications/speaker_rec/scripts/cloud/run_b6_mix_reproduction.sh').read_text()
        cls.guard = source[source.index('for key in NUM_EPOCHS '):source.index('\nexport REPO_ROOT')]
        cls.keys = cls.guard.splitlines()[0].removeprefix('for key in ').removesuffix('; do').split()

    def run_guard(self, extra):
        env = {k: v for k, v in os.environ.items() if k not in self.keys}
        env.update(extra)
        return subprocess.run(['bash', '-c', 'set -euo pipefail\n' + self.guard], env=env, text=True, capture_output=True, timeout=10)

    def test_every_forbidden_override_rejects_even_when_empty(self):
        for key in self.keys:
            for value in ('', '1'):
                with self.subTest(key=key, value=value):
                    run = self.run_guard({key: value})
                    self.assertEqual(run.returncode, 2, run.stderr)
                    self.assertIn(key, run.stderr)

    def test_absent_overrides_pass_the_guard(self):
        run = self.run_guard({})
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
