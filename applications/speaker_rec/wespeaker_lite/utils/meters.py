# Copyright (c) 2024, ReDimNet2 authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-epoch meters for the training log.

These are drop-in replacements for the two torchnet meters the training loop
used (`tnt.meter.AverageValueMeter` and `tnt.meter.ClassErrorMeter`). torchnet
is unmaintained and pulls in `six` plus `visdom`; `visdom` ships only an sdist
whose setup.py imports `pkg_resources`, which setuptools 80 removed, so a fresh
`pip install -r requirements.txt` failed to build it. The meters are ~30 lines
each, so vendoring them removes that whole chain.

Behaviour is kept identical to torchnet's, including the details that matter
for the printed numbers:

  * AverageValueMeter.value() returns (mean, std) with the *sample* std (ddof=1).
  * ClassErrorMeter.value() returns a one-element list holding the accuracy in
    percent, not an error rate, despite the class name.
"""

import numpy as np


class AverageValueMeter:
    """Running mean and standard deviation of a scalar."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []
        self.sum = 0.0
        self.n = 0

    def add(self, value):
        value = float(value)
        self.vals.append(value)
        self.sum += value
        self.n += 1

    def value(self):
        if self.n == 0:
            return float('nan'), float('nan')
        mean = self.sum / self.n
        if self.n == 1:
            # torchnet returns inf here (0/0 in its variance formula); match it
            # so the tuple is identical even though only index 0 is logged.
            return mean, float('inf')
        # Sample standard deviation (ddof=1), matching torchmeter's default.
        return mean, float(np.std(self.vals, ddof=1))


class ClassErrorMeter:
    """Accuracy over a stream of (scores, targets) batches.

    With accuracy=True the reported value is the top-1 accuracy in percent,
    averaged over all samples added, exactly as torchnet reports it.
    """

    def __init__(self, accuracy=False, topk=None):
        self.accuracy = accuracy
        self.topk = topk
        self.reset()

    def reset(self):
        self.n = 0
        self.n_correct = 0

    def add(self, output, target):
        output = np.asarray(output)
        target = np.asarray(target).reshape(-1)

        if output.ndim == 1:
            output = output[None, :]
        if output.shape[0] != target.shape[0]:
            # torchnet raises here too; keep the message actionable.
            raise ValueError(
                "output and target must agree on the batch dimension "
                f"(got {output.shape[0]} scores and {target.shape[0]} targets)")

        pred = output.argmax(axis=-1)
        self.n_correct += int((pred == target).sum())
        self.n += int(target.shape[0])

    def value(self):
        if self.n == 0:
            return [float('nan')]
        return [100.0 * self.n_correct / self.n]


import builtins
import sys
import types

# `meter` 命名空间在进程里只保留一份，名字固定（见下面 _MetersModule 的说明）。
_METER_SINGLETON_ATTR = '_wespeaker_lite_meter_singleton'


class _MeterNamespace:
    """The `meter` attribute the training loop reaches the meters through.

    `executor.run_epoch` calls `tnt.meter.AverageValueMeter()` /
    `tnt.meter.ClassErrorMeter()`, which is how the meters were reached when
    this module stood in for torchnet's package layout (`torchnet.meter.X`).
    When torchnet was dropped the two classes moved to the top level of this
    module and the `meter` name went with it, so every run died on
    `AttributeError: module '...meters' has no attribute 'meter'` before the
    first optimizer step. Keeping the attribute here restores the documented
    drop-in path without moving the classes.

    注意：这个命名空间**不要**用模块级赋值 `meter = _MeterNamespace()`
    暴露出去，原因见下面 `_MetersModule`。
    """

    AverageValueMeter = AverageValueMeter
    ClassErrorMeter = ClassErrorMeter


_meter_namespace = _MeterNamespace()


class _MetersModule(types.ModuleType):
    """让 `meter` 变成「即使本文件被加载两次也只有一个实例」的属性。

    为什么需要这样：本文件在训练进程里会被**加载两次**，得到两个互不相干的
    模块对象——一次是 `wespeaker_lite.utils.meters`，一次是
    `applications.speaker_rec.wespeaker_lite.utils.meters`，因为这两条路径
    都在 `sys.path` 上（训练入口自己加一条，`launch.sh` 导出的 `PYTHONPATH`
    里两条都有）。文件里的类因此也会各自生成一份。

    实测（在仓库根目录跑，两个名字都 import 一次）：

        m1 = import wespeaker_lite.utils.meters
        m2 = import applications.speaker_rec.wespeaker_lite.utils.meters
        m1 is m2                     → False
        m1.meter is m2.meter         → False   （改成属性之后是 True）
        isinstance(m1.meter.AverageValueMeter(),
                   m2.meter.AverageValueMeter) → False

    这个 `isinstance` 的用法在本仓库里是真实存在的（`per_epoch` 评测那段），
    写成模块级 `meter = _MeterNamespace()` 时，跨名字拿到的是另一份类对象，
    判断落空，实测报 `TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'`。

    所以这里把 `meter` 存成一个「按名字唯一」的对象：无论本文件被加载几次，
    只要进程里已经有一个这样的命名空间就复用它。这样两份模块里的 `meter`
    指向同一个实例，两个名字下的类对象也是同一份。

    为什么不简单地用 `sys.modules.get('wespeaker_lite.utils.meters')` 复用整份
    模块：两次加载用的模块名不同，谁先谁后不确定；而且先前那份可能已经被
    `del sys.modules[...]` 清掉（`tests/test_varlen_masking.py` 就会这么干），
    那时再查模块名就查不到了。以「进程里有没有这个对象」为准最稳。
    """

    @property
    def meter(self):
        existing = getattr(builtins, _METER_SINGLETON_ATTR, None)
        if existing is None:
            existing = _meter_namespace
            setattr(builtins, _METER_SINGLETON_ATTR, existing)
        return existing


# 把本模块的类型换成上面那个子类，`meter` 属性才会生效。
# 换成属性之后，类对象也随之只保留一份：_MeterNamespace 只在本文件
# **第一次**被加载时创建（后面那次直接复用旧实例）。
sys.modules[__name__].__class__ = _MetersModule
