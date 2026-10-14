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
    """

    AverageValueMeter = AverageValueMeter
    ClassErrorMeter = ClassErrorMeter


meter = _MeterNamespace()
