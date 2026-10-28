#!/usr/bin/env python3
"""Regression tests for multi-run log segmentation, timestamp choice, and ETA guards."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run_script(script: str, content: str, *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "train.log"
        log.write_text(content, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(HERE / script), str(log), *args],
            text=True,
            capture_output=True,
            check=False,
        )


class PseAnalyzerTests(unittest.TestCase):
    def test_latest_restart_segment_and_unknown_timezone(self):
        result = run_script(
            "analyze_pse_log.py",
            """[2026-08-30 10:00:00] [TrainState] step=100 x
[2026-08-30 10:10:00] [TrainState] step=200 x
Starting new training run
[2026-08-31 10:00:00] [TrainState] step=150 x
[2026-08-31 10:10:00] [TrainState] step=250 x
""",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("segments=2", result.stdout)
        self.assertIn("6.000", result.stdout)
        self.assertIn("时区未知", result.stdout)
        self.assertNotIn("580.000", result.stdout)

    def test_global_step_requires_epoch_baseline(self):
        result = run_script(
            "analyze_pse_log.py",
            """[2026-08-31 10:00:00.000] [TrainState] step=5200 x
[2026-08-31 10:10:00.000] [TrainState] step=5300 x
""",
            "--steps-per-epoch",
            "1000",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("请提供 --epoch-start-step", result.stdout)


class VadAnalyzerTests(unittest.TestCase):
    def test_latest_run_inner_timestamp_and_zero_based_remaining(self):
        result = run_script(
            "analyze_vad_log.py",
            """[2026-08-30 10:00:00] [2026-08-30 10:00:00] Epoch 3, Batch 0/5, Loss: 1
[2026-08-30 10:00:04] [2026-08-30 10:00:40] Epoch 3, Batch 4/5, Loss: 1
Starting new training run
[2026-08-31 10:00:00] [2026-08-31 10:00:00] Epoch 1, Batch 0/5, Loss: 1
[2026-08-31 10:00:04] [2026-08-31 10:00:40] Epoch 1, Batch 4/5, Loss: 1
""",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Epoch 1", result.stdout)
        self.assertIn("10.000 秒/batch", result.stdout)
        self.assertIn("剩余 0 batch", result.stdout)


if __name__ == "__main__":
    unittest.main()
