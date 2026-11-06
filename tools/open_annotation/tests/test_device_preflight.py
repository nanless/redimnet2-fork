"""v4 入口 st_dev 预检（device_preflight）的纯逻辑回归。

不需要 torch / GPU / 真实文件 / 网络：所有 stat 都由假对象提供。
覆盖三件事：
  1) 三个样本全命中 → 返回 0，且 stdout 是可解析的一行 JSON；
  2) 不匹配 / 文件缺失 → 返回 21，并打印可 grep 的 PREFLIGHT_DEVICE_MISMATCH；
  3) 入口接线正确 → main() 在 auto8 下会把 21 原样透传出去（不被 __main__ 兜底改成 2），
     且非 auto8 时明确打印跳过、不误伤 W2 调试。
"""
from __future__ import annotations
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import device_preflight as D  # noqa: E402


class FakeStat:
    """用固定 device 表替代 os.stat；可指定要抛 FileNotFoundError 的路径。"""

    def __init__(self, device=2097190, missing=()):
        self.device = device
        self.missing = set(missing)
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        if path in self.missing:
            raise FileNotFoundError(2, "No such file or directory", path)
        return types.SimpleNamespace(st_dev=self.device, st_ino=1234, st_size=5678)


def run_check(stat, samples=D.SAMPLES):
    buffer = io.StringIO()
    code = D.check(samples, stat=stat, out=buffer)
    return code, buffer.getvalue()


class DevicePreflightUnitTests(unittest.TestCase):
    def test_all_samples_match_returns_zero_and_prints_one_json_line(self):
        stat = FakeStat(device=D.EXPECTED_DEVICE)
        code, text = run_check(stat)
        self.assertEqual(code, 0)
        lines = [line for line in text.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1, "命中时应只输出一行 JSON")
        payload = json.loads(lines[0])
        self.assertEqual(payload["schema"], "annotation-device-preflight/v1")
        self.assertEqual(payload["expected_device"], D.EXPECTED_DEVICE)
        self.assertEqual(len(payload["samples"]), 3)
        roles = [row["role"] for row in payload["samples"]]
        self.assertEqual(roles, ["orphan_source", "committed_source", "group_shared_readonly"])
        for row in payload["samples"]:
            for key in ("path", "st_dev", "st_ino", "st_size"):
                self.assertIn(key, row)
                self.assertIsNotNone(row[key])
            self.assertTrue(row["device_ok"])
            self.assertIsNone(row["error"])
        # 三个样本都必须被真的 stat 到，顺序与 SAMPLES 一致
        self.assertEqual(stat.calls, [sample["path"] for sample in D.SAMPLES])

    def test_mismatch_returns_21_and_prints_grepable_marker(self):
        stat = FakeStat(device=57)          # dev 主机实测值
        code, text = run_check(stat)
        self.assertEqual(code, D.EXIT_MISMATCH)
        self.assertEqual(code, 21)
        self.assertIn("PREFLIGHT_DEVICE_MISMATCH expected=2097190 actual=", text)
        self.assertIn("orphan_source:57", text)
        # 第一行仍是 JSON，便于平台日志抓取
        payload = json.loads(text.splitlines()[0])
        self.assertTrue(all(not row["device_ok"] for row in payload["samples"]))

    def test_missing_sample_is_a_mismatch_not_a_crash(self):
        missing = D.SAMPLES[2]["path"]
        stat = FakeStat(device=D.EXPECTED_DEVICE, missing={missing})
        code, text = run_check(stat)
        self.assertEqual(code, 21)
        payload = json.loads(text.splitlines()[0])
        row = [r for r in payload["samples"] if r["role"] == "group_shared_readonly"][0]
        self.assertIsNone(row["st_dev"])
        self.assertIn("FileNotFoundError", row["error"])
        self.assertIn("group_shared_readonly:missing", text)

    def test_check_writes_nothing_to_disk(self):
        stat = FakeStat(device=D.EXPECTED_DEVICE)
        with tempfile.TemporaryDirectory() as tmp:
            before = sorted(os.listdir(tmp))
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                run_check(stat, samples=())     # 空样本也要覆盖到 out 分支
                run_check(stat)
            finally:
                os.chdir(cwd)
            self.assertEqual(sorted(os.listdir(tmp)), before, "预检不得写任何文件")


def _wired():
    """预检是否已接进 gpu_launch.main()（未接入时只跑单元层，不给假红）。"""
    try:
        import gpu_launch
    except Exception:
        return False
    return hasattr(gpu_launch, "device_preflight") and "device_preflight.check()" in Path(gpu_launch.__file__).read_text(encoding="utf-8")


@unittest.skipUnless(_wired(), "预检未接入 gpu_launch.main()")
class DevicePreflightEntryTests(unittest.TestCase):
    """入口接线：auto8 时 21 必须透传；非 auto8 时不阻断 W2 调试。"""

    def argv(self, target_uuids):
        values = {
            "bundle-sha": "a" * 64, "contract": "/tmp/contract.json", "contract-sha": "b" * 64,
            "shard-index": "/tmp/index.json", "index-sha": "c" * 64,
            "queue-root": "/tmp/queue", "output-root": "/tmp/out-labels", "output": "/tmp/attempt",
            "target-uuids": target_uuids, "run-root": "/tmp/run", "device-lock-root": "/tmp/locks",
        }
        argv = ["gpu_launch.py", "--execute"]
        for key, value in values.items():
            argv += ["--" + key, value]
        return argv

    def test_auto8_mismatch_exits_21_and_never_calls_run(self):
        import gpu_launch
        called = []
        with patch.object(gpu_launch.device_preflight, "check", return_value=21), \
             patch.object(gpu_launch, "run", lambda args: called.append(args)), \
             patch.object(sys, "argv", self.argv("auto8")):
            with self.assertRaises(SystemExit) as caught:
                gpu_launch.main()
        self.assertEqual(caught.exception.code, 21, "exit 21 必须原样透传，不能被兜底改成 2")
        self.assertEqual(called, [], "预检失败时绝不能进入 run()")

    def test_non_auto8_skips_loudly_and_runs(self):
        import gpu_launch
        called = []
        buffer = io.StringIO()
        with patch.object(gpu_launch.device_preflight, "check", side_effect=AssertionError("不应被调用")), \
             patch.object(gpu_launch, "run", lambda args: called.append(args)), \
             patch.object(sys, "argv", self.argv("GPU-11111111-2222-3333-4444-555555555555")), \
             contextlib.redirect_stdout(buffer):
            gpu_launch.main()
        self.assertEqual(len(called), 1)
        self.assertIn("PREFLIGHT_DEVICE_CHECK_SKIPPED", buffer.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
