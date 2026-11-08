#!/usr/bin/env python3
"""Refusal tests for tools/ptn_resume/lm_launch/lm_stage_gate.py.

Only the checks that happen *before* the real `model_115.pt` is hashed can be
exercised with synthetic fixtures — everything after that (checkpoint bytes,
existing run root, tampered runtime files) was exercised against the real
package on the training host with `--verify-only` (exit 0), which is recorded in
the implementation report. Nothing here touches a GPU: the gate never imports
torch, and `--verify-only` returns before any training is started.

Run:
    PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
      -s tools/ptn_resume/tests -p 'test_lm_stage_gate.py'
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
GATE = REPO / "tools/ptn_resume/lm_launch/lm_stage_gate.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class Gate(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="lm-gate-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.package = self.root / "package"
        (self.package / "runtime").mkdir(parents=True)
        runtime_file = self.package / "runtime/lm_launch.sh"
        runtime_file.write_text("#!/bin/bash\n", encoding="utf-8")
        config = self.package / "resume.yaml"
        config.write_text("num_epochs: 1\n", encoding="utf-8")
        # Full LM contract so the cheap checks pass and the speaker-map gate is
        # reachable without the real 650 MB checkpoint (its byte hash runs last).
        import yaml

        recipe = yaml.safe_load((REPO / "applications/speaker_rec/examples/vb2_vox2_cnc2"
                                        "/recipes/redimnet2/b6/official_repro_lm.yaml").read_text())
        recipe["exp_dir"] = str(self.root / "lm")
        recipe["model_ckpt"] = str(self.root / "model_115.pt")
        config.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (self.root / "model_115.pt").write_bytes(b"not-a-real-checkpoint")

        self.mapping = {f"spk_{i:03d}": i for i in range(4)}
        speaker_map = write_json(self.root / "spk2id.json", self.mapping)
        label_lines = [f"utt_{i:05d} spk_{i % 4:03d}" for i in range(8)]
        label = self.root / "utt2spk"
        label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        contract = {
            "schema": "lm-from-ptn115/v1",
            "runtime_admission": False,
            "LM_started": False,
            "checkpoint": {"path": str(self.root / "model_115.pt"),
                           "sha256": "65497b6a00efac412a82d563aebc1e767056fef23e7f4234f48f89cc8b3aad5c", "bytes": 1},
            "clock": {"stage": "lm", "start_epoch": 1, "target_epoch": 1, "epoch_iter": 13075,
                      "batch_size_per_gpu": 48, "margin": 0.3, "optimizer": "new", "scaler": "new"},
            "speaker_map": {"path": str(speaker_map), "sha256": sha(speaker_map),
                            "ptn_speakers": 4, "classes": 4},
            "lm_label": {"path": str(label)},
            "new_run_root": str(self.root / "lm"),
            "resume_config_path": "resume.yaml",
            "resume_config_sha256": sha(config),
            "runtime_files": {"lm_launch.sh": sha(runtime_file)},
        }
        self.contract = write_json(self.package / "resume-contract.json", contract)
        self.bundle = write_json(self.package / "bundle_files.json", {"resume.yaml": sha(config)})

    def run_gate(self, **overrides):
        args = {
            "--package-root": str(self.package),
            "--contract": str(self.contract),
            "--verify-only": "",
        }
        # Compute defaults lazily: some tests delete a file on purpose, so the
        # digest must be taken before the fixture changes (or supplied by them).
        if "--contract-sha" not in overrides and self.contract.exists():
            args["--contract-sha"] = sha(self.contract)
        if "--package-sha" not in overrides and self.bundle.exists():
            args["--package-sha"] = sha(self.bundle)
        args.update(overrides)
        command = [sys.executable, "-B", str(GATE)]
        for key, value in args.items():
            if value == "":
                command.append(key)
            else:
                command.extend([key, str(value)])
        return subprocess.run(command, capture_output=True, text=True)

    def test_wrong_package_sha_is_refused(self):
        result = self.run_gate(**{"--package-sha": "1" * 64})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("package sha", result.stdout)

    def test_wrong_contract_sha_is_refused(self):
        result = self.run_gate(**{"--contract-sha": "2" * 64})
        self.assertEqual(result.returncode, 2)
        self.assertIn("contract sha", result.stdout)

    def test_runtime_admission_true_is_refused(self):
        payload = json.loads(self.contract.read_text())
        payload["runtime_admission"] = True
        write_json(self.contract, payload)
        result = self.run_gate()
        self.assertEqual(result.returncode, 2)
        self.assertIn("runtime_admission", result.stdout)

    def test_lm_started_true_is_refused(self):
        payload = json.loads(self.contract.read_text())
        payload["LM_started"] = True
        write_json(self.contract, payload)
        result = self.run_gate()
        self.assertEqual(result.returncode, 2)
        self.assertIn("LM_started", result.stdout)

    def test_missing_bundle_is_refused(self):
        pinned = sha(self.bundle)
        self.bundle.unlink()
        result = self.run_gate(**{"--package-sha": pinned})
        self.assertEqual(result.returncode, 2)
        self.assertIn("bundle_files.json", result.stdout)

    def mutate_contract(self, mutate):
        payload = json.loads(self.contract.read_text())
        mutate(payload)
        write_json(self.contract, payload)
        return payload

    def test_missing_spk2id_is_refused(self):
        Path(self.refs_path()).unlink()
        result = self.run_gate()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("spk2id.json 不存在", result.stdout)

    def test_spk2id_sha_mismatch_is_refused(self):
        self.mutate_contract(lambda c: c["speaker_map"].update(sha256="0" * 64))
        result = self.run_gate()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("spk2id.json 摘要不符", result.stdout)

    def test_spk2id_speaker_count_mismatch_is_refused(self):
        self.mutate_contract(lambda c: c["speaker_map"].update(ptn_speakers=99))
        result = self.run_gate()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("spk2id 人数", result.stdout)

    def test_lm_label_speaker_count_mismatch_is_refused(self):
        self.mutate_contract(lambda c: c["speaker_map"].update(classes=99))
        result = self.run_gate()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("LM 说话人数", result.stdout)

    def test_lm_speaker_missing_from_map_is_refused(self):
        # 4 distinct speakers (matching classes=4) but one of them is not in the map
        label = self.root / "utt2spk"
        label.write_text("utt_1 spk_000\nutt_2 stranger\nutt_3 spk_001\nutt_4 spk_002\n", encoding="utf-8")
        result = self.run_gate()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("不在 spk2id", result.stdout)

    def refs_path(self):
        return Path(json.loads(self.contract.read_text())["speaker_map"]["path"])

    def test_execute_flag_is_a_noop(self):
        # The PTN entry.py takes --execute; entry commands copied from it append
        # the flag. The gate must behave identically with and without it.
        plain = self.run_gate()
        with_flag = self.run_gate(**{"--execute": ""})
        self.assertEqual(plain.returncode, with_flag.returncode)
        self.assertEqual(plain.stdout, with_flag.stdout)

    def test_execute_flag_does_not_rescue_a_refusal(self):
        Path(self.refs_path()).unlink()
        result = self.run_gate(**{"--execute": ""})
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("spk2id.json 不存在", result.stdout)

    def test_gate_never_imports_torch(self):
        source = GATE.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("torch.", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class PostflightSemantics(unittest.TestCase):
    """postflight 的 ok 要同时要求训练退出码为 0（审计 P0，2026-11-08）。"""

    def _run_root(self, marker=True, model=True):
        root = Path(tempfile.mkdtemp(prefix="lm-postflight-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        if model:
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "model_1.pt").write_bytes(b"synthetic-checkpoint")
        if marker:
            (root / "train.log").write_text(
                "cold start: model_ckpt loads model parameters only; optimizer/scaler start fresh\n",
                encoding="utf-8")
        return root

    def _postflight(self, root, code):
        import importlib.util
        spec = importlib.util.spec_from_file_location("lm_stage_gate_under_test", GATE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.postflight({"new_run_root": str(root)}, code)
        # postflight 把回执写到 <run_root>/lm_receipt.json（返回值是 None）
        return json.loads((root / "lm_receipt.json").read_text(encoding="utf-8"))

    def test_postflight_ok_requires_exit_zero(self):
        payload = self._postflight(self._run_root(), 1)
        self.assertTrue(payload["checks"]["model_1_exists"])
        self.assertTrue(payload["checks"]["cold_start_clean"])
        self.assertFalse(payload["checks"]["training_succeeded"])
        self.assertFalse(payload["checks"]["ok"],
                         "崩溃(exit=1)但留下 model_1 与 marker 不能记成成功")

    def test_postflight_ok_when_exit_zero(self):
        payload = self._postflight(self._run_root(), 0)
        self.assertTrue(payload["checks"]["training_succeeded"])
        self.assertTrue(payload["checks"]["ok"])
