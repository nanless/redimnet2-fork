#!/usr/bin/env python3
"""Synthetic CPU tests for the three-segment LM preparation checker.

Every fixture here is generated inside a temporary directory: no real audio,
no real checkpoint, no GPU, no network, and no access to the Mac experiment
directories. Counts and pins are supplied through the references file, which is
exactly the override channel the checker records in its evidence blocks; the
production references omit those blocks and therefore use the repository pins.

Run with:
    PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tools/ptn_resume/tests -p 'test_three_segment_lm_check.py'
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "ptn_resume"))

import three_segment_lm_check as checker  # noqa: E402

RECIPE = REPO / "applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/official_repro_lm.yaml"

COUNTS = {"ptn_speakers": 6, "lm_speakers": 4, "lm_utterances": 768}
TASKS = {
    "original": "t-20261019232824-8zqrl",
    "failed": "t-20261024154424-vggvm",
    "successor": "t-20261026074833-ctjsk",
}


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_checkpoint(path: Path, epoch: int) -> Path:
    """A torch.save-shaped zip whose data.pkl carries `epoch` as a real int."""
    payload = pickle.dumps(
        {"epoch": epoch, "model": {"backbone.weight": None}, "optimizer": {}, "scaler": {}},
        protocol=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{path.name}.tmp/data.pkl", payload)
        archive.writestr(f"{path.name}.tmp/version", "3\n")
    return path


def make_csv(path: Path, start: int, end: int, skip=None, extra=None, bad_row=None) -> Path:
    lines = ["epoch,lr,wd,margin,vox1-eer-8s"]
    for epoch in range(start, end + 1):
        if skip is not None and epoch == skip:
            continue
        if bad_row is not None and epoch == bad_row:
            lines.append(f"{epoch},nan,2e-05,0.2,1.0")
            continue
        lines.append(f"{epoch},0.72,2e-05,0.2,{1.0 + epoch * 0.001:.4f}")
    for epoch in extra or []:
        lines.append(f"{epoch},0.72,2e-05,0.2,1.5")
    return write_text(path, "\n".join(lines) + "\n")


class Fixture:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="three-seg-lm-"))

    def build(self):
        root = self.root
        run_original = root / "segments/original"
        run_failed = root / "segments/failed"
        run_successor = root / "segments/successor"
        write_csv = make_csv(run_original / "val_log.csv", 1, 35)
        failed_csv = make_csv(run_failed / "val_log.csv", 36, 47)
        successor_csv = make_csv(run_successor / "val_log.csv", 48, 120)

        ckpt115 = make_checkpoint(run_successor / "models/model_115.pt", 115)
        ckpt120 = make_checkpoint(run_successor / "models/model_120.pt", 120)

        receipt = write_json(run_successor / "ptn_success.json", {
            "LM_started": False,
            "checkpoint": str(ckpt120),
            "checkpoint_sha256": sha256(ckpt120),
            "epochs": list(range(48, 121)),
            "historical_numeric_root_cause_proven": False,
            "ptn_complete": True,
            "three_segment_LM_contract_required": True,
        })

        names = [f"spk_{index:03d}" for index in range(COUNTS["ptn_speakers"])]
        write_json(run_successor / "spk2id.json", {name: index for index, name in enumerate(names)})
        lm_names = names[: COUNTS["lm_speakers"]]
        rows = []
        for index in range(COUNTS["lm_utterances"]):
            rows.append(f"utt_{index:05d} {lm_names[index % len(lm_names)]}")
        lm_label = write_text(root / "data/mix_join_lm/utt2spk", "\n".join(rows) + "\n")

        reviewed = write_json(root / "evidence/reviewed_events.json", {
            "review_id": "synthetic-seven",
            "parent_task_id": TASKS["original"],
            "parent_run_root": str(run_original),
            "events": [{"event_id": f"ev-{i}"} for i in range(7)],
        })

        devices = [{"index": i, "used_mib": 0} for i in range(8)]
        resources = write_json(root / "evidence/resource_return.json", {
            "GPU_compute_PID_absent": True,
            "snapshots": [
                {"UTC": "2026-11-03T02:12:20Z", "devices": devices},
                {"UTC": "2026-11-03T02:12:25Z", "devices": devices},
            ],
        })

        successor_log = write_text(root / "segments/successor/launch.log",
                                   "[INFO] epoch 120 step 15001 loss 2.1\n[INFO] saved model_120.pt\n")

        tree = root / "source"
        members = []
        for index in range(190):
            relative = f"applications/speaker_rec/module_{index:03d}.py"
            content = f"# member {index}\nVALUE = {index}\n"
            write_text(tree / relative, content)
            members.append({
                "relative": relative,
                "bytes": len(content.encode()),
                "mode": 0o644,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            })
        manifest = write_json(root / "evidence/source_manifest.json", members)
        index189 = write_json(root / "evidence/source_189.json",
                              [{"relative": f"m{i}"} for i in range(189)])
        index95 = write_json(root / "evidence/source_95.json",
                             [{"relative": f"s{i}"} for i in range(95)])
        pinned_189 = sha256(index189)
        changed = [members[7], members[8]]
        delta = write_json(root / "evidence/source_delta.json", {
            "members_before": 189,
            "members_after": 190,
            "original_source189_sha256": pinned_189,
            "changes": [
                {"change": "replace", "relative": changed[0]["relative"],
                 "old_sha256": "1" * 64, "new_sha256": changed[0]["sha256"]},
                {"change": "add", "relative": changed[1]["relative"],
                 "old_sha256": None, "new_sha256": changed[1]["sha256"]},
            ],
        })
        # ---- synthetic frozen LM package -------------------------------------
        pkg = root / "package"
        package_recipe = write_text(pkg / "resume.yaml", RECIPE.read_text())
        executor = write_text(pkg / "runtime/lm_launch.sh", "#!/bin/bash\necho lm\n")
        contract = {
            "schema": "lm-from-ptn115/v1",
            "runtime_admission": False,
            "LM_started": False,
            "checkpoint": {"path": str(ckpt115), "sha256": sha256(ckpt115), "bytes": ckpt115.stat().st_size},
            "clock": {
                "stage": "lm", "start_epoch": 1, "target_epoch": 1,
                "epoch_iter": COUNTS["lm_utterances"] // 384, "batch_size_per_gpu": 48,
                "world_size": 8, "margin": 0.3, "optimizer": "new", "scaler": "new",
                "rng_restored": False, "saved_epoch": 115, "lr": 0.00045471497,
            },
            "speaker_map": {"path": str(run_successor / "spk2id.json"), "sha256": sha256(run_successor / "spk2id.json"),
                            "ptn_speakers": COUNTS["ptn_speakers"], "classes": COUNTS["lm_speakers"]},
            "new_run_root": str(root / "lm-from-ptn115/lm"),
            "resume_config_path": "resume.yaml",
            "resume_config_sha256": sha256(package_recipe),
            "runtime_files": {"lm_launch.sh": sha256(executor)},
            "executor": "runtime/lm_launch.sh",
            "legacy": {"bundle_files_sha256": "840235e7b5963ada274531a29401b9f2bd75fe92e985a8e8e560341ac48350d8"},
        }
        contract_path = write_json(pkg / "resume-contract.json", contract)
        write_json(pkg / "bundle_files.json", {
            "resume.yaml": sha256(package_recipe),
            "runtime/lm_launch.sh": sha256(executor),
            "resume-contract.json": sha256(contract_path),
        })

        terminal = write_json(root / "evidence/terminal.json", {
            "task_id": TASKS["successor"],
            "state": "Succeeded",
            "active_workers": 0,
            "observed_at": "2026-11-03T02:20:00Z",
        })

        refs = {
            "schema": checker.SCHEMA_REFS,
            "counts": COUNTS,
            "pins": {
                "model_115_sha256": sha256(ckpt115),
                "model_120_sha256": sha256(ckpt120),
                "original_189_sha256": pinned_189,
            },
            "segments": {
                "original": {
                    "task_id": TASKS["original"],
                    "run_root": str(run_original),
                    "validation_csv": str(write_csv),
                    "task_id_provenance": "platform-console-record",
                },
                "failed": {
                    "task_id": TASKS["failed"],
                    "run_root": str(run_failed),
                    "validation_csv": str(failed_csv),
                    "task_id_provenance": "platform-console-record",
                },
                "successor": {
                    "task_id": TASKS["successor"],
                    "run_root": str(run_successor),
                    "validation_csv": str(successor_csv),
                    "task_id_provenance": "repository-record, not CLI-verified",
                },
            },
            "checkpoints": [
                {"path": str(ckpt115), "epoch": 115},
                {"path": str(ckpt120), "epoch": 120},
            ],
            "success_receipt": str(receipt),
            "speaker_map": str(run_successor / "spk2id.json"),
            "lm_label": str(lm_label),
            "resource_return": str(resources),
            "reviewed_events": {"path": str(reviewed), "expected": 7},
            "successor_log": str(successor_log),
            "source": {
                "manifest_190": str(manifest),
                "delta": str(delta),
                "source_tree": str(tree),
                "original_189_index": str(index189),
                "original_95_index": str(index95),
                "original_95_sha256": sha256(index95),
            },
            "lm_recipe": str(RECIPE),
            "lm_contract": {
                "optimizer_fresh": True,
                "scaler_fresh": True,
                "loads_model_parameters_only": True,
                "restores_ptn_optimizer": False,
                "restores_ptn_scaler": False,
                "automatic_retry": False,
                "forbidden_existing_paths": [str(root / "lm-from-ptn115/lm")],
            },
            "platform": {"terminal": str(terminal)},
            "package": {
                "root": str(pkg),
                "bundle_files": str(pkg / "bundle_files.json"),
                "contract": str(contract_path),
            },
        }
        return refs

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class Base(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.cleanup)
        self.refs = self.fx.build()
        self.out = self.fx.root / "out"

    # -- helpers ---------------------------------------------------------
    def run_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = checker.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def write_refs(self, refs=None, name="refs.json"):
        path = self.out / name
        write_json(path, refs if refs is not None else self.refs)
        return path

    def build_contract(self, refs=None):
        refs_path = self.write_refs(refs)
        contract = self.out / "contract.json"
        code, out, err = self.run_main(["build", "--references", str(refs_path), "--output", str(contract)])
        return code, contract, out, err

    def check_contract(self, contract, name="report.json"):
        report = self.out / name
        code, out, err = self.run_main(["check", "--contract", str(contract), "--output", str(report)])
        payload = json.loads(report.read_text()) if report.exists() else None
        return code, payload, out, err

    def statuses(self, payload):
        return {check["name"]: check["status"] for check in payload["checks"]}

    def named(self, payload, name):
        for check in payload["checks"]:
            if check["name"] == name:
                return check
        raise AssertionError(f"missing check {name}: {list(self.statuses(payload))}")


class HappyPath(Base):
    def test_build_and_check_pass_without_granting_authority(self):
        code, contract, out, err = self.build_contract()
        self.assertEqual(code, 0, err)
        contract_payload = json.loads(contract.read_text())
        self.assertFalse(contract_payload["ready_for_lm_execution"])
        self.assertEqual(contract_payload["pin_source"], "references-override")
        self.assertEqual(contract_payload["counts_source"], "references-override")

        code, report, _out, err2 = self.check_contract(contract)
        self.assertEqual(code, 0, err2)
        self.assertTrue(report["checks_passed"])
        self.assertFalse(report["ready_for_lm_execution"])
        self.assertEqual(report["drift"], [])
        self.assertIn("none", report["execution_authority"])
        statuses = self.statuses(report)
        self.assertTrue(all(status == "pass" for status in statuses.values()), statuses)
        self.assertEqual(set(statuses), {
            "segments.validation_original", "segments.validation_failed", "segments.validation_successor",
            "segments.complete_coverage_1_120", "segments.task_identity", "checkpoints.identity",
            "success.receipt", "speaker_map.subset", "reviewed_events.scope",
            "successor.no_unreviewed_failures", "source.manifest_190_and_delta",
            "source.original_189_index", "source.original_95_index",
            "package.lm_contract", "lm.cold_start_contract", "resources.returned",
            "platform.terminal_observed", "contract.no_drift",
        })

    def test_stdout_mode_is_usable_without_writing_files(self):
        refs_path = self.write_refs()
        code, out, err = self.run_main(["build", "--references", str(refs_path), "--output", "-"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(payload["schema"], checker.SCHEMA_CONTRACT)
        self.assertFalse(payload["ready_for_lm_execution"])

    def test_references_from_environment(self):
        raw = json.dumps(self.refs).encode()
        os.environ["THREE_SEG_LM_REFS_B64"] = __import__("base64").b64encode(raw).decode()
        self.addCleanup(os.environ.pop, "THREE_SEG_LM_REFS_B64", None)
        code, out, err = self.run_main(["build", "--output", "-"])
        self.assertEqual(code, 0, err)
        self.assertFalse(json.loads(out)["ready_for_lm_execution"])


class Failures(Base):
    def expect_fail(self, name):
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 2, f"expected build to fail: {err}")
        payload = json.loads(contract.read_text())
        self.assertEqual(self.named(payload, name)["status"], "fail")
        self.assertFalse(payload["summary"]["checks_passed"])

    def test_validation_gap_in_failed_segment(self):
        self.refs["segments"]["failed"]["validation_csv"] = str(
            make_csv(self.fx.root / "bad/failed.csv", 36, 47, skip=41)
        )
        self.expect_fail("segments.validation_failed")

    def test_validation_extra_epoch(self):
        self.refs["segments"]["successor"]["validation_csv"] = str(
            make_csv(self.fx.root / "bad/successor.csv", 48, 120, extra=[121])
        )
        self.expect_fail("segments.validation_successor")

    def test_validation_nonfinite_metric(self):
        self.refs["segments"]["original"]["validation_csv"] = str(
            make_csv(self.fx.root / "bad/original.csv", 1, 35, bad_row=17)
        )
        self.expect_fail("segments.validation_original")

    def test_checkpoint_sha_mismatch(self):
        ckpt = Path(self.refs["checkpoints"][0]["path"])
        ckpt.write_bytes(ckpt.read_bytes() + b"tampered")
        self.expect_fail("checkpoints.identity")

    def test_checkpoint_epoch_mismatch(self):
        # The archive says 119 while the references demand 120: the saved-round
        # identity is what has to agree, not the file name.
        path = Path(self.refs["checkpoints"][1]["path"])
        make_checkpoint(path, 119)
        self.refs["pins"]["model_120_sha256"] = sha256(path)
        self.expect_fail("checkpoints.identity")

    def test_checkpoint_declared_epoch_outside_115_120_is_refused(self):
        self.refs["checkpoints"][1]["epoch"] = 119
        refs_path = self.write_refs()
        code, _out, err = self.run_main(["build", "--references", str(refs_path), "--output", "-"])
        self.assertEqual(code, 2)
        self.assertIn("115", err)

    def test_receipt_claims_root_cause_proven(self):
        receipt = Path(self.refs["success_receipt"])
        payload = json.loads(receipt.read_text())
        payload["historical_numeric_root_cause_proven"] = True
        write_json(receipt, payload)
        self.expect_fail("success.receipt")

    def test_receipt_lm_already_started(self):
        receipt = Path(self.refs["success_receipt"])
        payload = json.loads(receipt.read_text())
        payload["LM_started"] = True
        write_json(receipt, payload)
        self.expect_fail("success.receipt")

    def test_receipt_short_epochs(self):
        receipt = Path(self.refs["success_receipt"])
        payload = json.loads(receipt.read_text())
        payload["epochs"] = list(range(48, 120))
        write_json(receipt, payload)
        self.expect_fail("success.receipt")

    def test_speaker_ids_not_lexical(self):
        path = Path(self.refs["speaker_map"])
        payload = json.loads(path.read_text())
        flipped = dict(payload)
        names = sorted(payload)
        flipped[names[0]], flipped[names[1]] = payload[names[1]], payload[names[0]]
        write_json(path, flipped)
        self.expect_fail("speaker_map.subset")

    def test_lm_speaker_not_in_ptn(self):
        path = Path(self.refs["lm_label"])
        lines = path.read_text().splitlines()
        parts = lines[0].split()
        lines[0] = f"{parts[0]} unseen_speaker"
        write_text(path, "\n".join(lines) + "\n")
        self.expect_fail("speaker_map.subset")

    def test_reviewed_events_bound_to_other_task(self):
        path = Path(self.refs["reviewed_events"]["path"])
        payload = json.loads(path.read_text())
        payload["parent_task_id"] = TASKS["failed"]
        write_json(path, payload)
        self.expect_fail("reviewed_events.scope")

    def test_reviewed_events_wrong_count(self):
        path = Path(self.refs["reviewed_events"]["path"])
        payload = json.loads(path.read_text())
        payload["events"] = payload["events"][:6]
        write_json(path, payload)
        self.expect_fail("reviewed_events.scope")

    def test_successor_log_fatal_pattern(self):
        write_text(Path(self.refs["successor_log"]),
                   "[INFO] ok\nTraceback (most recent call last):\n")
        self.expect_fail("successor.no_unreviewed_failures")

    def test_source_manifest_wrong_member_count(self):
        path = Path(self.refs["source"]["manifest_190"])
        payload = json.loads(path.read_text())
        write_json(path, payload[:-1])
        self.expect_fail("source.manifest_190_and_delta")

    def test_source_tree_bytes_differ_from_manifest(self):
        member = json.loads(Path(self.refs["source"]["manifest_190"]).read_text())[3]
        write_text(Path(self.refs["source"]["source_tree"]) / member["relative"], "# changed\n")
        self.expect_fail("source.manifest_190_and_delta")

    def test_delta_not_two_changes(self):
        path = Path(self.refs["source"]["delta"])
        payload = json.loads(path.read_text())
        payload["changes"] = payload["changes"][:1]
        write_json(path, payload)
        self.expect_fail("source.manifest_190_and_delta")

    def test_lm_recipe_wrong_margin(self):
        text = RECIPE.read_text()
        tampered = text.replace("final_margin: 0.3", "final_margin: 0.2")
        self.assertNotEqual(text, tampered)
        self.refs["lm_recipe"] = str(write_text(self.fx.root / "bad/lm.yaml", tampered))
        self.expect_fail("lm.cold_start_contract")

    def test_lm_contract_claims_no_cold_start(self):
        self.refs["lm_contract"]["restores_ptn_optimizer"] = True
        self.expect_fail("lm.cold_start_contract")

    def test_lm_output_directory_already_exists(self):
        existing = Path(self.refs["lm_contract"]["forbidden_existing_paths"][0])
        existing.mkdir(parents=True)
        self.expect_fail("lm.cold_start_contract")

    def test_duplicate_json_key_rejected(self):
        path = self.refs["success_receipt"]
        text = Path(path).read_text()
        broken = text.replace('"ptn_complete": true', '"ptn_complete": true, "ptn_complete": true')
        self.assertNotEqual(text, broken)
        write_text(Path(path), broken)
        self.expect_fail("success.receipt")

    def test_unsupported_95_index_is_not_a_failure_but_blocks_passing(self):
        self.refs["source"]["original_95_index"] = None
        self.refs["source"]["original_95_sha256"] = None
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 3, err)
        payload = json.loads(contract.read_text())
        self.assertEqual(self.named(payload, "source.original_95_index")["status"], "unsupported")
        self.assertFalse(payload["summary"]["checks_passed"])

    def test_missing_platform_terminal_is_unsupported(self):
        self.refs["platform"] = {}
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 3, err)
        payload = json.loads(contract.read_text())
        self.assertEqual(self.named(payload, "platform.terminal_observed")["status"], "unsupported")
        self.assertFalse(payload["summary"]["checks_passed"])

    def test_missing_successor_log_is_unsupported(self):
        self.refs.pop("successor_log", None)
        code, contract, _out, _err = self.build_contract()
        self.assertEqual(code, 3)
        payload = json.loads(contract.read_text())
        self.assertEqual(self.named(payload, "successor.no_unreviewed_failures")["status"], "unsupported")


class Drift(Base):
    def test_artifact_changed_after_build_is_caught(self):
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 0, err)
        victim = Path(self.refs["lm_label"])
        victim.write_text(victim.read_text() + "\nextra_utt spk_000\n", encoding="utf-8")

        code, report, _out, _err = self.check_contract(contract)
        self.assertEqual(code, 2)
        self.assertFalse(report["checks_passed"])
        self.assertTrue(report["drift"])
        self.assertEqual(self.named(report, "contract.no_drift")["status"], "fail")
        artifacts = [entry["artifact"] for entry in report["drift"]]
        self.assertIn("lm_label", artifacts)

    def test_hand_written_contract_cannot_claim_authority(self):
        contract = self.out / "contract.json"
        code, _contract_path, _out, err = self.build_contract()
        self.assertEqual(code, 0, err)
        payload = json.loads(contract.read_text())
        payload["ready_for_lm_execution"] = True
        write_json(contract, payload)
        code, report, _out, _err = self.check_contract(contract)
        self.assertIsNone(report)
        self.assertEqual(code, 2)

    def test_check_rederives_instead_of_trusting_stored_verdicts(self):
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 0, err)
        payload = json.loads(contract.read_text())
        for check in payload["checks"]:
            check["status"] = "pass"
        write_json(contract, payload)
        code, report, _out, _err = self.check_contract(contract)
        self.assertEqual(code, 0)
        self.assertTrue(report["checks_passed"])

    def test_broken_contract_schema_is_refused(self):
        refs_path = self.write_refs()
        contract = write_json(self.out / "contract.json",
                              {"schema": "wrong/v1", "references": self.refs})
        code, _report, _out, err = self.check_contract(contract)
        self.assertEqual(code, 2)
        self.assertIn("schema", err)


class Package(Base):
    def rebuild_bundle(self):
        bundle_path = Path(self.refs["package"]["bundle_files"])
        bundle = json.loads(bundle_path.read_text())
        root = Path(self.refs["package"]["root"])
        bundle["resume.yaml"] = sha256(root / "resume.yaml")
        bundle["runtime/lm_launch.sh"] = sha256(root / "runtime/lm_launch.sh")
        bundle["resume-contract.json"] = sha256(root / "resume-contract.json")
        write_json(bundle_path, bundle)
        return bundle

    def rewrite_contract(self, mutate):
        path = Path(self.refs["package"]["contract"])
        payload = json.loads(path.read_text())
        mutate(payload)
        write_json(path, payload)
        self.rebuild_bundle()
        return payload

    def expect_package_fail(self, needle=""):
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 2, err)
        payload = json.loads(contract.read_text())
        check = self.named(payload, "package.lm_contract")
        self.assertEqual(check["status"], "fail")
        if needle:
            self.assertIn(needle, check["reason"])

    def test_happy_package_passes(self):
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 0, err)
        payload = json.loads(contract.read_text())
        check = self.named(payload, "package.lm_contract")
        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["evidence"]["members"], 3)
        self.assertEqual(check["evidence"]["executor"], "runtime/lm_launch.sh")

    def test_tampered_runtime_member(self):
        Path(self.refs["package"]["root"], "runtime/lm_launch.sh").write_text("#!/bin/bash\necho other\n")
        self.expect_package_fail("包成员摘要不符")

    def test_unlisted_file_in_package(self):
        write_text(Path(self.refs["package"]["root"], "runtime/stowaway.py"), "print('x')\n")
        self.expect_package_fail("未登记文件")

    def test_contract_restores_ptn_optimizer(self):
        self.rewrite_contract(lambda c: c["clock"].update(optimizer="restore"))
        self.expect_package_fail("优化器必须是 new")

    def test_contract_wrong_epoch_iter(self):
        self.rewrite_contract(lambda c: c["clock"].update(epoch_iter=13075))
        self.expect_package_fail("epoch_iter")

    def test_contract_new_run_root_exists(self):
        run_root = Path(self.refs["package"]["root"]).parent / "lm-from-ptn115/lm"
        run_root.mkdir(parents=True)
        self.expect_package_fail("已存在")

    def test_executor_not_in_bundle(self):
        bundle_path = Path(self.refs["package"]["bundle_files"])
        bundle = json.loads(bundle_path.read_text())
        bundle.pop("runtime/lm_launch.sh")
        write_json(bundle_path, bundle)
        self.expect_package_fail()

    def test_runtime_file_hash_mismatch(self):
        self.rewrite_contract(lambda c: c["runtime_files"].update({"lm_launch.sh": "0" * 64}))
        self.expect_package_fail("runtime 文件摘要不符")

    def test_missing_package_is_unsupported(self):
        self.refs.pop("package")
        code, contract, _out, err = self.build_contract()
        self.assertEqual(code, 3, err)
        payload = json.loads(contract.read_text())
        self.assertEqual(self.named(payload, "package.lm_contract")["status"], "unsupported")


class Primitives(unittest.TestCase):
    def test_epoch_extraction_from_torch_zip(self):
        root = Path(tempfile.mkdtemp(prefix="three-seg-zip-"))
        self.addCleanup(shutil.rmtree, root, True)
        path = make_checkpoint(root / "model_115.pt", 115)
        facts = checker.torch_zip_facts(path)
        self.assertEqual(facts["recorded_epoch"], 115)
        self.assertEqual(facts["zip_entries"], 2)

    def test_missing_epoch_key_is_rejected(self):
        root = Path(tempfile.mkdtemp(prefix="three-seg-zip-"))
        self.addCleanup(shutil.rmtree, root, True)
        path = root / "model_1.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("x.pt.tmp/data.pkl", pickle.dumps({"model": {}, "optimizer": {}, "scaler": {}}, protocol=2))
        with self.assertRaises(checker.Bad):
            checker.torch_zip_facts(path)

    def test_relative_path_is_rejected(self):
        with self.assertRaises(checker.Bad):
            checker.sha256_stream("relative/path.pt")

    def test_symlink_is_rejected(self):
        root = Path(tempfile.mkdtemp(prefix="three-seg-link-"))
        self.addCleanup(shutil.rmtree, root, True)
        target = write_text(root / "real.txt", "data\n")
        link = root / "link.txt"
        os.symlink(target, link)
        with self.assertRaises(checker.Bad):
            checker.sha256_stream(link)

    def test_lm_steps_constant_matches_recipe_arithmetic(self):
        self.assertEqual(checker.LM_STEPS, 13075)


if __name__ == "__main__":
    unittest.main(verbosity=2)
