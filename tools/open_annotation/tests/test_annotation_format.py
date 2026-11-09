"""Explicit cross-generation format dispatch: compatible profiles and refusals."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
import annotation_format as F
import audit_annotation_formats as A
import normalization
import schema


def record(clock=F.CLOCK_LEGACY, state="candidate", windows=True):
    value = {"schema": F.SIDECAR_SCHEMA, "contract": {}, "decision": {"state": state},
             "errors": [], "execution": {}, "provenance": {}, "quality_axes": {},
             "source": {}, "source_identity": {}, "units": [], "vad": {},
             "windows": [{"time_mapping": clock}] if windows else []}
    return value


def receipt(family="queue-v1", retained=False, tree="/tree/a", contract="c" * 64):
    source = {"source_id": "src_" + "a" * 64, "sidecar_relpath": "voxceleb2/id1/x/1.wav.annotation.json",
              "sidecar_sha256": "b" * 64, "sidecar_bytes": 10, "decision": "unknown",
              "job_key": "d" * 64, "commit_sha256": "e" * 64}
    if family == "bounded-queue-v1":
        source.update({"producer_contract_sha256": contract, "producer_sidecar_tree": tree,
                       "retained_existing": retained})
    return {"schema": F.RECEIPT_QUEUE if family == "queue-v1" else F.RECEIPT_BOUNDED,
            "shard_id": "s1", "index_sha256": "f" * 64, "contract_sha256": contract,
            "shard_sha256": "1" * 64, "source_count": 1, "attempt_id": "a",
            "committed_utc": "2026-11-09T00:00:00+00:00", "sources": [source]}


class SidecarProfileTests(unittest.TestCase):
    def test_both_historical_clocks_are_named_explicitly(self):
        self.assertEqual(F.sidecar_profile(record())["profile"], F.PROFILE_LEGACY)
        self.assertEqual(F.sidecar_profile(record(F.CLOCK_DECIMAL))["profile"], F.PROFILE_DECIMAL)

    def test_terminal_record_without_windows_has_no_clock(self):
        self.assertEqual(F.sidecar_profile(record(state="failed", windows=False)),
                         {"profile": "terminal-no-windows", "clock": None, "windows": 0})

    def test_unknown_schema_clock_mixed_clock_and_bad_decision_are_refused(self):
        cases = [
            ("schema", "redimnet2-open-annotation/v9", "unsupported sidecar schema"),
            ("clock", "source_offset_something_v3", "not a known generation clock"),
            ("decision", "maybe", "decision.state"),
        ]
        for field, value, message in cases:
            bad = record()
            if field == "clock":
                bad["windows"][0]["time_mapping"] = value
            elif field == "decision":
                bad["decision"]["state"] = value
            else:
                bad["schema"] = value
            with self.subTest(field=field), self.assertRaisesRegex(F.FormatError, message):
                F.sidecar_profile(bad)
        mixed = record()
        mixed["windows"].append({"time_mapping": F.CLOCK_DECIMAL})
        with self.assertRaisesRegex(F.FormatError, "mixes source clocks"):
            F.sidecar_profile(mixed)

    def test_missing_top_level_field_is_refused(self):
        bad = record(); bad.pop("vad")
        with self.assertRaisesRegex(F.FormatError, "missing top-level fields: vad"):
            F.sidecar_profile(bad)


class CommitTests(unittest.TestCase):
    def commit(self):
        return {"schema": F.COMMIT_SCHEMA, "source_id": "src_" + "a" * 64, "job_key": "b" * 64,
                "contract_sha256": "c" * 64, "source_record_sha256": "d" * 64,
                "sidecar_sha256": "e" * 64, "bytes": 42}

    def verify(self, commit):
        F.verify_commit(commit, sidecar_sha256="e" * 64, sidecar_bytes=42,
                        source_id="src_" + "a" * 64, contract_sha256="c" * 64)

    def test_matching_commit_passes(self):
        self.verify(self.commit())

    def test_divergent_or_foreign_commit_is_refused(self):
        for field, value, message in [
            ("schema", "redimnet2-sidecar-commit/v2", "unsupported sidecar commit schema"),
            ("sidecar_sha256", "0" * 64, "sha256 differs"),
            ("bytes", 43, "byte count differs"),
            ("source_id", "src_" + "0" * 64, "source_id differs"),
            ("contract_sha256", "0" * 64, "contract differs"),
        ]:
            bad = self.commit(); bad[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(F.FormatError, message):
                self.verify(bad)
        extra = self.commit(); extra["note"] = "x"
        with self.assertRaisesRegex(F.FormatError, "unexpected fields"):
            self.verify(extra)


class ReceiptTests(unittest.TestCase):
    def test_known_families_and_unknown_schema(self):
        self.assertEqual(F.receipt_family(receipt()), "queue-v1")
        self.assertEqual(F.receipt_family(receipt("bounded-queue-v1")), "bounded-queue-v1")
        with self.assertRaisesRegex(F.FormatError, "unsupported completion receipt schema"):
            F.receipt_family({"schema": "redimnet2-chunk-commit/v3"})

    def test_uniform_and_mixed_receipt_sets(self):
        self.assertEqual(F.assert_uniform_receipts([receipt(), receipt()]), "queue-v1")
        with self.assertRaisesRegex(F.FormatError, "mixes completion receipt generations"):
            F.assert_uniform_receipts([receipt(), receipt("bounded-queue-v1")])

    def test_v1_receipt_is_accepted_unchanged(self):
        value = receipt()
        self.assertEqual(F.normalize_completion_receipt(
            value, producer_contract_sha256="c" * 64, producer_sidecar_tree="/tree/a"), value)

    def test_homogeneous_bounded_receipt_is_explicitly_compatible(self):
        value = receipt("bounded-queue-v1")
        self.assertEqual(F.normalize_completion_receipt(
            value, producer_contract_sha256="c" * 64, producer_sidecar_tree="/tree/a"), value)

    def test_relocated_tree_with_same_version_name_is_accepted(self):
        # 11-09 receipts record the pre-migration experiments-root path while
        # the tree now lives under the shared root; version name + contract
        # pin the same content.
        value = receipt("bounded-queue-v1")
        self.assertEqual(F.normalize_completion_receipt(
            value, producer_contract_sha256="c" * 64,
            producer_sidecar_tree="/root/group-shared/new/root/a"), value)

    def test_bounded_receipt_with_retained_foreign_producer_is_refused(self):
        value = receipt("bounded-queue-v1", retained=True)
        with self.assertRaisesRegex(F.FormatError, "retained source from another producer"):
            F.normalize_completion_receipt(value, producer_contract_sha256="c" * 64,
                                           producer_sidecar_tree="/tree/a")
        for field, value_, message in [("producer_sidecar_tree", "/tree/b", "sidecar tree"),
                                       ("producer_contract_sha256", "9" * 64, "contract")]:
            bad = receipt("bounded-queue-v1"); bad["sources"][0][field] = value_
            with self.subTest(field=field), self.assertRaisesRegex(F.FormatError, message):
                F.normalize_completion_receipt(bad, producer_contract_sha256="c" * 64,
                                               producer_sidecar_tree="/tree/a")

    def test_missing_receipt_or_source_field_is_refused(self):
        bad = receipt(); bad.pop("shard_sha256")
        with self.assertRaisesRegex(F.FormatError, "receipt is missing fields"):
            F.normalize_completion_receipt(
                bad, producer_contract_sha256="c" * 64, producer_sidecar_tree="/tree/a")
        bad = receipt(); bad["sources"][0].pop("commit_sha256")
        with self.assertRaisesRegex(F.FormatError, "missing fields"):
            F.normalize_completion_receipt(bad, producer_contract_sha256="c" * 64,
                                           producer_sidecar_tree="/tree/a")


class LegacyProvenanceTests(unittest.TestCase):
    def test_legacy_free_text_profile_maps_to_explicit_policy(self):
        record = {"provenance": {"runtime": {"normalization": normalization.LEGACY_PROFILE_TEXT}}}
        self.assertEqual(schema._expected_normalization(record, "a'b"),
                         normalization.normalize_with_audit("a'b", normalization.POLICY))

    def test_unknown_profile_text_is_refused_not_defaulted(self):
        self.assertIsNone(schema._expected_normalization(
            {"provenance": {"runtime": {"normalization": "something else"}}}, "x"))

    def test_pre_provenance_record_keeps_bounded_policy_default(self):
        self.assertEqual(schema._expected_normalization({}, "a'b"),
                         normalization.normalize_with_audit("a'b", normalization.POLICY))

    def test_explicit_profile_wins_over_legacy_text(self):
        record = {"provenance": {"runtime": {
            "normalization_profile": normalization.APOSTROPHE_PROFILE,
            "normalization": normalization.LEGACY_PROFILE_TEXT}}}
        self.assertEqual(schema._expected_normalization(record, "a'b"),
                         normalization.normalize_with_audit("a'b", normalization.APOSTROPHE_PROFILE))


class TreeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        contract = "c" * 64
        identity = {"schema": F.SIDECAR_TREE_SCHEMA, "version": "fixture-v1",
                    "contract_sha256": contract}
        identity_path = self.root / "identity.json"
        identity_path.write_bytes(F.canonical_bytes(identity))
        identity_path.chmod(0o600)
        for source in A.SOURCES:
            directory = self.root / source / "id1"
            directory.mkdir(parents=True)
            relative = source + "/id1/a.wav.annotation.json"
            value = record(F.CLOCK_DECIMAL)
            value["source"] = {"dataset": source, "relative_path": "id1/a.wav",
                               "sidecar_relpath": relative, "source_id": "src_" + "a" * 64}
            raw = F.canonical_bytes(value)
            path = self.root / relative
            path.write_bytes(raw)
            path.chmod(0o600)
            commit = {"schema": F.COMMIT_SCHEMA, "source_id": "src_" + "a" * 64,
                      "job_key": "b" * 64, "contract_sha256": contract,
                      "source_record_sha256": "d" * 64,
                      "sidecar_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            path.with_name(path.name + ".commit.json").write_bytes(F.canonical_bytes(commit))

    def tearDown(self):
        self.temp.cleanup()

    def test_tree_audit_reads_all_sources_and_commits(self):
        report = A.audit_tree(self.root, per_source=2)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["identity"]["mode"], "0o600")
        for source in A.SOURCES:
            entry = report["sources"][source]
            self.assertEqual(entry["sampled"], 1)
            self.assertEqual(entry["profiles"], {F.PROFILE_DECIMAL: 1})
            self.assertEqual(entry["decisions"], {"candidate": 1})
            self.assertEqual(entry["entries"]["commit_matches"], 1)
            self.assertEqual(entry["entries"]["canonical"], 1)
            self.assertEqual(entry["modes"], {"0o600": 1})


if __name__ == "__main__":
    unittest.main()
