"""annotation_inventory 的去重、过滤、文件核对与幂等测试；只用临时合成收据。"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import annotation_inventory as M


def source_id(number):
    return "src_" + hashlib.sha256(f"source-{number}".encode()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source(number, dataset="voxceleb2", size=10, decision="candidate", relpath=None):
    return {
        "source_id": source_id(number),
        "sidecar_relpath": relpath or f"{dataset}/id{number:05d}/clip/00001.wav.annotation.json",
        "sidecar_sha256": digest(f"sidecar-{number}"),
        "sidecar_bytes": size,
        "decision": decision,
    }


def make_run(base, generation, profile="silver-profile"):
    root = Path(base) / generation / "run-v1"
    (root / "queue" / "completed").mkdir(parents=True)
    (root / "labels" / profile).mkdir(parents=True)
    return root


def put_receipt(root, name, sources, committed, schema="redimnet2-chunk-commit/v1"):
    path = Path(root) / "queue" / "completed" / f"{name}.json"
    document = {
        "schema": schema,
        "shard_id": name,
        "committed_utc": committed,
        "source_count": len(sources),
        "sources": sources,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def put_sidecar(root, relpath, payload=b"0123456789"):
    path = Path(root) / "labels" / "silver-profile" / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    # 1. 同源跨代重复：保留 committed_utc 最新的一行。
    def test_dedup_keeps_latest_commit(self):
        older = make_run(self.base, "portable-successor-20261027-v1")
        newer = make_run(self.base, "handoff-v4r2-20261108-v6")
        put_receipt(older, "partial-a", [source(1, size=11)], "2026-10-27T17:02:58.188381+00:00")
        put_receipt(newer, "partial-b", [source(1, size=22, relpath="voxblink2/id1/clip/00001.wav.annotation.json")],
                    "2026-11-08T07:11:29.153920+00:00")
        inventory = M.scan(M.specs_from_roots([older, newer]), progress_every=0)
        self.assertEqual(inventory.unique_sources, 1)
        self.assertEqual(inventory.total_rows, 2)
        row = inventory.rows[0]
        self.assertEqual(row["lineage"], "handoff-v4r2-20261108-v6/run-v1")
        self.assertEqual(row["committed_utc"], "2026-11-08T07:11:29.153920+00:00")
        self.assertEqual(row["sidecar_bytes"], 22)
        self.assertEqual(inventory.overlap_sources, 1)
        self.assertEqual(inventory.duplicate_rows, 1)

    # 2. 不同代之间的重叠计数与两两重叠。
    def test_overlap_across_lineages(self):
        left = make_run(self.base, "portable-successor-20261027-v1")
        right = make_run(self.base, "native-speed-successor-20261028-v1")
        put_receipt(left, "partial-a", [source(1), source(2)], "2026-10-27T17:02:58+00:00")
        put_receipt(right, "partial-b", [source(2), source(3)], "2026-10-28T11:48:26+00:00")
        inventory = M.scan(M.specs_from_roots([left, right]), progress_every=0)
        self.assertEqual(inventory.unique_sources, 3)
        self.assertEqual(inventory.overlap_sources, 1)
        pair = inventory.pair_overlaps[
            ("portable-successor-20261027-v1/run-v1", "native-speed-successor-20261028-v1/run-v1")
        ]
        self.assertEqual(pair, 1)
        self.assertEqual([state.unique for state in inventory.lineages], [2, 2])
        self.assertEqual([state.exclusive for state in inventory.lineages], [1, 1])
        report = M.format_report(inventory)
        self.assertIn("跨代重叠：**1**", report)
        self.assertIn("按数据集分布", report)

    # 3. filter 正确剔除命中源，并同步顶层 source_count。
    def test_filter_removes_done_sources(self):
        root = make_run(self.base, "generation-a")
        put_receipt(root, "partial-a", [source(1), source(2)], "2026-10-27T17:02:58+00:00")
        inventory = M.scan(M.specs_from_roots([root]), progress_every=0)
        done = {row["source_id"]: row for row in inventory.rows}
        queue = self.base / "queue.json"
        queue.write_text(json.dumps({
            "schema": "pending-queue/v1",
            "source_count": 3,
            "sources": [source(1), source(2), source(3)],
        }), encoding="utf-8")
        out = self.base / "filtered.json"
        summary = M.filter_queue_file(queue, out, done)
        self.assertEqual((summary["removed"], summary["remaining"]), (2, 1))
        filtered = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([item["source_id"] for item in filtered["sources"]], [source_id(3)])
        self.assertEqual(filtered["source_count"], 1)

    # 4. --check-files：侧车缺失或大小不符时保留该源，不误删。
    def test_missing_or_mismatched_sidecar_is_kept(self):
        root = make_run(self.base, "generation-a")
        good, missing, wrong = source(1, size=10), source(2, size=10), source(3, size=99)
        put_receipt(root, "partial-a", [good, missing, wrong], "2026-10-27T17:02:58+00:00")
        put_sidecar(root, good["sidecar_relpath"], b"0123456789")
        put_sidecar(root, wrong["sidecar_relpath"], b"short")
        inventory = M.scan(M.specs_from_roots([root]), progress_every=0)
        done = {row["source_id"]: row for row in inventory.rows}
        queue = self.base / "queue.json"
        queue.write_text(json.dumps({"sources": [good, missing, wrong]}), encoding="utf-8")
        out = self.base / "filtered.json"
        summary = M.filter_queue_file(
            queue, out, done, labels=M.labels_dirs(M.specs_from_roots([root])), check_files=True
        )
        self.assertEqual(summary["removed"], 1)
        self.assertEqual(summary["remaining"], 2)
        self.assertEqual(summary["missing_sidecar_kept"], 1)
        self.assertEqual(summary["size_mismatch_kept"], 1)
        kept = [item["source_id"] for item in json.loads(out.read_text(encoding="utf-8"))["sources"]]
        self.assertEqual(kept, [source_id(2), source_id(3)])

    # 5. 空的 completed 目录是合法的零结果。
    def test_empty_completed_directory(self):
        root = make_run(self.base, "generation-empty")
        inventory = M.scan(M.specs_from_roots([root]), progress_every=0)
        self.assertEqual(inventory.rows, [])
        self.assertEqual(inventory.unique_sources, 0)
        out = self.base / "done-index.jsonl"
        self.assertTrue(M.write_index(out, inventory.rows))
        self.assertEqual(out.read_bytes(), b"")
        self.assertEqual(M.load_index(out), [])
        self.assertIn("唯一源（去重后）：**0**", M.format_report(inventory))

    # 6. 索引与过滤重复跑都不改变结果（字节和 mtime 都不变）。
    def test_idempotent_rerun(self):
        roots = [make_run(self.base, "generation-a"), make_run(self.base, "generation-b")]
        put_receipt(roots[0], "partial-a", [source(1), source(2)], "2026-10-27T17:02:58+00:00")
        put_receipt(roots[1], "partial-b", [source(2), source(3)], "2026-10-28T11:48:26+00:00")
        specs = M.specs_from_roots(roots)
        first = M.serialize_rows(M.scan(specs, progress_every=0).rows)
        second = M.serialize_rows(M.scan(list(reversed(specs)), progress_every=0).rows)
        self.assertEqual(first, second)
        out = self.base / "done-index.jsonl"
        self.assertTrue(M.write_index(out, M.scan(specs, progress_every=0).rows))
        before = out.stat().st_mtime_ns
        self.assertFalse(M.write_index(out, M.scan(specs, progress_every=0).rows))
        self.assertEqual(out.stat().st_mtime_ns, before)
        queue = self.base / "queue.json"
        queue.write_text(json.dumps({"sources": [source(1), source(2), source(3)]}), encoding="utf-8")
        done = {row["source_id"]: row for row in M.load_index(out)}
        filtered = self.base / "filtered.json"
        M.filter_queue_file(queue, filtered, done)
        filtered_before = (filtered.read_bytes(), filtered.stat().st_mtime_ns)
        M.filter_queue_file(queue, filtered, done)
        self.assertEqual((filtered.read_bytes(), filtered.stat().st_mtime_ns), filtered_before)

    # 7. JSONL 队列按原行保留，不重排、不改写。
    def test_jsonl_queue_keeps_original_lines(self):
        root = make_run(self.base, "generation-a")
        put_receipt(root, "partial-a", [source(1)], "2026-10-27T17:02:58+00:00")
        inventory = M.scan(M.specs_from_roots([root]), progress_every=0)
        done = {row["source_id"]: row for row in inventory.rows}
        queue = self.base / "queue.jsonl"
        lines = [json.dumps(source(1)), json.dumps(source(2), sort_keys=True), json.dumps(source(3))]
        queue.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = self.base / "filtered.jsonl"
        summary = M.filter_queue_file(queue, out, done)
        self.assertEqual((summary["removed"], summary["remaining"]), (1, 2))
        self.assertEqual(out.read_text(encoding="utf-8"), "\n".join(lines[1:]) + "\n")

    # 8. 注册表里同名代指向两个根时拒绝，避免把老副本重复计入。
    def test_registry_duplicate_lineage_refused(self):
        left = make_run(self.base, "generation-a")
        right = make_run(self.base, "generation-b")
        registry = self.base / "registry.json"
        registry.write_text(json.dumps({"runs": [
            {"path": str(left), "lineage": "same-lineage"},
            {"path": str(right), "lineage": "same-lineage"},
        ]}), encoding="utf-8")
        with self.assertRaises(M.InventoryError):
            M.load_registry(registry)

    # 9. 收据里的 sidecar_relpath 不允许越界。
    def test_unsafe_sidecar_relpath_refused(self):
        root = make_run(self.base, "generation-a")
        put_receipt(root, "partial-a",
                    [source(1, relpath="../outside.wav.annotation.json")], "2026-10-27T17:02:58+00:00")
        with self.assertRaises(M.InventoryError):
            M.scan(M.specs_from_roots([root]), progress_every=0)

    # 10. 没有 queue/completed 的根默认拒绝；显式 optional 才允许记零。
    def test_missing_completed_requires_optional(self):
        root = self.base / "generation-c" / "run-v1"
        root.mkdir(parents=True)
        with self.assertRaises(M.InventoryError):
            M.scan([M.RunSpec(root=root, lineage="generation-c/run-v1")], progress_every=0)
        inventory = M.scan([M.RunSpec(root=root, lineage="generation-c/run-v1", optional=True)],
                           progress_every=0)
        self.assertEqual(inventory.unique_sources, 0)
        self.assertEqual(inventory.lineages[0].receipts, 0)

    # 11. CLI 端到端：建索引 + 报表 + 过滤 + 回执。
    def test_cli_report_and_filter(self):
        roots = [make_run(self.base, "generation-a")]
        put_receipt(roots[0], "partial-a", [source(1), source(2)], "2026-10-27T17:02:58+00:00")
        index = self.base / "done-index.jsonl"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = M.main(["--roots", *map(str, roots), "--output", str(index), "--report"])
        self.assertEqual(code, 0)
        self.assertEqual(len(M.load_index(index)), 2)
        self.assertIn("| 代（lineage） | 收据数 |", stdout.getvalue())
        queue = self.base / "queue.json"
        queue.write_text(json.dumps({"sources": [source(1), source(3)]}), encoding="utf-8")
        out = self.base / "filtered.json"
        receipt = self.base / "receipt.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = M.main(["--index", str(index), "--roots", *map(str, roots),
                           "--filter-queue", str(queue), "--out", str(out),
                           "--check-files", "--receipt", str(receipt)])
        self.assertEqual(code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual((summary["removed"], summary["remaining"]), (0, 2))
        self.assertEqual(summary["missing_sidecar_kept"], 1)
        self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["queue_items"], 2)

    # 12. --report 可以直接读已写出的 done-index（收据数显示 -，跨代重叠不可还原）。
    def test_report_from_index(self):
        root = make_run(self.base, "generation-a")
        put_receipt(root, "partial-a", [source(1), source(2)], "2026-10-27T17:02:58+00:00")
        index = self.base / "done-index.jsonl"
        M.write_index(index, M.scan(M.specs_from_roots([root]), progress_every=0).rows)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = M.main(["--index", str(index), "--report"])
        self.assertEqual(code, 0)
        report = stdout.getvalue()
        self.assertIn("唯一源（去重后）：**2**", report)
        self.assertIn("输入是已去重的 done-index", report)


if __name__ == "__main__":
    unittest.main()
