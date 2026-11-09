#!/usr/bin/env python3
"""Read-only audit of published annotation-tree generations.

Samples at least ``--per-source`` sidecars from each source directory and
reports the format axes that actually differ between the four historical
producers (schema, commit, path layout, decision set, clock profile, encoding,
modes).  Never writes to, repairs, or re-commits a label tree.

Usage::

    python3 tools/open_annotation/audit_annotation_formats.py \
        --root <labels>/<version> --per-source 20
    python3 tools/open_annotation/audit_annotation_formats.py \
        --root <labels>/<version> --queue <run>/queue
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_format import (  # noqa: E402
    FormatError, canonical_bytes, receipt_family, sidecar_profile, verify_commit,
)

SOURCES = ("cnceleb2", "voxblink2", "voxceleb2")


def _sample(root, source, per_source):
    base = Path(root) / source
    dirs = sorted(path for path in base.iterdir() if path.is_dir())
    if not dirs:
        return []
    if len(dirs) <= per_source:
        chosen = dirs
    else:
        chosen = [dirs[round(i * (len(dirs) - 1) / (per_source - 1))]
                  for i in range(per_source)] if per_source > 1 else [dirs[0]]
    sampled = []
    for directory in sorted(set(chosen)):
        files = sorted(path for path in directory.rglob("*.annotation.json") if path.is_file())
        sampled.extend(files[:2])
    return sampled


def audit_tree(root, per_source):
    root = Path(root)
    result = {"root": str(root), "identity": {}, "sources": {}, "errors": []}
    identity_path = root / "identity.json"
    if not identity_path.is_file():
        result["errors"].append({"path": str(identity_path), "error": "identity.json missing"})
        return result
    raw = identity_path.read_bytes()
    identity = json.loads(raw)
    result["identity"] = {
        "value": identity,
        "mode": oct(identity_path.stat().st_mode & 0o777),
        "canonical": raw == canonical_bytes(identity),
    }
    for source in SOURCES:
        base = root / source
        if not base.is_dir():
            result["sources"][source] = {"error": "source directory missing"}
            continue
        entries = collections.Counter()
        profiles = collections.Counter()
        decisions = collections.Counter()
        depths = collections.Counter()
        modes = collections.Counter()
        vad_variants = collections.Counter()
        runtime_variants = collections.Counter()
        sizes = []
        files = _sample(root, source, per_source)
        for path in files:
            relative = str(path.relative_to(root))
            parts = relative.split("/")
            entries["files"] += 1
            depths[len(parts)] += 1
            modes[oct(path.stat().st_mode & 0o777)] += 1
            sizes.append(path.stat().st_size)
            record = json.loads(path.read_bytes())
            try:
                profile = sidecar_profile(record)
                profiles[profile["profile"]] += 1
            except FormatError as exc:
                result["errors"].append({"path": relative, "error": str(exc)})
                continue
            decisions[record["decision"]["state"]] += 1
            if record["source"].get("sidecar_relpath") != relative:
                result["errors"].append({"path": relative, "error": "source.sidecar_relpath differs from disk path"})
            if record["source"].get("dataset") != source:
                result["errors"].append({"path": relative, "error": "source.dataset differs from top directory"})
            if not str(record["source"].get("source_id", "")).startswith("src_"):
                result["errors"].append({"path": relative, "error": "source_id is not src_<sha256>"})
            commit_path = path.with_name(path.name + ".commit.json")
            if not commit_path.is_file():
                result["errors"].append({"path": relative, "error": "commit sidecar missing"})
                entries["commit_missing"] += 1
            else:
                entries["commit_present"] += 1
                try:
                    verify_commit(
                        json.loads(commit_path.read_bytes()),
                        sidecar_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        sidecar_bytes=path.stat().st_size,
                        source_id=record["source"]["source_id"],
                        contract_sha256=identity["contract_sha256"],
                    )
                    entries["commit_matches"] += 1
                except FormatError as exc:
                    result["errors"].append({"path": relative, "error": "commit: " + str(exc)})
            entries["canonical" if path.read_bytes() == canonical_bytes(record) else "noncanonical"] += 1
            vad = record.get("vad", {})
            vad_variants[(vad.get("status"), "metadata_sha256" in vad)] += 1
            runtime_variants[tuple(sorted((record.get("provenance", {}).get("runtime") or {}).keys()))] += 1
        result["sources"][source] = {
            "sampled": len(files),
            "entries": dict(entries),
            "profiles": dict(profiles),
            "decisions": dict(decisions),
            "depths": dict(depths),
            "modes": dict(modes),
            "vad_variants": {str(key): value for key, value in vad_variants.items()},
            "runtime_variants": len(runtime_variants),
            "size_min": min(sizes) if sizes else None,
            "size_median": int(statistics.median(sizes)) if sizes else None,
            "size_max": max(sizes) if sizes else None,
        }
    return result


def audit_queue(queue_root, limit=None):
    queue = Path(queue_root)
    if (queue / "queue").is_dir():
        queue = queue / "queue"
    completed = queue / "completed"
    result = {"queue": str(queue), "files": 0, "families": {}, "sources": 0,
              "decisions": {}, "source_fields": {}, "sizes": [], "errors": []}
    entries = sorted(completed.glob("*.json")) if completed.is_dir() else []
    if limit:
        entries = entries[:limit]
    for path in entries:
        try:
            receipt = json.loads(path.read_text())
            family = receipt_family(receipt)
        except FormatError as exc:
            result["errors"].append({"path": str(path), "error": str(exc)})
            continue
        result["files"] += 1
        result["families"][family] = result["families"].get(family, 0) + 1
        result["sources"] += int(receipt.get("source_count") or 0)
        for source in receipt.get("sources", []):
            state = source.get("decision")
            result["decisions"][state] = result["decisions"].get(state, 0) + 1
            signature = tuple(sorted(source))
            result["source_fields"][str(signature)] = result["source_fields"].get(str(signature), 0) + 1
            if isinstance(source.get("sidecar_bytes"), int):
                result["sizes"].append(source["sidecar_bytes"])
    if result["sizes"]:
        sizes = sorted(result["sizes"])
        result["sidecar_bytes"] = {
            "count": len(sizes), "median": int(statistics.median(sizes)),
            "p95": sizes[min(len(sizes) - 1, int(len(sizes) * 0.95))], "max": sizes[-1],
        }
    result.pop("sizes")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="published labels/<version> directory")
    parser.add_argument("--per-source", type=int, default=20, help="speaker dirs sampled per source (default 20)")
    parser.add_argument("--queue", help="optional run or queue directory to census completion receipts")
    parser.add_argument("--json", help="write the full audit result to this path")
    args = parser.parse_args(argv)
    report = {"tree": audit_tree(args.root, args.per_source)}
    if args.queue:
        report["queue"] = audit_queue(args.queue)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
