"""Fast protocol accounting, and an independent CPU-only full-sidecar audit."""
from __future__ import annotations
import argparse
import contextlib
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import traceback
from contracts import canonical_bytes, digest, job_key, need, read_contract, safe_join, sha_file, utcnow
from schema import assert_valid_record
from work_queue import SLOT_BYTES, decode_completion_slot, read_index


@contextlib.contextmanager
def unique_source_scope(contract, index):
    """有限pilot用严格上限的set；正式全量旧分支保持原SQLite行为。"""
    count=index.get('source_count')
    bounded=(contract.get('purpose')=='pilot_validation' and type(count)is int and 0<count<=4096)
    if bounded:
        seen=set()
        def add(value):
            need(value not in seen,'duplicate source ID across finite pilot shards')
            need(len(seen)<4096,'finite pilot source-ID memory limit exceeded')
            seen.add(value)
        yield add, lambda:len(seen), lambda:None, 'bounded_pilot_memory_set'
    else:
        with tempfile.TemporaryDirectory(prefix='redimnet2-full-sidecar-audit-')as temp:
            connection=sqlite3.connect(str(Path(temp)/'seen.sqlite'))
            try:
                connection.execute('CREATE TABLE seen(source_id TEXT PRIMARY KEY)')
                yield (lambda value:connection.execute('INSERT INTO seen VALUES (?)',(value,)),
                       lambda:connection.execute('SELECT COUNT(*) FROM seen').fetchone()[0],
                       connection.commit, 'legacy_local_sqlite')
            finally:
                connection.close()


def verify_run(contract, index, queue_root, output_root, full=False, *, bounded_parent_lock_fd=None, bounded_terminal_pin=None, bounded_task_id=None):
    """No models/CUDA imported. full=False is completion-protocol verification.

    The light mode reads one sequential fixed-slot file, not millions of labels.
    It deliberately does not claim fresh byte-level sidecar or audio verification.
    full=True reopens every immutable chunk, sidecar and sidecar commit and uses a
    bounded in-memory source-ID set for pilot_validation <=4096 sources; other
    historical full modes retain their temporary local SQLite unique index.
    """
    queue_root, output_root = Path(queue_root), Path(output_root)
    errors = []; counts = Counter(); completed_sources = 0; completed_shards = 0; error_count = 0
    def add_error(value):
        nonlocal error_count
        error_count += 1
        if len(errors) < 1000: errors.append(value)
    identity = json.loads((queue_root / "identity.json").read_text())
    contract_sha, index_sha = identity["contract_sha256"], identity["index_sha256"]
    if isinstance(contract, (str, Path)):
        contract = read_contract(contract, contract_sha)
    if isinstance(index, (str, Path)):
        from bounded_speed_runtime import index_kind
        if index_kind(index, index_sha) == 'annotation-bounded-source-index/v1':
            need(not full, 'bounded successor forbids full-library audit; select metadata verification')
            from bounded_speed_queue import verify_bounded_terminal_metadata
            tree = output_root / contract['version']
            need(json.loads((tree/'identity.json').read_text()) == {'schema':'redimnet2-sidecar-tree/v1',
                 'version':contract['version'],'contract_sha256':contract_sha}, 'bounded new sidecar tree identity differs')
            value = verify_bounded_terminal_metadata(index,index_sha,queue_root,contract_sha,
                parent_lock_fd=bounded_parent_lock_fd,terminal_pin=bounded_terminal_pin,task_id=bounded_task_id)
            return dict(value,passed=value['metadata_complete'],database_created=False,
                        source_count=value['progress']['source_count'],contract_sha256=contract_sha,index_sha256=index_sha,
                        full_sidecar_readback_verified=False,sidecars_individually_reopened=0)
        need(bounded_parent_lock_fd is None and bounded_terminal_pin is None and bounded_task_id is None,
             'v1 report refuses bounded-only terminal options')
        index_path = Path(index).resolve(); index = read_index(index_path, index_sha)
    else:
        need(bounded_parent_lock_fd is None and bounded_terminal_pin is None and bounded_task_id is None,
             'v1 report refuses bounded-only terminal options')
        index_path = None
        need(isinstance(index, dict) and index.get("schema") == "redimnet2-shard-index/v1", "wrong loaded index")
    need(contract["version"], "contract version missing")
    tree = output_root / contract["version"]
    need(json.loads((tree / "identity.json").read_text()) == {"schema": "redimnet2-sidecar-tree/v1", "version": contract["version"], "contract_sha256": contract_sha}, "sidecar tree identity differs")
    broker = json.loads((queue_root / "broker.json").read_text())
    completion = queue_root / "completion_map.bin"
    need(completion.stat().st_size == len(index["shards"]) * SLOT_BYTES, "completion map shape differs")
    with completion.open("rb") as stream:
        for shard in index["shards"]:
            try:
                value = decode_completion_slot(stream.read(SLOT_BYTES))
                need(value["shard_id"] == shard["id"] and value["shard_sha256"] == shard["sha256"]
                     and value["source_count"] == shard["count"] and value["contract_sha256"] == contract_sha
                     and value["index_sha256"] == index_sha, "completion slot does not bind pinned shard")
                need(len(value["source_ids_sha256"]) == len(value["commit_sha256"]) == 64, "completion digest missing")
                if "source_ids_sha256" in shard: need(value["source_ids_sha256"] == shard["source_ids_sha256"], "source-ID digest differs from enhanced index")
                need(set(value["decision_counts"]) == {"candidate", "unknown", "failed"}
                     and all(type(n) is int and n >= 0 for n in value["decision_counts"].values())
                     and sum(value["decision_counts"].values()) == shard["count"], "completion decision denominator differs")
                counts.update(value["decision_counts"]); completed_sources += shard["count"]; completed_shards += 1
            except Exception as exc:
                add_error({"shard_id": shard["id"], "stage": "completion_map", "error": str(exc)})
    if not (broker["completed_chunks"] == len(index["shards"]) and not broker["leases"]):
        add_error({"stage": "broker", "error": "queue not globally complete", "completed_chunks": broker["completed_chunks"], "leases": len(broker["leases"])})
    if completed_sources != index["source_count"]: add_error({"stage": "denominator", "completed": completed_sources, "planned": index["source_count"]})
    full_count = 0; sidecar_bytes = 0; full_errors = 0; full_decisions = Counter(); source_storage='not_used'
    if full:
        need(index_path is not None, "full CPU verification requires original index path")
        # 有限pilot不创建任何数据库，逐源readback及唯一性检查仍完整。
        with unique_source_scope(contract,index)as(add_seen,unique_count,commit_seen,source_storage):
            for ordinal, shard in enumerate(index["shards"]):
                try:
                    source_path = index_path.parent / shard["path"]
                    need(sha_file(source_path) == shard["sha256"], "source shard bytes changed")
                    with source_path.open() as stream: sources = [json.loads(line) for line in stream]
                    need(len(sources) == shard["count"], "source shard count differs")
                    p = queue_root / "completed" / (shard["id"] + ".json")
                    value = json.loads(p.read_text())
                    need(value.get("schema") in ("redimnet2-chunk-commit/v1", "redimnet2-bounded-chunk-commit/v1"),
                         "unsupported completion receipt schema %r; supported: redimnet2-chunk-commit/v1, redimnet2-bounded-chunk-commit/v1"
                         % (value.get("schema"),))
                    need(value["schema"] == "redimnet2-chunk-commit/v1",
                         "bounded-chunk-commit/v1 receipt cannot be verified by the v1 run report; use the bounded verifier")
                    need(value["shard_id"] == shard["id"]
                         and value["source_count"] == shard["count"]
                         and value["contract_sha256"] == contract_sha and value["index_sha256"] == index_sha
                         and value["shard_sha256"] == shard["sha256"], "chunk receipt identity differs")
                    receipts = {r["source_id"]: r for r in value["sources"]}
                    need(len(receipts) == len(value["sources"]) == len(sources)
                         and set(receipts) == {s["source_id"] for s in sources}, "chunk source union differs")
                    with completion.open("rb") as f:
                        f.seek(ordinal * SLOT_BYTES); slot = decode_completion_slot(f.read(SLOT_BYTES))
                    need(sha_file(p) == slot["commit_sha256"] and digest(sorted(receipts)) == slot["source_ids_sha256"], "chunk receipt is not bound to completion map")
                    chunk_decisions = Counter()
                    for source in sources:
                        add_seen(source["source_id"])
                        receipt = receipts[source["source_id"]]
                        need(receipt["job_key"] == job_key(source, contract_sha), "source job key differs")
                        sidecar = safe_join(tree, source["sidecar_relpath"])
                        need(not sidecar.is_symlink(), "sidecar symlink refused")
                        raw = sidecar.read_bytes(); record = json.loads(raw); assert_valid_record(record)
                        need(record["source"] == source and record["contract"]["sha256"] == contract_sha
                             and record["execution"]["job_key"] == receipt["job_key"], "sidecar input/version differs")
                        need(hashlib.sha256(raw).hexdigest() == receipt["sidecar_sha256"] and len(raw) == receipt["sidecar_bytes"], "sidecar bytes differ")
                        commit = sidecar.with_name(sidecar.name + ".commit.json")
                        need(sha_file(commit) == receipt["commit_sha256"], "per-source commit bytes differ")
                        descriptor = json.loads(commit.read_text())
                        expected_descriptor = {"schema":"redimnet2-sidecar-commit/v1", "source_id":source["source_id"],
                            "job_key":receipt["job_key"], "contract_sha256":contract_sha,
                            "source_record_sha256":digest(source), "sidecar_sha256":hashlib.sha256(raw).hexdigest(), "bytes":len(raw)}
                        need(descriptor == expected_descriptor, "per-source commit contract differs")
                        need(receipt["decision"] == record["decision"]["state"], "decision ledger differs")
                        chunk_decisions[record["decision"]["state"]] += 1
                        full_decisions[record["decision"]["state"]] += 1
                        full_count += 1; sidecar_bytes += len(raw)
                    need(slot["decision_counts"] == {key:chunk_decisions[key] for key in ("candidate","unknown","failed")},
                         "completion aggregate differs from actual source decisions")
                    commit_seen()
                except Exception:
                    full_errors += 1
                    add_error({"stage": "full_sidecars", "shard_id": shard["id"], "traceback": traceback.format_exc()})
                if full_count and full_count % 10000 < shard["count"]:
                    print(json.dumps({"event": "full_verification_progress", "sources": full_count, "errors": full_errors}), flush=True)
            unique = unique_count()
            if full_count != index["source_count"] or unique != index["source_count"]:
                add_error({"stage": "full_denominator", "verified": full_count, "unique": unique, "planned": index["source_count"]})
    return {"schema": "redimnet2-production-verification/v1", "passed": error_count == 0,
            "mode": "full_sidecars" if full else "completion_map_protocol", "source_count": index["source_count"],
            "completed_source_count": completed_sources, "completed_shard_count": completed_shards,
            "decision_counts": dict(counts), "errors": errors, "error_count": error_count, "error_examples_truncated": error_count > len(errors),
            "full_sidecar_decision_counts": {key:full_decisions[key] for key in ("candidate","unknown","failed")} if full else None,
            "full_sidecar_readback_verified": bool(full and error_count == 0), "sidecars_individually_reopened": full_count,
            "sidecar_bytes_reopened": sidecar_bytes, "input_audio_rehashed_in_this_audit": False,
            "source_ID_global_disjointness_rechecked": bool(full and error_count == 0),
            "source_disjointness_storage":source_storage,"database_created":source_storage=="legacy_local_sqlite",
            "shard_ownership_union_disjoint_verified": bool(error_count == 0),
            "light_mode_meaning": "immutable input-shard binding and producer publication protocol, not a fresh per-file integrity or boundary-accuracy audit",
            "contract_sha256": contract_sha, "index_sha256": index_sha, "gpu_model_execution": False,
            "verified_utc": utcnow()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True); p.add_argument("--shard-index", required=True)
    p.add_argument("--queue-root", required=True); p.add_argument("--output-root", required=True)
    p.add_argument("--full", action="store_true"); p.add_argument("--report", required=True)
    p.add_argument('--bounded-terminal-observation'); p.add_argument('--bounded-terminal-sha'); p.add_argument('--bounded-task-id')
    args = p.parse_args(argv)
    need(bool(args.bounded_terminal_observation)==bool(args.bounded_terminal_sha)==bool(args.bounded_task_id),
         'bounded external terminal path/SHA/task must be complete')
    terminal = {'path':args.bounded_terminal_observation,'sha256':args.bounded_terminal_sha} if args.bounded_terminal_observation else None
    result = verify_run(args.contract, args.shard_index, args.queue_root, args.output_root, full=args.full,
                        bounded_terminal_pin=terminal,bounded_task_id=args.bounded_task_id)
    from contracts import atomic_new
    atomic_new(args.report, result); print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
