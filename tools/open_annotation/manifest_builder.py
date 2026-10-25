"""Stream declared source lists into bounded disk buckets and immutable shards.

No model/CUDA import, source audio scan, or source-side write. Original buckets
are retained as audit/recovery evidence. A final index is published only after
all declared text inputs were fully hashed and each shard was closed.
"""
from __future__ import annotations
import argparse
import base64
import collections
import datetime
import hashlib
import heapq
import json
import os
from pathlib import Path
import signal
import time
from typing import Iterator

from source_registry import (DATA, DATASET_ROOTS, SCHEMA, VERSION, classify_path,
    from_training_row, merge_records, normalise_record, path_spelling, source_id)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""): h.update(block)
    return h.hexdigest()


class TextInputs:
    def __init__(self, limit_mib_s=30):
        self.proofs = []
        self.limit = limit_mib_s * 1048576
        self.started = time.monotonic()
        self.total = 0

    def lines(self, path):
        p = Path(path); before = p.stat(); h = hashlib.sha256(); count = 0
        with p.open("rb") as f:
            for count, raw in enumerate(f, 1):
                h.update(raw); self.total += len(raw)
                if count % 4096 == 0 and self.limit:
                    delay = self.total / self.limit - (time.monotonic() - self.started)
                    if delay > 0: time.sleep(min(delay, 1.0))
                yield count, raw
        after = p.stat()
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed during read: " + str(p)
        self.proofs.append({"path": str(p), "sha256": h.hexdigest(), "bytes": before.st_size,
            "lines": count, "mtime_ns": before.st_mtime_ns, "inode": before.st_ino, "stable": True})


class DirectoryResolver:
    """Only conditional small evaluation maps use directory-name caching."""
    def __init__(self): self.names = {}; self.failures = {}
    def resolve(self, candidates):
        for raw in candidates:
            p = Path(raw)
            if str(p.parent) not in self.names:
                try:
                    with os.scandir(p.parent) as it:
                        self.names[str(p.parent)] = {x.name for x in it if x.is_file()}
                except (FileNotFoundError, NotADirectoryError):
                    self.names[str(p.parent)] = set()
            if p.name in self.names[str(p.parent)]: return str(p)
        return None


def training_records(spec, inputs):
    for line, raw in inputs.lines(spec["path"]):
        if not raw.strip(): continue
        try:
            row = json.loads(raw)
            ds, rel = classify_path(row["wav"])
            if ds not in {"voxceleb2", "voxblink2", "cnceleb2"}: raise ValueError("not a declared training corpus")
            # The compact staging form avoids writing duplicated verbose records.
            yield source_id(ds, rel), {"training": row, "role": spec["role"]}
        except Exception as exc:
            rid = hashlib.sha256(spec["path"].encode()).hexdigest()
            r = normalise_record({"dataset": "invalid-input", "relative_path": f"{rid}/line{line}.invalid",
                "audio_path": None, "roles": [spec["role"]], "protection": {"training_forbidden": True, "reasons": ["invalid_input_row"]},
                "input_error": {"path": spec["path"], "line": line, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_base64": base64.b64encode(raw).decode(), "exception": repr(exc)}})
            yield r["source_id"], {"source": r}


def endpoint_record(dataset, relative, audio_path, role, **extra):
    return normalise_record({"dataset": dataset, "relative_path": relative, "audio_path": audio_path,
        "roles": [role], "original_splits": ["provider_evaluation"], **extra})


def protocol_records(spec, inputs, resolver):
    name = spec["protocol"]; role = "evaluation:" + name; keys = set()
    if name.startswith("Vox1-"):
        for line, raw in inputs.lines(spec["path"]):
            if not raw.strip(): continue
            parts = raw.decode("utf-8-sig").strip().split(" ")
            if len(parts) != 3 or any(not v for v in parts): raise ValueError(f"{name} row {line} differs from the frozen single-space parser")
            int(parts[0]); keys.update(parts[1:])
        for key in sorted(keys):
            yield endpoint_record("voxceleb1", key, DATA + "/voxceleb1/wav/" + key, role)
    elif name == "VOICES":
        enrollment = {}
        for _, raw in inputs.lines(spec["enroll_map"]):
            if raw.strip():
                speaker, path = raw.decode().split(); enrollment[speaker] = Path(path).name
        skipped = 0
        for _, raw in inputs.lines(spec["path"]):
            parts = raw.decode().split()
            if len(parts) < 3: skipped += 1; continue
            a, b, label = parts[:3]
            if label not in {"target", "1", "tgt", "nontarget", "0", "imp", "nontgt"}:
                try: int(float(label))
                except ValueError: skipped += 1; continue
            keys.update((enrollment.get(a, a + ".wav"), Path(b).name))
        spec["parser_skipped_rows"] = skipped
        for key in sorted(keys):
            yield endpoint_record("voices2019", key, DATASET_ROOTS["voices2019"][0] + "/" + key, role)
    elif name == "CN-Celeb":
        groups = {}; members = set(); tests = set(); used_groups = set()
        for _, raw in inputs.lines(spec["enroll_map"]):
            parts = raw.decode().split()
            if len(parts) < 2: continue
            if parts[0] in groups: raise ValueError("duplicate CN enrollment group")
            groups[parts[0]] = parts[1:]; members.update(parts[1:])
        for _, raw in inputs.lines(spec["path"]):
            parts = raw.decode().split()
            if len(parts) < 3: continue
            e, t, label = parts[:3]
            if label not in {"target", "nontarget"}: int(label)
            used_groups.add(e)
            key = "eval/test/" + Path(t).name if not t.startswith("eval/") else t
            if t.startswith("test/"): key = "eval/" + t
            tests.add(key)
        if any(e not in groups for e in used_groups): raise ValueError("CN trial references an unknown enrollment group")
        root = DATASET_ROOTS["cnceleb1"][0]
        for key in sorted(members | tests):
            if key in members:
                rel = "data/" + key; path = root + "/" + rel; extra = {"enrollment_groups": sorted(g for g, values in groups.items() if key in values)}
            else:
                candidates = list(dict.fromkeys([root + "/" + key, root + "/data/" + key, root + "/eval/" + key,
                    root + "/eval/enroll/" + Path(key).name, root + "/eval/test/" + Path(key).name]))
                path = resolver.resolve(candidates); rel = str(Path(path).relative_to(root)) if path else key
                extra = {"resolver_candidates": candidates, "resolution_status": "directory_resolved_media_not_decoded" if path else "unresolved"}
            yield endpoint_record("cnceleb1", rel, path, role, **extra)
    else: raise ValueError("unknown declared protocol: " + name)
    spec["unique_endpoint_keys"] = len(keys) if keys else None


def wav_scp_records(spec, inputs):
    root = spec["audio_root"]; dataset = spec["dataset"]
    texts = {}; speakers = {}
    for field, target in [("text", texts), ("utt2spk", speakers)]:
        if spec.get(field):
            for line, raw in inputs.lines(spec[field]):
                parts = raw.decode("utf-8-sig").strip().split(maxsplit=1)
                if len(parts) == 2:
                    if parts[0] in target: raise ValueError(f"duplicate {field} key at line {line}")
                    target[parts[0]] = parts[1]
    for line, raw in inputs.lines(spec["path"]):
        if not raw.strip(): continue
        key, path = raw.decode("utf-8-sig").strip().split(maxsplit=1)
        if path.endswith("|"): raise ValueError("shell-pipe wav.scp is forbidden")
        path = path_spelling(path)
        if not path.startswith(root + "/"): raise ValueError("wav.scp source outside declared audio root")
        rel = path[len(root) + 1:]
        original = rel.split("/", 1)[0] if rel.split("/", 1)[0] in {"train", "dev", "test"} else "unspecified"
        role = spec.get("role", "supplemental:" + original)
        yield normalise_record({"dataset": dataset, "relative_path": rel, "audio_path": path,
            "roles": [role], "original_splits": [original], "consumer_keys": [key],
            "speaker_evidence": {"provider_label": speakers.get(key, Path(rel).parent.name), "verified_identity": False},
            "provider_transcript": {"text": texts.get(key), "kind": "provider_transcript_not_word_time_gold", "path": spec.get("text"), "key": key}})


def source_jsonl_records(spec, inputs):
    for _, raw in inputs.lines(spec["path"]):
        if raw.strip(): yield normalise_record(json.loads(raw))


def build(config, output, *, max_records=None):
    out = Path(output).resolve()
    if str(out).startswith(DATA): raise ValueError("output must be independent of original media")
    out.mkdir(parents=True, exist_ok=False); buckets = out / "source_buckets"; buckets.mkdir(); shards = out / "shards"; shards.mkdir()
    n = int(config.get("bucket_count", 256)); chunk_size = int(config.get("chunk_size", 256))
    if not 1 <= n <= 4096 or not 1 <= chunk_size <= 4096: raise ValueError("invalid bounded bucket/chunk count")
    inputs = TextInputs(config.get("read_limit_mib_s", 30)); resolver = DirectoryResolver(); started = time.monotonic()
    handles = [open(buckets / f"bucket-{i:04d}.jsonl", "xb", buffering=65536) for i in range(n)]
    total = 0; input_counts = []; counts = collections.Counter(); samples = {}; sample_limit = int(config.get("sample_pool_per_dataset", 2000))
    stage_samples = {}; stage_heaps = {}
    index = {"schema": "redimnet2-shard-index/v1", "format": "source-jsonl", "source_schema": SCHEMA,
        "version": VERSION, "source_count": 0, "shards": [], "input_proofs": [], "identity_scope": "declared dataset-relative source paths; no byte/PCM/person dedup claim"}
    try:
        for original_spec in config["inputs"]:
            spec = dict(original_spec); kind = spec["kind"]; before_count = total
            if kind == "training_list": iterator = training_records(spec, inputs)
            elif kind == "protocol": iterator = ((r["source_id"], {"source": r}) for r in protocol_records(spec, inputs, resolver))
            elif kind == "wav_scp": iterator = ((r["source_id"], {"source": r}) for r in wav_scp_records(spec, inputs))
            elif kind == "source_jsonl": iterator = ((r["source_id"], {"source": r}) for r in source_jsonl_records(spec, inputs))
            else: raise ValueError("unknown input kind: " + kind)
            for sid, value in iterator:
                slot = int(sid[4:12], 16) % n
                handles[slot].write(canonical(value) + b"\n"); total += 1
                if "training" in value: ds = classify_path(value["training"]["wav"])[0]
                else: ds = value["source"]["dataset"]
                pool = stage_samples.setdefault(ds, {}); heap = stage_heaps.setdefault(ds, []); priority = -int(sid[4:20], 16)
                if sid in pool:
                    candidate = from_training_row(value["training"], value["role"]) if "training" in value else value["source"]
                    pool[sid] = merge_records(pool[sid], candidate)
                elif len(heap) < sample_limit or priority > heap[0][0]:
                    if len(heap) >= sample_limit:
                        _, old_sid = heapq.heappop(heap); del pool[old_sid]
                    heapq.heappush(heap, (priority, sid))
                    pool[sid] = from_training_row(value["training"], value["role"]) if "training" in value else value["source"]
                if total % 100000 == 0: print(json.dumps({"phase": "register_text", "references": total, "elapsed_s": time.monotonic()-started}), flush=True)
                if max_records and total >= max_records: raise ValueError("max_records reached: intentionally incomplete registry, no final index")
            input_counts.append({"spec": spec, "registration_references": total-before_count})
    finally:
        for h in handles: h.close()
    with (out/"representative_pool_after_text_pass.jsonl").open("x") as f:
        for ds in sorted(stage_samples):
            for sid in sorted(stage_samples[ds]):f.write(canonical(stage_samples[ds][sid]).decode()+"\n")
    (out/"text_pass_complete.json").write_text(json.dumps({"registration_references":total,"input_proofs":inputs.proofs,
        "sample_scope":"deterministic lowest source hashes per dataset from every declared text row; source records sampled during the complete text pass; final index still pending merge",
        "sample_counts":{k:len(v) for k,v in stage_samples.items()}},ensure_ascii=False,indent=2)+"\n")
    for slot in range(n):
        merged = {}
        bucket_path = buckets / f"bucket-{slot:04d}.jsonl"
        with bucket_path.open("rb") as f:
            for raw in f:
                value = json.loads(raw); r = from_training_row(value["training"], value["role"]) if "training" in value else normalise_record(value["source"])
                sid = r["source_id"]
                merged[sid] = merge_records(merged[sid], r) if sid in merged else r
        chunk = []; chunk_id = 0
        def write_chunk(rows):
            nonlocal chunk_id
            chunk_name = f"source-{slot:04d}-{chunk_id:04d}"; p = shards / (chunk_name + ".jsonl")
            payload = b"".join(canonical(x)+b"\n" for x in rows)
            with p.open("xb") as stream: stream.write(payload)
            index["shards"].append({"id": chunk_name, "path": str(p.relative_to(out)), "sha256": hashlib.sha256(payload).hexdigest(), "count": len(rows), "duration_estimate_seconds": None})
            chunk_id += 1
        for sid in sorted(merged):
            r = merged[sid]; chunk.append(r); index["source_count"] += 1
            counts["dataset:"+r["dataset"]] += 1
            counts["resolution:"+r["resolution_status"]] += 1
            if r["protection"]["training_forbidden"]: counts["protected_sources"] += 1
            if r["split_conflicts"]: counts["source_conflicts"] += 1
            if "train_lm" in r["roles"] and "train_ptn" not in r["roles"]: counts["lm_sources_without_ptn"] += 1
            for role in r["roles"]: counts["role:"+role] += 1
            if r["audio_path"] is not None and r["dataset"] != "invalid-input":
                heap = samples.setdefault(r["dataset"], []); priority = -int(sid[4:20], 16)
                if len(heap) < sample_limit: heapq.heappush(heap, (priority, sid, r))
                elif priority > heap[0][0]: heapq.heapreplace(heap, (priority, sid, r))
            if len(chunk) == chunk_size: write_chunk(chunk); chunk = []
        if chunk: write_chunk(chunk)
        print(json.dumps({"phase": "merge_bucket", "bucket": slot, "source_count": index["source_count"], "shards": len(index["shards"]), "elapsed_s": time.monotonic()-started}), flush=True)
    index.update(input_proofs=inputs.proofs, registration_references=total, input_counts=input_counts,
        counts=dict(sorted(counts.items())), directory_resolver_cached_parents=len(resolver.names),
        source_media_stat_performed=False, full_media_sha_performed=False, model_calls=0,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
    index["registry_id"] = "registry_" + hashlib.sha256(canonical(index)).hexdigest()
    index["content_sha256"] = hashlib.sha256(canonical(index)).hexdigest()
    final = out/"index.json"; temp=out/".index.json.tmp"
    with temp.open("x") as f: json.dump(index,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp,final)
    final.with_suffix('.json.sha256').write_text(file_sha(final)+'  index.json\n')
    with (out/"representative_pool.jsonl").open("x") as f:
        for dataset in sorted(samples):
            for _, _, record in sorted(samples[dataset], key=lambda x: x[1]):f.write(canonical(record).decode()+"\n")
    summary = {k:v for k,v in index.items() if k != "shards"};summary["shard_count"] = len(index["shards"]); summary["index_file_sha256"] = file_sha(final)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"phase":"complete","source_count":index["source_count"],"shards":len(index["shards"]),"registry_id":index["registry_id"],"elapsed_s":time.monotonic()-started}),flush=True)
    return index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--max-wall-seconds",type=int,default=7200);a=p.parse_args()
    if a.max_wall_seconds <= 0: raise ValueError("wall bound must be positive")
    signal.alarm(a.max_wall_seconds)
    os.nice(10)
    build(json.loads(Path(a.config).read_text()),a.output)


if __name__ == "__main__": main()
