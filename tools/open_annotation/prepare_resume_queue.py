"""按 done-index 过滤三训练域源分片，并生成新索引构建配置。

续标注的前置步骤：先建 done-index（已标注源），再把全量源分片逐行过滤成
"只含未标注源"的候选分片，最后交给 manifest_builder 重新分片成新索引。
三个步骤都只读输入；输出目录不存在才创建，逐行保留原字节。

本模块不启动 GPU、不加载模型、不写旧队列或旧侧车。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SOURCE_ID = re.compile(rb'"source_id"\s*:\s*"(src_[0-9a-f]+)"')
DONE_IDS = None
MISSING_IDS = frozenset()


class Refused(Exception):
    """输入不符合约定时显式失败，不做部分写入。"""


def need(condition, message):
    if not condition:
        raise Refused(message)


def load_done_index(path):
    """读取 done-index.jsonl，返回已标注 source_id 集合。"""
    path = Path(path)
    need(path.is_file(), f"done index not found: {path}")
    ids = set()
    with path.open("rb") as handle:
        for line in handle:
            match = SOURCE_ID.search(line)
            if match:
                ids.add(match.group(1))
            elif line.startswith(b"src_"):
                ids.add(line.strip())
    need(ids, f"done index has no source_id rows: {path}")
    return ids


def filter_one(pair):
    src, dst = pair
    kept = skipped = bad = 0
    with open(src, "rb") as source, open(dst, "xb", buffering=1024 * 1024) as target:
        for line in source:
            match = SOURCE_ID.search(line)
            if not match:
                bad += 1
                continue
            if match.group(1) in DONE_IDS and match.group(1) not in MISSING_IDS:
                skipped += 1
                continue
            if not line.endswith(b"\n"):
                line += b"\n"
            target.write(line)
            kept += 1
    return {"file": src.name, "kept": kept, "skipped": skipped, "bad": bad}


def filter_shards(shards_dir, done_index, out_dir, workers=32, limit=0, missing_index=None):
    global DONE_IDS, MISSING_IDS
    shards_dir = Path(shards_dir)
    out_dir = Path(out_dir)
    need(shards_dir.is_dir(), f"shards directory not found: {shards_dir}")
    need(not out_dir.exists(), f"output already exists: {out_dir}")
    started = time.monotonic()
    DONE_IDS = load_done_index(done_index)
    MISSING_IDS = load_done_index(missing_index) if missing_index else frozenset()
    loaded = time.monotonic() - started
    shards = sorted(shards_dir.glob("*.jsonl"))
    need(shards, f"no shard jsonl under {shards_dir}")
    if limit:
        shards = shards[:limit]
    out_dir.mkdir(parents=True)
    pairs = [(path, out_dir / path.name) for path in shards]
    stats = {"files": 0, "kept": 0, "skipped": 0, "bad": 0}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(filter_one, pairs, chunksize=8), 1):
            stats["files"] += 1
            for key in ("kept", "skipped", "bad"):
                stats[key] += result[key]
            if index % 2000 == 0:
                print(json.dumps({"phase": "filter", "files": index, "of": len(pairs), **{k: stats[k] for k in ("kept", "skipped")}}), flush=True)
    stats["total_in"] = stats["kept"] + stats["skipped"] + stats["bad"]
    need(stats["bad"] == 0, "some rows have no source_id; refusing")
    stats["done_ids"] = len(DONE_IDS)
    stats["missing_ids"] = len(MISSING_IDS)
    stats["elapsed_s"] = round(time.monotonic() - started, 1)
    stats["done_load_s"] = round(loaded, 1)
    (out_dir / "filter-stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    return stats


def build_config(candidates_dir, out_path, bucket_count=256, chunk_size=256, read_limit_mib_s=400):
    candidates_dir = Path(candidates_dir)
    need(candidates_dir.is_dir(), f"candidates directory not found: {candidates_dir}")
    files = sorted(p for p in candidates_dir.glob("*.jsonl") if p.stat().st_size > 0)
    need(files, f"no non-empty candidate shards under {candidates_dir}")
    config = {
        "schema": "redimnet2-manifest-build-config/v1",
        "bucket_count": bucket_count,
        "chunk_size": chunk_size,
        "read_limit_mib_s": read_limit_mib_s,
        "sample_pool_per_dataset": 2000,
        "inputs": [{"kind": "source_jsonl", "path": str(path)} for path in files],
    }
    Path(out_path).write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n")
    return {"inputs": len(files), "config": str(out_path)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shards", type=Path, help="源分片目录（catalog 的 *.jsonl）")
    result.add_argument("--done-index", type=Path, help="done-index.jsonl")
    result.add_argument("--out", type=Path, help="过滤输出目录（必须不存在）")
    result.add_argument("--missing", type=Path, help="有收据但侧车缺失的 source_id 清单（这些源要保留）")
    result.add_argument("--workers", type=int, default=32)
    result.add_argument("--limit", type=int, default=0, help="只处理前 N 个分片（冒烟）")
    result.add_argument("--config-out", type=Path, help="写出 manifest_builder 配置到这个路径")
    result.add_argument("--bucket-count", type=int, default=256)
    result.add_argument("--chunk-size", type=int, default=256)
    result.add_argument("--read-limit-mib-s", type=int, default=400)
    return result


def main():
    arguments = parser().parse_args()
    need(arguments.shards and arguments.done_index and arguments.out, "--shards/--done-index/--out are required")
    stats = filter_shards(arguments.shards, arguments.done_index, arguments.out, arguments.workers, arguments.limit, arguments.missing)
    print(json.dumps({"phase": "filter_done", **stats}, ensure_ascii=False), flush=True)
    if arguments.config_out:
        config = build_config(arguments.out, arguments.config_out, arguments.bucket_count, arguments.chunk_size, arguments.read_limit_mib_s)
        print(json.dumps({"phase": "config", **config}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
