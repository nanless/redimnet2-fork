"""标注完成索引（done-index）与待跑队列过滤，纯标准库。

以后跑标注之前先 build index，再 filter-queue
==============================================

这个工具只读各代 ``<run>/queue/completed/*.json`` 收据，不读音频、不读侧车正文、
不启动模型，也不改任何既有队列或判据。它回答一个问题：哪些 ``source_id``
已经有结果，下一批可以跳过。

1）建索引：扫各代收据，按 ``source_id`` 去重（保留 ``committed_utc`` 最新的一代），
   写出 ``done-index.jsonl`` 并打印 markdown 报表：

    /root/miniforge3/envs/USE/bin/python -B \\
      tools/open_annotation/annotation_inventory.py \\
      --registry /root/group-shared/voiceprint/artifacts/redimnet2/annotation-inventory/registry.json \\
      --output   /root/group-shared/voiceprint/artifacts/redimnet2/annotation-inventory/done-index.jsonl \\
      --report

2）跑新标注之前过滤待跑队列（命中 done-index 的源剔除）：

    /root/miniforge3/envs/USE/bin/python -B \\
      tools/open_annotation/annotation_inventory.py \\
      --index /root/group-shared/voiceprint/artifacts/redimnet2/annotation-inventory/done-index.jsonl \\
      --roots <每一代的 run 根，供 --check-files 定位 labels 树> \\
      --filter-queue /path/to/queue.json --out /path/to/filtered.json \\
      --check-files --check-max 5000

索引每行固定 7 个字段：``source_id``、``sidecar_relpath``、``sidecar_sha256``、
``sidecar_bytes``、``decision``、``lineage``、``committed_utc``；按 ``source_id``
排序，重复运行字节不变。``--check-files`` 只核对侧车是否存在（及大小是否与收据
一致），不重算 SHA；它只适合抽样或小批量，命中数超过 ``--check-max``（默认 5000）
时直接拒绝，避免对十万级文件逐个 stat。侧车缺失或大小不符的源**保留**在待跑队列
里，不会被误删。

队列文件支持 JSON（顶层为源列表，或 ``sources`` / ``items`` / ``records`` /
``queue`` / ``rows`` 之一为源列表）和 JSONL（每行一条源记录）。分片索引
（``shards`` 指向 JSONL 分片）不在本工具范围内，会显式报错。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile

SCHEMA = "annotation-done-index/v1"
FILTER_SCHEMA = "annotation-inventory-filter/v1"
INDEX_FIELDS = (
    "source_id",
    "sidecar_relpath",
    "sidecar_sha256",
    "sidecar_bytes",
    "decision",
    "lineage",
    "committed_utc",
)
DATASETS = ("cnceleb2", "voxceleb2", "voxblink2")
SOURCE_ID_RE = re.compile(r"src_[0-9a-f]{64}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
QUEUE_LIST_KEYS = ("sources", "items", "records", "queue", "rows", "tasks")


class InventoryError(RuntimeError):
    """输入、收据或索引不满足本工具的显式约定。"""


def need(condition, message):
    if not condition:
        raise InventoryError(message)


def parse_utc(value, where):
    need(isinstance(value, str) and value.strip(), f"{where}: committed_utc missing")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.datetime.fromisoformat(text)
    except ValueError as error:
        raise InventoryError(f"{where}: bad committed_utc {value!r}: {error}") from error
    need(moment.tzinfo is not None, f"{where}: committed_utc needs a timezone")
    return moment.astimezone(datetime.timezone.utc)


def iso(moment):
    return moment.astimezone(datetime.timezone.utc).isoformat()


def derive_lineage(root):
    """由 run 根推出代名：``.../<代的目录>/run-vN`` -> ``<代的目录>/run-vN``。"""
    root = Path(root)
    if root.name.startswith("run-") and root.parent.name:
        return f"{root.parent.name}/{root.name}"
    return root.name


@dataclass(frozen=True)
class RunSpec:
    root: Path
    lineage: str
    optional: bool = False


@dataclass
class LineageStats:
    lineage: str
    root: str
    receipts: int = 0
    rows: int = 0
    unique: int = 0
    exclusive: int | None = None
    first_utc: str | None = None
    last_utc: str | None = None
    datasets: Counter = field(default_factory=Counter)


@dataclass
class Inventory:
    rows: list
    lineages: list
    total_receipts: int
    total_rows: int
    masks: Counter
    pair_overlaps: dict
    datasets: Counter
    decisions: Counter
    from_index: bool = False

    @property
    def unique_sources(self):
        return len(self.rows)

    @property
    def overlap_sources(self):
        return sum(count for mask, count in self.masks.items() if mask.bit_count() > 1)

    @property
    def duplicate_rows(self):
        return self.total_rows - len(self.rows)


# ---------------------------------------------------------------------------
# 注册表与扫描


def load_registry(path):
    path = Path(path)
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise InventoryError(f"cannot read registry {path}: {error}") from error
    if isinstance(document, dict):
        entries = document.get("runs")
    elif isinstance(document, list):
        entries = document
    else:
        entries = None
    need(isinstance(entries, list) and entries, f"{path}: registry needs a non-empty 'runs' list")
    specs = []
    labels = {}
    roots = {}
    for entry in entries:
        if isinstance(entry, str):
            entry = {"path": entry}
        need(isinstance(entry, dict), f"{path}: registry entry must be an object")
        raw = entry.get("path")
        need(isinstance(raw, str) and raw, f"{path}: registry entry needs 'path'")
        root = Path(raw).expanduser()
        need(root.is_absolute(), f"{path}: registry path must be absolute: {raw}")
        root = root.resolve()
        need(root.is_dir(), f"{path}: registry path is not a directory: {root}")
        lineage = entry.get("lineage") or derive_lineage(root)
        need(isinstance(lineage, str) and lineage, f"{path}: bad lineage for {root}")
        optional = entry.get("optional", False)
        need(isinstance(optional, bool), f"{path}: 'optional' must be boolean for {root}")
        spec = RunSpec(root=root, lineage=lineage, optional=optional)
        if lineage in labels:
            need(labels[lineage] == root, f"{path}: lineage {lineage!r} points at two roots; "
                                          f"remove the stale copy or give an explicit distinct lineage")
            continue
        if root in roots and roots[root] != lineage:
            raise InventoryError(f"{path}: root {root} listed with two lineages; scan it once")
        labels[lineage] = root
        roots[root] = lineage
        specs.append(spec)
    return specs


def specs_from_roots(roots):
    specs = []
    for raw in roots:
        root = Path(raw).expanduser().resolve()
        need(root.is_dir(), f"root is not a directory: {root}")
        specs.append(RunSpec(root=root, lineage=derive_lineage(root)))
    labels = {}
    for spec in specs:
        need(spec.lineage not in labels, f"two --roots share lineage {spec.lineage!r}; pass --registry with explicit lineages")
        labels[spec.lineage] = spec.root
    return specs


def _validate_source(source, where):
    need(isinstance(source, dict), f"{where}: source record must be an object")
    source_id = source.get("source_id")
    need(isinstance(source_id, str) and SOURCE_ID_RE.match(source_id), f"{where}: bad source_id")
    relpath = source.get("sidecar_relpath")
    need(isinstance(relpath, str) and relpath, f"{where}: sidecar_relpath missing for {source_id}")
    need("\x00" not in relpath and "\\" not in relpath, f"{where}: bad sidecar_relpath for {source_id}")
    pure = PurePosixPath(relpath)
    need(not pure.is_absolute() and ".." not in pure.parts and len(pure.parts) >= 2,
         f"{where}: sidecar_relpath must be a safe relative path: {relpath!r}")
    digest = source.get("sidecar_sha256")
    need(isinstance(digest, str) and SHA256_RE.match(digest), f"{where}: bad sidecar_sha256 for {source_id}")
    size = source.get("sidecar_bytes")
    need(type(size) is int and size >= 0, f"{where}: bad sidecar_bytes for {source_id}")
    decision = source.get("decision")
    need(isinstance(decision, str) and decision, f"{where}: bad decision for {source_id}")
    return source_id, relpath, digest, size, decision


def _receipt_sources(path):
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise InventoryError(f"cannot read receipt {path}: {error}") from error
    need(isinstance(document, dict), f"{path}: receipt must be an object")
    sources = document.get("sources")
    need(isinstance(sources, list), f"{path}: receipt has no 'sources' list")
    count = document.get("source_count")
    need(type(count) is int and count == len(sources),
         f"{path}: source_count {count!r} != {len(sources)} receipt rows")
    committed = document.get("committed_utc")
    moment = parse_utc(committed, str(path))
    return sources, committed, moment


def _dataset(relpath):
    head = relpath.split("/", 1)[0]
    return head if head in DATASETS else "other"


def scan(specs, progress_every=2000):
    """扫描所有 run 根；返回按 source_id 去重的 Inventory。"""
    index = {}
    stats = []
    for spec in specs:
        completed = spec.root / "queue" / "completed"
        if not completed.is_dir():
            need(spec.optional, f"queue/completed not found under {spec.root}; "
                                f"mark the run 'optional' only if it truly produced no receipts")
        stats.append(LineageStats(lineage=spec.lineage, root=str(spec.root)))
    total_receipts = 0
    seen_receipts = 0
    for position, spec in enumerate(specs):
        completed = spec.root / "queue" / "completed"
        if not completed.is_dir():
            continue
        acc = stats[position]
        acc._first_moment = None
        acc._last_moment = None
        bit = 1 << position
        for path in sorted(completed.glob("*.json")):
            sources, committed, moment = _receipt_sources(path)
            total_receipts += 1
            acc.receipts += 1
            acc.rows += len(sources)
            if acc.first_utc is None or moment < acc._first_moment:
                acc.first_utc, acc._first_moment = committed, moment
            if acc.last_utc is None or moment > acc._last_moment:
                acc.last_utc, acc._last_moment = committed, moment
            for source in sources:
                source_id, relpath, digest, size, decision = _validate_source(source, str(path))
                acc.datasets[_dataset(relpath)] += 1
                row = {
                    "source_id": source_id,
                    "sidecar_relpath": relpath,
                    "sidecar_sha256": digest,
                    "sidecar_bytes": size,
                    "decision": decision,
                    "lineage": spec.lineage,
                    "committed_utc": committed,
                }
                current = index.get(source_id)
                if current is None:
                    index[source_id] = [moment, row, bit]
                else:
                    current[2] |= bit
                    if (moment, spec.lineage, digest, relpath) > (
                        current[0],
                        current[1]["lineage"],
                        current[1]["sidecar_sha256"],
                        current[1]["sidecar_relpath"],
                    ):
                        current[0] = moment
                        current[1] = row
            seen_receipts += 1
            if progress_every and seen_receipts % progress_every == 0:
                print(f"scanned {seen_receipts} receipts / {len(index)} unique sources",
                      file=sys.stderr, flush=True)
    rows = [index[source_id][1] for source_id in sorted(index)]
    masks = Counter(entry[2] for entry in index.values())
    per_lineage_unique = [0] * len(specs)
    for mask, count in masks.items():
        for position in range(len(specs)):
            if mask >> position & 1:
                per_lineage_unique[position] += count
    for position, state in enumerate(stats):
        state.unique = per_lineage_unique[position]
        state.exclusive = masks.get(1 << position, 0)
    pair_overlaps = Counter()
    labels = [spec.lineage for spec in specs]
    for mask, count in masks.items():
        chosen = [position for position in range(len(specs)) if mask >> position & 1]
        for left in range(len(chosen)):
            for right in range(left + 1, len(chosen)):
                pair_overlaps[(labels[chosen[left]], labels[chosen[right]])] += count
    datasets = Counter(_dataset(row["sidecar_relpath"]) for row in rows)
    decisions = Counter(row["decision"] for row in rows)
    return Inventory(
        rows=rows,
        lineages=stats,
        total_receipts=total_receipts,
        total_rows=sum(state.rows for state in stats),
        masks=masks,
        pair_overlaps=dict(pair_overlaps),
        datasets=datasets,
        decisions=decisions,
    )


def inventory_from_rows(rows):
    """从已去重的 done-index 行重建报表（无法还原跨代重叠）。"""
    counts = Counter(row["lineage"] for row in rows)
    first = {}
    last = {}
    datasets = Counter()
    decisions = Counter()
    for row in rows:
        lineage = row["lineage"]
        moment = parse_utc(row["committed_utc"], f"index row {row['source_id']}")
        if lineage not in first or moment < first[lineage]:
            first[lineage] = moment
        if lineage not in last or moment > last[lineage]:
            last[lineage] = moment
        datasets[_dataset(row["sidecar_relpath"])] += 1
        decisions[row["decision"]] += 1
    lineages = [
        LineageStats(lineage=lineage, root="", receipts=0, rows=count, unique=count,
                     first_utc=iso(first[lineage]), last_utc=iso(last[lineage]))
        for lineage, count in sorted(counts.items())
    ]
    return Inventory(
        rows=rows,
        lineages=lineages,
        total_receipts=0,
        total_rows=len(rows),
        masks=Counter(),
        pair_overlaps={},
        datasets=datasets,
        decisions=decisions,
        from_index=True,
    )


# ---------------------------------------------------------------------------
# 索引读写


def serialize_rows(rows):
    pieces = []
    for row in rows:
        record = {name: row[name] for name in INDEX_FIELDS}
        pieces.append(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if not pieces:
        return b""
    return ("\n".join(pieces) + "\n").encode("utf-8")


def write_atomic(path, payload):
    """原子写；内容相同则不落盘，返回是否真的改了文件。"""
    path = Path(path)
    need(not path.is_symlink(), f"refusing symlink output: {path}")
    if path.exists():
        need(path.is_file(), f"output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return False
    need(path.parent.is_dir(), f"output parent directory missing: {path.parent}")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return True


def write_index(path, rows):
    return write_atomic(path, serialize_rows(rows))


def load_index(path):
    path = Path(path)
    need(path.is_file(), f"done-index not found: {path}")
    rows = []
    seen = set()
    ordered = True
    previous = None
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise InventoryError(f"{path}:{number}: bad JSON: {error}") from error
            need(isinstance(row, dict), f"{path}:{number}: row must be an object")
            missing = [name for name in INDEX_FIELDS if name not in row]
            need(not missing, f"{path}:{number}: missing fields {missing}")
            normalized = {name: row[name] for name in INDEX_FIELDS}
            _validate_source(normalized, f"{path}:{number}")
            parse_utc(normalized["committed_utc"], f"{path}:{number}")
            need(normalized["source_id"] not in seen, f"{path}:{number}: duplicate source_id")
            seen.add(normalized["source_id"])
            if previous is not None and normalized["source_id"] < previous:
                ordered = False
            previous = normalized["source_id"]
            rows.append(normalized)
    if not ordered:
        rows.sort(key=lambda row: row["source_id"])
    return rows


# ---------------------------------------------------------------------------
# 报表


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def format_report(inventory):
    lines = ["## 已完成标注索引", ""]
    if inventory.from_index:
        lines.append("输入是已去重的 done-index：收据数、跨代重叠无法从这里还原（显示为 -）。")
        lines.append("")
    receipts = f"{inventory.total_receipts:,}" if inventory.total_receipts else "-"
    lines.append(
        f"唯一源（去重后）：**{inventory.unique_sources:,}**；"
        f"源行合计 {inventory.total_rows:,}；"
        f"收据合计 {receipts}。"
    )
    lines.append("")
    rows = []
    for state in inventory.lineages:
        rows.append([
            state.lineage,
            f"{state.receipts:,}" if state.receipts else "-",
            f"{state.rows:,}",
            f"{state.unique:,}",
            "-" if state.exclusive is None else f"{state.exclusive:,}",
            state.first_utc or "-",
            state.last_utc or "-",
        ])
    rows.append([
        "**合计（去重后）**",
        f"**{inventory.total_receipts:,}**" if inventory.total_receipts else "-",
        f"**{inventory.total_rows:,}**",
        f"**{inventory.unique_sources:,}**",
        "-",
        "",
        "",
    ])
    lines.extend(_table(("代（lineage）", "收据数", "源行数", "代内唯一源", "独占源", "首次提交（UTC）", "末次提交（UTC）"), rows))
    lines.append("")
    lines.append("独占源 = 只出现在这一代、没有跨代重叠的源；跨代重叠的源不计入任何一代的独占源，"
                 "所以各代独占源之和小于去重后唯一源。")
    if not inventory.from_index:
        lines.append("")
        lines.append(
            f"跨代重叠：**{inventory.overlap_sources:,}** 个源出现在 2 代及以上；"
            f"重复行 {inventory.duplicate_rows:,}（含同代重复）。"
        )
        if inventory.pair_overlaps:
            lines.append("")
            pairs = [
                (left, right, count)
                for (left, right), count in sorted(inventory.pair_overlaps.items(), key=lambda item: (-item[1], item[0]))
            ]
            lines.extend(_table(("代 A", "代 B", "重叠源"), pairs))
    lines.append("")
    lines.append("按数据集分布（去重后唯一源）：")
    lines.append("")
    total = max(inventory.unique_sources, 1)
    dataset_names = list(DATASETS) + (["other"] if inventory.datasets.get("other") else [])
    dataset_rows = [
        (name, f"{inventory.datasets.get(name, 0):,}",
         f"{inventory.datasets.get(name, 0) * 100.0 / total:.2f}%")
        for name in dataset_names
    ]
    lines.extend(_table(("数据集", "唯一源", "占比"), dataset_rows))
    if inventory.decisions:
        decisions = ", ".join(f"{name}={count:,}" for name, count in sorted(inventory.decisions.items()))
        lines.extend(["", f"decision 分布（去重后）：{decisions}。（这里只记账，不改判据。）"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 待跑队列


@dataclass
class QueueDocument:
    path: Path
    kind: str
    data: object
    items: list
    key: str | None
    lines: list | None


def load_queue(path):
    path = Path(path)
    need(path.is_file(), f"queue not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except ValueError:
        lines = [line for line in text.splitlines() if line.strip()]
        need(lines, f"{path}: not JSON and not JSONL")
        items = []
        for number, line in enumerate(lines, 1):
            try:
                items.append(json.loads(line))
            except ValueError as error:
                raise InventoryError(f"{path}:{number}: bad JSONL: {error}") from error
        document, key = None, None
        kind = "jsonl"
        raw_lines = lines
    else:
        kind = "json"
        raw_lines = None
        if isinstance(document, list):
            items, key = document, None
        elif isinstance(document, dict):
            key = next((name for name in QUEUE_LIST_KEYS if isinstance(document.get(name), list)), None)
            need(key is not None,
                 f"{path}: queue JSON needs a source list under one of {QUEUE_LIST_KEYS}; "
                 f"shard-index files (with 'shards') are not supported here")
            items = document[key]
        else:
            items, key = None, None
        need(isinstance(items, list), f"{path}: queue source list missing")
    for number, item in enumerate(items, 1):
        need(isinstance(item, dict), f"{path}: queue item {number} must be an object")
        need(isinstance(item.get("source_id"), str) and item["source_id"],
             f"{path}: queue item {number} has no source_id")
    return QueueDocument(path=path, kind=kind, data=document, items=items, key=key, lines=raw_lines)


def _check_sidecar(row, labels_dirs):
    relpath = row["sidecar_relpath"]
    expected = row["sidecar_bytes"]
    mismatched = None
    for base in labels_dirs:
        candidate = base / relpath
        try:
            info = candidate.stat()
        except FileNotFoundError:
            continue
        except OSError as error:
            return "error", f"{candidate}: {error}"
        if not stat.S_ISREG(info.st_mode):
            continue
        if info.st_size == expected:
            return "ok", str(candidate)
        if mismatched is None:
            mismatched = (str(candidate), info.st_size)
    if mismatched is not None:
        return "size-mismatch", f"{mismatched[0]}: {mismatched[1]} != {expected}"
    return "missing", None


def labels_dirs(specs):
    found = []
    for spec in specs:
        base = spec.root / "labels"
        if not base.is_dir():
            print(f"warning: no labels/ tree under {spec.root}; --check-files cannot see it",
                  file=sys.stderr)
            continue
        for profile in sorted(base.iterdir()):
            if profile.is_dir() and not profile.name.startswith("."):
                found.append(profile)
    unique = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def filter_queue_file(path, out_path, done, labels=None, check_files=False, check_max=5000):
    document = load_queue(path)
    out_path = Path(out_path).resolve()
    need(out_path != document.path.resolve(), f"refusing to overwrite the input queue: {out_path}")
    items = document.items
    hits = [item for item in items if item["source_id"] in done]
    if check_files:
        need(labels, "--check-files needs at least one labels/ tree (pass --roots; with --index "
                     "also pass the run roots)")
        need(check_max == 0 or len(hits) <= check_max,
             f"--check-files would stat {len(hits)} sidecars; limit is {check_max}. "
             f"Filter a sampled queue or pass --check-max 0 explicitly")
    kept = []
    removed = 0
    missing = []
    mismatched = []
    errors = []
    for position, item in enumerate(items):
        source_id = item["source_id"]
        row = done.get(source_id)
        if row is None:
            kept.append(position)
            continue
        if not check_files:
            removed += 1
            continue
        status, detail = _check_sidecar(row, labels)
        if status == "ok":
            removed += 1
            continue
        kept.append(position)
        record = {"source_id": source_id, "sidecar_relpath": row["sidecar_relpath"]}
        if status == "missing":
            missing.append(record)
        elif status == "size-mismatch":
            record["detail"] = detail
            mismatched.append(record)
        else:
            record["detail"] = detail
            errors.append(record)
    if document.kind == "jsonl":
        kept_items = [document.items[position] for position in kept]
        kept_lines = [document.lines[position] for position in kept]
        payload = ("\n".join(kept_lines) + "\n").encode("utf-8") if kept_lines else b""
    else:
        kept_items = [items[position] for position in kept]
        if document.key is None:
            output_document = kept_items
        else:
            output_document = dict(document.data)
            output_document[document.key] = kept_items
            for count_key in ("source_count", "count"):
                if document.data.get(count_key) == len(items):
                    output_document[count_key] = len(kept_items)
        payload = (json.dumps(output_document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_atomic(out_path, payload)
    summary = {
        "schema": FILTER_SCHEMA,
        "queue": str(document.path),
        "out": str(out_path),
        "queue_items": len(items),
        "removed": removed,
        "remaining": len(kept_items),
        "check_files": bool(check_files),
        "sidecars_checked": len(hits) if check_files else 0,
        "missing_sidecar_kept": len(missing),
        "size_mismatch_kept": len(mismatched),
        "check_error_kept": len(errors),
        "done_index_unique": len(done),
        "missing_examples": missing[:20],
        "size_mismatch_examples": mismatched[:20],
        "check_error_examples": errors[:20],
    }
    return summary


# ---------------------------------------------------------------------------
# 入口


def parser():
    result = argparse.ArgumentParser(
        description="Build a done-index from annotation commit receipts and filter a pending queue.",
    )
    result.add_argument("--roots", nargs="+", type=Path,
                        help="run roots; each must contain queue/completed/*.json")
    result.add_argument("--registry", type=Path,
                        help="JSON registry: {\"runs\":[{\"path\":...,\"lineage\":...}]}")
    result.add_argument("--index", type=Path, dest="index_path", help="existing done-index.jsonl")
    result.add_argument("--output", "-o", type=Path, help="write done-index.jsonl (only when building)")
    result.add_argument("--report", action="store_true", help="print the markdown report")
    result.add_argument("--filter-queue", type=Path, help="pending queue JSON/JSONL to filter")
    result.add_argument("--out", type=Path, help="filtered queue output path")
    result.add_argument("--check-files", action="store_true",
                        help="verify each hit sidecar exists (and size matches); keeps missing ones")
    result.add_argument("--check-max", type=int, default=5000, metavar="N",
                        help="refuse --check-files when more than N hits (0 = no limit; default 5000)")
    result.add_argument("--receipt", type=Path, help="optional JSON receipt for the filter run")
    result.add_argument("--quiet", action="store_true", help="no progress lines")
    return result


def run(arguments):
    need(arguments.check_max >= 0, "--check-max must be >= 0")
    need(not (arguments.roots and arguments.registry), "choose --roots or --registry, not both")
    need(not (arguments.index_path and arguments.registry), "choose --index or --registry, not both")
    need(arguments.filter_queue or arguments.report or arguments.output,
         "nothing to do: pass --output, --report and/or --filter-queue")
    if arguments.filter_queue:
        need(arguments.out, "--filter-queue requires --out")
    else:
        need(not arguments.out, "--out only makes sense with --filter-queue")
        need(not arguments.check_files, "--check-files only makes sense with --filter-queue")
    specs = None
    inventory = None
    if arguments.index_path:
        need(not arguments.output, "--output cannot be combined with --index; an index is already an index")
        rows = load_index(arguments.index_path)
        if arguments.report:
            inventory = inventory_from_rows(rows)
    else:
        need(arguments.roots or arguments.registry, "pass --roots, --registry or --index")
        if arguments.registry:
            specs = load_registry(arguments.registry)
        else:
            specs = specs_from_roots(arguments.roots)
        inventory = scan(specs, progress_every=0 if arguments.quiet else 2000)
        rows = inventory.rows
    if arguments.output:
        changed = write_index(arguments.output, inventory.rows)
        if not arguments.quiet:
            print(f"index: {inventory.unique_sources} sources -> {arguments.output}"
                  + ("" if changed else " (unchanged)"), file=sys.stderr)
    if arguments.report:
        print(format_report(inventory))
    if arguments.filter_queue:
        if arguments.check_files:
            need(arguments.roots or specs, "--check-files needs --roots (or --registry) to locate labels/ trees")
            if specs is None:
                specs = specs_from_roots(arguments.roots)
            labels = labels_dirs(specs)
        else:
            labels = None
        done = {row["source_id"]: row for row in rows}
        summary = filter_queue_file(
            arguments.filter_queue,
            arguments.out,
            done,
            labels=labels,
            check_files=arguments.check_files,
            check_max=arguments.check_max,
        )
        if arguments.receipt:
            write_atomic(arguments.receipt,
                         (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        return run(arguments)
    except InventoryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
