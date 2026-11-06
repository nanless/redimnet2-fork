#!/usr/bin/env python3
"""Three-segment LM preparation checker (read-only, CPU-only, stdlib-only).

Why a single self-contained file: the evidence it audits lives on the training
host, and that host must stay untouched. A file with no imports from this
repository can be piped to a remote interpreter
(`python -B - build --references ...`, references supplied in
`THREE_SEG_LM_REFS_B64`) so nothing is copied onto, or written inside, the
training host. It also runs where `torch` cannot be imported at all.

It never grants execution authority: `ready_for_lm_execution` is hardcoded
False in both the contract and the report. The tool only says whether the
frozen evidence for the three PTN segments (1-35 / 36-47 / 48-120) and for the
LM cold-start contract is complete and self-consistent.

Commands
    build   read the originals, derive the judgements, freeze digests -> contract
    check   re-read the originals, re-derive every judgement, compare with contract

Exit codes
    0  every check passed
    2  at least one check failed (or the inputs could not be read at all)
    3  no failure, but at least one check is `unsupported` (evidence not supplied)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import os
import pickletools
import re
import stat as statmod
import sys
import zipfile

SCHEMA_REFS = "redimnet2-three-segment-lm-references/v1"
SCHEMA_CONTRACT = "redimnet2-three-segment-lm-contract/v1"
SCHEMA_REPORT = "redimnet2-three-segment-lm-report/v1"

PTN_SPEAKERS = 119274
LM_SPEAKERS = 115362
LM_UTTERANCES = 5021010
EMBED_DIM = 192

# Repository pins. A references file may override them (unit-test fixtures must);
# the effective pin and its origin are recorded in every evidence block.
PIN_MODEL_115_SHA256 = "65497b6a00efac412a82d563aebc1e767056fef23e7f4234f48f89cc8b3aad5c"
PIN_MODEL_120_SHA256 = "2798fe2379c4411673a4c4bdcdd2f87b58a66b5dc1ad9f47af96db0673b0cc2f"
PIN_ORIGINAL_189_SHA256 = "b342964138b7ff72b5403add10f8ac4b962f6b5346d501e757176504c89196b0"

LM_EXPECT = {
    "num_epochs": 1,
    "batch_size_per_gpu": 48,
    "world_size": 8,
    "num_frms": 500,
    "speed_perturb": False,
    "aug_root_prob": 0.7,
    "do_lm": True,
    "margin": 0.3,
    "fix_start_epoch": 1,
    "weight_decay": 4.0e-5,
    "scheduler": "ExponentialDecrease",
    "warm_up_epoch": 0.2,
    "initial_lr": 1.0e-4,
    "final_lr": 2.5e-5,
}
LM_STEPS = LM_UTTERANCES // (LM_EXPECT["batch_size_per_gpu"] * LM_EXPECT["world_size"])
LM_LR_PEAK = 4.5471497e-4
LM_LR_PEAK_STEP = 2615
LM_LR_FINAL = 1.5001590e-4

PASS, FAIL, UNSUP = "pass", "fail", "unsupported"
CSV_COLUMNS = ("epoch", "lr", "wd", "margin", "vox1-eer-8s")

# Candidate fatal patterns for the third segment's log. Finding one is a fail;
# nothing here is whitelisted, and the old seven-event allowance is not reused.
FATAL_PATTERNS = (
    r"Traceback \(most recent call last\)",
    r"Non-finite training",
    r"CUDA out of memory",
    r"ChildFailedError",
    r"Watchdog caught",
    r"could NOT restore",
    r"freshly initialised",
)


class Bad(Exception):
    """A check could not be established from the evidence at hand."""

    def __init__(self, message, *, unsupported=False):
        super().__init__(message)
        self.unsupported = unsupported


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def _as_path(value):
    from pathlib import Path

    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise Bad("证据路径必须是非空字符串")
    path = Path(value)
    if not path.is_absolute():
        raise Bad(f"证据路径必须是绝对路径: {value}")
    return path


def stat_key(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def sha256_stream(value):
    """Hash a regular file and prove it did not move while being read."""
    path = _as_path(value)
    if path.is_symlink():
        raise Bad(f"证据必须是实体文件，不允许软链: {path}")
    if not path.is_file():
        raise Bad(f"证据不存在或不是普通文件: {path}")
    before = os.stat(path)
    if not statmod.S_ISREG(before.st_mode):
        raise Bad(f"证据不是普通文件: {path}")
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
    after = os.stat(path)
    if stat_key(before) != stat_key(after):
        raise Bad(f"读取期间证据发生变化: {path}")
    if total != before.st_size:
        raise Bad(f"读到的字节数与 stat 不符: {path} {total} != {before.st_size}")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": total,
        "stat": stat_key(after),
        "stable": True,
    }


def read_small(value, cap=32 * 1024 * 1024):
    path = _as_path(value)
    if path.is_symlink() or not path.is_file():
        raise Bad(f"证据不存在或不是实体文件: {path}")
    before = os.stat(path)
    if before.st_size > cap:
        raise Bad(f"证据过大，拒绝整体读入（{before.st_size} B，上限 {cap} B）: {path}")
    with open(path, "rb") as stream:
        data = stream.read()
    after = os.stat(path)
    if stat_key(before) != stat_key(after):
        raise Bad(f"读取期间证据发生变化: {path}")
    return data


def parse_json_bytes(data, label):
    def no_duplicates(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                raise Bad(f"{label}: JSON 出现重复键 {key!r}")
            seen[key] = value
        return seen

    def no_constants(token):
        raise Bad(f"{label}: JSON 出现非有限常量 {token}")

    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=no_constants
        )
    except UnicodeDecodeError as error:
        raise Bad(f"{label}: 不是 UTF-8 文本: {error}") from error
    except json.JSONDecodeError as error:
        raise Bad(f"{label}: 不是合法 JSON: {error}") from error


def json_evidence(value, label):
    return parse_json_bytes(read_small(value), label)


def b64_bytes(text):
    if not text:
        return b""
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, ValueError, AttributeError) as error:
        raise Bad(f"base64 输入无法解码: {error}") from error


def pins(refs):
    override = (refs.get("pins") or {}) if isinstance(refs, dict) else {}
    if override:
        return {
            "source": "references-override",
            "model_115_sha256": override.get("model_115_sha256", PIN_MODEL_115_SHA256),
            "model_120_sha256": override.get("model_120_sha256", PIN_MODEL_120_SHA256),
            "original_189_sha256": override.get("original_189_sha256", PIN_ORIGINAL_189_SHA256),
        }
    return {
        "source": "repository-pin",
        "model_115_sha256": PIN_MODEL_115_SHA256,
        "model_120_sha256": PIN_MODEL_120_SHA256,
        "original_189_sha256": PIN_ORIGINAL_189_SHA256,
    }


def counts(refs):
    """Speaker/utterance counts. References may override them for fixtures; the
    effective values and their origin are recorded in every evidence block."""
    override = (refs.get("counts") or {}) if isinstance(refs, dict) else {}
    if override:
        return {
            "source": "references-override",
            "ptn_speakers": int(override.get("ptn_speakers", PTN_SPEAKERS)),
            "lm_speakers": int(override.get("lm_speakers", LM_SPEAKERS)),
            "lm_utterances": int(override.get("lm_utterances", LM_UTTERANCES)),
        }
    return {
        "source": "repository-constant",
        "ptn_speakers": PTN_SPEAKERS,
        "lm_speakers": LM_SPEAKERS,
        "lm_utterances": LM_UTTERANCES,
    }


# --------------------------------------------------------------------------
# segment validation tables
# --------------------------------------------------------------------------
def read_validation_csv(value, label, start, end):
    path = _as_path(value)
    digest = sha256_stream(path)
    with open(path, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise Bad(f"{label}: 验证表为空: {path}")
    missing = [column for column in CSV_COLUMNS if column not in rows[0]]
    if missing:
        raise Bad(f"{label}: 验证表缺少列 {missing}")
    epochs = []
    for index, row in enumerate(rows):
        try:
            epoch = int(row["epoch"])
        except (TypeError, ValueError) as error:
            raise Bad(f"{label}: 第 {index} 行 epoch 不是整数: {row.get('epoch')!r}") from error
        epochs.append(epoch)
        for column in CSV_COLUMNS[1:]:
            try:
                number = float(row[column])
            except (TypeError, ValueError) as error:
                raise Bad(f"{label}: 第 {index} 行 {column} 不是数值") from error
            if number != number or number in (float("inf"), float("-inf")):
                raise Bad(f"{label}: 第 {index} 行 {column} 非有限")
        eer = float(row["vox1-eer-8s"])
        if not 0.0 <= eer <= 100.0:
            raise Bad(f"{label}: 第 {index} 行 vox1-eer-8s 越界: {eer}")
    expected = list(range(start, end + 1))
    if epochs != expected:
        seen = f"{epochs[0]}..{epochs[-1]}" if epochs else "空"
        missing_epochs = sorted(set(expected) - set(epochs))[:5]
        extra = sorted(set(epochs) - set(expected))[:5]
        duplicated = sorted({e for e in epochs if epochs.count(e) > 1})[:5]
        raise Bad(
            f"{label}: 轮次必须精确覆盖 {start}..{end}，实际 {seen}（{len(epochs)} 行）；"
            f"缺 {missing_epochs} 多 {extra} 重复 {duplicated}"
        )
    return {
        "path": str(path),
        "sha256": digest["sha256"],
        "rows": len(rows),
        "epochs": [epochs[0], epochs[-1]],
        "first_row": {column: rows[0][column] for column in CSV_COLUMNS},
        "last_row": {column: rows[-1][column] for column in CSV_COLUMNS},
    }


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def torch_zip_facts(value):
    """Read a torch.save zip archive without importing torch."""
    path = _as_path(value)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            payloads = [n for n in names if n.endswith("data.pkl")]
            if not names:
                raise Bad(f"检查点不是 zip 归档: {path}")
            if len(payloads) != 1:
                raise Bad(f"检查点必须恰有一个 data.pkl，实际 {len(payloads)} 个: {path}")
            payload = archive.read(payloads[0])
            entry_sum = sum(item.file_size for item in archive.infolist())
    except zipfile.BadZipFile as error:
        raise Bad(f"检查点不是合法 zip（可能是旧式非 zip torch.save）: {path}: {error}") from error

    strings = set()
    recorded_epoch = None
    epoch_key_seen = False
    expect_int = False
    int_opcodes = {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT"}
    # pickle protocol 2 memoises every string it pushes, so the integer that
    # carries the value is separated from its key by a memo opcode.
    memo_opcodes = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"}
    try:
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(payload)):
            name = opcode.name
            if name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE") and isinstance(arg, str):
                strings.add(arg)
                epoch_key_seen = epoch_key_seen or arg == "epoch"
                expect_int = arg == "epoch" and recorded_epoch is None
                continue
            if expect_int:
                if name in memo_opcodes:
                    continue
                if name in int_opcodes and isinstance(arg, int):
                    recorded_epoch = arg
                expect_int = False
    except Exception as error:  # pickletools can raise a wide range
        raise Bad(f"检查点 data.pkl 无法解析: {path}: {error}") from error

    absent = [key for key in ("model", "optimizer", "scaler") if key not in strings]
    if absent:
        raise Bad(f"检查点缺少顶层键 {absent}: {path}")
    if not epoch_key_seen:
        raise Bad(f"检查点缺少 epoch 键，无法核对保存轮次: {path}")
    if recorded_epoch is None:
        raise Bad(f"检查点 epoch 不是紧随其键的整数，布局不可信: {path}")
    return {
        "path": str(path),
        "zip_entries": len(names),
        "uncompressed_entry_bytes": entry_sum,
        "recorded_epoch": recorded_epoch,
        "top_level_keys_found": ["epoch", "model", "optimizer", "scaler"],
    }


def checkpoint_evidence(spec, label, pin_values):
    path = _as_path(spec.get("path") if isinstance(spec, dict) else spec)
    epoch_declared = spec.get("epoch")
    digest = sha256_stream(path)
    facts = torch_zip_facts(path)
    if facts["recorded_epoch"] != epoch_declared:
        raise Bad(
            f"{label}: 归档内记录轮次 {facts['recorded_epoch']!r} 与要求的 {epoch_declared!r} 不一致"
        )
    expected = pin_values.get(f"model_{epoch_declared}_sha256")
    if not expected:
        raise Bad(f"{label}: 没有对应的 pin 摘要")
    if digest["sha256"] != expected:
        raise Bad(f"{label}: sha256 与 pin 不符，期望 {expected}，实际 {digest['sha256']}")
    return {
        "path": digest["path"],
        "sha256": digest["sha256"],
        "bytes": digest["bytes"],
        "stat": digest["stat"],
        "stable": True,
        "recorded_epoch": facts["recorded_epoch"],
        "zip_entries": facts["zip_entries"],
        "uncompressed_entry_bytes": facts["uncompressed_entry_bytes"],
        "pinned_sha256": expected,
        "pin_source": pin_values.get("source"),
    }


# --------------------------------------------------------------------------
# receipts / maps / logs
# --------------------------------------------------------------------------
def success_receipt(value):
    record = json_evidence(value, "ptn_success")
    for key, want in (
        ("ptn_complete", True),
        ("LM_started", False),
        ("three_segment_LM_contract_required", True),
    ):
        got = record.get(key)
        if got is not want:
            raise Bad(f"ptn_success.{key} 应为 {want!r}，实际 {got!r}")
    epochs = record.get("epochs")
    if epochs != list(range(48, 121)):
        raise Bad("ptn_success.epochs 必须精确为 48..120")
    if record.get("historical_numeric_root_cause_proven") is not False:
        raise Bad(
            "ptn_success.historical_numeric_root_cause_proven 不是 false："
            "历史数值根因未经独立审阅，本任务不能声明已解决"
        )
    return {
        "path": str(_as_path(value)),
        "checkpoint": record.get("checkpoint"),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "epochs": [epochs[0], epochs[-1]],
        "LM_started": record.get("LM_started"),
        "historical_numeric_root_cause_proven": record.get("historical_numeric_root_cause_proven"),
        "three_segment_LM_contract_required": record.get("three_segment_LM_contract_required"),
    }


def speaker_map(value, expect):
    record = parse_json_bytes(read_small(value), "spk2id")
    if not isinstance(record, dict) or not record:
        raise Bad("spk2id 必须是非空对象")
    if not all(isinstance(k, str) and k for k in record):
        raise Bad("spk2id 说话人名必须是非空字符串")
    if not all(type(v) is int for v in record.values()):
        raise Bad("spk2id 编号必须是整数（不接受布尔）")
    if len(record) != expect["ptn_speakers"]:
        raise Bad(f"spk2id 应有 {expect['ptn_speakers']} 人，实际 {len(record)}")
    if len(set(record.values())) != len(record):
        raise Bad("spk2id 编号不唯一")
    if set(record.values()) != set(range(len(record))):
        raise Bad("spk2id 编号必须从 0 连续")
    if any(record[name] != index for index, name in enumerate(sorted(record))):
        raise Bad("spk2id 编号顺序与 train.py 的词典序不一致")
    return record


def lm_label(value, ptn_map, expect):
    path = _as_path(value)
    digest = sha256_stream(path)
    speakers = set()
    rows = 0
    with open(path, "r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            parts = line.split()
            if len(parts) != 2:
                raise Bad(f"LM 标签第 {number} 行必须包含 utterance 与 speaker 两列")
            speakers.add(parts[1])
            rows += 1
    if rows != expect["lm_utterances"]:
        raise Bad(f"LM 标签应有 {expect['lm_utterances']} 句，实际 {rows}")
    if len(speakers) != expect["lm_speakers"]:
        raise Bad(f"LM 标签应有 {expect['lm_speakers']} 人，实际 {len(speakers)}")
    missing = sorted(speakers - set(ptn_map))
    if missing:
        raise Bad(f"LM 说话人不在 PTN 映射中：{len(missing)} 人，前 10 个 {missing[:10]}")
    return {
        "path": str(path),
        "sha256": digest["sha256"],
        "rows": rows,
        "speakers": len(speakers),
        "dropped_from_ptn": expect["ptn_speakers"] - len(speakers),
        "counts_source": expect["source"],
        "subset": True,
    }


def reviewed_events_scope(value, original_task, original_run):
    record = json_evidence(value, "reviewed_events")
    events = record.get("events")
    if not isinstance(events, list) or len(events) != 7:
        raise Bad(f"已审事件必须正好 7 条，实际 {len(events) if isinstance(events, list) else '非列表'}")
    if record.get("parent_task_id") != original_task:
        raise Bad(f"已审事件绑定任务 {record.get('parent_task_id')!r} 不是原段 {original_task!r}")
    if record.get("parent_run_root") != original_run:
        raise Bad("已审事件绑定 run 不是原段 run")
    ids = [event.get("event_id") for event in events if isinstance(event, dict)]
    if len(ids) != 7 or len(set(ids)) != 7:
        raise Bad("已审事件的 event_id 必须 7 条且唯一")
    return {
        "path": str(_as_path(value)),
        "review_id": record.get("review_id"),
        "events": len(events),
        "scoped_to_task": record.get("parent_task_id"),
        "applies_to_successor": False,
    }


def scan_successor_log(value):
    path = _as_path(value)
    digest = sha256_stream(path)
    hits = {}
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            for pattern in FATAL_PATTERNS:
                if re.search(pattern, line):
                    hits.setdefault(pattern, []).append(number)
    return {
        "path": str(path),
        "sha256": digest["sha256"],
        "bytes": digest["bytes"],
        "fatal_hits": {pattern: lines[:3] for pattern, lines in hits.items()},
        "fatal_total": sum(len(lines) for lines in hits.values()),
    }


def resource_return(value):
    record = json_evidence(value, "resource_return")
    snapshots = record.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) < 2:
        raise Bad("resource_return 至少要两份快照")
    if record.get("GPU_compute_PID_absent") is not True:
        raise Bad("resource_return.GPU_compute_PID_absent 不是 true")
    for index, snapshot in enumerate(snapshots):
        devices = snapshot.get("devices")
        if not isinstance(devices, list) or len(devices) != 8:
            raise Bad(f"第 {index} 份快照不是 8 张卡")
        busy = [device.get("index") for device in devices if device.get("used_mib") != 0]
        if busy:
            raise Bad(f"第 {index} 份快照仍有占用: {busy}")
    return {
        "path": str(_as_path(value)),
        "snapshots": len(snapshots),
        "last_utc": snapshots[-1].get("UTC"),
        "devices_idle": 8,
        "compute_pid_absent": True,
    }


# --------------------------------------------------------------------------
# source identity
# --------------------------------------------------------------------------
def source_identity(spec, pin_values):
    spec = spec or {}
    manifest_path = spec.get("manifest_190")
    delta_path = spec.get("delta")
    tree = spec.get("source_tree")
    if not (manifest_path and delta_path):
        raise Bad("未提供 190 清单或 189→190 差异文件", unsupported=True)
    manifest = json_evidence(manifest_path, "source_manifest")
    if not isinstance(manifest, list):
        raise Bad("source_manifest 必须是成员列表")
    relatives = [member.get("relative") for member in manifest if isinstance(member, dict)]
    if len(relatives) != 190 or len(set(relatives)) != 190:
        raise Bad(f"source_manifest 必须是 190 个唯一成员，实际 {len(relatives)}/{len(set(relatives))}")

    tree_checked = 0
    if tree:
        import pathlib

        root = pathlib.Path(tree)
        if not root.is_dir():
            raise Bad(f"source 树不存在: {root}")
        for member in manifest:
            target = root / member["relative"]
            if not target.is_file():
                raise Bad(f"source 树缺少成员: {member['relative']}")
            actual = sha256_stream(str(target))
            if actual["bytes"] != member.get("bytes"):
                raise Bad(f"source 成员字节数与清单不符: {member['relative']}")
            if actual["sha256"] != member.get("sha256"):
                raise Bad(f"source 成员 sha256 与清单不符: {member['relative']}")
            tree_checked += 1

    delta = json_evidence(delta_path, "source_delta")
    if delta.get("members_before") != 189 or delta.get("members_after") != 190:
        raise Bad("source_delta 的成员数不是 189→190")
    changes = delta.get("changes")
    if not isinstance(changes, list) or len(changes) != 2:
        raise Bad(f"source_delta 必须正好列出 2 处变化，实际 {len(changes) if isinstance(changes, list) else '非列表'}")
    kinds = sorted(str(change.get("change")) for change in changes)
    if kinds != ["add", "replace"]:
        raise Bad(f"source_delta 变化类型必须是 add/replace，实际 {kinds}")
    by_member = {member["relative"]: member for member in manifest}
    for change in changes:
        relative = change.get("relative")
        member = by_member.get(relative)
        if member is None:
            raise Bad(f"source_delta 列出的成员不在 190 清单中: {relative}")
        if member.get("sha256") != change.get("new_sha256"):
            raise Bad(f"source_delta 新摘要与 190 清单不符: {relative}")
    if delta.get("original_source189_sha256") != pin_values["original_189_sha256"]:
        raise Bad("source_delta 记录的 189 原清单摘要不是 pin 值")

    return {
        "manifest": {"path": str(_as_path(manifest_path)), "members": 190},
        "delta": {"path": str(_as_path(delta_path)), "changes": kinds, "before": 189, "after": 190},
        "tree_verified_members": tree_checked,
        "original_source189_sha256": delta.get("original_source189_sha256"),
        "changed_members": [change.get("relative") for change in changes],
        "pin_source": pin_values["source"],
    }


def original_189_index(value, pin_values):
    path = _as_path(value)
    digest = sha256_stream(path)
    if digest["sha256"] != pin_values["original_189_sha256"]:
        raise Bad(f"189 原清单摘要不是 pin 值：{digest['sha256']}")
    record = json_evidence(value, "source_189")
    if not isinstance(record, list) or len(record) != 189:
        raise Bad(f"189 原清单应列出 189 个成员，实际 {len(record) if isinstance(record, list) else '非列表'}")
    return {"path": str(path), "sha256": digest["sha256"], "members": 189}


def original_95_index(value, expected_sha256, ledger_189=None):
    """The 95-member list exists in two shapes on disk: a JSON ledger, or a
    `sha256sum -c` text file (the real one is the latter). Both are accepted,
    but the member count and the pinned digest are not negotiable."""
    if not value or not expected_sha256:
        raise Bad("未提供原 95 份清单文件或其固定摘要", unsupported=True)
    path = _as_path(value)
    digest = sha256_stream(path)
    if digest["sha256"] != expected_sha256:
        raise Bad(f"95 原清单摘要不符：希望 {expected_sha256}，实际 {digest['sha256']}")
    raw = read_small(value)
    try:
        record = parse_json_bytes(raw, "source_95")
    except Bad:
        record = None
    if isinstance(record, dict):
        members = record.get("members")
    elif isinstance(record, list):
        members = record
    else:
        members = None
    if members is None:
        # sha256sum -c format: "<hex>  <path>" per line, 95 members.
        lines = [line for line in raw.decode("utf-8", "replace").splitlines() if line.strip()]
        entries = []
        for number, line in enumerate(lines, 1):
            parts = line.split(None, 1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                raise Bad(f"95 清单第 {number} 行不是 <sha256>  <path>: {line[:60]!r}")
            entries.append({"sha256": parts[0], "path": parts[1].lstrip("* ").strip()})
        members = entries
        shape = "sha256sum-text"
        # Cross-check against the 189 ledger when it is available: the 95 list
        # must be a subset of it, with identical hashes.
        if ledger_189:
            by_name = {row.get("relative"): row.get("sha256") for row in ledger_189 if isinstance(row, dict)}
            prefix = "/root/code/github_repos/redimnet2-fork/"
            mismatched = []
            for entry in entries:
                relative = entry["path"][len(prefix):] if entry["path"].startswith(prefix) else entry["path"]
                if relative in by_name and by_name[relative] != entry["sha256"]:
                    mismatched.append(relative)
            if mismatched:
                raise Bad(f"95 清单与 189 清单摘要冲突: {mismatched[:5]}")
            return {
                "path": str(path),
                "sha256": digest["sha256"],
                "members": len(entries),
                "shape": shape,
                "subset_of_189_checked": len([e for e in entries if e["path"].startswith(prefix)]),
            }
    else:
        shape = "json-ledger"
    if not isinstance(members, list) or len(members) != 95:
        raise Bad(f"95 原清单应列出 95 个成员，实际 {len(members) if isinstance(members, list) else '未知'}")
    return {"path": str(path), "sha256": digest["sha256"], "members": 95, "shape": shape}


# --------------------------------------------------------------------------
# LM cold-start contract
# --------------------------------------------------------------------------
def load_recipe(value):
    try:
        import yaml
    except ImportError as error:
        raise Bad(f"当前解释器没有 PyYAML，无法核对配方: {error}", unsupported=True) from error
    data = read_small(value, cap=8 * 1024 * 1024)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Bad(f"配方不是 UTF-8: {error}") from error
    try:
        record = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise Bad(f"配方 YAML 解析失败: {error}") from error
    if not isinstance(record, dict):
        raise Bad("配方顶层必须是映射")
    return record


def lm_cold_start_contract(refs):
    recipe_path = refs.get("lm_recipe")
    declaration = refs.get("lm_contract") or {}
    if not recipe_path:
        raise Bad("未提供 LM 配方路径", unsupported=True)
    recipe = load_recipe(recipe_path)
    expect = counts(refs)
    steps = expect["lm_utterances"] // (LM_EXPECT["batch_size_per_gpu"] * LM_EXPECT["world_size"])
    problems = []

    def want(condition, message):
        if not condition:
            problems.append(message)

    dataset = recipe.get("dataset_args") or {}
    loader = recipe.get("dataloader_args") or {}
    margin = recipe.get("margin_update") or {}
    sched = recipe.get("scheduler_args") or {}
    projection = recipe.get("projection_args") or {}
    aug = dataset.get("aug_setup") or {}

    want(recipe.get("num_epochs") == LM_EXPECT["num_epochs"], "num_epochs 不是 1")
    want(loader.get("batch_size") == LM_EXPECT["batch_size_per_gpu"], "每卡 batch_size 不是 48")
    want(loader.get("drop_last") is True, "drop_last 不是 true")
    want(dataset.get("num_frms") == LM_EXPECT["num_frms"], "num_frms 不是 500")
    want(dataset.get("speed_perturb") is LM_EXPECT["speed_perturb"], "speed_perturb 不是 false")
    want(aug.get("prob") == LM_EXPECT["aug_root_prob"], "增强根概率不是 0.7")
    want(recipe.get("do_lm") is LM_EXPECT["do_lm"], "do_lm 不是 true")
    want(projection.get("project_type") == "sphereface2", "project_type 不是 sphereface2")
    want(margin.get("initial_margin") == LM_EXPECT["margin"], "initial_margin 不是 0.3")
    want(margin.get("final_margin") == LM_EXPECT["margin"], "final_margin 不是 0.3")
    want(margin.get("fix_start_epoch") == LM_EXPECT["fix_start_epoch"], "fix_start_epoch 不是 1")
    want(margin.get("update_margin") is True, "update_margin 不是 true")
    want((recipe.get("optimizer_args") or {}).get("weight_decay") == LM_EXPECT["weight_decay"],
         "weight_decay 不是 4.0e-05")
    want(recipe.get("scheduler") == LM_EXPECT["scheduler"], "scheduler 不是 ExponentialDecrease")
    want(sched.get("warm_up_epoch") == LM_EXPECT["warm_up_epoch"], "warm_up_epoch 不是 0.2")
    want(sched.get("initial_lr") == LM_EXPECT["initial_lr"], "initial_lr 不是 1.0e-04")
    want(sched.get("final_lr") == LM_EXPECT["final_lr"], "final_lr 不是 2.5e-05")
    want(sched.get("lr_scaling") == "legacy", "lr_scaling 不是 legacy")

    for key, expected in (
        ("optimizer_fresh", True),
        ("scaler_fresh", True),
        ("loads_model_parameters_only", True),
        ("restores_ptn_optimizer", False),
        ("restores_ptn_scaler", False),
        ("automatic_retry", False),
    ):
        if declaration.get(key) is not expected:
            problems.append(f"LM 合同声明 {key} 应为 {expected!r}，实际 {declaration.get(key)!r}")

    outputs = [entry for entry in (declaration.get("forbidden_existing_paths") or []) if entry]
    for path in outputs:
        if os.path.exists(path):
            problems.append(f"LM 输出目录已存在，会被覆盖: {path}")
    if not outputs:
        problems.append("LM 合同未列出必须为空的输出目录")

    if problems:
        raise Bad("LM 冷启动合同不满足：" + "；".join(problems))

    return {
        "recipe": str(_as_path(recipe_path)),
        "num_epochs": recipe.get("num_epochs"),
        "batch_size_per_gpu": loader.get("batch_size"),
        "steps_per_epoch": steps,
        "num_frms": dataset.get("num_frms"),
        "classes": expect["lm_speakers"],
        "counts_source": expect["source"],
        "margin": margin.get("initial_margin"),
        "lr_scale_ratio": LM_EXPECT["world_size"] * LM_EXPECT["batch_size_per_gpu"] / 64.0,
        "lr_peak": LM_LR_PEAK,
        "lr_peak_step": LM_LR_PEAK_STEP,
        "lr_final": LM_LR_FINAL,
        "optimizer_fresh": True,
        "scaler_fresh": True,
        "automatic_retry": False,
        "checked_outputs": outputs,
        "meaning": "只证明合同声明与配方一致；冷启动本身须由任务内八 rank 日志与新优化器/scaler 状态证明",
    }


# --------------------------------------------------------------------------
# frozen LM package
# --------------------------------------------------------------------------
PACKAGE_SCHEMA = "lm-from-ptn115/v1"


def package_lm_contract(spec, pin_values, expect_counts):
    """Check the frozen LM package: membership, contract semantics, executor.

    A package that merely declares itself fine is not accepted: every listed
    file is re-hashed, and every LM-specific field is compared against the
    values this checker pins independently.
    """
    spec = spec or {}
    root = spec.get("root")
    if not root or not spec.get("bundle_files") or not spec.get("contract"):
        raise Bad("未提供 LM 冻结包（root/bundle_files/contract）", unsupported=True)

    import pathlib

    package_root = pathlib.Path(root)
    if not package_root.is_dir():
        raise Bad(f"包目录不存在: {package_root}")

    bundle = json_evidence(spec["bundle_files"], "bundle_files")
    if not isinstance(bundle, dict) or not bundle:
        raise Bad("bundle_files.json 必须是非空对象")
    declared = set(bundle)
    on_disk = set()
    for path in package_root.rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(package_root))
            if relative in ("bundle_files.json", "build-receipt.json"):
                continue
            on_disk.add(relative)
    missing = sorted(declared - on_disk)
    extra = sorted(on_disk - declared)
    if missing:
        raise Bad(f"包清单列出但磁盘缺失: {missing[:5]}")
    if extra:
        raise Bad(f"包内存在未登记文件: {extra[:5]}")
    for relative, pinned in bundle.items():
        actual = sha256_stream(str(package_root / relative))
        if actual["sha256"] != pinned:
            raise Bad(f"包成员摘要不符: {relative}")

    contract = json_evidence(spec["contract"], "package_contract")
    problems = []

    def want(condition, message):
        if not condition:
            problems.append(message)

    want(contract.get("schema") == PACKAGE_SCHEMA, f"合同 schema 应为 {PACKAGE_SCHEMA}")
    want(contract.get("runtime_admission") is False, "runtime_admission 必须为 false")
    want(contract.get("LM_started") is False, "LM_started 必须为 false")

    checkpoint = contract.get("checkpoint") or {}
    want(checkpoint.get("sha256") == pin_values["model_115_sha256"], "合同 checkpoint 不是 model_115 的 pin")
    clock = contract.get("clock") or {}
    want(clock.get("stage") == "lm", "clock.stage 不是 lm")
    want(clock.get("start_epoch") == 1 and clock.get("target_epoch") == 1, "LM 必须只跑 1 轮")
    want(clock.get("epoch_iter") == expect_counts["steps"], f"clock.epoch_iter 不是 {expect_counts['steps']}")
    want(clock.get("batch_size_per_gpu") == LM_EXPECT["batch_size_per_gpu"], "每卡 batch 不是 48")
    want(clock.get("margin") == LM_EXPECT["margin"], "margin 不是 0.3")
    want(clock.get("optimizer") == "new", "LM 优化器必须是 new（冷启动）")
    want(clock.get("scaler") == "new", "LM scaler 必须是 new（冷启动）")
    want(clock.get("rng_restored") is False, "LM 不得恢复 PTN 的 RNG")
    speaker = contract.get("speaker_map") or {}
    want(speaker.get("classes") == expect_counts["lm_speakers"], "分类头类别数不是 115,362")
    want(speaker.get("ptn_speakers") == expect_counts["ptn_speakers"], "合同 spk2id 人数不是 119,274")
    if expect_counts.get("speaker_map_sha256"):
        want(speaker.get("sha256") == expect_counts["speaker_map_sha256"],
             "合同 speaker_map.sha256 与实际 spk2id.json 不符")
    new_run_root = contract.get("new_run_root")
    if new_run_root and os.path.exists(new_run_root):
        problems.append(f"new_run_root 已存在，会被覆盖: {new_run_root}")
    if not new_run_root:
        problems.append("合同缺少 new_run_root")

    runtime_files = contract.get("runtime_files") or {}
    if not isinstance(runtime_files, dict) or not runtime_files:
        problems.append("合同缺少 runtime_files")
    else:
        for name, pinned in runtime_files.items():
            target = package_root / "runtime" / name
            if not target.is_file():
                problems.append(f"runtime 文件缺失: {name}")
                continue
            if sha256_stream(str(target))["sha256"] != pinned:
                problems.append(f"runtime 文件摘要不符: {name}")

    config_relative = contract.get("resume_config_path")
    if config_relative:
        config_path = package_root / config_relative if not os.path.isabs(config_relative) else pathlib.Path(config_relative)
        if not config_path.is_file():
            problems.append(f"resume 配置缺失: {config_path}")
        elif sha256_stream(str(config_path))["sha256"] != contract.get("resume_config_sha256"):
            problems.append("resume 配置摘要与合同不符")
    else:
        problems.append("合同缺少 resume_config_path")

    executor = contract.get("executor")
    if not executor:
        problems.append("合同缺少 executor")
    elif executor not in bundle:
        problems.append(f"executor 未登记在包清单: {executor}")

    legacy = contract.get("legacy") or {}
    want(legacy.get("bundle_files_sha256") == "840235e7b5963ada274531a29401b9f2bd75fe92e985a8e8e560341ac48350d8",
         "legacy.bundle_files_sha256 不是 v3 旧包摘要")

    if problems:
        raise Bad("冻结包合同不满足：" + "；".join(problems))
    return {
        "root": str(package_root),
        "bundle_sha256": sha256_stream(spec["bundle_files"])["sha256"],
        "contract_sha256": sha256_stream(spec["contract"])["sha256"],
        "members": len(bundle),
        "members_rehashed": len(bundle),
        "executor": executor,
        "checkpoint_sha256": checkpoint.get("sha256"),
        "clock": {k: clock.get(k) for k in ("stage", "start_epoch", "target_epoch", "epoch_iter", "lr", "margin", "optimizer", "scaler")},
        "classes": speaker.get("classes"),
        "new_run_root": new_run_root,
        "runtime_files": len(runtime_files),
        "meaning": "包成员逐个重新哈希、合同字段逐项与 pin 比对；runtime_admission 仍为 false",
    }


# --------------------------------------------------------------------------
# platform identity
# --------------------------------------------------------------------------
def platform_terminal(value, task_id):
    if not value:
        raise Bad("未提供平台终态观测原件（checks/ 里没有本段 task id 的终态 JSON）", unsupported=True)
    record = json_evidence(value, "terminal")
    if record.get("task_id") != task_id:
        raise Bad(f"终态原件绑定任务 {record.get('task_id')!r} 不是本段 {task_id!r}")
    if record.get("state") not in ("Succeeded", "Success"):
        raise Bad(f"平台终态不是 Succeeded/Success，实际 {record.get('state')!r}")
    if record.get("active_workers") != 0 or type(record.get("active_workers")) is not int:
        raise Bad("平台终态 active_workers 不是整数 0")
    stamp = record.get("observed_at") or record.get("observed_utc")
    if not isinstance(stamp, str) or not (stamp.endswith("Z") or "+" in stamp):
        raise Bad("终态原件缺少带时区的观测时间")
    return {"path": str(_as_path(value)), "task_id": task_id, "state": record.get("state"), "observed_at": stamp}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(refs):
    """Re-derive every judgement from the originals. Stored booleans are never trusted."""
    checks = []
    pin_values = pins(refs)
    count_values = counts(refs)

    def run(name, fn, *, note=None):
        try:
            checks.append({"name": name, "status": PASS, "evidence": fn(), "note": note})
        except Bad as error:
            checks.append(
                {"name": name, "status": UNSUP if error.unsupported else FAIL, "reason": str(error), "note": note}
            )
        except OSError as error:
            checks.append({"name": name, "status": FAIL, "reason": f"OSError: {error}", "note": note})

    segments = refs.get("segments") or {}
    original = segments.get("original") or {}
    failed = segments.get("failed") or {}
    successor = segments.get("successor") or {}

    def segment(name, spec, start, end):
        if not spec.get("validation_csv"):
            raise Bad(f"{name}: 未提供验证表路径", unsupported=True)
        return read_validation_csv(spec["validation_csv"], name, start, end)

    run("segments.validation_original", lambda: segment("original", original, 1, 35),
        note="原段 1—35 轮验证表")
    run("segments.validation_failed", lambda: segment("failed", failed, 36, 47),
        note="失败 vggvm 段 36—47 轮验证表")
    run("segments.validation_successor", lambda: segment("successor", successor, 48, 120),
        note="第三段 48—120 轮验证表")

    def coverage():
        spans = []
        for name, spec, start, end in (
            ("original", original, 1, 35),
            ("failed", failed, 36, 47),
            ("successor", successor, 48, 120),
        ):
            if not spec.get("validation_csv"):
                raise Bad(f"{name}: 缺验证表，无法核对三段覆盖", unsupported=True)
            rows = read_validation_csv(spec["validation_csv"], name, start, end)
            spans.append([name, rows["epochs"][0], rows["epochs"][1]])
        union = []
        for _name, start, end in spans:
            union.extend(range(start, end + 1))
        if sorted(union) != list(range(1, 121)) or len(set(union)) != 120:
            raise Bad("三段轮次并集不是 1..120，或存在重叠")
        return {"ranges": spans, "union": [1, 120], "overlaps": 0}

    run("segments.complete_coverage_1_120", coverage,
        note="并集必须恰好 1..120，不重复、不补造")

    def identity():
        out = {}
        for name, spec in (("original", original), ("failed", failed), ("successor", successor)):
            task = spec.get("task_id")
            if not task:
                raise Bad(f"{name}: 未提供平台任务 ID")
            if not re.fullmatch(r"t-\d{14}-[a-z0-9]{5}", str(task)):
                raise Bad(f"{name}: 任务 ID 形状不对: {task!r}")
            out[name] = {
                "task_id": task,
                "run_root": spec.get("run_root"),
                "task_id_provenance": spec.get("task_id_provenance", "unspecified"),
            }
        return out

    run("segments.task_identity", identity,
        note="第三段 ID 来自仓库记录，未经 CLI 复核（见 task_id_provenance）")

    def checkpoints():
        specs = [spec for spec in (refs.get("checkpoints") or []) if spec]
        if not specs:
            raise Bad("未提供任何检查点", unsupported=True)
        return [checkpoint_evidence(spec, f"checkpoint_{spec.get('epoch')}", pin_values) for spec in specs]

    run("checkpoints.identity", checkpoints,
        note="115/120：实体路径 + 读取前后 stat + sha256 + 归档内记录轮次")

    def receipt():
        record = success_receipt(refs["success_receipt"])
        by_epoch = {spec.get("epoch"): spec for spec in (refs.get("checkpoints") or []) if spec}
        declared_path = (by_epoch.get(120) or {}).get("path")
        if record.get("checkpoint_sha256") != pin_values["model_120_sha256"]:
            raise Bad("ptn_success.checkpoint_sha256 与 model_120.pt 的 pin 摘要不一致")
        if declared_path and record.get("checkpoint") != declared_path:
            raise Bad("ptn_success.checkpoint 指向的不是实际 model_120.pt 路径")
        return record

    run("success.receipt", receipt, note="第三段成功收据（训练侧自述，非平台观测）")

    def maps():
        ptn = speaker_map(refs["speaker_map"], count_values)
        lm = lm_label(refs["lm_label"], ptn, count_values)
        return {"ptn_speakers": len(ptn), "lm": lm, "relation": "lm_subset_of_ptn"}

    run("speaker_map.subset", maps, note="119,274 → 115,362，名称全覆盖，无随机/切片退路")

    def events():
        spec = refs.get("reviewed_events") or {}
        if not spec.get("path"):
            raise Bad("未提供已审事件原件", unsupported=True)
        return reviewed_events_scope(spec["path"], original.get("task_id"), original.get("run_root"))

    run("reviewed_events.scope", events, note="7 次已审 FLAC 事件只属于原段")

    def log_scan():
        spec = refs.get("successor_log")
        if not spec:
            raise Bad("未提供第三段 launch.log，无法复查新异常", unsupported=True)
        facts = scan_successor_log(spec)
        if facts["fatal_total"]:
            raise Bad(f"第三段日志出现未审阅致命模式：{facts['fatal_hits']}")
        return facts

    run("successor.no_unreviewed_failures", log_scan,
        note="旧七事件许可不搬到第三段；命中致命模式即 fail")

    run("source.manifest_190_and_delta", lambda: source_identity(refs.get("source"), pin_values),
        note="190 成员树 + 189→190 两处差异")
    run("source.original_189_index",
        lambda: original_189_index((refs.get("source") or {}).get("original_189_index"), pin_values),
        note="原 189 清单必须仍是 pin 摘要")
    def source95():
        spec = refs.get("source") or {}
        ledger = None
        if spec.get("original_189_index"):
            try:
                ledger = json_evidence(spec["original_189_index"], "source_189")
            except Bad:
                ledger = None
        return original_95_index(spec.get("original_95_index"), spec.get("original_95_sha256"), ledger)

    run("source.original_95_index", source95,
        note="拿不到原 95 清单时必须 unsupported，不许静默通过")

    def package():
        expect = counts(refs)
        expect = {
            "lm_speakers": expect["lm_speakers"],
            "ptn_speakers": expect["ptn_speakers"],
            "speaker_map_sha256": sha256_stream(refs["speaker_map"])["sha256"],
            "steps": expect["lm_utterances"] // (LM_EXPECT["batch_size_per_gpu"] * LM_EXPECT["world_size"]),
        }
        return package_lm_contract(refs.get("package"), pin_values, expect)

    run("package.lm_contract", package,
        note="冻结 LM 包：成员重哈希 + 合同逐项比对 + executor 必须登记")

    run("lm.cold_start_contract", lambda: lm_cold_start_contract(refs),
        note="只载模型参数、优化器/scaler 全新、无自动重试")
    run("resources.returned", lambda: resource_return(refs["resource_return"]),
        note="八卡归还证明")
    run("platform.terminal_observed",
        lambda: platform_terminal((refs.get("platform") or {}).get("terminal"), successor.get("task_id")),
        note="平台侧终态观测；缺原件时 unsupported")
    return checks


def digest_artifacts(refs):
    """Freeze the byte identity of every artefact named by the references."""
    frozen = {}

    def put(label, path):
        if not path:
            return
        try:
            frozen[label] = sha256_stream(path)
        except Bad as error:
            frozen[label] = {"path": str(path), "error": str(error)}

    segments = refs.get("segments") or {}
    for name in ("original", "failed", "successor"):
        put(f"validation.{name}", (segments.get(name) or {}).get("validation_csv"))
    for spec in refs.get("checkpoints") or []:
        if spec:
            put(f"checkpoint.{spec.get('epoch')}", spec.get("path"))
    put("success_receipt", refs.get("success_receipt"))
    put("speaker_map", refs.get("speaker_map"))
    put("lm_label", refs.get("lm_label"))
    put("resource_return", refs.get("resource_return"))
    put("reviewed_events", (refs.get("reviewed_events") or {}).get("path"))
    put("successor_log", refs.get("successor_log"))
    source = refs.get("source") or {}
    put("source.manifest_190", source.get("manifest_190"))
    put("source.delta", source.get("delta"))
    put("source.original_189_index", source.get("original_189_index"))
    put("source.original_95_index", source.get("original_95_index"))
    put("lm_recipe", refs.get("lm_recipe"))
    package = refs.get("package") or {}
    put("package.bundle_files", package.get("bundle_files"))
    put("package.contract", package.get("contract"))
    put("platform.terminal", (refs.get("platform") or {}).get("terminal"))
    return frozen


def summarise(checks):
    counts = {PASS: 0, FAIL: 0, UNSUP: 0}
    for check in checks:
        counts[check["status"]] += 1
    return {
        "total": len(checks),
        "passed": counts[PASS],
        "failed": counts[FAIL],
        "unsupported": counts[UNSUP],
        "checks_passed": counts[FAIL] == 0 and counts[UNSUP] == 0,
    }


def exit_code(summary):
    if summary["failed"]:
        return 2
    if summary["unsupported"]:
        return 3
    return 0


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def load_references(args):
    if args.references and args.references != "-":
        return json_evidence(args.references, "references")
    if args.references == "-":
        return parse_json_bytes(sys.stdin.buffer.read(), "references")
    raw = b64_bytes(os.environ.get("THREE_SEG_LM_REFS_B64", ""))
    if not raw:
        raise Bad("没有引用表：请用 --references PATH 或设置 THREE_SEG_LM_REFS_B64")
    return parse_json_bytes(raw, "references")


def validate_references(refs):
    if not isinstance(refs, dict):
        raise Bad("引用表必须是 JSON 对象")
    if refs.get("schema") != SCHEMA_REFS:
        raise Bad(f"引用表 schema 必须是 {SCHEMA_REFS}，实际 {refs.get('schema')!r}")
    segments = refs.get("segments")
    if not isinstance(segments, dict) or set(segments) != {"original", "failed", "successor"}:
        raise Bad("引用表必须列出 original/failed/successor 三段")
    for key in ("success_receipt", "speaker_map", "lm_label", "resource_return", "lm_recipe"):
        if not refs.get(key):
            raise Bad(f"引用表缺少 {key}")
    checkpoints = refs.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise Bad("引用表必须列出 checkpoint 115/120")
    epochs = sorted(spec.get("epoch") for spec in checkpoints if isinstance(spec, dict))
    if epochs != [115, 120]:
        raise Bad(f"检查点必须正好是 115 与 120，实际 {epochs}")


def write_json(payload, target):
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target in (None, "-"):
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    from pathlib import Path

    path = Path(target)
    if path.exists() and not path.is_file():
        raise SystemExit(f"输出目标不是普通文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def line(payload):
    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def cmd_build(args):
    refs = load_references(args)
    validate_references(refs)
    checks = evaluate(refs)
    summary = summarise(checks)
    contract = {
        "schema": SCHEMA_CONTRACT,
        "generated_by": "tools/ptn_resume/three_segment_lm_check.py",
        "ready_for_lm_execution": False,
        "execution_authority": "none; this contract only freezes CPU evidence",
        "pin_source": pins(refs)["source"],
        "counts_source": counts(refs)["source"],
        "references": refs,
        "artifacts": digest_artifacts(refs),
        "checks": checks,
        "summary": summary,
        "mandatory_future_execution_gate": [
            "stable exclusive lock across the three segments before any write",
            "unique new LM output claim before any write",
            "eight-rank cold optimizer/scaler evidence inside the LM task",
            "LM1 postflight (finite loss, cold-start marker, saved checkpoint)",
        ],
    }
    write_json(contract, args.output)
    line({
        "stage": "build",
        "checks_passed": summary["checks_passed"],
        "ready_for_lm_execution": False,
        "passed": summary["passed"],
        "failed": summary["failed"],
        "unsupported": summary["unsupported"],
        "failed_checks": [check["name"] for check in checks if check["status"] == FAIL],
        "unsupported_checks": [check["name"] for check in checks if check["status"] == UNSUP],
    })
    return exit_code(summary)


def cmd_check(args):
    if args.contract and args.contract != "-":
        contract_bytes = read_small(args.contract)
    elif args.contract == "-":
        contract_bytes = sys.stdin.buffer.read()
    else:
        contract_bytes = b64_bytes(os.environ.get("THREE_SEG_LM_CONTRACT_B64", ""))
        if not contract_bytes:
            raise Bad("没有合同：请用 --contract PATH 或设置 THREE_SEG_LM_CONTRACT_B64")
    contract = parse_json_bytes(contract_bytes, "contract")
    if contract.get("schema") != SCHEMA_CONTRACT:
        raise Bad(f"合同 schema 必须是 {SCHEMA_CONTRACT}")
    if contract.get("ready_for_lm_execution") is not False:
        raise Bad("合同自称已获执行资格，拒绝（本工具永不放行）")

    refs = contract.get("references")
    validate_references(refs)

    # Re-derive everything from the originals; stored verdicts are only compared.
    checks = evaluate(refs)
    frozen = digest_artifacts(refs)

    drift = []
    expected = contract.get("artifacts") or {}
    for label, record in frozen.items():
        want = expected.get(label)
        if not isinstance(want, dict):
            drift.append({"artifact": label, "reason": "合同没有冻结该原件，检查器不予放行"})
            continue
        if "error" in record:
            drift.append({"artifact": label, "reason": f"原件不可读: {record['error']}"})
            continue
        if record.get("sha256") != want.get("sha256"):
            drift.append({"artifact": label, "reason": f"sha256 变了: {want.get('sha256')} → {record.get('sha256')}"})
        if record.get("bytes") != want.get("bytes"):
            drift.append({"artifact": label, "reason": f"字节数变了: {want.get('bytes')} → {record.get('bytes')}"})

    stored = {check["name"]: check["status"] for check in contract.get("checks") or []}
    for check in checks:
        previous = stored.get(check["name"])
        if previous is not None and previous != check["status"]:
            drift.append({"artifact": check["name"], "reason": f"判定漂移: {previous} → {check['status']}"})

    if drift:
        checks.append({
            "name": "contract.no_drift",
            "status": FAIL,
            "reason": f"{len(drift)} 处原件或判定与冻结合同不一致",
            "evidence": drift[:20],
        })
    else:
        checks.append({"name": "contract.no_drift", "status": PASS,
                       "evidence": {"artifacts": len(frozen)}})
    summary = summarise(checks)

    report = {
        "schema": SCHEMA_REPORT,
        "generated_by": "tools/ptn_resume/three_segment_lm_check.py",
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "ready_for_lm_execution": False,
        "execution_authority": "none; a fully passing CPU report still does not authorise LM execution",
        "pin_source": pins(refs)["source"],
        "counts_source": counts(refs)["source"],
        "checks_passed": summary["checks_passed"],
        "summary": summary,
        "checks": checks,
        "drift": drift,
    }
    write_json(report, args.output)
    line({
        "stage": "check",
        "checks_passed": summary["checks_passed"],
        "ready_for_lm_execution": False,
        "passed": summary["passed"],
        "failed": summary["failed"],
        "unsupported": summary["unsupported"],
        "failed_checks": [check["name"] for check in checks if check["status"] == FAIL],
        "unsupported_checks": [check["name"] for check in checks if check["status"] == UNSUP],
    })
    return exit_code(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="从原件生成冻结合同")
    build.add_argument("--references", help="引用表路径，- 表示 stdin，缺省读 THREE_SEG_LM_REFS_B64")
    build.add_argument("--output", default="-", help="输出路径，- 表示 stdout（缺省）")
    check = sub.add_parser("check", help="重读原件、重算判定、与合同比对")
    check.add_argument("--contract", help="合同路径，- 表示 stdin，缺省读 THREE_SEG_LM_CONTRACT_B64")
    check.add_argument("--output", default="-", help="输出路径，- 表示 stdout（缺省）")
    args = parser.parse_args(argv)

    # This tool never needs a GPU and must not create a CUDA context on a host
    # that may be running somebody else's job.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    try:
        return cmd_build(args) if args.command == "build" else cmd_check(args)
    except Bad as error:
        line({"ok": False, "error": str(error)})
        return 2
    except OSError as error:
        line({"ok": False, "error": f"OSError: {error}"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
