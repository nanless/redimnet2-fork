"""Immutable offline production contracts and small, stdlib-only helpers."""
from __future__ import annotations
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets


class Refused(RuntimeError):
    pass


ENGINE_MODULES = ("contracts.py", "audio_windows.py", "qwen_backend.py", "worker.py",
                  "schema.py", "sidecar.py", "work_queue.py", "report.py",
                  "source_registry.py", "manifest_builder.py", "normalization.py",
                  "alignment_observer.py", "quality_axes.py", "source_clock.py")
NATIVE_BATCH_POLICY = 'asr_fixed_cohort_fa_single/v1'
ASR_B1_FEATURE_POLICY = 'official_b1_features_then_pad/v1'
NATIVE_FORMAL_DECLARATION = dict(schema='annotation-native-formal-contract/v1',
    admission_schema='annotation-native-selected-pilot-admission/v2',
    execution_requires='authenticated_bounded_runtime_v2',default_enabled=False)


def asr_batch_profile(contract):
    """新合批只属于独立试验；原单窗和正式标注不会隐式接受批处理字段。"""
    runtime = contract['runtime']
    if runtime.get('execution_policy') != NATIVE_BATCH_POLICY:
        need('asr_batching' not in runtime, 'ASR batch declaration needs explicit native pilot policy')
        need('asr_preprocessing' not in runtime, 'B1 feature preprocessing needs explicit native policy')
        need('native_formal' not in contract, 'native formal declaration cannot change old policy')
        return None
    purpose=contract.get('purpose')
    if purpose=='silver_annotation':
        declaration=contract.get('native_formal')
        need(isinstance(declaration,dict)and declaration==NATIVE_FORMAL_DECLARATION and declaration.get('default_enabled')is False,
             'native ASR batches are pilot-only without explicit default-disabled formal declaration')
    else:
        need(purpose in {'pilot_validation','cpu_fixture'} and 'native_formal'not in contract,
             'native pilot must not contain a formal declaration')
    value = runtime.get('asr_batching')
    need(isinstance(value, dict) and set(value) == {'policy', 'max_requests', 'ordering'} and
         value['policy'] == 'fixed_cohort_until_durable/v1' and
         type(value['max_requests']) is int and value['max_requests'] in (1, 4, 8) and
         value['ordering'] == 'frames_then_row_id_within_source_group/v1', 'explicit native batch 1/4/8 and fixed ordering required')
    need(type(runtime.get('max_batch')) is int and runtime['max_batch'] == 1,
         'Ultra must retain max_batch=1 independently of ASR')
    return dict(value)


def asr_feature_profile(contract):
    """独立B1声学特征协议必须显式声明；旧合同不会被隐式修正。"""
    runtime=contract['runtime']
    if 'asr_preprocessing'not in runtime:return None
    value=runtime['asr_preprocessing']
    need(asr_batch_profile(contract)is not None and isinstance(value,dict)and value=={'policy':ASR_B1_FEATURE_POLICY},
         'explicit official B1 feature preprocessing declaration required')
    need(asr_compute_profile(contract)=={'dtype':'float32','attention':'sdpa'},
         'new B1 feature protocol initially requires reviewed FP32/SDPA')
    if contract.get('purpose')!='cpu_fixture':
        from native_b1_feature_padding import validate_source_bindings
        validate_source_bindings(contract,verify_files=False)
    return dict(value)


def engine_modules_for_contract(contract):
    return ENGINE_MODULES + (('native_transformers_batch.py',) if asr_batch_profile(contract) is not None else ()) + (
        ('native_b1_feature_padding.py',)if asr_feature_profile(contract)is not None else())


def need(condition, message):
    if not condition:
        raise Refused(message)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def native_json(value):
    """Keep invalid native values explicit; never silently convert them to 0."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_native_float": repr(value)}
    if isinstance(value, dict):
        return {str(k): native_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_json(v) for v in value]
    return value


def relative_path(raw):
    need(isinstance(raw, str) and raw and "\\" not in raw and "\x00" not in raw, "invalid relative path")
    p = PurePosixPath(raw)
    need(bool(p.parts) and not p.is_absolute() and all(x not in ("", ".", "..") for x in p.parts), "unsafe relative path")
    need(str(p) == raw, "relative path is not canonical")
    return p


def safe_join(root, raw):
    rel = relative_path(raw)
    root = Path(root).absolute()
    target = root.joinpath(*rel.parts)
    need(not root.is_symlink(), "output root symlink refused")
    current = root
    for part in rel.parts:
        current = current / part
        need(not current.is_symlink(), "output descendant symlink refused")
    need(root.resolve() in target.resolve().parents, "output escapes root")
    return target


def stat_identity(path):
    s = Path(path).stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


CONTENT_STAT_KEYS = ('inode', 'bytes', 'mtime_ns', 'ctime_ns')


def identity_matches(actual, recorded):
    """内容身份：device 是挂载命名空间标签，不参与；其余字段必须完全相等。

    返回 True 表示"只有 device 不同"（调用方据此记 device_relaxed）；
    False 表示完全一致；任何内容字段不符抛 ValueError 并点名该字段。
    """
    if not isinstance(actual, dict) or not isinstance(recorded, dict):
        raise ValueError('stat identity must be dicts')
    relaxed = actual.get('device') != recorded.get('device')
    for key in CONTENT_STAT_KEYS:
        if key not in recorded:
            raise ValueError('recorded stat identity missing field: ' + key)
        if actual.get(key) != recorded[key]:
            raise ValueError('source content changed: %s expected=%r actual=%r'
                             % (key, recorded[key], actual.get(key)))
    return relaxed


def atomic_new(path, value):
    """Atomic, no-replace publication; an interrupted tmp remains as evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else canonical_bytes(value)
    tmp = path.with_name("." + path.name + "." + secrets.token_hex(12) + ".tmp")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        # Do not delete the different losing attempt or overwrite a winner.
        if path.read_bytes() != data:
            raise Refused("different immutable publication already exists: " + str(path))
    else:
        fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    tmp.unlink()
    return hashlib.sha256(data).hexdigest()


def validate_execution_policy(contract):
    runtime = contract["runtime"]; policy = runtime.get("execution_policy")
    batch = asr_batch_profile(contract)
    if batch is not None:
        asr_feature_profile(contract)
        return policy
    if contract.get("purpose") == "silver_annotation":
        need(policy == "single_window_independent" and runtime.get("max_batch") == 1,
             "silver annotation requires single_window_independent max_batch=1; batch cohort is an observed semantic input")
    if policy == "single_window_independent":
        need(runtime.get("max_batch") == 1, "single-window policy cannot execute multirow batches")
    elif contract.get("purpose") != "cpu_fixture":
        need(contract.get("purpose") == "pilot_validation" and policy == "batch_sensitive_exploratory",
             "non-single-window batching is exploratory pilot evidence only")
    return policy


def asr_compute_profile(contract):
    """转写数值方式独立声明；缺省严格保留历史FP32/eager行为。

    此字段不修改Ultra定位的FP32/eager设置，也不证明新数值方式
    已通过质量验收。实际模型及资源证明仍由独立试验核实。
    """
    runtime = contract["runtime"]
    profile = runtime.get("asr_compute", {"dtype": "float32", "attention": "eager"})
    need(isinstance(profile, dict) and set(profile) == {"dtype", "attention"},
         "ASR compute profile requires exactly dtype and attention")
    need(isinstance(profile["dtype"], str) and profile["dtype"] in {"float32", "bfloat16"}, "unsupported ASR dtype")
    need(isinstance(profile["attention"], str) and profile["attention"] in {"eager", "sdpa"}, "unsupported ASR attention")
    return dict(profile)


def read_contract(path, expected_sha=None):
    path = Path(path)
    if expected_sha:
        need(sha_file(path) == expected_sha, "production contract bytes changed")
    value = json.loads(path.read_text())
    need(value.get("schema") == "redimnet2-production-contract/v2", "wrong production contract schema")
    relative_path(value.get("version"))
    need("/" not in value["version"], "version must be one directory name")
    need(value.get("purpose") in {"silver_annotation", "pilot_validation", "cpu_fixture"}, "unknown purpose")
    if value["purpose"] != "cpu_fixture":
        pins = value.get("engine_source_files", {})
        need(set(pins) == set(engine_modules_for_contract(value)), "production semantic engine source pins missing")
        for name, expected in pins.items():
            need(sha_file(Path(__file__).resolve().parent / name) == expected, "semantic engine bytes changed; use a new contract/version: " + name)
    need(value.get("model_API_calls_allowed") == 0, "formal model API must be zero")
    need(value.get("release_training_labels", False) is False, "automatic training release is not supported")
    runtime = value["runtime"]
    need(runtime.get("dtype") == "float32" and runtime.get("attention") == "eager" and runtime.get("TF32") is False, "unvalidated precision/backend")
    asr_compute_profile(value)
    need(type(runtime.get("max_batch")) is int and 1 <= runtime["max_batch"] <= 4, "batch must be measured range 1..4")
    validate_execution_policy(value)
    if value["purpose"] != "cpu_fixture":
        from source_clock import TIME_MAPPING
        from normalization import POLICY, SUPPORTED_POLICIES
        need(runtime.get("boundary_mapping") == TIME_MAPPING, "new engine requires explicit decimal-rational source clock; freeze a new contract")
        need(runtime.get("normalization_profile", POLICY) in SUPPORTED_POLICIES, "unknown explicit normalization profile")
        alignment = value.get("alignment", {})
        need(alignment.get("mode") == "ultra_no_head" and alignment.get("head_used") is False, "only measured Ultra no-head is admitted")
        need(alignment.get("repo_id") == "17slever17/Qwen3-ForcedAligner-Ultra" and alignment.get("revision") == "765dc1d714e4c80196782a5df5bfbcb3e1416736", "Ultra author identity differs")
        need(value["models"]["aligner"]["files"].get("model.safetensors", {}).get("sha256") == "e6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f", "Ultra main weights identity differs")
        need(runtime.get("max_batch") == 1 and
             (runtime.get("execution_policy") == "single_window_independent" or asr_batch_profile(value) is not None),
             "Ultra must remain single-window; native ASR policy is pilot-only")
    need(1 <= runtime.get("length_ratio_limit", 0) <= 4, "invalid batch length ratio")
    need(type(runtime.get("max_new_tokens")) is int and runtime["max_new_tokens"] > 0, "invalid token bound")
    window = runtime["window"]
    core, context = window["core_seconds"], window["context_seconds"]
    need(isinstance(core, (int, float)) and isinstance(context, (int, float)) and core > 0 and context >= 0, "invalid window")
    need(core + 2 * context <= 30, "local Qwen/FA window must not exceed 30 seconds")
    need(set(value["models"]) == {"asr", "aligner"}, "exact offline ASR/FA models required")
    for name, model in value["models"].items():
        need(Path(model["path"]).is_absolute() and model.get("files"), "model must be local and pinned: " + name)
        for rel, pin in model["files"].items():
            relative_path(rel)
            need(type(pin["bytes"]) is int and pin["bytes"] >= 0 and len(pin["sha256"]) == 64, "bad model file pin")
    return value


def verify_models(contract, full=False):
    facts = {}
    for name, model in contract["models"].items():
        facts[name] = {}
        for rel, pin in model["files"].items():
            path = Path(model["path"]) / rel
            before = stat_identity(path)
            need(before["bytes"] == pin["bytes"], "model bytes differ: " + str(path))
            if full:
                need(sha_file(path) == pin["sha256"], "model SHA differs: " + str(path))
            need(before == stat_identity(path), "model changed during audit")
            facts[name][str(path)] = before | {"sha256": pin["sha256"], "full_sha_checked": full}
    return facts


def verify_model_stats(facts):
    for files in facts.values():
        for path, before in files.items():
            need(stat_identity(path) == {k: before[k] for k in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")}, "model changed after admission: " + path)


def job_key(source, contract_sha):
    need(source.get("source_id"), "source ID missing")
    return digest({"source_id": source["source_id"], "source_record_sha256": digest(source), "contract_sha256": contract_sha})


def model_provenance(contract):
    return {name: {"path": value["path"], "files_sha256": digest(value["files"]),
                   "files": value["files"]} for name, value in contract["models"].items()}
