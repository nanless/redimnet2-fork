#!/usr/bin/env python3
"""CPU-only checks for a separate LM continuation. Default CLI never trains."""
import argparse
import ast
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from dataclasses import dataclass


class Refused(RuntimeError):
    pass


@dataclass(frozen=True)
class Policy:
    task_id: str = "t-20261019232824-8zqrl"
    parent_root: str = "/root/experiments/redimnet2/b6-mix-official-20261019/formal-main-7afb821-zero-20261019-2327"
    original_repo: str = "/root/code/github_repos/redimnet2-fork"
    commit: str = "7afb821ac7b0062c7d6edda0317d5dc67bbb15f8"
    manifest_sha: str = "5160f99e33caccf373563b6bc695d10f304044b780d10b5c40f6a4b0fab102d9"
    source_count: int = 95
    source_root: str = "/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/source-7afb821"
    source_tree_count: int = 189
    source_tree_inventory_sha: str = "cf2cc7be726504f9d97be5b85cf7259cc9f29bd2a9d66075e02f0fde6b1d1330"
    evidence_max_age_seconds: int = 600
    batches_per_epoch: int = 15001
    total_epochs: int = 120
    attempt: str = "20261019T160535-718"
    reviewed_ledger_file: str = "/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/event-review-20261023-v4/reviewed_events.json"
    reviewed_ledger_sha256: str = "527c0bc069e8c9a3cb695b0fc229ff1c2a3c2856f3d56089c8c3f81db5bcc11c"

    decoder_library: str = "/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/event-review-20261023-v4/runtime_audio/lib/libsndfile.so.1"


UTC = dt.timezone.utc
TERMINAL = {"Succeeded", "Failed", "Stopped"}


def need(condition, message):
    if not condition:
        raise Refused(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open() as f:
        value = json.load(f)
    need(isinstance(value, dict), f"JSON must be an object: {path}")
    return value


def absolute(path):
    need(isinstance(path, (str, os.PathLike)) and bool(str(path)), "Nonempty absolute path required")
    p = Path(path)
    need(p.is_absolute(), f"Absolute path required: {p}")
    return p


def beneath(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_source_tree(contract, policy):
    """Verify the complete immutable tree before a fresh interpreter can start."""
    source = absolute(contract["source_root"])
    need(str(source) == policy.source_root and source.resolve(strict=True) == source, "Wrong exact independent source path or symlink ancestor")
    provenance = source.parent / "provenance"
    need(provenance.is_dir() and not provenance.is_symlink(), "Source provenance directory must be real")
    inventory = provenance / "source_tree_inventory.json"
    for file in (inventory, provenance / "source_identity.json"):
        need(file.is_file() and not file.is_symlink() and stat.S_IMODE(file.stat().st_mode) == 0o444, "Source provenance must be a real read-only file")
    need(sha(inventory) == policy.source_tree_inventory_sha, "Complete source inventory bytes changed")
    value = read_json(inventory)
    need(value.get("source_root") == str(source) and value.get("frozen_commit") == policy.commit, "Complete source inventory identity differs")
    records = value.get("files")
    need(isinstance(records, list) and len(records) == policy.source_tree_count, "Wrong complete source member count")
    expected, directories = {}, set()
    for item in records:
        relative = item.get("relative")
        need(isinstance(relative, str), "Invalid complete source relative path")
        path = PurePosixPath(relative)
        need(not path.is_absolute() and path.parts and ".." not in path.parts and path.as_posix() == relative and relative not in expected, "Unsafe/duplicate complete source path")
        need(item.get("mode") in (0o444, 0o555) and isinstance(item.get("bytes"), int) and item["bytes"] >= 0 and re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")), "Invalid complete source record")
        expected[relative] = item
        for parent in path.parents:
            if parent != PurePosixPath("."): directories.add(str(parent))
    seen_files, seen_dirs = set(), set()
    pending = [source]
    while pending:
        directory = pending.pop()
        need(not directory.is_symlink() and stat.S_ISDIR(directory.lstat().st_mode) and stat.S_IMODE(directory.stat().st_mode) == 0o555, "Source directory must be real and mode555")
        for member in directory.iterdir():
            relative = member.relative_to(source).as_posix(); mode = member.lstat().st_mode
            if stat.S_ISDIR(mode):
                need(relative in directories, "Unexpected complete source directory: " + relative)
                seen_dirs.add(relative); pending.append(member)
            else:
                need(stat.S_ISREG(mode) and relative in expected, "Unexpected/symlink complete source member: " + relative)
                item = expected[relative]
                need(stat.S_IMODE(mode) == item["mode"] and member.stat().st_size == item["bytes"] and sha(member) == item["sha256"], "Complete source bytes/mode differ: " + relative)
                seen_files.add(relative)
    need(seen_files == set(expected) and seen_dirs == directories, "Complete source tree has missing members")
    identity = read_json(provenance / "source_identity.json")
    need(identity.get("source_root") == str(source) and identity.get("frozen_commit") == policy.commit and identity.get("parent_run") == policy.parent_root and identity.get("parent_task_id") == policy.task_id and identity.get("original_checksum_sha256") == policy.manifest_sha and identity.get("archive_files") == policy.source_tree_count and identity.get("frozen_files_matching") == policy.source_count, "Prepared source identity differs")
    return {"files_checked": len(seen_files), "inventory_sha256": policy.source_tree_inventory_sha, "identity_sha256": sha(provenance / "source_identity.json"), "readonly": True}


def validate_source(contract, policy):
    """Map immutable original absolute entries; never rewrite the original file."""
    manifest = absolute(contract["source_manifest"])
    need(sha(manifest) == policy.manifest_sha, "Frozen manifest bytes changed")
    root = absolute(contract["source_root"]).resolve(strict=True)
    active = Path(policy.original_repo).resolve()
    need(not beneath(root, active) and not beneath(active, root), "Active repository cannot overlap continuation source")
    need(contract["source_commit"] == policy.commit, "Wrong source commit")
    entries, seen = [], set()
    prefix = PurePosixPath(policy.original_repo)
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        need(match is not None, "Unsupported or escaped frozen manifest line")
        digest, raw = match.groups()
        source = PurePosixPath(raw)
        need(source.is_absolute() and ".." not in source.parts, "Unsafe original source path")
        try:
            relative = source.relative_to(prefix)
        except ValueError as exc:
            raise Refused("Frozen source outside original repository") from exc
        need(relative.parts and str(relative) not in seen, "Duplicate or empty frozen source entry")
        seen.add(str(relative))
        target = root.joinpath(*relative.parts)
        resolved = target.resolve(strict=True)
        need(beneath(resolved, root), "Snapshot symlink escapes independent source root")
        need(target.is_file() and sha(target) == digest, f"Frozen source mismatch: {relative}")
        entries.append({"original": raw, "relative": str(relative), "snapshot": str(resolved), "sha256": digest})
    need(len(entries) == policy.source_count, "Wrong frozen source entry count")
    validate_source_tree(contract, policy)
    return entries


def validate_terminal_evidence(contract, policy, now):
    for field in ("platform_state_file", "writer_state_file"):
        evidence = read_json(absolute(contract[field]))
        need(evidence.get("task_id") == policy.task_id, "Evidence belongs to another task")
        need(evidence.get("parent_run_root") == policy.parent_root, "Evidence belongs to another run")
        need(evidence.get("source") in {"volcengine-console", "volcengine-api"}, "Unknown evidence source")
        try:
            observed = dt.datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00"))
            need(observed.tzinfo is not None, "Evidence timestamp requires timezone")
            age = (now - observed).total_seconds()
        except (ValueError, KeyError, TypeError) as exc:
            raise Refused("Invalid evidence timestamp") from exc
        need(-5 <= age <= policy.evidence_max_age_seconds, "Missing, stale or future task/writer evidence")
        if field == "platform_state_file":
            need(evidence.get("state") in TERMINAL, "Original task is not confirmed terminal")
            need(evidence.get("active_workers") == 0, "Original platform task still has active workers")
        else:
            need(evidence.get("writers") == [], "Original run still has a writer")
            need(evidence.get("probe_complete") is True, "Writer probe is incomplete")


@contextlib.contextmanager
def parent_locks(root):
    """Read-only FDs, no writes/truncation to the parent's existing lock files."""
    handles = []
    try:
        for rel in (".reproduction.lock", "ptn/.cloud_train.lock"):
            handle = (root / rel).open("rb")
            handles.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Refused(f"Original run lock is held: {rel}") from exc
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def load_reviewed_ledger(contract, policy):
    """Code-pinned review; contract cannot add events or select another ledger."""
    need(contract.get("schema") == 2, "Unsupported contract schema")
    need(contract.get("parent_task_id") == policy.task_id and contract.get("parent_run_root") == policy.parent_root and contract.get("parent_attempt") == policy.attempt, "Wrong reviewed event parent provenance")
    path = absolute(contract.get("reviewed_event_ledger"))
    need(str(path) == policy.reviewed_ledger_file and path.resolve(strict=True) == path, "Wrong exact reviewed ledger path or symlink ancestor")
    need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444, "Reviewed ledger must be a real read-only file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Reviewed ledger changed during reading")
    need(hashlib.sha256(raw).hexdigest() == policy.reviewed_ledger_sha256, "Reviewed ledger bytes differ from code-pinned review")
    ledger = json.loads(raw)
    need(isinstance(ledger, dict), "Reviewed ledger must be an object")
    need(ledger.get("schema") == 1 and ledger.get("parent_task_id") == policy.task_id and ledger.get("parent_run_root") == policy.parent_root and ledger.get("parent_attempt") == policy.attempt, "Reviewed ledger identity differs")
    identity = read_json(Path(policy.parent_root) / "run_identity.json")
    need(identity == {"mode": "formal", "run_root": policy.parent_root, "source_checksum_sha256": policy.manifest_sha}, "Original run identity differs")
    audio = ledger.get("audio")
    need(isinstance(audio, list) and len(audio) == 3 and len({r.get("sample_key") for r in audio}) == 3, "Reviewed audio identities incomplete")
    by_key = {r["sample_key"]: r for r in audio}
    need(set(by_key) == {e["sample"]["key"] for e in ledger["events"]}, "Reviewed audio/event identity mismatch")
    need(all(by_key[e["sample"]["key"]]["path"] == e["sample"]["wav"] for e in ledger["events"]), "Reviewed audio path/event identity mismatch")
    return ledger


def audit_reviewed_events(contract, policy=Policy(), audio_checker=None, require_stable=False, decoder_runtime_checker=None):
    """Read-only occurrence/audio review, NEVER readiness to execute LM."""
    from reviewed_events import classify_reviewed_log, ReviewRefused
    from audio_review import audit_audio_files, AudioRefused
    from decoder_runtime import audit_loaded_decoder, RuntimeRefused
    ledger = load_reviewed_ledger(contract, policy)
    stdout = Path(policy.parent_root) / "checks" / policy.attempt / "ptn.stdout.log"
    need(stdout.resolve(strict=True) == stdout and stdout.is_file(), "Reviewed stdout must be the exact original file")
    before = stdout.stat()
    with stdout.open("rb") as handle: raw = handle.read(before.st_size)
    need(len(raw) == before.st_size, "Original stdout truncated while reading")
    try:
        events = classify_reviewed_log(raw, ledger)
        runtime = (decoder_runtime_checker or audit_loaded_decoder)(policy)
        audio = (audio_checker or audit_audio_files)(ledger)
    except (ReviewRefused, AudioRefused, RuntimeRefused) as exc:
        refusal = Refused(str(exc))
        if hasattr(exc, "evidence"): refusal.review_evidence = exc.evidence
        raise refusal from exc
    after = stdout.stat()
    need(not require_stable or (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) == (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev), "Original stdout changed during terminal review")
    return {"ready": False, "events_review_accepted": True, "review_id": ledger["review_id"],
            "reviewed_ledger_sha256": policy.reviewed_ledger_sha256, "events": events, "audio_audit": audio, "decoder_runtime": runtime,
            "stdout_scanned_bytes": len(raw), "stdout_scanned_sha256": hashlib.sha256(raw).hexdigest(),
            "stdout_grew_during_review": after.st_size != before.st_size,
            "meaning": "Read-only exact occurrence and current audio review only; PTN120, terminal task, writers, locks, checkpoints and LM execution are NOT granted."}


def model_summary(path, expected_head):
    """Trusted project weights, CPU only; called only after terminal/writer gates."""
    import torch
    torch.set_num_threads(1)
    obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    need(isinstance(obj, dict) and set(("model", "optimizer", "scaler", "epoch")) <= set(obj), "Incomplete checkpoint")
    need(isinstance(obj["model"], dict) and obj["model"], "Missing model state")
    need(isinstance(obj["optimizer"], dict) and obj["optimizer"].get("state") and obj["optimizer"].get("param_groups"), "Missing optimizer state")
    scaler = obj["scaler"]
    need(isinstance(scaler, dict) and all(k in scaler for k in ("scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker")), "Incomplete scaler state")
    need(all(isinstance(v, (int, float)) and math.isfinite(v) for v in scaler.values()), "Nonfinite scaler state")
    need(scaler["scale"] > 0 and scaler["growth_factor"] > 1 and 0 < scaler["backoff_factor"] < 1, "Invalid scaler range")
    for group in (obj["model"], *obj["optimizer"].get("state", {}).values()):
        for value in group.values():
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                need(torch.isfinite(value).all().item(), "Nonfinite checkpoint tensor")
    head = obj["model"].get("projection.weight")
    need(head is not None and list(head.shape) == expected_head, "Unexpected model head shape")
    return {"epoch": obj["epoch"], "head_shape": list(head.shape)}


def checkpoint_summary(path):
    return model_summary(path, [357822, 192])


def verify_import_paths(contract, policy=Policy()):
    """Fresh configured interpreter, no visible CUDA, no bytecode writes."""
    from decoder_runtime import verify_runtime_files, RuntimeRefused
    try:
        runtime_files = verify_runtime_files(policy)
    except RuntimeRefused as exc:
        raise Refused(str(exc)) from exc
    root = absolute(contract["source_root"]).resolve(strict=True)
    python = absolute(contract["python_bin"])
    need(python.is_file() and os.access(python, os.X_OK), "Configured Python is not executable")
    code = r'''
import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); active=Path(sys.argv[2]).resolve()
def under(p,r):
    try: p.relative_to(r); return True
    except ValueError: return False
sys.path=[str(root),str(root/'applications/speaker_rec')]+[s for s in sys.path if s and not under(Path(s).resolve(),active)]
names=['redimnet2','wespeaker_lite.dataset.processor','wespeaker_lite.utils.executor','wespeaker_lite.bin.train']
out={}
for name in names:
    mod=importlib.import_module(name); f=getattr(mod,'__file__',None)
    if not f or not under(Path(f).resolve(),root): raise RuntimeError('Source import escaped snapshot: '+name+' '+str(f))
    out[name]=str(Path(f).resolve())
sys.path.insert(0,sys.argv[3])
from decoder_runtime import audit_loaded_decoder
from types import SimpleNamespace
out['_decoder_runtime']=audit_loaded_decoder(SimpleNamespace(decoder_library=sys.argv[4]))
print('SOURCE_IMPORT_AUDIT '+json.dumps(out))
'''
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(root) + ":" + str(root / "applications/speaker_rec"),
                "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env.pop("LD_PRELOAD", None); env.pop("LD_AUDIT", None)
    env["LD_LIBRARY_PATH"] = str(Path(policy.decoder_library).parent)
    proc = subprocess.run([str(python), "-c", code, str(root), policy.original_repo, str(Path(__file__).resolve().parent), policy.decoder_library], cwd=root,
                          env=env, capture_output=True, text=True, timeout=180)
    need(proc.returncode == 0, "Independent interpreter/source import failed: " + proc.stderr[-3000:])
    markers = [line for line in proc.stdout.splitlines() if line.startswith("SOURCE_IMPORT_AUDIT ")]
    need(len(markers) == 1, "Missing import-path audit result")
    result = json.loads(markers[0].split(" ", 1)[1])
    decoder_runtime = result.pop("_decoder_runtime")
    need(set(result) == {"redimnet2", "wespeaker_lite.dataset.processor", "wespeaker_lite.utils.executor", "wespeaker_lite.bin.train"}, "Incomplete module-path audit")
    need(all(beneath(Path(v).resolve(), root) for v in result.values()), "Import-path audit escaped source root")
    return {"python": str(python), "modules": result, "cuda_visible_devices": "", "bytecode_writes": False, "decoder_runtime": decoder_runtime, "runtime_files": runtime_files}


def verify_checkpoint(path, epoch, reader):
    need(path.is_file(), f"PTN{epoch} checkpoint is missing")
    need(not path.is_symlink(), "Cross-run checkpoint symlink is not accepted")
    before = path.stat()
    summary = reader(path)
    need(summary.get("epoch") == epoch and summary.get("head_shape") == [357822, 192], "Wrong PTN checkpoint epoch/head identity")
    digest = sha(path)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Checkpoint changed while checking")
    return {"path": str(path), "sha256": digest, "epoch": epoch, "bytes": before.st_size}


def preflight(contract, policy=Policy(), now=None, reader=checkpoint_summary, locks_held=False, import_checker=verify_import_paths, event_audio_checker=None, decoder_runtime_checker=None):
    fixed_now = now
    now = fixed_now or dt.datetime.now(UTC)
    need(contract.get("schema") == 2, "Unsupported contract schema")
    need(contract.get("parent_task_id") == policy.task_id and contract.get("parent_run_root") == policy.parent_root, "Wrong parent provenance")
    need(contract.get("parent_attempt") == policy.attempt, "Wrong parent attempt")
    need(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", contract.get("continuation_id", "")) is not None, "Invalid continuation identity")
    root = Path(policy.parent_root).resolve(strict=True)
    output = absolute(contract["output_root"])
    resolved_output = output.resolve()
    source = absolute(contract["source_root"]).resolve(strict=True)
    need(not output.exists(), "One-time LM output already exists; no resume/overwrite")
    need(not beneath(resolved_output, root) and not beneath(root, resolved_output), "Continuation output overlaps original run")
    need(not beneath(resolved_output, source) and not beneath(source, resolved_output), "Continuation output overlaps source")
    active = Path(policy.original_repo).resolve()
    need(not beneath(resolved_output, active) and not beneath(active, resolved_output), "Continuation output overlaps active repository")
    identity = read_json(root / "run_identity.json")
    need(identity == {"mode": "formal", "run_root": policy.parent_root, "source_checksum_sha256": policy.manifest_sha}, "Original run identity differs")
    need((root / "checks/start_policy.txt").read_text().strip() == "fresh_zero_no_resume", "Wrong original start policy")
    need((root / "checks/source_git_commit.txt").read_text().strip() == policy.commit, "Wrong original commit")
    validate_terminal_evidence(contract, policy, now)
    mapped = validate_source(contract, policy)
    imports = import_checker(contract, policy)
    # A new runtime must use this verified snapshot, not active repository files.
    required = ["applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh",
                "applications/speaker_rec/scripts/cloud/check_b6_lm_init.py",
                "applications/speaker_rec/scripts/cloud/evaluate_b6_mix_reproduction.sh"]
    need(set(required) <= {v["relative"] for v in mapped}, "Required LM runtime file is not frozen")
    accepted = read_json(root / "checks/accepted_smoke_success.json")
    smoke_file = absolute(contract["smoke_success_file"])
    need(sha(smoke_file) == contract["expected_smoke_sha256"], "Wrong smoke proof bytes")
    smoke = read_json(smoke_file)
    need(smoke == accepted and smoke.get("success") is True and smoke.get("mode") == "smoke", "Smoke proof is not original accepted proof")
    need(smoke.get("source_checksum_sha256") == policy.manifest_sha, "Smoke/source lineage differs")
    for key, expected_key, smoke_key in [("data_audit_file", "expected_data_audit_sha256", "data_audit_sha256"), ("official_reference_file", "expected_official_sha256", "official_reference_sha256")]:
        need(sha(absolute(contract[key])) == contract[expected_key] == accepted.get(smoke_key), f"Wrong frozen dependency: {key}")
    audit = read_json(contract["data_audit_file"])
    need(audit.get("ok") is True and audit.get("lm_speakers_subset_of_ptn") is True, "Original data audit failed")
    train_text = (root / "ptn/train.log").read_text()
    need(re.search(r"\|\s*120\|\s*15001\|", train_text) is not None, "PTN120 has not completed its data batches")
    with (root / "ptn/val_log.csv").open() as f:
        values = list(csv.DictReader(f))
    need([int(v["epoch"]) for v in values] == list(range(1, 121)), "PTN validation is incomplete, duplicated or out of order")
    need(all(math.isfinite(float(v["vox1-eer-8s"])) and 0 <= float(v["vox1-eer-8s"]) <= 100 for v in values), "Invalid PTN validation metric")
    stdout = root / "checks" / contract["parent_attempt"] / "ptn.stdout.log"
    need(beneath(stdout.resolve(strict=True), (root / "checks").resolve()), "Attempt path escapes original checks")
    event_audit = audit_reviewed_events(contract, policy, audio_checker=event_audio_checker, require_stable=True, decoder_runtime_checker=decoder_runtime_checker)
    stdout_text = stdout.read_text(errors="strict")
    need(sha(stdout) == event_audit["stdout_scanned_sha256"], "Original stdout changed after occurrence review")
    caught = event_audit["events"]
    need("TRAIN_END status=ok reason=finished exit=0 last_epoch=120" in stdout_text, "PTN launcher did not report normal completion")
    with contextlib.nullcontext() if locks_held else parent_locks(root):
        checkpoints = [verify_checkpoint(root / f"ptn/models/model_{e}.pt", e, reader) for e in (115, 120)]
    # Re-check evidence freshness after potentially lengthy source/checkpoint I/O.
    validate_terminal_evidence(contract, policy, fixed_now or dt.datetime.now(UTC))
    return {"ready": True, "continuation_id": contract["continuation_id"], "parent_task_id": policy.task_id,
            "parent_run_root": policy.parent_root, "source_commit": policy.commit,
            "original_manifest_sha256": policy.manifest_sha, "source_root": str(source),
            "mapped_source": mapped, "complete_source_audit": validate_source_tree(contract, policy), "import_audit": imports, "checkpoints": checkpoints, "caught_read_events": caught,
            "reviewed_event_audit": event_audit, "stdout_sha256": sha(stdout), "output_root": str(output), "checked_at": dt.datetime.now(UTC).isoformat(),
            "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "dependency_identity": {k: {"path": str(absolute(contract[k]).resolve(strict=True)), "sha256": sha(contract[k])} for k in ("source_manifest", "smoke_success_file", "data_audit_file", "official_reference_file", "platform_state_file", "writer_state_file")},
            "meaning": "CPU preflight only. No LM/GPU process started."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-reviewed-events-only", action="store_true", help="Exact event/audio audit only; ready always false; no locks/checkpoints/LM")
    modes.add_argument("--audit-source-only", action="store_true", help="Audit mapped snapshot/imports only; never grants permission to start LM")
    args = parser.parse_args()
    try:
        contract = read_json(args.contract)
        if args.audit_reviewed_events_only:
            result = audit_reviewed_events(contract)
        elif args.audit_source_only:
            result = {"ready": False, "source_audit_complete": True,
                      "meaning": "Source/CPU import audit only; PTN120/terminal/writer checks NOT satisfied by this command.",
                      "mapped_source": validate_source(contract, Policy()), "complete_source_audit": validate_source_tree(contract, Policy()), "import_audit": verify_import_paths(contract)}
        else:
            result = preflight(contract)
    except (Refused, OSError, KeyError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"ready": False, "reason": str(exc), "review_evidence": getattr(exc, "review_evidence", None)}, ensure_ascii=False))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
