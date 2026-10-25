"""只读检查同输出恢复边界，拒绝旧保存点、脏日志及重复恢复尝试。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import stat


class ResumeRefused(ValueError):
    pass


def _stat(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _regular(path):
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ResumeRefused(f"not a regular non-symlink file: {path}")
    if path.stat().st_size == 0:
        raise ResumeRefused(f"empty file: {path}")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ResumeRefused(f"invalid integer {name}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ResumeRefused(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _constant(value):
    raise ResumeRefused(f"nonfinite JSON constant: {value}")


@contextmanager
def _inactive_writer(root, enabled):
    """仅rank0检查既有新实验锁；不创建、不写锁文件，离开前释放。"""
    if not enabled:
        yield
        return
    import fcntl
    path = root / ".training.lock"
    if path.is_symlink() or not path.is_file():
        raise ResumeRefused("same-output training lock file is missing or unsafe")
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResumeRefused("an existing training writer still owns this output") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _log(path, rank, step, batch_size, world_size, expected_exposure):
    _regular(path)
    before = _stat(path)
    sha, schedule = hashlib.sha256(), hashlib.sha256()
    count = 0
    last = None
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(16 * 1024 * 1024 + 1)
            if not raw:
                break
            if len(raw) > 16 * 1024 * 1024 or not raw.endswith(b"\n"):
                raise ResumeRefused(f"truncated or oversized exposure line: {path}")
            sha.update(raw)
            try:
                row = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_constant)
                _canonical(row)  # Also rejects overflowed nonfinite float literals.
            except (ValueError, UnicodeError, TypeError) as error:
                raise ResumeRefused(f"invalid exposure JSON at {path}:{count + 1}: {error}") from error
            if not isinstance(row, dict):
                raise ResumeRefused(f"non-object exposure line: {path}")
            count += 1
            if type(row.get("step")) is not int or row["step"] != count:
                raise ResumeRefused(f"duplicate, missing or reordered exposure step at {path}:{count}")
            if type(row.get("rank")) is not int or row["rank"] != rank:
                raise ResumeRefused(f"exposure rank differs: {path}:{count}")
            if row.get("optimizer_update") is not True:
                raise ResumeRefused(f"exposure line is not a committed optimizer update: {path}:{count}")
            if type(row.get("samples")) is not int or row["samples"] != batch_size:
                raise ResumeRefused(f"per-rank sample count differs: {path}:{count}")
            exposure = row.get("exposure_global")
            if not isinstance(exposure, dict) or type(exposure.get("samples")) is not int or exposure["samples"] != count * batch_size * world_size:
                raise ResumeRefused(f"global exposure position differs: {path}:{count}")
            seconds = row.get("seconds")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
                raise ResumeRefused(f"invalid exposure duration: {path}:{count}")
            schedule.update(_canonical([count, seconds, exposure]).encode())
            last = row
            if count > step:
                raise ResumeRefused(f"uncheckpointed optimizer update in exposure log: {path}")
    if count != step:
        raise ResumeRefused(f"exposure tail {count} does not equal checkpoint step {step}: {path}")
    if _canonical(last["exposure_global"]) != _canonical(expected_exposure):
        raise ResumeRefused(f"checkpoint and final exposure differ: {path}")
    if _stat(path) != before:
        raise ResumeRefused(f"exposure changed while checking: {path}")
    return {"rows": count, "sha256": sha.hexdigest(), "schedule_sha256": schedule.hexdigest(), "stat": before}


def check_resume_output(output, resume, *, world_size=None, rank=0, writer_lock_held=False):
    """检查后不修改任何文件；CLI在调用核心run()之前执行。

    仅允许恢复同一已存在输出目录的直接train_step_N.pt。没有跨输出/new-attempt
    隐式导入模式。核心训练仍须取得自己的写锁并验证配置/数据/source/RNG。
    writer_lock_held=True只供core内rank0取得同一输出写锁后的第二核使用，
    不向CLI暴露；跳过自冲突的锁探测，不跳过目录/日志/检查点核验。
    """
    if type(writer_lock_held) is not bool or (writer_lock_held and rank != 0):
        raise ResumeRefused("held-writer mode is only for rank0 inside the core writer lock")
    _integer(rank, "rank", minimum=0)
    if world_size is not None and rank >= _integer(world_size, "requested world_size"):
        raise ResumeRefused("rank is outside requested world size")
    raw_root, raw_checkpoint = Path(output).absolute(), Path(resume).absolute()
    root, checkpoint = raw_root.resolve(), raw_checkpoint.resolve()
    if not root.is_dir() or raw_root.is_symlink():
        raise ResumeRefused("resume requires its existing original output directory")
    if raw_checkpoint.is_symlink() or raw_checkpoint.parent != raw_root or checkpoint.parent != root:
        raise ResumeRefused("resume checkpoint must be a direct file in the same output; new attempt needs an explicit separate design")
    match = re.fullmatch(r"train_step_([1-9][0-9]*)\.pt", checkpoint.name)
    if not match:
        raise ResumeRefused("resume must use canonical train_step_N.pt, not embedding or renamed files")
    _regular(checkpoint)
    name_step = int(match[1])
    with _inactive_writer(root, rank == 0 and not writer_lock_held):
        inventory, steps = {}, {"train": set(), "embedding": set()}
        for path in root.iterdir():
            name = path.name
            if name.startswith(("train_step_", "embedding_step_")):
                found = re.fullmatch(r"(train|embedding)_step_([1-9][0-9]*)\.pt", name)
                if not found:
                    raise ResumeRefused(f"temporary or noncanonical checkpoint remains: {name}")
                _regular(path)
                inventory[name] = _stat(path)
                steps[found[1]].add(int(found[2]))
            if name.startswith("failure_"):
                raise ResumeRefused(f"previous attempt failure retained; explicit new-attempt review required: {name}")
            started = re.fullmatch(r"identity_(?:rank[0-9]+_)?start_step_([0-9]+)\.json", name)
            if started and int(started[1]) >= name_step:
                raise ResumeRefused(f"a resume attempt at/after this step already has identity evidence: {name}")
        if not steps["train"] or steps["train"] != steps["embedding"]:
            raise ResumeRefused("training/embedding published checkpoint steps are incomplete or unpaired")
        if max(steps["train"]) != name_step:
            raise ResumeRefused("resume is older than existing published output checkpoint")
        # Only the trusted same-output training format is loaded; tensor storage remains CPU-mapped.
        import torch
        before = _stat(checkpoint)
        try:
            payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        except Exception as error:
            raise ResumeRefused("unreadable same-output checkpoint") from error
        if not isinstance(payload, dict) or payload.get("schema") != "duration-training/v1":
            raise ResumeRefused("unsupported training checkpoint schema")
        step = _integer(payload.get("global_step"), "global_step")
        saved_world = _integer(payload.get("world_size"), "world_size")
        if rank >= saved_world:
            raise ResumeRefused("rank is outside checkpoint world size")
        if step != name_step:
            raise ResumeRefused("checkpoint payload step does not match filename")
        if world_size is not None and saved_world != _integer(world_size, "requested world_size"):
            raise ResumeRefused("requested world size differs from checkpoint")
        if not isinstance(payload.get("config"), dict):
            raise ResumeRefused("checkpoint config is not a mapping")
        batch_size = _integer(payload["config"].get("batch_size"), "batch_size")
        if type(payload.get("sampler_global_position")) is not int or payload["sampler_global_position"] != step * saved_world * batch_size:
            raise ResumeRefused("checkpoint sampler position differs")
        exposure = payload.get("exposure")
        if not isinstance(exposure, dict) or type(exposure.get("samples")) is not int or exposure["samples"] != step * saved_world * batch_size:
            raise ResumeRefused("checkpoint global exposure differs")
        logs = {}
        for path in root.glob("exposure_rank*.jsonl"):
            found = re.fullmatch(r"exposure_rank(0|[1-9][0-9]*)\.jsonl", path.name)
            if not found:
                raise ResumeRefused(f"noncanonical exposure filename: {path.name}")
            logs[int(found[1])] = path
        if set(logs) != set(range(saved_world)):
            raise ResumeRefused("exposure rank streams are missing or from another world size")
        log_evidence = {str(r): _log(logs[r], r, step, batch_size, saved_world, exposure) for r in range(saved_world)}
        if len({item["schedule_sha256"] for item in log_evidence.values()}) != 1:
            raise ResumeRefused("rank global exposure/duration histories differ")
        if _stat(checkpoint) != before:
            raise ResumeRefused("checkpoint changed during resume guard")
        final_inventory = {p.name: _stat(p) for p in root.iterdir() if p.name.startswith(("train_step_", "embedding_step_"))}
        if final_inventory != inventory:
            raise ResumeRefused("checkpoint directory changed during resume guard")
        return {"mode": "same-output-only", "global_step": step, "world_size": saved_world,
                "checkpoint": str(checkpoint), "checkpoint_stat": before,
                "published_steps": sorted(steps["train"]), "exposure_logs": log_evidence,
                "writer_lock_checked": rank == 0 and not writer_lock_held,
                "writer_lock_held": writer_lock_held,
                "limitations": "snapshot preflight; core writer lock/config/data/source/RNG checks still required; not a full tensor audit or cross-output migration"}
