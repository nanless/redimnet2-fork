"""One offline single-GPU replica; dynamic chunks and versioned sidecar output.

Called in-process by the authenticated resource bootstrap. Returning deliberately
keeps _ACTIVE_BACKEND alive until the bootstrap's terminal ready/ACK/drain ends.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from audio_windows import inspect_source, plan_windows, read_window, reuse_vad, units_from_windows, verify_decoder
from contracts import atomic_new, digest, job_key, model_provenance, native_json, need, read_contract, sha_file, stat_identity, utcnow, asr_batch_profile, safe_join
from schema import build_record
from sidecar import SidecarStore, StageCache
from work_queue import WorkQueue

_ACTIVE_BACKEND = None

PIPELINE_TIMERS = ('source_lookup', 'source_audit', 'VAD_and_plan', 'window_read',
                   'stage_cache', 'sidecar_publish', 'journal', 'chunk_finish')

# Whisper 特征提取器 n_fft=400、center=True 时在两侧各补 200 帧；torch.stft 内部的 F.pad
# 要求 pad < 输入长度，所以 <201 帧必然抛 RuntimeError。这里取 400（= 一个完整分析窗）
# 作为"可标注下限"，而不是刚好 201：12.5–25 ms 的窗口即使侥幸不崩也没有标注价值。
MIN_WINDOW_FRAMES = 400


def refuse_short_window(window, frames, context, counters):
    """把进不了 ASR 的短窗写成失败窗：不抛异常、不丢源、不进批次。

    形态与 process_group 里 read_window 失败的兜底一致（schema 的 preprocessing_failed
    谓词只认这四个字段），并把帧数与下限写进 errors，便于事后按原因统计。
    """
    error = ('asr_input_too_short: model input has %d frames at 16000 Hz (%.3f ms); '
             'WhisperFeatureExtractor needs >= %d frames (n_fft=400, center pad 200 per side)'
             % (frames, frames / 16.0, MIN_WINDOW_FRAMES))
    window.update(preprocessing_status="failed", input_pcm_sha256=None,
                  asr={"status": "failed", "raw_text": "", "normalized_text": "", "language": None,
                       "native": {"eos_observed": False}, "error": error},
                  alignment={"status": "not_applicable", "reason": "asr_input_too_short", "target_text": "",
                             "items": [], "native_timestamps_ms": [], "fixed_timestamps_ms": [],
                             "repair_changed": False})
    context["errors"].append({"stage": "asr_input_too_short", "window_id": window["window_id"],
                              "frames": frames, "min_frames": MIN_WINDOW_FRAMES, "traceback": error})
    counters["short_window_refused"] = counters.get("short_window_refused", 0) + 1
    return True


def timed_call(counters, stage, function, *args, **kwargs):
    """记录互不包含模型阶段的CPU/存储调用墙钟；异常仍完整向外传播。"""
    started = time.monotonic()
    try: return function(*args, **kwargs)
    finally:
        values = counters.setdefault('pipeline_seconds', {})
        values[stage] = values.get(stage, 0.0) + time.monotonic() - started


def progress_performance(backend, counters, process_started_utc, process_started_monotonic, previous=None):
    """只描述此worker进程的累计值和片间增量，不拼接旧attempt计时。"""
    now = time.monotonic()
    backend_values = backend.snapshot_stats()
    selected = ('assigned_source_rows', 'sidecars_committed', 'sidecar_cache_hits',
                'stage_cache_hits', 'model_windows', 'decoded_audio_seconds', 'pipeline_seconds',
                'short_window_refused')
    selected += tuple(key for key in ('retained_previous_producer_hits','retained_previous_decision_counts') if key in counters)
    current = {'worker_elapsed_seconds': now - process_started_monotonic,
               'backend': backend_values,
               'counters': {key: copy.deepcopy(counters.get(key,0)) for key in selected}}
    def increments(value, old):
        if isinstance(value, dict):
            return {k: v for k, raw in value.items()
                    if (v := increments(raw, old.get(k) if isinstance(old, dict) else None)) is not None}
        if type(value) in (int, float):
            difference = value - (old if type(old) in (int, float) else 0)
            need(difference >= 0, 'worker cumulative timing/count moved backwards')
            return difference
        return None
    # 参数/配置读回属于身份信息，不能对模型形状、dtype或名字计算“增量”。
    increment_source = copy.deepcopy(current)
    for key in ('loaded_compute', 'first_call_compute_observation'):
        increment_source['backend'].pop(key, None)
    return {'schema': 'annotation-worker-performance/v1', 'scope': 'this_worker_process_only',
            'worker_pid': os.getpid(), 'worker_started_utc': process_started_utc,
            'observed_utc': utcnow(), 'cumulative': current,
            'since_previous_chunk': increments(increment_source, previous),
            'prior_attempt_elapsed_included': False}, current


def _start_ticks(pid):
    raw = Path("/proc/%d/stat" % pid).read_text()
    return int(raw[raw.rfind(")") + 2:].split()[19])


def authenticate_resource(args):
    need(args.ticket and args.ticket_sha and args.resource_attestation, "GPU worker requires resource bootstrap evidence")
    need(sha_file(args.ticket) == args.ticket_sha, "resource ticket bytes changed")
    ticket = json.loads(Path(args.ticket).read_text())
    need(ticket["parent_pid"] == os.getppid() and ticket["parent_start_time"] == _start_ticks(os.getppid()), "actual resource parent differs")
    need(ticket["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "resource boot identity differs")
    need(0 <= time.monotonic() - ticket["created_monotonic"] <= 900, "resource ticket stale/future")
    need(str(ticket["worker_id"]) == str(args.worker_id) and ticket["contract_sha256"] == args.contract_sha
         and ticket["index_sha256"] == args.index_sha, "resource worker/input contract differs")
    uuid = ticket["target_uuid"]
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid, "worker must see exactly its assigned UUID")
    attestation = json.loads(Path(args.resource_attestation).read_text())
    need(attestation["schema"] == "open-annotation-resource-attestation/v1"
         and attestation["nonce"] == ticket["nonce"] and str(attestation["worker_id"]) == str(args.worker_id)
         and attestation["UUIDs"] == [uuid] and attestation["known_container_worker_pid"] == os.getpid()
         and attestation["ticket_sha"] == args.ticket_sha, "active CUDA context attestation differs")
    need(len(attestation["challenge_attestation_sha"]) == 64, "challenge digest missing")
    for entry in ticket["locks"]:
        s = Path(entry["path"]).stat(); held = Path("/proc/%d/fd/%d" % (ticket["parent_pid"], entry["fd"])).stat()
        need((s.st_dev, s.st_ino) == (entry["device"], entry["inode"]) == (held.st_dev, held.st_ino), "resource lock identity differs")
    bundle = Path(__file__).resolve().parent / "bundle_files.json"
    if bundle.is_file(): need(sha_file(bundle) == ticket["bundle_sha256"], "production bundle manifest differs from resource parent")
    return ticket


def _descriptor(contract, contract_sha, bundle_sha, decoder):
    return {"sha256": contract_sha, "version": contract["version"], "api_calls": 0,
            "purpose": contract["purpose"], "provenance": {"models": model_provenance(contract),
            "alignment_method": "cpu_fixture" if contract["purpose"] == "cpu_fixture" else contract.get("alignment", {}).get("mode"),
            "alignment_identity": contract.get("alignment"),
            "code_sha256": bundle_sha, "decoder": decoder, "runtime": contract["runtime"]}}


def _stage_inputs(stage, window, context, contract, contract_sha, models):
    value = {"stage": stage, "contract_sha256": contract_sha, "input_pcm_sha256": window["input_pcm_sha256"],
             "model_sample_rate": 16000, "model_input_frames": window["model_input_frames"],
             "model_files_sha256": models["asr" if stage == "asr" else "aligner"]["files_sha256"]}
    if stage == "aligner":
        value.update(target_text_sha256=digest(window["asr"]["normalized_text"]), language=window["asr"]["language"])
    return value


def process_group(sources, claim, store, cache, backend, contract, args, descriptor, counters):
    """单窗沿原缓冲；明确新试验先固定最多256窗整组，整批提交后才发布。"""
    native_profile=asr_batch_profile(contract)
    contexts = []; receipts = []
    for source in sources:
        claim.assert_member(source); counters["assigned_source_rows"] += 1
        previous = None
        if hasattr(getattr(claim, 'queue', None), 'previous_receipt'):
            previous = timed_call(counters, 'source_lookup', claim.queue.previous_receipt, claim, source)
        old = timed_call(counters, 'source_lookup', store.lookup, source)
        need(previous is None or old is None, 'ambiguous source has both old and new producer results; preserve and inspect')
        if previous is not None:
            receipts.append(previous)
            counters['retained_previous_producer_hits'] += 1
            counters['retained_previous_decision_counts'][previous['decision']] += 1
            counters['decision_counts'][previous['decision']] += 1
            timed_call(counters, 'journal', claim.journal, {'source_id':source['source_id'],
                       'status':'verified_previous_producer_retained', 'receipt':previous})
            continue
        if old is not None:
            receipts.append(old); counters["sidecar_cache_hits"] += 1; counters["decision_counts"][old["decision"]] += 1
            timed_call(counters, 'journal', claim.journal, {"source_id": source["source_id"], "status": "verified_sidecar_cache_hit", "receipt": old})
            continue
        started = utcnow(); identity, errors = timed_call(counters, 'source_audit', inspect_source, source); windows = []; vad = {"status": "missing", "reused": False, "segments": [], "errors": []}
        if identity["decode_status"] == "decoded_complete":
            vad = timed_call(counters, 'VAD_and_plan', reuse_vad, source, identity); errors.extend(vad.get("errors", []))
            if identity["frames"] > 0:
                try: windows = timed_call(counters, 'VAD_and_plan', plan_windows, identity, source, contract["runtime"], vad)
                except Exception: errors.append({"stage": "window_plan", "traceback": traceback.format_exc()})
        contexts.append({"source": source, "identity": identity, "vad": vad, "windows": windows,
                         "errors": errors, "started_utc": started})

    if native_profile is not None:
        total_windows=sum(len(c['windows'])for c in contexts)
        need(total_windows<=256, 'native pilot source group exceeds 256 fixed windows; preserve and reject, no regrouping')
        if total_windows:
            # 完整来源组（包括本次已命中的侧车）决定稳定地址，不能用剩余行另起计划。
            group=safe_join(store.root,'.native_torch_cohorts/'+claim.shard['id']+'/'+digest(sources))
            backend.prepare_native_group(group,args.contract_sha,descriptor['provenance']['code_sha256'])

    def flush(rows):
        if not rows: return
        asr_misses = []
        for row in rows:
            context, window = row["context"], row["window"]
            inputs = _stage_inputs("asr", window, context, contract, args.contract_sha, descriptor["provenance"]["models"])
            row["asr_inputs"] = inputs
            # 新策略只认完整批次文件事务，不允许逐窗缓存把未提交cohort拆散。
            cached = None if native_profile is not None else timed_call(counters, 'stage_cache', cache.get, "asr", inputs)
            if cached is None: asr_misses.append(row)
            else: window["asr"] = cached; window["asr_cache_hit"] = True; counters["stage_cache_hits"]["asr"] += 1
        answers = backend.transcribe(asr_misses)
        need(set(answers) == {r["row_id"] for r in asr_misses}, "ASR result mapping differs")
        for row in asr_misses:
            result = answers[row["row_id"]]
            hit=False
            if native_profile is not None:
                hit=result.get('durable_native_batch',{}).get('cache_hit')
                need(type(hit)is bool, 'native batch result needs actual per-row durable cache accounting')
                counters['stage_cache_hits']['asr']+=int(hit)
            elif result["status"] != "failed": result = timed_call(counters, 'stage_cache', cache.put, "asr", row["asr_inputs"], result)
            row["window"]["asr"] = result; row["window"]["asr_cache_hit"] = hit
        fa_misses = []
        for row in rows:
            context, window = row["context"], row["window"]; row["asr"] = window["asr"]
            inputs = _stage_inputs("aligner", window, context, contract, args.contract_sha, descriptor["provenance"]["models"])
            row["aligner_inputs"] = inputs; cached = timed_call(counters, 'stage_cache', cache.get, "aligner", inputs)
            if cached is None: fa_misses.append(row)
            else: window["alignment"] = cached; window["aligner_cache_hit"] = True; counters["stage_cache_hits"]["aligner"] += 1
        answers = backend.align(fa_misses)
        need(set(answers) == {r["row_id"] for r in fa_misses}, "FA result mapping differs")
        for row in fa_misses:
            result = answers[row["row_id"]]
            if result["status"] == "completed_native_candidates": result = timed_call(counters, 'stage_cache', cache.put, "aligner", row["aligner_inputs"], result)
            row["window"]["alignment"] = result; row["window"]["aligner_cache_hit"] = False
        for row in rows:
            window = row["window"]; context = row["context"]
            for stage in ("asr", "alignment"):
                if window[stage].get("error"): context["errors"].append({"stage": stage, "window_id": window["window_id"], "traceback": window[stage]["error"]})
            counters["model_windows"] += 1

    rows = []; pending_limit = 257 if native_profile is not None else contract["runtime"]["max_batch"] * 4
    for context in contexts:
        for window in context["windows"]:
            try:
                wave = timed_call(counters, 'window_read', read_window, context["source"], context["identity"], window)
                if len(wave) < MIN_WINDOW_FRAMES:
                    refuse_short_window(window, len(wave), context, counters)
                    continue
                window["preprocessing_status"] = "completed"
                rows.append({"row_id": context["source"]["source_id"] + ":" + window["window_id"],
                             "audio": wave, "context": context, "window": window})
            except Exception:
                error = traceback.format_exc(); window.update(preprocessing_status="failed", input_pcm_sha256=None,
                    asr={"status": "failed", "raw_text": "", "normalized_text": "", "language": None, "native": {"eos_observed": False}, "error": error},
                    alignment={"status": "not_applicable", "reason": "preprocessing_failed", "target_text": "", "items": [],
                               "native_timestamps_ms": [], "fixed_timestamps_ms": [], "repair_changed": False})
                context["errors"].append({"stage": "window_decode", "window_id": window["window_id"], "traceback": error})
            if len(rows) >= pending_limit: flush(rows); rows.clear()
    flush(rows); rows.clear()

    for context in contexts:
        source, identity = context["source"], context["identity"]
        units, flags = units_from_windows(context["windows"], identity) if identity["decode_status"] == "decoded_complete" else ([], [])
        context["errors"].extend(flags)
        if identity["decode_status"] == "decoded_complete":
            after = stat_identity(source["audio_path"])
            if after != identity["stat_after"]:
                identity.update(decode_status="decode_failed", stat_after=after); units = []
                context["errors"].append({"stage": "source", "reason": "source_changed_before_publication"})
        execution = {"job_key": job_key(source, args.contract_sha), "attempt_id": claim.attempt_id,
                     "worker_id": str(args.worker_id), "device": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu_fixture"),
                     "backend": backend.name, "started_utc": context["started_utc"], "finished_utc": utcnow(), "api_calls": 0}
        record = build_record(source, identity, descriptor, context["vad"], context["windows"], units, execution, context["errors"])
        record = native_json(record)
        receipt = timed_call(counters, 'sidecar_publish', store.publish, source, record, claim)
        receipts.append(receipt); counters["sidecars_committed"] += 1; counters["decision_counts"][receipt["decision"]] += 1
        if record["errors"]:
            counters["error_source_count"] += 1
            if len(counters["errors"]) < 100: counters["errors"].append({"source_id": source["source_id"], "sidecar_relpath": receipt.get("sidecar_relpath"), "error_count": len(record["errors"])})
        if identity["decode_status"] == "decoded_complete": counters["decoded_audio_seconds"] += identity["duration_seconds"]
        timed_call(counters, 'journal', claim.journal, {"source_id": source["source_id"], "status": "sidecar_committed", "receipt": receipt})
    return receipts


def main(argv=None, *, bounded_lease=None):
    global _ACTIVE_BACKEND
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "contract-sha", "shard-index", "index-sha", "queue-root", "output-root", "worker-id", "worker-output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resource-attestation"); parser.add_argument("--ticket"); parser.add_argument("--ticket-sha")
    parser.add_argument("--backend", choices=("qwen", "fixture"), default="qwen")
    parser.add_argument('--bounded-runtime-spec'); parser.add_argument('--bounded-runtime-sha')
    args = parser.parse_args(argv); started = utcnow(); start_time = time.monotonic()
    contract = read_contract(args.contract, args.contract_sha)
    need(contract["purpose"] == "cpu_fixture" if args.backend == "fixture" else contract["purpose"] != "cpu_fixture", "backend purpose differs")
    output = Path(args.worker_output); output.mkdir(parents=True, exist_ok=True)
    need(not (output / "worker_summary.json").exists(), "worker evidence output is not fresh")
    from bounded_speed_runtime import runtime_options, guard_native_formal_route
    bounded_pin = runtime_options(args)
    guard_native_formal_route(contract, bounded_pin, lease=bounded_lease, require_lease=True)
    need((bounded_pin is not None) == (bounded_lease is not None), 'authenticated bounded lineage lease required for thin index')
    if bounded_pin is None:
        queue = WorkQueue(args.shard_index, args.index_sha, args.queue_root, args.contract_sha)
    else:
        need(bounded_lease.pin == bounded_pin, 'worker inherited runtime pin differs')
        queue = bounded_lease.queue(args, contract)
    store = SidecarStore(args.output_root, contract, args.contract_sha); cache = StageCache(store.root / ".stage_cache")
    ticket = authenticate_resource(args) if args.backend == "qwen" else None
    bundle_sha = ticket["bundle_sha256"] if ticket else digest({"cpu_fixture_code": "production-worker/v1"})
    for path, expected in contract.get("source_files", {}).items(): need(sha_file(path) == expected, "pinned inference source changed: " + path)
    decoder = verify_decoder(contract["decoder"]) if args.backend == "qwen" else {"synthetic_fixture": True}
    descriptor = _descriptor(contract, args.contract_sha, bundle_sha, decoder)
    from qwen_backend import QwenBackend, FixtureBackend
    if ticket:
        from gpu_common import verify_model_stats as verify_admitted_model_stats
        verify_admitted_model_stats(ticket["model_facts"])
    _ACTIVE_BACKEND = QwenBackend(contract) if args.backend == "qwen" else FixtureBackend(contract)
    if ticket: verify_admitted_model_stats(ticket["model_facts"])
    if args.backend == "qwen":
        descriptor["provenance"]["runtime_qwen_modules"] = _ACTIVE_BACKEND.module_provenance
    counters = {"assigned_source_rows": 0, "sidecars_committed": 0, "sidecar_cache_hits": 0,
                "decision_counts": {"candidate": 0, "unknown": 0, "failed": 0}, "stage_cache_hits": {"asr": 0, "aligner": 0},
                "retained_previous_producer_hits": 0,
                "retained_previous_decision_counts": {"candidate":0,"unknown":0,"failed":0},
                "model_windows": 0, "decoded_audio_seconds": 0.0, "error_source_count": 0, "errors": [],
                "pipeline_seconds": {key: 0.0 for key in PIPELINE_TIMERS},
                "short_window_refused": 0}
    refs = []; assigned = 0; group_size = int(contract["runtime"].get("source_batch_rows", 8))
    previous_performance = None
    need(1 <= group_size <= 32, "source decode batch must remain bounded 1..32")
    while True:
        claim = queue.claim_next(args.worker_id)
        if claim is None:
            if queue.is_complete(): break
            time.sleep(.25); continue
        with claim:
            assigned += 1; sources = claim.sources(); receipts = []
            for offset in range(0, len(sources), group_size):
                receipts.extend(process_group(sources[offset:offset + group_size], claim, store, cache, _ACTIVE_BACKEND, contract, args, descriptor, counters))
            timed_call(counters, 'chunk_finish', claim.finish, receipts)
            refs.append({"id": claim.shard["id"], "path": str(queue.chunk_receipt_path(claim.shard["id"])),
                         "sha256": sha_file(queue.chunk_receipt_path(claim.shard["id"])), "source_count": len(receipts)})
            performance, previous_performance = progress_performance(_ACTIVE_BACKEND, counters, started, start_time, previous_performance)
            print(json.dumps({"event": "chunk_committed", "worker_id": args.worker_id, "shard_id": claim.shard["id"],
                              "sources": len(receipts), "decision_counts": counters["decision_counts"],
                              "performance": performance}, ensure_ascii=False), flush=True)
    metrics = _ACTIVE_BACKEND.metrics()
    summary = {"schema": "redimnet2-open-annotation-worker-summary/v1", "worker_id": str(args.worker_id),
               "contract_sha256": args.contract_sha, "index_sha256": args.index_sha, "backend": _ACTIVE_BACKEND.name,
               "device_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu_fixture"), "started_utc": started, "finished_utc": utcnow(),
               "elapsed_seconds": time.monotonic() - start_time, "chunks_assigned_count": assigned,
               "chunks_committed_count": len(refs), **counters, "model_API_calls": 0,
               "chunk_receipt_refs": refs, "backend_metrics": metrics, "synthetic_fixture": args.backend == "fixture"}
    atomic_new(output / "worker_summary.json", summary)
    return summary


if __name__ == "__main__":
    try: main()
    except BaseException:
        traceback.print_exc(); raise
