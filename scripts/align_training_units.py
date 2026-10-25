#!/usr/bin/env python3
"""Create bounded offline CPU forced-alignment candidates, never reviewed gold."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="single JSON records object, not JSONL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asr-model", required=True, type=Path)
    parser.add_argument("--aligner-model", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=1)
    parser.add_argument("--max-units-per-record", type=int, default=24)
    parser.add_argument("--max-audio-seconds", type=float, default=30)
    parser.add_argument("--context-ms", type=float, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="actually load CPU ASR and aligner; otherwise print plan only")
    args = parser.parse_args(argv)
    if args.max_records < 1 or args.max_units_per_record < 4 or args.cpu_threads < 1 or not 0 <= args.context_ms <= 40 or not 0 < args.max_audio_seconds <= 120:
        parser.error("positive counts/CPU threads, >=4 units for a complete family, context0..40ms, audio budget<=120s required")
    from word_ft.align import NativeQwenCPUBackend, align_training_record, identity_digest, file_sha
    from word_ft.schema import protect_output, write_dataset
    source = json.loads(args.input.read_text())
    rows = source.get("records")
    if not isinstance(rows, list) or not rows:
        parser.error("input must contain nonempty records list")
    if "dataset_id" in source and source["dataset_id"] != identity_digest({k:v for k,v in source.items() if k != "dataset_id"}):
        parser.error("input dataset identity differs")
    eligible = [r for r in rows if r.get("corpus") in ("C", "CNC2", "CN-Celeb2") or
                str(r.get("key", r.get("source_utterance_id", ""))).startswith("C_")]
    selected = eligible[:args.max_records]
    if not selected:
        parser.error("no explicitly identified CNC2 input records")
    out = protect_output(args.output)
    if out.exists():
        raise FileExistsError("candidate output already exists")
    plan = {"mode": "cpu_execute" if args.execute else "plan_only_no_model_loaded", "input": str(args.input.resolve()),
            "input_sha256": file_sha(args.input), "input_dataset_id": source.get("dataset_id"),
            "input_records": len(rows), "cnc2_records": len(eligible), "selected_records": len(selected),
            "max_units_per_record": args.max_units_per_record, "max_audio_seconds": args.max_audio_seconds,
            "context_ms": args.context_ms, "cpu_threads": args.cpu_threads,
            "asr_model": str(args.asr_model.resolve()), "aligner_model": str(args.aligner_model.resolve()),
            "evidence": "forced_alignment_candidate", "boundary_review": "pending", "transcript_origin": "asr_prediction",
            "gold_eligible": False, "vad_concatenation": False, "time_truncation": False,
            "family_counts": [1,2,3,6], "family_policy": "matched_prefix_family",
            "scope": "offline local CPU only; no automatic retries; long source is refused whole"}
    print(json.dumps(plan, ensure_ascii=False, allow_nan=False), flush=True)
    if not args.execute:
        return 0
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS=str(args.cpu_threads),
                      MKL_NUM_THREADS=str(args.cpu_threads), TOKENIZERS_PARALLELISM="false")
    out.mkdir(parents=True, exist_ok=False)
    (out/"plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    attempts, units = [], []
    try:
        backend = NativeQwenCPUBackend(args.asr_model, args.aligner_model, cpu_threads=args.cpu_threads)
    except Exception:
        (out/"model_load_failure.txt").write_text(traceback.format_exc())
        return 2
    (out/"model_identity.json").write_text(json.dumps(backend.identity, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    for record in selected:
        try:
            result = align_training_record(record, backend, max_units=args.max_units_per_record,
                                           context_ms=args.context_ms, max_audio_seconds=args.max_audio_seconds)
            units.extend(result["units"])
            attempts.append({"status": "candidate_pending_review", "audit": result["audit"],
                             "review_queue": result["review_queue"]})
        except Exception as exc:
            attempts.append({"status": "refused", "source_key": record.get("key", record.get("source_utterance_id")),
                             "reason": str(exc), "evidence": getattr(exc, "evidence", None),
                             "traceback": traceback.format_exc()})
        (out/"alignment_attempts.json").write_text(json.dumps(attempts, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    if units:
        write_dataset(out/"candidate_units.json", units,
                      {"plan": plan, "backend_identity": backend.identity,
                       "record_audits": [a["audit"] for a in attempts if "audit" in a],
                       "automatic_candidate_not_gold": True})
    queue = [q for a in attempts for q in a.get("review_queue", [])]
    (out/"human_review_queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    refused = sum(a["status"] == "refused" for a in attempts)
    print(json.dumps({"units": len(units), "attempted_records": len(attempts), "refused_records": refused,
                      "boundary_review": "pending", "gold_eligible": False}, ensure_ascii=False), flush=True)
    return 2 if refused or not units else 0


if __name__ == "__main__":
    raise SystemExit(main())
