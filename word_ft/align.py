"""Offline CPU ASR/alignment candidates; automatic timestamps are not gold.

No model or torch import occurs until NativeQwenCPUBackend is explicitly built.
Each span reads one continuous source interval, retaining internal pauses.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import unicodedata


class AlignmentRefused(ValueError):
    def __init__(self, reason, evidence=None):
        self.reason = reason
        self.evidence = evidence
        super().__init__(reason)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hanzi_text(text):
    """Keep punctuation-separated Han characters; never silently erase digits/Latin."""
    if not isinstance(text, str) or not text.strip():
        raise AlignmentRefused("missing_transcript")
    normalized = unicodedata.normalize("NFKC", text)
    chars = []
    for offset, ch in enumerate(normalized):
        point = ord(ch)
        if (ch == "〇" or 0x3400 <= point <= 0x4DBF or 0x4E00 <= point <= 0x9FFF or
                0xF900 <= point <= 0xFAFF or 0x20000 <= point <= 0x2FA1F):
            chars.append((ch, offset))
        elif ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        else:
            raise AlignmentRefused("unreviewed_non_han_spoken_form")
    if not chars:
        raise AlignmentRefused("empty_han_transcript")
    return normalized, chars


def capture_native_alignment(aligner, audio, text, language="Chinese"):
    """Record the actual native pre/post fix_timestamp vectors, in milliseconds.

    The temporary hook is confined to this caller-owned CPU aligner. It is
    restored even if model inference or native timestamp repair raises.
    A backend lacking this observed hook is refused, not called "raw".
    """
    processor = aligner.aligner_processor
    original = processor.fix_timestamp
    local = vars(processor)
    had_local, previous = "fix_timestamp" in local, local.get("fix_timestamp")
    captures = []

    def traced(data):
        raw = data.tolist() if hasattr(data, "tolist") else list(data)
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in raw):
            raise AlignmentRefused("invalid_raw_timestamp_vector")
        event = {"raw_timestamps_ms": list(raw), "fixed_timestamps_ms": None,
                 "repair_changed": None, "changes": [], "confidence": None}
        captures.append(event)
        fixed = list(original(data))
        if len(fixed) != len(raw) or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in fixed):
            raise AlignmentRefused("invalid_fixed_timestamp_vector", captures)
        event["fixed_timestamps_ms"] = fixed
        event["changes"] = [{"index": i, "raw_ms": a, "fixed_ms": b, "delta_ms": b-a}
                            for i, (a, b) in enumerate(zip(raw, fixed)) if a != b]
        event["repair_changed"] = bool(event["changes"])
        return fixed

    processor.fix_timestamp = traced
    try:
        results = aligner.align(audio=audio, text=text, language=language)
    except Exception as exc:
        raise AlignmentRefused("native_alignment_failed: " + str(exc), captures) from exc
    finally:
        if had_local:
            processor.fix_timestamp = previous
        else:
            delattr(processor, "fix_timestamp")
    if len(results) != 1 or len(captures) != 1:
        raise AlignmentRefused("expected_one_observed_timestamp_vector", captures)
    items = [{"text": str(x.text), "start_time": float(x.start_time),
              "end_time": float(x.end_time), "confidence": None} for x in results[0]]
    step = float(aligner.timestamp_segment_time)
    if not math.isfinite(step) or step <= 0:
        raise AlignmentRefused("invalid_timestamp_resolution", captures)
    return {"items": items, "timestamp_audit": captures[0], "time_resolution_ms": step,
            "timestamp_unit": "seconds for items; milliseconds for pre/post vectors",
            "boundary_accuracy_verified": False, "confidence": None}


def validate_character_alignment(text, items, frames, sample_rate):
    normalized, chars = hanzi_text(text)
    if sample_rate != 16000 or not isinstance(frames, int) or frames <= 0:
        raise AlignmentRefused("requires_mono_16k_positive_source_frames")
    if len(items) != len(chars) or [x.get("text") for x in items] != [x[0] for x in chars]:
        raise AlignmentRefused("transcript_alignment_character_coverage_mismatch")
    checked, previous_end = [], 0
    for index, (item, (char, offset)) in enumerate(zip(items, chars)):
        start, end = item.get("start_time"), item.get("end_time")
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or
                not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= frames/sample_rate):
            raise AlignmentRefused("invalid_character_time_geometry")
        # Convert timestamp endpoints only; never extend a zero-length item.
        a, b = round(start*sample_rate), round(end*sample_rate)
        if not 0 <= a < b <= frames or a < previous_end:
            raise AlignmentRefused("overlapping_or_zero_sample_character_geometry")
        checked.append({"text": char, "text_offset": offset, "char_index": index,
                        "start_sample": a, "end_sample": b, "confidence": None})
        previous_end = b
    return normalized, checked


def _coverage(start, end, segments):
    covered, previous = 0, 0
    for a, b in segments:
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b or a < previous:
            raise AlignmentRefused("invalid_vad_geometry")
        covered += max(0, min(end, b)-max(start, a))
        previous = b
    return covered/(end-start)


def build_candidate_spans(record, text, aligned, *, max_units=128, context_ms=0):
    """1/2/3/4/5/6-character contiguous source spans, never VAD concatenated.

    These are candidate dictionaries for the public schema adapter, not a
    claim of human-verified lexical completeness. Even unrepaired timestamps
    and high VAD coverage cannot set boundary_review to accepted.
    """
    if not isinstance(max_units, int) or max_units < 1 or not 0 <= context_ms <= 40:
        raise AlignmentRefused("invalid_candidate_budget_or_context")
    normalized, chars = validate_character_alignment(text, aligned["items"], record["source_frames"], record["sample_rate"])
    segment_list = record.get("segments", [])
    if any(b > record["source_frames"] for _, b in segment_list):
        raise AlignmentRefused("vad_beyond_source")
    candidates, requested = [], round(context_ms*record["sample_rate"]/1000)
    for first in range(len(chars)):
        for count in range(1, 7):
            last = first+count-1
            if last >= len(chars):
                break
            a, b = chars[first]["start_sample"], chars[last]["end_sample"]
            previous_end = chars[first-1]["end_sample"] if first else 0
            following_start = chars[last+1]["start_sample"] if last+1 < len(chars) else record["source_frames"]
            left, right = min(requested, (a-previous_end)//2), min(requested, (following_start-b)//2)
            start, end = a-left, b+right
            piece = "".join(x["text"] for x in chars[first:last+1])
            coverage = _coverage(a, b, segment_list)
            raw = {"source_key": record["key"], "speaker_id": record["speaker"],
                   "source_audio": record["audio"], "source_start_sample": start,
                   "source_end_sample": end, "core_start_sample": a, "core_end_sample": b,
                   "sample_rate": record["sample_rate"], "num_samples": end-start,
                   "text": piece, "normalized_text": piece, "unit_count": count,
                   "count_bin": "1" if count == 1 else "2-3" if count <= 3 else "4-6",
                   "char_start_index": first, "char_end_index_exclusive": last+1,
                   "transcript_offsets": [chars[first]["text_offset"], chars[last]["text_offset"]+1],
                   "evidence": "forced_alignment_candidate", "boundary_review": "pending",
                   "transcript_origin": "asr_prediction", "confidence": None,
                   "time_resolution_ms": aligned["time_resolution_ms"],
                   "timestamp_repaired": aligned["timestamp_audit"]["repair_changed"],
                   "vad_coverage": coverage, "vad_used_to_trim": False,
                   "context_left_samples": left, "context_right_samples": right,
                   "context_basis": "estimated_alignment_gap_unverified" if left or right else "none",
                   "internal_pauses_retained": True, "source_intervals": [[start, end]],
                   "gold_eligible": False, "session_id": record.get("session_id"),
                   "review_reasons": ["asr_transcript_requires_human_check", "character_onset_and_tail_require_listening",
                                      "no_calibrated_alignment_confidence"]}
            if raw["timestamp_repaired"]:
                raw["review_reasons"].append("native_timestamp_repair")
            if coverage < .5:
                raw["review_reasons"].append("low_predicted_vad_coverage_do_not_trim_target")
            raw["candidate_id"] = identity_digest({"source": record["key"], "span": [start,end],
                                                    "text": piece, "text_sha": identity_digest(normalized)})
            candidates.append(raw)
            if len(candidates) >= max_units:
                return candidates
    return candidates


def model_directory_identity(path):
    root = Path(path).resolve(strict=True)
    if not root.is_dir() or not (root/"config.json").is_file():
        raise AlignmentRefused("local_model_directory_required")
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            resolved = p.resolve(strict=True)
            if root not in resolved.parents:
                raise AlignmentRefused("model_file_outside_local_directory")
            files[str(p.relative_to(root))] = {"bytes": p.stat().st_size, "sha256": file_sha(p)}
    if not any(name.endswith(".safetensors") for name in files):
        raise AlignmentRefused("local_safetensors_required")
    return {"path": str(root), "files": files, "files_sha256": identity_digest(files)}


def matched_prefix_spans(record, text, aligned, *, max_units=24, context_ms=0):
    """Only complete 1/2/3/6-character families; never truncate a family."""
    if not isinstance(max_units, int) or max_units < 4:
        raise AlignmentRefused("at_least_four_units_required_for_complete_family")
    _, chars = validate_character_alignment(text, aligned["items"], record["source_frames"], record["sample_rate"])
    if len(chars) < 6:
        raise AlignmentRefused("no_complete_six_character_prefix_family")
    all_spans = build_candidate_spans(record, text, aligned, max_units=len(chars)*6, context_ms=context_ms)
    by_key = {(s["char_start_index"],s["unit_count"]):s for s in all_spans}
    output = []
    for anchor in range(min(len(chars)-5, max_units//4)):
        family = [by_key[(anchor, n)] for n in (1,2,3,6)]
        if (len({s["source_start_sample"] for s in family}) != 1 or
                any(a["source_end_sample"] >= b["source_end_sample"] for a,b in zip(family,family[1:])) or
                any(not family[-1]["text"].startswith(s["text"]) for s in family)):
            raise AlignmentRefused("invalid_matched_prefix_family")
        output.extend(family)
    return output


class NativeQwenCPUBackend:
    """Explicit, local-only float32 CPU backend; constructor loads real models."""
    def __init__(self, asr_path, aligner_path, *, cpu_threads=4):
        if (os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1") or
                os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
            raise AlignmentRefused("explicit_hidden_cuda_and_offline_environment_required")
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise AlignmentRefused("positive_cpu_thread_limit_required")
        self.identity = {"asr": model_directory_identity(asr_path),
                         "aligner": model_directory_identity(aligner_path)}
        import torch
        if torch.cuda.is_initialized():
            raise AlignmentRefused("cuda_already_initialized")
        torch.set_num_threads(cpu_threads)
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        import inspect
        from importlib.metadata import version
        common = dict(dtype=torch.float32, device_map="cpu", local_files_only=True,
                      trust_remote_code=False, attn_implementation="eager")
        self.asr = Qwen3ASRModel.from_pretrained(self.identity["asr"]["path"],
                                               max_inference_batch_size=1, max_new_tokens=512, **common)
        self.aligner = Qwen3ForcedAligner.from_pretrained(self.identity["aligner"]["path"], **common)
        for wrapper in (self.asr, self.aligner):
            if any(p.device.type != "cpu" for p in wrapper.model.parameters()):
                raise AlignmentRefused("model_has_non_cpu_parameters")
        if torch.cuda.is_initialized():
            raise AlignmentRefused("cuda_initialized_during_cpu_model_load")
        self.identity.update({"device": "cpu", "dtype": "float32", "cpu_threads": cpu_threads,
                              "qwen_asr_version": version("qwen-asr"), "torch_version": torch.__version__,
                              "time_resolution_ms": float(self.aligner.timestamp_segment_time),
                              "sources": {str(Path(inspect.getfile(cls)).resolve()): file_sha(inspect.getfile(cls))
                                          for cls in (Qwen3ASRModel, Qwen3ForcedAligner)}})

    def transcribe(self, waveform, sample_rate):
        rows = self.asr.transcribe(audio=(waveform, sample_rate), language="Chinese", return_time_stamps=False)
        if len(rows) != 1:
            raise AlignmentRefused("expected_one_asr_transcript")
        return {"text": rows[0].text, "language": rows[0].language,
                "transcript_origin": "asr_prediction", "confidence": None}

    def align(self, waveform, sample_rate, text):
        return capture_native_alignment(self.aligner, (waveform, sample_rate), text)


def align_training_record(record, backend, *, max_units=24, context_ms=0, max_audio_seconds=30):
    """Run one bounded whole-file candidate audit; no audio files are written.

    Input may be a reviewed duration_ft record or an explicit audio_path and
    speaker_id record. A source longer than the limit is refused, never cut
    at the limit. Every unit retains one common source_utterance_id so two
    overlapping or disjoint words from this turn cannot become target pairs.
    """
    import numpy as np
    import soundfile as sf
    from .schema import make_unit, normalize_text

    path = Path(record.get("audio_path", record.get("audio", "")))
    if not path.is_absolute() or not path.is_file():
        raise AlignmentRefused("absolute_local_source_audio_required")
    speaker = record.get("speaker_id", record.get("speaker"))
    if not isinstance(speaker, str) or not speaker:
        raise AlignmentRefused("speaker_identity_required")
    if not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0:
        raise AlignmentRefused("positive_audio_budget_required")
    before = path.stat()
    source = {"path": str(path), "realpath": str(path.resolve()), "bytes": before.st_size,
              "mtime_ns": before.st_mtime_ns, "sha256": file_sha(path)}
    if (record.get("source_stat", {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns}) !=
            {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns} or
            record.get("audio_realpath", source["realpath"]) != source["realpath"]):
        raise AlignmentRefused("source_attributes_differ_from_vad_input")
    info = sf.info(str(path))
    if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
        raise AlignmentRefused("source_must_be_mono_16k_positive_frames")
    if info.frames/info.samplerate > max_audio_seconds:
        raise AlignmentRefused("whole_source_exceeds_budget_no_time_truncation")
    if (record.get("source_frames", info.frames) != info.frames or
            record.get("sample_rate", info.samplerate) != info.samplerate):
        raise AlignmentRefused("source_header_differs_from_input")
    vad = dict(record.get("vad") or {})
    for name in ("rttm", "sidecar"):
        if name in record:
            observed = file_sha(record[name])
            if observed != record.get(name+"_sha256"):
                raise AlignmentRefused("source_vad_annotation_changed")
            vad[name] = record[name]
            vad[name+"_sha256"] = observed
    segments = record.get("segments", vad.get("segments_source_samples", []))
    if vad.get("source_sha256", source["sha256"]) != source["sha256"]:
        raise AlignmentRefused("vad_source_content_sha_differs")
    vad.update(segments_source_samples=segments, source_sha256=source["sha256"], used_to_trim=False, concatenated=False)
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.shape != (info.frames,) or rate != 16000 or not np.isfinite(waveform).all():
        raise AlignmentRefused("incomplete_or_nonfinite_source_decode")
    prediction = backend.transcribe(waveform, rate)
    text = prediction["text"]
    prepared = {"key": record.get("key", record.get("source_utterance_id", source["sha256"])),
                "speaker": speaker, "audio": str(path), "source_frames": info.frames,
                "sample_rate": rate, "segments": segments,
                "session_id": record.get("source_session_id", record.get("session_id"))}
    try:
        normalized = normalize_text(text, "zh")
        aligned = backend.align(waveform, rate, text)
        spans = matched_prefix_spans(prepared, text, aligned, max_units=max_units, context_ms=context_ms)
    except Exception as exc:
        raise AlignmentRefused(str(exc), {"source": source, "asr_prediction": prediction,
                               "alignment": locals().get("aligned"),
                               "native_failure_evidence": getattr(exc, "evidence", None)}) from exc
    audit = {"source": source, "source_header": {"frames": info.frames, "sample_rate": rate, "channels": 1},
             "asr_prediction": {**prediction, "normalized_text": normalized, "confidence": None},
             "alignment": aligned, "backend_identity": backend.identity,
             "decoder": {"soundfile_version": sf.__version__, "libsndfile_version": sf.__libsndfile_version__},
             "quality_checks": {"character_coverage_exact": True, "time_geometry_legal": True,
                                "transcript_correctness_verified": False, "boundary_completeness_verified": False,
                                "confidence_available": False, "vad_does_not_prove_completeness": True},
             "candidate_count": len(spans), "max_units": max_units, "max_audio_seconds": max_audio_seconds,
             "boundary_review": "pending", "evidence": "forced_alignment_candidate",
             "source_unchanged_after_alignment": True}
    audit_id = identity_digest(audit)
    units = []
    for span in spans:
        unit = make_unit(path, speaker, span["text"], start_sample=span["source_start_sample"],
                         end_sample=span["source_end_sample"], evidence="forced_alignment_candidate",
                         transcript_origin="asr_prediction", boundary_review="pending",
                         source_utterance_id=str(prepared["key"]), source_session_id=prepared["session_id"],
                         corpus=record.get("corpus", "CNC2"), split=record.get("split", "train"),
                         provenance={"record_audit_sha256": audit_id, "source_key": prepared["key"],
                                     "source_transcript_sha256": identity_digest(text)},
                         alignment={"time_resolution_ms": aligned["time_resolution_ms"], "confidence": None,
                                    "timestamp_audit": aligned["timestamp_audit"],
                                    "char_start_index": span["char_start_index"],
                                    "char_end_index_exclusive": span["char_end_index_exclusive"],
                                    "context_left_samples": span["context_left_samples"],
                                    "context_right_samples": span["context_right_samples"],
                                    "context_basis": span["context_basis"], "internal_pauses_retained": True,
                                    "core_source_interval": [span["core_start_sample"], span["core_end_sample"]]},
                         vad={**vad, "core_predicted_speech_coverage": span["vad_coverage"]})
        if unit["audio_sha256"] != source["sha256"]:
            raise AlignmentRefused("source_content_changed_during_unit_build")
        unit["family_id"] = identity_digest({k:unit[k] for k in
            ("audio_sha256","speaker_id","source_utterance_id","start_sample","language","counting_policy")})
        unit["review_reasons"] = span["review_reasons"]
        units.append(unit)
    after = path.stat()
    if ((after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or
            str(path.resolve()) != source["realpath"] or file_sha(path) != source["sha256"]):
        raise AlignmentRefused("source_changed_during_asr_alignment")
    audit["record_audit_sha256"] = audit_id
    return {"units": units, "audit": audit,
            "review_queue": [{"unit_id": u["unit_id"], "source_utterance_id": u["source_utterance_id"],
                              "text": u["spoken_text"], "audio_path": u["audio_path"],
                              "start_sample": u["start_sample"], "end_sample": u["end_sample"],
                              "family_id": u["family_id"], "boundary_review": "pending", "reasons": u["review_reasons"]} for u in units]}
