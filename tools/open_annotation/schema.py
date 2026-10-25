"""Production silver sidecar contract and cross-field checks (stdlib only).

Structural validity is independent of label usability. Unknown/unsupported/failed
sources are terminal records, while machine candidates are never human truth.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import unicodedata
from normalization import normalize_with_audit, POLICY, SUPPORTED_POLICIES
from quality_axes import quality_axes, native_observation_errors
from source_clock import TIME_MAPPING, SUPPORTED_MAPPINGS, sample_bounds

SCHEMA = "redimnet2-open-annotation/v2"
SOURCE_SCHEMA = "redimnet2-source/v1"
PCM_HASH_POLICY = "decoded_float32le_native_interleaved/v1"
SUCCESS = {"completed", "succeeded", "succeeded_candidate", "completed_native_candidates"}
SHA = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ID = re.compile(r"^src_[0-9a-f]{64}$")


def canonical_bytes(value):
    """Finite, deterministic JSON used for content identity and publication."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _integer(x):
    return type(x) is int


def _number(x):
    return type(x) in (int, float) and math.isfinite(x)


def _safe_relative(x):
    return (isinstance(x, str) and bool(x) and "\x00" not in x and
            "\\" not in x and not x.startswith("/") and
            all(p not in ("", ".", "..") for p in x.split("/")) and
            not PurePosixPath(x).is_absolute())


def _sha(x):
    return isinstance(x, str) and SHA.fullmatch(x) is not None


def _surface(text):
    if not isinstance(text, str):return None
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def _expected_normalization(record, raw):
    policy = record.get("provenance", {}).get("runtime", {}).get("normalization_profile", POLICY)
    if not isinstance(raw, str) or policy not in SUPPORTED_POLICIES:
        return None
    return normalize_with_audit(raw, policy)


def _finite_tree(value, path="record"):
    errors = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(path + ":nonfinite_json_number")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(path + ":nonstring_json_key")
            errors.extend(_finite_tree(item, path + "." + str(key)))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            errors.extend(_finite_tree(item, path + "[" + str(i) + "]"))
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        errors.append(path + ":non_json_value")
    return errors


def _raw_geometry_flags(window, identity):
    """Inspect the preserved native model output, not just filtered top units."""
    reasons = []
    al = window.get("alignment", {})
    items = al.get("items", [])
    if not isinstance(items, list):
        return ["alignment_items_not_list"]
    if al.get("repair_changed") is not False:
        reasons.append("alignment_repaired_or_repair_unknown")
    native = al.get("native_timestamps_ms", al.get("raw_timestamps_ms"))
    fixed = al.get("fixed_timestamps_ms")
    if not isinstance(native, list) or not isinstance(fixed, list):
        reasons.append("native_or_fixed_timestamps_missing")
    elif len(native) != 2 * len(items) or len(fixed) != len(native):
        reasons.append("timestamp_item_count_mismatch")
    elif any(not _number(x) for x in native + fixed):
        reasons.append("native_timestamp_nonfinite_or_invalid")
    elif native != fixed and al.get("repair_changed") is False:
        reasons.append("timestamp_repair_flag_contradiction")
    sr = identity.get("sample_rate")
    c0, c1 = window.get("context_start_sample"), window.get("context_end_sample")
    seconds = (c1 - c0) / sr if _integer(sr) and sr > 0 and _integer(c0) and _integer(c1) else None
    previous_end = None
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            reasons.append("native_item_not_object")
            continue
        a, b = item.get("start_time"), item.get("end_time")
        if not isinstance(item.get("text"), str) or not item.get("text", "").strip():
            reasons.append("native_empty_text")
        if not _number(a) or not _number(b):
            reasons.append("native_item_nonfinite_or_invalid")
        elif b <= a:
            reasons.append("native_zero_or_negative_duration")
        else:
            if a < 0 or (seconds is not None and b > seconds + 1e-8):
                reasons.append("native_outside_context")
            if previous_end is not None and a < previous_end - 1e-8:
                reasons.append("native_overlap_or_reversal")
            previous_end = b
            if isinstance(fixed, list) and len(fixed) == 2 * len(items) and all(_number(x) for x in fixed):
                if abs(a * 1000 - fixed[2*i]) > 1e-6 or abs(b * 1000 - fixed[2*i+1]) > 1e-6:
                    reasons.append("item_time_not_bound_to_fixed_native_vector")
    return sorted(set(reasons))


def quality_reasons(record):
    """Return conservative machine-quality reasons; never claim human accuracy."""
    reasons = []
    source, identity = record.get("source", {}), record.get("source_identity", {})
    windows, units = record.get("windows", []), record.get("units", [])
    if source.get("audio_path") is None or source.get("resolution_status") == "unresolved":
        reasons.append("source_unresolved")
    if identity.get("decode_status") != "decoded_complete":
        reasons.append("source_not_completely_decoded")
    if identity.get("frames") == 0:
        reasons.append("source_empty_pcm")
    if identity.get("stat_before") is None or identity.get("stat_before") != identity.get("stat_after"):
        reasons.append("source_stat_unstable_or_unknown")
    if record.get("errors"):
        reasons.append("processing_errors_preserved")
    if not isinstance(windows, list) or not windows:
        reasons.append("no_model_windows")
        windows = []
    if not isinstance(units, list) or not units:
        reasons.append("no_usable_lexical_units")
    sr, frames = identity.get("sample_rate"), identity.get("frames")
    owned = []
    for i, window in enumerate(windows):
        if not isinstance(window, dict):
            reasons.append("window_not_object")
            continue
        prefix = "window_" + str(i) + ":"
        runtime = record.get("provenance", {}).get("runtime", {})
        if isinstance(runtime, dict) and "boundary_mapping" in runtime and window.get("time_mapping") != runtime["boundary_mapping"]:
            reasons.append(prefix + "source_clock_not_bound_to_producer_runtime")
        if window.get("preprocessing_status", "completed") != "completed":
            reasons.append(prefix + "preprocessing_not_completed")
        asr, al = window.get("asr", {}), window.get("alignment", {})
        if not isinstance(asr, dict) or not isinstance(al, dict):
            reasons.append(prefix + "malformed_stage_observation")
            continue
        if asr.get("status") not in SUCCESS:
            reasons.append(prefix + "asr_not_succeeded")
        if asr.get("status") in SUCCESS:
            raw = asr.get("raw_text")
            expected_audit = _expected_normalization(record, raw)
            if expected_audit is None or asr.get("normalization_audit") != expected_audit:
                reasons.append(prefix + "normalization_audit_missing_or_not_reconstructible")
            if expected_audit is not None:
                if asr.get("normalized_text") != expected_audit["normalized_text"]:
                    reasons.append(prefix + "normalized_text_not_bound_to_raw_policy")
                if expected_audit["spoken_form_review_required"]:
                    reasons.append(prefix + "spoken_form_review_required")
        if not isinstance(asr.get("raw_text"), str) or not asr.get("raw_text", "").strip():
            reasons.append(prefix + "empty_asr_not_proven_noise")
        native_asr = asr.get("native") or {}
        if not isinstance(native_asr, dict):native_asr = {}
        if native_asr.get("eos_observed") is not True:
            reasons.append(prefix + "asr_eos_not_observed")
        if al.get("status") not in SUCCESS:
            reasons.append(prefix + "alignment_not_succeeded")
        if not al.get("items"):
            reasons.append(prefix + "no_native_alignment_items")
        if al.get("status") in SUCCESS:
            if al.get("target_text") != asr.get("normalized_text") or not isinstance(al.get("target_text"), str):
                reasons.append(prefix + "alignment_target_not_bound_to_asr")
            items = al.get("items", [])
            if isinstance(items, list) and all(isinstance(x, dict) and isinstance(x.get("text"), str) for x in items):
                if _surface(al.get("target_text")) != _surface("".join(x["text"] for x in items)):
                    reasons.append(prefix + "alignment_text_coverage_mismatch")
        reasons.extend(prefix + x for x in _raw_geometry_flags(window, identity))
        if record.get("provenance", {}).get("alignment_method") == "ultra_no_head":
            reasons.extend(prefix + x for x in native_observation_errors(al))
        a, b = window.get("owned_start_sample"), window.get("owned_end_sample")
        if _integer(a) and _integer(b):
            owned.append((a, b))
    if _integer(frames) and frames > 0:
        interval = source.get("source_interval")
        target = (0, frames) if interval is None else (interval.get("start_sample"), interval.get("end_sample"))
        ordered = sorted(owned)
        if not ordered or ordered[0][0] != target[0] or ordered[-1][1] != target[1] or any(a[1] != b[0] for a, b in zip(ordered, ordered[1:])):
            reasons.append("owned_windows_do_not_partition_complete_source")
    if not _integer(sr) or sr <= 0:
        reasons.append("source_sample_rate_unknown")
    previous = None
    for u in units if isinstance(units, list) else []:
        if not isinstance(u, dict):continue
        a, b = u.get("start_sample"), u.get("end_sample")
        if _integer(a) and _integer(b):
            if previous is not None and a < previous:
                reasons.append("source_units_overlap_or_out_of_order")
            previous = b
    return sorted(set(reasons))


def build_record(source, identity, contract, vad, windows, units, execution, errors):
    record = {"schema": SCHEMA, "source": deepcopy(source), "source_identity": deepcopy(identity),
              "contract": deepcopy(contract), "provenance": deepcopy(contract.get("provenance", {"models": contract.get("models", {})})),
              "vad": deepcopy(vad), "windows": deepcopy(windows), "units": deepcopy(units),
              "execution": deepcopy(execution), "errors": deepcopy(errors)}
    record["quality_axes"] = quality_axes(record)
    reasons = quality_reasons(record)
    hard_failed = identity.get("decode_status") == "decode_failed"
    record["decision"] = {"state": "failed" if hard_failed else ("unknown" if reasons else "candidate"),
                          "reasons": reasons, "annotation_completed": True,
                          "usable_for_training": False, "usable_for_evaluation": False,
                          "human_review": None, "open_model_generated": any(isinstance(w, dict) and isinstance(w.get("asr"), dict) and w["asr"].get("status") in SUCCESS for w in windows)}
    return record


def validate_record(record):
    errors = _finite_tree(record)
    if not isinstance(record, dict):
        return errors + ["record:not_object"]
    def need(ok, reason):
        if not ok:
            errors.append(reason)
    need(record.get("schema") == SCHEMA, "schema:wrong_version")
    required = ("source", "source_identity", "contract", "provenance", "vad", "execution", "decision")
    for name in required:
        need(isinstance(record.get(name), dict), name + ":not_object")
    if any(not isinstance(record.get(name), dict) for name in required):
        return sorted(set(errors))
    for name in ("windows", "units", "errors"):
        need(isinstance(record.get(name), list), name + ":not_list")
    if any(not isinstance(record.get(name), list) for name in ("windows", "units", "errors")):
        return sorted(set(errors))
    s, identity, contract, decision = (record[k] for k in ("source", "source_identity", "contract", "decision"))
    need(s.get("schema") == SOURCE_SCHEMA, "source:wrong_schema")
    need(isinstance(s.get("source_id"), str) and SOURCE_ID.fullmatch(s.get("source_id", "")) is not None, "source:invalid_id")
    need(isinstance(s.get("dataset"), str) and _safe_relative(s.get("dataset")) and "/" not in s.get("dataset", ""), "source:invalid_dataset")
    need(_safe_relative(s.get("relative_path")), "source:unsafe_relative_path")
    if _safe_relative(s.get("relative_path")) and isinstance(s.get("dataset"), str):
        expected = s["dataset"] + "/" + s["relative_path"] + ".annotation.json"
        need(s.get("sidecar_relpath") == expected, "source:sidecar_path_collision_or_mismatch")
        expected_id = "src_" + hashlib.sha256((s["dataset"] + "\0" + s["relative_path"]).encode()).hexdigest()
        need(s.get("source_id") == expected_id, "source:id_not_bound_to_dataset_relative_path")
    need(isinstance(s.get("roles"), list) and bool(s.get("roles")) and all(isinstance(x, str) and x for x in s.get("roles", [])), "source:roles_missing")
    need(isinstance(s.get("original_splits"), list), "source:original_splits_missing")
    need(isinstance(s.get("protection"), dict) and type(s.get("protection", {}).get("training_forbidden")) is bool, "source:protection_missing")
    need(s.get("channel_policy") == "mean_channels_to_mono", "source:channel_policy_not_frozen")
    need(identity.get("decode_status") in {"decoded_complete", "decode_failed", "missing"}, "identity:invalid_decode_status")
    if identity.get("decode_status") == "decoded_complete":
        for key in ("sample_rate", "channels", "frames", "bytes"):
            need(_integer(identity.get(key)) and identity[key] >= (0 if key in ("bytes", "frames") else 1), "identity:invalid_" + key)
        for key in ("file_sha256", "pcm_sha256"):
            need(_sha(identity.get(key)), "identity:missing_" + key)
        need(identity.get("pcm_hash_policy") == PCM_HASH_POLICY, "identity:pcm_hash_policy_missing")
        if _integer(identity.get("sample_rate")) and identity["sample_rate"] > 0 and _integer(identity.get("frames")):
            need(_number(identity.get("duration_seconds")) and abs(identity["duration_seconds"] - identity["frames"] / identity["sample_rate"]) < 1e-8, "identity:duration_frames_clock_mismatch")
        interval = s.get("source_interval")
        if interval is not None:
            need(isinstance(interval, dict), "source:interval_not_object")
            if isinstance(interval, dict):
                a, b = interval.get("start_sample"), interval.get("end_sample")
                need(_integer(a) and _integer(b) and 0 <= a < b <= identity.get("frames", -1) and interval.get("sample_rate", identity.get("sample_rate")) == identity.get("sample_rate"), "source:interval_clock_invalid")
    else:
        need(not record["units"], "identity:undecoded_source_has_units")
    need(_sha(contract.get("sha256")) and isinstance(contract.get("version"), str), "contract:identity_missing")
    need(contract.get("api_calls") == 0 and type(contract.get("api_calls")) is int, "contract:production_api_not_zero")
    ex = record["execution"]
    need(ex.get("api_calls") == 0 and type(ex.get("api_calls")) is int, "execution:production_api_not_zero")
    for key in ("job_key", "attempt_id", "worker_id", "started_utc", "finished_utc"):
        need(ex.get(key) is not None, "execution:missing_" + key)
    need(decision.get("state") in {"candidate", "unknown", "failed"}, "decision:invalid_state")
    need(decision.get("annotation_completed") is True and type(decision.get("open_model_generated")) is bool, "decision:not_terminal_machine_annotation")
    need(decision.get("human_review") is None, "decision:machine_output_claims_human_review")
    need(decision.get("usable_for_training") is False and decision.get("usable_for_evaluation") is False, "decision:unreviewed_consumer_release")
    need(isinstance(decision.get("reasons"), list) and all(isinstance(x, str) for x in decision.get("reasons", [])), "decision:invalid_reasons")
    provenance = record["provenance"]
    need(_sha(provenance.get("code_sha256", provenance.get("bundle_manifest_sha256"))), "provenance:code_identity_missing")
    models = provenance.get("models")
    need(isinstance(models, dict) and set(models) == {"asr", "aligner"}, "provenance:exact_models_missing")
    if isinstance(models, dict):
        for name in ("asr", "aligner"):
            model = models.get(name)
            if not isinstance(model, dict):continue
            need(isinstance(model.get("path"), str) and model["path"].startswith("/"), "provenance:" + name + "_not_local")
            pins = model.get("files")
            need(isinstance(pins, dict) and bool(pins) and _sha(model.get("files_sha256")), "provenance:" + name + "_weights_identity_missing")
            if isinstance(pins, dict):
                for path, pin in pins.items():
                    need(_safe_relative(path) and isinstance(pin, dict) and _sha(pin.get("sha256")) and _integer(pin.get("bytes")) and pin["bytes"] >= 0, "provenance:" + name + "_invalid_file_pin")
                need(model.get("files_sha256") == sha256_json(pins), "provenance:" + name + "_file_pin_digest_mismatch")
    vad = record["vad"]
    need(isinstance(vad.get("status"), str) and type(vad.get("reused")) is bool and isinstance(vad.get("segments"), list), "vad:status_or_segments_missing")
    if vad.get("reused") is True:
        for name in ("original_path", "metadata_path"):
            need(isinstance(vad.get(name), str) and vad[name].startswith("/"), "vad:missing_" + name)
        for name in ("original_sha256", "metadata_sha256"):
            need(_sha(vad.get(name)), "vad:missing_" + name)
        need(vad.get("identity_binding") in {"legacy_source_path_size_mtime_clock", "source_file_sha256"}, "vad:unknown_source_binding")
        if s.get("old_vad", {}).get("identity_binding") == "legacy_source_path_size_mtime_clock":
            need(vad.get("identity_binding") == "legacy_source_path_size_mtime_clock" and vad.get("old_source_sha256") is None and vad.get("old_source_byte_sha_binding_claimed", False) is False, "vad:legacy_binding_upgraded_to_historical_source_sha")
        if vad.get("identity_binding") == "source_file_sha256":
            need(_sha(vad.get("old_source_sha256")) and vad.get("old_source_sha256") == identity.get("file_sha256"), "vad:strong_source_binding_mismatch")
        for seg in vad.get("segments", []):
            need(isinstance(seg, dict), "vad:segment_not_object")
            if isinstance(seg, dict):
                a, b = seg.get("start_time"), seg.get("end_time")
                need(_number(a) and _number(b) and 0 <= a < b and _number(identity.get("duration_seconds")) and b <= identity["duration_seconds"] + 1e-8, "vad:segment_outside_source_clock")
                sr0 = identity.get("sample_rate")
                if _number(a) and _number(b) and _integer(sr0) and sr0 > 0:
                    need(seg.get("start_sample") == math.floor(a * sr0) and seg.get("end_sample") == math.ceil(b * sr0), "vad:segment_sample_mapping_mismatch")
    windows = {}
    frames, sr = identity.get("frames"), identity.get("sample_rate")
    for i, w in enumerate(record["windows"]):
        if not isinstance(w, dict):
            errors.append("window:not_object"); continue
        wid = w.get("window_id")
        need(isinstance(wid, str) and wid and wid not in windows, "window:duplicate_or_missing_id")
        if isinstance(wid, str):windows[wid] = w
        bounds = [w.get(x) for x in ("context_start_sample", "owned_start_sample", "owned_end_sample", "context_end_sample")]
        need(all(_integer(x) for x in bounds), "window:noninteger_source_clock")
        if all(_integer(x) for x in bounds):
            a, b, c, d = bounds
            need(0 <= a <= b < c <= d and (not _integer(frames) or d <= frames), "window:source_clock_outside_or_invalid")
        need(w.get("model_sample_rate") == 16000 and w.get("time_mapping") in SUPPORTED_MAPPINGS, "window:model_or_time_mapping_not_frozen")
        runtime = record.get("provenance", {}).get("runtime", {})
        if isinstance(runtime, dict) and "boundary_mapping" in runtime:
            need(w.get("time_mapping") == runtime["boundary_mapping"], "window:source_clock_not_bound_to_producer_runtime")
        preprocessing_failed = (w.get("preprocessing_status") == "failed" and w.get("input_pcm_sha256") is None
                                and isinstance(w.get("asr"), dict) and w["asr"].get("status") == "failed"
                                and isinstance(w.get("alignment"), dict) and w["alignment"].get("status") == "not_applicable")
        need(_sha(w.get("input_pcm_sha256")) or preprocessing_failed, "window:model_pcm_identity_missing")
        need(isinstance(w.get("asr"), dict) and isinstance(w.get("alignment"), dict), "window:stage_observation_missing")
        asr = w.get("asr", {})
        if isinstance(asr, dict) and asr.get("status") in SUCCESS:
            raw = asr.get("raw_text")
            expected_audit = _expected_normalization(record, raw)
            need(expected_audit is not None and asr.get("normalization_audit") == expected_audit,
                 "window:normalization_audit_not_reconstructible")
            if expected_audit is not None:
                need(asr.get("normalized_text") == expected_audit["normalized_text"], "window:normalized_text_raw_policy_mismatch")
    occurrences = {}
    seen = set()
    for u in record["units"]:
        if not isinstance(u, dict):errors.append("unit:not_object");continue
        w = windows.get(u.get("window_id"))
        need(w is not None, "unit:unknown_window")
        need(isinstance(u.get("text"), str) and bool(u.get("text", "").strip()), "unit:empty_text")
        need(_integer(u.get("native_index")) and u["native_index"] >= 0, "unit:bad_native_index")
        need(u.get("boundary_status") == "machine_candidate", "unit:machine_claims_reference_boundary")
        need(u.get("geometry_status") == "valid", "unit:invalid_top_level_geometry_status")
        need(u.get("sample_rate") == sr and _integer(sr) and sr > 0, "unit:wrong_source_sample_rate")
        a, b = u.get("start_sample"), u.get("end_sample")
        need(_integer(a) and _integer(b), "unit:noninteger_source_coordinate")
        if _integer(a) and _integer(b) and _integer(frames):need(0 <= a < b <= frames, "unit:invalid_source_geometry")
        text = u.get("text")
        if isinstance(text, str):
            need(u.get("occurrence_index") == occurrences.get(text, 0), "unit:repeated_text_occurrence_mismatch")
            occurrences[text] = occurrences.get(text, 0) + 1
        index = u.get("native_index")
        if w is not None and _integer(index) and isinstance(w.get("alignment"), dict):
            need(type(u.get("repair_applied")) is bool and u.get("repair_applied") == w["alignment"].get("repair_changed"), "unit:repair_flag_mismatch")
            items = w["alignment"].get("items", [])
            need(index < len(items), "unit:native_index_outside_items")
            if 0 <= index < len(items) and isinstance(items[index], dict):
                item = items[index]
                need(u.get("text") == item.get("text"), "unit:text_not_bound_to_native_occurrence")
                if _number(item.get("start_time")) and _number(item.get("end_time")) and _integer(sr) and sr > 0:
                    ca = w.get("context_start_sample")
                    if _integer(ca) and w.get("time_mapping") in SUPPORTED_MAPPINGS:
                        local_a, local_b = sample_bounds(item["start_time"], item["end_time"], sr, w["time_mapping"])
                        need(a == ca + local_a and b == ca + local_b, "unit:source_clock_mapping_mismatch")
            key = (u.get("window_id"), index)
            need(key not in seen, "unit:duplicate_native_occurrence");seen.add(key)
            if _integer(a) and _integer(b) and _integer(w.get("owned_start_sample")) and _integer(w.get("owned_end_sample")):
                midpoint = (a + b) / 2
                need(w["owned_start_sample"] <= midpoint < w["owned_end_sample"], "unit:wrong_owned_window")
    try:
        need(record.get("quality_axes") == quality_axes(record), "quality_axes:not_reconstructible_or_missing")
        if provenance.get("alignment_method") == "ultra_no_head":
            for w in record["windows"]:
                if isinstance(w, dict) and isinstance(w.get("alignment"), dict):
                    for reason in native_observation_errors(w["alignment"]):
                        need(False, "ultra_observation:" + reason)
        computed = quality_reasons(record)
        if decision.get("state") == "candidate":need(not computed, "decision:candidate_hides_quality_failure:" + ",".join(computed))
        if isinstance(decision.get("reasons"), list):need(set(computed).issubset(set(decision["reasons"])), "decision:quality_reasons_omitted")
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError) as e:
        errors.append("record:malformed_cross_field_structure:" + type(e).__name__)
    return sorted(set(errors))


def assert_valid_record(record):
    errors = validate_record(record)
    if errors:
        raise ValueError("sidecar validation refused: " + "; ".join(errors))
