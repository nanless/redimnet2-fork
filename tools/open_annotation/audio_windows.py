"""Bounded audio decoding and continuous-context windows in the source clock."""
from __future__ import annotations
import hashlib
import base64
import json
import math
from pathlib import Path, PurePosixPath
import traceback
from contracts import need, sha_file, stat_identity

PCM_POLICY = "decoded_float32le_native_interleaved/v1"
TIME_MAPPING = "source_offset_floor_start_ceil_end_v1"
_DECODER = None


def _shared_source_name(raw):
    """Only the two verified mount aliases; no basename/suffix matching."""
    need(isinstance(raw, str) and PurePosixPath(raw).is_absolute() and ".." not in PurePosixPath(raw).parts,
         "invalid absolute source path")
    prefixes = ("/root/group-shared/", "/shared/group-shared/")
    for prefix in prefixes:
        if raw.startswith(prefix): return "/verified-group-shared/" + raw[len(prefix):]
    return raw


def decoder_provenance():
    global _DECODER
    if _DECODER is None:
        import soundfile as sf
        maps = Path("/proc/self/maps")
        libs = sorted({line.split(None, 5)[5] for line in maps.read_text().splitlines()
                       if len(line.split(None, 5)) == 6 and "libsndfile" in line}) if maps.exists() else []
        _DECODER = {"soundfile_version": sf.__version__, "libsndfile_version": sf.__libsndfile_version__,
                    "mapped_libraries": [{"path": p, "sha256": sha_file(p)} for p in libs if Path(p).is_file()]}
    return _DECODER


def verify_decoder(pin):
    import soundfile as sf
    actual = decoder_provenance()
    need(actual["soundfile_version"] == pin["soundfile_version"]
         and actual["libsndfile_version"] == pin["libsndfile_version"], "actual decoder version differs")
    need(sha_file(sf.__file__) == pin["soundfile_sha256"], "actual soundfile Python source differs")
    matches = [x for x in actual["mapped_libraries"] if x["path"] == pin["library_path"] and x["sha256"] == pin["library_sha256"]]
    need(len(actual["mapped_libraries"]) == len(matches) == 1, "actual mapped libsndfile identity differs")
    return actual


def inspect_source(source):
    """Hash encoded bytes and every decoded native PCM frame with bounded RAM."""
    path = source.get("audio_path")
    missing = {"decode_status": "missing", "file_sha256": None, "pcm_sha256": None,
               "sample_rate": None, "channels": None, "frames": None, "bytes": None,
               "duration_seconds": None, "stat_before": None, "stat_after": None,
               "pcm_hash_policy": PCM_POLICY}
    if path is None or not Path(path).is_file():
        return missing, [{"stage": "source", "reason": "source_unresolved_or_missing", "audio_path": path}]
    identity = dict(missing, decode_status="decode_failed")
    try:
        import numpy as np
        import soundfile as sf
        before = stat_identity(path); identity.update(stat_before=before, bytes=before["bytes"])
        identity["file_sha256"] = sha_file(path)
        pcm = hashlib.sha256(); read_frames = 0
        with sf.SoundFile(path) as stream:
            sr, channels, frames = stream.samplerate, stream.channels, len(stream)
            need(sr > 0 and channels > 0 and frames >= 0, "invalid source clock")
            identity.update(sample_rate=sr, channels=channels, frames=frames, duration_seconds=frames / sr)
            while True:
                data = stream.read(65536, dtype="float32", always_2d=True)
                if len(data) == 0: break
                need(data.shape[1] == channels and bool(np.isfinite(data).all()), "invalid/nonfinite decoded PCM")
                pcm.update(data.astype("<f4", copy=False).tobytes(order="C")); read_frames += len(data)
        need(read_frames == frames, "decoder did not cover all declared frames")
        after = stat_identity(path); identity["stat_after"] = after
        need(before == after, "source changed during encoded/PCM audit")
        identity.update(decode_status="decoded_complete", pcm_sha256=pcm.hexdigest(), decoder=decoder_provenance())
        return identity, []
    except Exception:
        if path and Path(path).exists(): identity["stat_after"] = stat_identity(path)
        return identity, [{"stage": "source_decode", "traceback": traceback.format_exc()}]


def _capture_legacy_file(path, result, prefix, text_field, changed_message):
    """Capture read bytes before any decode/identity check can reject them."""
    before = stat_identity(path); result[prefix + "_stat_before"] = before
    raw = Path(path).read_bytes()
    result[prefix + "_sha256"] = hashlib.sha256(raw).hexdigest()
    result[prefix + "_bytes"] = len(raw)
    decode_error = None
    try:
        text = raw.decode("utf-8")
        result[text_field] = text; result[prefix + "_encoding"] = "utf-8"
    except UnicodeDecodeError as exc:
        decode_error = exc; text = None
        result[text_field] = None
        result[prefix + "_encoding"] = "invalid_utf8_preserved_as_base64"
        result[prefix + "_raw_base64"] = base64.b64encode(raw).decode("ascii")
        result[prefix + "_decode_error"] = traceback.format_exc()
    try:
        after = stat_identity(path); result[prefix + "_stat_after"] = after
    except Exception:
        result[prefix + "_stat_after"] = None
        result[prefix + "_stat_after_error"] = traceback.format_exc()
        raise
    need(before == after, changed_message)
    if decode_error is not None: raise decode_error
    return raw, text


def reuse_vad(source, identity):
    """Validate the actual old path/size/mtime/clock binding, not a fictitious SHA binding."""
    old = source.get("old_vad") or {}
    result = {"status": "missing", "reused": False, "original_path": old.get("path"),
              "metadata_path": old.get("metadata_path"), "original_sha256": None,
              "original_stat_before": None, "original_stat_after": None,
              "metadata_stat_before": None, "metadata_stat_after": None,
              "identity_binding": "legacy_source_path_size_mtime_clock", "segments": [], "errors": []}
    if not old.get("path") or not Path(old["path"]).is_file(): return result
    try:
        need(identity.get("decode_status") == "decoded_complete", "old VAD cannot bind an undecoded source")
        p = Path(old["path"])
        raw, text = _capture_legacy_file(p, result, "original", "original_text", "old VAD changed while reading")
        headers = {}; segments = []; result["headers"] = headers
        for line in text.splitlines():
            if line.startswith("#"):
                if "=" in line: k, v = line[1:].strip().split("=", 1); headers[k.strip()] = v.strip()
                continue
            if not line.strip(): continue
            parts = line.split(); need(len(parts) >= 5 and parts[0] == "SPEAKER", "invalid old RTTM row")
            start, duration = float(parts[3]), float(parts[4]); end = start + duration
            need(math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= identity["duration_seconds"] + 1e-6, "invalid old VAD interval")
            segments.append({"start_time": start, "end_time": end,
                             "start_sample": math.floor(start * identity["sample_rate"]),
                             "end_sample": math.ceil(end * identity["sample_rate"])})
        old_source = headers.get("source")
        if isinstance(old_source, str) and old_source.startswith("/"):
            need(_shared_source_name(old_source) == _shared_source_name(source["audio_path"]), "old VAD complete absolute source binding differs")
            result["binding_path_representation"] = "complete_absolute_path_with_two_verified_shared_mount_aliases"
        else:
            need(old_source == old.get("expected_source_relative"), "old VAD relative source binding differs")
            result["binding_path_representation"] = "complete_relative_source_path"
        expected = {"source_size": identity["bytes"], "source_mtime_ns": identity["stat_before"]["mtime_ns"],
                    "source_sample_rate": identity["sample_rate"], "source_channels": identity["channels"],
                    "source_frames": identity["frames"]}
        for key, value in expected.items(): need(int(headers[key]) == value, "old VAD clock/stat differs: " + key)
        for key, field in (("model_sha256", "expected_model_sha256"), ("inference_signature", "expected_inference_signature")):
            need(old.get(field) and headers.get(key) == old[field], "old VAD model/recipe binding differs: " + key)
        need(int(headers["segments"]) == len(segments), "old VAD segment count differs")
        metadata = None
        if old.get("metadata_path"):
            meta_path = Path(old["metadata_path"])
            meta_raw, meta_text = _capture_legacy_file(meta_path, result, "metadata", "metadata_raw_text", "old VAD metadata changed")
            try:
                metadata = json.loads(meta_text)
                result.update(metadata_parse_status="parsed", metadata=metadata)
            except Exception:
                result.update(metadata_parse_status="failed", metadata_parse_error=traceback.format_exc())
                raise
            for key in ("source_size", "source_mtime_ns", "source_sample_rate"):
                need(metadata[key] == expected[key], "old metadata binding differs: " + key)
            need(metadata["source_num_channels"] == identity["channels"] and metadata["segments"] == len(segments)
                 and metadata["inference_signature"] == old["expected_inference_signature"], "old VAD metadata identity differs")
            result["metadata_sha256"] = hashlib.sha256(meta_raw).hexdigest()
        result.update(status="reused" if segments else "reused_empty", reused=True,
                      original_sha256=hashlib.sha256(raw).hexdigest(), original_text=text,
                      headers=headers, metadata=metadata, segments=segments,
                      old_source_byte_sha_binding_claimed=False)
    except Exception:
        result.update(status="identity_mismatch_or_invalid", reused=False, segments=[],
                      errors=[{"stage": "reuse_vad", "traceback": traceback.format_exc()}])
    return result


def plan_windows(identity, source, runtime, vad=None):
    sr, frames = identity["sample_rate"], identity["frames"]
    interval = source.get("source_interval")
    start, stop = (0, frames) if interval is None else (interval["start_sample"], interval["end_sample"])
    if interval is not None: need(interval.get("sample_rate", sr) == sr, "source interval clock differs")
    need(type(start) is int and type(stop) is int and 0 <= start < stop <= frames, "source interval outside audio")
    policy = runtime["window"]; core = int(round(policy["core_seconds"] * sr)); context = int(round(policy["context_seconds"] * sr))
    need(core > 0 and context >= 0 and core + 2 * context <= 30 * sr, "invalid bounded model window")
    # VAD can suggest a quiet cut. No VAD interval is concatenated or omitted.
    quiet = []
    if vad and vad.get("reused"):
        segs = vad["segments"]
        for a, b in zip(segs, segs[1:]):
            if b["start_sample"] > a["end_sample"]: quiet.append((a["end_sample"] + b["start_sample"]) // 2)
    result = []; pos = start
    while pos < stop:
        end = min(stop, pos + core)
        if end != stop:
            candidates = [q for q in quiet if pos + max(1, core // 2) <= q <= min(stop, pos + 30 * sr - 2 * context)
                          and abs(q - end) <= sr]
            if candidates: end = min(candidates, key=lambda q: (abs(q - end), q))
        a, b = max(start, pos - context), min(stop, end + context)
        need(a <= pos < end <= b and b - a <= 30 * sr, "model window clock/length differs")
        result.append({"window_id": "window_%06d" % len(result), "context_start_sample": a,
                       "context_end_sample": b, "owned_start_sample": pos, "owned_end_sample": end,
                       "model_sample_rate": 16000, "time_mapping": TIME_MAPPING,
                       "left_context_limited": a == start, "right_context_limited": b == stop})
        pos = end
    need(result[0]["owned_start_sample"] == start and result[-1]["owned_end_sample"] == stop
         and all(x["owned_end_sample"] == y["owned_start_sample"] for x, y in zip(result, result[1:])), "windows do not cover source")
    return result


def read_window(source, identity, window):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    need(stat_identity(source["audio_path"]) == identity["stat_after"], "source changed before window read")
    a, b = window["context_start_sample"], window["context_end_sample"]
    with sf.SoundFile(source["audio_path"]) as stream:
        need(stream.samplerate == identity["sample_rate"] and stream.channels == identity["channels"], "window source clock changed")
        stream.seek(a); data = stream.read(b - a, dtype="float32", always_2d=True)
    need(len(data) == b - a and bool(np.isfinite(data).all()), "short or nonfinite window read")
    need(source["channel_policy"] == "mean_channels_to_mono", "unfrozen channel policy")
    mono = data.mean(axis=1, dtype=np.float32)
    sr = identity["sample_rate"]
    if sr != 16000:
        divisor = math.gcd(sr, 16000); mono = resample_poly(mono, 16000 // divisor, sr // divisor).astype(np.float32)
    need(mono.ndim == 1 and len(mono) > 0 and bool(np.isfinite(mono).all()), "invalid model PCM")
    need(len(mono) <= 30 * 16000 + 1, "resampled model window exceeds 30 seconds")
    need(stat_identity(source["audio_path"]) == identity["stat_after"], "source changed during window read")
    window["input_pcm_sha256"] = hashlib.sha256(mono.astype("<f4", copy=False).tobytes()).hexdigest()
    window["model_input_frames"] = len(mono)
    window["resampling_policy"] = "scipy_resample_poly_gcd_float32/v1"
    return mono


def units_from_windows(windows, identity):
    sr, frames = identity["sample_rate"], identity["frames"]; units = []; flags = []
    for w in windows:
        alignment = w.get("alignment", {})
        for i, item in enumerate(alignment.get("items", [])):
            a, b = item.get("start_time"), item.get("end_time")
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b)):
                flags.append({"window_id": w["window_id"], "native_index": i, "reason": "nonfinite_native_geometry"}); continue
            start = w["context_start_sample"] + math.floor(a * sr); end = w["context_start_sample"] + math.ceil(b * sr)
            if not (0 <= start < end <= frames and w["context_start_sample"] <= start < end <= w["context_end_sample"]):
                flags.append({"window_id": w["window_id"], "native_index": i, "reason": "invalid_native_geometry"}); continue
            midpoint = (start + end) / 2
            if not w["owned_start_sample"] <= midpoint < w["owned_end_sample"]: continue
            if not isinstance(item.get("text"), str) or not item["text"].strip(): continue
            units.append({"text": item["text"], "start_sample": start, "end_sample": end, "sample_rate": sr,
                          "window_id": w["window_id"], "native_index": i, "boundary_status": "machine_candidate",
                          "geometry_status": "valid", "repair_applied": bool(alignment.get("repair_changed"))})
    units.sort(key=lambda u: (u["start_sample"], u["end_sample"], u["window_id"], u["native_index"]))
    counts = {}
    for u in units:
        u["occurrence_index"] = counts.get(u["text"], 0); counts[u["text"]] = u["occurrence_index"] + 1
    for a, b in zip(units, units[1:]):
        if a["end_sample"] > b["start_sample"]:
            flags.append({"reason": "cross_window_or_native_overlap_unresolved", "left": [a["window_id"], a["native_index"]], "right": [b["window_id"], b["native_index"]]})
    return units, flags
