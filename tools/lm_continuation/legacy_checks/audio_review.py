"""Bounded, read-only CPU review of the exact audio files in an event ledger.

Only audit_audio_files() loads soundfile/numpy. A test decoder supplies versions,
open(path), and isfinite(block); its reader follows SoundFile's context-manager,
frames/samplerate/channels, seek(), tell(), and read() interface. Blocks support
shape/dtype, contiguous frame slicing and tobytes(order="C"). No audio is written.
"""
import contextlib
import datetime as dt
import hashlib
import numbers
import os
from pathlib import Path
import re
import stat


class AudioRefused(RuntimeError):
    """Any failed read or identity check refuses the entire review."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BLOCK_FRAMES = 65536
MAX_DURATION_SECONDS = 30 * 60
PRODUCTION_REQUESTS = (("ptn", 35464), ("lm", 88264))


def _need(condition, message):
    if not condition:
        raise AudioRefused(message)


@contextlib.contextmanager
def _single_thread_environment():
    previous = {key: os.environ.get(key) for key in THREAD_KEYS}
    try:
        for key in THREAD_KEYS:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _default_decoder():
    import soundfile
    import numpy

    class Decoder:
        versions = {"soundfile": soundfile.__version__,
                    "libsndfile": soundfile.__libsndfile_version__,
                    "numpy": numpy.__version__}

        @staticmethod
        def open(path):
            return soundfile.SoundFile(str(path), mode="r")

        @staticmethod
        def isfinite(block):
            return bool(numpy.isfinite(block).all())

    return Decoder()


def _state(info):
    return {"device": info.st_dev, "inode": info.st_ino,
            "bytes": info.st_size, "mode": info.st_mode,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def _paths(item):
    _need(isinstance(item, dict), "Audio ledger entry must be an object")
    for key in ("path", "resolved", "sample_key"):
        _need(isinstance(item.get(key), str) and bool(item[key]),
              "Audio ledger requires nonempty " + key)
    path, resolved = Path(item["path"]), Path(item["resolved"])
    _need(path.is_absolute() and resolved.is_absolute()
          and ".." not in path.parts and ".." not in resolved.parts,
          "Audio paths must be absolute and without traversal")
    _need(type(item.get("bytes")) is int and item["bytes"] > 0,
          "Audio byte length must be a positive integer")
    _need(isinstance(item.get("sha256"), str)
          and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
          "Audio ledger requires an exact SHA256")
    _need(path.resolve(strict=True) == resolved
          and resolved.resolve(strict=True) == resolved,
          "Audio resolved path differs from reviewed identity: " + str(path))
    _need(stat.S_ISREG(resolved.lstat().st_mode),
          "Audio target must be a regular file")
    return path, resolved


def _fingerprint(item):
    path, resolved = _paths(item)
    before = resolved.stat()
    _need(before.st_size == item["bytes"], "Audio size differs: " + str(path))
    digest, byte_count = hashlib.sha256(), 0
    with resolved.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        _need(_state(opened) == _state(before), "Audio changed before hashing")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
        ended = os.fstat(stream.fileno())
    after = resolved.stat()
    _need(_state(before) == _state(ended) == _state(after)
          and path.resolve(strict=True) == resolved,
          "Audio changed while hashing")
    _need(byte_count == item["bytes"] and digest.hexdigest() == item["sha256"],
          "Audio SHA256 differs: " + str(path))
    return {"before": _state(before), "after": _state(after),
            "sha256": digest.hexdigest(), "resolved": str(resolved)}


def _unchanged(item, expected):
    _, resolved = _paths(item)
    current = _state(resolved.stat())
    _need(current == expected, "Audio changed during decode: " + item["sample_key"])
    return current


def _header(reader, item):
    values = {key: getattr(reader, key) for key in ("frames", "samplerate", "channels")}
    _need(all(isinstance(v, numbers.Integral) and not isinstance(v, bool)
              and v > 0 for v in values.values()), "Invalid audio header")
    values = {key: int(value) for key, value in values.items()}
    _need(values["frames"] <= MAX_DURATION_SECONDS * values["samplerate"],
          "Audio exceeds 30 minute review bound")
    for key in values:
        if key in item:
            _need(item[key] == values[key], "Audio header differs from ledger: " + key)
    return values


def _read(reader, decoder, trial, kind, start, requested, expected, channels):
    record = {"kind": kind, "start_frame": start, "requested_frames": requested,
              "expected_frames": expected, "ok": False}
    trial["reads"].append(record)
    try:
        block = reader.read(requested, dtype="float32", always_2d=True)
        shape = getattr(block, "shape", None)
        _need(shape is not None and len(shape) == 2
              and all(isinstance(v, numbers.Integral) and not isinstance(v, bool)
                      for v in shape), "Decoded block must have a two-dimensional shape")
        record["received_frames"] = int(shape[0])
        _need(int(shape[0]) == expected and int(shape[1]) == channels,
              "Decoded frame/channel count differs from exact expected count")
        record["finite"] = bool(decoder.isfinite(block))
        _need(record["finite"], "Nonfinite decoded audio")
        record["ok"] = True
    except Exception as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        raise


def _decode_once(item, decoder, record, number, expected_state, expected_header):
    trial = {"open_number": number, "ok": False, "reads": []}
    record["opens"].append(trial)
    trial["file_before"] = _unchanged(item, expected_state)
    with decoder.open(item["resolved"]) as reader:
        header = _header(reader, item)
        trial["header"] = header
        if expected_header is not None:
            _need(header == expected_header, "Audio header changed across independent opens")
        frames, rate, channels = (header[k] for k in ("frames", "samplerate", "channels"))
        position = 0
        while position < frames:
            requested = min(BLOCK_FRAMES, frames - position)
            _read(reader, decoder, trial, "full", position, requested, requested, channels)
            position += requested
        _read(reader, decoder, trial, "full_eof", frames, 1, 0, channels)
        window = 3 * rate
        windows = (("start_3s", 0, window),
                   ("middle_3s", max(0, (frames - window) // 2), window),
                   ("end_3s", max(0, frames - window), window),
                   ("near_eof_50ms", max(0, frames - max(1, rate // 20)), max(1, rate // 20)))
        for kind, start, requested in windows:
            seek_record = {"kind": kind, "frame": start, "ok": False}
            trial.setdefault("seeks", []).append(seek_record)
            actual = reader.seek(start)
            _need(isinstance(actual, numbers.Integral) and not isinstance(actual, bool)
                  and int(actual) == start, "Audio seek returned an unexpected position")
            seek_record.update({"returned_frame": int(actual), "ok": True})
            _read(reader, decoder, trial, kind, start, requested,
                  min(requested, frames - start), channels)
    trial["file_after"] = _unchanged(item, expected_state)
    trial["ok"] = True
    return header


def _production_windows(item, header):
    _need(header["samplerate"] == 16000,
          "Production crop contract requires the reviewed 16000 Hz source")
    windows = []
    for stage, requested in PRODUCTION_REQUESTS:
        span = max(0, header["frames"] - requested)
        starts = {0, span // 2, max(0, span - 1)}
        # Generic fixed regression points apply to every source when legal.
        # This preserves the reviewed failure starts without publishing sample IDs.
        if stage == "ptn" and 24593 < span:
            starts.add(24593)
        boundaries = {point + delta for point in (159744, 319488)
                      for delta in (-2, -1, 0, 1, 2)}
        starts.update(start for start in boundaries if 0 <= start < span)
        for start in sorted(starts):
            windows.append({"stage": stage, "start_frame": start,
                            "requested_frames": requested,
                            "expected_frames": min(requested, header["frames"] - start)})
    return windows


def _tell(reader, expected):
    position = reader.tell()
    _need(isinstance(position, numbers.Integral) and not isinstance(position, bool)
          and int(position) == expected, "Production audio tell returned an unexpected position")
    return int(position)


def _production_read(reader, decoder, trial, start, requested, expected, channels):
    read = {"start_frame": start, "requested_frames": requested,
            "expected_frames": expected, "read_mode": "soundfile_default_float64",
            "ok": False, "step": "tell_before"}
    trial.setdefault("reads", []).append(read)
    try:
        read["tell_before"] = _tell(reader, start)
        read["step"] = "read"
        block = reader.read(requested)
        shape = getattr(block, "shape", None)
        expected_shape = (expected,) if channels == 1 else (expected, channels)
        _need(shape == expected_shape, "Production decoded frame/channel count differs")
        read["received_frames"] = int(shape[0])
        dtype = getattr(block, "dtype", None)
        _need(getattr(dtype, "kind", None) == "f" and getattr(dtype, "itemsize", None) == 8,
              "Production default read must return float64 PCM")
        read["finite"] = bool(decoder.isfinite(block))
        _need(read["finite"], "Nonfinite production decoded audio")
        read["step"] = "tell_after"
        read["tell_after"] = _tell(reader, start + expected)
        read["ok"] = True
        return block
    except Exception as exc:
        read["error"] = type(exc).__name__ + ": " + str(exc)
        raise


def _pcm_bytes(block, frames, channels):
    value = block.tobytes(order="C")
    _need(isinstance(value, bytes) and len(value) == frames * channels * 8,
          "Production PCM byte count differs")
    return value


def _production_review(item, decoder, record, expected_state, expected_header):
    windows = _production_windows(item, expected_header)
    frames, channels = expected_header["frames"], expected_header["channels"]
    reference = {"purpose": "independent_sequential_float64_reference", "ok": False,
                 "pcm_encoding": "float64_native_endian_interleaved_C",
                 "reads": [], "file_before": _unchanged(item, expected_state)}
    record["production_reference"] = reference
    digests = [hashlib.sha256() for _ in windows]
    with decoder.open(item["resolved"]) as reader:
        _need(_header(reader, item) == expected_header, "Production reference header changed")
        position = 0
        while position < frames:
            count = min(BLOCK_FRAMES, frames - position)
            block = _production_read(reader, decoder, reference, position, count, count, channels)
            for window, digest in zip(windows, digests):
                start = max(position, window["start_frame"])
                end = min(position + count, window["start_frame"] + window["expected_frames"])
                if start < end:
                    digest.update(_pcm_bytes(block[start-position:end-position], end-start, channels))
            position += count
    reference["file_after"] = _unchanged(item, expected_state)
    reference["ok"] = True
    record["production_opens"] = []
    for number, (window, digest) in enumerate(zip(windows, digests), 1):
        trial = dict(window, open_number=number, purpose="fresh_open_production_crop",
                     ok=False, reads=[], step="identity_before")
        record["production_opens"].append(trial)
        trial["file_before"] = _unchanged(item, expected_state)
        try:
            trial["step"] = "open"
            with decoder.open(item["resolved"]) as reader:
                _need(_header(reader, item) == expected_header, "Production crop header changed")
                trial["tell_on_open"] = _tell(reader, 0)
                trial["step"] = "seek"
                returned = reader.seek(window["start_frame"])
                _need(isinstance(returned, numbers.Integral) and not isinstance(returned, bool)
                      and int(returned) == window["start_frame"],
                      "Production audio seek returned an unexpected position")
                trial["seek_returned"] = int(returned)
                trial["step"] = "read"
                block = _production_read(reader, decoder, trial, window["start_frame"],
                                         window["requested_frames"], window["expected_frames"], channels)
                trial["pcm_sha256"] = hashlib.sha256(
                    _pcm_bytes(block, window["expected_frames"], channels)).hexdigest()
                trial["reference_pcm_sha256"] = digest.hexdigest()
                trial["step"] = "compare_sequential_pcm"
                _need(trial["pcm_sha256"] == trial["reference_pcm_sha256"],
                      "Production crop PCM differs from sequential reference slice")
            trial["step"] = "identity_after"
            trial["file_after"] = _unchanged(item, expected_state)
            trial["ok"] = True
        except Exception as exc:
            trial["error"] = type(exc).__name__ + ": " + str(exc)
            raise


def audit_audio_files(ledger, decoder=None):
    """Validate identity, two full/stress decodes and fresh production crops.

    audio entries: sample_key/path/resolved/bytes/sha256, optionally the exact
    frames/samplerate/channels. Duplicated samples or targets are refused.
    The fixed production requests apply only to the reviewed 16000 Hz contract.
    Each production crop matches its sequential float64 reference slice SHA256.
    Return JSON-safe metadata and every read, never PCM data. Any failure raises
    AudioRefused immediately; .evidence retains the partial audit for reporting.
    Thread limits are temporary, and the caller's CUDA visibility is untouched.
    """
    evidence = {"schema": 2, "ok": False, "readonly_cpu": True,
                "cuda_used": False, "decoder_injected": decoder is not None,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "thread_environment": {key: "1" for key in THREAD_KEYS}, "audio": []}
    try:
        _need(isinstance(ledger, dict) and isinstance(ledger.get("audio"), list)
              and bool(ledger["audio"]), "Ledger audio must be a nonempty list")
        seen_keys, seen_targets = set(), set()
        for item in ledger["audio"]:
            _, resolved = _paths(item)
            _need(item["sample_key"] not in seen_keys and str(resolved) not in seen_targets,
                  "Duplicate audio sample or target in ledger")
            seen_keys.add(item["sample_key"])
            seen_targets.add(str(resolved))
            initial = _fingerprint(item)
            evidence["audio"].append({"sample_key": item["sample_key"],
                                      "path": item["path"], "resolved": str(resolved),
                                      "initial_identity": initial, "opens": []})
        with _single_thread_environment():
            active_decoder = _default_decoder() if decoder is None else decoder
            versions = getattr(active_decoder, "versions", None)
            _need(isinstance(versions, dict) and all(
                isinstance(versions.get(key), str) and bool(versions[key])
                for key in ("soundfile", "libsndfile", "numpy")),
                "Decoder must report soundfile/libsndfile/numpy versions")
            evidence["backend_versions"] = dict(versions)
            for item, record in zip(ledger["audio"], evidence["audio"]):
                header = None
                for number in (1, 2):
                    header = _decode_once(item, active_decoder, record, number,
                                          record["initial_identity"]["after"], header)
                _production_review(item, active_decoder, record,
                                   record["initial_identity"]["after"], header)
                final = _fingerprint(item)
                _need(final["after"] == record["initial_identity"]["after"],
                      "Audio identity changed before final SHA verification")
                record["final_identity"] = final
                record["ok"] = True
        evidence["ok"] = True
        evidence["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        evidence["meaning"] = ("These bounded current reads passed; historical random crop "
                               "parameters are not reconstructed; the fixed fifth-file initial-seek regression is included.")
        return evidence
    except Exception as exc:
        evidence["error"] = type(exc).__name__ + ": " + str(exc)
        evidence["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        raise AudioRefused(str(exc), evidence=evidence) from exc
