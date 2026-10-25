"""真实长度、固定共同试验表的说话人验证评测。GPU必须显式授权并给预算。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


class EvaluationRefused(ValueError):
    """输入身份、真实长度或缓存内容不能满足评测条件。"""


STANDARD_O_SHA256 = "0bc0a0fe3e557f1a75fb71e566d862d460709e80a4fe28e80e49bc0ab3a536ea"
class AudioDecodeError(EvaluationRefused):
    """仅单条音频解码失败，可在有界错误预算内继续。"""


class ModelExecutionError(EvaluationRefused):
    """模型、设备或数值失败，停止后续全部裁片。"""


ALL_O_AXIS = (0.5, 1, 2, 3, "full")
COMMON_8_AXIS = (0.5, 1, 2, 3, 5, 8, "full")
PREPROCESS = {"schema": 1, "sample_rate": 16000, "decode": "soundfile-float32",
              "channels": "arithmetic-mean", "resample": False,
              "crop": "nested-center-floor-samples-v1", "vad": False,
              "wrap": False, "pad": False, "embedding": "L2", "precision": "float32"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stat_id(path: Path) -> list[int]:
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()



def protect_output(path: Path) -> None:
    resolved = Path(path).resolve()
    for root in ('/root/code/github_repos/redimnet2-fork',
                 '/root/experiments/redimnet2/b6-mix-official-20261019',
                 '/root/group-shared/voiceprint/data', '/shared/group-shared/voiceprint/data'):
        protected = Path(root).resolve()
        if resolved == protected or protected in resolved.parents:
            raise EvaluationRefused(f"output targets a protected original tree: {resolved}")


def read_trials(path: Path) -> tuple[list[dict], dict]:
    """保持原试验顺序；音频标识必须相对音频根目录，第一层是身份。"""
    path = Path(path)
    before = stat_id(path)
    raw = path.read_bytes()
    if stat_id(path) != before:
        raise EvaluationRefused("protocol changed while reading")
    trials = []
    for index, line in enumerate(raw.decode().splitlines()):
        fields = line.split()
        if len(fields) != 3 or fields[0] not in ("0", "1"):
            raise EvaluationRefused(f"bad protocol row {index + 1}")
        for value in fields[1:]:
            p = Path(value)
            if p.is_absolute() or ".." in p.parts or len(p.parts) < 3:
                raise EvaluationRefused(f"unsafe or speakerless relative audio key: {value}")
        if (fields[0] == '1') != (Path(fields[1]).parts[0] == Path(fields[2]).parts[0]):
            raise EvaluationRefused(f"label and speaker IDs disagree at row {index + 1}")
        trials.append({"trial_id": f"{index:08d}", "label": int(fields[0]),
                       "enroll": fields[1], "verify": fields[2],
                       "enroll_speaker": Path(fields[1]).parts[0],
                       "verify_speaker": Path(fields[2]).parts[0]})
    if not trials or {t["label"] for t in trials} != {0, 1}:
        raise EvaluationRefused("protocol needs positive and negative trials")
    return trials, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(trials)}


def audio_record(root: Path, relative: str, hash_audio: bool = True) -> dict:
    import soundfile as sf
    root = Path(root).resolve()
    path = root / relative
    # Existing test-set symlinks may resolve outside root; key traversal was rejected above.
    before = stat_id(path)
    info = sf.info(str(path))
    if info.samplerate != 16000 or info.frames <= 0 or info.channels < 1:
        raise EvaluationRefused(f"invalid audio header: {relative}: {info}")
    checksum = sha_file(path) if hash_audio else None
    after = stat_id(path)
    if before != after:
        raise EvaluationRefused(f"audio changed during indexing: {relative}")
    return {"relative": relative, "path": str(path.absolute()), "resolved": str(path.resolve()),
            "frames": int(info.frames), "sample_rate": int(info.samplerate),
            "channels": int(info.channels), "seconds": info.frames / info.samplerate,
            "format": info.format, "subtype": info.subtype, "stat": before, "sha256": checksum}


def audio_content_identity(record: dict) -> dict:
    # Device/inode numbers differ across worker namespaces; bind bytes/header instead.
    return {k: record[k] for k in ('relative', 'resolved', 'frames', 'sample_rate',
                                    'channels', 'format', 'subtype', 'sha256')}


def index_audio(root: Path, trials: list[dict]) -> dict[str, dict]:
    records = {}
    for rel in sorted({t[k] for t in trials for k in ("enroll", "verify")}):
        records[rel] = audio_record(root, rel)
    return records


def center_crop(record: dict, duration: float | str) -> dict:
    """所有时长围绕同一中心，整数样本区间互相嵌套，绝不补音频。"""
    frames, rate = int(record["frames"]), int(record["sample_rate"])
    if duration == "full":
        count = frames
    else:
        if isinstance(duration, bool):
            raise EvaluationRefused("boolean duration")
        seconds = float(duration)
        samples = seconds * rate
        if not math.isfinite(seconds) or seconds <= 0 or not math.isclose(samples, round(samples), abs_tol=1e-7):
            raise EvaluationRefused("duration must be positive finite and an integer sample count")
        count = int(round(samples))
        if count > frames:
            raise EvaluationRefused(f"too short for {duration}s: {record['relative']}")
    start = (frames - count) // 2
    return {"relative": record["relative"], "source_resolved": record["resolved"],
            "source_sha256": record["sha256"], "source_stat": record["stat"],
            "source_frames": frames, "sample_rate": rate, "source_channels": record["channels"],
            "duration": duration, "start_sample": start, "end_sample": start + count,
            "actual_samples": count, "actual_seconds": count / rate,
            "policy": PREPROCESS["crop"]}


def build_matrices(trials: list[dict], records: dict[str, dict], strict_standard: bool = False) -> list[dict]:
    eligible = [t for t in trials if all(records[t[k]]["frames"] >= 8 * records[t[k]]["sample_rate"]
                                       for k in ("enroll", "verify"))]
    if strict_standard and (len(trials) != 37611 or len(records) != 4708 or len(eligible) != 4460):
        raise EvaluationRefused("standard O identity/counts differ from frozen 37611/4708/4460")
    if not eligible or {t["label"] for t in eligible} != {0, 1}:
        raise EvaluationRefused("8-second common subset lacks both trial classes")
    matrices = []
    for name, subset, axis in [("all-O", trials, ALL_O_AXIS), ("common-8s", eligible, COMMON_8_AXIS)]:
        subset_sha = digest(subset)
        # Validate every endpoint against every axis before starting costly extraction.
        for rel in {t[k] for t in subset for k in ("enroll", "verify")}:
            for duration in axis:
                center_crop(records[rel], duration)
        matrices.append({"name": name, "axis": list(axis), "trials": subset,
                         "trialset_sha256": subset_sha, "trial_count": len(subset),
                         "positive": sum(t["label"] for t in subset),
                         "negative": sum(1 - t["label"] for t in subset),
                         "speakers": len({t[k] for t in subset for k in ("enroll_speaker", "verify_speaker")})})
    return matrices


def source_identity(repo: Path) -> dict:
    paths = sorted((Path(repo) / "redimnet2").rglob("*.py"))
    files = {str(p.relative_to(repo)): sha_file(p) for p in paths}
    files["duration_ft/evaluate.py"] = sha_file(Path(__file__))
    files["duration_ft/metrics.py"] = sha_file(Path(__file__).with_name("metrics.py"))
    return {"sha256": digest(files), "files": files}


def load_model(checkpoint: Path, device: str = "cpu", *, allow_gpu: bool = False,
               gpu_memory_mib: int | None = None):
    import torch
    import redimnet2
    from redimnet2 import ReDimNet2Wrap
    repo = Path(__file__).resolve().parents[1]
    try:
        Path(redimnet2.__file__).resolve().relative_to(repo)
    except ValueError as error:
        raise EvaluationRefused("model import escaped independent source tree") from error
    checkpoint = Path(checkpoint)
    before = stat_id(checkpoint)
    model_sha = sha_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), dict) or not isinstance(payload.get("state_dict"), dict):
        raise EvaluationRefused("evaluation requires strict released/export {model_config,state_dict}")
    model = ReDimNet2Wrap(**payload["model_config"])
    result = model.load_state_dict(payload["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise EvaluationRefused("strict model load failed")
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise EvaluationRefused(f"nonfinite checkpoint: {name}")
    if stat_id(checkpoint) != before:
        raise EvaluationRefused("checkpoint changed while loading")
    requested = torch.device(device)
    if requested.type == "cuda":
        if not allow_gpu or not gpu_memory_mib or gpu_memory_mib <= 0:
            raise EvaluationRefused("GPU requires --allow-gpu and a positive explicit memory MiB budget")
        index = requested.index if requested.index is not None else 0
        total = torch.cuda.get_device_properties(index).total_memory
        fraction = gpu_memory_mib * 2**20 / total
        if fraction > 0.9:
            raise EvaluationRefused("GPU budget must not exceed 90% of a device")
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif requested.type != "cpu":
        raise EvaluationRefused("only CPU/CUDA are supported")
    torch.set_num_threads(1)
    model.float().to(requested).eval()
    import soundfile as sf
    decoder_files = {}
    maps = Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            if 'libsndfile' in line:
                library = Path(line.split(maxsplit=5)[-1])
                if library.is_file():
                    decoder_files[str(library.resolve())] = sha_file(library)
    identity = {"checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": model_sha,
                "model_config": payload["model_config"], "code": source_identity(repo),
                "torch": torch.__version__, "soundfile": sf.__version__,
                "libsndfile": sf.__libsndfile_version__, "decoder_files": decoder_files,
                "soundfile_module_sha256": sha_file(Path(sf.__file__)), "numpy": np.__version__,
                "device": str(requested), "gpu_memory_mib": gpu_memory_mib,
                "precision": "float32", "tf32": False, "embedding_dim": payload["model_config"]["embed_dim"]}
    return model, identity


def cache_identity(model_identity: dict, crop: dict) -> dict:
    # Source byte SHA, exact sample interval, preprocessing, runtime and all model-code hashes.
    return {"schema": 1, "model": model_identity, "preprocess": PREPROCESS,
            "crop": {k: v for k, v in crop.items() if k != "source_stat"}}


class EmbeddingCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, identity: dict) -> Path:
        return self.directory / (digest(identity) + ".json")

    def get(self, identity: dict) -> np.ndarray | None:
        path = self.path(identity)
        if not path.exists():
            return None
        try:
            saved = json.loads(path.read_text())
            vector = np.asarray(saved["vector"], dtype=np.float32)
            if saved["identity"] != identity or saved["vector_sha256"] != digest(saved["vector"]):
                raise EvaluationRefused("cache identity/checksum differs")
            dim = identity["model"]["embedding_dim"]
            if vector.shape != (dim,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=2e-5):
                raise EvaluationRefused("invalid cached vector")
            return vector
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EvaluationRefused(f"refusing existing corrupt cache {path}: {error}") from error

    def put(self, identity: dict, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (identity["model"]["embedding_dim"],) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=2e-5):
            raise EvaluationRefused("refusing invalid embedding")
        values = vector.tolist()
        write_json(self.path(identity), {"identity": identity, "vector": values, "vector_sha256": digest(values)})


def extract_one(model, model_identity: dict, record: dict, crop: dict, cache: EmbeddingCache) -> tuple[np.ndarray, bool]:
    path = Path(record["path"])
    if stat_id(path) != record["stat"]:
        raise EvaluationRefused("audio changed since indexing")
    cached = cache.get(cache_identity(model_identity, crop))
    if cached is not None:
        return cached, True
    import torch
    import soundfile as sf
    path = Path(record["path"])
    if stat_id(path) != record["stat"]:
        raise EvaluationRefused("audio changed since indexing")
    try:
        audio, rate = sf.read(str(path), start=crop["start_sample"], frames=crop["actual_samples"], dtype="float32", always_2d=True)
    except Exception as error:
        if isinstance(error, MemoryError):
            raise
        raise AudioDecodeError(f"soundfile decode failed: {path}") from error
    if rate != record["sample_rate"] or len(audio) != crop["actual_samples"] or not np.isfinite(audio).all():
        raise AudioDecodeError("audio decode incomplete/nonfinite")
    if stat_id(path) != record["stat"]:
        raise EvaluationRefused("audio changed while decoding")
    waveform = np.mean(audio, axis=1, dtype=np.float32)
    try:
        with torch.inference_mode():
            result = model(torch.from_numpy(waveform).unsqueeze(0).to(model_identity["device"]))
            if result.shape != (1, model_identity["embedding_dim"]):
                raise ModelExecutionError("model returned wrong embedding shape")
            result = result.detach().float().cpu().numpy()[0]
    except Exception as error:
        raise ModelExecutionError(f"model/device failure: {type(error).__name__}: {error}") from error
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or not math.isfinite(norm) or norm <= 1e-12:
        raise ModelExecutionError("zero/nonfinite embedding")
    vector = result / norm
    cache.put(cache_identity(model_identity, crop), vector)
    return vector, False


def score_cell(trials: list[dict], enroll_vectors: dict, verify_vectors: dict) -> tuple[list[dict], list[float], list[int]]:
    rows, scores, labels = [], [], []
    for trial in trials:
        row = dict(trial)
        if trial["enroll"] not in enroll_vectors or trial["verify"] not in verify_vectors:
            row.update(status="missing_embedding", score=None)
        else:
            score = float(np.dot(enroll_vectors[trial["enroll"]], verify_vectors[trial["verify"]]))
            if not math.isfinite(score):
                raise EvaluationRefused("nonfinite cosine")
            row.update(status="ok", score=score)
            scores.append(score)
            labels.append(trial["label"])
        rows.append(row)
    return rows, scores, labels


def evaluate_matrix(checkpoint: Path, protocol: Path, audio_root: Path, output: Path, *,
                    device="cpu", allow_gpu=False, gpu_memory_mib=None, strict_standard=True,
                    calibration: Path | None = None, max_audio_errors: int = 10,
                    max_consecutive_audio_errors: int = 3) -> dict:
    """一次提取所有唯一裁片；两张表始终保留各自同一试验集合。"""
    from duration_ft.metrics import score_metrics, fixed_threshold_metrics
    output = Path(output)
    protect_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if max_audio_errors < 1 or max_consecutive_audio_errors < 1:
        raise EvaluationRefused("audio error limits must be positive")
    trials, protocol_identity = read_trials(protocol)
    if strict_standard and protocol_identity['sha256'] != STANDARD_O_SHA256:
        raise EvaluationRefused("standard Vox1-O protocol SHA differs")
    records, header_errors = {}, []
    for rel in sorted({t[k] for t in trials for k in ("enroll", "verify")}):
        try:
            records[rel] = audio_record(audio_root, rel)
        except Exception as error:
            header_errors.append({"relative": rel, "type": type(error).__name__, "message": str(error)})
    write_json(output / "index_errors.json", header_errors)
    if header_errors:
        raise EvaluationRefused("audio indexing failed; see index_errors.json; no trials silently dropped")
    matrices = build_matrices(trials, records, strict_standard)
    try:
        model, model_identity = load_model(checkpoint, device, allow_gpu=allow_gpu, gpu_memory_mib=gpu_memory_mib)
    except Exception as error:
        failure = {"complete": False, "phase": "model_load", "cells": [], "unique_crops": 0,
                   "cache_hits": 0, "extraction_errors": [{"type": type(error).__name__,
                   "message": str(error), "traceback": traceback.format_exc()}]}
        write_json(output / 'failures' / f'model-load-{time.time_ns()}.json', failure)
        write_json(output / 'summary.json', failure)
        return failure
    cache = EmbeddingCache(output / "embedding-cache")
    planned_crops = {}
    for matrix in matrices:
        for rel in sorted({t[k] for t in matrix["trials"] for k in ("enroll", "verify")}):
            for duration in matrix["axis"]:
                crop = center_crop(records[rel], duration)
                planned_crops[(rel, str(duration))] = {k: v for k, v in crop.items() if k != "source_stat"}
    comparison = {"schema": 1, "protocol_sha256": protocol_identity["sha256"],
                  "matrices": [{k: m[k] for k in ("name", "axis", "trialset_sha256")} for m in matrices],
                  "audio_content_sha256": digest({k: audio_content_identity(v) for k, v in records.items()}),
                  "crops_sha256": digest([planned_crops[k] for k in sorted(planned_crops)]),
                  "preprocess": PREPROCESS, "duration_unit": "seconds; full=natural waveform",
                  "crop_seed": None, "evaluation_code_sha256": sha_file(Path(__file__))}
    run_identity = {"schema": 1, "protocol": protocol_identity, "matrices": comparison['matrices'],
                    "audio_index_sha256": comparison["audio_content_sha256"],
                    "comparison_identity": comparison, "comparison_identity_sha256": digest(comparison),
                    "calibration_sha256": sha_file(Path(calibration)) if calibration else None,
                    "error_limits": {"total_audio": max_audio_errors, "consecutive_audio": max_consecutive_audio_errors},
                    "protocol_kind": "standard-Vox1-O" if strict_standard else "nonstandard-test-protocol", "model": model_identity, "preprocess": PREPROCESS}
    run_file = output / "run_identity.json"
    if run_file.exists() and json.loads(run_file.read_text()) != run_identity:
        raise EvaluationRefused("existing output identity differs; use a new output directory")
    write_json(run_file, run_identity)
    write_json(output / "audio_index.json", records)
    frozen_thresholds = None
    calibration_identity = None
    if calibration:
        calibration_identity = json.loads(Path(calibration).read_text())
        if calibration_identity.get("model_identity") != model_identity or not calibration_identity.get("o_speaker_disjoint"):
            raise EvaluationRefused("calibration model identity/O disjoint proof differs")
        o_speakers = {t[k] for t in trials for k in ("enroll_speaker", "verify_speaker")}
        if o_speakers.intersection(calibration_identity.get("speakers", [])):
            raise EvaluationRefused("calibration overlaps evaluation speakers")
        frozen_thresholds = calibration_identity["thresholds"]
    vectors, attempted, errors = {}, {}, []
    hits, consecutive, audio_error_count = 0, 0, 0
    abort_reason = None
    crop_records = {key: dict(crop, status="unattempted") for key, crop in planned_crops.items()}
    for key in sorted(planned_crops):
        rel, _ = key
        crop = center_crop(records[rel], planned_crops[key]["duration"])
        attempted[key] = crop
        try:
            vector, hit = extract_one(model, model_identity, records[rel], crop, cache)
            vectors[key] = vector
            hits += int(hit)
            consecutive = 0
            crop_records[key].update(status="ok", cache_hit=hit)
        except Exception as error:
            failure = {"crop": crop, "type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc(), "audio_decode_error": isinstance(error, AudioDecodeError)}
            errors.append(failure)
            crop_records[key]["status"] = "audio_error" if isinstance(error, AudioDecodeError) else "fatal"
            write_json(output / "failures" / f"crop-{time.time_ns()}.json", failure)
            write_json(output / "extraction_errors.json", errors)
            if isinstance(error, AudioDecodeError):
                audio_error_count += 1
                consecutive += 1
                if audio_error_count >= max_audio_errors or consecutive >= max_consecutive_audio_errors:
                    abort_reason = "bounded_audio_error_limit"
            else:
                abort_reason = "model_device_or_identity_failure"
            if abort_reason:
                break
        if len(attempted) % 100 == 0:
            write_json(output / 'progress.json', {"attempted": len(attempted), "total": len(planned_crops),
                       "cache_hits": hits, "errors": len(errors), "complete": False})
    write_json(output / "crops.json", [crop_records[key] for key in sorted(crop_records)])
    results = []
    for matrix in matrices:
        for left in matrix["axis"]:
            for right in matrix["axis"]:
                e = {r: vec for (r, d), vec in vectors.items() if d == str(left)}
                v = {r: vec for (r, d), vec in vectors.items() if d == str(right)}
                rows, scores, labels = score_cell(matrix["trials"], e, v)
                for row in rows:
                    row.update(test_speaker=row['verify_speaker'], enroll_duration=left,
                               test_duration=right, cohort_id=matrix['trialset_sha256'],
                               model_id=model_identity['checkpoint_sha256'])
                directory = output / matrix["name"] / f"{left}x{right}"
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / "scores.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                complete = len(scores) == len(rows)
                result = {"matrix": matrix["name"], "enroll_duration": left, "verify_duration": right,
                          "trialset_sha256": matrix["trialset_sha256"],
                          "comparison_identity_sha256": digest(comparison), "expected_trials": len(rows),
                          "scored_trials": len(scores), "positive": matrix["positive"], "negative": matrix["negative"],
                          "speakers": matrix["speakers"], "complete": complete,
                          "metrics": score_metrics(labels, scores) if complete else None,
                          "fixed_threshold": {name: fixed_threshold_metrics(labels, scores, float(value["threshold"])) for name, value in frozen_thresholds.items()} if complete and frozen_thresholds else {"status": "unavailable", "reason": "no independent frozen calibration supplied"},
                          "calibration_sha256": sha_file(Path(calibration)) if calibration else None,
                          "scores_csv": str((directory / "scores.csv").resolve()),
                          "scores_sha256": sha_file(directory / "scores.csv")}
                write_json(directory / "metrics.json", result)
                results.append(result)
    summary = {"run_identity": run_identity, "cells": results, "extraction_errors": errors,
               "unique_crops": len(planned_crops), "attempted_crops": len(attempted),
               "unattempted_crops": len(planned_crops) - len(attempted), "cache_hits": hits,
               "aborted": abort_reason is not None, "abort_reason": abort_reason,
               "audio_error_count": audio_error_count,
               "complete": not errors and all(r["complete"] for r in results),
               "calibration": calibration_identity, "note": "EER/minDCF thresholds use test scores only to describe discrimination; not deployment calibration."}
    write_json(output / "summary.json", summary)
    return summary


def prepare_calibration(audio_root: Path, o_protocol: Path, output: Path, *, speaker_count=100,
                        clips_per_speaker=5, min_seconds=3.0) -> dict:
    """O身份不重叠的开发校准表；每个身份每个video仅取一条。只读文件头，不提取向量。"""
    trials, o_identity = read_trials(o_protocol)
    excluded = {t[k] for t in trials for k in ("enroll_speaker", "verify_speaker")}
    root, output = Path(audio_root), Path(output)
    protect_output(output)
    if speaker_count < 2 or clips_per_speaker < 2 or not math.isfinite(min_seconds) or min_seconds <= 0:
        raise EvaluationRefused("calibration needs >=2 speakers and clips per speaker")
    selected, exclusions = {}, []
    for speaker in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("id") and p.name not in excluded):
        clips = []
        for video in sorted(p for p in speaker.iterdir() if p.is_dir()):
            for path in sorted(video.glob("*.wav")):
                rel = str(path.relative_to(root))
                try:
                    record = audio_record(root, rel)
                    if record["seconds"] < min_seconds:
                        continue
                except Exception as error:
                    exclusions.append({"relative": rel, "error": str(error)})
                    continue
                clips.append(record)
                break
            if len(clips) == clips_per_speaker:
                break
        if len(clips) == clips_per_speaker:
            selected[speaker.name] = clips
        if len(selected) == speaker_count:
            break
    if len(selected) != speaker_count:
        raise EvaluationRefused(f"only {len(selected)} eligible calibration speakers")
    if excluded.intersection(selected):
        raise EvaluationRefused("calibration O-speaker overlap")
    lines = []
    speakers = list(selected)
    for speaker in speakers:
        clips = selected[speaker]
        for i, left in enumerate(clips):
            for right in clips[i + 1:]:
                lines.append(f"1 {left['relative']} {right['relative']}")
    for index, speaker in enumerate(speakers):
        for other in speakers[index + 1:]:
            for left in selected[speaker]:
                for right in selected[other]:
                    lines.append(f"0 {left['relative']} {right['relative']}")
    output.mkdir(parents=True, exist_ok=True)
    table = output / "trials.txt"
    text = "\n".join(lines) + "\n"
    if table.exists() and table.read_text() != text:
        raise EvaluationRefused("existing calibration trial table differs")
    if not table.exists():
        table.write_text(text)
    info = {"schema": 1, "purpose": "O-only independent development calibration; not training, not E/H independent",
            "audio_root": str(root.resolve()), "o_protocol": o_identity, "o_speakers_excluded": sorted(excluded),
            "speakers": speakers, "speaker_count": len(speakers), "clips_per_speaker": clips_per_speaker,
            "clip_count": sum(map(len, selected.values())), "positive": speaker_count * clips_per_speaker * (clips_per_speaker - 1) // 2,
            "negative": speaker_count * (speaker_count - 1) // 2 * clips_per_speaker**2,
            "trials_sha256": sha_file(table), "selection": "lexicographic speaker/video/wav; first valid >= min seconds; one wav per video",
            "min_seconds": min_seconds, "threshold_condition": "full/full natural waveform, no VAD, no padding",
            "o_speaker_disjoint": True, "eh_independent": False,
            "pretrained_entity_overlap": "unknown; no cross-corpus person-identity mapping verified",
            "audio": selected, "excluded_unreadable": exclusions, "thresholds": None}
    existing = output / "selection.json"
    if existing.exists() and json.loads(existing.read_text()) != info:
        raise EvaluationRefused("existing calibration selection identity differs")
    if not existing.exists():
        write_json(existing, info)
    return info


def calibrate_model(checkpoint: Path, selection: Path, output: Path, *, device="cpu",
                    allow_gpu=False, gpu_memory_mib=None) -> dict:
    """只从O身份独立开发表的自然整段分数定阈值，绝不从O拟阈值。"""
    from duration_ft.metrics import calibrate_far_threshold
    selection, output = Path(selection), Path(output)
    protect_output(output)
    info = json.loads(selection.read_text())
    protocol = selection.parent / "trials.txt"
    if not info.get("o_speaker_disjoint") or info.get("eh_independent") is not False:
        raise EvaluationRefused("selection must declare exact O-only disjoint scope")
    if sha_file(protocol) != info["trials_sha256"]:
        raise EvaluationRefused("calibration trial identity changed")
    trials, _ = read_trials(protocol)
    if {t[k] for t in trials for k in ("enroll_speaker", "verify_speaker")} != set(info["speakers"]):
        raise EvaluationRefused("calibration speaker table differs")
    if set(info["speakers"]).intersection(info["o_speakers_excluded"]):
        raise EvaluationRefused("calibration overlaps O identities")
    records = index_audio(Path(info["audio_root"]), trials)
    expected = {r["relative"]: r for clips in info["audio"].values() for r in clips}
    if {k: audio_content_identity(v) for k, v in records.items()} != {k: audio_content_identity(v) for k, v in expected.items()}:
        raise EvaluationRefused("calibration audio identity changed")
    try:
        model, identity = load_model(checkpoint, device, allow_gpu=allow_gpu, gpu_memory_mib=gpu_memory_mib)
    except Exception as error:
        failure = {"complete": False, "phase": "calibration_model_load", "thresholds": None,
                   "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_json(output / "failures" / f"calibration-load-{time.time_ns()}.json", failure)
        write_json(output / "summary.json", failure)
        raise
    cache = EmbeddingCache(output / "embedding-cache")
    vectors, crops = {}, []
    for rel in sorted(records):
        crop = center_crop(records[rel], "full")
        crops.append(crop)
        # Any calibration failure aborts threshold fitting; no silent pair exclusion.
        try:
            vectors[rel], _ = extract_one(model, identity, records[rel], crop, cache)
        except Exception as error:
            failure = {"complete": False, "phase": "calibration_extraction", "crop": crop,
                       "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                       "thresholds": None, "attempted": len(crops), "completed_crops": len(vectors)}
            write_json(output / "calibration_error.json", failure)
            write_json(output / "failures" / f"calibration-{time.time_ns()}.json", failure)
            write_json(output / "summary.json", failure)
            raise
    rows, scores, labels = score_cell(trials, vectors, vectors)
    if any(row["status"] != "ok" for row in rows):
        raise EvaluationRefused("incomplete calibration")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output / "crops.json", crops)
    result = {"schema": 1, "purpose": info["purpose"], "model_identity": identity,
              "o_speaker_disjoint": True, "eh_independent": False, "speakers": info["speakers"],
              "o_protocol": info["o_protocol"], "selection_sha256": sha_file(selection),
              "trials_sha256": info["trials_sha256"], "condition": "full/full natural waveform, no VAD, no padding",
              "score_sha256": sha_file(output / "scores.csv"),
              "thresholds": {"far_0.01": calibrate_far_threshold(labels, scores, target_far=0.01),
                             "far_0.001": calibrate_far_threshold(labels, scores, target_far=0.001)}}
    destination = output / "thresholds.json"
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise EvaluationRefused("existing frozen calibration differs")
    if not destination.exists():
        write_json(destination, result)
    return result
