"""Register source paths and consumer roles without claiming content deduplication.

No audio/model imports, directory scan, or source modification occurs here.
File/PCM identity and old-VAD checks belong to the actual read-only consumer.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import PurePosixPath
from typing import Any

SCHEMA = "redimnet2-source/v1"
VERSION = "local-catalogue-20261024"
DATA = "/root/group-shared/voiceprint/data/speech/speaker_verification"
OLD_VAD_ROOT = "/root/group-shared/voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/rttm_firered"
FIRERED_MODEL_SHA = "63f4fb1b00a6b8607c118dd48efc18d5e40d67d99b7bf9aa7a8d61540cf23d71"
FIRERED_SIGNATURE = "a56629f741671964f03e3b7ba3f2087d0b3ed76622b9d527e68b501eff325810"

# These aliases describe source path spelling, not exact waveform equivalence.
DATASET_ROOTS = {
    "voxceleb2": [DATA + "/voxceleb/vox2/wav"],
    "voxblink2": [DATA + "/VoxBlink2/audio"],
    "cnceleb2": [DATA + "/cnceleb/CN-Celeb2_flac/data"],
    "voxceleb1": [DATA + "/voxceleb1/wav", DATA + "/voxceleb/vox1/wav"],
    "voices2019": [DATA + "/voices/extracted/VOiCES_Box_unzip/Speaker_Recognition/sid_eval"],
    "cnceleb1": [DATA + "/cnceleb/CN-Celeb_wav"],
    "childmandarin": [DATA + "/BAAI-ChildMandarin41.25H"],
    "kespeech": [DATA + "/KeSpeech/KeSpeech.splits/KeSpeech/Audio"],
}
_DEFAULT_ROOT_ITEMS = sorted(((len(root), dataset, root) for dataset, roots in DATASET_ROOTS.items() for root in roots), reverse=True)


def path_spelling(path: str) -> str:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("source path must be a nonempty string without NUL")
    p = PurePosixPath(path)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("source path must be absolute and contain no parent traversal")
    value = str(p)
    alias = "/shared/group-shared"
    if value == alias or value.startswith(alias + "/"):
        value = "/root/group-shared" + value[len(alias):]
    return value


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid relative source path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or str(p) == ".":
        raise ValueError("relative source path escapes its dataset")
    return str(p)


def source_id(dataset: str, relative: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset):
        raise ValueError("dataset identifier must be a safe single directory name")
    rel = relative_path(relative)
    return "src_" + hashlib.sha256((dataset + "\0" + rel).encode()).hexdigest()


def classify_path(path: str, roots: dict[str, list[str]] | None = None) -> tuple[str, str]:
    path = path_spelling(path)
    items = _DEFAULT_ROOT_ITEMS if roots is None else sorted(((len(root), dataset, path_spelling(root)) for dataset, rr in roots.items() for root in rr), reverse=True)
    for _, dataset, root in items:
        if path.startswith(root + "/"):
            return dataset, relative_path(path[len(root) + 1:])
    raise ValueError("source does not belong to a declared dataset root: " + path)


def sidecar_relative_path(source: dict[str, Any]) -> str:
    source_id(source["dataset"], source["relative_path"])
    return source["dataset"] + "/" + relative_path(source["relative_path"]) + ".annotation.json"


def legacy_vad_reference(path: str) -> dict[str, Any]:
    """Actual FireRed root_depth=2 mapping, checked against real existing files.

    Old metadata has source path/stat/clock, not a source-byte SHA. Do not turn
    a newly calculated audio SHA into a claim about the old model's input.
    """
    path = path_spelling(path)
    p = PurePosixPath(path)
    old = PurePosixPath(OLD_VAD_ROOT).joinpath(*p.parts[-3:])
    old = old.with_name(old.name + ".rttm")
    if not path.startswith(DATA + "/"):
        raise ValueError("old VAD source outside declared audio root")
    return {
        "path": str(old), "metadata_path": str(old.parent / ("." + old.name + ".json")),
        "format": "firered-rttm-v2", "status": "unverified",
        "expected_source_relative": path[len(DATA) + 1:],
        "expected_model_sha256": FIRERED_MODEL_SHA,
        "expected_inference_signature": FIRERED_SIGNATURE,
        "identity_binding": "legacy_source_path_size_mtime_clock",
    }


def normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    r = dict(record)
    dataset = r.get("dataset")
    rel = relative_path(r["relative_path"])
    expected_id = source_id(dataset, rel)
    if r.get("source_id", expected_id) != expected_id:
        raise ValueError("source_id does not match dataset and relative path")
    roles = sorted(set(r.get("roles", [])))
    if not roles or not all(isinstance(x, str) and x for x in roles):
        raise ValueError("source needs nonempty consumer roles")
    path = r.get("audio_path")
    if path is not None:
        path = path_spelling(path)
    else:
        candidates = r.get("resolver_candidates", [])
        if not candidates and not r.get("input_error"):
            raise ValueError("unresolved source must retain resolver candidates")
        r["resolver_candidates"] = list(dict.fromkeys(path_spelling(p) for p in candidates))
    protection = r.get("protection", {})
    if not isinstance(protection, dict):
        raise ValueError("protection must be an object")
    reasons = set(protection.get("reasons", []))
    if any(x.startswith("regression") for x in roles):
        reasons.add("regression_source_not_for_training")
    evaluation = (any(x.startswith(("evaluation:", "dev", "calibration", "test", "regression")) for x in roles)
        or any(x in {"test", "dev", "calibration"} for x in r.get("original_splits", [])))
    training = any(x.startswith("train") for x in roles)
    if evaluation:
        reasons.add("heldout_or_diagnostic_consumer")
    conflicts = list(r.get("split_conflicts", []))
    if evaluation and training:
        conflicts.append("same_source_has_training_and_heldout_consumers")
        reasons.add("source_role_conflict")
    r.update(
        schema=SCHEMA, source_id=expected_id, dataset=dataset,
        version=r.get("version", VERSION), relative_path=rel, audio_path=path,
        roles=roles, original_splits=sorted(set(r.get("original_splits", []))),
        parent_id=r.get("parent_id", "parent_" + hashlib.sha256((dataset + "\0" + str(PurePosixPath(rel).parent)).encode()).hexdigest()),
        parent_evidence=r.get("parent_evidence", "provider_path_group_not_verified_session"),
        speaker_evidence=r.get("speaker_evidence", {"provider_label": None, "verified_identity": False}),
        protection={"training_forbidden": bool(protection.get("training_forbidden", False) or reasons), "reasons": sorted(reasons)},
        channel_policy=r.get("channel_policy", "mean_channels_to_mono"),
        source_interval=r.get("source_interval"),
        resolution_status=r.get("resolution_status", "path_registered_media_unverified" if path else "unresolved"),
        split_conflicts=sorted(set(conflicts)),
        identity_status="path_registered_no_content_dedup",
    )
    r["sidecar_relpath"] = sidecar_relative_path(r)
    return r


def merge_records(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a, b = normalise_record(a), normalise_record(b)
    if a["source_id"] != b["source_id"] or a["version"] != b["version"]:
        raise ValueError("cannot merge different source identities/versions")
    out = dict(a)
    paths = sorted(set(a.get("path_aliases", []) + b.get("path_aliases", []) + [p for p in (a["audio_path"], b["audio_path"]) if p]))
    out["path_aliases"] = paths
    if not out["audio_path"]:
        out["audio_path"] = b["audio_path"]
        out["resolution_status"] = b["resolution_status"]
    out["roles"] = sorted(set(a["roles"] + b["roles"]))
    out["original_splits"] = sorted(set(a["original_splits"] + b["original_splits"]))
    out["protection"] = {
        "training_forbidden": a["protection"]["training_forbidden"] or b["protection"]["training_forbidden"],
        "reasons": sorted(set(a["protection"]["reasons"] + b["protection"]["reasons"])),
    }
    labels = sorted(set(x for x in [a["speaker_evidence"].get("provider_label"), b["speaker_evidence"].get("provider_label"), *a["speaker_evidence"].get("provider_labels", []), *b["speaker_evidence"].get("provider_labels", [])] if x))
    if len(labels) > 1:
        out["speaker_evidence"] = {"provider_label": None, "provider_labels": labels, "verified_identity": False}
        out["split_conflicts"] = out.get("split_conflicts", []) + ["provider_speaker_label_conflict"]
        out["protection"]["training_forbidden"] = True
        out["protection"]["reasons"].append("provider_speaker_label_conflict")
    if not out.get("old_vad") and b.get("old_vad"):
        out["old_vad"] = b["old_vad"]
    out["consumer_keys"] = sorted(set(a.get("consumer_keys", []) + b.get("consumer_keys", [])))
    return normalise_record(out)


def from_training_row(row: dict[str, Any], role: str) -> dict[str, Any]:
    path = path_spelling(row["wav"])
    dataset, rel = classify_path(path)
    if dataset not in {"voxceleb2", "voxblink2", "cnceleb2"}:
        raise ValueError("training row outside the declared three-corpus input")
    return normalise_record({
        "dataset": dataset, "relative_path": rel, "audio_path": path,
        "roles": [role], "original_splits": ["provider_train"],
        "speaker_evidence": {"provider_label": row.get("spk"), "verified_identity": False},
        "consumer_keys": [row["key"]] if row.get("key") else [],
        "old_vad": legacy_vad_reference(path),
    })
