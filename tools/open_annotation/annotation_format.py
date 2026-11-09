"""Published annotation formats: explicit generation profiles, never silent mixing.

The four historical label trees (2026-10-25, 10-27, 10-28, 11-09) share the
same sidecar top level (``redimnet2-open-annotation/v2``) and the same commit
sidecar (``redimnet2-sidecar-commit/v1``), but they are *not* one format:

* source clock: ``..._floor_start_ceil_end_v1`` (10-25/10-27) versus
  ``source_offset_decimal_floor_start_ceil_end_v2`` (10-28/11-09);
* runtime provenance: the later profile adds ``asr_batching``/``asr_compute``/
  ``asr_preprocessing``/``normalization_profile``;
* queue completion receipt: ``redimnet2-chunk-commit/v1`` (10-25/10-27) versus
  ``redimnet2-bounded-chunk-commit/v1`` with per-source producer fields
  (10-28/11-09);
* file modes: 0600/0700 under umask 077 versus 0644/0755 under umask 022.

This module centralises those differences.  Callers must ask which profile a
record belongs to and either handle it explicitly or refuse it.  Unknown
schemas/clocks are refused with the supported set in the message; a record or
receipt set that mixes producers is refused instead of being read as one.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath

SIDECAR_SCHEMA = "redimnet2-open-annotation/v2"
SIDECAR_TREE_SCHEMA = "redimnet2-sidecar-tree/v1"
COMMIT_SCHEMA = "redimnet2-sidecar-commit/v1"
DECISION_STATES = ("candidate", "unknown", "failed")

CLOCK_LEGACY = "source_offset_floor_start_ceil_end_v1"
CLOCK_DECIMAL = "source_offset_decimal_floor_start_ceil_end_v2"
PROFILE_LEGACY = "legacy-floor-ceil-v1"
PROFILE_DECIMAL = "decimal-rational-v2"
CLOCK_PROFILES = {CLOCK_LEGACY: PROFILE_LEGACY, CLOCK_DECIMAL: PROFILE_DECIMAL}
PROFILE_CLOCKS = {value: key for key, value in CLOCK_PROFILES.items()}

SIDECAR_TOP_LEVEL = frozenset({
    "contract", "decision", "errors", "execution", "provenance", "quality_axes",
    "schema", "source", "source_identity", "units", "vad", "windows",
})
COMMIT_FIELDS = frozenset({
    "bytes", "contract_sha256", "job_key", "schema", "sidecar_sha256",
    "source_id", "source_record_sha256",
})

RECEIPT_QUEUE = "redimnet2-chunk-commit/v1"
RECEIPT_BOUNDED = "redimnet2-bounded-chunk-commit/v1"
RECEIPT_FAMILIES = {RECEIPT_QUEUE: "queue-v1", RECEIPT_BOUNDED: "bounded-queue-v1"}
RECEIPT_SOURCE_FIELDS = frozenset({
    "source_id", "sidecar_relpath", "sidecar_sha256", "sidecar_bytes", "decision",
    "job_key", "commit_sha256",
})
BOUNDED_SOURCE_FIELDS = frozenset({
    "producer_contract_sha256", "producer_sidecar_tree", "retained_existing",
})
RECEIPT_REQUIRED_FIELDS = frozenset({
    "schema", "shard_id", "index_sha256", "contract_sha256", "shard_sha256",
    "source_count", "sources",
})


class FormatError(ValueError):
    """A reader met a format it cannot interpret without guessing."""


def _require(condition, message):
    if not condition:
        raise FormatError(message)


def _object(value, label):
    _require(isinstance(value, dict), label + " is not a JSON object")
    return value


def canonical_bytes(value):
    """Byte identity used by every published sidecar in all four generations."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sidecar_profile(record):
    """Return the explicit clock profile of one published sidecar record.

    Result: ``{"profile": ..., "clock": ..., "windows": n}``.  Records with no
    model windows (terminal undecodable sources) return ``clock=None`` and
    ``profile="terminal-no-windows"``; they can still be structurally valid but
    carry no clock to compare across generations.
    """
    record = _object(record, "sidecar record")
    schema = record.get("schema")
    _require(schema == SIDECAR_SCHEMA,
             "unsupported sidecar schema %r; supported: %r" % (schema, SIDECAR_SCHEMA))
    missing = sorted(SIDECAR_TOP_LEVEL - set(record))
    _require(not missing, "sidecar is missing top-level fields: " + ", ".join(missing))
    decision = record.get("decision")
    _require(isinstance(decision, dict) and decision.get("state") in DECISION_STATES,
             "sidecar decision.state %r is outside %r" % (
                 decision.get("state") if isinstance(decision, dict) else decision,
                 list(DECISION_STATES)))
    windows = record.get("windows")
    _require(isinstance(windows, list), "sidecar windows is not a list")
    if not windows:
        return {"profile": "terminal-no-windows", "clock": None, "windows": 0}
    clocks = set()
    for index, window in enumerate(windows):
        _require(isinstance(window, dict), "window %d is not an object" % index)
        clock = window.get("time_mapping")
        _require(clock in CLOCK_PROFILES,
                 "window %d time_mapping %r is not a known generation clock; supported: %r"
                 % (index, clock, sorted(CLOCK_PROFILES)))
        clocks.add(clock)
    _require(len(clocks) == 1,
             "sidecar mixes source clocks %r inside one record; refusing to pick one"
             % sorted(clocks))
    clock = clocks.pop()
    return {"profile": CLOCK_PROFILES[clock], "clock": clock, "windows": len(windows)}


def verify_commit(commit, *, sidecar_sha256, sidecar_bytes, source_id, contract_sha256):
    """Validate a ``.commit.json`` against the sidecar bytes and source identity.

    All four generations write the same commit schema; a missing or divergent
    commit is refused rather than treated as an older/newer variant.
    """
    commit = _object(commit, "sidecar commit")
    _require(commit.get("schema") == COMMIT_SCHEMA,
             "unsupported sidecar commit schema %r; supported: %r"
             % (commit.get("schema"), COMMIT_SCHEMA))
    missing = sorted(COMMIT_FIELDS - set(commit))
    extra = sorted(set(commit) - COMMIT_FIELDS)
    _require(not missing, "sidecar commit is missing fields: " + ", ".join(missing))
    _require(not extra, "sidecar commit has unexpected fields: " + ", ".join(extra))
    _require(commit["sidecar_sha256"] == sidecar_sha256,
             "sidecar commit sha256 differs from sidecar bytes")
    _require(commit["bytes"] == sidecar_bytes, "sidecar commit byte count differs")
    _require(commit["source_id"] == source_id, "sidecar commit source_id differs")
    _require(commit["contract_sha256"] == contract_sha256,
             "sidecar commit contract differs from tree identity")
    return commit


def receipt_family(receipt):
    """Return the explicit completion-receipt family; unknown schemas are refused."""
    receipt = _object(receipt, "completion receipt")
    schema = receipt.get("schema")
    _require(schema in RECEIPT_FAMILIES,
             "unsupported completion receipt schema %r; supported: %r"
             % (schema, sorted(RECEIPT_FAMILIES)))
    return RECEIPT_FAMILIES[schema]


def assert_uniform_receipts(receipts):
    """Refuse a directory that silently mixes receipt generations."""
    families = {}
    for receipt in receipts:
        family = receipt_family(receipt)
        families.setdefault(family, 0)
        families[family] += 1
    _require(len(families) <= 1,
             "queue mixes completion receipt generations %r; split by producer instead of reading both as one"
             % families)
    return next(iter(families), None)


def normalize_completion_receipt(receipt, *, producer_contract_sha256, producer_sidecar_tree):
    """Validate one receipt and return a v1-shaped view, or refuse it explicitly.

    A bounded receipt is accepted only when every source was produced by the
    same declared producer (tree and contract).  Retained rows (copied from a
    previous producer) make the receipt a mixed-producer record; this reader
    refuses it because reading them with one ``sidecar_tree`` would silently
    read the wrong files.
    """
    family = receipt_family(receipt)
    missing = sorted(RECEIPT_REQUIRED_FIELDS - set(receipt))
    _require(not missing, "completion receipt is missing fields: " + ", ".join(missing))
    sources = receipt.get("sources")
    _require(isinstance(sources, list), "completion receipt sources is not a list")
    _require(receipt.get("source_count") == len(sources),
             "completion receipt source_count differs from its source list")
    if family == "queue-v1":
        for index, source in enumerate(sources):
            _require(isinstance(source, dict), "receipt source %d is not an object" % index)
            absent = sorted(RECEIPT_SOURCE_FIELDS - set(source))
            _require(not absent, "receipt source %d is missing fields: %s"
                     % (index, ", ".join(absent)))
            _require(source["decision"] in DECISION_STATES,
                     "receipt source %d decision %r is outside %r"
                     % (index, source["decision"], list(DECISION_STATES)))
        return receipt
    # bounded-queue-v1: every row must point at the declared producer.
    for index, source in enumerate(sources):
        _require(isinstance(source, dict), "bounded receipt source %d is not an object" % index)
        absent = sorted((RECEIPT_SOURCE_FIELDS | BOUNDED_SOURCE_FIELDS) - set(source))
        _require(not absent, "bounded receipt source %d is missing producer fields: %s"
                 % (index, ", ".join(absent)))
        _require(source["decision"] in DECISION_STATES,
                 "bounded receipt source %d decision %r is outside %r"
                 % (index, source["decision"], list(DECISION_STATES)))
        _require(source["retained_existing"] is False,
                 "bounded receipt contains a retained source from another producer "
                 "(producer_sidecar_tree=%r); this reader refuses to mix producer trees"
                 % (source["producer_sidecar_tree"],))
        if source["producer_sidecar_tree"] != producer_sidecar_tree:
            # The 11-09 tree was relaid out under /root/group-shared after its
            # receipts were written, so the recorded absolute path is the old
            # location.  The version directory name plus the contract pin the
            # content identity; a different version name is still refused.
            _require(PurePosixPath(source["producer_sidecar_tree"]).name
                     == PurePosixPath(producer_sidecar_tree).name,
                     "bounded receipt source %d sidecar tree %r is not version %r of declared producer %r"
                     % (index, source["producer_sidecar_tree"],
                        PurePosixPath(producer_sidecar_tree).name, producer_sidecar_tree))
        _require(source["producer_contract_sha256"] == producer_contract_sha256,
                 "bounded receipt source %d contract %r differs from declared producer %r"
                 % (index, source["producer_contract_sha256"], producer_contract_sha256))
    return receipt
