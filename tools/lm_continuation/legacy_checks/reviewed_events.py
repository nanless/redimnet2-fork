"""Pure byte-level classification of seven individually reviewed PTN events.

The caller must independently pin the ledger's file SHA and provenance. This
module grants no training readiness, performs no I/O, and never edits logs.
Offsets delimit [Traceback start, first resumed Epoch start), in raw bytes.
Only CRLF -> LF and CR -> LF are used for the secondary normalized digest.
"""
import ast
import hashlib
import re


class ReviewRefused(RuntimeError):
    """Evidence differs from the exact reviewed occurrence ledger."""


_TRACE = b"Traceback (most recent call last):"
_DROP = re.compile(
    rb"\[dataset\]\[DROP:read_failed\][ \t]+LibsndfileError: "
    rb"Internal psf_fseek\(\) failed\.[ \t]+\|[ \t]*(\{[^\r\n]+\})")
_PROGRESS = re.compile(rb"Epoch:[ \t]*(\d+):[^\r\n]*?\b(\d+)/15001\b")
_FATAL = re.compile(
    rb"CUDA out of memory|Watchdog caught|NCCL (?:watchdog|error|timeout)|"
    rb"could NOT restore (?:optimizer|GradScaler)|"
    rb"(?:RuntimeError|FloatingPointError|OutOfMemoryError|ChildFailedError|"
    rb"MemoryError|OSError|SystemError):|\b(?:FATAL|SIGSEGV|SIGABRT)\b|"
    rb"TRAIN_END status=(?:failed|error)|"
    rb"\b(?:nan|[+-]?inf(?:inity)?|non[- ]?finite)\b", re.IGNORECASE)
_OTHER_EXCEPTION = re.compile(
    rb"\b(?:[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)|Error|Exception):", re.IGNORECASE)


def _need(ok, message):
    if not ok:
        raise ReviewRefused(message)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _integer(value):
    return type(value) is int


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _validate_ledger(ledger, length):
    _need(isinstance(ledger, dict) and type(ledger.get("schema")) is int
          and ledger["schema"] == 1, "Unsupported reviewed-event schema")
    size = ledger.get("stdout_prefix_bytes")
    _need(_integer(size) and 0 < size <= length, "Missing/truncated reviewed prefix")
    _need(_hash(ledger.get("stdout_prefix_sha256")), "Invalid prefix SHA256")
    events = ledger.get("events")
    _need(isinstance(events, list) and len(events) == 7,
          "Exactly seven individually reviewed events required")
    ids, identities, previous_end, previous_progress = set(), set(), -1, (0, -1)
    for event in events:
        _need(isinstance(event, dict), "Invalid reviewed event")
        name = event.get("event_id")
        _need(isinstance(name, str) and name and name not in ids, "Duplicate/invalid event_id")
        start, end = event.get("raw_start"), event.get("raw_end")
        _need(_integer(start) and _integer(end)
              and previous_end < start < end < size, "Invalid/overlapping event offsets")
        _need(_hash(event.get("raw_sha256")) and _hash(event.get("normalized_sha256")),
              "Invalid event digest")
        sample = event.get("sample")
        _need(isinstance(sample, dict) and set(sample) == {"key", "spk", "wav"}
              and all(isinstance(v, str) and v for v in sample.values()),
              "Complete key/spk/wav dictionary required")
        epoch, before, resumed = (event.get(k) for k in ("epoch", "before_batch", "resumed_batch"))
        _need(all(_integer(v) for v in (epoch, before, resumed))
              and 1 <= epoch <= 120 and 0 <= before < 15001 and resumed == before + 1,
              "Invalid exact event progress")
        identity = (epoch, before, resumed, sample["key"])
        _need(identity not in identities and (epoch, before) > previous_progress,
              "Duplicate/out-of-order occurrence identity")
        ids.add(name); identities.add(identity)
        previous_end, previous_progress = end, (epoch, before)
    return size, events


def classify_reviewed_log(raw_bytes, ledger_dict):
    """Return seven occurrence records or raise ReviewRefused; never readiness.

    Required ledger keys: schema=1, stdout_prefix_bytes, stdout_prefix_sha256,
    events (exactly seven, ordered). Each event requires event_id, raw_start,
    raw_end, raw_sha256, normalized_sha256, sample={key,spk,wav}, epoch,
    before_batch, resumed_batch. Extra audit metadata is allowed. A sample or
    block digest may repeat; an occurrence identity or event_id may not.
    """
    _need(isinstance(raw_bytes, bytes), "Raw log must be bytes")
    prefix_size, events = _validate_ledger(ledger_dict, len(raw_bytes))
    _need(_digest(raw_bytes[:prefix_size]) == ledger_dict["stdout_prefix_sha256"],
          "Reviewed stdout prefix bytes changed")
    _need(_FATAL.search(raw_bytes) is None, "Fatal/nonfinite marker in full log")
    starts = list(re.finditer(re.escape(_TRACE), raw_bytes))
    # Count generic words too: malformed extra markers must not disappear.
    _need(len(starts) == 7 and len(re.findall(rb"\bTraceback\b", raw_bytes, re.I)) == 7,
          "Unknown/missing/additional Traceback")
    _need(len(re.findall(rb"read_failed", raw_bytes, re.I)) == 7,
          "Unknown/missing/additional read_failed")
    _need([m.start() for m in starts] == [e["raw_start"] for e in events],
          "Traceback occurrence offsets/order differ")
    records = []
    for index, event in enumerate(events):
        start = event["raw_start"]
        limit = starts[index + 1].start() if index + 1 < len(starts) else len(raw_bytes)
        drops = list(_DROP.finditer(raw_bytes, start, limit))
        _need(len(drops) == 1, "Event has no single exact caught-read marker")
        drop = drops[0]
        preceding = list(_PROGRESS.finditer(raw_bytes, 0, start))
        _need(bool(preceding), "Event has no preceding progress")
        prior = preceding[-1]
        _need(tuple(map(int, prior.groups())) == (event["epoch"], event["before_batch"]),
              "Preceding epoch/batch differs")
        # Never search past an incorrect first progress to a later good one.
        resumed = _PROGRESS.search(raw_bytes, drop.end(), limit)
        _need(resumed is not None, "Event has no resumed progress")
        _need(tuple(map(int, resumed.groups())) == (event["epoch"], event["resumed_batch"]),
              "First resumed epoch/batch differs from exact +1")
        _need(resumed.start() == event["raw_end"] and resumed.end() <= prefix_size,
              "Raw end/resumed progress is outside reviewed occurrence")
        block = raw_bytes[start:resumed.start()]
        _need(block[:drop.start() - start].count(
            b"soundfile.LibsndfileError: Internal psf_fseek() failed.") == 1,
            "Unexpected exception identity")
        try:
            sample = ast.literal_eval(drop.group(1).decode("utf-8", errors="strict"))
        except (UnicodeError, SyntaxError, ValueError, TypeError, RecursionError) as exc:
            raise ReviewRefused("Malformed complete sample dictionary") from exc
        _need(sample == event["sample"], "Complete sample dictionary differs")
        normalized = block.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        raw_sha, normal_sha = _digest(block), _digest(normalized)
        _need(raw_sha == event["raw_sha256"] and normal_sha == event["normalized_sha256"],
              "Raw/normalized event bytes differ")
        records.append({"event_id": event["event_id"], "epoch": event["epoch"],
                        "before_batch": event["before_batch"], "resumed_batch": event["resumed_batch"],
                        "sample": dict(sample), "raw_start": start, "raw_end": resumed.start(),
                        "raw_bytes": len(block), "raw_sha256": raw_sha,
                        "normalized_bytes": len(normalized), "normalized_sha256": normal_sha})
    # An additional exception-only line cannot borrow the known exception class.
    cursor = 0
    for event in events:
        _need(_OTHER_EXCEPTION.search(raw_bytes, cursor, event["raw_start"]) is None,
              "Unreviewed exception outside exact event blocks")
        cursor = event["raw_end"]
    _need(_OTHER_EXCEPTION.search(raw_bytes, cursor) is None,
          "Unreviewed exception outside exact event blocks")
    return records
