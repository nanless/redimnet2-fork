"""Single-node durable broker cursor plus stable, non-expiring chunk locks.

No SQLite WAL on the shared filesystem. The broker JSON is O(active workers),
not O(corpus size); completed chunks have immutable receipts. Lock expiry never
authorizes takeover. A released stable flock and a new generation fence do.
"""
from __future__ import annotations
import contextlib
import fcntl
import json
import os
from pathlib import Path
import secrets
import struct
from contracts import atomic_new, canonical_bytes, digest, job_key, need, relative_path, sha_file, utcnow

SLOT_BYTES = 1024
SLOT_MAGIC = b"RDMAP001"


def encode_completion_slot(value):
    import hashlib
    data = canonical_bytes(value)
    need(len(data) <= SLOT_BYTES - 44, "completion receipt exceeds fixed slot")
    return SLOT_MAGIC + struct.pack(">I", len(data)) + hashlib.sha256(data).digest() + data + bytes(SLOT_BYTES - 44 - len(data))


def decode_completion_slot(data):
    import hashlib
    need(len(data) == SLOT_BYTES and data[:8] == SLOT_MAGIC, "missing/torn completion slot")
    size = struct.unpack(">I", data[8:12])[0]
    need(0 < size <= SLOT_BYTES - 44, "bad completion slot size")
    raw = data[44:44 + size]
    need(hashlib.sha256(raw).digest() == data[12:44], "completion slot checksum differs")
    return json.loads(raw)


def _replace(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + secrets.token_hex(12) + ".tmp")
    with tmp.open("xb") as f:
        f.write(canonical_bytes(value)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(fd)
    finally: os.close(fd)


@contextlib.contextmanager
def stable_lock(path, nonblocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.is_symlink(), "lock symlink refused")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        before = os.fstat(stream.fileno()); after = path.stat()
        need((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), "stable lock inode differs")
        yield stream


class ProductionRunLock:
    """Parent must retain this for the complete 2/8 worker lifetime."""
    def __init__(self, queue_root): self.path = Path(queue_root) / ".parent.lock"
    def __enter__(self):
        self.context = stable_lock(self.path, nonblocking=True)
        return self.context.__enter__()
    def __exit__(self, *args): return self.context.__exit__(*args)


def read_index(path, expected_sha=None):
    path = Path(path).resolve()
    if expected_sha: need(sha_file(path) == expected_sha, "shard index SHA changed")
    value = json.loads(path.read_text())
    need(value.get("schema") == "redimnet2-shard-index/v1", "wrong immutable shard index")
    shards = value.get("shards")
    need(isinstance(shards, list) and shards, "no shards")
    ids = set()
    for shard in shards:
        need(isinstance(shard["id"], str) and shard["id"] and "/" not in shard["id"] and shard["id"] not in ids, "duplicate/unsafe shard ID")
        ids.add(shard["id"]); relative_path(shard["path"])
        need(type(shard["count"]) is int and shard["count"] > 0 and len(shard["sha256"]) == 64, "invalid shard pin")
    need(sum(s["count"] for s in shards) == value["source_count"], "index denominator differs")
    return value


class WorkQueue:
    def __init__(self, index_path, index_sha, root, contract_sha):
        self.index_path = Path(index_path).resolve(); self.index_sha = index_sha
        self.index = read_index(self.index_path, index_sha)
        self.shards = self.index["shards"]; self.by_id = {s["id"]: s for s in self.shards}
        self.ordinal = {s["id"]: i for i, s in enumerate(self.shards)}
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.contract_sha = contract_sha
        self.identity = {"schema": "redimnet2-work-queue/v1", "index_sha256": index_sha,
                         "contract_sha256": contract_sha, "source_count": self.index["source_count"],
                         "shard_count": len(self.shards)}
        with stable_lock(self.root / "broker.lock"):
            p = self.root / "identity.json"
            if p.exists(): need(json.loads(p.read_text()) == self.identity, "queue belongs to another input/contract")
            else: atomic_new(p, self.identity)
            if not (self.root / "broker.json").exists():
                _replace(self.root / "broker.json", {"next": 0, "leases": {}, "completed_chunks": 0})
            completion = self.root / "completion_map.bin"
            if not completion.exists():
                with completion.open("xb") as f:
                    f.truncate(len(self.shards) * SLOT_BYTES); f.flush(); os.fsync(f.fileno())
            need(completion.stat().st_size == len(self.shards) * SLOT_BYTES, "completion map shape differs")

    def chunk_receipt_path(self, shard_id): return self.root / "completed" / (shard_id + ".json")

    def receipt(self, shard_id):
        path = self.chunk_receipt_path(shard_id)
        if not path.exists(): return None
        value = json.loads(path.read_text()); shard = self.by_id[shard_id]
        need(value["schema"] == "redimnet2-chunk-commit/v1" and value["shard_id"] == shard_id,
             "bad chunk commit identity")
        need(value["index_sha256"] == self.index_sha and value["contract_sha256"] == self.contract_sha
             and value["shard_sha256"] == shard["sha256"] and value["source_count"] == shard["count"],
             "chunk commit belongs to another input/contract")
        need(len(value["sources"]) == shard["count"] and len({s["source_id"] for s in value["sources"]}) == shard["count"], "bad committed denominator")
        expected = {s["source_id"]: s for s in self.read_shard(shard)}
        need(set(expected) == {s["source_id"] for s in value["sources"]}, "chunk receipt has foreign or missing source IDs")
        need(all(s["job_key"] == job_key(expected[s["source_id"]], self.contract_sha) for s in value["sources"]), "chunk receipt job/input differs")
        return value

    def read_shard(self, shard):
        path = self.index_path.parent / shard["path"]
        need(sha_file(path) == shard["sha256"], "immutable shard changed")
        result = []
        with path.open() as f:
            for line in f:
                need(bool(line.strip()), "blank source row")
                result.append(json.loads(line))
        need(len(result) == shard["count"] and len({r["source_id"] for r in result}) == len(result), "shard denominator/uniqueness differs")
        return result

    def write_completion(self, shard_id, receipt):
        value = {"shard_id": shard_id, "shard_sha256": self.by_id[shard_id]["sha256"],
                 "source_count": receipt["source_count"], "source_ids_sha256": digest(sorted(s["source_id"] for s in receipt["sources"])),
                 "commit_sha256": sha_file(self.chunk_receipt_path(shard_id)), "index_sha256": self.index_sha,
                 "contract_sha256": self.contract_sha, "decision_counts": {key: sum(s["decision"] == key for s in receipt["sources"]) for key in ("candidate", "unknown", "failed")}}
        data = encode_completion_slot(value); fd = os.open(str(self.root / "completion_map.bin"), os.O_RDWR)
        try:
            start = self.ordinal[shard_id] * SLOT_BYTES; written = 0
            while written < len(data):
                n = os.pwrite(fd, data[written:], start + written); need(n > 0, "short completion slot write"); written += n
            os.fsync(fd)
            need(os.pread(fd, SLOT_BYTES, start) == data, "completion slot readback differs")
        finally: os.close(fd)
        return value

    def claim_next(self, worker_id):
        with stable_lock(self.root / "broker.lock"):
            state = json.loads((self.root / "broker.json").read_text())
            candidates = []
            if state["next"] < len(self.shards):
                candidates.append(self.shards[state["next"]]["id"])
            else:
                # At most the number of active workers. No repeated corpus scan.
                candidates.extend(sorted(state["leases"]))
            for shard_id in candidates:
                lock_path = self.root / "claims" / (shard_id + ".lock")
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                need(not lock_path.is_symlink(), "claim lock symlink refused")
                handle = lock_path.open("a+b")
                try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: handle.close(); continue
                previous_receipt = self.receipt(shard_id)
                if previous_receipt is not None:
                    self.write_completion(shard_id, previous_receipt)
                    if shard_id in state["leases"]:
                        state["leases"].pop(shard_id); state["completed_chunks"] += 1
                    if state["next"] < len(self.shards) and self.shards[state["next"]]["id"] == shard_id:
                        state["next"] += 1
                    handle.close(); _replace(self.root / "broker.json", state)
                    continue
                token_path = self.root / "attempt_state" / (shard_id + ".json")
                generation = json.loads(token_path.read_text())["generation"] + 1 if token_path.exists() else 1
                token = {"generation": generation, "nonce": secrets.token_hex(24), "worker_id": str(worker_id),
                         "pid": os.getpid(), "claimed_utc": utcnow()}
                _replace(token_path, token)
                if state["next"] < len(self.shards) and self.shards[state["next"]]["id"] == shard_id: state["next"] += 1
                state["leases"][shard_id] = token
                _replace(self.root / "broker.json", state)
                return Claim(self, self.by_id[shard_id], handle, token)
            # If a committed stale lease was retired, broker state was saved above.
            return None

    def is_complete(self):
        with stable_lock(self.root / "broker.lock"):
            value = json.loads((self.root / "broker.json").read_text())
            return value["completed_chunks"] == len(self.shards) and not value["leases"]


class Claim:
    def __init__(self, queue, shard, handle, token):
        self.queue, self.shard, self.handle, self.token = queue, shard, handle, token
        self.attempt_id = "%08d-%s" % (token["generation"], token["nonce"])
        self.attempt_root = queue.root / "attempts" / shard["id"] / self.attempt_id
        self.attempt_root.mkdir(parents=True, exist_ok=False)
        atomic_new(self.attempt_root / "claim.json", token)
        self.closed = False

    def assert_current(self):
        need(not self.closed and not self.handle.closed, "claim already released")
        st = os.fstat(self.handle.fileno()); current = (self.queue.root / "claims" / (self.shard["id"] + ".lock")).stat()
        need((st.st_dev, st.st_ino) == (current.st_dev, current.st_ino), "claim inode replaced")
        token = json.loads((self.queue.root / "attempt_state" / (self.shard["id"] + ".json")).read_text())
        need(token == self.token, "stale attempt fence refused publication")

    def sources(self):
        self.assert_current()
        if not hasattr(self, "_sources"):
            self._sources = self.queue.read_shard(self.shard)
            self._source_map = {s["source_id"]: s for s in self._sources}
        return self._sources

    def assert_member(self, source):
        self.sources()
        need(self._source_map.get(source["source_id"]) == source, "foreign or changed source is not a member of this pinned shard")

    def journal(self, value):
        self.assert_current()
        with (self.attempt_root / "journal.jsonl").open("ab") as f:
            f.write(canonical_bytes(value)); f.flush(); os.fsync(f.fileno())

    def finish(self, receipts):
        self.assert_current()
        need(len(receipts) == self.shard["count"] and len({r["source_id"] for r in receipts}) == len(receipts), "cannot commit incomplete chunk")
        sources = {s["source_id"]: s for s in self.sources()}
        need(set(sources) == {r["source_id"] for r in receipts}, "cannot commit foreign sources as this chunk")
        need(all(r["job_key"] == job_key(sources[r["source_id"]], self.queue.contract_sha) for r in receipts), "chunk receipt job/input differs")
        value = {"schema": "redimnet2-chunk-commit/v1", "shard_id": self.shard["id"],
                 "index_sha256": self.queue.index_sha, "contract_sha256": self.queue.contract_sha,
                 "shard_sha256": self.shard["sha256"], "source_count": self.shard["count"],
                 "attempt_id": self.attempt_id, "sources": receipts, "committed_utc": utcnow()}
        atomic_new(self.queue.chunk_receipt_path(self.shard["id"]), value)
        self.assert_current(); self.queue.write_completion(self.shard["id"], value)
        with stable_lock(self.queue.root / "broker.lock"):
            self.assert_current()
            state = json.loads((self.queue.root / "broker.json").read_text())
            need(state["leases"].get(self.shard["id"]) == self.token, "broker fence differs")
            state["leases"].pop(self.shard["id"]); state["completed_chunks"] += 1
            _replace(self.queue.root / "broker.json", state)
        return value

    def close(self):
        if not self.closed:
            self.closed = True; self.handle.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def prepare_queue(index_path, index_sha, queue_root, contract_sha, output_root=None):
    """Idempotent initialization; caller retains .parent.lock throughout run."""
    queue = WorkQueue(index_path, index_sha, queue_root, contract_sha)
    return queue.identity
