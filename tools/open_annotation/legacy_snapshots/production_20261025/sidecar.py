"""Versioned mirror-tree sidecars, semantic readback and no-replace commits."""
from __future__ import annotations
import json
import os
from pathlib import Path
from contracts import Refused, atomic_new, canonical_bytes, digest, job_key, need, safe_join, sha_file, stat_identity, utcnow
from schema import assert_valid_record


class SidecarStore:
    def __init__(self, output_root, contract, contract_sha):
        self.contract, self.contract_sha = contract, contract_sha
        self.root = Path(output_root).resolve() / contract["version"]
        self.root.mkdir(parents=True, exist_ok=True)
        identity = {"schema": "redimnet2-sidecar-tree/v1", "version": contract["version"], "contract_sha256": contract_sha}
        p = self.root / "identity.json"
        if p.exists(): need(json.loads(p.read_text()) == identity, "output version already belongs to another contract")
        else: atomic_new(p, identity)

    def path(self, source):
        expected = source["dataset"] + "/" + source["relative_path"] + ".annotation.json"
        need(source["sidecar_relpath"] == expected, "sidecar mirror mapping differs")
        return safe_join(self.root, expected)

    def _receipt(self, source, path, raw, commit):
        return {"source_id": source["source_id"], "job_key": job_key(source, self.contract_sha),
                "sidecar_relpath": str(path.relative_to(self.root)),
                "sidecar_sha256": digest_bytes(raw), "sidecar_bytes": len(raw),
                "commit_sha256": sha_file(commit), "decision": json.loads(raw)["decision"]["state"]}

    def lookup(self, source, check_current_source=True):
        path = self.path(source); commit = path.with_name(path.name + ".commit.json")
        if not path.exists():
            need(not commit.exists(), "orphan commit has no sidecar")
            return None
        need(not path.is_symlink() and not commit.is_symlink(), "published sidecar/commit symlink refused")
        raw = path.read_bytes(); record = json.loads(raw); assert_valid_record(record)
        need(record["source"] == source and record["contract"]["sha256"] == self.contract_sha,
             "existing sidecar belongs to another source version/contract")
        need(record["execution"]["job_key"] == job_key(source, self.contract_sha), "existing sidecar job key differs")
        identity = record["source_identity"]
        if check_current_source and identity["decode_status"] == "decoded_complete":
            need(source.get("audio_path") and stat_identity(source["audio_path"]) == identity["stat_after"], "source changed since sidecar publication")
        descriptor = {"schema": "redimnet2-sidecar-commit/v1", "source_id": source["source_id"],
                      "job_key": job_key(source, self.contract_sha), "contract_sha256": self.contract_sha,
                      "source_record_sha256": digest(source), "sidecar_sha256": digest_bytes(raw),
                      "bytes": len(raw)}
        if commit.exists(): need(json.loads(commit.read_text()) == descriptor, "sidecar commit digest/identity differs")
        else:
            # Crash after sidecar publication but before commit: semantic readback
            # above is mandatory before the missing marker can be repaired.
            atomic_new(commit, descriptor)
        return self._receipt(source, path, raw, commit)

    def publish(self, source, record, claim, fault=None):
        claim.assert_current(); claim.assert_member(source); assert_valid_record(record)
        need(record["source"] == source and record["execution"]["job_key"] == job_key(source, self.contract_sha), "publication source/job differs")
        need(record["contract"]["sha256"] == self.contract_sha, "publication contract differs")
        path = self.path(source); raw = canonical_bytes(record)
        if fault: fault("before_sidecar")
        claim.assert_current(); atomic_new(path, raw)
        if fault: fault("after_sidecar_before_commit")
        need(path.read_bytes() == raw, "sidecar readback differs")
        claim.assert_current()
        result = self.lookup(source)
        if fault: fault("after_commit")
        need(result is not None, "sidecar commit missing after publish")
        return result


def digest_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


class StageCache:
    """Content-addressed raw stage results. Identity is checked on every hit."""
    def __init__(self, root): self.root = Path(root)
    def path(self, stage, key):
        need(stage in {"asr", "aligner"} and len(key) == 64 and all(c in "0123456789abcdef" for c in key), "unsafe stage cache key")
        return self.root / stage / key[:2] / key[2:4] / (key + ".json")
    def get(self, stage, inputs):
        key = digest(inputs); path = self.path(stage, key)
        if not path.exists(): return None
        need(not path.is_symlink(), "stage cache symlink refused")
        value = json.loads(path.read_text())
        need(value.get("schema") == "redimnet2-stage-cache/v1" and value["stage"] == stage
             and value["inputs"] == inputs and value["key"] == key and digest(value["output"]) == value["output_sha256"], "stage cache corrupt or wrong contract")
        return value["output"]
    def put(self, stage, inputs, output):
        key = digest(inputs); path = self.path(stage, key)
        value = {"schema": "redimnet2-stage-cache/v1", "stage": stage, "key": key,
                 "inputs": inputs, "output": output, "output_sha256": digest(output)}
        if path.exists():
            # An already verified winner is authoritative, never overwritten by
            # another attempt's floating-point or batch-dependent output.
            self.get(stage, inputs)
        else:
            try: atomic_new(path, value)
            except (FileExistsError, Refused):
                # Only a verified existing winner permits recovery. E.g. a
                # missing/corrupt file or another inputs contract still fails.
                need(path.is_file(), "cache publication failed without a winner")
                self.get(stage, inputs)
        return self.get(stage, inputs)
