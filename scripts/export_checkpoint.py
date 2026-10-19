#!/usr/bin/env python3
"""Export a training checkpoint to the released embedding-only format."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.checkpoint_utils import embedding_payload  # noqa: E402


def export_checkpoint(checkpoint: Path, training_config: Path | None, output: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing file: {output}")
    payload, _ = embedding_payload(checkpoint, training_config)
    if training_config is not None and "export_info" in payload:
        def sha256(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        payload["export_info"]["source_checkpoint_sha256"] = sha256(checkpoint)
        payload["export_info"]["source_training_config_sha256"] = sha256(training_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        # Atomic creation that also refuses an output created by another process.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return payload.get("export_info", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path,
                        help="Actual saved experiment config.yaml; required for training checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = export_checkpoint(args.checkpoint, args.training_config, args.output)
    print(f"Wrote {args.output}; source epoch={info.get('source_epoch')}; "
          f"discarded classifier tensors={len(info.get('discarded_projection_keys', []))}")


if __name__ == "__main__":
    main()
