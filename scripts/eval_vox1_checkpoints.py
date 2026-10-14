#!/usr/bin/env python3
"""Evaluate ReDimNet2 checkpoints on VoxCeleb1 cleaned protocols (O/E/H).

Scoring matches wespeaker_lite's *fast* validation path, eval_epoch_simple:
- extract L2-normalized embeddings for unique trial utterances
- score with cosine similarity
- report EER (%)

It does NOT match wespeaker_lite's full eval_epoch (5 crops x 3 s with TTA and
multi-crop averaging). That function is gated behind the `full_val` config key,
which no recipe in this repo sets, so it never runs during training.

Audio duration also differs from the training-time validation: this script reads
each utterance in full (soundfile read, variable length, batch_size=1), whereas
the fast path reads only the first 8 s (eval_dur=8.0, start_time=0.0) and wraps
the tail around when the clip is shorter. Hence these numbers, not val_log.csv's
`-eer-8s` column (formerly `-eer-fl`), are the ones comparable to the official
table.

Usage (from repo root, with redimnet2 env):
  python scripts/eval_vox1_checkpoints.py \
    --vox1-root /path/to/voxceleb1 \
    --ckpt-dir checkpoints/pretrained \
    --protocols O
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from redimnet2.redimnet2 import ReDimNet2Wrap  # noqa: E402


PROTOCOL_FILES = {
    "O": "veri_test2.txt",
    "E": "list_test_all2.txt",
    "H": "list_test_hard2.txt",
}

# Official README numbers (VoxCeleb2-dev trained, cleaned protocols)
OFFICIAL_VOX2 = {
    "b0-vox2-lm.pt": {"O": 1.04, "E": 1.16, "H": 1.97},
    "b1-vox2-lm.pt": {"O": 0.78, "E": 0.96, "H": 1.72},
    "b2-vox2-lm.pt": {"O": 0.57, "E": 0.76, "H": 1.41},
    "b3-vox2-lm.pt": {"O": 0.42, "E": 0.66, "H": 1.22},
    "b4-vox2-lm.pt": {"O": 0.37, "E": 0.58, "H": 1.07},
    "b5-vox2-lm.pt": {"O": 0.33, "E": 0.56, "H": 1.07},
    "b6-vox2-lm.pt": {"O": 0.29, "E": 0.52, "H": 0.99},
}

OFFICIAL_MIX = {
    "b6-vb2+vox2_v0-lm.pt": {"O": 0.23, "E": 0.35, "H": 0.67},
    "b6-vb2+vox2_v0-ptn.pt": {"O": 0.29, "E": 0.50, "H": 0.94},
    "b3-vb2+vox2+cnc2_v0-lm.pt": {"O": 0.51, "E": 0.60, "H": 1.14},
    "b3-vb2+vox2+cnc2_v0-ptn.pt": {"O": 0.69, "E": 0.82, "H": 1.47},
    "b6-vb2+vox2+cnc2_v0-lm.pt": {"O": 0.26, "E": 0.41, "H": 0.79},
    "b6-vb2+vox2+cnc2_v0-ptn.pt": {"O": 0.42, "E": 0.58, "H": 1.07},
}
OFFICIAL = {**OFFICIAL_VOX2, **OFFICIAL_MIX}


def eer_percent(y_true, y_score) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(np.around(eer * 100, 3))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


class TrialWavDataset(Dataset):
    def __init__(self, root: Path, rel_paths: list[str], wavs_subroot: str = "wav"):
        self.root = root
        self.wavs_subroot = wavs_subroot
        self.rel_paths = rel_paths
        missing = [
            p for p in rel_paths if not (root / wavs_subroot / p).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} wavs missing, e.g. {missing[:3]}"
            )

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        path = self.root / self.wavs_subroot / rel
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != 16000:
            raise ValueError(f"{path} sr={sr}, expected 16000")
        return torch.from_numpy(wav), rel


def load_protocol(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=" ", names=["label", "enroll", "verify"])


def resolve_protocol(vox1_root: Path, key: str) -> Path:
    name = PROTOCOL_FILES[key]
    candidates = [
        vox1_root / name,
        vox1_root / "protocols" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"protocol {name} not under {vox1_root}")


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataset: TrialWavDataset,
    device: torch.device,
    batch_size: int = 1,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: batch,
    )
    model.eval()
    out: dict[str, np.ndarray] = {}
    for batch in tqdm(loader, desc="embed", leave=False):
        # batch_size=1 for variable-length full utterances (official-style fl)
        for wav, rel in batch:
            x = wav.unsqueeze(0).to(device)  # (1, T)
            emb = model(x)
            emb = F.normalize(emb, p=2, dim=1)
            out[rel] = emb.squeeze(0).float().cpu().numpy()
    return out


def load_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_config" in ckpt and "state_dict" in ckpt:
        model = ReDimNet2Wrap(**ckpt["model_config"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    model.to(device)
    model.eval()
    return model


def eval_one(
    model: torch.nn.Module,
    vox1_root: Path,
    protocol_path: Path,
    device: torch.device,
    wavs_subroot: str,
) -> float:
    proto = load_protocol(protocol_path)
    utts = sorted(set(proto["enroll"]).union(set(proto["verify"])))
    ds = TrialWavDataset(vox1_root, utts, wavs_subroot=wavs_subroot)
    emb = extract_embeddings(model, ds, device)
    scores = [
        cosine(emb[r.enroll], emb[r.verify]) for r in proto.itertuples(index=False)
    ]
    return eer_percent(proto["label"].values, np.asarray(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vox1-root", type=Path, required=True)
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=REPO / "checkpoints" / "pretrained",
    )
    ap.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Specific .pt filenames; default = all *.pt in ckpt-dir",
    )
    ap.add_argument(
        "--protocols",
        default="O",
        help="Comma list among O,E,H (default O)",
    )
    ap.add_argument("--wavs-subroot", default="wav")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "checkpoints" / "eval_vox1_results.csv",
    )
    ap.add_argument("--only-lm", action="store_true", help="Only evaluate *-lm.pt")
    args = ap.parse_args()

    proto_keys = [p.strip().upper() for p in args.protocols.split(",") if p.strip()]
    for k in proto_keys:
        if k not in PROTOCOL_FILES:
            raise SystemExit(f"Unknown protocol {k}")

    if args.checkpoints:
        ckpts = [args.ckpt_dir / n for n in args.checkpoints]
    else:
        ckpts = sorted(args.ckpt_dir.glob("*.pt"))
        if args.only_lm:
            ckpts = [p for p in ckpts if p.name.endswith("-lm.pt")]

    if not ckpts:
        raise SystemExit(f"No checkpoints in {args.ckpt_dir}")

    device = torch.device(args.device)
    rows = []
    for ckpt_path in ckpts:
        print(f"\n=== {ckpt_path.name} ===")
        model = load_checkpoint(ckpt_path, device)
        row = {"checkpoint": ckpt_path.name}
        for key in proto_keys:
            ppath = resolve_protocol(args.vox1_root, key)
            eer = eval_one(model, args.vox1_root, ppath, device, args.wavs_subroot)
            row[f"Vox1-{key}"] = eer
            official = OFFICIAL.get(ckpt_path.name, {}).get(key)
            if official is not None:
                row[f"official-{key}"] = official
                row[f"delta-{key}"] = round(eer - official, 3)
            print(
                f"  Vox1-{key}: {eer}"
                + (f"  (official {official}, delta {eer - official:+.3f})" if official is not None else "")
            )
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out_csv}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
