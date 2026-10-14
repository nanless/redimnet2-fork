#!/usr/bin/env python3
"""Evaluate ReDimNet2 checkpoints on OOD sets: VOICES SID, CN-Celeb multi-enroll, Vox1-B.

Reuse cosine-EER protocol consistent with scripts/eval_vox1_checkpoints.py.
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

OFFICIAL_MIX = {
    "b6-vb2+vox2_v0-lm.pt": {"Vox1-B": 0.89, "SITW": 0.74, "VOICES": 2.55, "CN-Celeb": 7.49},
    "b6-vb2+vox2_v0-ptn.pt": {"Vox1-B": 1.27, "SITW": 1.01, "VOICES": 3.25, "CN-Celeb": 8.50},
    "b3-vb2+vox2+cnc2_v0-lm.pt": {"Vox1-B": 1.72, "SITW": 1.25, "VOICES": 3.87, "CN-Celeb": 4.78},
    "b3-vb2+vox2+cnc2_v0-ptn.pt": {"Vox1-B": 2.16, "SITW": 1.91, "VOICES": 5.96, "CN-Celeb": 5.29},
    "b6-vb2+vox2+cnc2_v0-lm.pt": {"Vox1-B": 1.15, "SITW": 0.87, "VOICES": 3.03, "CN-Celeb": 3.59},
    "b6-vb2+vox2+cnc2_v0-ptn.pt": {"Vox1-B": 1.51, "SITW": 1.15, "VOICES": 4.30, "CN-Celeb": 4.32},
}


def eer_percent(y_true, y_score) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(np.around(eer * 100, 3))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


class AbsWavDataset(Dataset):
    def __init__(self, paths: list[Path], keys: list[str]):
        self.paths = paths
        self.keys = keys
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} wavs missing, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != 16000:
            # allow resample-free fail; VOICES/CN-Celeb expected 16k
            raise ValueError(f"{path} sr={sr}, expected 16000")
        return torch.from_numpy(wav), self.keys[idx]


@torch.no_grad()
def extract_embeddings(model, dataset, device) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda b: b)
    model.eval()
    out = {}
    for batch in tqdm(loader, desc="embed", leave=False):
        for wav, key in batch:
            emb = F.normalize(model(wav.unsqueeze(0).to(device)), p=2, dim=1)
            out[key] = emb.squeeze(0).float().cpu().numpy()
    return out


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ReDimNet2Wrap(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def score_trials(emb, trials: list[tuple[int, str, str]]) -> float:
    """Cosine scores for L2-normalized embeddings; chunked for large trial lists."""
    n = len(trials)
    y = np.empty(n, dtype=np.int32)
    s = np.empty(n, dtype=np.float64)
    chunk = 250_000
    for i in range(0, n, chunk):
        part = trials[i : i + chunk]
        y[i : i + len(part)] = [t[0] for t in part]
        a = np.stack([emb[t[1]] for t in part])
        b = np.stack([emb[t[2]] for t in part])
        s[i : i + len(part)] = (a * b).sum(axis=1)
    return eer_percent(y, s)


def load_vox1b(vox1_root: Path):
    # Prefer full KAIST bilingual list (808574 trials) over the smaller copy.
    candidates = [
        vox1_root / "protocols" / "list_test_bilingual.txt",
        vox1_root / "protocols" / "voxceleb1-b" / "list_test_bilingual.txt",
        vox1_root / "list_test_bilingual.txt",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(f"Vox1-B protocol not under {vox1_root}")
    df = pd.read_csv(p, sep=" ", names=["label", "enroll", "verify"])
    utts = sorted(set(df["enroll"]).union(set(df["verify"])))
    root = vox1_root
    paths = [root / "wav" / u for u in utts]
    trials = [(int(r.label), r.enroll, r.verify) for r in df.itertuples(index=False)]
    return AbsWavDataset(paths, utts), trials


def load_voices(voices_root: Path):
    base = voices_root / "extracted" / "VOiCES_Box_unzip" / "Speaker_Recognition"
    audio = base / "sid_eval"
    lists = base / "sid_eval_lists"
    enroll_map = {}
    for line in (lists / "eval-enroll.lst").read_text().splitlines():
        if not line.strip():
            continue
        spk, rel = line.split()
        enroll_map[spk] = Path(rel).name
    post = (
        voices_root
        / "extracted"
        / "VOiCES_Box_unzip"
        / "VOiCES_challenge_2019_post-eval-release"
        / "VOiCES_challenge_2019_eval.SID.trial-keys.lst"
    )
    if not post.exists():
        raise FileNotFoundError(f"VOICES trial keys missing: {post}")
    labeled = []
    keys = set()
    for line in post.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        a, b, lab = parts[0], parts[1], parts[2]
        if lab in {"target", "1", "tgt"}:
            y = 1
        elif lab in {"nontarget", "0", "imp", "nontgt"}:
            y = 0
        else:
            try:
                y = int(float(lab))
            except ValueError:
                continue
        en = enroll_map.get(a, f"{a}.wav" if not a.endswith(".wav") else a)
        # enroll files named VOiCES_..._0001.wav
        if en not in enroll_map.values() and a in enroll_map:
            en = enroll_map[a]
        elif a in enroll_map:
            en = enroll_map[a]
        else:
            en = f"{a}.wav"
        tv = Path(b).name
        labeled.append((y, en, tv))
        keys.add(en)
        keys.add(tv)
    key_list = sorted(keys)
    paths = [audio / k for k in key_list]
    return AbsWavDataset(paths, key_list), labeled


def _resolve_cnceleb_wav(wav_root: Path, u: str) -> Path:
    for cand in (
        wav_root / u,
        wav_root / "data" / u,
        wav_root / "eval" / u,
        wav_root / "eval" / "enroll" / Path(u).name,
        wav_root / "eval" / "test" / Path(u).name,
    ):
        if cand.exists():
            return cand
    return wav_root / u


def load_cnceleb(cn_root: Path, mode: str = "avg"):
    """CN-Celeb enrollment trials.

    mode=avg (default, matches README multi-enrollment):
      average L2 embeddings of segments listed in enroll.map, then score
      official trials.lst (enroll_id test_rel label).

    mode=concat:
      score with single concatenated eval/enroll/*-enroll.wav per trial
      (CNC-Eval-Concat.lst or path-style Avg lists).
    """
    wav_root = cn_root / "CN-Celeb_wav"
    flac_lists = cn_root / "CN-Celeb_flac" / "eval" / "lists"
    mode = mode.lower().strip()

    if mode == "avg":
        emap = flac_lists / "enroll.map"
        if not emap.exists():
            emap = wav_root / "eval" / "lists" / "enroll.map"
        trl = flac_lists / "trials.lst"
        if not trl.exists():
            trl = wav_root / "eval" / "lists" / "trials.lst"
        if not emap.exists() or not trl.exists():
            raise FileNotFoundError(
                f"CN-Celeb Avg needs enroll.map + trials.lst under {cn_root}"
            )

        avg_groups: dict[str, list[str]] = {}
        seg_keys: set[str] = set()
        for line in emap.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            ek, segs = parts[0], parts[1:]
            avg_groups[ek] = segs
            seg_keys.update(segs)

        trials: list[tuple[int, str, str]] = []
        test_keys: set[str] = set()
        for line in trl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            # official: idXXXX-enroll test/....wav 0|1
            e, t, lab = parts[0], parts[1], parts[2]
            if lab in {"target", "nontarget"}:
                y = 1 if lab == "target" else 0
            else:
                y = int(lab)
            # normalize test key to basename under eval/test for embedding lookup
            t_key = f"eval/test/{Path(t).name}" if not t.startswith("eval/") else t
            if t.startswith("test/"):
                t_key = f"eval/{t}"
            trials.append((y, e, t_key))
            test_keys.add(t_key)

        key_list = sorted(seg_keys | test_keys)
        paths = []
        for k in key_list:
            if k in seg_keys:
                paths.append(wav_root / "data" / k)
            else:
                paths.append(_resolve_cnceleb_wav(wav_root, k))
        return AbsWavDataset(paths, key_list), trials, avg_groups

    if mode != "concat":
        raise ValueError(f"Unknown cnceleb mode: {mode}")

    candidates = [
        wav_root / "eval" / "lists" / "CNC-Eval-Concat.lst",
        wav_root / "eval" / "lists" / "CNC-Eval-Avg.lst",  # local path-style (concat audio)
        wav_root / "eval" / "lists" / "trials.lst",
        flac_lists / "trials.lst",
    ]
    lists = next((p for p in candidates if p.exists()), None)
    if lists is None:
        raise FileNotFoundError(f"No CN-Celeb trials under {cn_root}")

    trials = []
    keys: set[str] = set()
    for line in lists.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[0] in {"0", "1"}:
            y, e, t = int(parts[0]), parts[1], parts[2]
        elif parts[2] in {"target", "nontarget", "tgt", "imp"}:
            e, t, lab = parts[0], parts[1], parts[2]
            y = 1 if lab in {"target", "tgt", "1"} else 0
        else:
            e, t, y = parts[0], parts[1], int(parts[2])
        trials.append((y, e, t))
        keys.add(e)
        keys.add(t)

    key_list = sorted(keys)
    return (
        AbsWavDataset([_resolve_cnceleb_wav(wav_root, k) for k in key_list], key_list),
        trials,
        {},
    )


def average_enroll_embeddings(
    emb: dict[str, np.ndarray], avg_groups: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """Mean + re-L2-normalize multi-utterance enroll embeddings (wespeaker Avg)."""
    if not avg_groups:
        return emb
    out = dict(emb)
    for ek, segs in avg_groups.items():
        vecs = np.stack([emb[s] for s in segs], axis=0)
        m = vecs.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        out[ek] = m.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints" / "pretrained")
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--benchmarks", default="VOICES,CN-Celeb,Vox1-B")
    ap.add_argument("--vox1-root", type=Path, required=False)
    ap.add_argument("--voices-root", type=Path, required=False)
    ap.add_argument("--cnceleb-root", type=Path, required=False)
    ap.add_argument(
        "--cnceleb-mode",
        default="avg",
        choices=["avg", "concat"],
        help="CN-Celeb enrollment: avg=enroll.map multi-utt mean; concat=*-enroll.wav",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 默认写到仓库的 checkpoints/ 下，且用一个**不会覆盖**已发布汇总表的文件名
    # （已发布的是 eval_ood_all.csv）。想写到别处就显式传 --out-csv。
    ap.add_argument("--out-csv", type=Path,
                    default=REPO / "checkpoints" / "eval_ood_results.csv")
    ap.add_argument("--only-mix", action="store_true", help="Only mix-table checkpoints")
    args = ap.parse_args()

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if args.checkpoints:
        ckpts = [args.ckpt_dir / n for n in args.checkpoints]
    else:
        ckpts = sorted(args.ckpt_dir.glob("*.pt"))
        if args.only_mix:
            ckpts = [p for p in ckpts if p.name in OFFICIAL_MIX]

    device = torch.device(args.device)
    cache = {}
    rows = []
    for ckpt in ckpts:
        print(f"\n=== {ckpt.name} ===")
        model = load_checkpoint(ckpt, device)
        row = {"checkpoint": ckpt.name}
        for b in benches:
            if b not in cache:
                if b == "Vox1-B":
                    cache[b] = (*load_vox1b(args.vox1_root), {})
                elif b == "VOICES":
                    cache[b] = (*load_voices(args.voices_root), {})
                elif b == "CN-Celeb":
                    cache[b] = load_cnceleb(args.cnceleb_root, mode=args.cnceleb_mode)
                else:
                    raise SystemExit(f"Unknown benchmark {b}")
            ds, trials, avg_groups = cache[b]
            emb = extract_embeddings(model, ds, device)
            emb = average_enroll_embeddings(emb, avg_groups)
            eer = score_trials(emb, trials)
            row[b] = eer
            off = OFFICIAL_MIX.get(ckpt.name, {}).get(b)
            if off is not None:
                row[f"official-{b}"] = off
                row[f"delta-{b}"] = round(eer - off, 3)
            print(f"  {b}: {eer}" + (f" (official {off}, delta {eer-off:+.3f})" if off is not None else ""))
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", args.out_csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
