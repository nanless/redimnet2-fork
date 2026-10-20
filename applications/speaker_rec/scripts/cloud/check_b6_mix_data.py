#!/usr/bin/env python3
"""逐行核对三语料训练表，并抽查音频；不会修改数据或扫描全部训练音频。"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPECTED = {
    "mix_join": {"A": 1092009, "B": 9904382, "C": 524787},
    "mix_join_lm": {"A": 754757, "B": 3831606, "C": 434647},
}

def normalized(path):
    return os.path.normpath(str(path).replace("/root/group-shared/", "/shared/group-shared/", 1))

def audio_check(path, min_seconds=0):
    import soundfile as sf
    import numpy as np
    path = Path(path)
    resolved = path.resolve(strict=True)
    if normalized(path).startswith("/shared/group-shared/") and not str(resolved).startswith("/shared/group-shared/"):
        raise ValueError(f"音频软链接跳出预定共享盘，需要增加挂载: {path} -> {resolved}")
    info = sf.info(path)
    # 抽中的文件完整解码，避免只读取文件头却遗漏损坏的后半部分。
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000 or x.shape[1] != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError(f"音频不符合 16 kHz、单声道、有限数值要求: {path}")
    if len(x) / sr + 1e-4 < min_seconds:
        raise ValueError(f"音频短于 {min_seconds} 秒: {path}")
    return {"path": str(path), "resolved": str(path.resolve(strict=True)),
            "frames": info.frames, "sample_rate": sr, "channels": info.channels,
            "seconds": info.duration, "bytes": path.stat().st_size}

def audit_stage(root, name, samples, eval_paths):
    expected = EXPECTED[name]
    counts = collections.Counter()
    roots = collections.Counter()
    speakers = set()
    selected = []
    digests = [hashlib.sha256(), hashlib.sha256()]
    base = root / "vox2_vb2_cnc2_v0" / name
    speech = normalized(root.parent / "speech" / "speaker_verification")
    allowed = {"A": speech + "/voxceleb/vox2/wav/",
               "B": speech + "/VoxBlink2/audio/",
               "C": speech + "/cnceleb/CN-Celeb2_flac/data/"}
    overlap = 0
    count = 0
    print(f"开始逐行核对 {base}", flush=True)
    with (base / "data.list").open("rb") as data, (base / "utt2spk").open("rb") as labels:
        for count, pair in enumerate(itertools.zip_longest(data, labels), 1):
            dl, ul = pair
            if dl is None or ul is None:
                raise ValueError(f"{name} 的 data.list 与 utt2spk 行数不同")
            digests[0].update(dl)
            digests[1].update(ul)
            obj = json.loads(dl)
            fields = ul.decode().split()
            if len(fields) != 2 or fields != [obj["key"], obj["spk"]]:
                raise ValueError(f"{name} 第 {count} 行样本编号或说话人不一致")
            source = obj["spk"].split("_", 1)[0]
            wav = normalized(obj["wav"])
            if source not in allowed or not wav.startswith(allowed[source]):
                raise ValueError(f"{name} 第 {count} 行来源或音频目录不符合三语料约定: {obj}")
            if not obj["key"].startswith(obj["spk"] + "/"):
                raise ValueError(f"{name} 第 {count} 行样本编号与说话人前缀不一致")
            counts[source] += 1
            roots[obj["wav"].split("/voiceprint/", 1)[0]] += 1
            speakers.add(obj["spk"])
            overlap += wav in eval_paths
            stride = max(1, expected[source] // samples)
            if counts[source] == 1 or (counts[source] % stride == 0 and counts[source] // stride < samples):
                selected.append({"line": count, "source": source, "key": obj["key"],
                                 **audio_check(obj["wav"], 5.0 if name.endswith("_lm") else 0)})
            if count % 2000000 == 0:
                print(f"{name}: 已核对 {count} 行", flush=True)
    if dict(counts) != expected or overlap:
        raise ValueError(f"{name} 数量不符或训练音频出现在评测列表: {dict(counts)}, overlap={overlap}")
    result = {"directory": str(base), "rows": count, "source_counts": dict(counts),
              "speakers": len(speakers), "path_prefix_counts": dict(roots),
              "sha256": dict(zip(["data.list", "utt2spk"], (d.hexdigest() for d in digests))),
              "sampled_audio": selected, "exact_eval_path_overlap": overlap,
              "limits": "训练表全部逐行检查；仅所列抽样音频完整解码。未逐条 stat 全部训练音频，也未做人名身份或音频内容去重。"}
    print(f"{name}: {count} 行，{len(speakers)} 个说话人，{len(selected)} 条音频抽查通过", flush=True)
    return result, speakers

def audit_eval(speech, root, samples):
    # 直接使用正式评测程序的数据读取函数，避免另写一套路径解释规则。
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / "scripts"))
    import eval_ood_checkpoints as ood
    import eval_vox1_checkpoints as vox
    results = {}
    path_set = set()
    for name in ("O", "E", "H"):
        p = vox.resolve_protocol(speech / "voxceleb1", name)
        df = vox.load_protocol(p)
        keys = sorted(set(df["enroll"]) | set(df["verify"]))
        paths = [speech / "voxceleb1" / "wav" / k for k in keys]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Vox1-{name}: {len(missing)} 条音频缺失，例如 {missing[:3]}")
        path_set.update(normalized(p) for p in paths)
        selected = [paths[i] for i in range(0, len(paths), max(1, len(paths) // samples))][:samples]
        results[f"Vox1-{name}"] = {"protocol": str(p), "trials": len(df), "unique_audio": len(keys),
                                  "missing": len(missing), "sampled_audio": [audio_check(p) for p in selected]}
        print(f"Vox1-{name}: {len(df)} 对试验，{len(keys)} 条音频，全部路径存在", flush=True)
    for name, load in (("Vox1-B", lambda: ood.load_vox1b(speech / "voxceleb1")),
                       ("VOICES", lambda: ood.load_voices(speech / "voices")),
                       ("CN-Celeb", lambda: ood.load_cnceleb(speech / "cnceleb", mode="avg"))):
        loaded = load()
        dataset, trials = loaded[:2]
        path_set.update(normalized(p) for p in dataset.paths)
        selected = dataset.paths[::max(1, len(dataset.paths) // samples)][:samples]
        results[name] = {"trials": len(trials), "unique_audio": len(dataset.paths), "missing": 0,
                         "sampled_audio": [audio_check(p) for p in selected]}
        if name == "CN-Celeb":
            results[name]["enrollment_mode"] = "avg"
            results[name]["enrollment_speakers"] = len(loaded[2])
        print(f"{name}: {len(trials)} 对试验，{len(dataset.paths)} 条音频，全部路径存在", flush=True)
    # 训练过程使用单独的小目录，文件本身可能为软链接；逐条核对其解析后的去向。
    p = root / "vox1-test" / "veri_test2.txt"
    df = vox.load_protocol(p)
    keys = sorted(set(df["enroll"]) | set(df["verify"]))
    paths = [root / "vox1-test" / "wav" / k for k in keys]
    resolved = [p.resolve(strict=True) for p in paths]
    path_set.update(normalized(p) for p in resolved)
    results["training_validation"] = {"root": str(root / "vox1-test"), "trials": len(df),
                                      "unique_audio": len(paths), "missing": 0,
                                      "resolved_roots": sorted({str(p).split("/wav/")[0] for p in resolved})}
    return results, path_set

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("/root/group-shared/voiceprint/data/wespeaker_data"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples-per-source", type=int, default=12)
    ap.add_argument("--only-paths", action="store_true", help="仅验证增强和评测路径，不扫描两阶段完整训练表")
    args = ap.parse_args()
    if args.samples_per_source < 1:
        ap.error("--samples-per-source 必须大于零")
    start = time.time()
    root = args.data_root
    report = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hostname": os.uname().nodename, "data_root": str(root),
              "resolved_data_root": str(root.resolve(strict=True)), "ok": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        speech = root.parent / "speech" / "speaker_verification"
        report["evaluation"], eval_paths = audit_eval(speech, root, args.samples_per_source)
        report["augmentation"] = {}
        for name, expected in (("musan/music", 660), ("musan/noise", 930), ("musan/speech", 426), ("rirs/simulated_rirs", 60000)):
            directory = root / "raw" / name
            paths = sorted(directory.rglob("*.wav"))
            if len(paths) != expected:
                raise ValueError(f"{directory}: {len(paths)} 个 wav，应为 {expected}")
            # 混响可能很短，但应为单声道 16 kHz；完整解码少量均匀抽样。
            selected = paths[::max(1, len(paths) // args.samples_per_source)][:args.samples_per_source]
            report["augmentation"][name] = {"directory": str(directory), "resolved": str(directory.resolve(strict=True)),
                                            "wav_files": len(paths), "sampled_audio": [audio_check(p) for p in selected]}
        if not args.only_paths:
            report["ptn"], ptn_speakers = audit_stage(root, "mix_join", args.samples_per_source, eval_paths)
            report["lm"], lm_speakers = audit_stage(root, "mix_join_lm", args.samples_per_source, eval_paths)
            if not lm_speakers.issubset(ptn_speakers):
                raise ValueError("微调阶段存在主训练阶段没有的说话人")
            report["lm_speakers_subset_of_ptn"] = True
            report["ptn_speakers_absent_from_lm"] = len(ptn_speakers - lm_speakers)
        report["mount"] = subprocess.check_output(["findmnt", "-T", str(root), "-J"], text=True)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = round(time.time() - start, 3)
        temp = args.output.with_name(args.output.name + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temp, args.output)
        print(f"检查结果已写入 {args.output}，ok={report['ok']}", flush=True)

if __name__ == "__main__":
    main()
