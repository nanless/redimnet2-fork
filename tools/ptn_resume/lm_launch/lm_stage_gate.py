#!/usr/bin/env python3
"""LM stage gate: verify the frozen package, then delegate to the frozen trainer.

This file deliberately contains **no training logic**. It checks the package and
the contract, hands the run to the frozen source tree's ordinary STAGE=lm entry
(`applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh`), and then
reads back the cold-start evidence. That keeps the amount of new code on the
training path small enough to review.

What it enforces before any training starts:
  * `bundle_files.json` and `resume-contract.json` match their external SHAs
  * the contract's `runtime_admission` is false and `LM_started` is false
  * `model_115.pt` exists and its sha256 is the pinned value
  * the new run root does not exist yet (unique new output)
  * the frozen `resume.yaml` still declares the LM geometry this package froze

What it checks afterwards:
  * `models/model_1.pt` exists (a real saved epoch, not just a log)
  * the log contains the cold-start marker and **no** optimizer/GradScaler restore
  * writes `lm_receipt.json` next to the run (never overwrites)

CUDA note: it never imports torch itself; the trainer does. `--verify-only`
performs the CPU checks and exits without touching the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat as statmod
import subprocess
import sys

PIN_MODEL_115_SHA256 = "65497b6a00efac412a82d563aebc1e767056fef23e7f4234f48f89cc8b3aad5c"
NUMBER = 13075
BATCH = 48
WORLD = 8
MARGIN = 0.3
COLD_START_MARKER = (
    "cold start: model_ckpt loads model parameters only; optimizer/scaler start fresh"
)
FORBIDDEN_LOG_LINES = (
    "resume: restored optimizer state from",
    "resume: restored GradScaler state from",
    "could NOT restore optimizer",
    "could NOT restore GradScaler",
)


class Refused(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise Refused(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(path):
    before = os.stat(path)
    value = sha256_file(path)
    after = os.stat(path)
    key = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    need(key(before) == key(after), f"文件在读取期间变化: {path}")
    need(statmod.S_ISREG(after.st_mode) and not os.path.islink(path), f"证据必须是实体文件: {path}")
    return value, after.st_size


def load_json(path):
    with open(path, "rb") as stream:
        return json.loads(stream.read().decode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--package-sha", required=True)
    parser.add_argument("--verify-only", action="store_true", help="只做 CPU 校验，不启动训练")
    # Compatibility only: the PTN entry.py takes --execute, so entry commands
    # copied from PTN tend to append it. Executing is already the default here;
    # accepting the flag must not change any judgement.
    parser.add_argument("--execute", action="store_true",
                        help="兼容旧入口：仅表示执行（本工具默认即执行），不影响任何判据")
    args = parser.parse_args(argv)

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        contract = verify(args)
        print("LM_GATE_VERIFIED " + json.dumps({
            "checkpoint_sha256": contract["checkpoint"]["sha256"],
            "new_run_root": contract["new_run_root"],
            "clock": contract["clock"]["epoch_iter"],
            "spk2id": contract.get("_verified_speaker_map"),
        }, ensure_ascii=False), flush=True)
        if args.verify_only:
            return 0
        return run(args, contract)
    except Refused as error:
        print("LM_GATE_REFUSED " + json.dumps({"error": str(error)}, ensure_ascii=False), flush=True)
        return 2
    except OSError as error:
        # A missing/unreadable evidence file is a refusal, not a crash.
        print("LM_GATE_REFUSED " + json.dumps({"error": f"OSError: {error}"}, ensure_ascii=False), flush=True)
        return 2


def verify(args):
    root = args.package_root
    need(os.path.isdir(root), f"包目录不存在: {root}")

    bundle_path = os.path.join(root, "bundle_files.json")
    need(stable_sha256(bundle_path)[0] == args.package_sha, "bundle_files.json 与外部 package sha 不符")
    need(stable_sha256(args.contract)[0] == args.contract_sha, "resume-contract.json 与外部 contract sha 不符")

    contract = load_json(args.contract)
    need(contract.get("runtime_admission") is False, "合同的 runtime_admission 不是 false")
    need(contract.get("LM_started") is False, "合同的 LM_started 不是 false")

    checkpoint = contract.get("checkpoint") or {}
    path = checkpoint.get("path")
    need(path and os.path.isfile(path), f"model_115.pt 不存在: {path}")
    need(checkpoint.get("sha256") == PIN_MODEL_115_SHA256, "合同 checkpoint sha 不是 pin 值")
    need(os.path.basename(path) == "model_115.pt", "LM 必须从第 115 轮初始化")

    clock = contract.get("clock") or {}
    need(clock.get("stage") == "lm", "clock.stage 不是 lm")
    need(clock.get("start_epoch") == 1 and clock.get("target_epoch") == 1, "LM 只跑 1 轮")
    need(clock.get("epoch_iter") == NUMBER, f"clock.epoch_iter 不是 {NUMBER}")
    need(clock.get("batch_size_per_gpu") == BATCH, "每卡 batch 不是 48")
    need(clock.get("margin") == MARGIN, "margin 不是 0.3")
    need(clock.get("optimizer") == "new" and clock.get("scaler") == "new", "LM 必须冷启动")

    run_root = contract.get("new_run_root")
    need(run_root and not os.path.exists(run_root), f"new_run_root 已存在，拒绝覆盖: {run_root}")

    config = os.path.join(root, contract.get("resume_config_path", "resume.yaml"))
    need(os.path.isfile(config), f"resume 配置缺失: {config}")
    need(stable_sha256(config)[0] == contract.get("resume_config_sha256"), "resume 配置摘要与合同不符")
    verify_recipe(config, contract)

    speaker = verify_speaker_map(contract)

    for name, pinned in (contract.get("runtime_files") or {}).items():
        target = os.path.join(root, "runtime", name)
        need(os.path.isfile(target), f"runtime 文件缺失: {name}")
        need(stable_sha256(target)[0] == pinned, f"runtime 文件摘要不符: {name}")

    # Most expensive check last: 650 MB. Everything cheap has already passed.
    actual, size = stable_sha256(path)
    need(actual == PIN_MODEL_115_SHA256, f"model_115.pt 摘要不符: {actual}")
    need(size == checkpoint.get("bytes"), "model_115.pt 字节数与合同不符")

    contract = dict(contract)
    contract["_verified_speaker_map"] = speaker
    return contract


def verify_speaker_map(contract):
    """Re-hash spk2id.json and re-count the LM head — the fallback guard.

    If `spk2id_map_path` is missing or unreadable, train.py silently falls back
    to slicing the first N rows of the PTN head (357,822 → 115,362). That shape
    happens to be legal, so nothing else would complain: the run would train
    against the wrong centroids. This check makes the map an execution gate:
    exact bytes, exact PTN speaker count, exact LM speaker count, full coverage.
    """
    speaker = contract.get("speaker_map") or {}
    path = speaker.get("path")
    need(path and os.path.isfile(path), f"spk2id.json 不存在（会退回按索引切片）: {path}")
    actual, size = stable_sha256(path)
    need(actual == speaker.get("sha256"), f"spk2id.json 摘要不符: {actual}")
    need(size > 0, "spk2id.json 为空")
    with open(path, "r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    need(isinstance(mapping, dict) and mapping, "spk2id.json 不是非空对象")
    need(all(type(value) is int for value in mapping.values()), "spk2id 编号不是整数")
    ptn_speakers = speaker.get("ptn_speakers")
    need(ptn_speakers is not None, "合同缺少 speaker_map.ptn_speakers")
    need(len(mapping) == ptn_speakers, f"spk2id 人数 {len(mapping)} != 合同 {ptn_speakers}")

    label = (contract.get("lm_label") or {}).get("path")
    need(label and os.path.isfile(label), f"LM 标签不存在: {label}")
    speakers = set()
    rows = 0
    with open(label, "r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            need(len(parts) == 2, f"LM 标签第 {rows + 1} 行不是两列")
            speakers.add(parts[1])
            rows += 1
    classes = speaker.get("classes")
    need(classes is not None, "合同缺少 speaker_map.classes")
    need(len(speakers) == classes, f"LM 说话人数 {len(speakers)} != 合同 {classes}")
    missing = sorted(speakers - set(mapping))
    need(not missing, f"LM 说话人不在 spk2id 中: {len(missing)} 人，前 5 个 {missing[:5]}")
    return {
        "sha256": actual,
        "ptn_speakers": len(mapping),
        "lm_speakers": len(speakers),
        "lm_utterances": rows,
        "coverage": "all_lm_speakers_present",
    }


def verify_recipe(config, contract):
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - the training env has PyYAML
        raise Refused(f"无法核对配方，缺少 PyYAML: {error}") from error
    with open(config, "r", encoding="utf-8") as stream:
        recipe = yaml.safe_load(stream)
    need(isinstance(recipe, dict), "resume 配置顶层不是映射")
    loader = recipe.get("dataloader_args") or {}
    dataset = recipe.get("dataset_args") or {}
    margin = recipe.get("margin_update") or {}
    sched = recipe.get("scheduler_args") or {}
    need(recipe.get("num_epochs") == 1, "配方轮数不是 1")
    need(loader.get("batch_size") == BATCH, "配方每卡 batch 不是 48")
    need(dataset.get("num_frms") == 500, "配方 num_frms 不是 500")
    need(dataset.get("speed_perturb") is False, "配方 speed_perturb 不是 false")
    need((dataset.get("aug_setup") or {}).get("prob") == 0.7, "配方增强根概率不是 0.7")
    need(recipe.get("do_lm") is True, "配方 do_lm 不是 true")
    need(margin.get("initial_margin") == MARGIN and margin.get("final_margin") == MARGIN, "配方 margin 不是恒 0.3")
    need((recipe.get("optimizer_args") or {}).get("weight_decay") == 4.0e-5, "配方 weight_decay 不是 4e-5")
    need(recipe.get("scheduler") == "ExponentialDecrease", "配方 scheduler 不是 ExponentialDecrease")
    need(sched.get("warm_up_epoch") == 0.2, "配方 warm_up_epoch 不是 0.2")
    need(sched.get("lr_scaling") == "legacy", "配方 lr_scaling 不是 legacy")
    need(recipe.get("exp_dir") == contract["new_run_root"], "配方 exp_dir 与合同 new_run_root 不一致")
    need(recipe.get("model_ckpt") == contract["checkpoint"]["path"], "配方 model_ckpt 与合同 checkpoint 不一致")

    # The frozen recipe wins over the environment: no overrides may slip in.
    for key in ("NUM_EPOCHS", "BATCH_SIZE", "SAMPLE_NUM_PER_EPOCH", "SAVE_EPOCH_INTERVAL", "RECIPE"):
        need(not os.environ.get(key), f"拒绝外部覆盖 {key}={os.environ.get(key)}")


def run(args, contract):
    source = os.path.join(args.package_root, "source")
    script = os.path.join(source, "applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh")
    need(os.path.isfile(script), f"冻结源码树缺少云端入口: {script}")

    # 验的就是跑的：REPRO=1 只会改 DEFAULT_RECIPE，显式 RECIPE 优先，
    # 因此把包内 resume.yaml 直接交给 trainer，避免"验包内配方、跑仓库配方"。
    recipe = os.path.join(args.package_root, contract.get("resume_config_path", "resume.yaml"))
    need(os.path.isfile(recipe), f"包内配方不存在: {recipe}")

    env = dict(os.environ)
    env.update({
        "REPO_ROOT": source,
        "RECIPE": recipe,
        "DATA_ROOT": env.get("DATA_ROOT", "/root/group-shared/voiceprint/data/wespeaker_data"),
        "EXP_DIR": contract["new_run_root"],
        "PTN_EXP_DIR": os.path.dirname(os.path.dirname(contract["checkpoint"]["path"])),
        "MODEL_CKPT": contract["checkpoint"]["path"],
        "NUM_GPUS": str(WORLD),
        "STAGE": "lm",
        "REPRO": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    need(env["RECIPE"] == recipe, "RECIPE 未指向包内 resume.yaml，验的与跑的可能不是同一份")
    env.pop("PYTHONPATH", None)
    print("LM_GATE_DELEGATE " + json.dumps(
        {"entry": script, "exp_dir": env["EXP_DIR"], "recipe": env["RECIPE"]}), flush=True)
    result = subprocess.run(["bash", script], env=env)
    postflight(contract, result.returncode)
    return result.returncode


def postflight(contract, code):
    run_root = contract["new_run_root"]
    receipt = os.path.join(run_root, "lm_receipt.json")
    payload = {"training_exit_code": code, "new_run_root": run_root, "checks": {}}
    try:
        checkpoint = os.path.join(run_root, "models", "model_1.pt")
        payload["checks"]["model_1_exists"] = os.path.isfile(checkpoint)
        if payload["checks"]["model_1_exists"]:
            payload["checks"]["model_1_sha256"] = sha256_file(checkpoint)

        # 冷启动标记由 shell 包装层打到 cloud_launch_lm.log；train.log 只有 trainer 输出。
        # 两处都读，避免 2026-11-07 那次"训练成功但 ok:false"的假阴性。
        candidate_logs = [os.path.join(run_root, "train.log"),
                          os.path.join(run_root, "cloud_launch_lm.log")]
        cold = forbidden = None
        texts = []
        for log_path in candidate_logs:
            if os.path.isfile(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as stream:
                    texts.append(stream.read())
        if texts:
            cold = any(COLD_START_MARKER in text for text in texts)
            forbidden = [line for text in texts for line in FORBIDDEN_LOG_LINES if line in text]
            payload["checks"]["cold_start_logs_seen"] = [p for p in candidate_logs if os.path.isfile(p)]
        payload["checks"]["cold_start_marker"] = cold
        payload["checks"]["forbidden_restore_lines"] = forbidden or []
        payload["checks"]["cold_start_clean"] = bool(cold) and not forbidden
        # ok 要同时看训练退出码：崩溃但恰好留下 model_1.pt 与 marker 时，
        # 不能记成成功（审计 P0：307 行原实现只看 model_1 + marker）。
        payload["checks"]["training_succeeded"] = (code == 0)
        payload["checks"]["ok"] = (
            bool(payload["checks"].get("model_1_exists"))
            and bool(payload["checks"]["cold_start_clean"])
            and payload["checks"]["training_succeeded"]
        )
    except OSError as error:
        payload["checks"]["error"] = str(error)
        payload["checks"]["ok"] = False
    if os.path.exists(receipt):
        print("LM_GATE_NOTE " + json.dumps({"note": "receipt 已存在，不覆盖", "path": receipt}), flush=True)
        return
    with open(receipt, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("LM_GATE_POSTFLIGHT " + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
