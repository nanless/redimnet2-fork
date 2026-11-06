#!/usr/bin/env python3
"""Build a frozen LM package derived from the reviewed PTN model47 package.

The existing `tools/ptn_resume/build.py` cannot build this: its spec schema,
`config_from_legacy()` and the emitted contract are hardcoded to the PTN
47→48 resume (num_epochs 120, batch 96, num_frms 200, num_class 357822,
scale_ratio 12, checkpoint == model_47, restore optimizer/scaler). An LM package
needs the opposite of several of those, so this builder is a separate file. It
follows the same rules as the PTN builder:

  * refuses to write into an existing directory (no overwrite, no retry-in-place)
  * re-hashes every inherited member instead of trusting a copied manifest
  * writes YAML through `yaml.safe_dump` and re-reads it to prove the numeric
    types survived (the 2e-05 / "2e-05" trap)
  * records `builder_sha256`, `spec_sha256`, package and contract SHA so the
    caller can pin them externally
  * never imports the trainer, never touches a GPU, never submits anything

Single self-contained file on purpose: it belongs on the training host, where
nothing can be installed, and it must be pipeable there
(`THREE... LM_BUILD_SPEC_B64=… python -B - build`) so no code is copied around.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import shutil
import stat as statmod
import sys

SCHEMA_SPEC = "lm-package-build-spec/v1"
SCHEMA_CONTRACT = "lm-from-ptn115/v1"
SPEC_KEYS = {
    "schema", "derived_from", "package_root", "new_run_root", "checkpoint", "speaker_map",
    "recipe", "terminals", "legacy", "locks", "runtime_source", "lm_label",
}
PIN_MODEL_115_SHA256 = "65497b6a00efac412a82d563aebc1e767056fef23e7f4234f48f89cc8b3aad5c"
LEGACY_BUNDLE_SHA256 = "840235e7b5963ada274531a29401b9f2bd75fe92e985a8e8e560341ac48350d8"
SOURCE189_SHA256 = "b342964138b7ff72b5403add10f8ac4b962f6b5346d501e757176504c89196b0"
LM_BATCH = 48
LM_WORLD = 8
LM_STEPS = 13075
LM_MARGIN = 0.3
LM_LR = {"initial": 1.0e-4, "peak": 4.5471497e-4, "peak_step": 2615, "final": 1.5001590e-4}
SKIP = {"bundle_files.json", "build-receipt.json"}


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


def path_of(value, label):
    need(isinstance(value, (str, os.PathLike)) and str(value), f"{label} 路径为空")
    path = os.path.abspath(str(value))
    need(os.path.isabs(path), f"{label} 必须是绝对路径")
    return path


def stable(path):
    need(os.path.isfile(path), f"文件不存在: {path}")
    need(not os.path.islink(path), f"不接受软链: {path}")
    before = os.stat(path)
    value = sha256_file(path)
    after = os.stat(path)
    key = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    need(key(before) == key(after), f"读取期间文件变化: {path}")
    need(statmod.S_ISREG(after.st_mode), f"不是普通文件: {path}")
    return {"path": path, "sha256": value, "bytes": after.st_size, "stat": list(key(after))}


def load_json(path):
    with open(path, "rb") as stream:
        return json.loads(stream.read().decode("utf-8"))


def copy_verified(source, target, expected_sha, label):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(source, target)
    actual = sha256_file(target)
    need(actual == expected_sha, f"{label} 复制后摘要不符: {source}")
    return actual


def build(spec, builder_sha256, spec_sha256):
    need(isinstance(spec, dict), "spec 必须是 JSON 对象")
    need(set(spec) == SPEC_KEYS, f"spec 键集合不符：多 {sorted(set(spec) - SPEC_KEYS)} 少 {sorted(SPEC_KEYS - set(spec))}")
    need(spec.get("schema") == SCHEMA_SPEC, f"spec schema 必须是 {SCHEMA_SPEC}")

    package_root = path_of(spec["package_root"], "package_root")
    need(not os.path.exists(package_root), f"package_root 已存在，拒绝覆盖: {package_root}")
    new_run_root = path_of(spec["new_run_root"], "new_run_root")
    need(not os.path.exists(new_run_root), f"new_run_root 已存在，拒绝覆盖: {new_run_root}")

    # -- 1. provenance of the package this one is derived from ----------------
    origin = spec.get("derived_from") or {}
    origin_root = path_of(origin.get("package_root"), "derived_from.package_root")
    origin_bundle = path_of(os.path.join(origin_root, "bundle_files.json"), "origin bundle")
    origin_contract = path_of(os.path.join(origin_root, "resume-contract.json"), "origin contract")
    need(stable(origin_bundle)["sha256"] == origin.get("bundle_files_sha256"),
         "被派生包的 bundle_files.json 摘要与 spec 不符")
    need(stable(origin_contract)["sha256"] == origin.get("contract_sha256"),
         "被派生包的 resume-contract.json 摘要与 spec 不符")
    origin_manifest = path_of(os.path.join(origin_root, "source_manifest.json"), "origin manifest")
    origin_delta = path_of(os.path.join(origin_root, "source_delta.json"), "origin delta")
    origin_contract_value = load_json(origin_contract)
    origin_runtime = origin_contract_value.get("runtime_files") or {}
    manifest = load_json(origin_manifest)
    need(isinstance(manifest, list) and len(manifest) == 190, "被派生包的 source_manifest 不是 190 成员")

    # -- 2. LM inputs --------------------------------------------------------
    checkpoint = spec["checkpoint"]
    checkpoint_path = path_of(checkpoint.get("path"), "checkpoint.path")
    need(os.path.basename(checkpoint_path) == "model_115.pt", "LM 必须从 model_115.pt 初始化")
    checkpoint_fact = stable(checkpoint_path)
    need(checkpoint_fact["sha256"] == PIN_MODEL_115_SHA256, "model_115.pt 摘要不是 pin 值")
    need(checkpoint_fact["sha256"] == checkpoint.get("sha256"), "spec 里的 checkpoint 摘要与实测不符")

    speaker_map = spec["speaker_map"]
    speaker_path = path_of(speaker_map.get("path"), "speaker_map.path")
    speaker_fact = stable(speaker_path)
    need(speaker_fact["sha256"] == speaker_map.get("sha256"), "speaker_map 摘要与实测不符")

    recipe_path = path_of(spec["recipe"].get("path"), "recipe.path")
    recipe_fact = stable(recipe_path)
    need(recipe_fact["sha256"] == spec["recipe"].get("sha256"), "recipe 摘要与实测不符")

    terminals = {}
    for name in ("predecessor", "original"):
        entry = (spec.get("terminals") or {}).get(name)
        need(entry, f"terminals.{name} 缺失")
        path = path_of(entry.get("path"), f"terminals.{name}.path")
        fact = stable(path)
        payload = load_json(path)
        terminals[name] = {
            "reference": {"path": path, "sha256": fact["sha256"]},
            "payload": payload,
            "authority": "externally obtained control-plane observation; file validation is not platform authentication",
        }

    legacy = spec.get("legacy") or {}
    need(legacy.get("bundle_files_sha256") == LEGACY_BUNDLE_SHA256, "legacy.bundle_files_sha256 不是 v3 旧包摘要")
    need((legacy.get("source189_sha256") or SOURCE189_SHA256) == SOURCE189_SHA256, "legacy.source189_sha256 不是 pin")
    source95 = legacy.get("source95") or {}
    source95_fact = stable(path_of(source95.get("path"), "legacy.source95.path"))
    need(source95_fact["sha256"] == source95.get("sha256"), "legacy.source95 摘要与实测不符")

    locks = spec.get("locks") or {}
    for key in ("logical_chain_lock", "admission_lock", "required_existing_locks"):
        need(locks.get(key), f"locks.{key} 缺失")

    runtime_source = spec.get("runtime_source") or {}
    for name in ("lm_launch.sh", "lm_stage_gate.py"):
        need(runtime_source.get(name), f"runtime_source.{name} 缺失")

    # -- 3. copy the frozen source tree -------------------------------------
    os.makedirs(package_root)
    source_target = os.path.join(package_root, "source")
    copied = 0
    for member in manifest:
        relative = member["relative"]
        source = os.path.join(origin_root, "source", relative)
        target = os.path.join(source_target, relative)
        copied_sha = copy_verified(source, target, member["sha256"], f"source/{relative}")
        need(os.path.getsize(target) == member["bytes"], f"source 成员字节数不符: {relative}")
        copied += 1

    # -- 4. copy the inherited runtime skeleton -----------------------------
    runtime_target = os.path.join(package_root, "runtime")
    for name, pinned in sorted(origin_runtime.items()):
        copy_verified(os.path.join(origin_root, "runtime", name), os.path.join(runtime_target, name),
                      pinned, f"runtime/{name}")

    # -- 5. add the LM runtime files ---------------------------------------
    runtime_files = dict(origin_runtime)
    for name in ("lm_launch.sh", "lm_stage_gate.py"):
        entry = runtime_source[name]
        target = os.path.join(runtime_target, name)
        if isinstance(entry, dict):
            # Inline content: the maintainer side may be the only place the new
            # runtime files exist, and the training host must stay untouched
            # apart from the new package directory.
            need(entry.get("content_b64"), f"runtime_source.{name} 内容为空")
            data = base64.b64decode(entry["content_b64"], validate=True)
            with open(target, "wb") as stream:
                stream.write(data)
            need(entry.get("sha256") in (None, sha256_file(target)),
                 f"runtime_source.{name} 内联内容与声明的 sha 不符")
        else:
            source_path = path_of(entry, f"runtime_source.{name}")
            if name in origin_runtime:
                copy_verified(source_path, target, origin_runtime[name], f"runtime/{name}")
            else:
                shutil.copyfile(source_path, target)
        if name.endswith(".sh"):
            os.chmod(target, 0o755)
        runtime_files[name] = sha256_file(target)

    # -- 6. write the LM config --------------------------------------------
    import yaml

    with open(recipe_path, "r", encoding="utf-8") as stream:
        recipe = yaml.safe_load(stream)
    need(isinstance(recipe, dict), "recipe 顶层不是映射")
    need(recipe.get("num_epochs") == 1, "recipe 不是 1 轮")
    need((recipe.get("dataloader_args") or {}).get("batch_size") == LM_BATCH, "recipe 每卡 batch 不是 48")
    need((recipe.get("margin_update") or {}).get("final_margin") == LM_MARGIN, "recipe margin 不是 0.3")

    config = dict(recipe)
    config["exp_dir"] = new_run_root
    config["model_ckpt"] = checkpoint_path
    config["spk2id_map_path"] = speaker_path
    need("checkpoint" not in config, "LM 配方不应带 PTN 的 checkpoint 键")

    config_path = os.path.join(package_root, "resume.yaml")
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    with open(config_path, "r", encoding="utf-8") as stream:
        readback = yaml.safe_load(stream)
    need(readback == config, "resume.yaml 回读与写入对象不一致（数值类型可能在 YAML 里变了）")
    config_fact = stable(config_path)

    # -- 7. write the contract ---------------------------------------------
    contract = {
        "schema": SCHEMA_CONTRACT,
        "runtime_admission": False,
        "LM_started": False,
        "builder_sha256": builder_sha256,
        "spec_sha256": spec_sha256,
        "derived_from": {
            "package_root": origin_root,
            "bundle_files_sha256": origin.get("bundle_files_sha256"),
            "contract_sha256": origin.get("contract_sha256"),
        },
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_fact["sha256"], "bytes": checkpoint_fact["bytes"]},
        "checkpoint_stat_at_build": checkpoint_fact["stat"],
        "speaker_map": {
            "path": speaker_path,
            "sha256": speaker_fact["sha256"],
            "ptn_speakers": 119274,
            "classes": 115362,
            "lm_utterances": 5021010,
            "removed_from_ptn": 3912,
        },
        "lm_label": {"path": path_of(spec["lm_label"], "lm_label")},
        "clock": {
            "stage": "lm",
            "start_epoch": 1,
            "target_epoch": 1,
            "epoch_iter": LM_STEPS,
            "batch_size_per_gpu": LM_BATCH,
            "world_size": LM_WORLD,
            "workers_per_rank": 24,
            "lr": LM_LR,
            "margin": LM_MARGIN,
            "optimizer": "new",
            "scaler": "new",
            "rng_restored": False,
            "saved_epoch": 115,
            "num_frms": 500,
            "speed_perturb": False,
            "aug_root_prob": 0.7,
        },
        "new_run_root": new_run_root,
        "resume_config_path": "resume.yaml",
        "resume_config_sha256": config_fact["sha256"],
        "source_root": source_target,
        "source_manifest_sha256": sha256_file(os.path.join(package_root, "source_manifest.json"))
        if os.path.exists(os.path.join(package_root, "source_manifest.json")) else None,
        "runtime_files": runtime_files,
        "ptn_runtime_inherited": sorted(origin_runtime),
        "executor": "runtime/lm_launch.sh",
        "legacy": {
            "bundle_root": path_of(legacy.get("bundle_root"), "legacy.bundle_root"),
            "bundle_files_sha256": LEGACY_BUNDLE_SHA256,
            "source189_sha256": SOURCE189_SHA256,
            "source95": {"path": source95_fact["path"], "sha256": source95_fact["sha256"]},
        },
        "terminals": terminals,
        "logical_chain_lock": locks["logical_chain_lock"],
        "admission_lock": locks["admission_lock"],
        "required_existing_locks": list(locks["required_existing_locks"]),
        "must_recheck_under_exclusive_locks": [
            "fresh control-plane terminal for the third segment",
            "stable inode locks of the three segments plus logical/admission locks",
            "model_115.pt bytes and speaker map bytes",
            "unique new LM output claim before any write",
            "eight-rank cold optimizer/scaler evidence inside the LM task",
        ],
        "historical_rng_missing": True,
        "cold_start": {
            "loads_model_parameters_only": True,
            "restores_ptn_optimizer": False,
            "restores_ptn_scaler": False,
            "automatic_retry": False,
            "marker": "cold start: model_ckpt loads model parameters only; optimizer/scaler fresh",
        },
        "notes": [
            "runtime/{entry,worker}.py 是从 PTN 包原样继承的，仅作来源记录，LM 不使用它们",
            "spk2id.json 是执行门：缺失或摘要不符会退回按索引切片，gate 在训练前拒绝",
            "LM 训练走 source/ 里的 STAGE=lm 普通入口，由 runtime/lm_stage_gate.py 在前后把关",
        ],
    }

    # source manifest / delta are byte-identical copies (the tree is identical)
    for name in ("source_manifest.json", "source_delta.json"):
        target = os.path.join(package_root, name)
        shutil.copyfile(os.path.join(origin_root, name), target)
        need(sha256_file(target) == stable(os.path.join(origin_root, name))["sha256"], f"{name} 复制后不一致")
    contract["source_manifest_sha256"] = sha256_file(os.path.join(package_root, "source_manifest.json"))
    contract["source_delta_sha256"] = sha256_file(os.path.join(package_root, "source_delta.json"))

    contract_path = os.path.join(package_root, "resume-contract.json")
    with open(contract_path, "w", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    contract_sha = sha256_file(contract_path)

    # -- 8. package manifest ------------------------------------------------
    bundle = {}
    for base, _dirs, files in os.walk(package_root):
        for name in files:
            target = os.path.join(base, name)
            relative = os.path.relpath(target, package_root).replace(os.sep, "/")
            if relative in SKIP:
                continue
            bundle[relative] = sha256_file(target)
    need(bundle, "包为空")
    bundle_path = os.path.join(package_root, "bundle_files.json")
    with open(bundle_path, "w", encoding="utf-8") as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    package_sha = sha256_file(bundle_path)

    receipt = {
        "schema": "lm-package-build-receipt/v1",
        "package_root": package_root,
        "package_sha256": package_sha,
        "contract_sha256": contract_sha,
        "builder_sha256": builder_sha256,
        "spec_sha256": spec_sha256,
        "members": len(bundle),
        "source_members": copied,
        "runtime_members": len(runtime_files),
        "derived_from": contract["derived_from"],
        "checkpoint_sha256": checkpoint_fact["sha256"],
        "resume_config_sha256": config_fact["sha256"],
    }
    with open(os.path.join(package_root, "build-receipt.json"), "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default=None, help="spec 路径；缺省读 LM_BUILD_SPEC_B64")
    parser.add_argument("--output", default="-", help="回执输出路径，- 表示 stdout")
    args = parser.parse_args(argv)

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    try:
        if args.spec and args.spec != "-":
            with open(args.spec, "rb") as stream:
                raw = stream.read()
        else:
            text = os.environ.get("LM_BUILD_SPEC_B64", "")
            need(text, "没有 spec：请用 --spec 或 LM_BUILD_SPEC_B64")
            raw = base64.b64decode(text.encode("ascii"), validate=True)
        spec = json.loads(raw.decode("utf-8"))
        # When this file is piped to `python -B -` there is no real __file__;
        # the caller then supplies the digest of the exact bytes it piped.
        try:
            builder_sha = sha256_file(os.path.abspath(__file__))
        except OSError:
            builder_sha = os.environ.get("LM_BUILDER_SHA256", "")
        need(builder_sha and len(builder_sha) == 64, "无法确定 builder 摘要：请设置 LM_BUILDER_SHA256")
        receipt = build(spec, builder_sha, hashlib.sha256(raw).hexdigest())
    except (Refused, binascii.Error, ValueError, KeyError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    text = json.dumps({"ok": True, **receipt}, ensure_ascii=False, indent=2) + "\n"
    if args.output in (None, "-"):
        sys.stdout.write(text)
    else:
        with open(args.output, "x", encoding="utf-8") as stream:
            stream.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
