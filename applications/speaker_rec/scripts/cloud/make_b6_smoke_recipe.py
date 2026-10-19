#!/usr/bin/env python3
"""Derive a short B6 smoke recipe without compressing the official step clock."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
# Counts independently audited on 2026-10-19; the wrapper verifies the exact
# full training-table hashes before this recipe can be used.
FULL_COUNTS = {"ptn": 11521178, "lm": 5021010}
BATCH_SIZES = {"ptn": 96, "lm": 48}
FULL_EPOCHS = {"ptn": 120, "lm": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_schedulers():
    source = REPO_ROOT / "applications/speaker_rec/wespeaker_lite/utils/schedulers.py"
    spec = importlib.util.spec_from_file_location("actual_b6_schedulers", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProbeOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]


class ProbeProjection:
    def __init__(self):
        self.margin = None

    def update(self, margin):
        self.margin = margin


def make_lr(module, recipe, epoch_iter, epochs):
    arguments = dict(recipe["scheduler_args"])
    arguments.update(num_epochs=epochs, epoch_iter=epoch_iter,
                     scale_ratio=8 * recipe["dataloader_args"]["batch_size"] / 64)
    return getattr(module, recipe["scheduler"])(ProbeOptimizer(), **arguments)


def compare_lr(module, original, derived, full_steps, short_steps, epochs, steps):
    reference = make_lr(module, original, full_steps, FULL_EPOCHS[original["_proof_stage"]])
    candidate = make_lr(module, derived, short_steps, epochs)
    maximum = 0.0
    for step in steps:
        reference.current_iter = candidate.current_iter = step
        expected, actual = reference.get_current_lr(), candidate.get_current_lr()
        require(math.isfinite(expected) and math.isfinite(actual), "nonfinite learning rate in clock proof")
        difference = abs(expected - actual)
        maximum = max(maximum, difference)
        require(math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-14),
                f"learning-rate clock differs at step {step}: full={expected}, short={actual}")
    return maximum


def compare_margin(module, original, derived, full_steps, short_steps, count):
    arguments = original.get("margin_update")
    if not arguments or not arguments.get("update_margin", False):
        return {"enabled": False}
    old_args = dict(arguments, epoch_iter=full_steps)
    new_args = dict(derived["margin_update"], epoch_iter=short_steps)
    old = module.MarginScheduler(SimpleNamespace(projection=ProbeProjection()), **old_args)
    new = module.MarginScheduler(SimpleNamespace(projection=ProbeProjection()), **new_args)
    points = set(range(count))
    for key in ["increase_start_epoch", "fix_start_epoch"]:
        boundary = round((arguments[key] - 1) * full_steps)
        points.update([max(0, boundary - 1), boundary, boundary + 1])
    maximum = 0.0
    for step in sorted(points):
        old.step(step)
        new.step(step)
        difference = abs(old.get_margin() - new.get_margin())
        maximum = max(maximum, difference)
        require(math.isclose(old.get_margin(), new.get_margin(), rel_tol=1e-12, abs_tol=1e-14),
                f"margin clock differs at step {step}")
    return {"enabled": True, "points_checked": len(points), "max_abs_margin_difference": maximum}


def derive(args):
    require(args.num_gpus == 8, "this reproduction requires exactly eight GPUs")
    require(args.sample_num_per_epoch > 0, "sample count must be positive")
    require(not args.output.exists(), f"refusing to overwrite derived recipe: {args.output}")
    sidecar = args.output.with_suffix(".timing.json")
    require(not sidecar.exists(), f"refusing to overwrite derivation proof: {sidecar}")
    with args.recipe.open() as stream:
        original = yaml.safe_load(stream)
    require(isinstance(original, dict), "recipe must contain a mapping")
    require(original["dataloader_args"]["batch_size"] == BATCH_SIZES[args.stage], "unexpected B6 per-GPU batch")
    require(original["num_epochs"] == FULL_EPOCHS[args.stage], "expected the unchanged full official recipe")
    require(original["dataset_args"].get("sample_num_per_epoch", 0) == 0,
            "input must be the full-data official recipe, not an already shortened recipe")
    require(bool(original.get("do_lm", False)) == (args.stage == "lm"), "recipe stage does not match --stage")
    require(original["scheduler_args"].get("lr_scaling", "legacy") == "legacy", "official legacy scaling must stay unchanged")
    batch = BATCH_SIZES[args.stage]
    full_steps = FULL_COUNTS[args.stage] // args.num_gpus // batch
    short_steps = args.sample_num_per_epoch // args.num_gpus // batch
    require(short_steps == 32, "this validated smoke test requires exactly 32 updates per epoch")
    require(args.sample_num_per_epoch == short_steps * args.num_gpus * batch,
            "smoke sample count must be a whole global batch")
    ratio = full_steps / short_steps
    derived = copy.deepcopy(original)
    derived["dataset_args"]["sample_num_per_epoch"] = args.sample_num_per_epoch
    derived["num_epochs"] = 1
    scheduler = derived["scheduler_args"]
    changes = {}
    if args.stage == "ptn":
        require(original["scheduler"] == "ExponentialDecayPlateauWithWarmup", "unexpected PTN scheduler")
        require(not scheduler.get("decay_to_final_lr", False), "PTN must retain the official half-life formula")
        for key in ["warm_up_epoch", "plateau_epochs", "decay_epochs"]:
            before = scheduler[key]
            scheduler[key] = before * ratio
            changes[f"scheduler_args.{key}"] = {"full": before, "smoke": scheduler[key]}
        scheduler["decay_to_final_lr"] = False
    else:
        require(original["scheduler"] == "ExponentialDecrease", "unexpected LM scheduler")
        require(original["num_epochs"] == 1, "LM derivation assumes one complete fine-tuning epoch")
        before = scheduler["warm_up_epoch"]
        scheduler["warm_up_epoch"] = before * ratio
        changes["scheduler_args.warm_up_epoch"] = {"full": before, "smoke": scheduler["warm_up_epoch"]}
        initial, final = scheduler["initial_lr"], scheduler["final_lr"]
        require(initial > 0 and final > 0, "LM learning rates must be positive")
        scheduler["final_lr"] = initial * math.exp(math.log(final / initial) * short_steps / full_steps)
        changes["scheduler_args.final_lr"] = {"full": final, "smoke": scheduler["final_lr"]}
    margin = derived.get("margin_update", {})
    if margin.get("update_margin", False):
        for key in ["increase_start_epoch", "fix_start_epoch"]:
            before = margin[key]
            margin[key] = (before - 1) * ratio + 1
            changes[f"margin_update.{key}"] = {"full": before, "smoke": margin[key]}
    module = load_schedulers()
    original_for_proof = dict(original, _proof_stage=args.stage)
    count = 64 if args.stage == "ptn" else 32
    epochs_to_check = [1, 2] if args.stage == "ptn" else [1]
    errors = {str(epochs): compare_lr(module, original_for_proof, derived, full_steps,
                                    short_steps, epochs, range(count))
              for epochs in epochs_to_check}
    margin_proof = compare_margin(module, original, derived, full_steps, short_steps, count)
    reference = make_lr(module, original, full_steps, FULL_EPOCHS[args.stage])
    rates = {}
    for update in [1, 25, 32, 33, 64]:
        if update > count:
            continue
        reference.current_iter = update - 1
        rates[str(update)] = {"global_step_zero_based": update - 1,
                              "learning_rate": reference.get_current_lr()}
    report = {"ok": True, "stage": args.stage, "source_recipe": str(args.recipe.resolve()),
              "output": str(args.output.resolve()), "timing_json": str(sidecar.resolve()),
              "full_sample_count": FULL_COUNTS[args.stage], "smoke_sample_count": args.sample_num_per_epoch,
              "num_gpus": args.num_gpus, "per_gpu_batch": batch,
              "full_epoch_iter": full_steps, "short_epoch_iter": short_steps,
              "epoch_scale": ratio, "changes": changes,
              "proof": {"steps_checked": count, "short_num_epochs_checked": epochs_to_check,
                        "max_abs_lr_difference": max(errors.values()), "lr_difference_per_num_epochs": errors,
                        "margin": margin_proof},
              "lr_at_update_number": rates,
              "formal_recipe_changed": False,
              "meaning": "Only epoch-valued clock parameters are re-expressed; the short run uses the same absolute-update LR and margin as the full run."}
    derived["smoke_derivation"] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write("# Isolated smoke recipe: preserves the full-data absolute-update schedule.\n")
        stream.write("# Generated by make_b6_smoke_recipe.py; never use this file for a formal run.\n")
        yaml.safe_dump(derived, stream, sort_keys=False, allow_unicode=True)
    with sidecar.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["ptn", "lm"])
    parser.add_argument("--sample-num-per-epoch", required=True, type=int)
    parser.add_argument("--num-gpus", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = derive(args)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
