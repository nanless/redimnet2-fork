#!/usr/bin/env python3
"""Check the exact three-corpus B6 PTN -> LM initialization before training.

SphereFace2 has one centroid per class. Its t=3 is a score transform, not
three subcenters. Speed perturbation, rather than subcenters, triples PTN
classes. This command checks every LM centroid against its named PTN row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "applications/speaker_rec"))
from wespeaker_lite.models.projections import get_projection
from wespeaker_lite.utils.checkpoint import extract_state_dict
from wespeaker_lite.utils.utils import reassemble_projection_head

PTN_SPEAKERS = 119274
LM_SPEAKERS = 115362
LM_UTTERANCES = 5021010
EMBED_DIM = 192
SPEED_CLASSES = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key in speaker mapping: {key!r}")
        result[key] = value
    return result


def load_mapping(path):
    with path.open() as stream:
        mapping = json.load(stream, object_pairs_hook=unique_object)
    require(isinstance(mapping, dict) and mapping, "PTN speaker mapping must be a nonempty object")
    require(all(isinstance(k, str) and k for k in mapping), "speaker names must be nonempty strings")
    require(all(type(v) is int for v in mapping.values()), "speaker IDs must be integers, not booleans")
    require(len(set(mapping.values())) == len(mapping), "PTN speaker IDs are not unique")
    require(set(mapping.values()) == set(range(len(mapping))), "PTN speaker IDs must be contiguous from zero")
    require(len(mapping) == PTN_SPEAKERS,
            f"expected {PTN_SPEAKERS} PTN speakers, found {len(mapping)}")
    require(all(mapping[name] == index for index, name in enumerate(sorted(mapping))),
            "PTN mapping does not match the lexical speaker-ID order used by train.py")
    return mapping


def load_lm_mapping(path, old):
    speakers = set()
    rows = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.split()
            require(len(parts) == 2, f"LM label line {line_number} must contain utterance and speaker")
            speakers.add(parts[1])
            rows += 1
    missing = sorted(speakers - old.keys())
    require(not missing, f"LM speakers missing from PTN mapping: count={len(missing)}, first={missing[:10]}")
    require(len(speakers) == LM_SPEAKERS,
            f"expected {LM_SPEAKERS} LM speakers, found {len(speakers)}")
    require(rows == LM_UTTERANCES,
            f"expected {LM_UTTERANCES} LM utterances, found {rows}")
    # Exactly the lexical ordering used by train.py's spk2id().
    return {speaker: index for index, speaker in enumerate(sorted(speakers))}, rows


def check(args):
    mapping = load_mapping(args.ptn_spk2id)
    lm_mapping, lm_rows = load_lm_mapping(args.lm_label, mapping)
    with args.ptn_recipe.open() as stream:
        recipe = yaml.safe_load(stream)
    projection_args = dict(recipe["projection_args"])
    require(projection_args.get("project_type") == "sphereface2", "PTN recipe must use sphereface2")
    require(projection_args.get("t") == 3, "PTN SphereFace2 score transform t must be 3")
    require(recipe["model_args"].get("embed_dim") == EMBED_DIM, "B6 embedding dimension must be 192")
    require(projection_args.get("embed_dim") == EMBED_DIM, "projection embedding dimension must be 192")
    require(recipe["dataset_args"].get("speed_perturb") is True, "PTN speed perturbation must be enabled")
    require(not recipe.get("do_lm", False), "expected a PTN recipe, not an LM recipe")
    projection_args["num_class"] = 1
    tiny_projection = get_projection(projection_args)
    require(type(tiny_projection).__name__ == "SphereFace2", "unexpected projection implementation")
    require(tuple(tiny_projection.weight.shape) == (1, EMBED_DIM),
            "SphereFace2 implementation changed: expected one centroid per class")
    require(tuple(tiny_projection.bias.shape) == (1, 1), "unexpected SphereFace2 bias layout")
    del tiny_projection

    # mmap avoids copying the optimizer buffers from a large training checkpoint.
    # Legacy, non-zip torch.save files are supported by the bounded fallback.
    try:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state, extra = extract_state_dict(raw)
    require(isinstance(state, dict), "checkpoint does not contain a training state dictionary")
    projection_keys = {key for key in state if key.startswith("projection.")}
    require(projection_keys == {"projection.weight", "projection.bias"},
            f"unexpected projection state layout: {sorted(projection_keys)}")
    require(any(key.startswith("backbone.") for key in state), "checkpoint has no trained backbone")
    weight = state["projection.weight"]
    bias = state["projection.bias"]
    expected_shape = (len(mapping) * SPEED_CLASSES, EMBED_DIM)
    require(isinstance(weight, torch.Tensor) and tuple(weight.shape) == expected_shape,
            f"PTN projection.weight must have shape {expected_shape}, found {getattr(weight, 'shape', None)}")
    require(weight.is_floating_point(), "PTN projection weights must be floating point")
    require(isinstance(bias, torch.Tensor) and tuple(bias.shape) == (1, 1), "PTN projection.bias must have shape (1, 1)")
    require(bool(torch.isfinite(bias).all()), "PTN projection bias contains nonfinite values")
    for start in range(0, weight.shape[0], 4096):
        require(bool(torch.isfinite(weight[start:start + 4096]).all()),
                f"PTN projection weights contain nonfinite values near row {start}")
    recorded_epoch = extra.get("epoch")
    if args.expected_epoch is not None:
        require(type(recorded_epoch) is int and recorded_epoch == args.expected_epoch,
                f"expected recorded epoch {args.expected_epoch}, found {recorded_epoch!r}")
        match = re.fullmatch(r"model_(\d+)\.pt", args.checkpoint.name)
        require(match is not None and int(match.group(1)) == args.expected_epoch,
                "checkpoint filename must agree with the required PTN epoch")
    new_weight, stats = reassemble_projection_head(weight, mapping, lm_mapping)
    require(stats["new"] == 0 and stats["matched"] == LM_SPEAKERS,
            f"LM head must have complete name coverage without random initialization: {stats}")
    require(tuple(new_weight.shape) == (LM_SPEAKERS, EMBED_DIM), "unexpected reconstructed LM head shape")
    names = list(lm_mapping)
    for start in range(0, len(names), 4096):
        old_rows = torch.tensor([mapping[name] for name in names[start:start + 4096]], dtype=torch.long)
        expected = weight.index_select(0, old_rows)
        require(torch.equal(new_weight[start:start + len(old_rows)], expected),
                f"reconstructed LM centroids do not exactly match named PTN rows near LM row {start}")
    return {
        "ok": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "ptn_spk2id": str(args.ptn_spk2id.resolve()),
        "lm_label": str(args.lm_label.resolve()),
        "ptn_recipe": str(args.ptn_recipe.resolve()),
        "recorded_epoch": recorded_epoch,
        "required_epoch": args.expected_epoch,
        "projection_type": "SphereFace2",
        "type_basis": "required PTN recipe and verified implementation/state layout; checkpoint has no class-name metadata",
        "score_transform_t": 3,
        "subcenters_per_class": 1,
        "speed_classes": SPEED_CLASSES,
        "ptn_projection_shape": list(weight.shape),
        "lm_projection_shape": list(new_weight.shape),
        "lm_utterances": lm_rows,
        "reassembly": stats,
        "all_lm_rows_exact": True,
        "random_or_slice_fallback_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ptn-spk2id", required=True, type=Path)
    parser.add_argument("--lm-label", required=True, type=Path)
    parser.add_argument("--ptn-recipe", type=Path, default=REPO_ROOT / "applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/official_repro_ptn.yaml")
    parser.add_argument("--expected-epoch", type=int,
                        help="formal run must use 115; omit only for an isolated short test")
    args = parser.parse_args()
    try:
        report = check(args)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
