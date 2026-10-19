#!/usr/bin/env python3
"""Regression checks for checkpoint safety and optional real-training export parity."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from redimnet2.redimnet2 import ReDimNet2Wrap
from scripts.checkpoint_utils import embedding_payload
from scripts.export_checkpoint import export_checkpoint
from scripts.eval_vox1_checkpoints import load_checkpoint as load_vox1
from scripts.eval_ood_checkpoints import load_checkpoint as load_ood


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {"F": 8, "C": 4, "causal": "none", "stages_setup": [((1, 1), 1, 1, [(3, 3)], 4)],
                       "feat_type": "tf", "embed_dim": 8}
        self.model = ReDimNet2Wrap(**self.config).eval()
        self.recipe = self.root / "config.yaml"
        self.recipe.write_text(yaml.safe_dump({"model": "ReDimNet2Custom", "model_args": self.config}))
        self.state = {"backbone." + k: v for k, v in self.model.state_dict().items()}
        self.state["projection.weight"] = torch.randn(5, 8)
        self.source = self.root / "model_1.pt"
        self.write(self.state)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, state):
        torch.save({"epoch": 1, "model": state, "optimizer": {"state": {}}}, self.source)

    def test_real_model_export_and_both_eval_loaders(self):
        output = self.root / "embedding.pt"
        export_checkpoint(self.source, self.recipe, output)
        wave = torch.randn(1, 16000)
        with torch.inference_mode():
            expected = self.model(wave)
            for load in (load_vox1, load_ood):
                torch.testing.assert_close(load(self.source, torch.device("cpu"), self.recipe)(wave), expected, rtol=0, atol=0)
                torch.testing.assert_close(load(output, torch.device("cpu"))(wave), expected, rtol=0, atol=0)
        released = torch.load(output, weights_only=False)
        self.assertEqual(set(released), {"model_config", "state_dict", "export_info"})
        self.assertEqual(released["export_info"]["discarded_projection_keys"], ["projection.weight"])
        with self.assertRaises(FileExistsError):
            export_checkpoint(self.source, self.recipe, output)

    def test_missing_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --training-config"):
            embedding_payload(self.source)

    def test_wrapped_and_legacy_bare_training_state(self):
        wrapped = {"module._orig_mod." + k.replace("backbone.", "backbone._orig_mod.", 1): v
                   for k, v in self.state.items()}
        torch.save(wrapped, self.source)
        _, loaded = embedding_payload(self.source, self.recipe)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)

    def test_duplicate_wrapper_keys_rejected(self):
        state = dict(self.state)
        key = next(iter(state))
        state["module." + key] = state[key]
        self.write(state)
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            embedding_payload(self.source, self.recipe)

    def test_missing_embedding_tensor_rejected(self):
        self.state.pop(next(iter(self.state)))
        self.write(self.state)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            embedding_payload(self.source, self.recipe)

    def test_unknown_branch_rejected(self):
        self.state["other_network.weight"] = torch.ones(1)
        self.write(self.state)
        with self.assertRaisesRegex(ValueError, "outside backbone/projection"):
            embedding_payload(self.source, self.recipe)

    def test_incompatible_recipe_rejected(self):
        self.config["embed_dim"] = 9
        self.recipe.write_text(yaml.safe_dump({"model": "ReDimNet2Custom", "model_args": self.config}))
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            embedding_payload(self.source, self.recipe)

    def test_original_released_format_remains_supported(self):
        torch.save({"model_config": self.config, "state_dict": self.model.state_dict()}, self.source)
        payload, _ = embedding_payload(self.source)
        self.assertNotIn("export_info", payload)


def real_training_check(checkpoint, recipe, official):
    """Compare against the original bare backbone, not the exporter's own loader."""
    torch.manual_seed(42)
    config = yaml.safe_load(Path(recipe).read_text())["model_args"]
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw["model"]
    reference = ReDimNet2Wrap(**config).eval()
    # The repository's real saver already removes compiler and DDP wrappers.
    reference.load_state_dict({k[len("backbone."):]: v for k, v in state.items()
                               if k.startswith("backbone.")}, strict=True)
    del raw, state
    wave = torch.randn(1, 32000)
    with torch.inference_mode():
        expected = reference(wave)
    del reference
    with tempfile.TemporaryDirectory(prefix="redimnet2-export-check-") as temp:
        output = Path(temp) / "embedding.pt"
        export_checkpoint(Path(checkpoint), Path(recipe), output)
        for label, load, source, source_recipe in (
            ("Vox1 training", load_vox1, checkpoint, recipe),
            ("OOD training", load_ood, checkpoint, recipe),
            ("Vox1 exported", load_vox1, output, None),
            ("OOD exported", load_ood, output, None),
        ):
            model = load(Path(source), torch.device("cpu"), Path(source_recipe) if source_recipe else None)
            with torch.inference_mode():
                actual = model(wave)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(f"{label}: max_abs_diff={(actual - expected).abs().max().item():.9g}; shape={tuple(actual.shape)}")
            del model
        print(f"Training checkpoint={checkpoint}; exported bytes={output.stat().st_size}")
    if official:
        for load in (load_vox1, load_ood):
            model = load(Path(official), torch.device("cpu"))
            with torch.inference_mode():
                actual = model(wave)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise AssertionError("Official checkpoint inference failed")
            del model
        print(f"Official checkpoint accepted by both loaders: {official}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoint", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--official-checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExportTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.training_checkpoint:
        if not args.training_config:
            parser.error("--training-config is required with --training-checkpoint")
        real_training_check(args.training_checkpoint, args.training_config, args.official_checkpoint)
