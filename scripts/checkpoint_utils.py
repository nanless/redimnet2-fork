"""Strict ReDimNet2 embedding loaders for released and training checkpoints."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import yaml

from redimnet2.redimnet2 import ReDimNet2Wrap


def _normalise_training_keys(state):
    """Remove DDP / torch.compile wrapper names, rejecting ambiguous keys."""
    result = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Training model must contain only string-to-tensor entries")
        parts = key.split(".")
        while parts and parts[0] in {"module", "_orig_mod"}:
            parts.pop(0)
        parts = [part for part in parts if part != "_orig_mod"]
        normalised = ".".join(parts)
        if normalised in result:
            raise ValueError(f"Wrapper-prefix removal produces duplicate key: {normalised}")
        result[normalised] = value
    return result


def embedding_payload(path: Path, training_config: Path | None = None):
    """Return a strictly validated released-format payload, without optimizer state.

    Released files keep their existing model_config/state_dict interpretation.
    Training files require the actual experiment YAML (its model_args defines
    the embedding network). Only the explicitly named projection.* classifier
    is discarded; all embedding parameters and buffers must match strictly.
    Load only checkpoints and YAML files from trusted sources.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a checkpoint mapping: {path}")
    if "model_config" in checkpoint or "state_dict" in checkpoint:
        if "model_config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError("Released checkpoint requires both model_config and state_dict")
        config, state = checkpoint["model_config"], checkpoint["state_dict"]
        metadata = checkpoint.get("export_info")
    else:
        if training_config is None:
            raise ValueError(
                f"Training checkpoint {path} requires --training-config pointing to "
                "the experiment's config.yaml (or export it with export_checkpoint.py)"
            )
        with Path(training_config).open() as stream:
            recipe = yaml.safe_load(stream)
        if not isinstance(recipe, Mapping) or recipe.get("model") not in {
            "ReDimNet2Custom", "ReDimNet2Wrap"
        } or not isinstance(recipe.get("model_args"), Mapping):
            raise ValueError("Training YAML must specify ReDimNet2Custom/Wrap and model_args")
        config = dict(recipe["model_args"])
        raw_state = checkpoint.get("model", checkpoint)
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise ValueError("Training checkpoint has no non-empty model state")
        normalised = _normalise_training_keys(raw_state)
        unknown = [k for k in normalised if not k.startswith(("backbone.", "projection."))]
        if unknown:
            raise ValueError(f"Unexpected training parameters outside backbone/projection: {unknown[:8]}")
        state = {k[len("backbone."):]: v for k, v in normalised.items() if k.startswith("backbone.")}
        metadata = {
            "source_checkpoint": str(path.resolve()),
            "source_training_config": str(Path(training_config).resolve()),
            "source_epoch": checkpoint.get("epoch"),
            "discarded_projection_keys": sorted(k for k in normalised if k.startswith("projection.")),
        }
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("model_config and state_dict must be mappings")
    model = ReDimNet2Wrap(**dict(config))
    model.load_state_dict(state, strict=True)
    # Return the tensors from the validated module, without classifier or optimizer.
    payload = {"model_config": dict(config), "state_dict": model.state_dict()}
    if metadata is not None:
        payload["export_info"] = metadata
    return payload, model.eval()


def load_embedding_checkpoint(path: Path, device: torch.device,
                              training_config: Path | None = None):
    _, model = embedding_payload(path, training_config)
    return model.to(device).eval()
