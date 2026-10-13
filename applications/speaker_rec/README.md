# Speaker Recognition Training

This application contains the lightweight [wespeaker](https://github.com/wenet-e2e/wespeaker)-forked
training stack used for ReDimNet2 speaker-recognition experiments.

## Contents

- `wespeaker_lite/`: training code, datasets, augmentations, schedulers, and
  checkpoint utilities.
- `examples/vox2/`: VoxCeleb2-only data preparation and b0–b6 recipes.
- `examples/vb2_vox2_cnc2/`: VoxCeleb2 + VoxBlink2 + CN-Celeb2 training list
  files (`data.list`, `utt2spk`) and b3/b6 recipes.
- `launch.md`: torchrun launch details and environment setup.
- `pyproject.toml`: Poetry environment with the Python dependencies needed by
  `wespeaker_lite` and the included data-preparation scripts.

Run commands from the repository root so both `redimnet2` and
`applications/speaker_rec/wespeaker_lite` are importable.

## Environment

The training code is GPU-oriented and uses `torchrun` with NCCL. Install a
PyTorch/torchaudio build that matches your CUDA driver. The Poetry file keeps
the dependency list lightweight and does not pin a CUDA wheel source.

```bash
cd applications/speaker_rec
poetry install --no-root
cd ../..
```

If your cluster requires a specific PyTorch wheel index, install matching
`torch` and `torchaudio` in the Poetry environment first, then run
`poetry install --no-root`:

```bash
cd applications/speaker_rec
poetry run python -m pip install torch torchaudio --index-url <pytorch-wheel-index>
poetry install --no-root
```

System tools are still required for data preparation and optional codec
augmentation:

- `ffmpeg` for VoxCeleb audio conversion and codec augmentation.
- `libsndfile` for the Python `soundfile` package.
- Extra `/dev/shm` only if you enable the legacy option
  `dataset_args.lmdb_to_ram: true` (current ReDimNet2 recipes do **not** use
  LMDB; they read MUSAN/RIRS from raw wav directories via `dataset_dir`).

## Typical Workflow

Prepare data with one of the example READMEs, create a small local YAML overlay
for machine-specific paths, then launch from the repository root:

```bash
cd applications/speaker_rec
poetry run bash -lc 'cd ../.. && ./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vox2/recipes/redimnet2/b3/ptn.yaml \
  8 \
  --config_overrides /tmp/vox2-ptn.local.yaml'
```

For the combined VoxCeleb2 + VoxBlink2 + CN-Celeb2 recipes, start with
`examples/vb2_vox2_cnc2/README.md`. That example uses `pyarrow` and
`soundfile` to build duration tables when filtering short utterances for the
large-margin stage, in addition to the core training dependencies.

## Recipe keys that have no effect

Some keys appear in the `recipes/redimnet2/**/*.yaml` files but are **not read
by the training code**. They are leftovers from other training stacks and can
be safely ignored; do not expect them to change anything.

| Key | Why it does nothing |
|-----|---------------------|
| `gpus` | Never referenced by `train.py` or `launch.sh`. The number of GPU processes is the second positional argument of `launch.sh`, not a config value. Setting `gpus: [0]` will **not** restrict training to a single GPU. |
| `grad_accum_steps` | Gradient accumulation is not implemented in this training stack. The value is unused, so the effective batch size is `dataloader_args.batch_size` times the number of GPU processes. |
| `eval_step_interval` | Validation is triggered by epoch, not by step (`_run_validation` runs once per epoch in `train.py`). The value is unused. |
| `scheduler_args.scale_ratio` | Overwritten at startup by `train.py`: `configs['scheduler_args']['scale_ratio'] = world_size * batch_size / 64`. Whatever value you put in the YAML is discarded. |

If you need to limit the run to specific GPUs, use `CUDA_VISIBLE_DEVICES`
before calling `launch.sh`, and pass the number of processes as the second
argument.

