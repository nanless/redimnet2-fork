# Launching Training

Run from the repository root. The launcher sets `PYTHONPATH` so the local
`redimnet2` package and `wespeaker_lite` training package are both importable.

```bash
./applications/speaker_rec/launch.sh \
  applications/speaker_rec/examples/vox2/recipes/redimnet2/b3/ptn.yaml \
  8 \
  --config_overrides /tmp/vox2-ptn.local.yaml
```

The second positional argument is the number of local GPU processes. It
defaults to `8`.

The launcher expands to:

```bash
export PYTHONPATH="$(pwd):$(pwd)/applications/speaker_rec:${PYTHONPATH}"
export LOGLEVEL=ERROR
export PYTHONFAULTHANDLER=1

torchrun --nnodes=1:1 --nproc_per_node="$NUM_GPUS" --rdzv-backend=static \
  --rdzv-endpoint="${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT}" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  applications/speaker_rec/wespeaker_lite/bin/train.py \
  --config "$CONFIG" "$@"
```

`MASTER_ADDR` and `MASTER_PORT` can be set in the environment. If `MASTER_PORT`
is not set, `launch.sh` picks a free port automatically, so two training runs on
the same machine no longer collide. **Multi-node runs must set `MASTER_PORT`
explicitly** — every node has to use the same one.

Training auto-resumes from the newest `model_*.pt` in the configured
`exp_dir/models/`.

The load order is `checkpoint` (a config key, unset in every recipe) →
newest file in `exp_dir/models/` → `model_ckpt`, first non-empty wins. So:

- **PTN** resumes from its own latest checkpoint, as expected.
- **LM fine-tuning** bootstraps from the PTN weights given by `model_ckpt` only
  on the first run, when `exp_dir/models/` is still empty; after an interruption
  it resumes from its own newest checkpoint instead of redoing the epochs.

> **To restart an LM run from scratch you must clear `exp_dir` first.** Because
> a checkpoint in `exp_dir/models/` outranks `model_ckpt`, editing `model_ckpt`
> alone has no effect — it is silently ignored while an old checkpoint is
> present. Use the `checkpoint:` config key to pin a specific starting file.
> See `docs/04-训练流程.md` section 6.
