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
  --rdzv-endpoint="${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29400}" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  applications/speaker_rec/wespeaker_lite/bin/train.py \
  --config "$CONFIG" "$@"
```

`MASTER_ADDR` and `MASTER_PORT` can be overridden in the environment. Training
auto-resumes from the newest `model_*.pt` in the configured `exp_dir/models/`.

> **Caveat for the large-margin fine-tuning stage.** The `lm` recipes set
> `model_ckpt`, which takes priority over the auto-detected checkpoint
> (`train.py`: `ckpt_to_load = configs.get('model_ckpt', None) or checkpoint`).
> So restarting an `lm` run **re-initializes backbone and projection head from
> the PTN weights again**; only the starting epoch number comes from the newest
> file. PTN runs do not set `model_ckpt` and resume normally. If an `lm` run is
> interrupted, prefer clearing `exp_dir` and starting over. See
> `docs/04-训练流程.md` section 6.
