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



### 维护版接续时的轮次与说话人映射检查

自动接续从`models/model_N.pt`中选轮数最大的完整候选文件，不按修改时间选；把第35轮重新复制进已有第42轮的目录，不会让训练退回第36轮。`checkpoint:`仍优先使用显式指定文件，允许明确指定较早检查点；这属于调用者有意选择，不是自动回退。临时文件、`best.pt`以及带其他前缀的名字不参与自动选择；同一轮出现`model_2.pt`和`model_02.pt`会因歧义被拒绝。文件是否完整仍由实际检查点读取判断，不因名字较新而跳过损坏文件。

同阶段恢复时，文件名轮数与保存内容中的`epoch`必须一致；重命名的训练检查点使用保存轮数恢复时钟。旧的纯权重文件没有优化器或缩放器记录，不能把它说成已经完整恢复了这两项状态。分类头形状改变的跨阶段初始化同时清空优化器、梯度缩放器和阶段时钟，从新阶段第1轮开始；`model_ckpt`仍表示只初始化模型的新阶段入口。

在同阶段恢复旧分类头和动量前，入口还核全部模型参数名、形状与原`spk2id.json`。说话人数相同但实际说话人或分类行编号改变，仍会拒绝。默认读取检查点所属实验目录的`spk2id.json`，移动过检查点时可用`spk2id_map_path`明确给出原映射；缺少映射不能猜。所有进程读完旧映射、模型和轮数之后，主进程才保存当前阶段的映射，避免提前覆盖来源证据。

这些是仓库维护版本的CPU回归结果，现有服务器任务的冻结源码、检查点和映射没有因此被修改，也没有继承新版本的GPU验收。
