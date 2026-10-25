# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from pathlib import Path

import torch

# Keys of the dict-format training checkpoint written by save_checkpoint().
# `model` holds the bare state dict; `optimizer`/`scaler` are optional.
CKPT_MODEL_KEY = 'model'
CKPT_OPTIMIZER_KEY = 'optimizer'
CKPT_SCALER_KEY = 'scaler'
CKPT_EPOCH_KEY = 'epoch'


def select_resume_checkpoint(model_dir, pinned_checkpoint=None):
    """显式文件优先；自动恢复按轮数选最新，不能按拷贝后的mtime倒退。"""
    if pinned_checkpoint is not None:
        path = Path(pinned_checkpoint)
        if not path.is_file():
            raise ValueError(f'explicit checkpoint is not a file: {path}')
        return str(path)
    root = Path(model_dir)
    if not root.exists():
        return None
    candidates = []
    for path in root.iterdir():
        found = re.fullmatch(r'model_([0-9]+)\.pt', path.name)
        if found and path.is_file():
            epoch = int(found[1])
            if epoch < 1:
                raise ValueError('checkpoint epoch must be positive')
            candidates.append((epoch, path))
    if not candidates:
        return None
    epochs = [epoch for epoch, _ in candidates]
    if len(epochs) != len(set(epochs)):
        raise ValueError('ambiguous checkpoint epoch names')
    return str(max(candidates, key=lambda item: item[0])[1])


def resume_start_epoch(checkpoint, extra, *, cross_stage=False):
    """核文件名与保存轮数；跨阶段重新从第1轮计，不继承主训练时钟。"""
    if type(cross_stage) is not bool:
        raise ValueError('cross_stage must be boolean')
    if checkpoint is None:
        return 1
    found = re.fullmatch(r'model_([0-9]+)\.pt', Path(checkpoint).name)
    named_epoch = int(found[1]) if found else None
    recorded_epoch = extra.get(CKPT_EPOCH_KEY)
    if named_epoch is not None and named_epoch < 1:
        raise ValueError('checkpoint filename epoch must be positive')
    if recorded_epoch is not None and (type(recorded_epoch) is not int or recorded_epoch < 1):
        raise ValueError('invalid checkpoint payload epoch')
    if named_epoch is not None and recorded_epoch is not None and named_epoch != recorded_epoch:
        raise ValueError('checkpoint filename and payload epoch differ')
    if cross_stage:
        return 1
    epoch = named_epoch if named_epoch is not None else recorded_epoch
    return 1 if epoch is None else epoch + 1


def validate_same_stage_resume(state_dict, model, checkpoint, speaker_map,
                               speaker_map_path=None):
    """恢复旧动量前，核完整模型集合与原说话人到分类行的映射；本函数不写文件。"""
    expected = model.state_dict()
    if set(state_dict) != set(expected) or any(
            not torch.is_tensor(state_dict[key]) or tuple(state_dict[key].shape) != tuple(value.shape)
            for key, value in expected.items()):
        raise ValueError('same-stage checkpoint model/head keys or shapes differ')
    path = Path(speaker_map_path) if speaker_map_path is not None else Path(checkpoint).parent.parent/'spk2id.json'
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate speaker-map key')
            result[key] = value
        return result
    if not path.is_file():
        raise ValueError('same-stage resume requires original spk2id.json or spk2id_map_path')
    old = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)
    if (not isinstance(old, dict) or any(type(v) is not int for v in old.values())
            or old != speaker_map):
        raise ValueError('same-stage speaker mapping differs; source evidence was not overwritten')
    return path


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def is_training_checkpoint(obj) -> bool:
    """True for the dict written by save_checkpoint(), False for a bare state dict."""
    if not isinstance(obj, dict):
        return False
    inner = obj.get(CKPT_MODEL_KEY)
    # torch.save() of a plain state dict has no 'model' key, so a bare state dict
    # fails this test and is loaded as-is.
    return isinstance(inner, dict) and all(
        isinstance(k, str) for k in inner.keys())


def extract_state_dict(checkpoint):
    """Return (state_dict, extra) from either checkpoint format.

    `extra` carries the optimizer/scaler/epoch that only the dict format has;
    it is an empty dict for a bare state-dict checkpoint.
    """
    if is_training_checkpoint(checkpoint):
        extra = {k: v for k, v in checkpoint.items() if k != CKPT_MODEL_KEY}
        return checkpoint[CKPT_MODEL_KEY], extra
    # Legacy / released-weight style: the file *is* the state dict.
    return checkpoint, {}


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict, _ = extract_state_dict(checkpoint)
    load_res = model.load_state_dict(state_dict, strict=False)
    print(f"LOAD RES : {load_res}")


def save_checkpoint(model: torch.nn.Module,
                    path: str,
                    optimizer=None,
                    scaler=None,
                    epoch: int = None):
    """Write a training checkpoint atomically.

    Passing no `optimizer`/`scaler`/`epoch` keeps the historical layout: the file
    is a bare state dict. With any of them the file becomes
    ``{'epoch':..., 'model': state_dict, 'optimizer':..., 'scaler':...}`` so an
    interrupted run can resume with its SGD momentum buffers and AMP GradScaler
    state intact. Readers accept both layouts (see extract_state_dict), which is
    what keeps the 20 published weight files and the existing bare checkpoints
    loadable.

    There is deliberately NO scheduler argument. The LR and margin schedules in
    utils/schedulers.py are plain classes, not `torch.optim.lr_scheduler`
    objects, and none of them defines `state_dict()`/`load_state_dict()` -- so
    asking one for its state would raise `AttributeError` at the first save.
    Nothing is lost: both schedules are driven purely by the absolute iteration
    `(epoch - 1) * epoch_iter + i`, and the starting epoch is checked against the
    `model_<epoch>.pt` filename and saved epoch, so they re-derive themselves on resume.
    """
    state_dict = _unwrap_model(model).state_dict()
    # Strip torch.compile's `_orig_mod.` prefix so checkpoints stay
    # transparent to whether the module was compiled at save time.
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

    if optimizer is None and scaler is None and epoch is None:
        payload = state_dict
    else:
        payload = {CKPT_MODEL_KEY: state_dict}
        if epoch is not None:
            payload[CKPT_EPOCH_KEY] = epoch
        if optimizer is not None:
            # The optimizer is saved as built here (plain torch.optim over the
            # raw module parameters), not in any post-DDP form. That is fine:
            # DDP only all-reduces gradients and never re-parameterises the
            # module, so the `state` dict is keyed against the same parameter
            # tensors either way.
            payload[CKPT_OPTIMIZER_KEY] = optimizer.state_dict()
        if scaler is not None:
            payload[CKPT_SCALER_KEY] = scaler.state_dict()

    # Write to a sibling temp file and rename into place. A direct torch.save()
    # can leave a truncated-but-present model_N.pt behind if training is killed
    # mid-write; that file also has the newest mtime, so the resume logic in
    # bin/train.py would pick it and crash inside torch.load. os.replace() is
    # atomic within a filesystem, so readers see either the old file or the
    # complete new one. The temp name must stay out of the model_(\d+).pt
    # pattern the resume scan matches on.
    tmp_path = "{}.tmp.{}".format(path, os.getpid())
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
