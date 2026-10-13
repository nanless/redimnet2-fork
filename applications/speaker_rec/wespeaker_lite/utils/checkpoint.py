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

import torch

def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location='cpu')
    load_res = model.load_state_dict(checkpoint, strict=False)
    print(f"LOAD RES : {load_res}")

def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # Strip torch.compile's `_orig_mod.` prefix so checkpoints stay
    # transparent to whether the module was compiled at save time.
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    # Write to a sibling temp file and rename into place. A direct torch.save()
    # can leave a truncated-but-present model_N.pt behind if training is killed
    # mid-write; that file also has the newest mtime, so the resume logic in
    # bin/train.py would pick it and crash inside torch.load. os.replace() is
    # atomic within a filesystem, so readers see either the old file or the
    # complete new one. The temp name must stay out of the model_(\d+).pt
    # pattern the resume scan matches on.
    tmp_path = "{}.tmp.{}".format(path, os.getpid())
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
