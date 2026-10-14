# MIT License
# 
# Copyright (c) 2024 ID R&D, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import functools
import numpy as np
import torch.nn as nn
from typing import List
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Iterable, Optional

class LayerNorm(nn.Module): # ⚡
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, T, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, T).
    """
    def __init__(self, C, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.C = (C, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            w = self.weight
            b = self.bias
            for _ in range(x.ndim-2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b # ⚡
            return x
        
    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k,v in {"C" : self.C, "data_format" : self.data_format, "eps" : self.eps}.items()])


class MaskedLayerNorm(LayerNorm):
    """``LayerNorm`` whose `channels_first` statistics ignore padded frames.

    In that mode `LayerNorm` reduces over both the channel and the time axes,
    so on a padded batch the padded frames would otherwise shift the statistics
    of the real ones.  This is a plain `LayerNorm` subclass, so parameter names
    and checkpoint keys are unchanged.
    """

    mask_aware = True

    def forward(self, x, mask=None):
        if mask is None or self.data_format != "channels_first":
            if mask is None:
                return super().forward(x)
            return super().forward(x), mask
        # x is (B, C, ..., T); reduce over every axis except the batch one.
        from redimnet2.layers.masking import as_mask_like
        m = as_mask_like(x, mask)
        dims = tuple(range(1, x.ndim))
        count = m.sum(dim=dims, keepdim=True).clamp_min(1.0)
        u = (x * m).sum(dim=dims, keepdim=True) / count
        s = ((x - u) ** 2 * m).sum(dim=dims, keepdim=True) / count
        x = (x - u) / torch.sqrt(s + self.eps)
        w = self.weight
        b = self.bias
        for _ in range(x.ndim - 2):
            w = w.unsqueeze(-1)
            b = b.unsqueeze(-1)
        x = w * x + b
        return x * m, mask
