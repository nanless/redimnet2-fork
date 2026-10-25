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

"""Helpers for carrying a validity mask through ReDimNet2.

A mask is a float tensor shaped ``(B, 1, T)``.  ``1`` marks a real frame (or a
real sample, for the sample-level mask at the very input), ``0`` marks the zero
padding used to make a batch rectangular.  ``(B, 1, T)`` broadcasts both against
1D feature maps ``(B, C, T)`` and against 2D ones ``(B, C, F, T)``.

补齐区域在每次混合时间位置的运算前必须归零，避免补零参与有效区间的统计。
这只是减少补齐污染的必要条件，不能保证补齐成批与逐条推理逐位相等：前端
取帧、卷积边界以及跨时间操作仍可能不同。MaskSequential让支持mask的模块
自行传递掩码，并在其他模块后归零；完整字质量和实际向量偏差需另做实验。
"""

import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_sample_mask(lengths, max_len, device=None, dtype=torch.float32):
    """Build a ``(B, 1, T)`` sample-level mask from per-utterance sample counts.

    ``lengths`` may be a tensor, a list or a numpy array.  Samples are assumed
    to be padded on the right, so entry ``(b, 0, t)`` is 1 for ``t < lengths[b]``.
    每项必须是1到max_len内的有限整数采样点数；布尔、分数、空批和多维输入
    不具有合法音频长度含义，不能先转long再静默截断。
    """
    if not isinstance(max_len, numbers.Integral) or isinstance(max_len, bool) or max_len <= 0:
        raise ValueError('max_len必须是正整数采样点数')
    if isinstance(lengths, (list, tuple)) and any(
        isinstance(value, (bool, np.bool_)) or
        (torch.is_tensor(value) and value.dtype == torch.bool)
        for value in lengths
    ):
        raise ValueError('真实长度不能是布尔值')
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths)
    if lengths.ndim != 1 or lengths.numel() == 0:
        raise ValueError('lengths必须是非空一维真实长度数组')
    if lengths.dtype == torch.bool or lengths.is_complex():
        raise ValueError('真实长度必须是有限整数采样点数')
    if not bool(torch.isfinite(lengths).all()):
        raise ValueError('真实长度包含非有限值')
    if lengths.is_floating_point() and not bool((lengths == lengths.round()).all()):
        raise ValueError('真实长度不能静默截断小数')
    lengths = lengths.to(device=device, dtype=torch.long)
    # 先统一整数存储类型，保留NumPy uint64等合法输入；超出int64的正数转换
    # 后为负数，会由下面的范围门拒绝，不能变成有效长度。
    if not bool(((lengths >= 1) & (lengths <= max_len)).all()):
        raise ValueError('真实长度必须在1到输入采样点数之间')
    idx = torch.arange(max_len, device=lengths.device)
    return (idx.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype).unsqueeze(1)


def sample_lengths(mask):
    """Recover per-utterance sample counts from a right-padded sample mask."""
    return mask.reshape(mask.size(0), -1).sum(dim=1).long()


def frame_count(lengths, win_length, hop_length, pad=None):
    """Frame count of a strided framing convolution.

    ``F.conv1d(x, kernel, stride=hop_length, padding=pad)`` returns
    ``(T + 2 * pad - win_length) // hop_length + 1`` frames for an input of
    ``T`` samples, so that is the number of frames a single, unpadded utterance
    of ``lengths`` samples produces.  The result is clamped to at least one so
    that degenerate masks do not make the pooling layer divide by zero.
    """
    if pad is None:
        pad = hop_length // 2
    n = (lengths + 2 * pad - win_length) // hop_length + 1
    return n.clamp_min(1)


def frame_mask_from_lengths(lengths, n_frames, win_length, hop_length, pad=None,
                            dtype=torch.float32):
    """Build a ``(B, 1, n_frames)`` frame-level mask from sample counts."""
    per_sample = frame_count(lengths, win_length, hop_length, pad=pad)
    idx = torch.arange(n_frames, device=per_sample.device)
    return (idx.unsqueeze(0) < per_sample.unsqueeze(1)).to(dtype).unsqueeze(1)


def frame_mask_from_sample_mask(sample_mask, win_length, hop_length, pad=None):
    """Turn a sample-level ``(B, 1, T)`` mask into a frame-level one.

    Convolution frame ``j`` covers the input positions
    ``[j * hop - pad, j * hop - pad + win_length)``.  A strict convolution only
    reads real positions, so frame ``j`` is reproducible from the real samples
    alone iff **every** position in that window is real; the "min over the
    window" below is exactly that test.  Positions that fall outside the signal
    are the convolution's own zero padding, which a lone utterance and a padded
    batch agree on, so they are treated as valid (the ``value=1.0`` above).

    有效帧的整个取样窗必须落在实际输入允许的位置。注意：单条前端的右侧
    卷积补齐，与批内显式补齐的无效区域不是同一标记；某些长度下此严格规则
    会比单条取帧公式少一个边界帧，不能承诺两条路径帧数或向量完全一致。
    """
    if pad is None:
        pad = hop_length // 2
    m = sample_mask.reshape(sample_mask.size(0), 1, sample_mask.size(-1))
    m = (m > 0.5).to(torch.float32)
    m_padded = F.pad(m, (pad, pad), mode='constant', value=1.0)
    # "-max(-x)" is a min filter, available through `max_pool1d` in the same
    # geometry as the framing convolution.
    return 1.0 - F.max_pool1d(1.0 - m_padded, kernel_size=win_length, stride=hop_length)



def ensure_nonempty(mask):
    """Force at least one valid position per utterance.

    Every mask in this file is a right-aligned prefix, so this only matters for
    utterances too short to produce a single valid frame; keeping the first
    frame valid avoids an all-masked softmax.
    """
    if mask.size(-1) == 0:
        return mask
    empty = mask.sum(dim=-1, keepdim=True) == 0
    ones = torch.zeros_like(mask)
    ones[..., 0:1] = 1.0
    return torch.where(empty, ones, mask)


def downsample_frame_mask(mask, stride):
    """Reduce a frame mask the way a stride-``stride`` time convolution does.

    A ``Conv`` with ``kernel_size = stride`` and no padding covers the input
    frames ``[j * stride, (j + 1) * stride)`` for output frame ``j``, so an
    output frame is valid only if all of its input frames are.
    """
    if stride == 1:
        return mask
    n_out = mask.size(-1) // stride
    if n_out == 0:
        return mask[..., :1]
    windowed = mask[..., :n_out * stride].reshape(mask.size(0), 1, n_out, stride)
    return windowed.amin(dim=-1)


def as_mask_like(x, mask):
    """Reshape a ``(B, 1, T)`` mask to broadcast against ``x``, which is
    ``(B, C, ..., T)``.  The intermediate axes become size 1 (so it also works
    for the 4D ``(B, C, F, T)`` tensors of the 2D stages)."""
    if mask is None:
        return None
    if x.ndim == mask.ndim:
        return mask
    return mask.reshape(mask.size(0), *([1] * (x.ndim - 2)), mask.size(-1))


def mask_mean(x, mask, dim):
    """Mean of ``x`` over ``dim`` counting only positions where ``mask`` is 1."""
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * mask).sum(dim=dim, keepdim=True) / count


def mask_var(x, mask, dim, mean=None):
    """Biased variance of ``x`` over ``dim`` counting only valid positions."""
    if mean is None:
        mean = mask_mean(x, mask, dim)
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * x * mask).sum(dim=dim, keepdim=True) / count - mean * mean


class MaskAware(nn.Module):
    """Base class for modules whose ``forward`` takes ``(x, mask=None)``.

    Subclasses return ``(out, mask)`` when a mask is supplied and ``out`` when
    it is not, which is what lets ``MaskSequential`` thread one through.
    """

    mask_aware = True


class MaskSequential(nn.Sequential):
    """An ``nn.Sequential`` that threads a frame mask through its children.

    With ``mask=None`` this is an ordinary ``nn.Sequential``, down to the order
    of operations.  With a mask, children that accept one update it, and every
    other child is followed by a re-zeroing of the padded frames so that the
    next time-mixing layer sees exactly zeros there.
    """

    def forward(self, x, mask=None):
        if mask is None:
            return super().forward(x)
        for module in self:
            if getattr(module, 'mask_aware', False):
                x, mask = module(x, mask)
            else:
                x = module(x) * as_mask_like(x, mask)
        return x, mask


class MaskDownConv2d(nn.Conv2d):
    """``nn.Conv2d`` that also reduces the frame mask when it strides in time.

    Used for the stage entry convolutions, whose kernel and stride are both
    ``(freq_stride, time_stride)``.  It is a plain ``nn.Conv2d`` subclass, so
    parameter names and checkpoint keys are unchanged.
    """

    mask_aware = True

    def forward(self, x, mask=None):
        if mask is None:
            return super().forward(x)
        # Re-zero before the convolution, then again with the reduced mask:
        # the second pass clears whatever it wrote into the frames it is about
        # to drop, so the two time branches of a dual stage agree.
        out = super().forward(rezero(x, mask))
        out_mask = ensure_nonempty(downsample_frame_mask(mask, self.stride[1]))
        return rezero(out, out_mask), out_mask


class MaskResample(nn.Module):
    """Frame resampler (``nn.Upsample``) that keeps the frame mask in step.

    ``mode='nearest'`` copies input frame ``j`` to output frames
    ``[j * factor, (j + 1) * factor)``, which is exactly ``repeat_interleave``;
    a fractional factor falls back to a nearest-neighbour mask lookup.
    """

    mask_aware = True

    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.up = nn.Upsample(scale_factor=factor, mode='nearest')

    def forward(self, x, mask=None):
        out = self.up(x)
        if mask is None:
            return out
        if float(self.factor).is_integer():
            return out, mask.repeat_interleave(int(self.factor), dim=-1)
        n_out = out.size(-1)
        idx = (torch.arange(n_out, device=mask.device) // self.factor).clamp_max(mask.size(-1) - 1)
        return out, mask.index_select(-1, idx)

    def extra_repr(self) -> str:
        return f"factor={self.factor}"


class MaskedGroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` whose statistics ignore the padded frames.

    ``nn.GroupNorm`` normalizes over both the channels of a group and the time
    axis, so on a padded batch the padded frames would otherwise shift the mean
    and variance of the real ones.  This is a plain ``nn.GroupNorm`` subclass,
    so parameter names and checkpoint keys are unchanged.
    """

    mask_aware = True

    def forward(self, x, mask=None):
        if mask is None:
            return super().forward(x)
        b, c, t = x.shape
        groups = self.num_groups
        # The channel axis is split into (group, per-group), so the mask needs a
        # singleton there: `(B, 1, 1, T)` broadcasts against `(B, G, C/G, T)`.
        m = as_mask_like(x, mask).to(x.dtype).reshape(b, 1, 1, t)
        per_group = c // groups
        xg = x.reshape(b, groups, per_group, t) * m
        # Number of real (group, time) cells: valid time positions * channels
        # per group.
        count = m.sum(dim=-1, keepdim=True) * per_group
        mean = xg.sum(dim=(2, 3), keepdim=True) / count
        var = xg.pow(2).sum(dim=(2, 3), keepdim=True) / count - mean * mean
        xn = ((xg - mean) / torch.sqrt(var.clamp_min(0.0) + self.eps)).reshape(b, c, t)
        out = xn * self.weight.reshape(1, c, 1) + self.bias.reshape(1, c, 1)
        return out * as_mask_like(x, mask), mask


def rezero(x, mask):
    """Zero the padded frames of ``x``.

    A convolution with ``padding='same'`` computes its boundary outputs using
    the positions just outside its input, and a lone utterance supplies zeros
    there.  For a padded batch to behave the same way, the padded region must
    read exactly zero *before* the next time-mixing layer runs.  Doing this
    after a single convolution is enough for the value at the real frames, but
    the padded frames themselves then hold non-zero junk, so a block with two
    stacked convolutions must re-zero in between -- which is why the blocks
    below call this between their own convolutions rather than once at the end.
    """
    if mask is None:
        return x
    return x * as_mask_like(x, mask)


def mask_aware_seq(layers):
    """Build a ``MaskSequential`` from plain layers.

    Only the reshape-only ``to1d`` / ``to2d`` children are wrapped; everything
    else is passed through unchanged, so the parameter names and checkpoint
    keys are identical to the ``nn.Sequential`` this replaces.
    """
    from redimnet2.layers.redim_structural import to1d, to2d
    return MaskSequential(*[
        _MaskPassthrough(layer) if type(layer) in (to1d, to2d) else layer
        for layer in layers
    ])


class _MaskPassthrough(nn.Module):
    """Reshape-only layer that forwards the frame mask untouched.

    Handles the `to1d` / `to2d` family: they reorder the frequency and channel
    axes into the feature axis but always leave time last, so the mask is
    unaffected and only its broadcast shape needs to be re-derived.
    """

    mask_aware = True

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, mask=None):
        out = self.inner(x)
        if mask is None:
            return out
        return out, mask
