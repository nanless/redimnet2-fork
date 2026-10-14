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

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F

from redimnet2.layers.masking import (
    MaskSequential,
    ensure_nonempty,
    frame_mask_from_sample_mask,
    mask_mean,
    mask_var,
)

class NormalizeAudio(nn.Module):
    """Per-utterance zero-mean unit-variance normalization of the waveform."""
    mask_aware = True

    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self,x,mask=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if mask is None:
            x = (x - x.mean(dim=2, keepdims=True)) /\
                     (x.std(dim=2, keepdims=True, unbiased=False) + self.eps)
            return x
        # Statistics come from the real samples only.
        m = mask.reshape(x.size(0), 1, x.size(-1)).to(x.dtype)
        mean = mask_mean(x, m, dim=2)
        var = mask_var(x, m, dim=2, mean=mean)
        return (x - mean) / (torch.sqrt(var.clamp_min(0.0)) + self.eps), mask

class PreEmphasis(torch.nn.Module):
    mask_aware = True

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x, mask=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        x = F.conv1d(x, self.flipped_filter).squeeze(1)
        if mask is None:
            return x
        # The leading sample is built by reflection from real data, not by
        # padding, so the mask passes through unchanged.
        return x, mask

class FbankAug(nn.Module):
    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10), freq_start_bin=0):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.freq_start_bin = freq_start_bin
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(self.freq_start_bin, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class MelBanks(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min = 20, 
        f_max = 7600, 
        n_mels=80, 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        spec_norm='mn',
        freq_start_bin = 0,
        num_apply_spec_aug = 1,
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
    ):
        super(MelBanks, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.hop_length = hop_length
        # torchaudio frames with `center=True`: it reflect-pads the signal by
        # `n_fft // 2` on both sides and windows with a `n_fft`-long, zero
        # padded window, so the frame count follows the `(T + 2 * pad - n_fft)`
        # rule with those two numbers.
        self.win_length = n_fft
        self.frame_pad = n_fft // 2
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, f_min=f_min, f_max=f_max, n_mels=n_mels,
            window_fn=torch.hamming_window)
        # A convolution front end (torchaudio >= 0.10) frames the signal with a
        # single strided conv and can update the frame mask with it.  The
        # legacy STFT-based one cannot, so there the caller supplies the frame
        # mask directly from the workload lengths.
        self.mel_is_conv = isinstance(mel_spec.spectrogram, nn.Conv1d)
        self.torchfbank = MaskSequential(
            NormalizeAudio() if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),
            mel_spec,
            )
        self.spec_norm = spec_norm
        if spec_norm == 'mn':
            def _mn(x, m=None):
                # Without a mask this is the plain per-utterance mean removal
                # the unmasked path has always used; `mask_mean` would need a
                # mask, so keep both entry points on one function.
                if m is None:
                    return x - torch.mean(x, dim=-1, keepdim=True)
                return x - mask_mean(x, m, dim=-1)
            self.spec_norm = _mn
        elif spec_norm == 'mvn':
            def _mvn(x, m=None):
                if m is None:
                    mean = torch.mean(x, dim=-1, keepdim=True)
                    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
                else:
                    mean = mask_mean(x, m, dim=-1)
                    var = mask_var(x, m, dim=-1, mean=mean)
                return (x - mean) / (torch.sqrt(var.clamp_min(0.0)) + 1e-8)
            self.spec_norm = _mvn
        elif spec_norm == 'bn':
            self.spec_norm = nn.BatchNorm1d(n_mels)
        else:
            self.spec_norm = None
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width) # Spec augmentation
        else:
            self.specaug = nn.Identity()


    def forward(self, x, mask=None):
        xdtype = x.dtype
        x = x.float()
        base_mask = mask
        with torch.no_grad(), torch.amp.autocast(x.device.type, enabled=False):
            if mask is None:
                x = self.torchfbank(x)+1e-6
                x = x.log()
                # print(f"spec : {x.size()}")
                # x = x - torch.mean(x, dim=-1, keepdim=True)
                x = self.spec_norm(x)
            else:
                if not self.mel_is_conv:
                    # Legacy STFT-based torchaudio front end: framing happens
                    # inside MelSpectrogram, which cannot carry a mask, so run
                    # the front end unmasked and derive the frame mask
                    # separately with the same geometry (center=True
                    # reflect-pads by n_fft // 2, hence win_length == n_fft and
                    # pad == n_fft // 2).
                    x = self.torchfbank(x) + 1e-6
                    x = x.log()
                    frame_mask = frame_mask_from_sample_mask(
                        mask, self.win_length, self.hop_length,
                        pad=self.frame_pad)
                    # 极短音频会得到全 0 的帧掩码，ASTP 的 softmax 会因此
                    # 产生 NaN 并在训练模式下扩散到整批；保证至少留一帧有效。
                    frame_mask = ensure_nonempty(frame_mask)
                    x = x * frame_mask
                else:
                    x, frame_mask = self.torchfbank(x, mask)
                    x = torch.log(x + 1e-6) * frame_mask
                if self.spec_norm is not None:
                    x = self.spec_norm(x, frame_mask)
            if self.training:
                for _ in range(self.num_apply_spec_aug):
                    x = self.specaug(x)
        x = x.to(xdtype)
        if base_mask is None:
            return x
        return x, frame_mask.to(xdtype)