# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pooling functions to aggregate frame-level deep features
into segment-level speaker embeddings

High-order statistics are surprisingly effective, TSDP acts similarly as TSTP,
even though we remove the mean statistic, on Voxceleb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from redimnet2.layers.masking import as_mask_like


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        if x.ndim == 3:
            pooling_mean = x.mean(dim=-1)
        elif x.ndim == 4:
            pooling_mean = x.mean(dim=(-1,-2))
        # To be compatable with 2D input
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TMAP(nn.Module):
    """
    Temporal max-average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TMAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        if x.ndim == 3:
            (x1, _) = torch.max(x, dim=-1)
            x2 = torch.mean(x, dim=-1)
        elif x.ndim == 4:
            # (x1, _) = torch.max(x, dim=(-1,-2))
            x1 = torch.max(
                torch.max(x, dim=-1)[0],
            dim=-1)[0]
            x2 = torch.mean(x, dim=(-1,-2))
        max_avg = x1 + x2
        return max_avg

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim

class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSDP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim

class TSTP_var(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_var = torch.var(x, dim=-1)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_var = pooling_var.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_var), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim

class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    mask_aware = True

    def __init__(self,
                 in_dim,
                 bottleneck_dim=128,
                 global_context_att=False,
                 **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x, mask=None):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        mask: optional (B,1,T) or (B,T) frame mask, 1 on real frames.
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, 0]
            m = as_mask_like(x, mask)  # (B, 1, T) against (B, C, T)

        if self.global_context_att:
            if mask is None:
                context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
                context_std = torch.sqrt(
                    torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            else:
                # The context must be built from the real frames only, both
                # here and in the attention below.
                count = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
                mean = (x * m).sum(dim=-1, keepdim=True) / count
                var = (x * x * m).sum(dim=-1, keepdim=True) / count - mean * mean
                # 与无掩码那一支用同一个估计量。
                #
                # 上面无掩码用的是 `torch.var`，它默认是**无偏**的（除以 n-1）；
                # 而下面这段矩是除以 n 的**有偏**版本。掩码全有效时两条路必须
                # 给出同样的结果，否则"只是传了个 lengths"就会改变输出。
                # 实测（小随机输入）差异正好是 sqrt(n/(n-1)) 这个比值：
                # n=5 时 7.8e-03，n=17 时 1.4e-03，与两个估计量的理论比值一致。
                # 这里换算成无偏，让两条路对齐。
                var = var * count / (count - 1.0).clamp_min(1.0)
                context_mean = mean.expand_as(x)
                context_std = torch.sqrt(var.clamp_min(0.0) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        logits = self.linear2(alpha)
        if mask is not None:
            logits = logits.masked_fill(m[:, 0].unsqueeze(1) < 0.5, float('-inf'))
        alpha = torch.softmax(logits, dim=2)
        if mask is None:
            mean = torch.sum(alpha * x, dim=2)
            var = torch.sum(alpha * (x**2), dim=2) - mean**2
        else:
            # Padded frames have alpha == 0, but their feature values may be
            # arbitrary; restrict the moments to the real frames.
            alpha = alpha * m
            mean = torch.sum(alpha * x, dim=2)
            var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        if mask is None:
            return torch.cat([mean, std], dim=1)
        return torch.cat([mean, std], dim=1), mask

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MHASTP(torch.nn.Module):
    """ Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 head_num=2,
                 d_s=1,
                 bottleneck_dim=64,
                 **kwargs):
        super(MHASTP, self).__init__()
        assert (in_dim % head_num
                ) == 0  # make sure that head num can be divided by input_dim
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):
                att_trans.add_module(
                    'att_' + str(i),
                    nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module(
                'att_' + str(layer_num - 1),
                nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num],
                          1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        # split
        chunks_out = []
        # for i in range(self.head_num):
        #     att_score = self.heads_att_trans[i](chunks[i])
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i]**2, dim=2) - mean**2
            std = torch.sqrt(var.clamp(min=1e-7))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(torch.nn.Module):
    """ An attentive pooling
    Reference:
        multi query multi head attentive statistics pooling
        https://arxiv.org/pdf/2110.05042.pdf
    Args:
        in_dim: the feature dimension of input
        layer_num: the number of layer in the pooling layer
        query_num: the number of querys
        head_num: the number of heads
        bottleneck_dim: the bottleneck dimension

    SA (H = 1, Q = 1, n = 2, d_s = 1) ref:
        https://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    MHA (H > 1, Q = 1, n = 1, d_s = 1) ref:
        https://arxiv.org/pdf/1906.09890.pdf
    AS (H = 1, Q > 1, n = 2, d_s = 1) ref:
        https://arxiv.org/pdf/1803.10963.pdf
    VSA (H = 1, Q > 1, n = 2, d_s = d_h) ref:
        http://www.interspeech2020.org/uploadfile/pdf/Mon-2-10-5.pdf
    """

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 query_num=2,
                 head_num=8,
                 d_s=2,
                 bottleneck_dim=64,
                 **kwargs):
        super(MQMHASTP, self).__init__()
        self.n_query = nn.ModuleList([
            MHASTP(in_dim,
                   layer_num=layer_num,
                   head_num=head_num,
                   d_s=d_s,
                   bottleneck_dim=bottleneck_dim) for i in range(query_num)
        ])
        self.query_num = query_num
        self.in_dim = in_dim

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = torch.cat(res, dim=-1)
        return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim


if __name__ == '__main__':
    data = torch.randn(16, 512, 10, 35)
    # model = StatisticsPooling()
    model = MQMHASTP(512 * 10)
    model = MHASTP(512 * 10)
    model = MQMHASTP(512 * 10, context=False)
    print(model)

    out = model(data)
    print(out.shape)
    print(model.get_out_dim())
