# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
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

import io
# import kaldiio
import json
import sys
import resampy
import warnings
import logging
import random
import traceback
import tarfile
import soundfile as sf
from pathlib import Path
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from contextlib import nullcontext

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

# 判定时间是否越界时用的容差（秒）。分段文件里的时间是"起点 + 时长"加出来的，
# 末段的终点常比文件末尾多出一个浮点尾数（实测差 5e-16），不留容差就会把这种
# 数值毛毛雨当成真的越界。
EDGE_EPS = 1e-6

# ── dropped-sample counters ──────────────────────────────────────────
# Samples dropped by the readers below used to be invisible: the only signal
# was a `logging.warning`, and bin/train.py raises the root logger's level to
# CRITICAL, which suppresses WARNING records entirely. A data.list / spk2id
# mismatch or a corrupt audio file therefore shrank the training set with no
# trace. These counters make every drop observable: the first few of each kind
# are printed to stderr, and report_drop_counters() prints a summary. stderr is
# not filtered by the logging level, so this survives train.py's CRITICAL.
# Counters are per-process, so with num_workers>0 each worker reports its own.
DROP_COUNTERS = {
    'short_audio': 0,       # read returned None (clip shorter than min_duration)
    'read_failed': 0,       # exception while reading/decoding the audio
    'vad_unsupported': 0,   # line carries a 'vad' key this reader cannot use
    'unknown_spk': 0,       # spk missing from spk2id (would have been label -1)
    'vad_no_rttm': 0,       # 按语音活动边界取样，但旁边没有分段文件（走回退策略）
    'vad_bad_rttm': 0,      # 分段文件在，但读不出可用区间（走回退策略）
    'vad_low_speech': 0,    # 选出的片段里真正的语音太少，丢掉或修理过
    # 下面是"被切开/被丢掉"的计数。vad_cut_inside 比较特别：样本**没有**被
    # 丢掉，只是切在了语音中间，所以它不进 DROP_COUNTERS 的汇总口径——
    # 放在这里是为了让 print 出来的前几条和别的丢弃一样可见。
    'vad_cut_inside': 0,    # 单个语音段比 max_sec 还长，只能在段内部切开
}

# Print at most this many occurrences of each kind, to surface a problem
# without flooding stderr when a data list is systematically broken.
DROP_PRINT_LIMIT = 5


def _drop_counter_incr(name, detail=None):
    """Count one dropped sample and print the first few occurrences to stderr."""
    DROP_COUNTERS[name] = DROP_COUNTERS.get(name, 0) + 1
    if DROP_COUNTERS[name] <= DROP_PRINT_LIMIT:
        msg = f"[dataset][DROP:{name}]"
        if detail is not None:
            msg += f" {detail}"
        print(msg, file=sys.stderr, flush=True)


def report_drop_counters(prefix="[dataset] dropped-sample summary"):
    """Print the accumulated drop counters to stderr (visible at any log level).

    只负责本进程的计数。开了 num_workers 之后每个 worker 是独立进程，
    它自己的计数不会汇总到主进程；所以主进程这边调用时可能一条都不打
    （这也是为什么调用方还要另外核对"一个 epoch 实际读到了多少条样本"）。
    """
    active = {k: v for k, v in DROP_COUNTERS.items() if v}
    if not active:
        return
    parts = ", ".join(f"{k}={v}" for k, v in sorted(active.items()))
    print(f"{prefix}（本进程 / 本 worker）: {parts}", file=sys.stderr, flush=True)


# ── 按语音活动（VAD）边界取片段 ──────────────────────────────────────
#
# 固定长度那条路径是「从文件里随机位置切同样长的一段」，切点落在哪完全随机，
# 经常把一句话从中间切断。这里改成：先读该音频旁边的分段文件，拿到一串
# 「哪里在说话」的时间区间（下称语音段），再**从段边界到段边界**挑一段
# 长度合适的片段。这样送进模型的片段要么是一整段语音，要么是连着的好几段，
# 不会把一个字切成两半。
#
# 分段文件用 NIST RTTM 文本格式，只写语音段：
#
#     SPEAKER <文件编号> 1 <起点秒> <时长秒> <NA> <NA> <NA> <NA> <NA>
#
# 时间都是秒、保留三位小数。默认文件名是音频文件名后面直接补 ".rttm"
# （`00001.wav` 对应 `00001.wav.rttm`），后缀可以用 `rttm_suffix` 改；
# 分段也可以放在另一棵目录树里，规则见 dataset.py 的 `rttm_path_for_audio`。

def report_vad_stats(prefix="[dataset] VAD 取样统计"):
    """把按语音活动取样的计数打到 stderr（不受日志级别影响）。

    计数是**按进程**的：开了 num_workers 之后每个 worker 各算各的，主进程这边
    通常一条都看不到。没开这个功能时字典是空的，这里不打印任何东西。
    """
    active = {k: v for k, v in VAD_STATS.items() if v}
    if not active:
        return
    parts = ", ".join(f"{k}={v}" for k, v in sorted(active.items()))
    print(f"{prefix}（本进程 / 本 worker）: {parts}", file=sys.stderr, flush=True)


def parse_rttm_segments(path):
    """读分段文件，返回按起点排好序的 [(起点秒, 终点秒), ...]。

    比直接 `split()` 宽松：空行跳过，字段不够的行跳过，时间读不出来的行跳过，
    段尾用起点加时长算出来（RTTM 的时长可以写成负数表示到文件末尾，这里按
    0 处理）。文件不存在返回 None，表示「没有分段可用」，调用方据此走回退策略；
    文件在但一行都没读出来，返回空列表。
    """
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except OSError:
        return None
    segs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 只认语音段：字段数够，且第 1 列是 SPEAKER（SPEECH 之类同样接受，
        # 因为它们的时间字段位置一致，都是"第 4 列起点、第 5 列时长"）。
        if len(parts) < 5:
            continue
        if parts[0].upper() not in ('SPEAKER', 'SPEECH'):
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        if dur < 0:
            dur = 0.0
        if start < 0:
            continue
        end = start + dur
        if end <= start:
            continue
        segs.append((start, end))
    segs.sort(key=lambda s: (s[0], s[1]))
    return segs


def merge_segments(segments, gap_sec=0.0):
    """把挨得太近或互相重叠的语音段并成一段。

    `gap_sec` 是允许并起来的最大静音间隔：两段之间的空隙小于等于它就合成
    一段（0 表示只并重叠或首尾相接的）。并起来之后，选片段就不会落在
    「其实不算停顿」的缝隙里。
    """
    if not segments:
        return []
    out = [list(segments[0])]
    for start, end in segments[1:]:
        if start - out[-1][1] <= gap_sec:
            if end > out[-1][1]:
                out[-1][1] = end
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def _speech_overlap_sec(choices, start, end):
    """片段 [start, end] 与哪些语音段有交集，交集一共多少秒。"""
    total = 0.0
    for s, e in choices:
        lo = max(s, start)
        hi = min(e, end)
        if hi > lo:
            total += hi - lo
    return total


def select_vad_span(segments,
                    min_sec,
                    max_sec,
                    target_sec=None,
                    pad_sec=0.0,
                    merge_gap_sec=0.0,
                    cut_policy='middle',
                    min_speech_sec=None,
                    rng=random,
                    clip_duration=None,
                    stats=None):
    """从语音段列表里挑一段「整段语音」的区间。

    返回字典：`start` / `end`（秒）、`mode`、`speech_sec`、`covered`、`aligned`。
    挑不出来返回 None。

    规则（按顺序）：

    1. **先合并**间隔不超过 `merge_gap_sec` 的相邻语音段（默认 0，只并重叠或
       首尾相接的）。合并后的区间就是"可选的整段"，片段的起点终点只能落在
       它们的边界上。
    2. **在所有连续几段的组合里**，取总时长落在 `[min_sec, max_sec]` 且最接近
       `target_sec`（默认取 `(min_sec + max_sec) / 2`）的那一个。也就是"每段
       都是整段"，绝不从中间切断；同一层里有多个一样接近的起点时随机取一个。
    3. **只能在段内部切开时**（单独一段就超过 `max_sec`）：按 `cut_policy`
       取这一段的中间或随机一小段，并记 `mode='cut_inside'`，让调用方计数。
       这是唯一一种会切到语音中间的情况。
    4. **两端各留 `pad_sec` 的余量**，再裁到整条音频的范围里。余量是对称加的，
       所以两头要么都完整加上、要么一起被截短，加完之后**仍然以语音段边界为
       中心**（`aligned=True` 表示两头的位移相等，即对称性没被破坏）。
    5. 片段里的真实语音少于 `min_speech_sec` 就返回 None，由调用方按
       `low_speech_policy` 丢掉或修理。注意 `pad_sec` 加出来的余量**不算**
       语音，它只是给分段器留的余量。

    > 为什么第 1 步的 `merge_gap_sec` 不跟着 `pad_sec` 走：余量是**对称地往
    > 两头扩**的，所以它只覆盖片段两端以外的范围，**不会**把片段内部那条静音
    > 缝隙补上；第 2 步选中的区间若横跨缝隙，扩充之后缝隙仍在正中间。
    > 早先的写法让 `merge_gap_sec` 默认等于 `pad_sec`，于是「扩完余量就盖住了
    > 缝隙」的区间被并成一段，随后又按"不切段"的规则把起点放在合并后区间的
    > 左端——而它中间那条缝隙其实还在。实测 `id03722/.../00011.wav` 就是
    > 这样：选出 (2.060, 5.888)，而那里的语音段是 (2.26,3.36) 与
    > (3.68,5.888)，区间正中躺着一整段 0.32 秒的静音，起点也不在任何语音段
    > 的边界上。

    Args:
        segments: [(起点秒, 终点秒), ...]，`parse_rttm_segments` 的输出
        min_sec / max_sec: 片段允许的时长范围
        target_sec: 想要的时长，默认取上下限中点
        pad_sec: 两端各留的余量秒数（可为 0）
        cut_policy: 'middle' 取中段，'random' 随机取一小段
        min_speech_sec: 片段内真实语音的下限，None 表示不检查
        rng: 随机数来源（测试时可注入固定种子）
        clip_duration: 整条音频的时长（秒）；给了就把片段裁进这个范围
        stats: 可选字典，用来累加这次挑选的计数（mode / cut_inside 等）

    Returns:
        dict 或 None
    """
    segments = [s for s in segments if s[1] > s[0]]
    if not segments:
        return None
    if target_sec is None:
        target_sec = 0.5 * (min_sec + max_sec)
    target_sec = min(max(target_sec, min_sec), max_sec)
    if min_speech_sec is None:
        min_speech_sec = min_sec
    if cut_policy not in ('middle', 'random'):
        raise ValueError(f"unknown cut_policy: {cut_policy!r}")

    # 1. 合并靠得很近的段。这里的间隔阈值**不跟 pad_sec 走**：
    # 余量是往两头对称扩的，不会补上片段内部那条缝隙，所以"扩完余量才连起来"
    # 的区间不能被并成一段（详见上面文档说明里的实测例子）。
    if merge_gap_sec is None:
        merge_gap_sec = 0.0
    choices = merge_segments(segments, gap_sec=max(0.0, merge_gap_sec))
    if not choices:
        return None

    n = len(choices)
    durs = [e - s for s, e in choices]

    # 2. 把连续的若干个**整段**并起来，凑到最接近 `target_sec` 的那个区间。
    #
    #    这里要遍历所有可能的合并段数（1 段、2 段、3 段……），在全部合法区间里
    #    取与 `target_sec` 最接近的那个，而不是"能用 1 段就停"。
    #    两者差别很大：假设 4 段语音各约 0.7~1.4 秒，目标 3.0 秒，
    #    "能用 1 段就停"只会给出 1.4 秒（差 1.6 秒），
    #    而把相邻两段并起来能得到 2.6 或 3.4 秒（差 0.4 秒）——
    #    后者才是"凑到目标时长、且每段都是整段"的本意。
    cut_source_sec = None   # 只有 cut_inside 才会填：被切的那一整段有多长
    best = None          # (与目标的差距, [平手项列表])
    max_merge = max(1, min(n, 32))
    # 平手要用容差判断，不能直接比浮点数是否相等：段的时间是「起点+时长」算出来的，
    # 0.0+1.2 与 2.6-1.4 的表示可能差一个 ulp（实测一处是 0.0，另一处是 2.22e-16）。
    # 按精确相等挑平手，会让本该并列的第四段永远轮不到，等于悄悄偏向靠前的语音段：
    # 实测四段各 1.2 秒、target=1.2 时，精确比较只有前两段能轮到（60 个种子下
    # 只出现 (0.0,1.2) 和 (1.3,2.5) 两个结果），加容差后四个平手项都能轮到。
    tol = max(1e-9, 1e-3 * max(target_sec, 1e-9))
    for n_merge in range(1, max_merge + 1):
        spans = []
        for i in range(0, n - n_merge + 1):
            j = i + n_merge - 1
            dur = choices[j][1] - choices[i][0]
            if dur < min_sec or dur > max_sec:
                continue
            spans.append((abs(dur - target_sec), i, j))
        if not spans:
            continue
        layer_best = min(s[0] for s in spans)
        ties = [s for s in spans if s[0] <= layer_best + tol]
        # 所有层一起比，取全局最接近的；同分时把各层的平手项并到一起。
        if best is None or layer_best < best[0] - tol:
            best = (layer_best, ties)
        elif abs(layer_best - best[0]) <= tol:
            best = (best[0], best[1] + ties)

    if best is not None:
        _, ties = best
        _, i, j = ties[rng.randrange(len(ties))]
        start, end = choices[i][0], choices[j][1]
        mode = 'merge' if j > i else 'single'
    else:
        # 3. 没有任何组合能落进 [min_sec, max_sec]：只能从某一段内部切。
        order = sorted(range(n), key=lambda k: durs[k])
        k = order[-1]              # 最长的一段
        if durs[k] < min_sec:
            return None            # 整条音频的语音加起来都不够短
        seg_start, seg_end = choices[k]
        mode = 'cut_inside'
        # 记下**被切的那一整段**有多长。注意不要拿切完之后的长度去说事：
        # 切出来当然是 target_sec 那么长，拿它去和 max_sec 比会得出
        # 「2.251s > 6.000s」这种自相矛盾的话。
        cut_source_sec = durs[k]
        if cut_policy == 'middle':
            span = min(target_sec, durs[k])
            start = seg_start + 0.5 * (durs[k] - span)
        else:
            span = min(target_sec, durs[k])
            start = seg_start + rng.uniform(0.0, durs[k] - span)
        end = start + span
        # 这里**不**单独累加 cut_inside：函数末尾会统一按 `mode` 加一次
        # （`stats[mode] += 1`，而 mode 就是 'cut_inside'）。
        # 两处都加会把一个样本算成 2 个 —— 实测踩过这个坑：
        # 单个段内切样本让统计变成 {'cut_inside': 2}，
        # 于是"段内切占多少比例"虚报成两倍。

    # 记下加余量之前的区间：它就是「算出来的语音边界」，检查/排查时看这个。
    raw_start, raw_end = start, end

    # 4. 两端留余量，再落到音频范围里。规则（这就是 `aligned` 的含义）：
    #
    #    * **能加就两端都加满** `pad_sec`，此时
    #      `start == raw_start - pad_sec` 且 `end == raw_end + pad_sec`，
    #      片段以语音段边界为中心（`aligned=True`）。
    #    * 文件长度不够、某一头放不下时，**片段整体平移**，让放不下的那一头
    #      贴住文件边缘，另一头的余量原样保留，片段长度不变。此时不再是
    #      "以边界为中心"，记 `aligned=False`；但片段仍然完整覆盖那些语音段。
    #    * 只有片段比整条音频还长，才真的夹短，也记 `aligned=False`。
    #
    #    为什么不做"两头各缩一半"的对称补偿：那会缩短片段，而且实测会把两个
    #    边界**同时**推离语音段边界（片段最短只剩 3.5 秒，比目标短一截）。
    #    平移则保证时长不受影响。
    #
    #    判定"有没有超出"要用一点容差 `EDGE_EPS`：分段文件里的时间是
    #    「起点 + 时长」加出来的，末段的终点常落在文件末尾再往后一点点
    #    （实测 `1.030 + 4.538 = 5.5680000000000005`，而文件正好 5.568 秒）。
    aligned = True
    if pad_sec > 0:
        start -= pad_sec
        end += pad_sec
        if clip_duration is not None:
            # 整段平移，让放不下的那一头贴住文件边缘。
            #
            # **但不能因此把另一头的余量撑大。** 平移会把起点（或终点）
            # 推到原来的语音边界之外，那一头就多出来一截不是语音的余量。
            # 实测过：分段 [(0.0,7.640),(7.890,9.984)]、音频 9.984 秒、pad=0.2 时，
            # 右端放不下 -> 整体左移 0.200 -> 左余量变成 0.400（请求的两倍），
            # 而右余量变成 0。也就是说"给两端各留一点余量"这个承诺没兑现，
            # 左边白送了一倍的静音。修法：平移之后把多出来的那头夹回去，
            # 让实际余量不超过请求的 `pad_sec`。长度会因此变短一点，
            # 这是有意的取舍 —— 宁可短一点，也不要悄悄多塞静音进去。
            if start < -EDGE_EPS and end <= clip_duration + EDGE_EPS:
                end += -start                  # 左端放不下：整体右移
                start = 0.0
                aligned = False
            elif end > clip_duration + EDGE_EPS and start >= -EDGE_EPS:
                start -= end - clip_duration   # 右端放不下：整体左移
                end = clip_duration
            # 不管走了哪条分支，最后都把两头夹进"原始区间 ± pad_sec"里，
            # 保证任何一头的余量都不会超过请求值。
            if raw_start - pad_sec > start:
                start = max(0.0, min(raw_start - pad_sec, clip_duration))
            if raw_end + pad_sec < end:
                end = max(0.0, min(raw_end + pad_sec, clip_duration))
                aligned = False
            elif start < -EDGE_EPS and end > clip_duration + EDGE_EPS:
                aligned = False                # 两头都放不下：只能夹短
            start = max(0.0, min(start, clip_duration))
            end = max(0.0, min(end, clip_duration))
        elif start < 0.0:
            start = 0.0
    elif clip_duration is not None:
        start = max(0.0, min(start, clip_duration))
        end = max(0.0, min(end, clip_duration))
    if end <= start:
        return None

    # 5. 语音占比检查（余量不算语音）。
    overlap = _speech_overlap_sec(choices, start, end)
    if overlap + 1e-9 < min_speech_sec:
        if stats is not None:
            stats['low_speech'] = stats.get('low_speech', 0) + 1
        return None

    if stats is not None:
        stats[mode] = stats.get(mode, 0) + 1

    return dict(start=start,
                end=end,
                mode=mode,
                speech_sec=overlap,
                cut_source_sec=cut_source_sec,
                covered=(raw_start, raw_end),
                aligned=aligned)


def _equal_spans(choices, target_sec, min_sec, max_sec):
    """列出所有合法区间的 (与目标差距, 起点下标, 终点下标)，供随机取平手用。

    只给测试和排查用；真正的挑选走 `select_vad_span` 里那层「按段数从少到多」
    的循环，逻辑一致但不用把所有区间都存下来。
    """
    n = len(choices)
    out = []
    max_merge = max(1, min(n, 32))
    for n_merge in range(1, max_merge + 1):
        for i in range(0, n - n_merge + 1):
            j = i + n_merge - 1
            dur = choices[j][1] - choices[i][0]
            if dur < min_sec or dur > max_sec:
                continue
            out.append((abs(dur - target_sec), i, j))
    return out








def read_audio_section(fp,
        samplerate,
        duration=3.0,
        min_duration=1.0,
        start_time=None,
        extend_to_duration=False,
    ):
    # Read `duration` seconds starting at `start_time` and return them at
    # `samplerate`, resampling on the fly if the file differs. The returned
    # rate is always `samplerate`.
    #
    # extend_to_duration=True (the fixed-length training path): the requested
    # window is chunk_len * 1.1 (the caller's speed-perturb margin), and for
    # clips shorter than that the tail is looped -- np.pad(..., 'wrap') -- until
    # the window is full. How often that triggers depends on the chunk window:
    # VoxCeleb2 clips are at least 3.97 s, so a ptn window (2.02 s, target
    # 2.22 s) never wraps (0%), while an lm window (6.02 s, target 6.62 s)
    # wraps on ~61% of clips, duplicating ~24% of the samples on those clips.
    # Measured on VoxCeleb2 dev, stratified across speakers (every 5th-20th
    # speaker, 2-3 clips each, ~900-2400 clips). Stratification matters:
    # data.list is grouped by speaker and clip length varies by speaker, so a
    # contiguous first-N-lines slice under-reports (~58%). Note P(wrap) equals
    # the share of clips shorter than the 6.62 s target (once the clip is at
    # least that long the uniform start can no longer run off the end); that
    # share is ~61% and the bare 6.02 s window is ~52-56%.
    # WeSpeaker upstream behaves the same way; this is deliberate and must not
    # be turned into zeros or dropped, as that would change training.
    #
    # duration=None plus extend_to_duration=False (the variable-length training
    # path) is the escape hatch from that looped padding: read the clip's own
    # audio and return it at whatever length it happens to have, so a short
    # utterance stays short instead of being stretched with copies of itself.
    # Callers on that path must pass an explicit start_time, since there is no
    # window to pick a random offset within.
    #
    # No Path(fp).exists() pre-check: sf.SoundFile already raises on a missing
    # file (callers handle that identically), and the explicit stat() was a
    # redundant virtiofs round-trip (~0.65 ms/sample, ~2 stat calls/sample).
    f = sf.SoundFile(fp)

    tar_dur = duration

    can_seek = f.seekable() # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = f.samplerate
    ns = f.frames
    tot_dur = ns / sr

    if min_duration > tot_dur:
        return None, None

    if start_time is None:
        rnd_start_range = max(0,tot_dur-tar_dur)
        t = np.random.uniform()
        start_time = rnd_start_range*t
    start_frame = int(sr * start_time)
    # print(f"start_frame : {start_frame}, tot len : {ns}")

    # Number of *source* frames to read, expressed at the file's own rate.
    if duration is not None:
        src_frames_to_read = int(sr * duration)
    else:
        src_frames_to_read = ns

    f.seek(start_frame)

    samples = f.read(src_frames_to_read)

    if sr != samplerate:
        # warnings.warn(f"Runtime resampling : {fp}")
        samples = resampy.resample(samples, sr, samplerate)

    # `samples` is now at `samplerate`, so the output length is counted at the
    # target rate. This was previously the same name as the source-frame count
    # above while meaning a different unit, which is what let the stale `sr`
    # slip into the return value.
    #
    # duration=None (the variable-length path) means "keep whatever the clip
    # has", so there is no output length to count towards and nothing to pad:
    # `out_frames` becomes the number of samples actually read, which makes the
    # two lines below no-ops. They are still executed rather than branched
    # around so the fixed-length path keeps exactly the same sequence of
    # operations it had before this escape hatch existed.
    out_frames = len(samples) if duration is None else int(samplerate * duration)
    if extend_to_duration:
        shortage = out_frames - len(samples)
        if shortage > 0:
            samples = np.pad(samples, (0, shortage), 'wrap')

    samples = samples[:out_frames]

    # `samples` are at `samplerate`; reporting the file's own rate here made
    # processor.resample() resample a second time (44.1 kHz inputs kept only
    # 16/44.1 of their length and were pitched up by the same factor).
    return samples, samplerate

def parse_raw_fast_and_filter(
        data,
        chunk_len,
        data_type='shard/raw/feat',
        min_num_frames=100,
        max_num_frames=800,
        frame_shift=10,
    ):
    """ Parse key/wav/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/spk

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    assert data_type == 'raw_fast'

    for sample in data:
        # Read example & audio
        assert 'src' in sample
        json_line = sample['src']
        try:
            obj = json.loads(json_line)
        except Exception as e:
            logging.warning(f"Failed to parse json line: {json_line}")
            raise e
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        try:
            sample_rate = 16_000

            # Only a lower bound on utterance length is enforced, through
            # read_audio_section's min_duration: clips shorter than
            # min_num_frames frames (100 frames * 10 ms = 1.0 s by default) are
            # dropped. `max_num_frames` used to be turned into a `max_len` here
            # and then never read, so long clips were never discarded despite
            # the parameter name. Reinstating an upper bound would drop 29.7%
            # of VoxCeleb2 dev clips (the 8.0 s implied by the default 800),
            # which is a training-data change, not a bug fix; the parameter is
            # therefore kept for signature compatibility but left unused.
            min_len = int(frame_shift / 1000 * min_num_frames * sample_rate)

            ext_spd_pert_dur = (chunk_len/sample_rate) * 1.1
            # Keep the requested minimum duration: read_audio_section returns
            # (None, None) for a too-short clip and would otherwise clobber
            # `sample_rate` with None, which the drop message below needs.
            min_duration_sec = min_len/sample_rate

            waveform, sample_rate = read_audio_section(wav_file,
                        samplerate=sample_rate,
                        min_duration=min_duration_sec,
                        duration=ext_spd_pert_dur, # Extended for speed perturb
                        start_time=None,
                        extend_to_duration=True)

            if waveform is None:
                # Clip shorter than min_duration: dropped. print() to stderr, not
                # logging.warning, because train.py pins the root logger to
                # CRITICAL and would swallow the warning.
                _drop_counter_incr(
                    'short_audio',
                    f"shorter than {min_duration_sec:.2f}s: {obj}")
                continue

            waveform = waveform[:int(ext_spd_pert_dur*sample_rate)]

            waveform = torch.from_numpy(waveform).unsqueeze(0).float()

            if 'vad' in obj:
                # VAD-segmented shard lines are not supported by this reader.
                # Raise a real exception: `NotImplemented` is a sentinel object,
                # not a class, so `raise NotImplemented()` raised a confusing
                # TypeError("'NotImplementedType' object is not callable").
                # The sample is still skipped by the handler below, matching the
                # other per-sample failures (missing file, bad json); this
                # reader never emits `vad` keys itself, so a data list that
                # carries them is being read by the wrong pipeline and the
                # per-line skip is the intended degradation.
                raise NotImplementedError(
                    "'vad' segments are not supported by this reader")

            sample = dict(key=key,
                           spk=spk,
                           path=wav_file,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield sample
        except Exception as ex:
            traceback.print_exc()
            # Also count it: same reason as above, a logging.warning is
            # suppressed by train.py's CRITICAL root level.
            _drop_counter_incr(
                'read_failed',
                f"{type(ex).__name__}: {ex} | {obj}")
            logging.warning('Failed to read & filter {}'.format(str(obj)))


def parse_raw_fast_varlen(
        data,
        min_sec=1.0,
        max_sec=8.0,
        data_type='raw_fast',
        min_num_frames=100,
        frame_shift=10,
    ):
    """ Read a clip's own audio for the variable-length training path.

        This is the variable-length sibling of `parse_raw_fast_and_filter`.
        The fixed-length path reads a window of `chunk_len * 1.1` seconds and
        loops the tail of short clips to fill it, so every sample leaves this
        stage with the same number of samples; this one instead returns each
        clip at whatever length it really has, and the batch is zero padded
        later by `collate_varlen`.

        Clips longer than `max_sec` are trimmed to a random `max_sec`-long
        window, so the length distribution is capped but no sample is stretched
        or dropped outright. Clips shorter than `min_sec` are dropped.

        Args:
            data: Iterable[str], str is a json line has key/wav/spk
            min_sec: clips shorter than this are dropped
            max_sec: upper bound on the emitted length, in seconds

        Returns:
            Iterable[{key, spk, path, wav, sample_rate, num_samples}]
    """
    assert data_type == 'raw_fast'
    assert max_sec > min_sec, (min_sec, max_sec)

    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        try:
            obj = json.loads(json_line)
        except Exception as e:
            logging.warning(f"Failed to parse json line: {json_line}")
            raise e
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        try:
            sample_rate = 16_000

            # duration=None keeps the clip at its own length: read_audio_section
            # then reads from `start_time` (which must be explicit, since there
            # is no window to pick a random offset within) to the end of the
            # file, without the `np.pad(..., 'wrap')` tail loop.
            waveform, sample_rate = read_audio_section(wav_file,
                        samplerate=sample_rate,
                        duration=None,
                        min_duration=min_sec,
                        start_time=0.0,
                        extend_to_duration=False)

            if waveform is None:
                # Silent drop before; count it so it is visible (see
                # DROP_COUNTERS: train.py's CRITICAL level hides warnings).
                _drop_counter_incr(
                    'short_audio',
                    f"shorter than {min_sec:.2f}s: {obj}")
                continue

            # 记下**裁剪之前**的真实长度。超过 max_sec 的音频下面会被裁成
            # 随机的 max_sec 窗口，但按语音边界取段要用的是**文件真实时长**
            # （RTTM 里的时间是原始文件里的绝对时间），不能拿裁剪后的长度去当。
            full_num_samples = int(len(waveform))

            max_samples = int(max_sec * sample_rate)
            if len(waveform) > max_samples:
                # Trim to a random window of at most max_sec. Reusing
                # get_random_chunk keeps the "random offset, clone the result"
                # behaviour identical to the fixed-length path.
                waveform = get_random_chunk(waveform, max_samples)

            if len(waveform) < int(min_sec * sample_rate):
                _drop_counter_incr(
                    'short_audio',
                    f"shorter than {min_sec:.2f}s after read: {obj}")
                continue

            waveform = torch.from_numpy(np.ascontiguousarray(waveform))
            waveform = waveform.unsqueeze(0).float()

            if 'vad' in obj:
                raise NotImplementedError(
                    "'vad' segments are not supported by this reader")

            sample = dict(key=key,
                          spk=spk,
                          path=wav_file,
                          wav=waveform,
                          sample_rate=sample_rate,
                          num_samples=waveform.size(-1),
                          full_num_samples=full_num_samples)
            yield sample
        except Exception as ex:
            traceback.print_exc()
            _drop_counter_incr(
                'read_failed',
                f"{type(ex).__name__}: {ex} | {obj}")
            logging.warning('Failed to read & filter {}'.format(str(obj)))


def read_audio_span(fp, samplerate, start_sec, end_sec):
    """读音频文件在 [start_sec, end_sec) 这段时间里的波形。

    与 `read_audio_section` 的区别：这里两端都是**确切的时间**，不补零也不循环
    复制；文件末尾不够长时读到哪算哪。返回 (波形, 实际采样率)。
    这一段是为了按语音活动边界取样专门写的：起点、终点都是算好的秒数，
    不需要再随机挑位置，也不该被"补足到某个长度"的逻辑改动。
    """
    f = sf.SoundFile(fp)
    if not f.seekable():
        raise ValueError("Not compatible with seeking")
    sr = f.samplerate
    tot_dur = f.frames / sr
    start_sec = max(0.0, float(start_sec))
    end_sec = min(float(end_sec), tot_dur)
    if end_sec <= start_sec:
        return None, sr
    start_frame = int(sr * start_sec)
    n_frames = max(1, int(round(sr * (end_sec - start_sec))))
    # 起点在文件末尾之后、或算出来的长度超过文件剩余部分时，都以文件为准。
    n_frames = min(n_frames, f.frames - start_frame)
    if n_frames <= 0:
        return None, sr
    f.seek(start_frame)
    samples = f.read(n_frames)
    if sr != samplerate:
        samples = resampy.resample(samples, sr, samplerate)
    return samples, samplerate


# 按语音活动取样这一步的计数（每个进程一份；开了 num_workers 后各 worker 独立）。
VAD_STATS = {}


def vad_select_segment(data,
                       rttm_suffix='.rttm',
                       rttm_root=None,
                       rttm_root_depth=2,
                       audio_root=None,
                       min_sec=1.0,
                       max_sec=6.0,
                       target_sec=None,
                       pad_sec=0.1,
                       merge_gap_sec=None,
                       min_speech_sec=None,
                       cut_policy='middle',
                       low_speech_policy='resample',
                       low_speech_retries=2,
                       missing_policy='fallback',
                       fallback_min_sec=None,
                       fallback_max_sec=None,
                       fallback_chunk_sec=3.0,
                       read_fail_policy='fallback',
                       stats=None):
    """按语音活动（VAD）边界从音频里取一段完整的话。

    这是 `random_chunk_range` 的替代品：不用随机位置切，而是先读音频旁边那份
    分段文件（默认就是文件名后面补 `.rttm`），再从**语音段的边界**到**语音段的
    边界**挑一段，长度落在 [min_sec, max_sec] 里，尽量接近 target_sec。
    挑选规则见 `select_vad_span`。

    为了能"按自己的长度"送进模型，上游必须先跑
    `parse_raw_fast_varlen`（它给每条样本带上 `path` 和真实时长），
    所以这一段的输入应当是变长读取的输出。

    几种坏情况的处理，都可以在配置里改：

    * **分段文件不在** → `missing_policy`：
        - `fallback`（默认）：退回随机位置切一段（复用 `random_chunk_range`
          的取法）。样本不丢，但这一条不保证落在语音边界上。
        - `drop`：丢掉这条样本。适合"只想要干净的整段语音"的场景。
    * **分段文件在但读不出可用区间**（空文件、时间字段坏掉）→ `missing_policy`
      同上，计数走 `vad_bad_rttm`。
    * **选出来的片段语音太少**（少于 `min_speech_sec`）→ `low_speech_policy`：
        - `resample`（默认）：换一个起点再挑，最多试 `low_speech_retries` 次，
          仍然不行就退回随机切一段。
        - `keep`：不管语音多少，直接用。
        - `drop`：丢掉这条样本。
    * **单段语音比 max_sec 还长，只能在段内部切开** → 按 `cut_policy`
      取中段或随机一小段，并计入 `vad_cut_inside`。这是唯一一种会切到
      语音中间的情况。

    与"补零凑批"的接口不变：仍然写 `num_samples`，下游 `collate_varlen`
    照旧右侧零填充并回传 `lengths`；增广链照旧作用在这段波形上。
    Args:
        data: Iterable[{key, wav, label, path, sample_rate, ...}]
        rttm_suffix: 分段文件名后缀，默认 `.rttm`（`a.wav` → `a.wav.rttm`）
        min_sec / max_sec: 片段时长范围
        target_sec: 想要的时长，默认取上下限中点
        pad_sec: 两端各留的余量，默认 0.1 秒（保证不会正好切在语音的第一帧上，
            因为挑剔一点的分段器通常会把边界前的十几毫秒也标成静音）
        merge_gap_sec: 相邻语音段间隔小于这个值就先并成一段；默认为 0.0
        min_speech_sec: 片段里真实语音的下限，默认等于 min_sec
        cut_policy: 'middle' | 'random'，见上
        low_speech_policy: 'resample' | 'keep' | 'drop'，见上
        low_speech_retries: `resample` 最多再试几次
        missing_policy: 'fallback' | 'drop'，见上
        fallback_min_sec / fallback_max_sec: 回退随机切时的长度范围；
            默认分别是 min_sec 和 max(2*min_sec, target_sec)
        fallback_chunk_sec: 拿不到音频真实长度时，随机切多长（秒）
        read_fail_policy: 'fallback' | 'drop'，选好的片段读失败时怎么办
        stats: 可选字典，用来累加本段的选择/回退次数（不传就用进程内那份）

    Returns:
        Iterable[{key, wav, label, sample_rate, num_samples, vad_span}]
    """
    if stats is None:
        stats = VAD_STATS
    if min_speech_sec is None:
        min_speech_sec = min_sec
    if merge_gap_sec is None:
        # 默认**不**跟着 pad_sec 走，理由见 select_vad_span 的说明：
        # 余量只扩片段两端，不会补上片段内部的静音缝隙。
        merge_gap_sec = 0.0
    if fallback_min_sec is None:
        fallback_min_sec = min_sec
    if fallback_max_sec is None:
        fallback_max_sec = max(2.0 * min_sec, target_sec or
                               0.5 * (min_sec + max_sec))
    if missing_policy not in ('fallback', 'drop'):
        raise ValueError(f"unknown missing_policy: {missing_policy!r}")
    if low_speech_policy not in ('resample', 'keep', 'drop'):
        raise ValueError(f"unknown low_speech_policy: {low_speech_policy!r}")

    def _random_span(wav, sample_rate):
        """回退用：随机位置切一段（和 `random_chunk_range` 同样的取法）。"""
        target = int(random.uniform(fallback_min_sec, fallback_max_sec)
                     * sample_rate)
        target = max(1, target)
        if len(wav) > target:
            wav = get_random_chunk(wav, target)
        return wav

    for sample in data:
        assert 'key' in sample
        assert 'wav' in sample
        wav = sample['wav'][0]
        sample_rate = sample.get('sample_rate', 16000)
        wav_path = sample.get('path', None)
        # 这一条音频**真实**有多长。
        #
        # 优先用上游记下的 `full_num_samples`：变长读取器会把超过 `max_sec`
        # 的音频裁成随机的 `max_sec` 窗口，这时 `wav.size(-1)` 已经不是文件长度了，
        # 而 RTTM 里的时间是**原始文件**里的绝对时间 —— 拿裁剪后的长度去判断
        # "片段有没有越界"，会把本来合法的片段判成越界、错误地退回随机切。
        # 实测：10002 条里 3386 条 >6 秒，其中 743 条（21.9%）因此错误回退，
        # 而这 743 条用真实时长全都能成功。
        full_n = sample.get('full_num_samples', None)
        if full_n:
            clip_duration = float(full_n) / float(sample_rate)
        else:
            clip_duration = wav.size(-1) / float(sample_rate)

        segments = None
        rttm_path = None
        if wav_path:
            # 这里才 import：dataset.py 在本文件之前被加载，放在模块顶层会绕成
            # 循环 import。
            from wespeaker_lite.dataset.dataset import rttm_path_for_audio
            rttm_path = rttm_path_for_audio(str(wav_path), rttm_suffix,
                                            rttm_root=rttm_root,
                                            rttm_root_depth=rttm_root_depth,
                                            audio_root=audio_root)
            segments = parse_rttm_segments(rttm_path)

        def _fallback(reason):
            """回退随机切；要求 drop 就返回 True（表示这条样本丢掉）。"""
            if missing_policy == 'drop':
                _drop_counter_incr(reason, f"{wav_path} s={rttm_path}")
                return True
            stats['fallback_random'] = stats.get('fallback_random', 0) + 1
            sample['wav'] = _random_span(wav, sample_rate).unsqueeze(0)
            sample['num_samples'] = sample['wav'].size(-1)
            sample['vad_span'] = None
            sample['vad_mode'] = 'fallback_random'
            return False

        if wav_path is None:
            if _fallback('vad_no_rttm'):
                continue
            yield sample
            continue

        if segments is None:
            if _fallback('vad_no_rttm'):
                continue
            yield sample
            continue

        if not segments:
            if _fallback('vad_bad_rttm'):
                continue
            yield sample
            continue

        span = None
        attempts = 1
        if low_speech_policy == 'resample':
            attempts = 1 + max(0, int(low_speech_retries))
        for attempt in range(attempts):
            this_min_speech = (0.0 if low_speech_policy == 'keep'
                               else min_speech_sec)
            # **重试必须换一个目标时长，否则等于没重试。**
            #
            # 原来每次都用同一个 `target_sec`，而 `select_vad_span` 是确定性的：
            # 只要"最接近目标"的那个区间是唯一的，重试多少次都会挑到同一段，
            # 结果自然还是语音太少。实测 200 条真实分段文件里，
            # 唯一最优区间 155 条、并列 0 条 —— 也就是说重试能改变结果的概率是 0，
            # `low_speech_retries` 这个开关实际上不起作用。
            #
            # 改成：第 2 次起把目标时长在 [min_sec, max_sec] 里挪一挪，
            # 让它去考虑别的区间（更靠中间的、或更短的），
            # 这样"多试几次"才真的多试了几个候选。
            if attempt == 0:
                this_target = target_sec
            elif target_sec is None:
                this_target = None
            else:
                lo = min_sec
                hi = max_sec
                # 在合法区间里取一个和上次不同的目标：交替偏向更短 / 更长
                bias = 0.25 if attempt % 2 == 1 else 0.75
                this_target = min(max(lo + bias * (hi - lo), lo), hi)
            span = select_vad_span(segments,
                                   min_sec=min_sec,
                                   max_sec=max_sec,
                                   target_sec=this_target,
                                   pad_sec=pad_sec,
                                   merge_gap_sec=merge_gap_sec,
                                   cut_policy=cut_policy,
                                   min_speech_sec=this_min_speech,
                                   rng=random,
                                   clip_duration=clip_duration,
                                   stats=None)
            if span is not None:
                break

        if span is None:
            # 语音含量怎么都到不了下限：送不进模型就先丢，丢不掉就回退。
            stats['low_speech'] = stats.get('low_speech', 0) + 1
            if low_speech_policy == 'drop':
                _drop_counter_incr('vad_low_speech', f"{wav_path}")
                continue
            # 记成 vad_low_speech，**不要**记成 vad_bad_rttm：
            # 走到这里说明分段文件本身是好的、也读出了区间，只是选出来的
            # 片段里语音太少。记错名字会让人去查数据，而真正该调的
            # 是 min_speech_sec 或 min_sec。
            if _fallback('vad_low_speech'):
                continue
            yield sample
            continue

        if span['mode'] == 'cut_inside':
            # 报的是"被切的那一整段"有多长，不是切完之后的长度；
            # 否则会出现 2.251s > 6.000s 这种自相矛盾的话（实测踩到过）。
            _src = span.get('cut_source_sec')
            _src_txt = (f"{_src:.3f}s" if _src is not None else "未知")
            _drop_counter_incr(
                'vad_cut_inside',
                f"{wav_path} 单段语音长 {_src_txt}，超过 max_sec={max_sec:.3f}s，"
                f"只能在段内部切开（切出 {span['end'] - span['start']:.3f}s）")
        # 只按 mode 计一次。**不要**在 cut_inside 分支里另外再加一次
        # `stats['cut_inside']` —— 那样一个样本会被算成 2 个，
        # "段内切占多少比例"就虚报成两倍，文档里的分模式统计表也会和日志对不上。
        _mode = span['mode']
        stats[_mode] = stats.get(_mode, 0) + 1

        # 读盘可能**抛异常**（文件缺失、损坏、权限），不只是"返回空"。
        # 原来这里没有 try：一个坏文件会直接把异常抛穿 DataLoader worker、
        # 把训练打断，而配置里的 `read_fail_policy` 对这条最常见的失败路径
        # 完全不起作用（实测：fallback 与 drop 两种取值都会抛 LibsndfileError）。
        # 现在统一成"读失败"，交给下面的策略处理。
        try:
            chunk, _ = read_audio_span(wav_path, sample_rate,
                                       span['start'], span['end'])
        except Exception as exc:
            chunk = None
            stats['read_error'] = stats.get('read_error', 0) + 1
            _drop_counter_incr('read_failed',
                               f"{type(exc).__name__}: {exc} | {wav_path}")
        if chunk is None or len(chunk) == 0:
            stats['read_failed'] = stats.get('read_failed', 0) + 1
            if read_fail_policy == 'drop':
                _drop_counter_incr('read_failed',
                                   f"span read returned nothing: {wav_path}")
                continue
            if _fallback('read_failed'):
                continue
            yield sample
            continue

        sample['wav'] = torch.from_numpy(
            np.ascontiguousarray(chunk)).unsqueeze(0).float()
        sample['num_samples'] = sample['wav'].size(-1)
        # 记下这一段是从哪儿来的，便于核对"片段边界就是语音段边界"。
        sample['vad_span'] = (round(span['start'], 6), round(span['end'], 6))
        sample['vad_mode'] = span['mode']
        yield sample


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav/feat, spk}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav/feat, spk}]
    """
    # Use a deterministic RNG seeded from the worker's initial random state.
    # This makes the local shuffle reproducible regardless of how many random
    # calls other pipeline stages have made.
    rng = random.Random(random.getrandbits(64))
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            rng.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    rng.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    # A speaker that is absent from spk2id used to be emitted with label = -1.
    # That is worse than a plain bad label: under speed perturbation
    # speed_perturb() adds num_spks * speed_idx, so -1 becomes num_spks - 1 or
    # 2 * num_spks - 1 -- both of which ARE valid speaker ids -- and the sample
    # is silently trained as a real, wrong speaker (label corruption that is
    # invisible in the loss curve).
    #
    # We drop such samples rather than raising: a label/data-list mismatch is
    # usually a partial data problem (one corpus in a mixed list not yet
    # regenerated, a speaker dropped by filtering), and aborting a multi-day
    # training run on it is worse than training on the remaining samples. The
    # count is reported through DROP_COUNTERS so the mismatch is still visible;
    # if the count is large, fix data.list/utt2spk instead of ignoring it.
    for sample in data:
        assert 'spk' in sample
        if sample['spk'] in spk2id:
            sample['label'] = spk2id[sample['spk']]
        else:
            _drop_counter_incr(
                'unknown_spk',
                f"spk={sample['spk']!r} key={sample.get('key')!r} "
                f"not in spk2id ({len(spk2id)} speakers); sample dropped")
            continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, num_spks):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = sample.pop('speed_idx', random.randint(0, 2))
        if speed_idx > 0:
            wav, _ = torchaudio.functional.speed(
                waveform, sample_rate, speeds[speed_idx])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_spks * speed_idx
            # 变速会改变波形长度，`num_samples` 必须跟着更新。
            #
            # 原来这里不更新，靠下游（`random_chunk` / `random_chunk_range`）
            # 重新算。那两条路都排在变速之后，所以一直是巧合地正确。
            # 但 `vad_select_segment` 现在要排在变速**之前**（见 dataset.py 的说明），
            # 于是变速成了最后一个改波形的步骤，这里不更新就会留下一个
            # 与波形不符的 num_samples，下游按它补零、建掩码就会错位。
            if 'num_samples' in sample:
                sample['num_samples'] = wav.size(-1)

        yield sample


def get_random_chunk(data, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # Tile-and-crop, for input shorter than chunk_len. Not dead code, but
        # note which callers actually reach it:
        #
        #   * Training chunking (parse_raw_fast_and_filter) never does: it reads
        #     chunk_len * 1.1 samples and speed perturb only shortens back to
        #     chunk_len, so the input is always >= chunk_len here.
        #   * The current recipes do not either. They use the composable
        #     Sequential/OneOf tree from dataset/augs.py, whose CustomNoise pads
        #     short noise via its own _crop_or_pad helper.
        #   * The legacy add_reverb_noise_gain path does, and it is the reason
        #     this branch is kept: it requests a full chunk of noise, and 298 of
        #     the 930 MUSAN noise files are shorter than a 6 s chunk. Without the
        #     tile, short noise files would come back undersized and the later
        #     `audio + scale * noise` would fail on mismatched shapes.
        #
        # Keep it: removing it would break that legacy path (and any future
        # caller that hands in an undersized buffer) for no gain.
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (
            repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data




def random_chunk(data, chunk_len, data_type='shard/raw/feat'):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            feat = get_random_chunk(feat, chunk_len)
            sample['feat'] = feat
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            wav = get_random_chunk(wav, chunk_len)
            sample['wav'] = wav.unsqueeze(0)
        yield sample

def random_chunk_range(data, min_sec=1.0, max_sec=3.0):
    """ Random chunk each sample to a length drawn in [min_sec, max_sec]

        Variable-length sibling of `random_chunk`. The chunk length varies per
        sample, so the emitted waveforms still differ in length and the batch
        must be padded by `collate_varlen`. Samples already shorter than the
        drawn length are left as they are -- growing them would need the same
        tail loop the variable-length path exists to avoid.

        Args:
            data: Iterable[{key, wav, label}]
            min_sec: lower bound of the drawn chunk length, in seconds
            max_sec: upper bound of the drawn chunk length, in seconds

        Returns:
            Iterable[{key, wav, label, num_samples}]
    """
    assert max_sec >= min_sec, (min_sec, max_sec)
    for sample in data:
        assert 'key' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        wav = sample['wav'][0]
        target = int(random.uniform(min_sec, max_sec) * sample_rate)
        target = max(1, target)
        if len(wav) > target:
            wav = get_random_chunk(wav, target)
        sample['wav'] = wav.unsqueeze(0)
        sample['num_samples'] = wav.size(-1)
        yield sample


def norm_rir(rir_audio):
    return rir_audio / np.sqrt(np.sum(rir_audio**2))



def apply_augmentor(data, augmentor, resample_rate=16000):
    """Apply a deserialized Augmentor (from dataset.augs) to each sample.

    Bridges the Sequential/OneOf augmentor tree (used in distillation) into
    the standard training data pipeline. The augmentor mutates sample['wav']
    in place via a {'wav', 'sample_rate'} dict view.
    """
    for sample in data:
        ex = {'wav': sample['wav'],
              'sample_rate': sample.get('sample_rate', resample_rate)}
        ex = augmentor(ex)
        sample['wav'] = ex['wav']
        yield sample







def add_reverb_noise_gain(data,
                          reverb_source,
                          noise_source,
                          resample_rate=16000,
                          aug_prob=0.6,
                          aug_setup='default',
                          gain_prob=1.0,
                          gain_db_range=(-18.0, 6.0),
                          speech_rms_mode='full',
                          peak_protect=True,
                          noise_categories=None,
                          eps=1e-12):
    """Reverb -> additive noise at target SNR -> random gain. No output RMS norm.

    Level-faithful augmentation chain for training without input z-norm
    (``spec_params.norm_signal: false``). The absolute level reaching the
    model is never renormalized: the training-level distribution is the
    natural utterance levels spread by the final random gain.

    Per sample, with probability ``aug_prob``, one of {reverb+noise, reverb,
    noise} is applied:

    1. Reverb: convolution with a unit-energy RIR. Only the *filter* is
       normalized; the convolved output is left as-is.
    2. Additive noise: scaled to a target SNR measured against the *current*
       speech (i.e. after reverb, matching what the model hears) and added.
       The mixture is not rescaled afterwards, so the sampled SNR
       distribution is preserved exactly.

    Independently of the above, with probability ``gain_prob`` a random gain
    (uniform in dB over ``gain_db_range``) is applied to every sample,
    augmented or not. A uniform scale moves the absolute level but not SNR.

    ``peak_protect`` rescales the final signal only if its peak exceeds 1.0
    (real capture cannot exceed full scale); uniform rescale preserves SNR.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        reverb_source / noise_source: LMDB sources. noise_source must support
            category sampling (LmdbDataMap) so SNR ranges match noise types.
        aug_setup: SNR preset, 'default' | 'medium' | 'hard'.
        speech_rms_mode: 'full' = power over the whole chunk (same SNR
            semantics as the older presets); 'active' = power over 25 ms
            frames within 40 dB of the loudest frame (active-speech RMS).

    Returns:
        Iterable[{key, wav, label, sample_rate, applied_augs}]
    """
    noisesnr = {
        'default': {'noise': [0, 15], 'music': [5, 15], 'speech': [13, 20], 'bubble': [13, 20]},
        'medium':  {'noise': [0, 12], 'music': [3, 15], 'speech': [5, 18],  'bubble': [5, 18]},
        'hard':    {'noise': [0, 12], 'music': [1, 12], 'speech': [2, 12],  'bubble': [2, 12]},
    }
    # [reverb+noise, reverb-only, noise-only]
    aug_type_prob = {
        'default': [0.25, 0.1, 0.65],
        'medium':  [0.25, 0.1, 0.65],
        'hard':    [0.25, 0.1, 0.65],
    }
    numnoise = {'noise': [1, 1], 'speech': [1, 1], 'bubble': [3, 7], 'music': [1, 1]}

    if noise_categories is not None:
        allowed = set(noise_categories)
        noisesnr = {k: {c: v for c, v in cats.items() if c in allowed}
                    for k, cats in noisesnr.items()}
        numnoise = {c: v for c, v in numnoise.items() if c in allowed}

    assert aug_setup in noisesnr, f"Unknown preset: {aug_setup}"
    assert speech_rms_mode in ('full', 'active'), speech_rms_mode
    assert gain_db_range[0] <= gain_db_range[1], gain_db_range

    frame_len = max(1, int(0.025 * resample_rate))

    def _full_power(x):
        return float(np.dot(x, x) / max(x.shape[0], 1))

    def _speech_power(x):
        if speech_rms_mode != 'active':
            return _full_power(x)
        n = x.shape[0] // frame_len
        if n < 2:
            return _full_power(x)
        fp = np.mean(x[:n * frame_len].reshape(n, frame_len) ** 2, axis=1)
        pmax = float(fp.max())
        if pmax <= 0.0:
            return 0.0
        return float(fp[fp >= pmax * 1e-4].mean())



    for sample in iter(data):

        rnd_ctx = sample.pop('rnd_ctx', nullcontext())

        with rnd_ctx:
            assert 'wav' in sample
            assert 'key' in sample
            curr_aug_setup = []
            audio = sample['wav'].numpy()[0]
            audio_len = audio.shape[0]
            modified = False

            if aug_prob > random.random():
                aug_type = random.choices([0, 1, 2],
                                        weights=aug_type_prob[aug_setup], k=1)[0]

                if aug_type in [0, 1]:
                    # Reverb first: the additive noise below models noise at the
                    # receiver, so it must not pass through the room filter.
                    rir_key, rir_data = reverb_source.random_one()
                    rir_sr, rir_audio = wavfile.read(io.BytesIO(rir_data))
                    rir_audio = rir_audio.astype(np.float32)
                    if rir_sr != resample_rate:
                        rir_audio = signal.resample(
                            rir_audio,
                            int(len(rir_audio) / rir_sr * resample_rate))
                    rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2) + eps)
                    audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
                    modified = True
                    curr_aug_setup.append(dict(aug='reverb', rir_key=rir_key))

                if aug_type in [0, 2]:
                    cat = random.choice(list(numnoise.keys()))
                    sample_cat = 'speech' if cat == 'bubble' else cat
                    min_num, max_num = numnoise[cat]
                    if min_num < max_num:
                        # numnoise ranges are inclusive ([3, 7] means 3..7
                        # noises, the same inclusive convention as the snr
                        # ranges right above). np.random.randint is exclusive on
                        # the upper bound, so the old call never produced
                        # max_num: bubble only ever drew {3,4,5,6}. +1 restores
                        # the documented range.
                        num_noises = np.random.randint(min_num, max_num + 1)
                    else:
                        num_noises = max_num

                    full_noise = []
                    for _ in range(num_noises):
                        _, noise_data = noise_source.random_one(sample_cat)
                        noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
                        # Chunk in the file's native dtype BEFORE float conversion:
                        # decoding whole files (minutes of music) to float32 was
                        # the chain's dominant cost (profiling/profile_augs_v2.py).
                        if noise_sr != resample_rate:
                            noise_audio = get_random_chunk(
                                noise_audio,
                                int(audio_len / resample_rate * noise_sr))
                            noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                            noise_audio = signal.resample(noise_audio, audio_len)
                        else:
                            noise_audio = get_random_chunk(noise_audio, audio_len)
                            noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                        full_noise.append(noise_audio)
                    if num_noises > 1:
                        noise_audio = np.mean(np.array(full_noise), axis=0)
                    else:
                        noise_audio = full_noise[0]

                    snr_db = random.uniform(*noisesnr[aug_setup][cat])
                    speech_power = _speech_power(audio)
                    noise_power = _full_power(noise_audio)
                    if speech_power > 0.0 and noise_power > 0.0:
                        # float(): a np.float64 scalar would promote the mixture
                        # and everything downstream to float64
                        scale = float(np.sqrt(
                            10**(-snr_db / 10) * speech_power / noise_power))
                        audio = audio + scale * noise_audio
                        modified = True
                        curr_aug_setup.append(dict(aug='noise', noise_cat=cat,
                                                snr_db=float(snr_db),
                                                scale=scale))

            if gain_prob > random.random():
                gain_db = random.uniform(gain_db_range[0], gain_db_range[1])
                audio = audio * (10.0**(gain_db / 20.0))
                modified = True
                curr_aug_setup.append(dict(aug='gain', gain_db=float(gain_db)))

            if peak_protect and modified:
                peak = float(np.max(np.abs(audio)))
                if peak > 1.0:
                    audio = audio * (0.99 / peak)
                    curr_aug_setup.append(dict(aug='peak_protect',
                                            peak=float(peak)))

            if modified:
                sample['wav'] = torch.from_numpy(audio).unsqueeze(0).float()
            sample['applied_augs'] = json.dumps(curr_aug_setup)
            yield sample
