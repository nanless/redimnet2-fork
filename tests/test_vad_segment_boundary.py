#!/usr/bin/env python3
"""按语音活动（VAD）边界取片段的回归测试。

这个脚本回答一个问题：**取出来的片段，起止是不是正好落在 RTTM 语音段的
边界上**。全部用合成数据，不需要真实语料，也不需要 GPU。

覆盖三种情形：

* ``single``：一段语音就够长，片段 = 那一段（可能再两边各留一点余量）；
* ``merge``：一段不够，连着好几段一起用；
* ``cut_inside``：单独一段就比 ``max_sec`` 还长，只能在段内部切开
  （这是唯一允许切进语音中间的情况，另外用 ``vad_cut_inside`` 计数）。

还检查：余量怎么加、音频不够长时片段怎么平移、语音太少时返回 None。

把这几种情形钉成测试的原因：这些规则都踩过坑。比如"合并相邻语音段"的阈值
一开始跟着余量走，结果把「加了余量才连起来」的区间并成一段，随后又把起点放在
合并区间的左端——而它中间那条静音缝隙其实还在，片段边界因此落在语音段之外。
现在合并阈值默认是 0，只并真正重叠或首尾相接的段。

用法::

    /root/miniforge3/envs/redimnet2/bin/python tests/test_vad_segment_boundary.py

退出码 0 表示检查通过；非 0 表示有检查没通过，并会打印差在哪。
"""
from __future__ import annotations

import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, 'applications', 'speaker_rec')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wespeaker_lite.dataset import processor as P  # noqa: E402

EPS = 1e-6

def fmt(segs):
    return "[" + ", ".join(f"({s:.3f}, {e:.3f}, {e-s:.3f}s)" for s, e in segs) + "]"

def check(name, segs, min_sec, max_sec, target_sec=None, pad_sec=0.0,
          clip_duration=None, cut_policy='middle', expect_mode=None, seed=0):
    rng = random.Random(seed)
    stats = {}
    span = P.select_vad_span(segs, min_sec=min_sec, max_sec=max_sec,
                             target_sec=target_sec, pad_sec=pad_sec,
                             cut_policy=cut_policy, rng=rng,
                             clip_duration=clip_duration, stats=stats)
    print(f"\n=== {name} ===")
    print(f"  RTTM 语音段      : {fmt(segs)}")
    print(f"  参数             : min_sec={min_sec} max_sec={max_sec} "
          f"target_sec={target_sec} pad_sec={pad_sec} clip={clip_duration}")
    if span is None:
        print("  选中区间         : 没有可用片段（返回 None）")
        print(f"  计数             : {stats}")
        return None
    s, e = span['start'], span['end']
    print(f"  选中区间         : ({s:.3f}, {e:.3f})  时长={e-s:.3f}s")
    print(f"  模式             : {span['mode']}   语音时长={span['speech_sec']:.3f}s  "
          f"aligned={span['aligned']}")
    # 边界判定：pad_sec=0 时，起点必须等于某个语音段的起点，终点必须等于某个语音段的终点
    starts = {round(x[0], 6) for x in segs}
    ends = {round(x[1], 6) for x in segs}
    on_start = round(s, 6) in starts
    on_end = round(e, 6) in ends
    print(f"  起点落在某个段起点上: {on_start}   终点落在某个段终点上: {on_end}")
    print(f"  计数             : {stats}")
    assert min_sec - EPS <= (e - s) <= max_sec + EPS, "长度越界"
    if expect_mode is not None:
        assert span['mode'] == expect_mode, f"模式应为 {expect_mode}，实际 {span['mode']}"
    if pad_sec > 0 and span['mode'] != 'cut_inside' and span['aligned']:
        # 加了余量之后，边界应当是「段起点-pad / 段终点+pad」；两头对称地让掉
        # 放不下的余量时 aligned 会变 False，那种情况这里不检查。
        on_start = round(s + pad_sec, 6) in starts
        on_end = round(e - pad_sec, 6) in ends
        print(f"  减掉余量后落在段边界上: 起点 {on_start} / 终点 {on_end}")
        assert on_start and on_end, "减掉余量后边界没有落在语音段边界上！"
    if pad_sec == 0.0 and span['mode'] != 'cut_inside':
        assert on_start and on_end, "边界没有落在语音段边界上！"
    return span

# ── A. 单段足够 ──
segs_a = [(0.500, 1.800), (2.400, 4.100), (5.000, 6.300), (7.200, 8.000)]
check("A1 单段就够，target 落在第 2 段上（single）", segs_a, 1.0, 6.0,
      target_sec=1.7, seed=1, expect_mode='single')
check("A2 单段就够，target 落在第 3 段上（single）", segs_a, 1.0, 6.0,
      target_sec=1.3, seed=1, expect_mode='single')

# ── B. 必须并两段以上（merge）──
#    每段都不到 2 秒，一段凑不出 min_sec=2.0，只能往后并
segs_b = [(0.100, 1.400), (1.700, 3.000), (3.400, 4.600), (5.000, 6.200)]
check("B 合并多段（merge）", segs_b, 2.0, 6.0, seed=2, expect_mode='merge')

# ── B2: 目标 5 秒，必须并多段（每段都不够 5 秒）──
segs_b2 = [(0.000, 1.500), (1.600, 2.700), (2.800, 4.000), (4.100, 5.900)]
check("B2 目标 5s 必须并多段（merge）", segs_b2, 2.0, 6.0, target_sec=5.0, seed=3,
      expect_mode='merge')

# ── B3: 一段就有 3 秒但目标 5 秒，只能并两段 ──
segs_b3 = [(1.000, 4.000), (4.500, 7.000), (9.000, 12.000)]
sp = check("B3 目标 5s、两段并起来正好 6s（merge）", segs_b3, 2.0, 6.0,
      target_sec=5.0, seed=8, expect_mode='merge')
# target=3.6：(1)+(2) 并起来正好 3.6s，比任何单段都接近目标
segs_b4 = [(1.000, 2.800), (3.000, 5.400), (8.000, 14.000)]
check("B4 target=3.6，第一第二段并起来正好命中（merge）", segs_b4, 2.0, 6.0,
      target_sec=3.6, seed=9, expect_mode='merge')

# ── C. 单段超长，必须切（cut_inside）──
segs_c = [(0.200, 12.700)]
sp = check("C 单段 12.5s > max_sec=6s（cut_inside, middle）", segs_c, 1.0, 6.0,
           seed=4, expect_mode='cut_inside')
print(f"  该段中点 = {(0.200+12.700)/2:.3f}；选中区间中点 = {(sp['start']+sp['end'])/2:.3f}"
      f"  → 取的是中间 {'OK' if abs((sp['start']+sp['end'])/2 - 6.45) < 1e-6 else 'NO'}")

# ── D. 带余量 pad_sec：仍然对称地落在边界两侧 ──
check("D pad_sec=0.15，边界仍对齐", segs_a, 1.0, 6.0, pad_sec=0.15, seed=5)
check("D2 pad_sec=0.15，音频开头不够长时整段平移（保长度）", [(0.100, 2.000), (3.0, 5.0)],
      1.0, 6.0, pad_sec=0.15, seed=6, clip_duration=6.0)

# ── E. 语音太少的边界情形 ──
print("\n=== E 语音太少 ===")
segs_e = [(0.000, 0.300)]
sp = P.select_vad_span(segs_e, min_sec=0.5, max_sec=6.0, min_speech_sec=1.0,
                       rng=random.Random(7))
print(f"  语音段 {fmt(segs_e)}，min_sec=0.5，min_speech_sec=1.0 → {sp}")
assert sp is None, "应当因为语音不足返回 None"

print("\n所有合成用例通过。")

# ── 解析 RTTM 文本本身也要稳 ──
def check_parse():
    import tempfile
    text = (
        "# 这一行是注释，应该忽略\n"
        "SPEAKER 00001 1 0.000 1.250 <NA> <NA> <NA> <NA> <NA>\n"
        "\n"
        "SPEAKER 00001 1 1.500 2.000 <NA> <NA> <NA> <NA> <NA>\n"
        "这一行不是 SPEAKER，忽略\n"
        "SPEAKER 00001 1 4.000 -1.000 <NA> <NA> <NA> <NA> <NA>\n"  # 负时长：丢掉
        "SPEAKER 00001 1 时间坏掉了 1.000 <NA> <NA> <NA> <NA> <NA>\n"  # 丢掉
    )
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as f:
        f.write(text)
        path = f.name
    try:
        segs = P.parse_rttm_segments(path)
        assert segs == [(0.0, 1.25), (1.5, 3.5)], segs
        print("\n=== RTTM 解析 ===")
        print(f"  注释行/空行/坏行都跳过了，读出: {segs}")
        # 文件不存在要返回 None（表示"没有分段可用"，调用方据此走回退策略）
        assert P.parse_rttm_segments(path + '.does_not_exist') is None
        print("  文件不存在时返回 None（调用方走回退策略）")
        # 合并：间隔 <= 阈值的相邻段并成一段
        print(f"  合并（间隔<=0.3）: {P.merge_segments(segs, gap_sec=0.3)}")
        assert P.merge_segments(segs, gap_sec=0.3) == [(0.0, 3.5)]
        assert P.merge_segments(segs, gap_sec=0.0) == [(0.0, 1.25), (1.5, 3.5)]
    finally:
        os.unlink(path)

check_parse()
print("\n所有检查通过。")
