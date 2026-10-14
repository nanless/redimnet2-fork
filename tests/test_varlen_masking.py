#!/usr/bin/env python3
"""变长输入与掩码的回归测试。

这个脚本回答两个问题，并且把答案变成可以自动比较的数字：

1. **向后兼容**：不传 `lengths` 时，模型的行为有没有变？
   基准是"改动前的那份代码"的结果，要求逐位一致（最大绝对差 0.0）。

2. **掩码是否真的在起作用**：
   * 掩码**全有效**时，结果应当与不传掩码几乎相同（实测余弦 > 0.999）。
     这一条如果不成立，说明掩码本身接错了。
   * 掩码**半有效**（一条短音频补零凑批）时，结果会与逐条推理有差距，
     但必须明显好于"补零却不传掩码"。这一条是防退化的下限。

为什么要专门测第二条：变长批量与逐条推理**不可能逐位等价**。
补齐会改变总帧数，进而改变下采样后的时间上下文，
最后一个真实帧的卷积感受野也会读到补零区。实测在补齐比 2 倍时余弦约 0.95。
所以这里不做"必须等于 1"的断言，而是记录一条下限：
一旦某次改动让这个数字掉下去，测试就会失败。

用法（必须用装了 torch 的环境）::

    /root/miniforge3/envs/redimnet2/bin/python tests/test_varlen_masking.py
    /root/miniforge3/envs/redimnet2/bin/python tests/test_varlen_masking.py --quick
    /root/miniforge3/envs/redimnet2/bin/python tests/test_varlen_masking.py \
        --baseline-pkg /path/to/old/redimnet2_parent_dir

退出码 0 表示全部通过；非 0 表示有检查未通过，并会打印差在哪。
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CKPT_DIR_DEFAULT = os.path.join(REPO, 'checkpoints', 'pretrained')

# 各项检查的下限。这些数字来自实测，改动模型层后如果掉下去就说明退化了。
MIN_COS_ALL_VALID = 0.999      # 掩码全有效 vs 不传掩码
MIN_COS_HALF_VALID = 0.90      # 半有效掩码 vs 逐条（实测约 0.95，留余量）

# 「补零但不传掩码应当明显更差」这一条不能用固定的绝对阈值。
#
# 原因：不同权重的余弦尺度差别很大。实测非对角余弦（同一模型对两段互不相关
# 音频的余弦）在 b0-vox2-lm 上是 0.64，而在混合数据那几个权重上高达 0.99——
# 也就是说后者的向量高度相关，余弦被压缩在一个很窄的区间里，
# 在那里「0.81」其实已经是很大的差异了。
# 原来写死 `<= 0.6` 会让 5 个混合数据权重（b3/b6 的 vb2/cnc2）误报失败。
#
# 改成**与模型自身尺度比较**：拿该模型对两段无关音频的余弦当基准
# （代表这个空间的"区分度尺度"），要求 no_mask 比基准至少低这么多。
# 实测 20 个权重全部满足，最小差值 0.158、最大 0.650。
MIN_NO_MASK_MARGIN = 0.12      # no_mask 至少要比该模型自身的基准低这么多


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


def load_model(pkg_root, pkg_name, ckpt_path):
    """从指定目录加载一份 redimnet2 包，并载入权重。"""
    for mod in list(sys.modules):
        if mod == pkg_name or mod.startswith(pkg_name + '.'):
            del sys.modules[mod]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        os.path.join(pkg_root, pkg_name, '__init__.py'),
        submodule_search_locations=[os.path.join(pkg_root, pkg_name)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = module.ReDimNet2Wrap(**state['model_config'])
    result = model.load_state_dict(state['state_dict'])
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f'{ckpt_path}: 权重与结构对不上 '
            f'missing={result.missing_keys} unexpected={result.unexpected_keys}')
    model.eval()
    return model


def pick_ckpts(ckpt_dir, quick):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
    if not ckpts:
        raise SystemExit(f'在 {ckpt_dir} 下没有找到 .pt 权重')
    if quick:
        # 只挑最小与最大的各一个，兼顾快与代表性
        ckpts = [ckpts[0], ckpts[-1]]
    return ckpts


def check_backward_compat(ckpt, baseline_root, quick):
    """不传 lengths 时，与改动前的代码逐位比较。"""
    if baseline_root is None:
        return None
    n_samples = 16000 if quick else 32000
    ref = load_model(baseline_root, 'redimnet2', ckpt)
    new = load_model(REPO, 'redimnet2', ckpt)
    worst = 0.0
    with torch.no_grad():
        for duration in (16000, 32000):
            wave = torch.randn(1, duration,
                               generator=torch.Generator().manual_seed(duration))
            worst = max(worst, (ref(wave) - new(wave)).abs().max().item())
    return worst


def check_mask_effect(ckpt, quick):
    """掩码全有效 / 半有效 / 不传掩码 三种情况的余弦。"""
    model = load_model(REPO, 'redimnet2', ckpt)
    long_len, short_len = 32000, 16000
    if quick:
        long_len, short_len = 16000, 8000

    wave_short = torch.randn(1, short_len,
                             generator=torch.Generator().manual_seed(1))
    wave_long = torch.randn(1, long_len,
                            generator=torch.Generator().manual_seed(2))

    padded = torch.zeros(2, long_len)
    padded[0, :long_len] = wave_long[0]
    padded[1, :short_len] = wave_short[0]

    with torch.no_grad():
        ref_short = model(wave_short)[0]
        ref_long = model(wave_long)[0]
        emb = model(padded, lengths=torch.tensor([long_len, short_len]))
        emb_all_valid = model(padded, lengths=torch.tensor([long_len, long_len]))
        emb_no_lengths = model(padded)

        # 该模型自身的"余弦尺度"：两段互不相关、等长的音频之间的余弦。
        # 用它当基准，才能判断「no_lengths 掉了多少」，而不是跟一个固定的
        # 绝对阈值比（见文件开头关于 MIN_NO_MASK_MARGIN 的说明）。
        unrelated_a = torch.randn(1, long_len,
                                  generator=torch.Generator().manual_seed(3))
        unrelated_b = torch.randn(1, long_len,
                                  generator=torch.Generator().manual_seed(4))
        baseline = cos(model(unrelated_a)[0], model(unrelated_b)[0])

    no_lengths = cos(emb_no_lengths[1], ref_short)
    return {
        'all_valid': cos(emb_all_valid[0], emb_no_lengths[0]),
        'half_valid': cos(emb[1], ref_short),
        'long_unaffected': cos(emb[0], ref_long),
        'no_lengths': no_lengths,
        'baseline': baseline,
        'no_mask_margin': baseline - no_lengths,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='变长输入与掩码的回归测试')
    parser.add_argument('--ckpt-dir', default=CKPT_DIR_DEFAULT,
                        help='存放已发布权重的目录')
    parser.add_argument('--quick', action='store_true',
                        help='只用两个权重、较短音频，快速跑一遍')
    parser.add_argument('--baseline-pkg', default=None,
                        help='改动前的 redimnet2 包所在目录（其下应有 redimnet2/）。'
                             '给了就额外做一次逐位向后兼容比较')
    args = parser.parse_args(argv)

    torch.set_grad_enabled(False)
    ckpts = pick_ckpts(args.ckpt_dir, args.quick)
    print(f'待测权重 {len(ckpts)} 个：'
          f'{[os.path.basename(c) for c in ckpts]}')

    failures = []

    if args.baseline_pkg:
        print('\n[1] 向后兼容：不传 lengths 时与改动前逐位比较')
        for ckpt in ckpts:
            worst = check_backward_compat(ckpt, args.baseline_pkg, args.quick)
            ok = worst == 0.0
            print(f'  {os.path.basename(ckpt):32s} 最大绝对差 {worst:.3e} '
                  f'{"通过" if ok else "未通过"}')
            if not ok:
                failures.append(f'{ckpt}: 向后兼容最大绝对差 {worst:.3e} != 0')
    else:
        print('\n[1] 向后兼容：未给 --baseline-pkg，跳过')

    print('\n[2] 掩码效果（补齐比 2 倍）')
    for ckpt in ckpts:
        r = check_mask_effect(ckpt, args.quick)
        name = os.path.basename(ckpt)
        ok_all = r['all_valid'] >= MIN_COS_ALL_VALID
        ok_half = r['half_valid'] >= MIN_COS_HALF_VALID
        ok_gap = r['no_mask_margin'] >= MIN_NO_MASK_MARGIN
        ok_long = r['long_unaffected'] >= MIN_COS_ALL_VALID
        print(f'  {name}')
        print(f'    掩码全有效 vs 不传掩码 : {r["all_valid"]:.6f}  '
              f'(要求 >= {MIN_COS_ALL_VALID})          '
              f'{"通过" if ok_all else "未通过"}')
        print(f'    半有效掩码 vs 逐条     : {r["half_valid"]:.6f}  '
              f'(要求 >= {MIN_COS_HALF_VALID})          '
              f'{"通过" if ok_half else "未通过"}')
        print(f'    不传掩码   vs 逐条     : {r["no_lengths"]:.6f}  '
              f'(该模型基准 {r["baseline"]:.6f}，'
              f'要求至少低 {MIN_NO_MASK_MARGIN})  '
              f'{"通过" if ok_gap else "未通过"}')
        print(f'    同批里长的那条不受影响 : {r["long_unaffected"]:.6f}  '
              f'(要求 >= {MIN_COS_ALL_VALID})          '
              f'{"通过" if ok_long else "未通过"}')
        if not ok_all:
            failures.append(
                f'{name}: 掩码全有效余弦 {r["all_valid"]:.6f} < {MIN_COS_ALL_VALID}')
        if not ok_half:
            failures.append(
                f'{name}: 半有效余弦 {r["half_valid"]:.6f} < {MIN_COS_HALF_VALID}')
        if not ok_gap:
            failures.append(
                f'{name}: 不传掩码余弦 {r["no_lengths"]:.6f} 只比该模型基准 '
                f'{r["baseline"]:.6f} 低 {r["no_mask_margin"]:.6f}，'
                f'不足 {MIN_NO_MASK_MARGIN}，掩码似乎没有起作用')
        if not ok_long:
            failures.append(
                f'{name}: 同批长条余弦 {r["long_unaffected"]:.6f} < {MIN_COS_ALL_VALID}')

    print()
    if failures:
        print('检查未通过，具体是：')
        for item in failures:
            print(f'  - {item}')
        return 1
    print('全部检查通过。')
    print('注意：这里没有断言"变长批量 == 逐条推理"。')
    print('实测二者不可能逐位等价（补齐改变总帧数、边界帧感受野不同），')
    print('所以上面记录的是下限，而不是等号。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
