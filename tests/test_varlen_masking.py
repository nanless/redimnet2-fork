#!/usr/bin/env python3
"""变长输入与掩码的回归测试。

这个脚本回答两个问题，并且把答案变成可以自动比较的数字：

1. **向后兼容**：不传 `lengths` 时，模型的行为有没有变？
   基准是"改动前的那份代码"的结果，要求逐位一致（最大绝对差 0.0）。

2. **掩码是否真的在起作用**（两条判据，一起看）：
   * **掩码机制算得对不对**：同一批数据，传一个**全有效**的 `lengths`
     与完全不传 `lengths`，结果应当几乎相同（实测相对 L2 约 2e-06）。
     这条能发现"带掩码的路径把全有效算错了"。
     **但它发现不了"掩码被整个忽略"** —— 那种情况下两条路径完全一样、
     这个数反而变成 0。所以它只是上限检查，下面那条成对比较才是主力。
   * **掩码带来的实际改善**（成对比较）：把一条短音频补零凑进一批，
     传 `lengths` 的结果必须比不传的明显更接近逐条推理。
     这一条是真正能发现"掩码失效"的判据。

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
MIN_COS_ALL_VALID = 0.999      # 同批里较长那条不受补齐影响
# 「掩码机制本身对不对」的判据：同一段音频，全有效掩码 vs 不传掩码，
# 相对 L2 误差应当很小（实测约 5e-07）。
MAX_REL_L2_ALL_VALID = 1e-4
MIN_COS_HALF_VALID = 0.90      # 半有效掩码 vs 逐条（实测约 0.95，留余量）

# 判据要怎么设计：**必须成对比较，不能用单点的绝对阈值，也不能只看"不传掩码有多差"。**
#
# 走过两次弯路，记在这里免得再犯：
#
# 1. 最早写死「不传掩码的余弦 <= 0.6」。不行，因为不同权重的余弦尺度差别很大：
#    实测非对角余弦（同一模型对两段互不相关音频的余弦）在 b0-vox2-lm 上是 0.64，
#    而混合数据那几个权重高达 0.99——后者向量高度相关，余弦被压在很窄的区间里，
#    在那里「0.81」已经是大差异。这个阈值让 5 个混合数据权重误报失败。
#
# 2. 后来改成「不传掩码要比该模型自身的基准低至少 0.12」。**这条方向是反的**：
#    掩码**失效**时，短的那条会被补零污染，「不传掩码」的结果当然是差的、
#    离基准很远，margin 反而更大，于是**照样通过**。
#    实测（b0-vox2-lm，把掩码整个删掉）：掩码正常与掩码失效两种情况的
#    margin 都是 0.312112 —— 也就是说这条判据对"掩码有没有生效"完全没有分辨力。
#
# 所以现在改成**成对比较**：同一批数据、同一个模型，
# **传 lengths 的结果**必须比**不传 lengths 的结果**明显更接近逐条推理。
# 掩码生效时这两个数差很多（实测 0.955 对 0.325），掩码失效时两者相等（都是 0.325），
# 于是这条判据在掩码坏掉时会**确实失败**。
MIN_MASK_BENEFIT = 0.30        # 传 lengths 要比不传至少好这么多（成对比较）


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
        # 全有效掩码：把两条都标成"整条有效"，于是掩码全为 1。
        # 它拿来测的是"走带掩码分支"与"走不带掩码分支"是否一致。
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

    # 「掩码机制本身有没有算错」：同一批数据，传一个**全有效**的 lengths
    # 与完全不传 lengths，结果应当几乎一样（实测相对 L2 约 2e-06）。
    #
    # **这条判据要说清它能测什么、不能测什么**（我在这上面来回错过两次）：
    #   * 能测：带掩码的那条代码路径有没有把"全有效"算错 —— 比如归一化时
    #     用错了统计量、或者下采样掩码时把有效区间缩掉了。算错就会远超 2e-06。
    #   * **不能测**：掩码有没有被整个忽略。实测把掩码删掉之后这个数变成
    #     0.000e+00（两条路径完全一样，当然相等），反而"更好看"。
    #     所以它只是一个**上限**检查，不是"掩码生效"的证据。
    #     真正能发现掩码失效的是下面成对比较的 mask_benefit。
    all_valid_rel_l2 = ((emb_all_valid[0] - emb_no_lengths[0]).norm()
                        / emb_no_lengths[0].norm()).item()

    half_valid = cos(emb[1], ref_short)
    no_lengths = cos(emb_no_lengths[1], ref_short)
    return {
        'all_valid_rel_l2': all_valid_rel_l2,
        # 关键判据：同一批数据、同一个模型，传 lengths 是否明显更接近逐条推理
        'half_valid': half_valid,
        'no_lengths': no_lengths,
        'mask_benefit': half_valid - no_lengths,
        'long_unaffected': cos(emb[0], ref_long),
        # 只作参考：该模型对两段无关音频的余弦，代表这个空间的"区分度尺度"
        'baseline': baseline,
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
        ok_mech = r['all_valid_rel_l2'] <= MAX_REL_L2_ALL_VALID
        ok_half = r['half_valid'] >= MIN_COS_HALF_VALID
        ok_benefit = r['mask_benefit'] >= MIN_MASK_BENEFIT
        ok_long = r['long_unaffected'] >= MIN_COS_ALL_VALID
        print(f'  {name}')
        print(f'    全有效掩码机制（相对L2）: {r["all_valid_rel_l2"]:.3e}  '
              f'(要求 <= {MAX_REL_L2_ALL_VALID:.0e})   '
              f'{"通过" if ok_mech else "未通过"}')
        print(f'    传 lengths   vs 逐条   : {r["half_valid"]:.6f}  '
              f'(要求 >= {MIN_COS_HALF_VALID})          '
              f'{"通过" if ok_half else "未通过"}')
        print(f'    不传 lengths vs 逐条   : {r["no_lengths"]:.6f}  (只作对照)')
        print(f'    掩码带来的改善（成对）  : {r["mask_benefit"]:.6f}  '
              f'(要求 >= {MIN_MASK_BENEFIT})          '
              f'{"通过" if ok_benefit else "未通过"}')
        print(f'    同批里长的那条不受影响 : {r["long_unaffected"]:.6f}  '
              f'(要求 >= {MIN_COS_ALL_VALID})          '
              f'{"通过" if ok_long else "未通过"}')
        if not ok_mech:
            failures.append(
                f'{name}: 全有效掩码 vs 不传掩码的相对 L2 有 '
                f'{r["all_valid_rel_l2"]:.3e}，超过 {MAX_REL_L2_ALL_VALID:.0e} —— '
                f'带掩码的路径把"全有效"算错了')
        if not ok_half:
            failures.append(
                f'{name}: 传 lengths 的余弦 {r["half_valid"]:.6f} < {MIN_COS_HALF_VALID}')
        if not ok_benefit:
            failures.append(
                f'{name}: 传 lengths({r["half_valid"]:.6f}) 并不比不传'
                f'({r["no_lengths"]:.6f}) 好多少，差 {r["mask_benefit"]:.6f}，'
                f'不足 {MIN_MASK_BENEFIT} —— 掩码很可能没有生效')
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
