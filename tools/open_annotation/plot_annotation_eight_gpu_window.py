"""只用已去身份的公开聚合数据画八卡利用率，不访问服务器或重新查询GPU。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(data: Path, output: Path, font_file: Path | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    value = json.loads(data.read_text(encoding='utf-8'))
    if value['schema'] != 'annotation-public-eight-GPU-120s/v1':
        raise ValueError('必须提供明确的公开八卡窗口数据')
    rows, samples = value['per_GPU'], value['samples']
    if [row['gpu_index'] for row in rows] != list(range(8)) or not samples:
        raise ValueError('必须包含按实际索引排序的八卡和非空采样')
    # 图只描述这个有限窗口，不把它改写成全程或同批实验结论。
    if value['same_cohort_benchmark'] is not False or value['whole_job_mean'] is not False:
        raise ValueError('此图不能冒充同批比较或全程平均值')
    times = [row['relative_seconds'] for row in samples]
    values = [row['GPU_values_percent'] for row in samples]
    if times != sorted(times) or any(not 0 <= t < value['window_seconds'] for t in times):
        raise ValueError('采样时间顺序或窗口不同')
    if any(len(row) != 8 or any(type(n) not in (int, float) or not 0 <= n <= 100 for n in row) for row in values):
        raise ValueError('八卡利用率必须是0至100的有效数值')
    font_file = font_file or Path('/System/Library/Fonts/STHeiti Medium.ttc')
    if not font_file.is_file():
        raise ValueError('请用--font-file指定本机已安装的中文字体；不联网安装字体')
    font = FontProperties(fname=str(font_file))
    mean = value['overall_per_timestamp_eight_GPU_mean']['mean_percent']
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    left, right = axes
    means = [row['mean_percent'] for row in rows]
    left.bar(range(8), means, color='#278b83', width=.66)
    left.axhline(mean, color='#335a85', linestyle='--', linewidth=1.2)
    left.set_xticks(range(8), [f'GPU{i}' for i in range(8)])
    left.set_ylim(0, 100)
    left.set_title('每张卡的窗口平均利用率', fontproperties=font)
    left.set_ylabel('利用率（%）', fontproperties=font)
    for i, number in enumerate(means):
        left.text(i, number + 2, f'{number:.1f}', ha='center', fontsize=10)
    right.fill_between(times, [min(row) for row in values], [max(row) for row in values],
                       color='#bcd8e6', alpha=.55, label='同一时刻八卡范围')
    right.plot(times, [sum(row) / 8 for row in values], color='#225f8e', linewidth=1.7, label='八卡平均值')
    right.set_xlim(0, value['window_seconds'])
    right.set_ylim(0, 105)
    right.set_xlabel('从窗口开始计算的秒数', fontproperties=font)
    right.set_title('实际连续采样；阴影为八卡范围', fontproperties=font)
    right.legend(prop=font, loc='upper right', fontsize=9)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle(f'正式八卡标注已经运行：120秒窗口平均 {mean:.2f}%', fontproperties=font, fontsize=16)
    fig.text(.5, .075, '八个工作进程首片均已完成后取窗；98个时间采样、784个GPU观测，缺测为0。',
             ha='center', fontproperties=font, fontsize=10)
    fig.text(.5, .025, '这是当前数据分布下的短窗口，不是同批对照或全程均值；瞬时0%不表示设备可被其他进程使用。',
             ha='center', fontproperties=font, fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    assets = Path(__file__).resolve().parents[2] / 'docs/reviews/assets/2026-10-28'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=assets / 'native-eight-GPU-firstchunk-120s-public.json')
    parser.add_argument('--output', type=Path, default=assets / 'native-eight-GPU-firstchunk-120s.png')
    parser.add_argument('--font-file', type=Path, help='本机已有的中文字体；默认使用macOS黑体')
    args = parser.parse_args()
    plot(args.data, args.output, args.font_file)
