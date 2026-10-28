"""绘制新16源有限试验的公开汇总图，不读取音频、侧车或私有样本ID。

从仓库公开的聚合JSON读取数字。字体只使用系统已有中文字体，不联网、
不安装依赖；跨主机可显式传入已有中文字体。运行示例：
python tools/open_annotation/plot_native_feature_benchmark.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(data_path: Path, output: Path, font_file: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont

    data = json.loads(data_path.read_text(encoding="utf-8"))
    if data["schema"] != "annotation-native-feature-public-benchmark/v1":
        raise ValueError("必须使用明确的新16源公开汇总协议")
    profiles = data["profiles"]
    if [row["profile"] for row in profiles] != ["B1", "B4", "B8"]:
        raise ValueError("必须保留B1、B4、B8的同协议顺序")
    if font_file is None:
        candidates = (
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
        )
        font_file = next((p for p in candidates if p.is_file()), None)
    if font_file is None:
        # 仅查本机已经安装的字体；失败时明确拒绝缺字图。
        for name in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei"):
            try:
                font_file = Path(findfont(name, fallback_to_default=False))
                break
            except ValueError:
                continue
    if font_file is None or not font_file.is_file():
        raise ValueError("未找到现有中文字体，请用--font-file指定；不自动安装字体")
    font = FontProperties(fname=str(font_file))
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    colors = ["#758399", "#2d86b8", "#229773"]
    labels = [p["profile"] for p in profiles]
    series = (
        ([p["cold_wall_seconds"] / 60 for p in profiles], "处理同一批3000条音频的完整耗时", "分钟", 34),
        ([sum(p["steady_gpu_mean_percent"]) / 2 for p in profiles], "相同口径下的双卡稳态利用率", "利用率（%）", 70),
    )
    for axis, (values, title, ylabel, upper) in zip(axes, series):
        axis.bar(labels, values, color=colors, width=0.6)
        axis.set_title(title, fontproperties=font, fontsize=13)
        axis.set_ylabel(ylabel, fontproperties=font, fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_ylim(0, upper)
        for i, value in enumerate(values):
            axis.text(i, value + 0.65, f"{value:.2f}", ha="center", fontsize=12)
    fig.suptitle("先保留每条音频的官方B1特征，再合批转写", fontproperties=font, fontsize=17)
    fig.text(0.5, 0.08, "B1每批1个窗口，B4最多4个窗口，B8最多8个窗口；三组均使用相同修正协议。",
             ha="center", fontproperties=font, fontsize=11)
    fig.text(0.5, 0.035, "3000个来源、3067个窗口：有效输入、文字、有效token、字界和判定均无变化。B4提速1.644倍，B8提速1.943倍。",
             ha="center", fontproperties=font, fontsize=10)
    fig.text(0.5, -0.005, "稳态排除加载、首片和末尾不均衡；本图不代表人工字界金标通过，也不代替正式接续验收。",
             ha="center", fontproperties=font, fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    assets = root / "docs/reviews/assets/2026-10-28"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=assets / "new16-main-public-benchmark.json")
    parser.add_argument("--output", type=Path, default=assets / "new16-main-speed-util-and-quality.png")
    parser.add_argument("--font-file", type=Path)
    args = parser.parse_args()
    plot(args.data, args.output, args.font_file)


if __name__ == "__main__":
    main()
