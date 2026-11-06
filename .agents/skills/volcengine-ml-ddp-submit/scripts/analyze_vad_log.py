#!/usr/bin/env python3
"""从 VAD 最新连续 run 的 Epoch/Batch 日志稳健估算训练时长。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import statistics
from typing import Optional


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)")
BATCH_RE = re.compile(r"Epoch\s+(\d+),\s*Batch\s+(\d+)/(\d+)")
RESTART_RE = re.compile(r"Starting new training run", re.IGNORECASE)


@dataclass(frozen=True)
class Point:
    timestamp: datetime
    epoch: int
    batch: int
    total_batches: int
    timestamp_layers: int


def parse_timestamp(raw: str) -> datetime:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in raw else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(raw, fmt)


def timestamp_index(source: str, count: int) -> int:
    if source == "outer":
        return 0
    if source == "inner":
        return count - 1
    if source.startswith("index:"):
        index = int(source.split(":", 1)[1])
        if index < 0:
            index += count
        if 0 <= index < count:
            return index
    raise ValueError(f"timestamp source {source!r} 不适用于 {count} 层时间戳")


def parse_runs(path: Path, source: str, line_filter: Optional[str]) -> list[list[Point]]:
    filter_re = re.compile(line_filter) if line_filter else None
    runs: list[list[Point]] = []
    current: list[Point] = []

    def finish() -> None:
        nonlocal current
        if current:
            runs.append(current)
            current = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if RESTART_RE.search(line):
            finish()
            continue
        if filter_re and not filter_re.search(line):
            continue
        batch_match = BATCH_RE.search(line)
        timestamps = DATE_RE.findall(line)
        if batch_match is None or not timestamps:
            continue
        try:
            index = timestamp_index(source, len(timestamps))
        except (ValueError, IndexError) as exc:
            raise SystemExit(str(exc)) from exc
        epoch, batch, total = map(int, batch_match.groups())
        if total <= 0 or epoch <= 0 or not 0 <= batch <= total:
            raise ValueError("日志中的epoch、batch或总batch无效")
        point = Point(parse_timestamp(timestamps[index]), epoch, batch, total, len(timestamps))
        if current:
            previous = current[-1]
            if point.timestamp < previous.timestamp or point.epoch < previous.epoch:
                finish()
            elif point.epoch == previous.epoch and point.batch < previous.batch:
                finish()
        current.append(point)
    finish()
    return runs


def latest_monotonic_segment(points: list[Point], epoch: int) -> list[Point]:
    selected = [point for point in points if point.epoch == epoch]
    if not selected:
        raise ValueError(f"所选 run 中没有 Epoch {epoch}")
    segments: list[list[Point]] = [[]]
    for point in selected:
        current = segments[-1]
        if current and (
            point.batch <= current[-1].batch
            or point.total_batches != current[-1].total_batches
            or point.timestamp <= current[-1].timestamp
        ):
            segments.append([])
            current = segments[-1]
        current.append(point)
    return segments[-1]


def seconds_per_batch(first: Point, last: Point) -> float:
    delta = last.batch - first.batch
    if delta <= 0:
        raise ValueError("batch 必须单调增加")
    return (last.timestamp - first.timestamp).total_seconds() / delta


def rolling_rates(points: list[Point], span: int) -> list[float]:
    rates = []
    for index, first in enumerate(points[:-1]):
        for last in points[index + 1 :]:
            if last.batch - first.batch >= span:
                rates.append(seconds_per_batch(first, last))
                break
    return rates


def recent_rate(points: list[Point], recent_batches: int) -> float:
    last = points[-1]
    target = last.batch - recent_batches
    candidates = [point for point in points[:-1] if point.batch <= target]
    return seconds_per_batch(candidates[-1] if candidates else points[0], last)


def remaining_batch_count(point: Point, base: int) -> int:
    if base not in (0, 1) or not base <= point.batch < point.total_batches + base:
        raise ValueError("batch超出声明的0/1-based范围")
    return point.total_batches - (point.batch - base + 1)


def format_hours(hours: float) -> str:
    total_minutes = round(hours * 60)
    return f"{total_minutes // 60}小时{total_minutes % 60:02d}分"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--run-index", type=int, default=-1, help="连续 run 索引；默认最后一个")
    parser.add_argument("--epoch", type=int, help="所选 run 内的 epoch；默认最后出现的 epoch")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--recent-batches", type=int, default=200)
    parser.add_argument(
        "--timestamp-source",
        default="inner",
        help="outer、inner 或 index:N；平台聚合日志默认优先原始 inner logger 时间",
    )
    parser.add_argument("--line-filter", help="只保留匹配此正则的 worker/rank 日志行")
    parser.add_argument(
        "--batch-index-base",
        choices=("auto", "0", "1"),
        default="auto",
        help="当前 VAD trainer 为 0-based；auto 看到 batch0 时确认，否则仍按 0-based 并提示",
    )
    args = parser.parse_args()

    if args.window <= 0 or args.recent_batches <= 0:
        parser.error("--window 和 --recent-batches 必须为正整数")
    if not args.log.is_file():
        raise SystemExit(f"日志不存在: {args.log}")
    runs = parse_runs(args.log, args.timestamp_source, args.line_filter)
    if not runs:
        raise SystemExit("没有可解析的 VAD batch 日志")
    try:
        run = runs[args.run_index]
    except IndexError as exc:
        raise SystemExit(f"--run-index 越界，共 {len(runs)} 个 run") from exc
    epoch = args.epoch if args.epoch is not None else run[-1].epoch
    try:
        points = latest_monotonic_segment(run, epoch)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(points) < 2:
        raise SystemExit(f"Epoch {epoch} 的最新单调区间不足 2 条日志")

    first, last = points[0], points[-1]
    overall = seconds_per_batch(first, last)
    recent = recent_rate(points, args.recent_batches)
    rates: dict[str, float] = {"全观察区间": overall, "近期区间": recent}
    for window in sorted({args.window, 100}):
        values = rolling_rates(points, window)
        if values:
            rates[f"{window}-batch 窗口中位数"] = statistics.median(values)

    candidates = list(rates.values())
    center, lower, upper = statistics.median(candidates), min(candidates), max(candidates)
    if args.batch_index_base == "1":
        base = 1
    else:
        base = 0
    if args.batch_index_base == "auto" and not any(point.batch == 0 for point in run):
        print("batch index auto: 未观察到 batch0；按当前 VAD trainer 的 0-based 合同计算")
    try:
        for point in points:
            remaining_batch_count(point, base)
        remaining_batches = remaining_batch_count(last, base)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    epoch_hours = last.total_batches * center / 3600
    epoch_lower = last.total_batches * lower / 3600
    epoch_upper = last.total_batches * upper / 3600
    remaining_hours = remaining_batches * center / 3600
    remaining_lower = remaining_batches * lower / 3600
    remaining_upper = remaining_batches * upper / 3600

    layer_counts = sorted({point.timestamp_layers for point in run})
    print(
        f"runs={len(runs)}; selected={args.run_index}; timestamp_source={args.timestamp_source}; "
        f"observed_layers={layer_counts}"
    )
    print(f"Epoch {epoch}: {len(points)} samples, {first.batch} -> {last.batch}/{last.total_batches}")
    print(f"观察区间: {first.timestamp} -> {last.timestamp}")
    for name, value in rates.items():
        print(f"{name}: {value:.3f} 秒/batch")
    print(
        f"单轮纯训练: {format_hours(epoch_hours)} "
        f"（范围 {format_hours(epoch_lower)}～{format_hours(epoch_upper)}）"
    )
    print(
        f"剩余 {remaining_batches} batch: {format_hours(remaining_hours)} "
        f"（范围 {format_hours(remaining_lower)}～{format_hours(remaining_upper)}）"
    )
    eta_lower = last.timestamp + timedelta(hours=remaining_lower)
    eta_upper = last.timestamp + timedelta(hours=remaining_upper)
    print(f"纯训练预计完成: {eta_lower} ～ {eta_upper}（沿用日志时钟，时区未另核实）")
    print("提示: 未包含轮末 validation、checkpoint 和平台调度开销；聚合日志应使用 --line-filter 固定 rank。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
