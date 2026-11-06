#!/usr/bin/env python3
"""从 rank-0 PSE 日志的最新连续 run 估算吞吐与可证明的 epoch ETA。"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import statistics
from pathlib import Path
from typing import Optional


TRAIN_STATE_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} [^]]+)\].*?"
    r"\[TrainState\]\s+step=(?P<step>\d+)"
)
RESTART_RE = re.compile(r"Starting new training run", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="rank-0 持久训练日志")
    parser.add_argument("--steps-per-epoch", type=int, help="每轮 optimizer step 数")
    parser.add_argument(
        "--epoch-start-step",
        type=int,
        help="当前 epoch 起点的 global optimizer step；后续 epoch ETA 必须提供",
    )
    parser.add_argument("--window-steps", type=int, default=5000)
    parser.add_argument(
        "--log-timezone",
        help="日志 naive 时间戳的 IANA 时区，如 UTC 或 Asia/Shanghai；省略表示未知",
    )
    parser.add_argument("--display-timezone", default="Asia/Shanghai")
    return parser.parse_args()


def parse_timestamp(raw: str) -> dt.datetime:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in raw else "%Y-%m-%d %H:%M:%S"
    return dt.datetime.strptime(raw, fmt)


def load_segments(path: Path) -> list[list[tuple[int, dt.datetime]]]:
    segments: list[list[tuple[int, dt.datetime]]] = []
    current: list[tuple[int, dt.datetime]] = []

    def finish() -> None:
        nonlocal current
        if current:
            segments.append(current)
            current = []

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if RESTART_RE.search(line):
                finish()
                continue
            match = TRAIN_STATE_RE.search(line)
            if not match:
                continue
            try:
                sample = (
                    int(match.group("step")),
                    parse_timestamp(match.group("timestamp")),
                )
            except ValueError:
                continue
            if current:
                previous_step, previous_time = current[-1]
                if sample[0] < previous_step or sample[1] < previous_time:
                    finish()
                elif sample[0] == previous_step:
                    current[-1] = sample  # 同一 segment 内保留该 step 的最后时间戳
                    continue
            current.append(sample)
    finish()
    return segments


def interval_stats(samples: list[tuple[int, dt.datetime]]):
    intervals = []
    for (step_a, time_a), (step_b, time_b) in zip(samples, samples[1:]):
        delta_steps = step_b - step_a
        delta_seconds = (time_b - time_a).total_seconds()
        if delta_steps > 0 and delta_seconds >= 0:
            intervals.append((step_a, step_b, delta_steps, delta_seconds))
    return intervals


def fmt_seconds(seconds: float) -> str:
    hours, remainder = divmod(int(round(max(0.0, seconds))), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def display_time(value: dt.datetime, log_timezone: Optional[str], display_timezone: str) -> str:
    if not log_timezone:
        return f"{value:%Y-%m-%d %H:%M:%S} (时区未知)"
    try:
        from zoneinfo import ZoneInfo

        source = ZoneInfo(log_timezone)
        target = ZoneInfo(display_timezone)
    except Exception as exc:
        raise SystemExit(f"无效时区: {exc}") from exc
    return value.replace(tzinfo=source).astimezone(target).isoformat(sep=" ", timespec="seconds")


def main() -> int:
    args = parse_args()
    if not args.log.is_file():
        raise SystemExit(f"日志不存在: {args.log}")
    segments = load_segments(args.log)
    if not segments:
        raise SystemExit("没有找到至少含两个单调 [TrainState] 样本的连续 run")
    samples = segments[-1]
    if len(samples) < 2:
        raise SystemExit("最新连续 run 不足两个样本；不使用旧 run 估算当前 ETA")
    if args.window_steps <= 0:
        raise SystemExit("--window-steps 必须为正整数")
    intervals = interval_stats(samples)
    if not intervals:
        raise SystemExit("最新 run 没有单调递增的 step 区间")

    first_step, first_time = samples[0]
    last_step, last_time = samples[-1]
    total_steps = last_step - first_step
    total_seconds = (last_time - first_time).total_seconds()
    weighted = total_seconds / total_steps
    med = statistics.median(seconds / steps for _, _, steps, seconds in intervals)

    window_start = max(first_step, last_step - max(1, args.window_steps))
    window = [item for item in intervals if item[1] > window_start]
    window_steps = sum(item[2] for item in window)
    window_seconds = sum(item[3] for item in window)
    recent = window_seconds / window_steps if window_steps else weighted

    print(f"日志: {args.log}")
    print(f"连续 run segments={len(segments)}; 使用最后一段，samples={len(samples)}")
    print(f"step: {first_step} -> {last_step}; raw time: {first_time} -> {last_time}")
    print(
        f"显示时间: {display_time(first_time, args.log_timezone, args.display_timezone)} -> "
        f"{display_time(last_time, args.log_timezone, args.display_timezone)}"
    )
    print(f"观测耗时: {fmt_seconds(total_seconds)}")
    print(f"全区间加权秒/step: {weighted:.3f}")
    print(f"样本区间中位秒/step: {med:.3f}")
    print(f"最近 {window_steps} step 秒/step: {recent:.3f}")

    slow = sorted(intervals, key=lambda item: item[3] / item[2], reverse=True)[:5]
    print("最慢区间:")
    for step_a, step_b, delta_steps, delta_seconds in slow:
        print(f"  {step_a}->{step_b}: {delta_seconds:.1f}s, {delta_seconds / delta_steps:.3f}s/step")

    if args.steps_per_epoch is None:
        print("ETA: 请提供 --steps-per-epoch；不要从 scheduler 周期推断")
        return 0
    if args.steps_per_epoch <= 0:
        raise SystemExit("--steps-per-epoch 必须为正整数")

    epoch_start = args.epoch_start_step
    if epoch_start is None:
        print("ETA: 请提供 --epoch-start-step；当前global step较小也不能证明这是从0开始的第一轮")
        return 0
    if epoch_start < 0:
        raise SystemExit("--epoch-start-step 不能为负")
    progress = last_step - epoch_start
    if progress < 0 or progress > args.steps_per_epoch:
        print(
            f"ETA: 无效 baseline，current progress={progress}, "
            f"steps_per_epoch={args.steps_per_epoch}"
        )
        return 0

    remaining = args.steps_per_epoch - progress
    central = remaining * recent
    low = remaining * min(recent, weighted)
    high = remaining * max(recent, weighted)
    eta = last_time + dt.timedelta(seconds=central)
    print(f"epoch progress: {progress}/{args.steps_per_epoch}; remaining={remaining}")
    print(f"单轮纯训练估算: {fmt_seconds(args.steps_per_epoch * recent)}")
    print(f"剩余: {fmt_seconds(low)} - {fmt_seconds(high)}; 中心值 {fmt_seconds(central)}")
    print(f"中心 ETA: {display_time(eta, args.log_timezone, args.display_timezone)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
