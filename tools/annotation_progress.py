#!/usr/bin/env python3
"""标注进度看板：从队列收据与运行目录直接算，不依赖平台接口。

用法：
    python3 tools/annotation_progress.py                       # 打印看板（markdown）
    python3 tools/annotation_progress.py --out docs/ANNOTATION-PROGRESS.md
    python3 tools/annotation_progress.py --run <run 目录> --window 60

为什么不用 `find labels | wc -l`：GPFS 上几十万文件要十几分钟；分片收据只有几千个，
秒级完成，而且统计的是"已提交分片"，不含正在写的半成品。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

TOTAL = 11_521_178
RETAINED = 1_617_437          # 本 lineage 起算基数（1027 保留）
TARGET_REMAINING = TOTAL - RETAINED
TASK_RE = re.compile(r"t-\d{14}-[a-z0-9]+")

# 存储规范（2026-11-09 起）：侧车与 run 目录写在
# /root/group-shared/voiceprint/open_annotation/redimnet2-silver-<日期>/ 下；
# 新基址优先；老代次在 02 根的历史树（2026-11-09 迁移前在 /root/experiments，
# 旧路径已不存在），只读。
SILVER_ROOT = Path("/root/group-shared/voiceprint/open_annotation")
HIST_ROOT = Path("/root/dsh-workspaces-mac/02_实验与标注_ReDimNet2/redimnet2/open-annotation-ultra-20261025")


def candidate_roots(root=None):
    if root:
        return [Path(root)]
    return [SILVER_ROOT, HIST_ROOT]


def latest_run(root: Path):
    # 旧布局 <root>/<handoff>/run-*；新布局 <root>/redimnet2-silver-<日期>/<handoff>/run-*
    cands = sorted(root.glob("*/run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands += sorted(root.glob("*/*/run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def detect_task(run: Path):
    """从 publish-work 的文件名取任务号。

    不读 terminal-*.json 里的 task_id 字段：那个字段已知会错绑旧任务
    （t-20261025054739-4kzhp），只有文件名是真的。
    """
    work = run.parent / "publish-work"
    if not work.is_dir():
        return None
    found = set()
    for p in work.glob("*.json"):
        m = TASK_RE.search(p.name)
        if m:
            found.add(m.group(0))
    return sorted(found)[-1] if found else None


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect(run: Path, window_minutes: int):
    broker = read_json(run / "queue" / "broker.json") or {}
    done_dir = run / "queue" / "completed"
    now = time.time()
    done_total = 0
    done_window = 0
    chunk_count = 0
    if done_dir.is_dir():
        for f in done_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            n = int(d.get("source_count") or 0)
            done_total += n
            chunk_count += 1
            if now - f.stat().st_mtime <= window_minutes * 60:
                done_window += n
    res = run / "resources-attempt1"
    workers = len([p for p in res.glob("worker-*") if p.is_dir()]) if res.is_dir() else 0
    return {
        "done_total": done_total,
        "done_window": done_window,
        "chunks": chunk_count,
        "next": broker.get("next"),
        "completed": broker.get("completed_chunks"),
        "leases": len(broker.get("leases") or {}),
        "workers": workers,
        "failure": (res / "failure.json").exists(),
        "quarantine": (run / "queue" / "quarantine.json").exists(),
        "run": run,
    }


def fmt(n):
    return f"{n:,}"


def render(m, window_minutes: int, task: str | None):
    done = m["done_total"]
    overall = RETAINED + done
    pct = overall / TOTAL * 100
    window_rate = m["done_window"] / (window_minutes / 60) if window_minutes else 0
    left = TARGET_REMAINING - done
    eta_h = left / window_rate if window_rate > 0 else None
    lines = []
    lines.append("# 标注进度看板（自动生成，不手改）")
    lines.append("")
    lines.append("> 生成：`python3 tools/annotation_progress.py --out docs/ANNOTATION-PROGRESS.md`。")
    lines.append("> 数据来源：`queue/broker.json` 与 `queue/completed/*.json` 的 `source_count` 求和；")
    lines.append("> 不用文件计数——GPFS 上几十万文件要十几分钟，而收据只有几千个。")
    lines.append("> 存储规范（2026-11-09 起）：新 run 根写 "
                 "`/root/group-shared/voiceprint/experiments/redimnet2/<实验名>/`，侧车写 "
                 "`.../open_annotation/redimnet2-silver-<日期>/`；四代均已归档；2026-11-09 迁移前的旧树在 02 根（历史路径），"
                 "见 `docs/20-标注结果归档与跳过规则.md`。")
    lines.append("")
    lines.append("## 1. 总量")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    if task:
        lines.append(f"| 任务 | `{task}` |")
    lines.append(f"| 本轮目录 | `{m['run']}` |")
    lines.append(f"| 分母（三域全量） | {fmt(TOTAL)} |")
    lines.append(f"| 本 lineage 起算基数 | {fmt(RETAINED)}（1027 保留） |")
    lines.append(f"| 本轮已完成 | **{fmt(done)}** 源 / {fmt(m['chunks'])} 分片 |")
    lines.append(f"| 折算总进度 | {fmt(overall)} / {fmt(TOTAL)} = **{pct:.2f}%** |")
    lines.append(f"| 本轮还剩 | {fmt(max(left, 0))} 源 |")
    lines.append("")
    lines.append("## 2. 速率与预计")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 最近 {window_minutes} 分钟 | {fmt(m['done_window'])} 源 → **{fmt(int(window_rate))} 源/小时** |")
    if eta_h:
        lines.append(f"| 按此速率跑完本轮 | **{eta_h/24:.1f} 天**（{eta_h:.0f} 小时） |")
    else:
        lines.append("| 按此速率跑完本轮 | 未能计算（窗口内无新增） |")
    lines.append("")
    lines.append("## 3. 队列与进程")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| broker next / completed_chunks | {m['next']} / {m['completed']} |")
    lines.append(f"| 正在处理的分片（leases） | {m['leases']} |")
    lines.append(f"| worker 目录数 | {m['workers']} |")
    lines.append(f"| 失败记录 `failure.json` | {'**存在**' if m['failure'] else '无'} |")
    lines.append(f"| 隔离片 `quarantine.json` | {'**存在**' if m['quarantine'] else '无'} |")
    lines.append("")
    lines.append("## 4. 怎么读这张表")
    lines.append("")
    lines.append("- **折算总进度**用 1,617,437 作基数：新生产者的 lineage 从 1027 保留量起算，")
    lines.append("  1028-v1 的 4,447,318 源不在本 lineage 内、会被重算。所以 52.64% 与本表的百分比")
    lines.append("  不是同一分母口径，不能直接相减。")
    lines.append("- **重启的代次**：11-08 的 v2–v5 四代都没产出侧车（v5 的 `run-v5` 是空目录），")
    lines.append("  v6 才重新开始计数；本表只算当前 run 目录里的分片收据。")
    lines.append("- **速率**只看最近一个窗口：整体平均会把早期的低速和现在的稳态混在一起，")
    lines.append("  判断“还要多久”用窗口速率更准。")
    lines.append("- 出现 `failure.json` 或 `quarantine.json` 时不等于停跑，先看内容：")
    lines.append("  隔离片是失败超阈值的分片被移出候选，其他片照常推进。")
    lines.append("- 任务号默认从 `publish-work/observation-t-*.json` 的文件名取；")
    lines.append("  `terminal-*.json` 里的 `task_id` 字段会错绑旧任务，不读它。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"采样时刻：{time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="侧车或旧实验根；省略时先找新基址"
                    "（/root/group-shared/voiceprint/open_annotation），找不到再回退 02 根的历史树")
    ap.add_argument("--run", help="直接指定 run 目录；省略则取最新")
    ap.add_argument("--window", type=int, default=60, help="速率窗口（分钟，默认 60）")
    ap.add_argument("--task", help="任务 ID（省略则从 publish-work 文件名自动取）")
    ap.add_argument("--out", help="写入文件；省略则打印到标准输出")
    args = ap.parse_args()

    if args.run:
        run = Path(args.run)
    else:
        run = next((r for r in (latest_run(x) for x in candidate_roots(args.root)) if r), None)
    if run is None or not (run / "queue").is_dir():
        raise SystemExit("找不到可读的 run 目录，用 --run 指定一个含 queue/ 的目录")
    task = args.task or detect_task(run)
    text = render(collect(run, args.window), args.window, task)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}（{len(text.splitlines())} 行）")
    else:
        print(text)


if __name__ == "__main__":
    main()
