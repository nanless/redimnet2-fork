#!/usr/bin/env python3
"""标注进度看板 HTTP 服务：浏览器打开即看，页面每 30 秒自刷新。

数据每次请求现算（读 queue/broker.json 与 queue/completed/*.json 的 source_count 求和），
不缓存、不落库，因此页面上的数字始终是请求那一刻的现场。

    python3 tools/annotation_dashboard_server.py --port 8899
    # 本机浏览器： http://127.0.0.1:8899/
    # 内网浏览器： http://10.43.6.199:8899/
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_progress import RETAINED, TOTAL, TARGET_REMAINING, collect, fmt, latest_run  # noqa: E402


def snapshot(run: Path, window: int, task: str | None):
    m = collect(run, window)
    done = m["done_total"]
    overall = RETAINED + done
    rate = m["done_window"] / (window / 60) if window else 0
    left = max(TARGET_REMAINING - done, 0)
    eta_days = left / rate / 24 if rate > 0 else None
    return {
        "task": task,
        "run": str(run),
        "done": done,
        "chunks": m["chunks"],
        "overall": overall,
        "total": TOTAL,
        "pct": overall / TOTAL * 100,
        "left": left,
        "window": window,
        "window_done": m["done_window"],
        "rate": rate,
        "eta_days": eta_days,
        "next": m["next"],
        "completed": m["completed"],
        "leases": m["leases"],
        "workers": m["workers"],
        "failure": m["failure"],
        "quarantine": m["quarantine"],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>标注进度看板</title>
<meta http-equiv="refresh" content="30">
<style>
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;padding:24px;
background:#0f1115;color:#e6e6e6}}
h1{{font-size:20px;margin:0 0 4px}}
.sub{{color:#9aa0a6;font-size:13px;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#171a21;border:1px solid #232733;border-radius:10px;padding:14px 16px}}
.k{{color:#9aa0a6;font-size:12px;margin-bottom:6px}}
.v{{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}}
.v small{{font-size:13px;color:#9aa0a6;font-weight:400}}
.bar{{height:10px;background:#232733;border-radius:6px;overflow:hidden;margin:10px 0 6px}}
.fill{{height:100%;background:linear-gradient(90deg,#3b82f6,#22c55e);width:{pct:.2f}%}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{text-align:left;padding:6px 10px;border-bottom:1px solid #232733}}
th{{color:#9aa0a6;font-weight:500}}
.ok{{color:#22c55e}} .bad{{color:#ef4444}} .warn{{color:#f59e0b}}
</style></head><body>
<h1>标注进度看板</h1>
<div class="sub">任务 {task} · 采样 {ts} · 每 30 秒自动刷新 · 数据现算不缓存</div>

<div class="bar"><div class="fill"></div></div>
<div class="sub">折算总进度 <b>{pct:.2f}%</b>（{overall} / {total}），本轮还剩 {left} 源</div>

<div class="grid">
  <div class="card"><div class="k">本轮已完成</div><div class="v">{done}<small> 源</small></div></div>
  <div class="card"><div class="k">已完成分片</div><div class="v">{chunks}<small> 片</small></div></div>
  <div class="card"><div class="k">最近 {window} 分钟速率</div><div class="v">{rate}<small> 源/小时</small></div></div>
  <div class="card"><div class="k">预计剩余</div><div class="v">{eta}<small> 天</small></div></div>
  <div class="card"><div class="k">worker</div><div class="v">{workers}</div></div>
  <div class="card"><div class="k">在跑分片 / 队列</div><div class="v">{leases} <small>/ next {next}</small></div></div>
</div>

<table>
<tr><th>项</th><th>状态</th></tr>
<tr><td>failure.json</td><td class="{fc}">{ft}</td></tr>
<tr><td>quarantine.json</td><td class="{qc}">{qt}</td></tr>
<tr><td>本轮目录</td><td><code>{run}</code></td></tr>
<tr><td>进度基数说明</td><td>基数 {retained}（1027 保留）；1028 的 4,447,318 源不在本 lineage，会被重算</td></tr>
</table>
</body></html>"""


def render_page(s):
    return PAGE.format(
        task=s["task"] or "（未指定）",
        ts=s["ts"],
        pct=s["pct"],
        overall=fmt(s["overall"]),
        total=fmt(s["total"]),
        left=fmt(s["left"]),
        done=fmt(s["done"]),
        chunks=fmt(s["chunks"]),
        window=s["window"],
        rate=fmt(int(s["rate"])),
        eta=("%.1f" % s["eta_days"]) if s["eta_days"] else "—",
        workers=s["workers"],
        leases=s["leases"],
        next=s["next"],
        fc="bad" if s["failure"] else "ok",
        ft="存在（检查内容）" if s["failure"] else "无",
        qc="warn" if s["quarantine"] else "ok",
        qt="存在（隔离片已移出候选）" if s["quarantine"] else "无",
        run=s["run"],
        retained=fmt(RETAINED),
    )


def make_handler(run, window, task):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data.json"):
                body = json.dumps(snapshot(run, window, task), ensure_ascii=False).encode()
                ctype = "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                body = render_page(snapshot(run, window, task)).encode()
                ctype = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmtstr, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/root/experiments/redimnet2/open-annotation-ultra-20261025")
    ap.add_argument("--run")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    run = Path(args.run) if args.run else latest_run(Path(args.root))
    if run is None or not (run / "queue").is_dir():
        raise SystemExit("找不到 run 目录，用 --run 指定")
    print("serving %s on http://%s:%d/  (window=%dmin)" % (run, args.bind, args.port, args.window), flush=True)
    ThreadingHTTPServer((args.bind, args.port), make_handler(run, args.window, None)).serve_forever()


if __name__ == "__main__":
    main()
