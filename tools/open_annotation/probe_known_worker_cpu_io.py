"""在真实容器只读已登记标注worker的CPU/I/O增量，不枚举进程或使用CUDA。

资源目录来自实际父进程，不接受手填待结束PID。主机名要显式匹配，
避免把共享盘中的云PID映射到开发机/proc。输出只发标准输出，由操作者保存。
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import stat
import time


def need(value, message):
    if not value:
        raise RuntimeError(message)


def small_json(path):
    need(path.resolve() == path, 'identity symlink refused')
    with path.open('rb') as stream:
        facts = os.fstat(stream.fileno())
        need(stat.S_ISREG(facts.st_mode) and facts.st_size <= 65536, 'small regular identity required')
        raw = stream.read(65537)
        need(len(raw) == facts.st_size, 'identity changed while reading')
    return json.loads(raw)


def observe(identity, proc_root=Path('/proc')):
    pid, started = identity.get('container_pid'), identity.get('started')
    need(type(pid) is int and pid > 0 and type(started) is int and started > 0,
         'registered PID/start identity required')
    p = proc_root / str(pid)
    fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
    need(int(fields[19]) == started and fields[0] not in ('Z', 'X'), 'known worker changed/dead')
    row = dict(pid=pid, start_ticks=int(fields[19]), state=fields[0],
               utime_ticks=int(fields[11]), stime_ticks=int(fields[12]),
               wchan=(p / 'wchan').read_text().strip())
    try:
        row['io'] = {line.split(':', 1)[0]: int(line.split(':', 1)[1])
                     for line in (p / 'io').read_text().splitlines()}
    except PermissionError:
        row['io_permission_denied'] = True
    return row


def probe(root, expected_hostname, workers=8, interval=10):
    need(socket.gethostname() == expected_hostname, 'wrong actual cloud container')
    need(type(workers) is int and 1 <= workers <= 8, 'worker count must be 1..8')
    need(type(interval) in (int, float) and 1 <= interval <= 30, 'interval must be 1..30 seconds')
    root = Path(root)
    need(root.is_absolute() and root.is_dir() and root.resolve() == root, 'physical resource root required')
    identities = []
    for i in range(workers):
        identity_path = root / ('worker-' + str(i)) / 'worker_identity.json'
        identity = small_json(identity_path)
        need(identity.get('worker_id') == 'worker-' + str(i), 'registered worker order differs')
        identities.append(identity)
    need(len({v['container_pid'] for v in identities}) == workers and
         len({v['uuid'] for v in identities}) == workers, 'duplicate worker PID or UUID')
    start = time.monotonic()
    before = [observe(v) for v in identities]
    time.sleep(interval)
    after = [observe(v) for v in identities]
    elapsed, hz, rows = time.monotonic() - start, os.sysconf('SC_CLK_TCK'), []
    for identity, a, b in zip(identities, before, after):
        need((a['pid'], a['start_ticks']) == (b['pid'], b['start_ticks']), 'worker changed during read')
        delta = {k: b['io'][k] - a['io'][k] for k in a.get('io', {})} if 'io' in a and 'io' in b else None
        ticks = b['utime_ticks'] + b['stime_ticks'] - a['utime_ticks'] - a['stime_ticks']
        need(ticks >= 0 and (delta is None or all(v >= 0 for v in delta.values())), 'counter went backwards')
        rows.append(dict(worker_id=identity['worker_id'], UUID=identity['uuid'],
                         cpu_percent_one_core_100=100 * ticks / hz / elapsed,
                         io_delta=delta, before=a, after=b))
    return dict(schema='known-worker-readonly-cpu-io/v1',
                observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                hostname=socket.gethostname(), elapsed_seconds=elapsed, rows=rows,
                CUDA_context_created=False, signals_sent=False, proc_enumerated=False,
                meaning='CPU百分比以一个核为100%；字符I/O包含缓存和文件系统语义，块计数为0不证明没有I/O；不定位具体CPU函数。')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources', required=True, type=Path)
    p.add_argument('--expected-hostname', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--interval-seconds', type=float, default=10)
    a = p.parse_args()
    print(json.dumps(probe(a.resources, a.expected_hostname, a.workers, a.interval_seconds),
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
