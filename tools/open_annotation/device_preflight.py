"""v4 入口的 st_dev 预检：零写入，早于任何 cohort / 租约 / 侧车。

为什么需要它（2026-11-01 停摆与本方案续跑的共同前提）：
侧车里 `source_identity.stat_after` 记录 `device = 2097190`（集群 GPFS），
而 `sidecar.lookup()` 用 `stat_identity()` 做**整字典相等**比较。容器若换了
挂载（dev 主机读到 `device = 57`，inode/bytes/mtime/ctime 全同），`lookup()`
会在第一条孤儿源上抛 `source changed since sidecar publication` —— 与 11-01
同形态的"启动即崩"。

放在正式入口里做，是为了省掉一次单独的探针任务：反正 v4 本来就要建、要提交，
提交那一刻顺手把三个已知文件的 `st_dev` 量出来，比事后补探针便宜且更可信
（量的是**同一个容器、同一次挂载**）。

约束：
* 只 `stat` + 只写 stdout；不创建目录、不写文件、不开锁、不碰队列。
* 由 `gpu_launch.main()` 在 `run()` 之前调用，因此 exit 21 时：
  未 claim 任何租约、未打开任何 cohort、未发布任何侧车、无半成品。
* 不匹配时打印可 grep 的 `PREFLIGHT_DEVICE_MISMATCH expected=… actual=…`
  并以 **21** 退出（与 `__main__` 兜底的 2 区分开）。
"""
from __future__ import annotations
import json
import os
import sys

SCHEMA = 'annotation-device-preflight/v1'
EXPECTED_DEVICE = 2097190
EXIT_MISMATCH = 21
MISMATCH_MARKER = 'PREFLIGHT_DEVICE_MISMATCH'

# 三个已知源：两个音频（分别被"孤儿侧车"和"已提交分片"覆盖）+ 一个同 GPFS 上的
# 只读文件（验证模型挂载与数据挂载的 st_dev 一致，且路径前缀不同也仍然成立）。
SAMPLES = (
    {'role': 'orphan_source',
     'path': '/root/group-shared/voiceprint/data/speech/speaker_verification/VoxBlink2/audio/id104645/SnHjF5uK3V4/00001.wav',
     'note': 'attempt1 已发布但未提交的侧车（744 条之一）覆盖此文件',
     'recorded': {'device': 2097190, 'inode': 26453836791, 'bytes': 367438}},
    {'role': 'committed_source',
     'path': '/root/group-shared/voiceprint/data/speech/speaker_verification/VoxBlink2/audio/id013397/7p-9eMvF29A/00007.wav',
     'note': 'three-train-24196 已提交分片覆盖此文件',
     'recorded': {'device': 2097190, 'inode': 2870920464, 'bytes': 64078}},
    {'role': 'group_shared_readonly',
     'path': '/shared/group-shared/voiceprint/models/a800_2gpus/code/own/download_xiaoxue_online_audios_20260428/models/Qwen3-ASR-1.7B/config.json',
     'note': '非音频只读文件；走 /shared 前缀，验证与数据路径同一文件系统',
     'recorded': None},
)


def probe(samples=SAMPLES, *, stat=None):
    """对每个样本做一次只读 stat，返回逐条结果；永不抛（异常记进 error 字段）。"""
    stat = stat or os.stat
    rows = []
    for sample in samples:
        row = {'role': sample['role'], 'path': sample['path'], 'note': sample.get('note'),
               'recorded': sample.get('recorded'),
               'st_dev': None, 'st_ino': None, 'st_size': None,
               'device_ok': False, 'error': None}
        try:
            value = stat(sample['path'])
            row['st_dev'], row['st_ino'], row['st_size'] = value.st_dev, value.st_ino, value.st_size
            row['device_ok'] = value.st_dev == EXPECTED_DEVICE
        except OSError as error:
            row['error'] = type(error).__name__ + ': ' + str(error)
        rows.append(row)
    return rows


def check(samples=SAMPLES, *, stat=None, out=None):
    """打印一行 JSON + 必要时一行 mismatch；返回 0 或 21。除了 stdout 不写任何东西。"""
    out = out or sys.stdout
    rows = probe(samples, stat=stat)
    out.write(json.dumps({'schema': SCHEMA, 'expected_device': EXPECTED_DEVICE,
                          'samples': rows}, ensure_ascii=False, sort_keys=True) + '\n')
    out.flush()
    bad = [row for row in rows if not row['device_ok']]
    if not bad:
        return 0
    actual = ','.join('%s:%s' % (row['role'], 'missing' if row['st_dev'] is None else row['st_dev'])
                      for row in bad)
    out.write('%s expected=%d actual=%s\n' % (MISMATCH_MARKER, EXPECTED_DEVICE, actual))
    out.flush()
    return EXIT_MISMATCH
