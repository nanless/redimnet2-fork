"""v4 入口的 st_dev 预检：零写入，早于任何 cohort / 租约 / 侧车。

背景（2026-11-01 停摆 / 2026-11-07 v2 失败）：侧车里 `source_identity.stat_after`
记录 `device = 2097190`（集群 GPFS 命名空间），而容器里同一文件读到 `device = 75`
（另一套挂载命名空间）。inode/bytes/mtime/ctime 完全相同，内容没有变化。

因此 device 只是挂载命名空间的标签，不是内容身份。2026-11-07 v2 生产任务在
`PREFLIGHT_DEVICE_MISMATCH expected=2097190 actual=orphan_source:75,...` 处 exit 21，
任务在 CUDA 之前就停了——预检把"命名空间不同"误判成"文件变了"。

现在的语义：
* device 不符：打印 `PREFLIGHT_DEVICE_MISMATCH`，回执里记
  `device_mismatch_observed:true` / `device_relaxed:true`，继续（不退出）。
* 内容不符（inode / bytes，以及记录里给出时的 mtime_ns / ctime_ns）：
  打印 `PREFLIGHT_IDENTITY_MISMATCH` 并以 exit 21 退出：这说明文件内容确实变了，拒绝。
* 只 `stat` + 只写 stdout；不创建目录、不写文件、不开锁、不碰队列。
* 由 `gpu_launch.main()` 在 `run()` 之前调用，因此 exit 21 时未 claim 任何租约。

`strict=False`（W2 pilot / 调试）保留原行为：device 不符也只打印 NONFATAL 标记。
"""
from __future__ import annotations
import json
import os
import sys

SCHEMA = 'annotation-device-preflight/v1'
EXPECTED_DEVICE = 2097190
EXIT_MISMATCH = 21
MISMATCH_MARKER = 'PREFLIGHT_DEVICE_MISMATCH'
NONFATAL_MARKER = 'PREFLIGHT_DEVICE_MISMATCH_NONFATAL'
IDENTITY_MARKER = 'PREFLIGHT_IDENTITY_MISMATCH'

# 三个已知源：两个音频（分别被"孤儿侧车"和"已提交分片"覆盖）+ 一个同 GPFS 上的
# 只读文件（验证模型挂载与数据挂载在读侧一致，且路径前缀不同也仍然成立）。
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

# 内容身份字段：device 之外的这些字段任何一个变了，才是真的文件变了。
IDENTITY_FIELDS = ('inode', 'bytes', 'mtime_ns', 'ctime_ns')


def probe(samples=SAMPLES, *, stat=None):
    """对每个样本做一次只读 stat，返回逐条结果；不抛异常（异常记进 error 字段）。"""
    stat = stat or os.stat
    rows = []
    for sample in samples:
        row = {'role': sample['role'], 'path': sample['path'], 'note': sample.get('note'),
               'recorded': sample.get('recorded'),
               'st_dev': None, 'st_ino': None, 'st_size': None,
               'st_mtime_ns': None, 'st_ctime_ns': None,
               'device_ok': False, 'identity_ok': True, 'identity_diffs': [], 'error': None}
        try:
            value = stat(sample['path'])
            row.update(st_dev=value.st_dev, st_ino=value.st_ino, st_size=value.st_size,
                       st_mtime_ns=getattr(value, 'st_mtime_ns', None),
                       st_ctime_ns=getattr(value, 'st_ctime_ns', None))
            row['device_ok'] = value.st_dev == EXPECTED_DEVICE
            recorded = sample.get('recorded') or {}
            actual = {'inode': value.st_ino, 'bytes': value.st_size,
                      'mtime_ns': getattr(value, 'st_mtime_ns', None),
                      'ctime_ns': getattr(value, 'st_ctime_ns', None)}
            diffs = [name for name in IDENTITY_FIELDS
                     if recorded.get(name) is not None and actual[name] is not None
                     and recorded[name] != actual[name]]
            row['identity_diffs'] = diffs
            row['identity_ok'] = not diffs
        except OSError as error:
            row['error'] = type(error).__name__ + ': ' + str(error)
            row['identity_ok'] = False
            row['identity_diffs'] = ['stat_error']
        rows.append(row)
    return rows


def check(samples=SAMPLES, *, stat=None, out=None, strict=True, relax_device=True):
    """总是打印一行 JSON；返回 0 或 21。

    relax_device=True（默认，生产 auto8 亦然）：device 不符只记录不退出；
    内容字段不符仍然 exit 21。
    relax_device=False：恢复旧的严格行为（device 不符即 exit 21）。
    strict=False（W2 pilot）：沿用 NONFATAL 标记，且从不因 device 退出。
    除 stdout 外不写任何东西。
    """
    out = out or sys.stdout
    rows = probe(samples, stat=stat)
    bad_device = [row for row in rows if not row['device_ok']]
    bad_identity = [row for row in rows if not row['identity_ok']]
    relaxed = bool(bad_device) and (relax_device or not strict)
    receipt = {
        'schema': SCHEMA,
        'expected_device': EXPECTED_DEVICE,
        'samples': rows,
        'device_mismatch_observed': bool(bad_device),
        'device_relaxed': bool(relaxed),
        'identity_mismatch_observed': bool(bad_identity),
        'device_semantics': 'device_is_mount_namespace_label; content identity = inode/bytes/mtime/ctime',
    }
    out.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + '\n')
    out.flush()

    if bad_identity:
        detail = ','.join('%s:%s' % (row['role'], '+'.join(row['identity_diffs'])) for row in bad_identity)
        out.write('%s expected_content_identity_stable actual=%s\n' % (IDENTITY_MARKER, detail))
        out.flush()
        return EXIT_MISMATCH

    if not bad_device:
        return 0

    actual = ','.join('%s:%s' % (row['role'], 'missing' if row['st_dev'] is None else row['st_dev'])
                      for row in bad_device)
    out.write('%s expected=%d actual=%s\n'
              % (MISMATCH_MARKER if relaxed else NONFATAL_MARKER, EXPECTED_DEVICE, actual))
    out.flush()
    return 0 if relaxed else EXIT_MISMATCH
