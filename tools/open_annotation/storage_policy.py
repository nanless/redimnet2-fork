"""标注大数据默认落盘策略：统一落到 group-shared/voiceprint 的两个基址。

2026-11-09 起用户硬性规定：/root（及 /tmp）只放代码仓库与临时日志，不放数据。
- 开源标注侧车与整轮 run 产物 → ``/root/group-shared/voiceprint/open_annotation/redimnet2-silver-<日期>/``
- 准入材料、账本、journal、证据包 → ``/root/group-shared/voiceprint/artifacts/redimnet2/<用途>/``

本模块只给默认基址和「是否落在旧数据根」的判定，供新入口与测试使用。
运行期显式传入的路径优先；这里不改写调用者已给的绝对路径。
注意 ``/root/group-shared`` 是指向 ``/shared/group-shared`` 的符号链接，而
``resume_existing_annotation.physical`` 等校验器拒绝符号链接祖先，写进
spec 的路径必须用无符号链接的 ``/shared/group-shared/...`` 拼法。
"""
from __future__ import annotations
from pathlib import Path

GROUP_SHARED = Path('/root/group-shared/voiceprint')
GROUP_SHARED_PHYSICAL = Path('/shared/group-shared/voiceprint')
OPEN_ANNOTATION = GROUP_SHARED / 'open_annotation'
ARTIFACTS = GROUP_SHARED / 'artifacts' / 'redimnet2'
SILVER_PREFIX = 'redimnet2-silver-'

# 旧数据根：只允许历史记录与带标记的例外，不再作为默认落盘位置。
FORBIDDEN_DATA_ROOTS = ('/root/experiments', '/root/codex_artifacts', '/root/annotations')
# /root 下非数据用途的固定前缀：解释器/环境与共享盘别名，可用于 spec 值。
NON_DATA_ROOT_PREFIXES = ('/root/miniforge3/', '/root/group-shared/voiceprint/')


def run_root(date):
    """整轮 run 与侧车根：date 为 YYYYMMDD，返回 .../open_annotation/redimnet2-silver-<date>。"""
    text = str(date)
    if len(text) != 8 or not text.isdigit():
        raise ValueError('silver run date must be YYYYMMDD: ' + text)
    return OPEN_ANNOTATION / (SILVER_PREFIX + text)


def artifact_root(purpose):
    """准入材料/账本/证据包根：purpose 为用途目录名，如 native-speed-successor-20261028-v1。"""
    raw = str(purpose)
    text = raw.strip('/')
    if not text or raw.startswith('/') or '..' in Path(text).parts:
        raise ValueError('artifact purpose must be a relative directory name: ' + repr(purpose))
    return ARTIFACTS / text


def forbidden(path):
    """该绝对路径是否落在旧数据根（默认落盘不得使用）。"""
    text = str(path)
    return any(text == root or text.startswith(root + '/') for root in FORBIDDEN_DATA_ROOTS)


def physical(path):
    """把 /root/group-shared 别名换成无符号链接的 /shared/group-shared 拼法。"""
    text = str(path)
    alias = str(GROUP_SHARED)
    if text == alias or text.startswith(alias + '/'):
        return GROUP_SHARED_PHYSICAL.joinpath(text[len(alias):].lstrip('/'))
    return Path(text)
