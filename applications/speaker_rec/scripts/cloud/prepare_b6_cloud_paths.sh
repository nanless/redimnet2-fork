#!/usr/bin/env bash
# 让清单内部 /root/group-shared 与 /shared/group-shared 两类绝对路径均可读取。
# 只创建目录别名；不改训练表、不复制音频、不覆盖已有非空目录或错误软链接。
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHARED_DATA_MOUNT="${SHARED_DATA_MOUNT:-/shared/group-shared}"
export SHARED_DATA_MOUNT
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["SHARED_DATA_MOUNT"])
if not source.is_absolute() or not source.is_dir():
    raise SystemExit(f"共享数据挂载不存在或不是绝对目录: {source}")
source = source.resolve(strict=True)
# 先验证这是目标训练数据所在的盘，避免给其他空挂载创建别名。
for relative in ("voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/data.list",
                 "voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join_lm/data.list",
                 "voiceprint/data/wespeaker_data/raw/musan/music",
                 "voiceprint/data/wespeaker_data/raw/rirs/simulated_rirs"):
    # 增强目录可能经由尚待创建的 /root/group-shared 绝对链接；此处先验证目录项存在。
    path = source / relative
    if "raw/" in relative:
        # 检查其真实源目录，不能因别名尚未创建而误判挂载为空。
        actual = source / ("voiceprint/data/noise/musan/musan/music" if "musan" in relative else
                           "voiceprint/data/noise/rirs_noises/RIRS_NOISES/simulated_rirs")
        if not actual.is_dir():
            raise SystemExit(f"增强数据目录缺失: {actual}")
    elif not path.is_file() or not os.access(path, os.R_OK):
        raise SystemExit(f"训练表不可读: {path}")

for target in (Path("/shared/group-shared"), Path("/root/group-shared")):
    if target == source:
        print(f"共享盘已挂载: {target}")
        continue
    # 不能让源目录位于待替换的目标目录内部，否则会构成失效链接。
    if target in source.parents:
        raise SystemExit(f"拒绝操作: 源目录 {source} 位于目标 {target} 内")
    if target.is_symlink():
        if target.resolve() != source:
            raise SystemExit(f"拒绝覆盖不同的软链接: {target} -> {os.readlink(target)}")
        print(f"已有正确目录链接: {target} -> {source}")
        continue
    if target.exists():
        if target.is_dir() and target.resolve() == source:
            print(f"目录已经指向正确共享盘: {target}")
            continue
        if not target.is_dir() or any(target.iterdir()):
            raise SystemExit(f"拒绝覆盖已有非空目录或文件: {target}")
        # rmdir 只接受空目录；并发写入导致其不再为空时会报错退出。
        target.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    print(f"已创建目录链接: {target} -> {source}")

for path in (Path("/root/group-shared/voiceprint/data/wespeaker_data/raw/musan/music"),
             Path("/root/group-shared/voiceprint/data/wespeaker_data/raw/rirs/simulated_rirs")):
    if not path.is_dir():
        raise SystemExit(f"建立目录链接后增强数据仍不可读: {path}")
print("训练表中的两类共享盘绝对路径已具备相同根目录。仍须运行 check_b6_mix_data.py 检查表内音频和评测数据。")
PY
