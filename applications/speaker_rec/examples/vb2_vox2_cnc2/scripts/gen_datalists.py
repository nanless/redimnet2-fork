#!/usr/bin/env python3
"""为一个说话人数据集生成 data.list 与 utt2spk 两张样本清单。

脚本假定音频放在 ``root_dir/<audio-subdir>/`` 下面，并支持两种常见布局：

1. 分层布局（VoxCeleb2 / VoxBlink2 习惯）::

       audio/<说话人>/<视频目录>/<片段>.wav

2. 扁平布局（CN-Celeb2 习惯）::

       audio/<说话人>/<片段>.flac

说话人一律取**相对 ``audio_root`` 的第一段目录名**，所以上面两种布局都能正确处理。

用法::

    python gen_datalists.py <root_dir> <output_dir> \
        --audio-subdir audio --exts .wav

输出::

    output_dir/data.list   每行一条 JSON：{"key": ..., "spk": ..., "wav": ...}
    output_dir/utt2spk     每行两列：样本编号 说话人

说明（为什么写成现在这样）：

* **不再按路径深度过滤。** 旧版本要求相对路径至少有两段（``len(parts) < 2`` 就跳过），
  这条规则会把"说话人目录下直接放音频"的场景**静默丢掉**，只在别的布局下侥幸通过。
  现在只跳过**直接躺在 ``audio_root`` 下、不属于任何说话人**的文件，并单独计数告警。
* **先写临时文件、成功后改名。** 旧版本直接写目标文件，中途被中断（Ctrl-C、被 kill、
  磁盘满）会在目标位置留下**看起来正常、其实只有一半**的清单，下一次运行因为文件已存在
  而把它当成完整结果复用。现在就地把输出写成 ``*.tmp``，全部写完后 ``os.replace`` 原子改名。
* **键（key）就是相对路径**，data.list 与 utt2spk 使用同一个键，保证两张表按行一一对应。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

AUDIO_EXTS_DEFAULT = ('.wav',)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='为一个说话人数据集生成 data.list 与 utt2spk 两张样本清单')
    parser.add_argument('root_dir',
                        help='语料根目录，音频在其 <audio-subdir> 子目录下')
    parser.add_argument('output_dir',
                        help='写出 data.list 与 utt2spk 的目录')
    parser.add_argument('--audio-subdir', default='audio',
                        help='root_dir 下存放说话人目录的子目录名（默认 audio）')
    parser.add_argument('--exts', nargs='+', default=list(AUDIO_EXTS_DEFAULT),
                        help='要收集的音频扩展名，例如 --exts .wav .flac（默认 .wav）')
    parser.add_argument('--allow-empty', action='store_true',
                        help='一条音频都没找到时也退出码 0（默认是报错退出）')
    return parser.parse_args(argv)


def normalize_exts(exts):
    """把扩展名统一成小写、带前导点、元组形式，便于 str.endswith 判断。"""
    normalized = []
    for ext in exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized.append(ext)
    if not normalized:
        raise SystemExit('--exts 至少要给一个非空扩展名')
    return tuple(normalized)


def collect_entries(audio_root, ext_tuple, warn):
    """遍历 audio_root，返回 (entries, stats)。

    ``entries`` 是 ``(key, spk_id, full_path)`` 的列表；``stats`` 记录各种被跳过的
    情况，用于最后如实汇报——**跳过的东西必须能被看见**，不能像旧版本那样静默丢弃。
    """
    entries = []
    n_scanned = 0
    n_wrong_ext = 0
    n_at_root = 0
    loose_files = []

    for dirpath, dirnames, filenames in os.walk(audio_root):
        dirnames.sort()  # 让遍历顺序稳定，便于复现
        for fn in sorted(filenames):
            n_scanned += 1
            if not fn.lower().endswith(ext_tuple):
                n_wrong_ext += 1
                continue
            full_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(full_path, audio_root)
            parts = rel_path.split(os.sep)

            # 直接躺在 audio_root 下的音频不属于任何说话人目录：
            # 它没有可信的说话人归属，只能跳过，但要记下来告警。
            if len(parts) < 2:
                n_at_root += 1
                loose_files.append(rel_path)
                continue

            spk_id = parts[0]
            key = rel_path
            entries.append((key, spk_id, full_path))

    stats = {
        'scanned': n_scanned,
        'wrong_ext': n_wrong_ext,
        'at_root': n_at_root,
        'loose_files': loose_files,
    }
    return entries, stats


def write_lists(entries, data_list_path, utt2spk_path):
    """把两张清单写成临时文件，全部成功后再原子改名到最终位置。

    这样中途失败不会在目标位置留下"看起来完整、实际残缺"的清单。
    """
    data_tmp = data_list_path + '.tmp'
    utt_tmp = utt2spk_path + '.tmp'
    try:
        with open(data_tmp, 'w') as f_data, open(utt_tmp, 'w') as f_utt:
            for key, spk_id, full_path in entries:
                f_data.write(json.dumps({'key': key, 'spk': spk_id,
                                         'wav': full_path}) + '\n')
                # 与 data.list 用同一个 key，保证两张表逐行对应
                f_utt.write(f'{key} {spk_id}\n')
            f_data.flush()
            os.fsync(f_data.fileno())
            f_utt.flush()
            os.fsync(f_utt.fileno())
        os.replace(data_tmp, data_list_path)
        os.replace(utt_tmp, utt2spk_path)
    except BaseException:
        # 失败时清掉临时文件，不给下一次运行留误导性的残骸
        for tmp in (data_tmp, utt_tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def main(argv=None):
    args = parse_args(argv)
    ext_tuple = normalize_exts(args.exts)

    audio_root = os.path.join(args.root_dir, args.audio_subdir)
    if not os.path.isdir(audio_root):
        raise SystemExit(f'找不到音频目录: {audio_root}')

    os.makedirs(args.output_dir, exist_ok=True)
    data_list_path = os.path.join(args.output_dir, 'data.list')
    utt2spk_path = os.path.join(args.output_dir, 'utt2spk')

    entries, stats = collect_entries(
        audio_root, ext_tuple, warn=lambda msg: print(msg, file=sys.stderr))

    if not entries:
        message = (f'在 {audio_root} 下一个 {"/".join(ext_tuple)} 音频都没找到'
                   f'（共扫到 {stats["scanned"]} 个文件，'
                   f'其中扩展名不符 {stats["wrong_ext"]} 个，'
                   f'直接放在根目录、无说话人归属 {stats["at_root"]} 个）')
        if args.allow_empty:
            print(f'警告: {message}', file=sys.stderr)
        else:
            raise SystemExit(message)

    entries.sort(key=lambda x: x[0])

    # 键必须唯一，否则 data.list 与 utt2spk 会出现同键不同说话人，
    # 训练时按说话人取标签就会串。
    seen = set()
    duplicates = []
    for key, _, _ in entries:
        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)
    if duplicates:
        preview = ', '.join(duplicates[:5])
        raise SystemExit(
            f'发现 {len(duplicates)} 个重复的样本编号（键），例如: {preview}。'
            f'键必须唯一，否则 data.list 与 utt2spk 会错位。')

    spk_set = {e[1] for e in entries}
    print(f'共找到 {len(entries)} 句，来自 {len(spk_set)} 个说话人')
    if stats['at_root']:
        preview = ', '.join(stats['loose_files'][:5])
        print(f'注意: 跳过 {stats["at_root"]} 个直接放在 {audio_root} 下、'
              f'没有说话人目录归属的音频文件，例如: {preview}',
              file=sys.stderr)
    if stats['wrong_ext']:
        print(f'注意: 跳过 {stats["wrong_ext"]} 个扩展名不在 '
              f'{" ".join(ext_tuple)} 里的文件', file=sys.stderr)

    write_lists(entries, data_list_path, utt2spk_path)

    print(f'已写出 {data_list_path}（{len(entries)} 行）')
    print(f'已写出 {utt2spk_path}（{len(entries)} 行）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
