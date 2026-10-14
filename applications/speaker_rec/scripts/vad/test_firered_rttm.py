#!/usr/bin/env python3
"""RTTM 标注工具的单元自检：不读音频、不用 GPU，几秒钟跑完。

用法::

    /root/miniforge3/bin/python test_firered_rttm.py

覆盖的内容：

* RTTM 行的渲染与解析是往返一致的，且恰好是十列；
* 结构不合法（时长非正、起点为负、声道不对、列数不对、不是 SPEAKER 行）
  的行会被拒绝，不会留在可用的语音段列表里，并且一定伴随一条错误；
* 有序/不重叠/不越界的检查能抓出问题；
* 镜像树的输出路径在不同语料之间不会撞名；
* 分片按行号取模，各 rank 合起来正好覆盖整个清单，不重复不遗漏。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from firered_rttm_common import (  # noqa: E402
    AudioRecord,
    check_mirror_uniqueness,
    check_segments,
    format_rttm_line,
    mirror_relative_path,
    parse_rttm,
    render_rttm,
    shard_coverage,
    sidecar_path,
)

FAILURES: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  通过  {name}")
    else:
        print(f"  未通过 {name} {detail}")
        FAILURES.append(name)


def test_format_roundtrip() -> None:
    print("[1] RTTM 行渲染与解析")
    line = format_rttm_line("00001", 0.44, 1.38)
    check("恰好十列", len(line.split()) == 10, line)
    check(
        "字段内容正确",
        line == "SPEAKER 00001 1 0.440 1.380 <NA> <NA> <NA> <NA> <NA>",
        line,
    )
    parsed = parse_rttm(line + "\n")
    check("解析无错误", parsed.errors == [], str(parsed.errors))
    check("解析出一条语音段", len(parsed.segments) == 1)
    if parsed.segments:
        segment = parsed.segments[0]
        check(
            "起止时间往返一致",
            abs(segment.start - 0.44) < 1e-9 and abs(segment.end - 1.82) < 1e-9,
            f"{segment.start} {segment.end}",
        )
    check("文件编号正确", parsed.file_id == "00001")


def test_rejects_bad_rows() -> None:
    print("[2] 不合法的行必须被拒绝，而不是当成可用语音段")
    cases = {
        "时长为负": "SPEAKER f 1 1.000 -0.500 <NA> <NA> <NA> <NA> <NA>",
        "时长为零": "SPEAKER f 1 1.000 0.000 <NA> <NA> <NA> <NA> <NA>",
        "起点为负": "SPEAKER f 1 -1.000 0.500 <NA> <NA> <NA> <NA> <NA>",
        "声道不是 1": "SPEAKER f 2 1.000 0.500 <NA> <NA> <NA> <NA> <NA>",
        "列数不足": "SPEAKER f 1 1.000 0.500 <NA> <NA>",
        "列数过多": "SPEAKER f 1 1.000 0.500 <NA> <NA> <NA> <NA> <NA> extra",
        "关键字不对": "SPEECH f 1 1.000 0.500 <NA> <NA> <NA> <NA> <NA>",
        "小数点位数不够": "SPEAKER f 1 1.0 0.5 <NA> <NA> <NA> <NA> <NA>",
        "未用列不是 NA": "SPEAKER f 1 1.000 0.500 spk1 <NA> <NA> <NA> <NA>",
        "时间不是数字": "SPEAKER f 1 abc 0.500 <NA> <NA> <NA> <NA> <NA>",
    }
    for name, row in cases.items():
        parsed = parse_rttm(row + "\n")
        check(f"{name}：没有留下可用语音段", parsed.segments == [], str(parsed.segments))
        check(f"{name}：报告了错误", len(parsed.errors) >= 1, str(parsed.errors))

    # 空文件是合法的：表示「标注过了，但没有语音」。
    empty = parse_rttm("# segments=0\n")
    check("空文件：没有错误", empty.errors == [], str(empty.errors))
    check("空文件：零条语音段", empty.segments == [])
    check("空文件：仍然有文件编号以外的头部", empty.file_id is None)


def test_check_segments() -> None:
    print("[3] 有序、不重叠、不越界检查")
    good = parse_rttm(
        "SPEAKER f 1 0.000 1.000 <NA> <NA> <NA> <NA> <NA>\n"
        "SPEAKER f 1 1.000 2.000 <NA> <NA> <NA> <NA> <NA>\n"
    )
    check("正常两段没有问题", check_segments(good.segments, 3.0) == [],
          str(check_segments(good.segments, 3.0)))
    over = parse_rttm(
        "SPEAKER f 1 0.000 1.000 <NA> <NA> <NA> <NA> <NA>\n"
        "SPEAKER f 1 0.500 2.000 <NA> <NA> <NA> <NA> <NA>\n"
    )
    check("重叠能被抓出", len(check_segments(over.segments, 3.0)) >= 1)
    beyond = parse_rttm("SPEAKER f 1 0.000 9.000 <NA> <NA> <NA> <NA> <NA>\n")
    check("越界能被抓出", len(check_segments(beyond.segments, 3.0)) >= 1)


def test_mirror_paths() -> None:
    print("[4] 镜像树输出路径")
    audio_root = Path("/data/speech/speaker_verification")
    cases = {
        "A": (
            "/data/speech/speaker_verification/voxceleb/vox2/wav/spk0001/vid0001/00001.wav",
            "spk0001/vid0001/00001.wav.rttm",
        ),
        "B": (
            "/data/speech/speaker_verification/VoxBlink2/audio/spk0001/vid0001/00001.wav",
            "spk0001/vid0001/00001.wav.rttm",
        ),
        "C": (
            "/data/speech/speaker_verification/cnceleb/CN-Celeb2_flac/data/spk0001/a.flac",
            "data/spk0001/a.flac.rttm",
        ),
    }
    for name, (audio, expected) in cases.items():
        got = sidecar_path(
            Path(audio), audio_root, Path("/out"), ".rttm", 2
        ).relative_to("/out").as_posix()
        check(f"{name} 路径正确", got == expected, f"{got} != {expected}")

    # 说话人目录名在三套语料里不会重复，所以只留两层目录就够了；
    # 下面这两条路径恰好一样是正常的：同一个说话人 ID 只在 VoxBlink2 里出现。
    print("[4b] 同名文件不再互相覆盖")
    # 这两个文件在过去只保留文件名时会撞在一起。
    a = "/data/speech/speaker_verification/voxceleb/vox2/wav/id06536/kTOEfLNM3IY/00137.wav"
    b = "/data/speech/speaker_verification/VoxBlink2/audio/id122341/gUY2pdKeuDg/00003.wav"
    c = "/data/speech/speaker_verification/voxceleb/vox2/wav/id06051/bc2NudrojMA/00137.wav"
    paths = {
        sidecar_path(Path(p), audio_root, Path("/out"), ".rttm", 2) for p in (a, b, c)
    }
    check("三个不同音频给出三个不同路径", len(paths) == 3, str(paths))
    shallow = {
        sidecar_path(Path(p), audio_root, Path("/out"), ".rttm", 0) for p in (a, c)
    }
    check(
        "只保留文件名时确实会撞（说明 depth 默认值是必要的）",
        len(shallow) == 1,
        str(shallow),
    )

    records = [
        AudioRecord(index=0, path=a, key=None, spk=None, line_number=1),
        AudioRecord(index=1, path=c, key=None, spk=None, line_number=2),
    ]
    report = check_mirror_uniqueness(records, audio_root, Path("/out"), ".rttm", 0)
    check("前置检查能发现撞名", not report["unique"], str(report))
    report2 = check_mirror_uniqueness(records, audio_root, Path("/out"), ".rttm", 2)
    check("调大 depth 后不再撞名", report2["unique"], str(report2))


def test_sharding() -> None:
    print("[5] 按行号分片")
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "data.list"
        rows = [
            json.dumps({"key": f"k{i}", "spk": f"s{i}", "wav": f"/audio/{i}.wav"})
            for i in range(101)  # 故意用质数，制造不整除
        ]
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

        for world_size in (1, 2, 3, 7):
            report = shard_coverage(manifest, world_size)
            check(
                f"world_size={world_size}：恰好覆盖清单一次",
                report["consistent"] and report["covered"] == 101,
                json.dumps(report, ensure_ascii=False),
            )
            counts = list(report["per_rank"].values())
            check(
                f"world_size={world_size}：各 rank 条数至多差 1",
                max(counts) - min(counts) <= 1,
                str(counts),
            )

        limited = shard_coverage(manifest, 4, limit=5)
        check(
            "--limit 时覆盖前 limit*world_size 条",
            limited["consistent"] and limited["covered"] == 20,
            json.dumps(limited, ensure_ascii=False),
        )

        # 分片是「按行号取模」而不是「切段」，所以顺序交错但合起来完整。
        seen = []
        for rank in range(3):
            from firered_rttm_common import iter_audio_shard

            seen.extend(r.index for r in iter_audio_shard(manifest, rank, 3))
        check("三片拼起来正好是 0..100", sorted(seen) == list(range(101)))


def test_audio_path_keys() -> None:
    print("[6] 样本清单里音频路径字段的兼容性")
    from firered_rttm_common import iter_audio_records

    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "data.list"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"key": "a", "spk": "s", "wav": "/a.wav"}),
                    json.dumps({"path": "/b.flac"}),
                    json.dumps({"audio_filepath": "/c.wav"}),
                    "",  # 空行要被跳过，而且不占行号
                    json.dumps({"key": "d", "spk": "s", "wav": "/d.wav"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        records = list(iter_audio_records(manifest))
        check("读到四条", len(records) == 4, str([r.path for r in records]))
        check(
            "顺序与内容正确",
            [r.path for r in records] == ["/a.wav", "/b.flac", "/c.wav", "/d.wav"],
            str([r.path for r in records]),
        )
        check("空行没有占行号", records[-1].index == 3, str(records[-1].index))

        manifest.write_text(json.dumps({"key": "a"}) + "\n", encoding="utf-8")
        try:
            list(iter_audio_records(manifest))
            check("缺少音频路径时报错", False, "没有抛异常")
        except ValueError:
            check("缺少音频路径时报错", True)


def main() -> int:
    print("=" * 60)
    print("RTTM 标注工具自检（不需要 GPU，不读音频）")
    print("=" * 60)
    test_format_roundtrip()
    test_rejects_bad_rows()
    test_check_segments()
    test_mirror_paths()
    test_sharding()
    test_audio_path_keys()
    print("=" * 60)
    if FAILURES:
        print(f"有 {len(FAILURES)} 项未通过：")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
