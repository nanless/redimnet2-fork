#!/usr/bin/env python3
"""Shared helpers for FireRedVAD offline RTTM annotation.

This module holds the parts that both the annotator and any external checker
need:

* how a training sample list (``data.list``) is read and how one process picks
  its own share of it by line index;
* how an audio file is decoded to 16 kHz mono and turned into features with the
  same front end FireRedVAD uses;
* how an RTTM sidecar is rendered, parsed and checked;
* how a sidecar proves which audio file it was made from (source size and
  modification time) so that a later run can tell "already done" from
  "stale, must redo";
* where the sidecar for one audio file is written.

Nothing here imports torch or FireRedVAD at module level, so it can be used by
lightweight checkers and by worker processes alike.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import soundfile as sf
from scipy import signal

# ---------------------------------------------------------------------------
# Constants that describe the audio and RTTM contract.
# ---------------------------------------------------------------------------

#: Everything is resampled to this rate before the model sees it.
TARGET_SAMPLE_RATE = 16000
#: FireRedVAD's front end: 25 ms window, 10 ms hop, 80 mel bins.
FRAME_LENGTH_SAMPLES = 400
FRAME_SHIFT_SAMPLES = 160
FRAME_SHIFT_MS = 10
FRAME_LENGTH_MS = 25

#: Version of the sidecar file layout. Bump when the meaning of a field
#: changes, so that old sidecars stop being accepted as "already done".
SCHEMA_VERSION = 2

#: The frame-level probability array contract, now that probabilities are
#: stored in a sidecar ``.npz`` next to the RTTM.
PROBABILITY_SCHEMA_VERSION = 1

#: Default suffix appended to the audio file name to get its RTTM path.
#: ``a/b/00001.wav`` -> ``a/b/00001.wav.rttm``.
DEFAULT_RTTM_SUFFIX = ".rttm"

#: Name of the small sidecar that records a human-readable header (the
#: ``#`` comment lines) plus the source file fingerprint.
#: ``a/b/00001.wav`` -> ``a/b/.00001.wav.rttm.json``
RTTM_META_SUFFIX = ".json"

#: Written next to the RTTM only when ``--store-probs`` is requested.
RTTM_PROB_SUFFIX = ".prob.npz"

#: Keys accepted inside one line of the sample list for the audio path.
AUDIO_PATH_KEYS = ("wav", "path", "audio", "audio_filepath", "wav_path")

#: RTTM tokens for the fields this task leaves unspecified. The NIST format
#: wants nine columns; the model only gives one speaker-free speech stream, so
#: every left-over column is filled with the format's own "not available".
RTTM_NA = "<NA>"


# ---------------------------------------------------------------------------
# Small filesystem helpers.
# ---------------------------------------------------------------------------


def abspath(path: "str | os.PathLike[str]") -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that readers never see a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    atomic_write_text(
        path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


# ---------------------------------------------------------------------------
# Sample list (data.list) reading and sharding.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioRecord:
    """One line of the sample list."""

    #: Position of this line among the non-empty lines, counted from zero.
    index: int
    #: Path to the audio file, exactly as written in the list.
    path: str
    #: The sample key, when the list has one.
    key: Optional[str]
    #: The speaker id, when the list has one.
    spk: Optional[str]
    #: 1-based line number in the file, for error messages.
    line_number: int


def iter_audio_records(manifest: Path) -> Iterator[AudioRecord]:
    """Read a sample list. Blank lines are skipped and do not get an index.

    Accepts both shapes used in this repository:

    * a ``data.list`` with ``{"key": ..., "spk": ..., "wav": ...}``;
    * any JSON line object that carries the audio path under one of
      :data:`AUDIO_PATH_KEYS`.
    """
    manifest = Path(manifest)
    index = 0
    # utf-8-sig：清单若带 BOM（Windows 上编辑过、或某些导出工具会加），
    # 用普通 utf-8 读会把 BOM 当成正文，第一行 json.loads 直接失败、
    # 整个任务以一段原始 traceback 结束。utf-8-sig 会把这个前缀吃掉，
    # 没有 BOM 时行为与 utf-8 完全一样。
    with manifest.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"第 {line_number} 行不是合法的 JSON（{manifest}）：{exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象：{manifest}")
            path = None
            for candidate in AUDIO_PATH_KEYS:
                value = item.get(candidate)
                if isinstance(value, str) and value:
                    path = value
                    break
            if path is None:
                raise ValueError(
                    f"第 {line_number} 行没有音频路径字段"
                    f"（可接受 {'/'.join(AUDIO_PATH_KEYS)}）：{manifest}"
                )
            yield AudioRecord(
                index=index,
                path=path,
                key=item.get("key") if isinstance(item.get("key"), str) else None,
                spk=item.get("spk") if isinstance(item.get("spk"), str) else None,
                line_number=line_number,
            )
            index += 1


def count_audio_records(manifest: Path) -> int:
    """Number of sample lines, i.e. the number of files the list points at."""
    total = 0
    for _ in iter_audio_records(manifest):
        total += 1
    return total


def iter_audio_shard(
    manifest: Path,
    rank: int,
    world_size: int,
    limit: Optional[int] = None,
) -> Iterator[AudioRecord]:
    """Yield the records this rank owns, by exact line index modulo world size.

    Rank ``r`` takes the records whose zero-based index ``i`` satisfies
    ``i % world_size == r``. Every index belongs to exactly one rank, so the
    ``world_size`` ranks together cover each line exactly once: no rank has to
    pad its share to a common length, and no two ranks write the same file.
    """
    if world_size <= 0:
        raise ValueError(f"world_size 必须为正：{world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank 必须在 [0, world_size) 内：{rank}/{world_size}")
    selected = 0
    for record in iter_audio_records(manifest):
        if record.index % world_size != rank:
            continue
        yield record
        selected += 1
        if limit is not None and selected >= limit:
            return


def shard_coverage(manifest: Path, world_size: int, limit: Optional[int] = None) -> Dict[str, object]:
    """Check that the shards partition the list: no gap, no overlap.

    Returns a small report; ``consistent`` is true only when every line is
    owned by exactly one rank.
    """
    owners: Dict[int, int] = {}
    duplicates: List[int] = []
    for rank in range(world_size):
        for record in iter_audio_shard(manifest, rank, world_size, limit):
            if record.index in owners:
                duplicates.append(record.index)
            owners[record.index] = rank
    total = count_audio_records(manifest)
    if limit is not None:
        # With --limit each rank takes at most `limit` records, so the covered
        # set is the first `limit * world_size` indices, not the whole list.
        expected = set(range(min(total, limit * world_size)))
    else:
        expected = set(range(total))
    missing = sorted(expected - set(owners))
    per_rank = {rank: 0 for rank in range(world_size)}
    for index, rank in owners.items():
        per_rank[rank] += 1
    return {
        "manifest": str(manifest),
        "manifest_records": total,
        "world_size": world_size,
        "limit": limit,
        "covered": len(owners),
        "expected_covered": len(expected),
        "per_rank": per_rank,
        "duplicate_indices": duplicates[:50],
        "duplicate_count": len(duplicates),
        "missing_count": len(missing),
        "missing_indices_head": missing[:50],
        "consistent": not duplicates and not missing,
    }


# ---------------------------------------------------------------------------
# Where the sidecar goes.
# ---------------------------------------------------------------------------


#: How much of the source path the mirrored tree keeps. See
#: :func:`mirror_relative_path` for why the default is not "all of it".
DEFAULT_ROOT_DEPTH = 2


def mirror_relative_path(
    audio_path: Path,
    audio_root: Optional[Path],
    root_depth: int = DEFAULT_ROOT_DEPTH,
) -> Path:
    """Work out the path an audio file gets inside the mirrored output tree.

    Why this is not simply "the path below ``audio_root``":

    The sample list points at three corpora that live under one shared parent::

        .../speaker_verification/voxceleb/vox2/wav/<speaker>/<video>/00001.wav
        .../speaker_verification/VoxBlink2/audio/<speaker>/<video>/00001.wav
        .../speaker_verification/cnceleb/CN-Celeb2_flac/data/<speaker>/x.flac

    The last two path parts of two different corpora collide easily: the file
    name ``00001.wav`` appears in almost every speaker directory of both
    VoxCeleb2 and VoxBlink2. Measured on the first 200 lines of the mixed
    training list, 13 different file names each occurred more than once, so a
    mirror that kept only the file name would have silently overwritten
    annotations.

    The rule used here instead: **keep the last ``root_depth`` path parts
    before the file name, plus the file name.** With the default
    ``root_depth=2`` the examples above become::

        <output_root>/vox2/wav/<speaker>/<video>/00001.wav.rttm
        <output_root>/audio/<speaker>/<video>/00001.wav.rttm
        <output_root>/data/<speaker>/x.flac.rttm

    Tokens shared by every file of one corpus (here ``voxceleb``,
    ``VoxBlink2``, ``cnceleb``) are dropped because they carry no
    distinguishing information.

    When ``audio_root`` is given, the path is first made relative to it, which
    removes the shared prefix exactly and keeps the token that names the
    corpus. Otherwise the absolute path is trimmed to its last
    ``root_depth + 1`` parts.

    The caller is responsible for checking that the result is unique across the
    sample list; :func:`check_mirror_uniqueness` does that.
    """
    audio_path = abspath(audio_path)
    parts: Tuple[str, ...]
    if audio_root is not None:
        try:
            parts = audio_path.relative_to(abspath(audio_root)).parts
        except ValueError:
            parts = audio_path.parts
    else:
        parts = audio_path.parts
    if root_depth > 0:
        # Keep the file name plus the last `root_depth` directory names.
        parts = parts[-(root_depth + 1) :]
    else:
        parts = parts[-1:]
    return Path(*parts)


def sidecar_path(
    audio_path: Path,
    audio_root: Optional[Path],
    output_root: Optional[Path],
    suffix: str = DEFAULT_RTTM_SUFFIX,
    root_depth: int = DEFAULT_ROOT_DEPTH,
) -> Path:
    """Return the RTTM path for one audio file.

    Two layouts are supported:

    * **Beside the audio (default).** When ``output_root`` is empty the RTTM
      sits in the same directory as the audio file, named
      ``<audio name><suffix>``. Training code that is handed an audio path can
      therefore find the annotation without a lookup table. This layout is
      only usable when the audio tree may be written to.
    * **Mirrored tree.** When ``output_root`` is given, the audio path is
      rebuilt below ``output_root`` by the rule described in
      :func:`mirror_relative_path`. Use this when the audio tree is read-only,
      and set ``output_root`` to the directory that matches the audio
      directory one-to-one.
    """
    audio_path = abspath(audio_path)
    if output_root is None:
        return audio_path.parent / f"{audio_path.name}{suffix}"
    output_root = abspath(output_root)
    relative = mirror_relative_path(audio_path, audio_root, root_depth)
    return output_root / relative.parent / f"{relative.name}{suffix}"


def check_mirror_uniqueness(
    records: "Sequence[AudioRecord]",
    audio_root: Optional[Path],
    output_root: Optional[Path],
    suffix: str = DEFAULT_RTTM_SUFFIX,
    root_depth: int = DEFAULT_ROOT_DEPTH,
    beside: bool = False,
) -> Dict[str, object]:
    """Check that no two sample lines would write the same RTTM path.

    A collision would mean one annotation silently overwriting another, so a
    run must refuse to start when this check fails. Returns the colliding
    paths together with the sample lines that produced them.
    """
    owners: Dict[str, List[str]] = {}
    for record in records:
        path = sidecar_path(
            Path(record.path), None if beside else audio_root, None if beside else output_root,
            suffix, root_depth,
        )
        owners.setdefault(str(path), []).append(f"{record.index}:{record.path}")
    collisions = {path: sources for path, sources in owners.items() if len(sources) > 1}
    return {
        "layout": "beside-audio" if beside else "mirror-tree",
        "audio_root": str(audio_root) if audio_root else None,
        "output_root": str(output_root) if output_root else None,
        "root_depth": root_depth,
        "records": len(records),
        "distinct_outputs": len(owners),
        "collision_count": len(collisions),
        "collisions_head": [
            {"output": path, "sources": sources[:4]}
            for path, sources in sorted(collisions.items())[:20]
        ],
        "unique": not collisions,
    }


def meta_path(rttm_path: Path) -> Path:
    """Path of the small JSON header file that belongs to an RTTM."""
    return Path(rttm_path).parent / f".{Path(rttm_path).name}{RTTM_META_SUFFIX}"


def probability_path(rttm_path: Path) -> Path:
    return Path(str(rttm_path) + RTTM_PROB_SUFFIX)


# ---------------------------------------------------------------------------
# Audio decoding.
# ---------------------------------------------------------------------------


def read_audio_mono_16k(path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """Read any supported audio file as 16 kHz mono int16 samples.

    Steps, in order:

    1. read with libsndfile as int16, keeping all channels;
    2. downmix to mono by averaging channels in float and rounding;
    3. resample to 16 kHz with a polyphase filter when the source rate is not
       already 16 kHz.

    Returns the samples plus what was observed in the source file, so the
    caller can write it into the sidecar header.
    """
    path = Path(path)
    stat = path.stat()
    audio, source_rate = sf.read(str(path), dtype="int16", always_2d=True)
    if audio.shape[0] <= 0:
        raise ValueError("音频里没有任何采样点（0 帧）")
    source_frames = int(audio.shape[0])
    source_channels = int(audio.shape[1])
    if source_channels <= 0:
        raise ValueError("声道数为 0")
    if source_channels == 1:
        mono = audio[:, 0]
    else:
        mono = np.rint(audio.astype(np.float64).mean(axis=1))
        mono = np.clip(mono, -32768, 32767).astype(np.int16)
    source_rate = int(source_rate)
    if source_rate <= 0:
        raise ValueError("采样率读到 0")
    if source_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(source_rate, TARGET_SAMPLE_RATE)
        as_float = mono.astype(np.float32) / 32768.0
        resampled = signal.resample_poly(
            as_float, TARGET_SAMPLE_RATE // divisor, source_rate // divisor
        ).astype(np.float32, copy=False)
        mono = np.rint(np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16)
    return np.ascontiguousarray(mono), {
        "sample_length": int(mono.shape[0]),
        "source_sample_rate": source_rate,
        "source_num_frames": source_frames,
        "source_num_channels": source_channels,
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def expected_feature_frames(sample_length: int) -> int:
    """How many 10 ms frames a signal of this many 16 kHz samples produces."""
    if sample_length < FRAME_LENGTH_SAMPLES:
        return 0
    return (sample_length - FRAME_LENGTH_SAMPLES) // FRAME_SHIFT_SAMPLES + 1


# ---------------------------------------------------------------------------
# RTTM rendering, parsing and checking.
# ---------------------------------------------------------------------------


def format_rttm_line(file_id: str, start: float, duration: float) -> str:
    """Render one speech segment as a NIST RTTM ``SPEAKER`` line.

    The nine columns are, in order:

    ``SPEAKER``, the file id, the channel (always ``1`` because the model sees
    a mono signal), the start time in seconds, the length in seconds, then
    four columns of speaker information that this task does not use, and
    finally the confidence column, which is not used either. Times always
    carry three decimals.
    """
    if duration <= 0:
        raise ValueError(f"时长为非正数的语音段：start={start} dur={duration}")
    return (
        f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} "
        f"{RTTM_NA} {RTTM_NA} {RTTM_NA} {RTTM_NA} {RTTM_NA}"
    )


def render_rttm(
    file_id: str,
    timestamps: Sequence[Tuple[float, float]],
    header_lines: Sequence[str],
) -> str:
    """Build the whole RTTM file text: header comments, then one line per
    speech segment."""
    lines: List[str] = [f"# {text}" for text in header_lines]
    for start, end in timestamps:
        if end <= start:
            continue
        lines.append(format_rttm_line(file_id, float(start), float(end - start)))
    return "\n".join(lines) + "\n"


def file_id_for(audio_path: Path) -> str:
    """The RTTM file id: the audio file name without its last extension.

    ``/a/b/00001.wav`` -> ``00001``. This is stable for a given audio file and
    does not depend on where the corpus is mounted.
    """
    return Path(audio_path).stem


@dataclass(frozen=True)
class RttmSegment:
    start: float
    duration: float
    channel: str
    raw: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class RttmParseResult:
    file_id: Optional[str]
    segments: List[RttmSegment]
    errors: List[str]


def parse_rttm(text: str) -> RttmParseResult:
    """Parse RTTM text back into segments, collecting every problem found.

    Which line is what:

    * a line whose first token is ``SPEAKER`` carries one speech segment;
    * a line starting with ``#`` is a comment written by this tool (the
      header block) and is ignored;
    * **any other non-empty line is an error.** It is recorded in ``errors``
      and makes the whole file fail validation. Nothing is guessed and
      nothing is silently dropped: an unknown line means the file was not
      written by this tool, or was written by a version whose meaning we do
      not know.

    A row is only *returned* in ``segments`` when it is structurally usable:
    ten fields, ``SPEAKER`` keyword, channel ``1``, the five optional columns
    set to ``<NA>``, times parseable and written with exactly three decimals,
    ``start >= 0``, and ``duration > 0``. A row that fails any of those is
    recorded in ``errors`` and **is not added to** ``segments``, so a caller
    that only looks at ``segments`` cannot end up with an inverted span
    (``end < start``) or an out-of-range value. Callers should still treat a
    non-empty ``errors`` list as a hard failure — the two views are meant to
    be used together, and every rejected row appears in ``errors``.
    """
    errors: List[str] = []
    segments: List[RttmSegment] = []
    file_id: Optional[str] = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        fields = stripped.split()
        if fields[0] != "SPEAKER":
            errors.append(f"第 {line_number} 行不是 SPEAKER 行：{stripped[:80]!r}")
            continue
        if len(fields) != 10:
            errors.append(
                f"第 {line_number} 行字段数 {len(fields)}，应为 10：{stripped[:80]!r}"
            )
            continue
        _, this_file_id, channel, start_text, duration_text = fields[:5]
        if file_id is None:
            file_id = this_file_id
        elif this_file_id != file_id:
            errors.append(
                f"第 {line_number} 行的文件编号 {this_file_id!r} 与前面的 {file_id!r} 不一致"
            )
        row_rejected = False
        try:
            start = float(start_text)
            duration = float(duration_text)
        except ValueError:
            errors.append(f"第 {line_number} 行的时间不是数字：{start_text!r} {duration_text!r}")
            continue
        if not (start_text.count(".") == 1 and len(start_text.split(".")[1]) == 3):
            errors.append(f"第 {line_number} 行起始时间不是三位小数：{start_text!r}")
            row_rejected = True
        if not (duration_text.count(".") == 1 and len(duration_text.split(".")[1]) == 3):
            errors.append(f"第 {line_number} 行时长不是三位小数：{duration_text!r}")
            row_rejected = True
        for tail_index, tail in enumerate(fields[5:], start=5):
            if tail != RTTM_NA:
                errors.append(
                    f"第 {line_number} 行第 {tail_index + 1} 列应为 {RTTM_NA}，实际是 {tail!r}"
                )
                row_rejected = True
        if channel != "1":
            errors.append(f"第 {line_number} 行声道应为 1，实际是 {channel!r}")
            row_rejected = True
        if start < 0:
            errors.append(f"第 {line_number} 行起点为负数：{start}")
            row_rejected = True
        if duration <= 0:
            errors.append(
                f"第 {line_number} 行时长为非正数：{duration_text!r}；"
                "这样的段起点与终点会颠倒，已丢弃"
            )
            row_rejected = True
        if row_rejected:
            continue
        segments.append(
            RttmSegment(start=start, duration=duration, channel=channel, raw=stripped)
        )
    return RttmParseResult(file_id=file_id, segments=segments, errors=errors)


def check_segments(
    segments: Sequence[RttmSegment],
    audio_duration: float,
    tolerance: float = 0.0015,
) -> List[str]:
    """Check ordering, overlap and bounds of the parsed segments.

    ``tolerance`` allows for the fact that both the audio length and the
    segment times are rounded to milliseconds; the default is a little over
    one millisecond.
    """
    errors: List[str] = []
    previous_end = None
    for position, segment in enumerate(segments):
        if segment.duration <= 0:
            errors.append(f"第 {position + 1} 段时长非正：{segment.raw!r}")
        if segment.start < -tolerance:
            errors.append(f"第 {position + 1} 段起点为负：{segment.start}")
        if previous_end is not None and segment.start < previous_end - tolerance:
            errors.append(
                f"第 {position + 1} 段与前一段重叠：{segment.start} < {previous_end}"
            )
        if segment.end > audio_duration + tolerance:
            errors.append(
                f"第 {position + 1} 段超出音频长度：{segment.end} > {audio_duration:.3f}"
            )
        previous_end = segment.end
    return errors


# ---------------------------------------------------------------------------
# Sidecar header (the ``#`` comment lines).
# ---------------------------------------------------------------------------

#: Field order of the header lines. Kept in one place so that the writer and
#: the reader cannot drift apart.
HEADER_KEYS = (
    "source",
    "source_size",
    "source_mtime_ns",
    "source_sample_rate",
    "source_channels",
    "source_frames",
    "sample_rate",
    "samples_16k",
    "duration_s",
    "frames",
    "frame_shift_ms",
    "model",
    "model_sha256",
    "firered_git_sha",
    "tool_git_sha",
    "vad",
    "rttm_suffix",
    "schema_version",
    "inference_signature",
    "segments",
    "speech_s",
    "speech_ratio",
)


def render_header_block(header: Dict[str, object]) -> List[str]:
    """Turn the header dictionary into sorted ``key=value`` comment lines."""
    lines: List[str] = []
    for key in HEADER_KEYS:
        if key in header:
            lines.append(f"{key}={header[key]}")
    for key in sorted(header):
        if key not in HEADER_KEYS:
            lines.append(f"{key}={header[key]}")
    return lines


def parse_header_block(text: str) -> Dict[str, str]:
    """Read back the ``key=value`` comment lines of an RTTM file."""
    header: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped.lstrip("#").strip()
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        header[key.strip()] = value.strip()
    return header
