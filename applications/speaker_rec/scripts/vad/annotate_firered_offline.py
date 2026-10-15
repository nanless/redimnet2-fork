#!/usr/bin/env python3
"""用 FireRedVAD（非流式离线模式）给说话人验证训练音频标注语音段，每条音频写一个 RTTM 文件。

用法示例::

    # 试跑：只处理本 rank 的前 200 条
    /root/miniforge3/bin/python annotate_firered_offline.py \
        --mode infer \
        --manifest /root/group-shared/voiceprint/data/wespeaker_data/vox2_dev/data.list \
        --output-root /dev/shm/rttm_smoke \
        --firered-root /root/code/github_repos/FireRedVAD \
        --model-dir /root/code/github_repos/FireRedVAD/pretrained_models/FireRedVAD/VAD \
        --audio-root /root/group-shared/voiceprint/data/speech/speaker_verification \
        --rank 0 --world-size 2 --limit 200

    # 检查已经写出来的 RTTM
    ... --mode validate --output-root /dev/shm/rttm_smoke --rank 0 --world-size 2

各模式的职责：

``infer``
    按行号分片取样本清单的一部分，读音频、算特征、跑模型、写 RTTM。
    已经存在且检查通过的 RTTM 会被跳过，所以这个模式可以反复跑、可以续跑。
``validate``
    只读地重新检查每一条音频对应的 RTTM：能不能解析、时间是否越界、
    是否有序不重叠、来源指纹是否还对得上。
``oracle``
    数值对照：把一批音频合在一个批次里跑，和逐条单独跑的结果比，
    确认批处理没有改变结果。

这个脚本不写任何东西到音频目录之外（除非显式打开 ``--store-probs``），
也不会修改、移动、删除任何音频文件。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from firered_rttm_common import (  # noqa: E402
    AudioRecord,
    DEFAULT_RTTM_SUFFIX,
    DEFAULT_ROOT_DEPTH,
    HEADER_KEYS,
    PROBABILITY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TARGET_SAMPLE_RATE,
    abspath,
    atomic_write_json,
    atomic_write_text,
    check_mirror_uniqueness,
    check_segments,
    expected_feature_frames,
    file_id_for,
    iter_audio_records,
    iter_audio_shard,
    meta_path,
    parse_header_block,
    parse_rttm,
    probability_path,
    read_audio_mono_16k,
    render_header_block,
    render_rttm,
    sha256_of_file,
    shard_coverage,
    sidecar_path,
)

# ---------------------------------------------------------------------------
# 模型参数。默认值与 FireRedVAD 官方 vad.py 里的 FireRedVadConfig 一致。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VadParameters:
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000

    def as_dict(self) -> Dict[str, object]:
        return {
            "smooth_window_size": self.smooth_window_size,
            "speech_threshold": self.speech_threshold,
            "min_speech_frame": self.min_speech_frame,
            "max_speech_frame": self.max_speech_frame,
            "min_silence_frame": self.min_silence_frame,
            "merge_silence_frame": self.merge_silence_frame,
            "extend_speech_frame": self.extend_speech_frame,
            "chunk_max_frame": self.chunk_max_frame,
        }

    def compact(self) -> str:
        return (
            f"sw{self.smooth_window_size}-th{self.speech_threshold}"
            f"-mins{self.min_speech_frame}-minn{self.min_silence_frame}"
            f"-maxs{self.max_speech_frame}-merges{self.merge_silence_frame}"
            f"-ext{self.extend_speech_frame}-chunk{self.chunk_max_frame}"
        )


@dataclass
class FeatureItem:
    """一条音频读完之后的所有中间结果。"""

    audio_path: str
    index: int
    output_path: str
    meta_path: str
    prob_path: str
    feat: Optional[torch.Tensor] = None
    sample_length: int = 0
    source_sample_rate: int = 0
    source_num_frames: int = 0
    source_num_channels: int = 0
    source_size: int = 0
    source_mtime_ns: int = 0
    error: Optional[str] = None

    @property
    def num_frames(self) -> int:
        return 0 if self.feat is None else int(self.feat.shape[0])

    @property
    def duration(self) -> float:
        return self.sample_length / float(TARGET_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# 各进程共用的全局量。用 spawn 启动子进程时由 _worker_init 填好。
# ---------------------------------------------------------------------------

_WORKER_FEAT = None
_WORKER_AUDIO_ROOT: Optional[Path] = None
_WORKER_OUTPUT_ROOT: Optional[Path] = None
_WORKER_SUFFIX = DEFAULT_RTTM_SUFFIX
_WORKER_ROOT_DEPTH = DEFAULT_ROOT_DEPTH


def _worker_init(
    firered_root: str,
    model_dir: str,
    audio_root: str,
    output_root: str,
    suffix: str,
    root_depth: int,
) -> None:
    global _WORKER_FEAT, _WORKER_AUDIO_ROOT, _WORKER_OUTPUT_ROOT, _WORKER_SUFFIX
    global _WORKER_ROOT_DEPTH
    if firered_root not in sys.path:
        sys.path.insert(0, firered_root)
    from fireredvad.core.audio_feat import AudioFeat

    _WORKER_FEAT = AudioFeat(str(Path(model_dir) / "cmvn.ark"))
    _WORKER_AUDIO_ROOT = abspath(audio_root) if audio_root else None
    _WORKER_OUTPUT_ROOT = abspath(output_root) if output_root else None
    _WORKER_SUFFIX = suffix
    _WORKER_ROOT_DEPTH = root_depth
    torch.set_num_threads(1)


def extract_feature(record: AudioRecord) -> FeatureItem:
    """读完一条音频并算出 80 维滤波器组特征。这一步在子进程里做。"""
    audio_path = abspath(record.path)
    assert _WORKER_FEAT is not None
    out_path = sidecar_path(
        audio_path,
        _WORKER_AUDIO_ROOT,
        _WORKER_OUTPUT_ROOT,
        _WORKER_SUFFIX,
        _WORKER_ROOT_DEPTH,
    )
    item = FeatureItem(
        audio_path=str(audio_path),
        index=record.index,
        output_path=str(out_path),
        meta_path=str(meta_path(out_path)),
        prob_path=str(probability_path(out_path)),
    )
    try:
        audio, metadata = read_audio_mono_16k(audio_path)
        feat, duration = _WORKER_FEAT.extract((audio, TARGET_SAMPLE_RATE))
        feat = feat.contiguous()
        expected_frames = expected_feature_frames(metadata["sample_length"])
        if feat.ndim != 2 or feat.shape[1] != 80:
            raise ValueError(f"特征形状异常：{tuple(feat.shape)}")
        if int(feat.shape[0]) != expected_frames:
            raise ValueError(
                f"特征帧数 {feat.shape[0]} 与按采样点算出的 {expected_frames} 不一致"
            )
        expected_duration = metadata["sample_length"] / float(TARGET_SAMPLE_RATE)
        if abs(float(duration) - expected_duration) > 1e-9:
            raise ValueError(f"时长 {duration} 与 {expected_duration} 不一致")
        for key, value in metadata.items():
            setattr(item, key, value)
        item.feat = feat
        del audio
        return item
    except Exception:
        item.error = traceback.format_exc()
        return item


# ---------------------------------------------------------------------------
# 批处理计划与模型前向。
# ---------------------------------------------------------------------------


def plan_batches(
    items: Sequence[FeatureItem], batch_size: int, max_padded_frames: int
) -> List[List[FeatureItem]]:
    """把一批音频排成若干次前向。

    先按帧数从小到大排序，再把长度接近的放进同一批，这样补零最少。
    同一批里还要满足两个上限：条数不超过 ``batch_size``，
    「最长那条的帧数 × 本批条数」不超过 ``max_padded_frames``。
    """
    if batch_size <= 0 or max_padded_frames <= 0:
        raise ValueError("batch_size 与 max_padded_frames 必须为正")
    ordered = sorted(items, key=lambda item: item.num_frames)
    batches: List[List[FeatureItem]] = []
    current: List[FeatureItem] = []
    current_max = 0
    for item in ordered:
        candidate_max = max(current_max, item.num_frames)
        if current and (
            len(current) >= batch_size
            or candidate_max * (len(current) + 1) > max_padded_frames
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, item.num_frames)
    if current:
        batches.append(current)
    return batches


def load_vad(firered_root: Path, model_dir: Path, params: VadParameters):
    """载入 FireRedVAD 模型。

    这里只改一处：FireRedVAD 自带的 ``get_mask_from_lengths`` 返回 ``uint8``
    掩码，PyTorch 2.5 的 ``masked_fill`` 要求布尔掩码。补零的位置和数值都不变，
    只把类型换成布尔。逐条前向（也就是 oracle 对照用的那条路）完全没动。
    """
    if str(firered_root) not in sys.path:
        sys.path.insert(0, str(firered_root))
    import fireredvad.core.detect_model as detect_model_module
    from fireredvad.vad import FireRedVad, FireRedVadConfig

    def _boolean_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
        max_length = int(torch.max(lengths).item())
        frame_index = torch.arange(max_length, device=lengths.device).unsqueeze(0)
        return frame_index >= lengths.unsqueeze(1)

    detect_model_module.get_mask_from_lengths = _boolean_mask_from_lengths

    config = FireRedVadConfig(use_gpu=True, **params.as_dict())
    return FireRedVad.from_pretrained(str(model_dir), config)


def forward_chunked_single(vad, feat: torch.Tensor, chunk_max_frame: int) -> np.ndarray:
    """超长音频按 chunk_max_frame 切开、逐块前向，再按时间顺序拼回来。"""
    if feat.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    chunks: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in feat.split(chunk_max_frame, dim=0):
            prob, _ = vad.vad_model.forward(chunk.unsqueeze(0).cuda(non_blocking=True))
            chunks.append(prob.detach().cpu().reshape(-1))
    return torch.cat(chunks).numpy().astype(np.float32, copy=False)


def forward_batch_once(vad, items: Sequence[FeatureItem]) -> List[np.ndarray]:
    """把同一批里所有音频补零成同样长，一次性前向，再按各自长度截回。"""
    if not items:
        return []
    nonempty = [item for item in items if item.num_frames > 0]
    outputs: Dict[str, np.ndarray] = {
        item.audio_path: np.zeros(0, dtype=np.float32)
        for item in items
        if item.num_frames == 0
    }
    if nonempty:
        lengths = torch.tensor(
            [item.num_frames for item in nonempty], dtype=torch.long, device="cuda"
        )
        max_frames = int(lengths.max().item())
        padded = torch.zeros(
            (len(nonempty), max_frames, 80), dtype=torch.float32, device="cuda"
        )
        for index, item in enumerate(nonempty):
            padded[index, : item.num_frames].copy_(item.feat, non_blocking=True)
        with torch.inference_mode():
            hidden, _ = vad.vad_model.dfsmn(padded, input_lengths=lengths)
            probs = torch.sigmoid(vad.vad_model.out(hidden))[..., 0]
        probs_cpu = probs.detach().cpu()
        for index, item in enumerate(nonempty):
            outputs[item.audio_path] = (
                probs_cpu[index, : item.num_frames]
                .contiguous()
                .numpy()
                .astype(np.float32, copy=False)
            )
    return [outputs[item.audio_path] for item in items]


def forward_batch_resilient(vad, items: Sequence[FeatureItem]) -> List[np.ndarray]:
    """显存不够时把这一批对半拆开重试，直到能跑下为止。"""
    try:
        return forward_batch_once(vad, items)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) <= 1:
            raise
        middle = len(items) // 2
        return forward_batch_resilient(vad, items[:middle]) + forward_batch_resilient(
            vad, items[middle:]
        )


# ---------------------------------------------------------------------------
# 写出与检查。
# ---------------------------------------------------------------------------


def make_inference_signature(
    model_sha256: str, firered_git_sha: str, params: VadParameters, suffix: str
) -> str:
    """把所有会影响输出的设置压成一个短标识，写进 RTTM 头部。

    以后只要模型、参数或输出命名方式变了，这个标识就会变，
    旧的 RTTM 就不会被当成「已经做过」而跳过。
    """
    import hashlib

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "firered_git_sha": firered_git_sha,
        "vad_parameters": params.as_dict(),
        "audio_contract": {
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "channels": "多声道先按浮点求平均再四舍五入回 int16，单声道原样",
            "resample": "scipy.signal.resample_poly，float32，32768 归一化",
        },
        "output": {"rttm_suffix": suffix, "layout": "beside-or-mirror-tree"},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_header(
    item: FeatureItem,
    params: VadParameters,
    timestamps: Sequence[Tuple[float, float]],
    model_dir: Path,
    model_sha256: str,
    firered_git_sha: str,
    tool_git_sha: str,
    signature: str,
    suffix: str,
    source_relative: str,
) -> Dict[str, object]:
    """一条 RTTM 文件开头那几行 ``#`` 注释的内容。"""
    speech_seconds = sum(max(0.0, end - start) for start, end in timestamps)
    duration = item.duration
    return {
        "source": source_relative,
        "source_size": item.source_size,
        "source_mtime_ns": item.source_mtime_ns,
        "source_sample_rate": item.source_sample_rate,
        "source_channels": item.source_num_channels,
        "source_frames": item.source_num_frames,
        "sample_rate": TARGET_SAMPLE_RATE,
        "samples_16k": item.sample_length,
        "duration_s": f"{duration:.3f}",
        "frames": item.num_frames,
        "frame_shift_ms": 10,
        "model": str(model_dir),
        "model_sha256": model_sha256,
        "firered_git_sha": firered_git_sha,
        "tool_git_sha": tool_git_sha,
        "vad": params.compact(),
        "rttm_suffix": suffix,
        "schema_version": SCHEMA_VERSION,
        "inference_signature": signature,
        "segments": len(timestamps),
        "speech_s": f"{speech_seconds:.3f}",
        "speech_ratio": f"{(speech_seconds / duration) if duration > 0 else 0.0:.4f}",
    }


def write_sidecar(
    item: FeatureItem,
    text: str,
    header: Dict[str, object],
    signature: str,
) -> None:
    """先写临时文件再改名，保证别的进程读到的永远是完整的 RTTM。"""
    atomic_write_text(Path(item.output_path), text)
    atomic_write_json(
        Path(item.meta_path),
        {
            "schema_version": SCHEMA_VERSION,
            "inference_signature": signature,
            "source_size": item.source_size,
            "source_mtime_ns": item.source_mtime_ns,
            "source_sample_rate": item.source_sample_rate,
            "source_num_channels": item.source_num_channels,
            "sample_length": item.sample_length,
            "frames": item.num_frames,
            "segments": header["segments"],
        },
    )


def write_probabilities(item: FeatureItem, raw_prob: np.ndarray) -> None:
    """可选：把逐帧语音概率存成 npz，方便以后换参数重切，不必重跑模型。"""
    path = Path(item.prob_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            np.savez(
                stream,
                schema_version=np.asarray(PROBABILITY_SCHEMA_VERSION, dtype=np.int32),
                raw_prob=np.asarray(raw_prob, dtype=np.float32),
                sample_rate=np.asarray(TARGET_SAMPLE_RATE, dtype=np.int32),
                sample_length=np.asarray(item.sample_length, dtype=np.int64),
                num_frames=np.asarray(item.num_frames, dtype=np.int64),
                frame_shift_ms=np.asarray(10, dtype=np.int32),
                source_sample_rate=np.asarray(item.source_sample_rate, dtype=np.int32),
                source_num_channels=np.asarray(item.source_num_channels, dtype=np.int32),
                source_size=np.asarray(item.source_size, dtype=np.int64),
                source_mtime_ns=np.asarray(item.source_mtime_ns, dtype=np.int64),
            )
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


def validate_sidecar(
    audio_path: Path,
    output_path: Path,
    meta_output_path: Path,
    signature: str,
    audio_root: Optional[Path],
    check_source_stat: bool = True,
    expected_sample_length: Optional[int] = None,
) -> List[str]:
    """重新检查一条音频对应的 RTTM 是否可用。

    返回问题清单，空列表表示通过。检查的项目：

    1. RTTM 存在、能按 NIST 九列格式解析；
    2. 文件编号与音频文件名对应；
    3. 每段时长非负、在音频长度之内、按时间排好且互不重叠；
    4. 头部记录里的来源指纹（文件大小、修改时间）与现在的音频文件一致；
    5. 头部记录里的 ``inference_signature`` 与本次运行的设置一致。

    这个函数只读，不改任何东西。
    """
    errors: List[str] = []
    output_path = Path(output_path)
    if not output_path.is_file():
        return [f"RTTM 不存在：{output_path}"]
    try:
        text = output_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"RTTM 读取失败：{exc}"]
    parsed = parse_rttm(text)
    errors.extend(parsed.errors)
    expected_id = file_id_for(Path(audio_path))
    if parsed.file_id is not None and parsed.file_id != expected_id:
        errors.append(f"文件编号 {parsed.file_id!r} 与音频名推出的 {expected_id!r} 不一致")

    header = parse_header_block(text)
    if header.get("inference_signature") != signature:
        errors.append(
            f"推理标识不一致：文件里是 {header.get('inference_signature')!r}，"
            f"本次运行是 {signature!r}"
        )
    try:
        stored_schema = int(header.get("schema_version", "-1"))
    except ValueError:
        stored_schema = -1
    if stored_schema != SCHEMA_VERSION:
        errors.append(f"schema_version {stored_schema} != {SCHEMA_VERSION}")

    try:
        duration_text = header.get("duration_s")
        if duration_text is None:
            raise ValueError("头部缺少 duration_s")
        audio_duration = float(duration_text)
    except Exception as exc:
        errors.append(f"头部 duration_s 无法解析：{exc}")
        audio_duration = None

    if expected_sample_length is not None:
        stored_samples = header.get("samples_16k")
        try:
            stored_samples_int = int(stored_samples) if stored_samples is not None else -1
        except ValueError:
            stored_samples_int = -1
        if stored_samples_int != expected_sample_length:
            errors.append(
                f"头部 samples_16k {stored_samples_int} 与本次读到的 {expected_sample_length} 不一致"
            )

    # 头部声明了多少段，正文就必须有多少行 SPEAKER。
    #
    # 少了这一比对会有一个很隐蔽的后果：把正文的 SPEAKER 行删掉、只留头部，
    # 这个文件在解析器看来与"合法地没有语音"完全一样（0 段、0 错误），
    # 于是 validate 判它通过、infer 又因为"已经写过且检查通过"而跳过它 ——
    # 语音标注就这么被永久地丢掉了，而且外面看到的是"全部完成、状态 ok"。
    declared = header.get("segments")
    if declared is not None:
        try:
            declared_int = int(declared)
        except (TypeError, ValueError):
            errors.append(f"头部 segments 不是整数：{declared!r}")
            declared_int = None
        if declared_int is not None and declared_int != len(parsed.segments):
            errors.append(
                f"头部声明 {declared_int} 段语音，正文里只有 "
                f"{len(parsed.segments)} 行 SPEAKER —— 两者必须一致"
                f"（正文被截断过？）"
            )

    if audio_duration is not None and not errors:
        errors.extend(check_segments(parsed.segments, audio_duration))

    if check_source_stat:
        try:
            stat = Path(audio_path).stat()
        except Exception as exc:
            errors.append(f"读不到音频文件状态：{exc}")
        else:
            try:
                stored_size = int(header.get("source_size", "-1"))
                stored_mtime = int(header.get("source_mtime_ns", "-1"))
            except ValueError:
                errors.append("头部 source_size/source_mtime_ns 无法解析")
            else:
                if stored_size != int(stat.st_size):
                    errors.append(
                        f"音频大小变了：RTTM 记的是 {stored_size}，现在是 {stat.st_size}"
                    )
                if stored_mtime != int(stat.st_mtime_ns):
                    errors.append("音频修改时间变了，RTTM 可能过期")
    if not Path(meta_output_path).is_file():
        errors.append(f"缺少配套的说明文件：{meta_output_path}")
    return errors


# ---------------------------------------------------------------------------
# 进度与失败记录。
# ---------------------------------------------------------------------------


def write_progress(path: Path, counters: Dict[str, object]) -> None:
    payload = dict(counters)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, payload)


def source_relative(audio_path: Path, audio_root: Optional[Path]) -> str:
    """相对音频根目录的路径，写进 RTTM 头部；不在根目录下就直接写全路径。"""
    audio_path = abspath(audio_path)
    if audio_root is not None:
        try:
            return audio_path.relative_to(abspath(audio_root)).as_posix()
        except ValueError:
            pass
    return audio_path.as_posix()


def resolve_reusable_sidecar(
    audio_path: Path,
    args: argparse.Namespace,
    audio_root: Optional[Path],
    output_root: Optional[Path],
) -> Optional[Path]:
    """找一个已经写好的、可以复用的 RTTM（续跑时用）。

    先看本次要写的位置，再依次看 ``--existing-output-root`` 列出的旧位置。
    只有内容检查全部通过的那一个才会被复用；检查不通过就当它不存在，
    后面照常重算并覆盖。
    """
    candidates: List[Path] = [
        sidecar_path(
            audio_path, audio_root, output_root, args.rttm_suffix, args.root_depth
        )
    ]
    for extra_root in args.existing_output_root or []:
        extra_root = abspath(extra_root)
        candidates.append(
            sidecar_path(
                audio_path, audio_root, extra_root, args.rttm_suffix, args.root_depth
            )
        )
        # 也试一下「只按文件名」的老布局，方便接上早期试跑留下的结果。
        candidates.append(
            extra_root / audio_path.parent.name / f"{audio_path.name}{args.rttm_suffix}"
        )
        candidates.append(extra_root / f"{audio_path.name}{args.rttm_suffix}")
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        errors = validate_sidecar(
            audio_path,
            candidate,
            meta_path(candidate),
            args._signature,
            audio_root,
            check_source_stat=not args.skip_source_stat,
        )
        if not errors:
            return candidate
    return None


def pending_records(
    args: argparse.Namespace,
    signature: str,
    counters: Dict[str, object],
) -> Iterator[AudioRecord]:
    """按分片顺序遍历本 rank 的样本；已经做好且检查通过的直接跳过。"""
    audio_root = abspath(args.audio_root) if args.audio_root else None
    output_root = abspath(args.output_root) if args.output_root else None
    for record in iter_audio_shard(args.manifest, args.rank, args.world_size, args.limit):
        counters["selected"] = int(counters["selected"]) + 1
        audio_path = abspath(record.path)
        if not args.force:
            reused = resolve_reusable_sidecar(audio_path, args, audio_root, output_root)
            if reused is not None:
                counters["skipped_valid"] = int(counters["skipped_valid"]) + 1
                continue
        yield record


# ---------------------------------------------------------------------------
# infer 模式。
# ---------------------------------------------------------------------------


def run_infer(args: argparse.Namespace, params: VadParameters) -> int:
    audio_root = abspath(args.audio_root) if args.audio_root else None
    output_root = abspath(args.output_root) if args.output_root else None
    manifest = abspath(args.manifest)
    model_dir = abspath(args.model_dir)
    firered_root = abspath(args.firered_root)
    model_sha256 = sha256_of_file(model_dir / "model.pth.tar")
    signature = make_inference_signature(
        model_sha256, args.firered_git_sha, params, args.rttm_suffix
    )
    args._signature = signature

    runtime_dir = (output_root or manifest.parent) / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tag = f"rank{args.rank:05d}-of{args.world_size:05d}"

    # 前置检查：会不会有两条音频算出同一个输出路径。
    # 一旦重复，后写的会悄悄盖掉先写的，所以宁可在这里停下来。
    #
    # **必须检查整份清单，不能只查本 rank 那一片。** 只查本片会漏掉跨 rank 的
    # 重名：清单里同一行音频出现两次时，分片会把它分到两个不同的 rank，
    # 两个 rank 各自看自己那一片都是"唯一的"，于是都去写同一个输出路径，
    # 最后两边都报成功、磁盘上只有一个文件。实测过这个情形
    # （两行重复的清单、world_size=2：两个 rank 都 written=1，磁盘上只有 1 个文件）。
    #
    # 代价说明：这里要把整份清单读一遍（一千多万行的清单约需十几秒），
    # 只在启动时做一次，相对于整个任务的时长可以忽略。
    if not args.no_collision_check:
        if args.world_size > 1 and args.limit is None:
            owned_all = list(iter_audio_records(manifest))
        else:
            # 分片或限流的情况下，"整份清单"没有意义（limit 是每 rank 的上限），
            # 退回查本片，并在输出里标明范围。
            owned_all = list(
                iter_audio_shard(manifest, args.rank, args.world_size, args.limit)
            )
        owned = owned_all
        collision = check_mirror_uniqueness(
            owned,
            audio_root,
            output_root,
            args.rttm_suffix,
            args.root_depth,
            beside=output_root is None,
        )
        collision["rank"] = args.rank
        atomic_write_json(runtime_dir / f"collision_check.{tag}.json", collision)
        if not collision["unique"]:
            print(
                json.dumps(
                    {"state": "collision_check_failed", **collision},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            print(
                "本 rank 的输出路径不唯一，已停止，避免互相覆盖。"
                "请把 --root-depth 调大（建议 2 或 3），或改用 --audio-root 让语料名进入路径。",
                file=sys.stderr,
            )
            return 2
    if args.rank == 0:
        atomic_write_json(
            runtime_dir / "run_attestation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "tool": "annotate_firered_offline.py",
                "tool_git_sha": args.tool_git_sha,
                "firered_git_sha": args.firered_git_sha,
                "model_dir": str(model_dir),
                "model_sha256": model_sha256,
                "manifest": str(manifest),
                "manifest_sha256": args.manifest_sha256,
                "audio_root": str(audio_root) if audio_root else None,
                "output_root": str(output_root) if output_root else None,
                "output_layout": "mirror-tree" if output_root else "beside-audio",
                "rttm_suffix": args.rttm_suffix,
                "world_size": args.world_size,
                "physical_gpu_count": args.physical_gpu_count,
                "processes_per_gpu": args.processes_per_gpu,
                "vad_parameters": params.as_dict(),
                "inference_signature": signature,
            },
        )

    progress_path = runtime_dir / f"progress.{tag}.json"
    failure_path = runtime_dir / f"failures.{tag}.jsonl"
    counters: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "rank": args.rank,
        "world_size": args.world_size,
        "selected": 0,
        "skipped_valid": 0,
        "decoded": 0,
        "written": 0,
        "failed": 0,
        "batches": 0,
        "segments": 0,
        "audio_seconds": 0.0,
        "speech_seconds": 0.0,
        "feature_frames": 0,
        "empty_rttm": 0,
        "inference_signature": signature,
        "state": "starting",
    }
    write_progress(progress_path, counters)

    context = mp.get_context("spawn")
    pool = context.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(
            str(firered_root),
            str(model_dir),
            str(audio_root) if audio_root else "",
            str(output_root) if output_root else "",
            args.rttm_suffix,
            args.root_depth,
        ),
    )
    vad = None
    failure_tmp: Optional[Path] = None
    started = time.monotonic()
    exit_code = 0
    try:
        vad = load_vad(firered_root, model_dir, params)
        records = pending_records(args, signature, counters)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(runtime_dir),
            prefix=f".{failure_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as failure_stream:
            failure_tmp = Path(failure_stream.name)
            feature_iter = pool.imap(extract_feature, records, chunksize=args.worker_chunksize)
            feature_pool: List[FeatureItem] = []

            def record_failure(item: FeatureItem, error: str) -> None:
                counters["failed"] = int(counters["failed"]) + 1
                failure_stream.write(
                    json.dumps(
                        {"wav": item.audio_path, "index": item.index, "error": error},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            def flush_pool() -> None:
                nonlocal feature_pool, exit_code
                if not feature_pool:
                    return
                write_futures: List[Tuple[FeatureItem, Future[None]]] = []

                def save(item: FeatureItem, raw_prob: np.ndarray) -> None:
                    if raw_prob.shape != (item.num_frames,):
                        raise ValueError(
                            f"概率长度 {raw_prob.shape} 与帧数 ({item.num_frames},) 不一致："
                            f"{item.audio_path}"
                        )
                    decisions = vad.vad_postprocessor.process(raw_prob.tolist())
                    timestamps = vad.vad_postprocessor.decision_to_segment(
                        decisions, item.duration
                    )
                    header = build_header(
                        item,
                        params,
                        timestamps,
                        model_dir,
                        model_sha256,
                        args.firered_git_sha,
                        args.tool_git_sha,
                        signature,
                        args.rttm_suffix,
                        source_relative(Path(item.audio_path), audio_root),
                    )
                    text = render_rttm(
                        file_id_for(Path(item.audio_path)),
                        timestamps,
                        render_header_block(header),
                    )
                    write_sidecar(item, text, header, signature)
                    if args.store_probs:
                        write_probabilities(item, raw_prob)
                    errors = validate_sidecar(
                        Path(item.audio_path),
                        Path(item.output_path),
                        Path(item.meta_path),
                        signature,
                        audio_root,
                        check_source_stat=False,
                        expected_sample_length=item.sample_length,
                    )
                    if errors:
                        raise ValueError("写完之后自检没通过：" + "；".join(errors))
                    return header

                with ThreadPoolExecutor(
                    max_workers=args.writer_threads,
                    thread_name_prefix=f"rttm-writer-{tag}",
                ) as writer_pool:
                    for batch in plan_batches(
                        feature_pool, args.batch_size, args.max_padded_frames
                    ):
                        try:
                            long_items = [
                                item
                                for item in batch
                                if item.num_frames > params.chunk_max_frame
                            ]
                            normal_items = [
                                item
                                for item in batch
                                if item.num_frames <= params.chunk_max_frame
                            ]
                            pairs: List[Tuple[FeatureItem, np.ndarray]] = []
                            if normal_items:
                                pairs.extend(
                                    zip(
                                        normal_items,
                                        forward_batch_resilient(vad, normal_items),
                                    )
                                )
                                counters["batches"] = int(counters["batches"]) + 1
                            for item in long_items:
                                pairs.append(
                                    (
                                        item,
                                        forward_chunked_single(
                                            vad, item.feat, params.chunk_max_frame
                                        ),
                                    )
                                )
                                counters["batches"] = int(counters["batches"]) + 1
                        except Exception:
                            exit_code = 1
                            batch_error = traceback.format_exc()
                            for item in batch:
                                record_failure(item, batch_error)
                            continue
                        for item, raw_prob in pairs:
                            write_futures.append(
                                (item, writer_pool.submit(save, item, raw_prob))
                            )
                    for item, future in write_futures:
                        try:
                            header = future.result()
                            counters["written"] = int(counters["written"]) + 1
                            counters["segments"] = int(counters["segments"]) + int(
                                header["segments"]
                            )
                            counters["audio_seconds"] = float(counters["audio_seconds"]) + (
                                item.duration
                            )
                            counters["speech_seconds"] = float(
                                counters["speech_seconds"]
                            ) + float(header["speech_s"])
                            counters["feature_frames"] = int(
                                counters["feature_frames"]
                            ) + item.num_frames
                            if int(header["segments"]) == 0:
                                counters["empty_rttm"] = int(counters["empty_rttm"]) + 1
                        except Exception:
                            exit_code = 1
                            record_failure(item, traceback.format_exc())
                feature_pool = []
                counters["state"] = "running"
                counters["elapsed_seconds"] = time.monotonic() - started
                write_progress(progress_path, counters)

            last_report = time.monotonic()
            for item in feature_iter:
                counters["decoded"] = int(counters["decoded"]) + 1
                if item.error is not None:
                    exit_code = 1
                    record_failure(item, item.error)
                else:
                    feature_pool.append(item)
                if len(feature_pool) >= args.batch_pool_size:
                    flush_pool()
                if args.report_every and time.monotonic() - last_report >= args.report_every:
                    done = int(counters["decoded"])
                    elapsed = max(1e-9, time.monotonic() - started)
                    print(
                        f"[infer {tag}] decoded={done} written={counters['written']} "
                        f"skipped={counters['skipped_valid']} failed={counters['failed']} "
                        f"{done / elapsed:.2f} files/s",
                        flush=True,
                    )
                    last_report = time.monotonic()
            flush_pool()
            failure_stream.flush()
            os.fsync(failure_stream.fileno())
        os.replace(failure_tmp, failure_path)
        failure_tmp = None
    finally:
        pool.close()
        pool.join()
        if failure_tmp is not None:
            try:
                failure_tmp.unlink()
            except FileNotFoundError:
                pass
        counters["elapsed_seconds"] = time.monotonic() - started
        counters["state"] = "failed" if exit_code else "completed"
        write_progress(progress_path, counters)
    print(json.dumps(counters, ensure_ascii=False, sort_keys=True), flush=True)
    return exit_code


# ---------------------------------------------------------------------------
# validate 模式。
# ---------------------------------------------------------------------------


def run_validate(args: argparse.Namespace, params: VadParameters) -> int:
    audio_root = abspath(args.audio_root) if args.audio_root else None
    output_root = abspath(args.output_root) if args.output_root else None
    manifest = abspath(args.manifest)
    model_dir = abspath(args.model_dir)
    signature = make_inference_signature(
        sha256_of_file(model_dir / "model.pth.tar"),
        args.firered_git_sha,
        params,
        args.rttm_suffix,
    )
    args._signature = signature
    runtime_dir = (output_root or manifest.parent) / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tag = f"rank{args.rank:05d}-of{args.world_size:05d}"
    summary_path = runtime_dir / f"validation.{tag}.json"
    failure_path = runtime_dir / f"validation_failures.{tag}.jsonl"
    processed = valid = invalid = empty = 0
    segments_total = 0
    speech_seconds = 0.0
    audio_seconds = 0.0
    source_rates: Dict[str, int] = {}
    channels: Dict[str, int] = {}
    tmp_path: Optional[Path] = None
    started = time.monotonic()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(runtime_dir),
            prefix=f".{failure_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            for record in iter_audio_shard(
                manifest, args.rank, args.world_size, args.limit
            ):
                processed += 1
                audio_path = abspath(record.path)
                out_path = sidecar_path(
                    audio_path,
                    audio_root,
                    output_root,
                    args.rttm_suffix,
                    args.root_depth,
                )
                # 允许从 --existing-output-root 指出的旧位置取结果，和续跑时一致。
                reusable = resolve_reusable_sidecar(
                    audio_path, args, audio_root, output_root
                )
                if reusable is not None:
                    out_path = reusable
                errors = validate_sidecar(
                    audio_path,
                    out_path,
                    meta_path(out_path),
                    signature,
                    audio_root,
                    check_source_stat=not args.skip_source_stat,
                )
                if errors:
                    invalid += 1
                    stream.write(
                        json.dumps(
                            {"wav": str(audio_path), "errors": errors},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    valid += 1
                    header = parse_header_block(out_path.read_text(encoding="utf-8"))
                    try:
                        # 从**正文实际解析出的行**统计，而不是头部声明的数字：
                        # 头部是文件自己写的，正文才是真正的内容。
                        parsed_body = parse_rttm(
                            out_path.read_text(encoding="utf-8"))
                        segments_total += len(parsed_body.segments)
                        speech_seconds += float(header.get("speech_s", "0"))
                        audio_seconds += float(header.get("duration_s", "0"))
                    except ValueError:
                        pass
                    if int(header.get("segments", "0")) == 0:
                        empty += 1
                    source_rates[header.get("source_sample_rate", "?")] = (
                        source_rates.get(header.get("source_sample_rate", "?"), 0) + 1
                    )
                    channels[header.get("source_channels", "?")] = (
                        channels.get(header.get("source_channels", "?"), 0) + 1
                    )
                if args.progress_every and processed % args.progress_every == 0:
                    print(
                        f"[validate {tag}] processed={processed} valid={valid} "
                        f"invalid={invalid}",
                        flush=True,
                    )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, failure_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "rank": args.rank,
        "world_size": args.world_size,
        "processed": processed,
        "valid": valid,
        "invalid": invalid,
        "empty_rttm": empty,
        "segments_total": segments_total,
        "speech_seconds": round(speech_seconds, 3),
        "audio_seconds": round(audio_seconds, 3),
        "speech_ratio": round(speech_seconds / audio_seconds, 6) if audio_seconds else None,
        "source_sample_rate_counts": source_rates,
        "source_channel_counts": channels,
        "elapsed_seconds": elapsed,
        "files_per_second": (processed / elapsed) if elapsed > 0 else None,
        "inference_signature": signature,
        "failure_list": str(failure_path),
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if invalid == 0 else 1


# ---------------------------------------------------------------------------
# oracle 模式。
# ---------------------------------------------------------------------------


def run_oracle(args: argparse.Namespace, params: VadParameters) -> int:
    """把批量推理的结果和逐条单独推理的结果做数值对照。

    对照两层：

    * 逐帧语音概率：最大绝对误差必须不超过 ``--oracle-atol``；
    * 切出来的语音段时间戳：必须逐个完全相同。

    逐条那条路走的就是官方 ``vad_model.forward``（单条、不补零），
    批量那条路是本脚本实际用的路径。
    """
    audio_root = abspath(args.audio_root) if args.audio_root else None
    output_root = abspath(args.output_root) if args.output_root else None
    firered_root = abspath(args.firered_root)
    model_dir = abspath(args.model_dir)
    _worker_init(
        str(firered_root),
        str(model_dir),
        str(audio_root) if audio_root else "",
        str(output_root) if output_root else "",
        args.rttm_suffix,
        args.root_depth,
    )
    records = list(
        iter_audio_shard(abspath(args.manifest), args.rank, args.world_size, args.limit or 16)
    )
    items = [extract_feature(record) for record in records]
    errors = [item for item in items if item.error]
    if errors:
        raise RuntimeError(errors[0].error)
    vad = load_vad(firered_root, model_dir, params)

    batched: Dict[str, np.ndarray] = {}
    for batch in plan_batches(items, args.batch_size, args.max_padded_frames):
        for item, prob in zip(batch, forward_batch_resilient(vad, batch)):
            batched[item.audio_path] = prob

    maximum_error = 0.0
    checked = 0
    for item in items:
        if item.num_frames == 0:
            single = np.zeros(0, dtype=np.float32)
        else:
            with torch.inference_mode():
                single_tensor, _ = vad.vad_model.forward(item.feat.unsqueeze(0).cuda())
            single = (
                single_tensor.detach()
                .cpu()
                .reshape(-1)
                .numpy()
                .astype(np.float32, copy=False)
            )
        candidate = batched[item.audio_path]
        if single.shape != candidate.shape:
            raise AssertionError(f"形状不一致：{item.audio_path} {single.shape} vs {candidate.shape}")
        if single.size:
            maximum_error = max(maximum_error, float(np.max(np.abs(single - candidate))))
        if not np.allclose(single, candidate, rtol=args.oracle_rtol, atol=args.oracle_atol):
            raise AssertionError(
                f"批量概率与逐条概率不一致：{item.audio_path} "
                f"max_abs={float(np.max(np.abs(single - candidate)))}"
            )
        single_segments = vad.vad_postprocessor.decision_to_segment(
            vad.vad_postprocessor.process(single.tolist()), item.duration
        )
        candidate_segments = vad.vad_postprocessor.decision_to_segment(
            vad.vad_postprocessor.process(candidate.tolist()), item.duration
        )
        if list(single_segments) != list(candidate_segments):
            raise AssertionError(
                f"语音段时间戳不一致：{item.audio_path}\n"
                f"  逐条：{single_segments}\n  批量：{candidate_segments}"
            )
        checked += 1
    result = {
        "oracle_items": len(items),
        "segments_compared": checked,
        "maximum_absolute_probability_error": maximum_error,
        "segment_timestamps_identical": True,
        "rtol": args.oracle_rtol,
        "atol": args.oracle_atol,
        "status": "passed",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


# ---------------------------------------------------------------------------
# shard-plan 模式：不开 GPU，只打印分片覆盖情况，证明没有重复也没有遗漏。
# ---------------------------------------------------------------------------


def run_shard_plan(args: argparse.Namespace) -> int:
    report = shard_coverage(abspath(args.manifest), args.world_size, args.limit)
    report["manifest_sha256"] = args.manifest_sha256
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["consistent"] else 1


def run_collision_only(args: argparse.Namespace) -> int:
    """只检查"有没有两条音频会写到同一个输出文件"，然后打印结果退出。

    单独做成一个入口，是为了让启动脚本**只做一次**。
    检查要看整份清单（一千多万行、约 23 分钟纯 CPU），
    如果让每个 rank 各做一遍，4 个 rank 就是 4 倍空转，
    而且期间 GPU 一直空闲、日志只有一行，很容易被当成卡死。
    """
    output_root = abspath(args.output_root) if args.output_root else None
    audio_root = abspath(args.audio_root) if args.audio_root else None
    report = check_mirror_uniqueness(
        iter_audio_records(abspath(args.manifest)),
        audio_root,
        output_root,
        args.rttm_suffix,
        args.root_depth,
        beside=output_root is None,
    )
    report["manifest"] = str(abspath(args.manifest))
    report["manifest_sha256"] = args.manifest_sha256
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["unique"] else 1


# ---------------------------------------------------------------------------
# 命令行。
# ---------------------------------------------------------------------------


def parameters_from_args(args: argparse.Namespace) -> VadParameters:
    return VadParameters(
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        min_speech_frame=args.min_speech_frame,
        max_speech_frame=args.max_speech_frame,
        min_silence_frame=args.min_silence_frame,
        merge_silence_frame=args.merge_silence_frame,
        extend_speech_frame=args.extend_speech_frame,
        chunk_max_frame=args.chunk_max_frame,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("infer", "validate", "oracle", "shard-plan"),
        required=True,
        help="要做什么：写标注 / 检查标注 / 数值对照 / 只看分片",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="样本清单（data.list 或同形状 JSONL）")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="写 RTTM 的根目录；不填就写在音频文件旁边",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="音频根目录，镜像树布局时用来算相对路径；也写进 RTTM 头部",
    )
    parser.add_argument(
        "--rttm-suffix",
        default=DEFAULT_RTTM_SUFFIX,
        help=f"RTTM 文件名后缀，默认 {DEFAULT_RTTM_SUFFIX}（00001.wav -> 00001.wav.rttm）",
    )
    parser.add_argument(
        "--root-depth",
        type=int,
        default=DEFAULT_ROOT_DEPTH,
        help=(
            "镜像树布局时，除文件名外再保留几层上层目录名，默认 2；"
            "保留太少会让不同语料里同名的音频互相覆盖，启动前会检查"
        ),
    )
    parser.add_argument(
        "--collision-only",
        action="store_true",
        help="只做「输出路径是否重名」的自检然后退出"
             "（供启动脚本调用一次，避免每个 rank 各做一遍）",
    )
    parser.add_argument(
        "--no-collision-check",
        action="store_true",
        help="跳过「输出路径是否唯一」的前置检查（不推荐）",
    )
    parser.add_argument(
        "--existing-output-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "上一次运行已经写好的输出根目录；续跑时先在这里找同名 RTTM，"
            "找到就直接用，适用于中途换了输出布局的情况"
        ),
    )
    parser.add_argument("--firered-root", type=Path, required=True, help="FireRedVAD 代码目录")
    parser.add_argument("--model-dir", type=Path, required=True, help="FireRedVAD 模型目录")
    parser.add_argument("--firered-git-sha", default="unknown", help="FireRedVAD 代码版本，写进 RTTM 头部")
    parser.add_argument("--tool-git-sha", default="unknown", help="本仓库代码版本，写进 RTTM 头部")
    parser.add_argument("--manifest-sha256", default="unknown", help="样本清单的 sha256，前置检查用")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--physical-gpu-count", type=int)
    parser.add_argument("--processes-per-gpu", type=int)
    parser.add_argument("--limit", type=int, help="每个 rank 最多处理多少条，用于试跑")
    parser.add_argument("--num-workers", type=int, default=16, help="读音频、算特征的子进程数")
    parser.add_argument("--worker-chunksize", type=int, default=8)
    parser.add_argument("--writer-threads", type=int, default=8, help="写 RTTM 的线程数")
    parser.add_argument("--batch-size", type=int, default=1024, help="一次前向最多多少条")
    parser.add_argument("--batch-pool-size", type=int, default=2048)
    parser.add_argument(
        "--max-padded-frames",
        type=int,
        default=1200000,
        help="一批里「最长帧数 × 条数」的上限，控制显存",
    )
    parser.add_argument("--progress-every", type=int, default=1000, help="validate 模式每多少条打印一次")
    parser.add_argument("--report-every", type=float, default=60.0, help="infer 模式每多少秒打印一次进度")
    parser.add_argument("--force", action="store_true", help="不管已有的 RTTM，全部重算")
    parser.add_argument(
        "--skip-source-stat",
        action="store_true",
        help="续跑检查时不去读音频文件状态（音频目录不可读时用）",
    )
    parser.add_argument("--store-probs", action="store_true", help="额外保存逐帧概率 npz（占空间）")
    parser.add_argument("--oracle-rtol", type=float, default=1e-5)
    parser.add_argument("--oracle-atol", type=float, default=1e-5)
    parser.add_argument("--smooth-window-size", type=int, default=5)
    parser.add_argument("--speech-threshold", type=float, default=0.4)
    parser.add_argument("--min-speech-frame", type=int, default=20)
    parser.add_argument("--max-speech-frame", type=int, default=2000)
    parser.add_argument("--min-silence-frame", type=int, default=20)
    parser.add_argument("--merge-silence-frame", type=int, default=0)
    parser.add_argument("--extend-speech-frame", type=int, default=0)
    parser.add_argument("--chunk-max-frame", type=int, default=30000)
    args = parser.parse_args(argv)

    if args.root_depth < 0:
        parser.error("--root-depth 不能为负")
    if args.num_workers <= 0:
        parser.error("--num-workers 必须为正")
    if args.writer_threads <= 0:
        parser.error("--writer-threads 必须为正")
    if args.batch_pool_size < args.batch_size:
        parser.error("--batch-pool-size 不能小于 --batch-size")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        parser.error("--rank/--world-size 不合法")
    if not args.rttm_suffix or "/" in args.rttm_suffix:
        parser.error("--rttm-suffix 不能为空、也不能含斜杠")
    if args.output_root is None and args.audio_root is not None:
        pass  # 旁边写也可以给 audio-root，只用于头部里的相对路径
    if args.physical_gpu_count is None and args.processes_per_gpu is None:
        args.physical_gpu_count = args.world_size
        args.processes_per_gpu = 1
    elif args.physical_gpu_count is None or args.processes_per_gpu is None:
        parser.error("physical-gpu-count 与 processes-per-gpu 要么都给，要么都不给")
    if args.physical_gpu_count <= 0 or args.processes_per_gpu <= 0:
        parser.error("GPU 数与每卡进程数必须为正")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    params = parameters_from_args(args)
    if args.mode == "infer":
        return run_infer(args, params)
    if args.mode == "validate":
        return run_validate(args, params)
    if args.mode == "shard-plan":
        if getattr(args, "collision_only", False):
            return run_collision_only(args)
        return run_shard_plan(args)
    return run_oracle(args, params)


if __name__ == "__main__":
    raise SystemExit(main())
