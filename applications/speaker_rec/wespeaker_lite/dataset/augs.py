"""Composable audio augmentors for speaker verification training.

Supports noise addition, reverberation, codec compression, and low-pass
filtering. Augmentors can be composed via Sequential and OneOf combinators,
and deserialized from YAML/dict configs via ``deserialize_augmentor()``.

Noise and RIR data can be loaded from either:
  - LMDB files (via ``lmdb_file`` parameter, uses LmdbData/LmdbDataMap)
  - Disk directories (via ``dataset_dir`` parameter, globs for audio files)

Usage::

    aug_config = {
        'aug_type': 'Sequential',
        'prob': 1.0,
        'augmentors': [
            {'aug_type': 'OneOf', 'prob': 0.65, 'augmentors': [
                {'aug_type': 'CustomNoise', 'lmdb_file': '/path/to/musan/lmdb',
                 'snr': [0, 15], 'prob': 1.0},
            ]},
            {'aug_type': 'Reverb', 'lmdb_file': '/path/to/rirs/lmdb',
             'prob': 0.25},
            {'aug_type': 'CodecAug', 'prob': 0.3},
        ]
    }
    augmentor = deserialize_augmentor(aug_config)
    # augmentor expects dict with 'wav' (torch Tensor [1,T]) and 'sample_rate'
"""

import io
import os
import glob
import random

import numpy as np
import torch
import soundfile as sf
from scipy import signal as sp_signal
from scipy.io import wavfile
from abc import abstractmethod
from typing import Union, List, Tuple


# ── 是否使用官方（修复前）的增强行为 ──────────────────────────────────
#
# 这个开关是为了**逐位复现官方训练结果**而存在的，平时保持 False。
#
# 本文件有两处修复改变了增强的随机流与内容（都是修真实缺陷）：
#   1. `_random_int` 的上界：官方用 `np.random.randint(lo, hi)`，
#      而本模块的范围是**两端都含**的，于是 hi 永远取不到
#      （babble 只可能抽到 3~6，抽不到 7）。
#   2. `CustomNoise` 取噪声：官方对短文件用 `np.pad(..., 'wrap')` 循环补齐，
#      实测 5%~15% 的抽取会命中比目标短的噪声文件，补出来的尾巴是开头的
#      逐字复制（有一个 0.065 秒的文件被补成 97% 是重复内容）。
#
# 置 True 时这两处都退回官方行为，用于和官方数字做逐位对比；
# **新实验不要打开它**。由配置项 `legacy_aug` 控制（见 dataset.py）。
LEGACY_AUG_BEHAVIOR = False


# ── helpers ──────────────────────────────────────────────────────────

def _random_int(val: Union[int, Tuple[int, int], List[int]]) -> int:
    if isinstance(val, (tuple, list)):
        lo, hi = val
        # The ranges in this module are inclusive at both ends: [3, 7] means
        # "3 to 7 noises" (see the AudioSource.random_one docstring: babble does
        # "3-7 reads/call", and the snr ranges use the same inclusive
        # convention). np.random.randint is exclusive on the upper bound, so the
        # old call never produced hi -- babble only ever drew {3,4,5,6}. The +1
        # restores the documented range. [1, 1] still yields exactly 1.
        if LEGACY_AUG_BEHAVIOR:
            # 官方行为：上界开区间，因此 hi 取不到（保留缺陷，只为复现）。
            return lo if lo >= hi else np.random.randint(lo, hi)
        return lo if lo >= hi else np.random.randint(lo, hi + 1)
    return int(val)


def _random_float(val: Union[float, Tuple[float, float], List[float]]) -> float:
    if isinstance(val, (tuple, list)):
        lo, hi = val
        return lo + np.random.rand() * (hi - lo)
    return float(val)


def _crop_or_pad(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Crop a long signal, or loop-pad a short one with np.pad(..., 'wrap').

    Kept for callers that have no way to fetch a different file. CustomNoise
    does NOT use it for the short case any more -- see
    ``_draw_noise_at_least``: loop-padding a short noise file makes the tail a
    verbatim copy of its head, so the added noise is periodic and the network
    can learn the loop rather than the noise. That helper instead re-draws a
    longer file, and only falls back to this wrap-padding if it cannot find one.
    """
    if len(signal) > target_len:
        # NOTE: the original exclusive-bound crop offset is kept deliberately.
        # Including the last possible start (randint(0, len-target+1)) would be
        # a more uniform crop, but it is an unrequested change to the crop
        # distribution and is not part of this fix.
        start = np.random.randint(0, len(signal) - target_len)
        return signal[start:start + target_len]
    elif len(signal) < target_len:
        return np.pad(signal, (0, target_len - len(signal)), mode='wrap')
    return signal


def _draw_noise_at_least(source, target_len, category=None,
                         max_retry=8, read_frames=None):
    """Draw noise from ``source`` and return a signal of exactly target_len.

    Prefers a file that is already at least ``target_len`` long, so a short
    target is filled with real audio instead of a looped copy of itself (the
    measured problem: on real MUSAN files, 5-15% of draws hit a file shorter
    than the target, and the resulting wrap duplicates ~50% of the samples --
    one 0.065 s file became 97% repeated audio).

    Bounded retry, then the old behaviour: after ``max_retry`` attempts that all
    returned a short file, fall back to ``_crop_or_pad`` (wrap) so this can
    never raise or loop forever. Returns ``(noise, noise_sr)``.
    """
    if read_frames is None:
        read_frames = target_len * 2
    last = None
    for _ in range(max_retry):
        noise, noise_sr = source.random_one(category, frames=read_frames)
        last = (noise, noise_sr)
        if len(noise) >= target_len:
            break
    noise, noise_sr = last
    if len(noise) < target_len:
        noise = _crop_or_pad(noise, target_len)
    return noise, noise_sr


# ── audio source (LMDB or disk) ─────────────────────────────────────

class AudioSource:
    """Lazy-loading audio source from LMDB or a directory of audio files."""

    def __init__(self, lmdb_file=None, dataset_dir=None, file_pattern='**/*.wav',
                 categorized=False):
        self._lmdb = None
        self._disk_files = None
        self._categorized = categorized

        if lmdb_file is not None:
            self._lmdb_path = lmdb_file
            self._source_type = 'lmdb'
        elif dataset_dir is not None:
            self._dataset_dir = dataset_dir
            self._file_pattern = file_pattern
            self._source_type = 'disk'
        else:
            raise ValueError("Either lmdb_file or dataset_dir must be provided")

    def _ensure_loaded(self):
        if self._source_type == 'lmdb' and self._lmdb is None:
            if self._categorized:
                from wespeaker_lite.dataset.lmdb_data import LmdbDataMap
                self._lmdb = LmdbDataMap(self._lmdb_path)
            else:
                from wespeaker_lite.dataset.lmdb_data import LmdbData
                self._lmdb = LmdbData(self._lmdb_path)
        elif self._source_type == 'disk' and self._disk_files is None:
            pattern = os.path.join(self._dataset_dir, self._file_pattern)
            self._disk_files = sorted(glob.glob(pattern, recursive=True))
            if not self._disk_files:
                raise FileNotFoundError(
                    f"No files found matching {pattern}")

    def random_one(self, category=None, frames=None) -> np.ndarray:
        """Return a random audio signal as float32 numpy array.

        If ``frames`` is given, decode at most that many samples from a
        random offset (disk source only). MUSAN files are 5-15 MB each and
        full decode dominates CustomNoise cost (esp. babble: 3-7 reads/call);
        partial decode is ~15x faster with statistically identical sampling
        (uniform random crop, just moved from post-decode to seek-time).
        """
        self._ensure_loaded()

        if self._source_type == 'lmdb':
            _, data = self._lmdb.random_one(category)
            sr, audio = wavfile.read(io.BytesIO(data))
            audio = audio.astype(np.float32)
            if audio.dtype == np.int16 or np.max(np.abs(audio)) > 2.0:
                audio = audio / (1 << 15)
            return audio, sr

        # Retry on read failure: libsndfile/scipy can raise on malformed
        # files. Use context manager so file handles always close (avoids FD
        # leak across millions of opens per worker over an epoch).
        last_err = None
        for _ in range(5):
            path = random.choice(self._disk_files)
            try:
                if frames is not None:
                    with sf.SoundFile(path) as f:
                        sr, n = f.samplerate, f.frames
                        if n > frames:
                            start = np.random.randint(0, n - frames)
                            f.seek(start)
                            audio = f.read(frames, dtype='float32',
                                           always_2d=False)
                        else:
                            audio = f.read(dtype='float32', always_2d=False)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    return audio, sr

                sr, audio = wavfile.read(path)
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 2.0:
                    audio = audio / (1 << 15)
                return audio, sr
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"AudioSource.random_one: 5 consecutive read failures "
            f"(last path={path}, last_err={last_err})")


# ── base augmentor ───────────────────────────────────────────────────

class Augmentor:
    def __init__(self, prob=0.5, aug_type="Augmentor"):
        self.prob = prob

    @abstractmethod
    def __apply__(self, example: dict) -> dict:
        pass

    def __call__(self, example: dict) -> dict:
        if np.random.rand() <= self.prob:
            return self.__apply__(example)
        return example


# ── combinators ──────────────────────────────────────────────────────

class OneOf(Augmentor):
    def __init__(self, augmentors: List[Augmentor], prob=0.5, aug_type="OneOf"):
        super().__init__(prob, aug_type)
        self.augmentors = augmentors

    def __apply__(self, example: dict) -> dict:
        idx = np.random.randint(0, len(self.augmentors))
        return self.augmentors[idx](example)

    def __repr__(self):
        return f"OneOf([{', '.join(repr(a) for a in self.augmentors)}], prob={self.prob})"


class Sequential(Augmentor):
    def __init__(self, augmentors: List[Augmentor], prob=1.0, aug_type="Sequential"):
        super().__init__(prob, aug_type)
        self.augmentors = augmentors

    def __apply__(self, example: dict) -> dict:
        for aug in self.augmentors:
            example = aug(example)
        return example

    def __repr__(self):
        return f"Sequential([{', '.join(repr(a) for a in self.augmentors)}], prob={self.prob})"


# ── noise augmentor ──────────────────────────────────────────────────

class CustomNoise(Augmentor):
    """Add noise at random SNR. Supports babble (multi-noise averaging).

    Noise loaded from LMDB (``lmdb_file``) or disk directory (``dataset_dir``).
    """

    def __init__(self, snr=(0, 15), babble=False,
                 lmdb_file=None, dataset_dir=None, file_pattern='**/*.wav',
                 noise_category=None,
                 prob=0.5, eps=1e-5, aug_type="CustomNoise", **kwargs):
        super().__init__(prob, aug_type)
        self.snr = snr
        self.num_noises = [3, 7] if babble else [1, 1]
        self.eps = eps
        self.noise_category = noise_category
        self._source = AudioSource(
            lmdb_file=lmdb_file, dataset_dir=dataset_dir,
            file_pattern=file_pattern,
            categorized=(lmdb_file is not None and noise_category is not None))

    def __apply__(self, example: dict) -> dict:
        audio = example['wav'].numpy()[0]
        audio_len = audio.shape[0]
        sample_rate = example['sample_rate']

        num_noises = _random_int(self.num_noises)
        noises = []
        # Read with margin for resampling (assume <=2x SR mismatch); full
        # path falls back if SR differs after read.
        read_frames = audio_len * 2
        for _ in range(num_noises):
            # Re-draw a file that is at least audio_len long instead of
            # loop-padding a short one (see _draw_noise_at_least); the resample
            # below can shrink the signal, so request the margin in source
            # frames and let _crop_or_pad handle any remaining shortfall.
            if LEGACY_AUG_BEHAVIOR:
                # 官方行为：直接从源里抽一条，短了就用循环补齐（保留缺陷）。
                noise, noise_sr = self._source.random_one(
                    self.noise_category, frames=read_frames)
            else:
                noise, noise_sr = _draw_noise_at_least(
                    self._source, read_frames, self.noise_category,
                    read_frames=read_frames)
            if noise_sr != sample_rate:
                from scipy.signal import resample as sp_resample
                noise = sp_resample(noise, int(len(noise) / noise_sr * sample_rate))
            noise = _crop_or_pad(noise, audio_len)
            noises.append(noise)

        noise_audio = np.mean(noises, axis=0)
        snr_db = _random_float(self.snr)

        audio_power = np.mean(audio ** 2) + self.eps
        noise_power = np.mean(noise_audio ** 2) + self.eps
        scale = np.sqrt(10 ** ((10 * np.log10(audio_power) -
                                10 * np.log10(noise_power) - snr_db) / 10))

        example['wav'] = torch.from_numpy(audio + noise_audio * scale).unsqueeze(0).float()
        return example

    def __repr__(self):
        return f"CustomNoise(snr={self.snr}, num_noises={self.num_noises}, prob={self.prob})"


# ── reverb augmentor ─────────────────────────────────────────────────

class Reverb(Augmentor):
    """Convolve with a room impulse response.

    RIR loaded from LMDB (``lmdb_file``) or disk directory (``dataset_dir``).
    """

    def __init__(self, rir_length_sec=1.0,
                 lmdb_file=None, dataset_dir=None, file_pattern='**/*.wav',
                 prob=0.5, aug_type="Reverb", **kwargs):
        super().__init__(prob, aug_type)
        self.rir_length_sec = rir_length_sec
        self._source = AudioSource(
            lmdb_file=lmdb_file, dataset_dir=dataset_dir,
            file_pattern=file_pattern)

    def __apply__(self, example: dict) -> dict:
        audio = example['wav'].numpy()[0]
        audio_len = audio.shape[0]
        sample_rate = example['sample_rate']

        rir, rir_sr = self._source.random_one()
        if rir_sr != sample_rate:
            from scipy.signal import resample as sp_resample
            rir = sp_resample(rir, int(len(rir) / rir_sr * sample_rate))

        rir = rir[:int(self.rir_length_sec * sample_rate)]
        rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-10)
        out = sp_signal.convolve(audio, rir, mode='full')[:audio_len]

        example['wav'] = torch.from_numpy(out).unsqueeze(0).float()
        return example

    def __repr__(self):
        return f"Reverb(rir_len={self.rir_length_sec}s, prob={self.prob})"


# ── codec augmentor ──────────────────────────────────────────────────

class CodecAug(Augmentor):
    """Apply random codec compression (mp3, opus, vorbis, g722, g711).

    Requires FFmpeg and torchaudio with AudioEffector support.
    """

    def __init__(self, sample_rate=16000, prob=0.5, aug_type="CodecAug", **kwargs):
        super().__init__(prob, aug_type)
        self.sample_rate = sample_rate
        from torchaudio.io import AudioEffector, CodecConfig
        self.effectors = {
            "mp3-lo":     AudioEffector(format="mp3", codec_config=CodecConfig(qscale=9)),
            "mp3-mid":    AudioEffector(format="mp3", codec_config=CodecConfig(qscale=5)),
            "mp3-hi":     AudioEffector(format="mp3", codec_config=CodecConfig(qscale=2)),
            "vorbis-lo":  AudioEffector(format="ogg", encoder="vorbis", codec_config=CodecConfig(qscale=1)),
            "vorbis-hi":  AudioEffector(format="ogg", encoder="vorbis", codec_config=CodecConfig(qscale=10)),
            "opus-11k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=11000)),
            "opus-16k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=16000)),
            "opus-32k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=32000)),
            "opus-64k":   AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=64000)),
            "g722":       AudioEffector(format="g722"),
            "g711-mulaw": AudioEffector(format="wav", encoder="pcm_mulaw"),
            "g711-alaw":  AudioEffector(format="wav", encoder="pcm_alaw"),
        }

    def __apply__(self, example: dict) -> dict:
        audio = example['wav']  # (1, T) torch tensor
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        effect_name = random.choice(list(self.effectors.keys()))
        effect = self.effectors[effect_name]

        out = effect.apply(audio.T, self.sample_rate).T  # (1, T')
        # Codec may change length slightly — match original
        orig_len = example['wav'].shape[-1]
        if out.shape[-1] > orig_len:
            out = out[..., :orig_len]
        elif out.shape[-1] < orig_len:
            # Keep the tail consistent with the rest of the augmentation chain:
            # wrap (repeat the audio) rather than zero-pad. Zero padding inserted
            # outright silence, which is a different and much easier cue than the
            # codec artefact this augmentor is supposed to teach the model to
            # ignore. The shortfall is tiny (a codec changes length by a few
            # samples, unlike the multi-second noise case), so wrap is both
            # cheap and inaudible here. Note: no current recipe enables CodecAug,
            # so this only affects configs that opt in; it is kept deliberately
            # conservative and does not change the length semantics (output is
            # still exactly orig_len samples).
            pad = orig_len - out.shape[-1]
            if pad >= out.shape[-1]:
                reps = pad // out.shape[-1] + 1
                out = out.repeat(1, reps)[..., :orig_len]
            else:
                out = torch.cat([out, out[..., :pad]], dim=-1)

        example['wav'] = out.float()
        return example

    def __repr__(self):
        return f"CodecAug(codecs={len(self.effectors)}, prob={self.prob})"


# ── low-pass filter augmentor ────────────────────────────────────────

class LowPassAug(Augmentor):
    """Apply a smooth low-pass filter in the frequency domain."""

    def __init__(self, cutoff_ratio=0.5, transition_width=0.1,
                 prob=0.5, aug_type="LowPassAug", **kwargs):
        super().__init__(prob, aug_type)
        self.cutoff_ratio = cutoff_ratio
        self.transition_width = transition_width

    def __apply__(self, example: dict) -> dict:
        audio = example['wav'].numpy()[0]
        T = audio.shape[0]

        rfft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(T)

        cutoff = self.cutoff_ratio * freqs[-1]
        tw = self.transition_width * freqs[-1]

        mask = np.ones_like(freqs)
        if tw <= 0:
            # transition_width=0 means "infinitely sharp cut". The cosine ramp
            # below divides by tw, which produced NaN (0/0) for every bin inside
            # the (empty) transition band and made the whole output NaN. With no
            # transition band the intended filter is the plain brick wall: pass
            # everything below the cutoff, drop everything above it.
            mask[freqs >= cutoff] = 0
            filtered = np.fft.irfft(rfft * mask, n=T)
            example['wav'] = torch.from_numpy(filtered).unsqueeze(0).float()
            return example

        mask[freqs > cutoff + tw / 2] = 0
        transition = np.logical_and(freqs >= cutoff - tw / 2,
                                     freqs <= cutoff + tw / 2)
        mask[transition] = 0.5 * (1 + np.cos(
            np.pi * (freqs[transition] - cutoff) / tw))

        filtered = np.fft.irfft(rfft * mask, n=T)

        example['wav'] = torch.from_numpy(filtered).unsqueeze(0).float()
        return example

    def __repr__(self):
        return f"LowPassAug(cutoff={self.cutoff_ratio}, prob={self.prob})"


# ── deserialization ──────────────────────────────────────────────────

_AUG_CLASSES = {
    'Sequential': Sequential,
    'OneOf': OneOf,
    'CustomNoise': CustomNoise,
    'Reverb': Reverb,
    'CodecAug': CodecAug,
    'LowPassAug': LowPassAug,
}


def deserialize_augmentor(config: dict) -> Augmentor:
    """Build an augmentor tree from a nested dict (YAML-friendly).

    Example config::

        {
            'aug_type': 'Sequential',
            'prob': 1.0,
            'augmentors': [
                {'aug_type': 'CustomNoise', 'lmdb_file': '...', 'snr': [0,15], 'prob': 0.65},
                {'aug_type': 'Reverb', 'lmdb_file': '...', 'prob': 0.25},
                {'aug_type': 'CodecAug', 'prob': 0.3},
            ]
        }
    """
    config = dict(config)  # shallow copy
    aug_type = config.pop('aug_type')

    if aug_type not in _AUG_CLASSES:
        raise ValueError(f"Unknown augmentor type: {aug_type}")

    cls = _AUG_CLASSES[aug_type]

    if aug_type in ('Sequential', 'OneOf'):
        children = [deserialize_augmentor(c) for c in config.pop('augmentors')]
        return cls(augmentors=children, **config)
    else:
        return cls(**config)
