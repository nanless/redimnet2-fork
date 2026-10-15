# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from wespeaker_lite.utils.file_utils import read_lists
from wespeaker_lite.dataset.lmdb_data import DiskDataMap, LmdbData, LmdbDataMap
import wespeaker_lite.dataset.processor as processor
from wespeaker_lite.dataset.processor import read_audio_section


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        try:
            self.source.set_epoch(epoch)
        except:
            pass

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)

    def __len__(self):
        try:
            return len(self.source)
        except:
            return None


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        # 说明（实测过，不是猜测）：`data[rank::world_size]` 对**最后一个 rank**
        # 才是整除的，前面的 rank 会多出 1 条。以本机 VoxCeleb2 开发集
        # （1,092,009 条）为例：
        #     8 卡: rank0 136,502 条，其余 rank 各 136,501 条
        #     4 卡: rank0 273,003 条，其余各 273,002 条
        #     2 卡: rank0 546,005 条，其余各 546,004 条
        # 也就是每个 rank 之间最多差 1 条。
        #
        # 这**不影响训练**：训练循环按 `epoch_iter` 截断
        # （`executor.run_epoch` 里 `if (i + 1) == epoch_iter: break`），
        # 而 `epoch_iter` 取的是 `总数 // world_size // batch_size`，
        # 比任何 rank 的实际条数都小，所以多出来的那一条永远读不到。
        #
        # 之所以不改成"均匀切分"：这是老实验依赖的行为，改了会让已有实验
        # 看到的数据顺序变化。只在注释里说清，不动代码。
        return data


class DataList(IterableDataset):

    def __init__(self,
                 lists,
                 shuffle=True,
                 partition=True,
                 repeat_dataset=True):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.lists)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            # Base seed for deterministic re-seeding at each data pass.
            # Derived from the worker's initial random state (set by seed_worker).
            _base_seed = random.getrandbits(64)
            _pass = 0
            while True:
                if counter % indexes_len == 0:
                    # Re-seed random/np at each pass over the data so that
                    # augmentation/shuffle stay deterministic regardless of
                    # prefetch timing between workers.
                    random.seed(_base_seed + _pass)
                    np.random.seed((_base_seed + _pass) % (2**31))
                    _pass += 1
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data


def is_varlen_enabled(configs):
    """ Whether the variable-length training path is switched on.

        The switch lives at ``dataset_args.varlen.enable`` and defaults to
        false, so every recipe written before this feature keeps the
        fixed-length behaviour bit for bit.
    """
    varlen_cfg = configs.get('varlen', None)
    if not isinstance(varlen_cfg, dict):
        return False
    return bool(varlen_cfg.get('enable', False))


def get_varlen_length_bounds(configs):
    """ (min_sec, max_sec) bounds used by the variable-length path.

        Shared by the reader stage, the random-crop stage and the collate-time
        batch cap so all three agree on what a legal length is.
    """
    varlen_cfg = configs.get('varlen', None)
    if not isinstance(varlen_cfg, dict):
        varlen_cfg = {}
    min_sec = float(varlen_cfg.get('min_sec', 1.0))
    max_sec = float(varlen_cfg.get('max_sec', 8.0))
    assert max_sec > min_sec > 0, (min_sec, max_sec)
    return min_sec, max_sec


# 按语音活动（VAD）边界取片段的默认值，集中放一处，方便文档和测试引用。
VAD_SEGMENT_DEFAULTS = {
    'enable': False,
    'rttm_suffix': '.rttm',
    # 分段文件和音频放在一起（默认），还是放在另一棵树里按同样的相对路径镜像过去。
    # 生成分段的那条流程两种都支持，训练这边必须能对上：
    #   rttm_root: /some/where/rttm   音频 /a/b/c/00001.wav → /some/where/rttm/b/c/00001.wav.rttm
    #   rttm_root_depth: 2            保留音频路径最后 2 层目录（b/c/00001.wav）
    # 不写 rttm_root 就是「放在音频旁边」。
    'rttm_root': None,
    'rttm_root_depth': 2,
    'target_sec': None,        # None 表示取 min_sec 与 max_sec 的中点
    'pad_sec': 0.1,
    'min_speech_sec': None,    # None 表示等于 min_sec
    'merge_gap_sec': None,     # None 表示 0.0（只并真正重叠或首尾相接的段）
    'cut_policy': 'middle',    # 单段语音长于 max_sec 时：middle | random
    'missing_policy': 'fallback',       # 没有分段文件：fallback | drop
    'low_speech_policy': 'resample',    # 语音太少：resample | keep | drop
    'low_speech_retries': 2,
    'read_fail_policy': 'fallback',     # 片段读失败：fallback | drop
    'fallback_chunk_sec': 3.0,
    'audio_root': None,        # 用 rttm_root 镜像时，音频路径的公共前缀
}


def is_vad_segments_enabled(configs):
    """ 是否按语音活动边界取样段。

        开关在 `dataset_args.vad_segments.enable`，默认 false，
        所以没写这一块的配方行为完全不变。
    """
    cfg = configs.get('vad_segments', None)
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get('enable', False))


def get_vad_segment_config(configs):
    """ 取按语音活动取样的配置，缺的项用默认值补齐并做基本检查。 """
    cfg = configs.get('vad_segments', None)
    if not isinstance(cfg, dict):
        cfg = {}
    out = dict(VAD_SEGMENT_DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k in VAD_SEGMENT_DEFAULTS})
    # 时长范围：没写就沿用变长路径那一套，保证两处对"合法长度"的理解一致。
    varlen_min, varlen_max = get_varlen_length_bounds(configs)
    out['min_sec'] = float(cfg.get('min_sec', varlen_min))
    out['max_sec'] = float(cfg.get('max_sec', varlen_max))
    assert out['max_sec'] > out['min_sec'] > 0, (out['min_sec'], out['max_sec'])
    if out['target_sec'] is not None:
        out['target_sec'] = float(out['target_sec'])
    if out['min_speech_sec'] is not None:
        out['min_speech_sec'] = float(out['min_speech_sec'])
    out['pad_sec'] = float(out['pad_sec'])
    assert out['pad_sec'] >= 0.0, out['pad_sec']
    assert out['rttm_suffix'], "rttm_suffix must not be empty"
    assert out['cut_policy'] in ('middle', 'random'), out['cut_policy']
    assert out['missing_policy'] in ('fallback', 'drop'), out['missing_policy']
    assert out['low_speech_policy'] in ('resample', 'keep', 'drop'), \
        out['low_speech_policy']
    assert out['read_fail_policy'] in ('fallback', 'drop'), \
        out['read_fail_policy']
    out['low_speech_retries'] = int(out['low_speech_retries'])
    out['rttm_root_depth'] = int(out['rttm_root_depth'])
    assert out['rttm_root_depth'] >= 1, out['rttm_root_depth']
    return out


def rttm_path_for_audio(wav_path, rttm_suffix='.rttm',
                        rttm_root=None, rttm_root_depth=2,
                        audio_root=None):
    """音频路径 → 分段文件路径。

    和生成分段的那条流程（`applications/speaker_rec/scripts/vad/`）用同一套规则：

    * 不写 `rttm_root`：分段就放在音频旁边，文件名后面直接补后缀
      （`a/b/00001.wav` → `a/b/00001.wav.rttm`）。
    * 写了 `rttm_root`：把音频路径的最后 `rttm_root_depth` 层目录连同文件名
      搬到 `rttm_root` 下面（音频树只读、或分段单独放一处时用这种）。
      给了 `audio_root` 就先减掉这个前缀再搬。

    这里只做路径拼装，不判断文件在不在——「在不在」由读文件那一步决定。
    """
    name = os.path.basename(wav_path) + rttm_suffix
    if not rttm_root:
        return wav_path + rttm_suffix
    path = wav_path
    if audio_root:
        try:
            rel = os.path.relpath(wav_path, audio_root)
        except ValueError:
            rel = wav_path
        # relpath 会给出 ../.. 这样的结果，说明不在同一棵树里，那就退回整条路径。
        path = wav_path if rel.startswith('..') else os.path.join(audio_root, rel)
    parts = path.split(os.sep)
    parts = parts[-(int(rttm_root_depth) + 1):]
    # 文件名要单独补上后缀：path 是音频路径，最后一段还是 `00001.wav`。
    parts[-1] = parts[-1] + rttm_suffix
    return os.path.join(rttm_root, *parts)


def vad_stats():
    """ 按语音活动取样这一步的计数（每个进程一份）。

        真正的定义和打印都在 processor.py 里（计数是在那一层产生的），
        这里只是给个方便的入口，省得调用方去记在哪个模块。
    """
    return processor.VAD_STATS


def report_vad_stats(prefix="[dataset] VAD 取样统计"):
    """ 打印上面那份计数。 """
    processor.report_vad_stats(prefix=prefix)



def Dataset(data_type,
            data_list_file,
            configs,
            spk2id_dict,
            verbose=False,
            whole_utt=False,
            reverb_lmdb_file=None,
            noise_lmdb_file=None,
            repeat_dataset=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw/feat file level. The second is local shuffle
        at training samples level.

        Args:
            data_type(str): shard/raw/raw_fast
            data_list_file: data list file
            configs: dataset configs
            spk2id_dict: spk2id dict
            reverb_lmdb_file: reverb data source lmdb file
            noise_lmdb_file: noise data source lmdb file
            whole_utt: use whole utt or random chunk
    """
    assert data_type == 'raw_fast'
    # Global shuffle
    lists = read_lists(data_list_file)
    shuffle = configs.get('shuffle', False)
    vadseg = is_vad_segments_enabled(configs)
    if vadseg and not is_varlen_enabled(configs):
        # 按语音活动取样的片段长度本来就是变化的，只有走变长读取才拿得到
        # 真实长度、也只有变长那条 collate 会补零并回传 lengths。
        # 直接报错，好过悄悄按固定长度跑出一个长度不对的结果。
        raise ValueError(
            "dataset_args.vad_segments.enable=true 需要同时打开 "
            "dataset_args.varlen.enable=true：按语音活动边界取样的片段长度"
            "是变化的，必须走变长读取与右侧补零那条路径。")
    varlen = is_varlen_enabled(configs)
    min_sec, max_sec = get_varlen_length_bounds(configs)
    dataset = DataList(lists, shuffle=shuffle, repeat_dataset=repeat_dataset)

    num_frms = configs.get('num_frms', 200)
    frame_shift = configs['fbank_args'].get('frame_shift', 10)
    frame_length = configs['fbank_args'].get('frame_length', 25)
    resample_rate = configs.get('resample_rate', 16000)
    chunk_len = ((num_frms - 1) * frame_shift +
                 frame_length) * resample_rate // 1000
    filter_conf = configs.get('filter_args', {})
    if varlen:
        print(f"VARLEN : enabled, clips read at their own length, "
              f"min_sec={min_sec}, max_sec={max_sec}")
        dataset = Processor(dataset, processor.parse_raw_fast_varlen,
                            min_sec, max_sec, data_type, **filter_conf)
    else:
        dataset = Processor(dataset, processor.parse_raw_fast_and_filter,
                            chunk_len, data_type, **filter_conf)


    # Local shuffle
    if shuffle:
        dataset = Processor(dataset, processor.shuffle,
                            **configs['shuffle_args'])

    # spk2id
    dataset = Processor(dataset, processor.spk_to_id, spk2id_dict)

    # resample
    resample_rate = configs.get('resample_rate', 16000)
    dataset = Processor(dataset, processor.resample, resample_rate)
    # speed perturb
    speed_perturb_flag = configs.get('speed_perturb', True)
    if speed_perturb_flag:
        dataset = Processor(dataset, processor.speed_perturb,
                            len(spk2id_dict))
    if (not whole_utt):
        # random chunk
        if vadseg:
            # 按语音活动边界取片段：从"段边界到段边界"挑一段完整的语音，
            # 替代随机位置的切法。片段长度仍落在 [min_sec, max_sec] 里，
            # 所以下游的分桶与补零逻辑原样可用。
            vad_cfg = get_vad_segment_config(configs)
            print(f"VAD_SEG : enabled, {vad_cfg}")
            dataset = Processor(dataset, processor.vad_select_segment,
                                rttm_suffix=vad_cfg['rttm_suffix'],
                                rttm_root=vad_cfg['rttm_root'],
                                rttm_root_depth=vad_cfg['rttm_root_depth'],
                                audio_root=vad_cfg['audio_root'],
                                min_sec=vad_cfg['min_sec'],
                                max_sec=vad_cfg['max_sec'],
                                target_sec=vad_cfg['target_sec'],
                                pad_sec=vad_cfg['pad_sec'],
                                merge_gap_sec=vad_cfg['merge_gap_sec'],
                                min_speech_sec=vad_cfg['min_speech_sec'],
                                cut_policy=vad_cfg['cut_policy'],
                                low_speech_policy=vad_cfg['low_speech_policy'],
                                low_speech_retries=vad_cfg['low_speech_retries'],
                                missing_policy=vad_cfg['missing_policy'],
                                read_fail_policy=vad_cfg['read_fail_policy'],
                                fallback_chunk_sec=vad_cfg['fallback_chunk_sec'])
        elif varlen:
            # Variable-length path: the crop is a random length inside the
            # configured bounds rather than a fixed chunk_len, so the batch
            # still carries utterances of differing true lengths.
            dataset = Processor(dataset, processor.random_chunk_range,
                                min_sec, max_sec)
        else:
            num_frms = configs.get('num_frms', 200)
            frame_shift = configs['fbank_args'].get('frame_shift', 10)
            frame_length = configs['fbank_args'].get('frame_length', 25)
            chunk_len = ((num_frms - 1) * frame_shift +
                         frame_length) * resample_rate // 1000
            dataset = Processor(dataset, processor.random_chunk, chunk_len,
                                data_type)

    # ── Standard augmentation ──
    if configs.get("aug_setup", None) is not None:
        aug_cfg = configs['aug_setup']
        # New format: composable Sequential/OneOf tree from dataset.augs.
        # Detected by presence of 'aug_type' at the root (vs legacy 'type').
        if 'aug_type' in aug_cfg:
            from wespeaker_lite.dataset.augs import deserialize_augmentor
            augmentor = deserialize_augmentor(aug_cfg)
            print(f"AUG_TYPE : augmentor_chain -> {repr(augmentor)}")
            dataset = Processor(dataset, processor.apply_augmentor,
                                augmentor, resample_rate)
        else:
            # Legacy string format: both 'wespeaker_augs' and 'wespeaker_augs_v2'
            # use the level-faithful chain (reverb -> noise@SNR -> random gain, no
            # RMS norm) via add_reverb_noise_gain. Prefer raw wav dirs; LMDB keys
            # remain for backward compatibility.
            aug_type = aug_cfg['type']
            print(f"AUG_TYPE : {aug_type}")
            if aug_cfg.get('reverb_dataset_dir'):
                reverb_data = DiskDataMap(
                    aug_cfg['reverb_dataset_dir'], categorized=False)
            elif aug_cfg.get('reverb_lmdb_file'):
                reverb_data = LmdbDataMap(aug_cfg['reverb_lmdb_file'])
            else:
                raise ValueError(
                    "aug_setup requires reverb_dataset_dir or reverb_lmdb_file")
            if aug_cfg.get('noise_dataset_dir'):
                noise_data = DiskDataMap(
                    aug_cfg['noise_dataset_dir'], categorized=True)
            elif aug_cfg.get('noise_lmdb_file'):
                noise_data = LmdbDataMap(aug_cfg['noise_lmdb_file'])
            else:
                raise ValueError(
                    "aug_setup requires noise_dataset_dir or noise_lmdb_file")
            dataset = Processor(dataset, processor.add_reverb_noise_gain,
                                reverb_data, noise_data, resample_rate,
                                aug_cfg.get('aug_prob', 0.6),
                                aug_cfg.get('preset', 'default'),
                                gain_prob=aug_cfg.get('gain_prob', 1.0),
                                gain_db_range=tuple(
                                    aug_cfg.get('gain_db_range', (-18.0, 6.0))),
                                speech_rms_mode=aug_cfg.get(
                                    'speech_rms_mode', 'full'),
                                peak_protect=aug_cfg.get('peak_protect', True),
                                noise_categories=aug_cfg.get(
                                    'noise_categories', None))

    return dataset

def collate_func_single(batch):
    assert len(batch) == 1
    (x0,x1),y = batch[0]
    return (x0,x1),y

class Vox1TestEvalDataset(torch.utils.data.Dataset):
    def __init__(self,
                    vox1_test_root,
                    wavs_subroot='wav',
                    vox1_veri_test2_list=None,
                    samplerate = 16000,
                    chunk_len_sec = 3.0,
                    num_chunks = 5
                ):
        self.vox1_test_root = Path(vox1_test_root)
        self.wavs_subroot = wavs_subroot

        if vox1_veri_test2_list is None:
            vox1_veri_test2_list = self.vox1_test_root/'veri_test2.txt'

        vox1_protocol = pd.read_csv(vox1_veri_test2_list,sep=" ",
                                         names=['label','enroll','verify'])

        trials = vox1_protocol['enroll'].values.tolist() + vox1_protocol['verify'].values.tolist()
        self.trials = sorted(list(set(trials)))
        assert all([(self.vox1_test_root/self.wavs_subroot/rlp).exists() for rlp in self.trials])
        trials_ind_map = {key:ind for ind, key in enumerate(self.trials)}

        vox1_protocol['enroll_ind'] = vox1_protocol.enroll.map(trials_ind_map)
        vox1_protocol['verify_ind'] = vox1_protocol.verify.map(trials_ind_map)
        self.vox1_protocol = vox1_protocol

        self.samplerate = samplerate
        self.chunk_len_sec = chunk_len_sec
        self.num_chunks = num_chunks

    def __len__(self):
        return len(self.trials)

    @staticmethod
    def slice_samples_into_chunks(
        samples,
        samplerate=16_000,
        chunk_len_sec=3.0,
        num_chunks=5
    ):
        chunk_len = int(chunk_len_sec*samplerate)
        n = len(samples)

        if n < chunk_len:
            # Audio shorter than one chunk. np.linspace(0, n - chunk_len, ...)
            # produced NEGATIVE offsets here, and each slice then came back with
            # a different length, so np.stack raised
            # "all input arrays must have the same shape" (e.g. a 47999-sample
            # clip against a 48000-sample chunk). Pad the tail by wrapping so
            # every chunk is exactly chunk_len, matching how the training path
            # fills a short window; the caller still gets (num_chunks, chunk_len).
            padded = np.pad(samples, (0, chunk_len - n), mode='wrap')
            return np.stack([padded[:chunk_len]] * num_chunks)

        # Long enough for a full chunk: unchanged behaviour. Offsets are spread
        # over [0, n - chunk_len], so every slice is exactly chunk_len long.
        chunked = []
        for shift in np.linspace(0, n - chunk_len, num_chunks).astype(int):
            chunked.append(samples[shift: shift + chunk_len])
        return np.stack(chunked)

    def __getitem__(self,index):
        path = self.vox1_test_root/self.wavs_subroot/self.trials[index]
        label = str(path.relative_to(self.vox1_test_root))

        samples, sr = sf.read(path)
        assert sr == self.samplerate
        chunks = self.slice_samples_into_chunks(samples,
                    samplerate=self.samplerate,
                    chunk_len_sec=self.chunk_len_sec,
                    num_chunks=self.num_chunks)

        x0 = torch.from_numpy(samples).float()[None,:]
        x1 = torch.from_numpy(chunks).float()
        y = torch.tensor(index).long().unsqueeze(0)

        return (x0,x1), y

def collate_func_mult(batch):
    tbatch = tuple(zip(*batch))
    return [torch.stack(xyz,dim=0) for xyz in tbatch]


def collate_varlen(batch, max_batch_sec=None, sample_rate=16000):
    """ Zero pad a list of variable-length utterances to the batch maximum.

        Used as the DataLoader `collate_fn` on the variable-length training
        path. Each item is `{key, label, wav (1, T_b), num_samples}` with a
        possibly different `T_b`; the returned `wav` is `(B, 1, T_max)` with
        every shorter utterance zero padded **on the right**, which is the
        layout `make_sample_mask` assumes, and `lengths` is the matching
        `(B,)` long tensor of real sample counts in samples of that padded
        waveform.

        `max_batch_sec` (optional, in seconds) caps how much audio one batch
        may hold: items are taken in the order the loader produced them until
        the next one would push the total past the cap, so a batch can come
        back smaller than `batch_size`. The first item is always kept, so a
        batch can never be empty whatever the cap is.

        Args:
            batch: List[{key, label, wav, num_samples}]
            max_batch_sec: optional cap on the summed true duration
            sample_rate: samples per second of `wav`, used for the cap

        Returns:
            Dict{key, label, wav, lengths}
    """
    # Bucketing assembles its own batches and hands them over wrapped so that
    # torch's auto-collation leaves them alone. The DataLoader still gathers
    # `batch_size` of those wrappers per step, so what arrives here is a list
    # of wrappers, each already holding a whole bucket batch. Flatten them into
    # the single list of samples this function expects; a plain list of samples
    # (the ordinary unbucketed path) passes through unchanged.
    if isinstance(batch, PreBatched):
        batch = batch.items
    if len(batch) and any(isinstance(x, PreBatched) for x in batch):
        flat = []
        for x in batch:
            flat.extend(x.items if isinstance(x, PreBatched) else [x])
        batch = flat

    if max_batch_sec is not None and max_batch_sec > 0:
        cap = int(max_batch_sec * sample_rate)
        taken = [batch[0]]
        total = int(batch[0]['num_samples'])
        for item in batch[1:]:
            n = int(item['num_samples'])
            if total + n > cap:
                break
            taken.append(item)
            total += n
        batch = taken

    keys = [item['key'] for item in batch]
    # `spk_to_id` stores a plain int and nothing later turns it into a tensor,
    # so accept both rather than assuming the item is already a 1-element
    # tensor (the fixed-length path stack-collates and gets the tensor for
    # free).
    labels = torch.as_tensor(
        [int(item['label']) for item in batch], dtype=torch.long)
    lengths = torch.tensor([int(item['num_samples']) for item in batch],
                           dtype=torch.long)
    max_len = int(lengths.max().item())
    wav = torch.zeros(len(batch), 1, max_len, dtype=torch.float32)
    for i, item in enumerate(batch):
        n = int(item['num_samples'])
        wav[i, :, :n] = item['wav'][0, :n]
    return dict(key=keys, label=labels, wav=wav, lengths=lengths)


# ── length bucketing ─────────────────────────────────────────────────
#
# Bucketing is a training strategy, not part of the model: it only decides
# which utterances end up in the same batch, so that the zero padding a batch
# needs stays small. It is therefore a separate, optional block
# (`dataset_args.bucketing`) and never touches ReDimNet2 itself.
#
# The training dataset is an infinite IterableDataset whose samples arrive in
# speaker-shuffled order, so there is no indexable table to group by length up
# front. The sampler below instead keeps one buffer per length bucket, fills
# them from the incoming stream, and emits a whole batch from a single bucket
# once that bucket has enough samples. Utterances from different buckets are
# never mixed, which is the point of the strategy.

def bucket_index(length_sec, bucket_width_sec, bucket_count):
    """ Which bucket a clip of `length_sec` seconds belongs to.

        Buckets are half-open intervals of `bucket_width_sec` starting at 0;
        everything at or beyond the last edge goes into the last bucket, so no
        clip is ever dropped for being too long.
    """
    idx = int(length_sec / bucket_width_sec)
    return min(idx, bucket_count - 1)


class PreBatched:
    """A batch already assembled by the dataset, not by the DataLoader.

    `BucketedLengthDataset` decides the batch itself (it must: one batch has to
    come from one length bucket), but the `DataLoader` it is handed to was
    built with the recipe's `batch_size`, so torch would otherwise try to
    batch these batches again -- wrapping each yielded group of `batch_size`
    dicts into a list-of-`batch_size`-items and handing *that* to
    `collate_varlen`, which then fails with
    `TypeError: list indices must be integers or slices, not str`.

    Wrapping the group in an object that is not a list/tuple/dict keeps torch's
    auto-collation from touching it (see `_utils/collate.default_convert`,
    which only recurses into those three container types). `collate_varlen`
    unwraps the group and pads it. It is a `list` subclass, so `len()` and
    iteration keep working if anything else looks at it.
    """

    __slots__ = ('items',)

    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


class BucketedLengthDataset(IterableDataset):
    """ Wrap a training dataset so each batch comes from one length bucket.

        An `IterableDataset` rather than a generator function for two reasons
        that both matter to the training loop:

          * `training_loop` calls `train_dataset.set_epoch(epoch)` once per
            epoch, and that call has to reach the inner `DataList.sampler`,
            which is what actually re-shuffles the data. A bare generator has
            no `set_epoch`, so wrapping the dataset in one broke training with
            `AttributeError: 'generator' object has no attribute 'set_epoch'`.
          * Being an `IterableDataset`, it can be handed straight to the
            `DataLoader` and gets the usual worker forking, which is what the
            per-worker stream split below relies on.

        Args:
            dataset: the training dataset, yielding `{key, wav, label,
                num_samples, ...}` with `num_samples` set by the
                variable-length reader
            batch_size: samples per emitted batch
            num_workers: DataLoader worker count, used only when no worker
                context is active (tests, `num_workers=0`); the real worker
                count is read from `torch.utils.data.get_worker_info()`
            bucket_count: number of length buckets
            bucket_width_sec: width of one bucket, in seconds
            shuffle_buckets: pick the next bucket at random instead of in a
                fixed order, so the batch length varies over training
            max_fill_rounds: how many times the loop may look for a full bucket
                before giving up and emitting the fullest buffer it has
    """

    def __init__(self, dataset, batch_size, num_workers=1, bucket_count=8,
                 bucket_width_sec=1.0, shuffle_buckets=True,
                 max_fill_rounds=4):
        assert batch_size >= 1, batch_size
        assert bucket_count >= 1, bucket_count
        assert bucket_width_sec > 0, bucket_width_sec
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bucket_count = bucket_count
        self.bucket_width_sec = bucket_width_sec
        self.shuffle_buckets = shuffle_buckets
        self.max_fill_rounds = max_fill_rounds

    def set_epoch(self, epoch):
        # Forward to the inner dataset; this is what keeps the per-epoch
        # shuffle working through the bucketing wrapper.
        try:
            self.dataset.set_epoch(epoch)
        except AttributeError:
            pass

    def __len__(self):
        try:
            return len(self.dataset)
        except Exception:
            return None

    def __iter__(self):
        batch_size = self.batch_size
        bucket_count = self.bucket_count
        bucket_width_sec = self.bucket_width_sec
        shuffle_buckets = self.shuffle_buckets
        max_fill_rounds = self.max_fill_rounds

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers_effective = (self.num_workers if worker_info is None
                                 else worker_info.num_workers)
        # With no worker context (num_workers=0, or a direct call) there is no
        # stream to split, so treat it as a single worker.
        num_workers_effective = max(1, int(num_workers_effective))

        rng = random.Random(random.getrandbits(64) + worker_id)

        buffers = [[] for _ in range(bucket_count)]
        full = [False] * bucket_count
        in_flight = 0  # samples held by this worker across all buckets
        stream = iter(self.dataset)
        exhausted = False
        # 这里**不要再按 DataLoader worker 切一次流**。
        #
        # 内层的 `DataList.sampler.sample()` 已经做过 worker 分片了
        # （`data[self.worker_id::self.num_workers]`），每个 worker 拿到的本来就是
        # 「每 num_workers 条取一条」的子流。之前这里又按
        # `cursor % num_workers_effective == worker_id` 过滤了一遍，
        # 于是每个 worker 只剩自己那 1/num_workers，合起来只有 1/num_workers 的语料。
        #
        # 实测（160 条清单、batch_size=8、读到 40 批为止，统计出现过多少个不同样本）：
        #     num_workers=1  普通 160 / 分桶 160
        #     num_workers=2  普通 160 / 分桶  80
        #     num_workers=4  普通 160 / 分桶  40
        #     num_workers=8  普通 160 / 分桶  20
        # 丢失比例正好 1/num_workers。而配方里 num_workers 是 16，
        # 也就是说开了分桶之后每个 epoch 实际只看到约 1/16 的语料。
        # 丢掉的是固定步长、不是随机，所以损失曲线照样正常下降，
        # 只有收敛变慢——从日志上几乎看不出。
        #
        # 所以这里直接消费内层给的流，不再过滤。

        while True:
            if not exhausted and not any(full):
                # Read until some bucket holds a whole batch.
                #
                # This used to stop after `num_workers * batch_size` samples,
                # which is far too few: with `bucket_count` buckets in play a
                # single bucket only receives about 1/bucket_count of the
                # stream, so no bucket could ever reach `batch_size` within that
                # budget. The loop then fell through to the "emit the fullest
                # buffer" branch below and handed back batches of one or two
                # items, which stalled the DataLoader and broke the training
                # loop's step count. The budget below leaves room for a whole
                # batch to land in the least populated bucket, with a generous
                # multiplier for a skewed length distribution.
                lookahead_limit = bucket_count * batch_size * max(1, num_workers_effective) * max_fill_rounds
                for _ in range(lookahead_limit):
                    if any(full):
                        break
                    try:
                        sample = next(stream)
                    except StopIteration:
                        exhausted = True
                        break
                    width = int(sample.get('num_samples', sample['wav'].size(-1)))
                    sec = width / float(sample.get('sample_rate', 16000))
                    b = bucket_index(sec, bucket_width_sec, bucket_count)
                    buffers[b].append(sample)
                    in_flight += 1
                    if len(buffers[b]) >= batch_size:
                        full[b] = True

            if any(full):
                ready_buckets = [b for b in range(bucket_count) if full[b]]
                b = rng.choice(ready_buckets) if shuffle_buckets else ready_buckets[0]
                out = buffers[b][:batch_size]
                del buffers[b][:batch_size]
                in_flight -= len(out)
                full[b] = len(buffers[b]) >= batch_size
                yield PreBatched(out)
                continue

            if not exhausted:
                # Nothing has filled yet even after the lookahead above. This
                # cannot happen for a live stream (the lookahead only stops
                # early when a bucket filled), but guard against an infinite
                # loop rather than spinning forever if it ever does.
                continue

            # Stream is over. Emit whatever is left, longest bucket first, with
            # the short tail as the final batch. The training loader is fed an
            # infinite stream in normal operation, so these tail batches only
            # occur when the source itself was finite (tests, small data lists).
            remaining = [b for b in range(bucket_count) if buffers[b]]
            if not remaining:
                return
            biggest = max(remaining, key=lambda b: len(buffers[b]))
            out = buffers[biggest]
            buffers[biggest] = []
            full[biggest] = False
            in_flight -= len(out)
            yield PreBatched(out)

class Vox1TestEvalDatasetSimple(torch.utils.data.Dataset):
    def __init__(self,
                    vox1_test_root,
                    wavs_subroot='wav',
                    vox1_veri_test2_list=None,
                    samplerate = 16000,
                    duration = 8.0
                ):
        self.vox1_test_root = Path(vox1_test_root)
        self.wavs_subroot = wavs_subroot
        if vox1_veri_test2_list is None:
            vox1_veri_test2_list = self.vox1_test_root/'veri_test2.txt'

        vox1_protocol = pd.read_csv(vox1_veri_test2_list,sep=" ",
                                         names=['label','enroll','verify'])

        trials = vox1_protocol['enroll'].values.tolist() + vox1_protocol['verify'].values.tolist()
        self.trials = sorted(list(set(trials)))
        assert all([(self.vox1_test_root/self.wavs_subroot/rlp).exists() for rlp in self.trials])
        trials_ind_map = {key:ind for ind, key in enumerate(self.trials)}

        vox1_protocol['enroll_ind'] = vox1_protocol.enroll.map(trials_ind_map)
        vox1_protocol['verify_ind'] = vox1_protocol.verify.map(trials_ind_map)
        self.vox1_protocol = vox1_protocol

        self.samplerate = samplerate
        self.duration = duration

    def __len__(self):
        return len(self.trials)

    def __getitem__(self,index):
        path = self.vox1_test_root/self.wavs_subroot/self.trials[index]
        label = str(path.relative_to(self.vox1_test_root))

        samples, sr = read_audio_section(path,
            samplerate=self.samplerate,
            duration=self.duration,
            min_duration=0.0,
            start_time=0.0,
            extend_to_duration=self.duration,
        )
        x = torch.from_numpy(samples).float()[None,:]
        y = torch.tensor(index).long().unsqueeze(0)

        return x, y
