"""Strict acoustic-VAD records and deterministic, unpadded speech-time crops."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
from functools import lru_cache
import numpy as np

SCHEMA = 'duration-vad-dataset/v1'
SIGNATURE = 'a56629f741671964f03e3b7ba3f2087d0b3ed76622b9d527e68b501eff325810'

class DataError(ValueError):
    def __init__(self, reason, detail=''):
        self.reason = reason
        super().__init__(f'{reason}: {detail}')

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_output_path(path,source_root=None):
    root=Path(path).resolve()
    forbidden=[Path('/root/code/github_repos/redimnet2-fork'),
               Path('/root/experiments/redimnet2/b6-mix-official-20261019'),
               Path('/root/group-shared/voiceprint/data'),Path('/shared/group-shared/voiceprint/data')]
    if source_root is not None:forbidden.append(Path(source_root).resolve())
    for protected in forbidden:
        if root==protected or protected in root.parents or root in protected.parents:
            raise ValueError('protected output path')
    return root

def source_stat(path):
    st = Path(path).stat()
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}

def annotation_paths(audio, root, depth=2):
    p = Path(root).joinpath(*Path(audio).parts[-(depth + 1):])
    p = p.with_name(p.name + '.rttm')
    return p, p.with_name('.' + p.name + '.json')

def inspect_record(obj, rttm_root, signature=SIGNATURE, minimum_seconds=5.0):
    """Read metadata only; every accepted segment and origin must be auditable."""
    for key in ('key', 'spk', 'wav'):
        if not isinstance(obj.get(key), str) or not obj[key]:
            raise DataError('bad_record', key)
    audio = Path(obj['wav'])
    try:
        stat = source_stat(audio)
    except OSError as exc:
        raise DataError('missing_audio', str(exc)) from exc
    if not stat['bytes']:
        raise DataError('empty_audio', str(audio))
    annotation, sidecar = annotation_paths(audio, rttm_root)
    if not annotation.is_file():
        raise DataError('missing_rttm', str(annotation))
    if not sidecar.is_file():
        raise DataError('missing_sidecar', str(sidecar))
    try:
        raw = annotation.read_text(); meta = json.loads(sidecar.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError('bad_annotation', str(exc)) from exc
    headers = {}; rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith('#'):
            entry = line[1:].strip()
            if '=' in entry:
                key, value = entry.split('=', 1)
                if key in headers:
                    raise DataError('duplicate_header', key)
                headers[key] = value
            continue
        parts = line.split()
        if len(parts) != 10 or parts[0] != 'SPEAKER' or parts[1] != audio.stem or parts[2] != '1':
            raise DataError('bad_rttm_row', line)
        try:
            start, duration = float(parts[3]), float(parts[4])
        except ValueError as exc:
            raise DataError('bad_rttm_time', line) from exc
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
            raise DataError('bad_rttm_time', line)
        if parts[5:] != ['<NA>'] * 5:
            raise DataError('bad_rttm_row', line)
        rows.append((start, start + duration))
    try:
        if meta['source_size'] != stat['bytes'] or meta['source_mtime_ns'] != stat['mtime_ns']:
            raise DataError('stale_annotation', str(audio))
        if headers['inference_signature'] != signature or meta['inference_signature'] != signature:
            raise DataError('signature_mismatch', str(audio))
        if int(headers['source_size']) != stat['bytes'] or int(headers['source_mtime_ns']) != stat['mtime_ns']:
            raise DataError('stale_annotation', str(audio))
        rate = int(headers['source_sample_rate']); frames = int(headers['source_frames'])
        if rate != 16000 or int(headers['source_channels']) != 1 or int(headers['sample_rate']) != 16000:
            raise DataError('unsupported_audio_format', str(audio))
        if meta['source_sample_rate'] != rate or meta['source_num_channels'] != 1:
            raise DataError('sidecar_mismatch', str(audio))
        if frames <= 0 or int(headers['samples_16k']) != frames or meta['sample_length'] != frames:
            raise DataError('sidecar_mismatch', str(audio))
        if int(headers['segments']) != len(rows) or meta['segments'] != len(rows):
            raise DataError('sidecar_mismatch', str(audio))
        if meta['schema_version'] != 2 or int(headers['schema_version']) != 2:
            raise DataError('schema_mismatch', str(audio))
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError('bad_annotation_metadata', str(exc)) from exc
    if not rows:
        raise DataError('empty_vad', str(audio))
    segments = []; previous_end = 0
    for start, end in rows:
        # RTTM is rounded to milliseconds: allow at most one millisecond at EOF.
        if start < previous_end - 1e-6 or end > frames / rate + 0.001:
            raise DataError('overlap_or_out_of_bounds', str(audio))
        a, b = round(start * rate), min(frames, round(end * rate))
        if b <= a:
            raise DataError('bad_rttm_time', str(audio))
        segments.append([a, b]); previous_end = end
    total = sum(b-a for a,b in segments)
    if total < round(minimum_seconds * rate):
        raise DataError('insufficient_speech', str(audio))
    return {'key': obj['key'], 'speaker': obj['spk'], 'corpus': obj['key'].split('_',1)[0],
            'audio': str(audio), 'audio_realpath': str(audio.resolve()), 'source_stat': stat,
            'sample_rate': rate, 'source_frames': frames, 'segments': segments,
            'speech_samples': total, 'rttm': str(annotation), 'rttm_sha256': file_sha(annotation),
            'sidecar': str(sidecar), 'sidecar_sha256': file_sha(sidecar), 'signature': signature}

def verify_record(record, check_audio_header=True):
    if source_stat(record['audio']) != record['source_stat'] or str(Path(record['audio']).resolve()) != record['audio_realpath']:
        raise DataError('source_changed', record['key'])
    for field in ('rttm', 'sidecar'):
        if file_sha(record[field]) != record[field + '_sha256']:
            raise DataError('annotation_changed', record['key'])
    if check_audio_header:
        import soundfile as sf
        info = sf.info(record['audio'])
        if info.samplerate != 16000 or info.channels != 1 or info.frames != record['source_frames']:
            raise DataError('audio_header_mismatch', record['key'])

def seal_dataset(records, speaker_map, selection):
    body = {'schema': SCHEMA, 'records': records, 'speaker_map': speaker_map, 'selection': selection}
    return dict(body, dataset_id=digest(body))

def load_dataset(path, verify=True):
    obj = json.loads(Path(path).read_text())
    body = {k:v for k,v in obj.items() if k != 'dataset_id'}
    if obj.get('schema') != SCHEMA or obj.get('dataset_id') != digest(body):
        raise DataError('dataset_identity_mismatch', str(path))
    records = obj['records']; speakers = obj['speaker_map']
    if not records or len({r['key'] for r in records}) != len(records):
        raise DataError('duplicate_or_empty_records', str(path))
    if sorted(speakers.values()) != list(range(len(speakers))):
        raise DataError('bad_speaker_map', str(path))
    for record in records:
        segments=record.get('segments',[])
        previous=0
        for a,b in segments:
            if not isinstance(a,int) or not isinstance(b,int) or not 0<=a<b<=record['source_frames'] or a<previous:
                raise DataError('bad_segment_geometry',record['key'])
            previous=b
        if not segments or sum(b-a for a,b in segments)!=record['speech_samples'] or record.get('signature')!=SIGNATURE:
            raise DataError('bad_segment_identity',record['key'])
        if record['speaker'] not in speakers or record['speech_samples'] < 80000:
            raise DataError('ineligible_record', record['key'])
        if verify:
            verify_record(record)
    return obj

def deterministic_seed(*parts):
    return int(hashlib.sha256(canonical(parts).encode()).hexdigest()[:16],16)

def length_for_step(step, seed, lengths, probabilities):
    if type(step) is not int or step < 0 or type(seed) is not int or seed < 0:
        raise ValueError('step and seed must be nonnegative integers')
    if not isinstance(lengths,(list,tuple)) or not isinstance(probabilities,(list,tuple)) or not lengths or len(lengths) != len(probabilities):
        raise ValueError('invalid length plan')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in list(lengths) + list(probabilities)):
        raise ValueError('lengths/probabilities must be finite positive')
    if not math.isclose(sum(probabilities),1.0,abs_tol=1e-12):
        raise ValueError('probabilities must sum to one')
    return random.Random(deterministic_seed(seed, 'length', step)).choices(lengths, probabilities)[0]

@lru_cache(maxsize=2)
def _epoch_order(count,seed,epoch):
    # At most two int64 permutations per process, not a new Python-int list each step.
    order=np.random.default_rng(deterministic_seed(seed,'order',epoch)).permutation(count)
    order.setflags(write=False)
    return order

def balanced_batch_indices(count, step, per_rank_batch, world_size, seed, permutation=None):
    """No dropped tails and no duplicate source within a global optimizer step.

    A permutation is shared until lcm(pool, global batch), where batch and
    epoch boundaries coincide. New source identity refuses older resumes.
    """
    if any(type(v) is not int for v in (count,step,per_rank_batch,world_size,seed)) or min(count,per_rank_batch,world_size)<1 or min(step,seed)<0:
        raise ValueError('invalid sampler arguments')
    size=per_rank_batch*world_size
    if count<size:raise ValueError('pool must cover a complete distinct global step')
    cycle_size=math.lcm(count,size);get_order=permutation or _epoch_order;output=[]
    for position in range(step*size,(step+1)*size):
        cycle,offset=divmod(position,cycle_size)
        output.append(int(get_order(count,seed,cycle)[offset%count]))
    if len(set(output))!=size:raise AssertionError('global batch uniqueness violated')
    return output


def global_batch_indices(count, step, per_rank_batch, world_size, seed):
    return balanced_batch_indices(count,step,per_rank_batch,world_size,seed)

def speech_crop_plan(record, seconds, seed, exposure_index, budget_seconds=5.0):
    rate=record['sample_rate']; n=round(seconds*rate); budget=round(budget_seconds*rate)
    total=record['speech_samples']
    if n<=0 or n>budget or total<budget:
        raise DataError('insufficient_speech', record['key'])
    # M1 and M2 share the same five-second parent; shorter requests nest at center.
    rng=random.Random(deterministic_seed(seed,'crop',exposure_index,record['key']))
    origin=rng.randrange(total-budget+1)+(budget-n)//2
    stop=origin+n; cursor=0; intervals=[]
    for a,b in record['segments']:
        size=b-a; lo=max(origin,cursor);hi=min(stop,cursor+size)
        if hi>lo:intervals.append([a+lo-cursor,a+hi-cursor])
        cursor+=size
        if cursor>=stop:break
    if sum(b-a for a,b in intervals)!=n:
        raise DataError('crop_plan_length',record['key'])
    return {'key':record['key'],'sample_rate':rate,'seconds':seconds,'samples':n,
            'speech_axis_start':origin,'intervals':intervals,
            'splice_offsets':_splice_offsets(intervals)}

def _splice_offsets(intervals):
    total=0; out=[]
    for a,b in intervals[:-1]:total+=b-a;out.append(total)
    return out

def read_speech_crop(record, plan):
    import numpy as np
    import soundfile as sf
    verify_record(record,check_audio_header=False)
    chunks=[]
    try:
        with sf.SoundFile(record['audio']) as stream:
            if stream.samplerate!=16000 or stream.channels!=1 or stream.frames!=record['source_frames']:
                raise DataError('audio_header_mismatch',record['key'])
            for a,b in plan['intervals']:
                stream.seek(a); piece=stream.read(b-a,dtype='float32',always_2d=False)
                if len(piece)!=b-a or not np.isfinite(piece).all():
                    raise DataError('invalid_audio_read',record['key'])
                chunks.append(piece)
    except Exception as exc:
        if isinstance(exc,DataError):raise
        raise DataError('decode_failed',f"{record['key']} plan={canonical(plan)} {type(exc).__name__}: {exc}") from exc
    audio=np.concatenate(chunks)
    if len(audio)!=plan['samples']:raise DataError('crop_length_mismatch',record['key'])
    return audio
