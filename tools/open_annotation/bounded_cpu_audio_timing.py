"""有限原音的只读CPU／共享盘时延测量，不导入模型或使用GPU。

每原件最多两次只读打开：先读编码原件并在内存解码核PCM，再重读解码。
第二次解码结果在内存切出原context窗口，保留重采样及PCM摘要耗时。
这不是生产八卡阶段profile，不保证冷缓存，也不写原件、侧车或数据库。
"""
from __future__ import annotations
import argparse
import base64
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
import traceback

MAX_SOURCES = 24
MAX_BYTES = 256 * 1024**2
MAX_PCM_BYTES = 64 * 1024**2
HEX = re.compile('[0-9a-f]{64}')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def measured(function):
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    return value, {'wall_seconds': time.perf_counter()-wall, 'process_cpu_seconds': time.process_time()-cpu}


def read_encoded(path, maximum):
    """第一次只读打开；只接受稳定普通文件，读量受集合剩余预算限制。"""
    initial = path.stat()
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'bounded regular source required')
        need(stamp(initial) == stamp(before), 'source changed before encoded open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(maximum+1)
        need(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(fd)) == stamp(path.stat()),
             'source changed during encoded read')
        return raw, stamp(before)
    finally:
        os.close(fd)


def validate_input(value):
    need(isinstance(value, dict) and value.get('schema') == 'bounded-cpu-audio-timing-input/v1', 'explicit input schema required')
    sources = value.get('sources')
    need(isinstance(sources, list) and 1 <= len(sources) <= MAX_SOURCES, 'select 1..24 sources')
    need(len({r['source_id'] for r in sources}) == len(sources), 'duplicate source')
    total = 0
    for row in sources:
        need(Path(row['audio_path']).is_absolute(), 'absolute selected audio path required')
        need(all(isinstance(row.get(k), str) and HEX.fullmatch(row[k]) for k in ('file_sha256','pcm_sha256')), 'frozen source SHA required')
        for key in ('bytes','frames','channels','sample_rate'):
            need(type(row.get(key)) is int and row[key] > 0, 'positive integer source geometry required')
        need(row['frames']*row['channels']*4 <= MAX_PCM_BYTES, 'source PCM exceeds bounded RAM')
        total += row['bytes']
        windows = row.get('windows')
        need(isinstance(windows,list) and 1 <= len(windows) <= 8, 'bounded explicit context windows required')
        for window in windows:
            a,b=window['context_start_sample'],window['context_end_sample']
            need(type(a)is int and type(b)is int and 0<=a<b<=row['frames'], 'invalid context sample range')
            need((b-a)/row['sample_rate']<=30.0001 and HEX.fullmatch(window['input_pcm_sha256']), 'bounded model window SHA required')
    need(total <= MAX_BYTES, 'aggregate encoded source budget exceeded')
    return sources


def measure_sources(value, expected_hostname):
    sources = validate_input(value)
    need(socket.gethostname() == expected_hostname, 'actual CPU observation host differs')
    need(not any(n == 'torch' or n.startswith(('torch.','qwen_asr','vllm')) for n in sys.modules), 'heavy model module already imported')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    rows=[]
    for source in sources:
        row={'source_id':source['source_id'],'audio_path':source['audio_path'],'source_file_opens':0,'source_seconds':source['frames']/source['sample_rate'],'windows':[]}
        path=Path(source['audio_path'])
        try:
            row['source_file_opens']+=1
            (encoded, original_stamp), row['encoded_read']=measured(lambda:read_encoded(path,source['bytes']))
            digest,row['encoded_hash']=measured(lambda:sha(encoded))
            need(digest==source['file_sha256'] and len(encoded)==source['bytes'],'encoded original identity differs')
            def decode_first():
                with sf.SoundFile(io.BytesIO(encoded)) as stream:
                    need((stream.samplerate,stream.channels,len(stream))==(source['sample_rate'],source['channels'],source['frames']),'source header differs')
                    data=stream.read(dtype='float32',always_2d=True)
                need(np.isfinite(data).all(),'nonfinite native PCM')
                return sha(data.astype('<f4',copy=False).tobytes())
            first_pcm,row['memory_decode_and_pcm_hash']=measured(decode_first)
            need(first_pcm==source['pcm_sha256'],'first native PCM identity differs')
            row['source_file_opens']+=1
            def decode_second():
                fd=os.open(path,os.O_RDONLY|getattr(os,'O_NONBLOCK',0))
                try:
                    need(stamp(os.fstat(fd))==original_stamp,'source changed before second read')
                    with sf.SoundFile(fd,closefd=False)as stream:
                        need((stream.samplerate,stream.channels,len(stream))==(source['sample_rate'],source['channels'],source['frames']),'second source header differs')
                        data=stream.read(dtype='float32',always_2d=True)
                    need(stamp(os.fstat(fd))==original_stamp==stamp(path.stat()),'source changed during second read')
                    return data
                finally:os.close(fd)
            data,row['second_file_decode']=measured(decode_second)
            second_pcm,row['second_pcm_hash']=measured(lambda:sha(data.astype('<f4',copy=False).tobytes()))
            need(np.isfinite(data).all() and second_pcm==source['pcm_sha256'],'second native PCM identity differs')
            for window in source['windows']:
                a,b=window['context_start_sample'],window['context_end_sample']
                def prepare_window():
                    mono=data[a:b].mean(axis=1,dtype=np.float32)
                    if source['sample_rate']!=16000:
                        divisor=math.gcd(source['sample_rate'],16000)
                        mono=resample_poly(mono,16000//divisor,source['sample_rate']//divisor).astype(np.float32)
                    return len(mono),sha(mono.astype('<f4',copy=False).tobytes())
                (frames,digest),timing=measured(prepare_window)
                need(digest==window['input_pcm_sha256'],'context model PCM differs')
                row['windows'].append(dict(window_id=window['window_id'],model_frames=frames,input_pcm_sha256=digest,preparation=timing))
            row.update(status='identity_verified',file_sha256=source['file_sha256'],pcm_sha256=first_pcm,stat_observed=list(original_stamp))
        except Exception:
            row.update(status='failed_preserved',traceback=traceback.format_exc())
        rows.append(row)
    return dict(schema='bounded-cpu-audio-timing-result/v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),sources=rows,passed=all(r['status']=='identity_verified'for r in rows),source_count=len(rows),physical_source_opens=sum(r['source_file_opens']for r in rows),soundfile_version=sf.__version__,libsndfile_version=sf.__libsndfile_version__,numpy_version=np.__version__,GPU_context_created=False,model_modules_imported=False,source_or_sidecar_writes=0,databases_created=0,cold_cache_claim=False,actual_cloud_stage_profile=False,meaning='CPU-only bounded sample; first encoded read plus memory decode/hash, second full file decode, in-memory exact original context preparation; page-cache and development-node effects retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-base64',required=True)
    parser.add_argument('--inputs-sha',required=True)
    parser.add_argument('--expected-hostname',required=True)
    args=parser.parse_args()
    raw=base64.b64decode(args.inputs_base64,validate=True)
    need(len(raw)<=1024**2 and sha(raw)==args.inputs_sha,'explicit bounded input SHA differs')
    value=json.loads(raw)
    try:os.nice(15)
    except (AttributeError,PermissionError):pass
    result=measure_sources(value,args.expected_hostname)
    result['inputs_sha256']=sha(raw)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    raise SystemExit(0 if result['passed']else 2)

if __name__=='__main__':
    main()
