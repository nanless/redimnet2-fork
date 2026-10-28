"""给少量已保存原音及侧车画波形、频谱和原生/FIX/源钟时间。

只读显式清单中的原件；不调用模型、网络或GPU，不修改任何标注。
图是机器定位诊断，不是人工金标。真实音频、清单、图和逐字文本只写私有输出。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

MAX_AUDIO = 64 * 1024**2
MAX_JSON = 16 * 1024**2
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pinned(path, expected, maximum):
    """在读前后核文件身份及实际摘要，不跟随最终软链接。"""
    path = Path(path)
    need(path.is_absolute() and not path.is_symlink(), '输入须为绝对普通文件路径')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, '输入不是有界普通文件')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    try:
        need(stamp(before) == stamp(os.fstat(fd)), '输入在打开前改变')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        need(len(raw) == before.st_size and len(raw) <= maximum and
             stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()), '输入读取期间改变')
        need(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) and sha(raw) == expected,
             '输入SHA不符')
        return raw
    finally:
        os.close(fd)


def strict_json(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            need(k not in d, 'JSON重复字段')
            d[k] = v
        return d
    def finite(text):
        x = float(text)
        need(math.isfinite(x), 'JSON非有限数')
        return x
    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=lambda _: need(False, 'JSON非有限数'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def interval_rows(record):
    """保留原生和作者FIX各自时间；源钟单位单独一轨，不改零长或倒置。"""
    sr = record['source_identity']['sample_rate']
    need(type(sr) is int and sr > 0, '源采样率无效')
    rows = []
    for window in record['windows']:
        a = window.get('alignment', {})
        words = [x.get('text', '') for x in a.get('items', [])]
        offset = window['context_start_sample'] / sr
        for field, label in [('native_timestamps_ms', 'native'), ('fixed_timestamps_ms', 'FIX')]:
            times = a.get(field, [])
            need(isinstance(times, list) and len(times) % 2 == 0, '原生时间向量不是成对端点')
            for j in range(len(times) // 2):
                start, end = times[2*j:2*j+2]
                need(all(type(v) in (int, float) and math.isfinite(v) for v in (start, end)), '非有限时间')
                rows.append({'track': label, 'window_id': window['window_id'], 'native_index': j,
                             'text': words[j] if j < len(words) else '',
                             'start_seconds': offset + start / 1000,
                             'end_seconds': offset + end / 1000})
    for unit in record['units']:
        rows.append({'track': 'source units', 'window_id': unit['window_id'],
                     'native_index': unit['native_index'], 'text': unit.get('text', ''),
                     'start_seconds': unit['start_sample'] / sr, 'end_seconds': unit['end_sample'] / sr})
    for line in record.get('vad', {}).get('original_text', '').splitlines():
        if line.startswith('#') or not line.strip():
            continue
        fields = line.split()
        if fields[0] != 'SPEAKER' or len(fields) < 5:
            continue
        start, duration = float(fields[3]), float(fields[4])
        need(math.isfinite(start) and math.isfinite(duration), 'VAD非有限时间')
        rows.append({'track': 'old VAD', 'window_id': None, 'native_index': None,
                     'text': '', 'start_seconds': start, 'end_seconds': start + duration})
    return rows


def zoom_ranges(record, rows, duration):
    """首尾、窗口接缝与有限异常点；合并重复，不把1采样点差伪装成声学分辨率。"""
    sr = record['source_identity']['sample_rate']
    points = [(0., '开头'), (duration, '结尾')]
    for w in record['windows'][1:]:
        points.append((w['owned_start_sample'] / sr, '窗口接缝'))
    for e in record.get('errors', []):
        ref = e.get('left')
        if ref:
            for r in rows:
                if r['track'] == 'source units' and [r['window_id'], r['native_index']] == ref:
                    points.append((r['end_seconds'], '源钟重叠'))
        elif 'native_index' in e:
            for r in rows:
                if r['track'] == 'FIX' and r['window_id'] == e.get('window_id') and r['native_index'] == e['native_index']:
                    points.append((r['start_seconds'], '零长或异常'))
    # 首尾、每个接缝及最多8个异常附近；所有端点仍完整保留在JSON/整窗图。
    result = []
    for center, reason in points[:12]:
        a, b = max(0., center - .35), min(duration, center + .35)
        if b <= a or any(abs(center - x['center_seconds']) < .025 for x in result):
            continue
        result.append({'start_seconds': a, 'end_seconds': b, 'center_seconds': center, 'reason': reason})
    return result


def load_case(case, audio_root):
    """兼容私有选择清单；编码原音须与侧车记录的原件SHA/时钟一致。"""
    rid = case['review_id']
    need(isinstance(rid, str) and ID.fullmatch(rid), 'review_id无效')
    row = case['row']
    sidecar_raw = read_pinned(case['local_sidecar'], row['sidecar_sha256'], MAX_JSON)
    record = strict_json(sidecar_raw)
    suffix = Path(row['audio_path']).suffix.lower()
    need(suffix in ('.wav', '.flac'), '仅绘制WAV或FLAC原音')
    audio = Path(audio_root) / rid / ('source' + suffix)
    raw = read_pinned(audio, row['audio_recorded_file_sha256'], MAX_AUDIO)
    need(record['source']['source_id'] == row['source_id'] and
         record['source_identity']['file_sha256'] == row['audio_recorded_file_sha256'], '侧车与清单源身份不符')
    import numpy as np
    import soundfile as sf
    pcm, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    identity = record['source_identity']
    need(sr == identity['sample_rate'] and pcm.shape == (identity['frames'], identity['channels']), '实际音频钟/声道/帧数与侧车不符')
    need(0 < len(pcm) <= sr * 180 and np.isfinite(pcm).all(), '音频时长或PCM数值无效')
    pcm_sha = sha(pcm.astype('<f4', copy=False).tobytes(order='C'))
    need(identity.get('pcm_hash_policy') == 'decoded_float32le_native_interleaved/v1' and
         pcm_sha == identity['pcm_sha256'], '解码原采样率多声道PCM SHA与侧车不符')
    proof = {'encoded_sha256': sha(raw), 'pcm_sha256': pcm_sha, 'sample_rate': sr,
             'frames': len(pcm), 'channels': pcm.shape[1], 'gain_applied': False, 'resampled': False,
             'soundfile_version': sf.__version__, 'libsndfile_version': sf.__libsndfile_version__}
    # 绘图声道平均不作为生产模型输入，也不重写原PCM证明。
    return record, pcm.mean(axis=1), sr, audio, proof


def window_table(window, record):
    """字词、原生/FIX毫秒和最终源采样位置并列，不把声学图当时间真值。"""
    a = window.get('alignment', {}); native = a.get('native_timestamps_ms', [])
    fixed = a.get('fixed_timestamps_ms', []); items = a.get('items', [])
    units = {u['native_index']: u for u in record['units'] if u['window_id'] == window['window_id']}
    body = ['<table><tr><th>项号</th><th>字词</th><th>原生时间 ms（窗内）</th><th>作者FIX ms（窗内）</th><th>源采样点 start/end</th></tr>']
    for i in range(max(len(items), len(native)//2, len(fixed)//2)):
        word = items[i].get('text', '') if i < len(items) else ''
        n, f = native[2*i:2*i+2], fixed[2*i:2*i+2]; u = units.get(i)
        samples = [u['start_sample'], u['end_sample']] if u else '未进入owned units'
        marked = ' style="background:#fff1df"' if n != f or (len(f)==2 and f[1] <= f[0]) else ''
        body.append('<tr'+marked+'>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (i,word,n,f,samples))+'</tr>')
    body.append('</table>')
    return ''.join(body)


def make_figure(pcm, sr, record, rows, start, end, title, output):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    a, b = max(0, int(math.floor(start*sr))), min(len(pcm), int(math.ceil(end*sr)))
    y = pcm[a:b]
    need(len(y) >= 2, '绘图区间过短')
    fig, axes = plt.subplots(3, 1, figsize=(16, 7.6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.6, 1.7]}, layout='constrained')
    # 各分箱保存min/max，长音频也不丢掉短脉冲。
    edges = np.linspace(0, len(y), min(10000, len(y)) + 1, dtype=int)
    centers = (edges[:-1] + edges[1:]) / (2*sr) + a/sr
    lo = np.array([y[l:r].min() for l, r in zip(edges[:-1], edges[1:])])
    hi = np.array([y[l:r].max() for l, r in zip(edges[:-1], edges[1:])])
    axes[0].fill_between(centers, lo, hi, color='#305b87', linewidth=.3)
    axes[0].set_ylabel('Amplitude')
    n = min(len(y), max(16, int(sr*.025)))
    _, times, z = stft(y, sr, nperseg=n, noverlap=min(n-1, int(n*.8)), boundary=None)
    freq = np.fft.rfftfreq(n, 1/sr)
    db = 20*np.log10(np.maximum(np.abs(z), 1e-6))
    axes[1].pcolormesh(times+a/sr, freq, db, shading='auto', cmap='magma', vmin=-90, vmax=-10, rasterized=True)
    axes[1].set_ylim(0, min(8000, sr/2)); axes[1].set_ylabel('Hz')
    tracks = ['old VAD', 'native', 'FIX', 'source units']; colors = ['#999999', '#3568aa', '#dd7a21', '#268176']
    shown = [r for r in rows if max(r['start_seconds'], r['end_seconds']) >= start and min(r['start_seconds'], r['end_seconds']) <= end]
    label = len(shown) <= 100
    for r in shown:
        j = tracks.index(r['track']); left, right = r['start_seconds'], r['end_seconds']
        if right > left:
            axes[2].plot([left, right], [j, j], color=colors[j], lw=4, solid_capstyle='butt')
            axes[2].plot([left, right], [j, j], '|', color=colors[j], ms=10)
        else:
            axes[2].plot([left], [j], 'x', color='#b51b28', ms=9)
        if label and r['text']:
            axes[2].text((left+right)/2, j+.12, r['text'][:18], ha='center', va='bottom', fontsize=7, clip_on=True)
    for w in record['windows']:
        for k in ('owned_start_sample', 'owned_end_sample'):
            x = w[k]/sr
            if start < x < end:
                for ax in axes:
                    ax.axvline(x, ls='--', color='#1b9a8e', lw=.8, alpha=.8)
    for ax in axes:
        ax.axvline(len(pcm)/sr, color='#b51b28', ls=':', lw=1)
        ax.grid(axis='x', alpha=.15)
    axes[2].set_yticks(range(4), tracks); axes[2].set_ylim(-.4, 3.6)
    # 源尾外留0.001秒，能显示VAD尾端超界；不补音频。
    axes[2].set_xlim(start, end + (.001 if end == len(pcm)/sr else 0))
    axes[2].set_xlabel('Source time (seconds); x = zero/reversed interval; dashed = owned-window edge')
    fig.suptitle(title + '\nMachine observations only; not human gold or label release', fontsize=12)
    fig.savefig(output, dpi=150); plt.close(fig)


def render(selection, selection_sha, audio_root, output):
    execution_source_sha = sha(Path(__file__).read_bytes())
    selected = strict_json(read_pinned(selection, selection_sha, MAX_JSON))
    need(isinstance(selected, list) and 1 <= len(selected) <= 32, '只接受1—32份显式选择')
    ids = [x['review_id'] for x in selected]
    need(len(ids) == len(set(ids)), 'review_id重复')
    output = Path(output)
    need(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(), '输出须为不存在的新绝对目录')
    # 先核全部原件再建输出目录，任何缺失/摘要不符不生成看似完整的报告。
    cases = [(c, *load_case(c, audio_root)) for c in selected]
    output.mkdir(mode=0o700)
    summaries = []
    with tempfile.TemporaryDirectory(prefix='saved-boundary-mpl-') as cache:
        old_cache = os.environ.get('MPLCONFIGDIR'); os.environ.setdefault('MPLCONFIGDIR', cache)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            installed = {f.name for f in font_manager.fontManager.ttflist}
            font = next((f for f in ['Songti SC', 'PingFang SC', 'Heiti TC', 'Noto Sans CJK SC'] if f in installed), 'DejaVu Sans')
            plt.rcParams.update({'font.family': [font, 'DejaVu Sans'], 'axes.unicode_minus': False})
            for c, record, pcm, sr, audio, decode_proof in cases:
                rid = c['review_id']; folder = output/rid; folder.mkdir()
                rows = interval_rows(record); duration = len(pcm)/sr
                title = f'{rid} | {record["source"]["dataset"]} | {record["decision"]["state"]} | {duration:.6f}s'
                figures = ['whole.png']; make_figure(pcm, sr, record, rows, 0, duration, title, folder/figures[-1])
                for i, w in enumerate(record['windows']):
                    name = f'window-{i:02d}.png'; figures.append(name)
                    make_figure(pcm, sr, record, rows, w['context_start_sample']/sr, w['context_end_sample']/sr, title+f' | window {i}', folder/name)
                zooms = zoom_ranges(record, rows, duration)
                for i, z in enumerate(zooms):
                    name=f'zoom-{i:02d}.png'; figures.append(name)
                    make_figure(pcm, sr, record, rows, z['start_seconds'], z['end_seconds'], title+' | '+z['reason'], folder/name)
                value = {'review_id':rid,'sample_id':c['sample_id'],'audio_path':str(audio),
                         'audio_sha256':c['row']['audio_recorded_file_sha256'],'sidecar_sha256':c['row']['sidecar_sha256'],
                         'human_gold':False,'label_eligibility_released':False,'models_called':False,
                         'source_identity':record['source_identity'],'local_decode_proof':decode_proof,'decision':record['decision'],
                         'windows':record['windows'],'interval_rows':rows,'errors':record['errors'],'zooms':zooms,'figures':figures}
                (folder/'observations.json').write_bytes(json_bytes(value))
                body=[f'<h1>{html.escape(title)}</h1><p>只展示原始机器结果与声学图，不是人工金标。x表示零长或倒置，虚线为该窗口负责的源范围边界。频谱时间分辨率约25毫秒，不能据图确认1采样点差的听觉真值。</p>', '<p>'+html.escape('; '.join(record['decision']['reasons']))+'</p>']
                for w in record['windows']:
                    asr=w['asr'];body.append('<h2>'+html.escape(w['window_id'])+'</h2><p>ASR原文：'+html.escape(asr.get('raw_text',''))+'</p><p>标准化：'+html.escape(asr.get('normalized_text',''))+'</p>'+window_table(w, record))
                for name in figures: body.append(f'<h2>{name}</h2><img loading="lazy" src="{name}">')
                (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:system-ui;max-width:1500px;margin:24px auto;padding:20px}img{width:100%}p{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:5px;text-align:left}</style>'+''.join(body))
                summaries.append({'review_id':rid,'sample_id':c['sample_id'],'decision':record['decision']['state'],
                                  'figures':len(figures),'duration_seconds':duration,'local_decode_proof':decode_proof,'path':rid+'/index.html'})
        finally:
            if old_cache is None: os.environ.pop('MPLCONFIGDIR',None)
    links=''.join(f'<li><a href="{x["path"]}">{x["review_id"]} / {x["sample_id"]} / {x["decision"]} / {x["duration_seconds"]:.6f}s</a></li>' for x in summaries)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>有限原音与字词边界诊断</h1><p>原始机器结果，不是人工金标。保留unknown和全部原生/FIX/源钟位置。</p><ul>'+links+'</ul>')
    receipt={'schema':'saved-boundary-review-output/v1','selection_sha256':selection_sha,
             'source_sha256':execution_source_sha,'cases':summaries,'human_gold':False,
             'label_eligibility_released':False,'model_GPU_API_execution':False,
             'outputs_sha256':{str(p.relative_to(output)):sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}}
    need(sha(Path(__file__).read_bytes()) == execution_source_sha, '绘图期间维护源码改变，输出仅为未完成尝试')
    (output/'receipt.json').write_bytes(json_bytes(receipt))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha',required=True)
    p.add_argument('--audio-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=render(a.selection,a.selection_sha,a.audio_root,a.output)
    print(json.dumps({'case_count':len(r['cases']),'figure_count':sum(x['figures'] for x in r['cases']),
                      'human_gold':False,'output':str(a.output)},ensure_ascii=False))


if __name__=='__main__': main()
