"""CPU-only, explicitly selected, source-clock diagnostic figures; never gold.

Inputs and source bytes are read only. Output must be a fresh disjoint tree.
No model, CUDA, API, normalisation, resampling, gain or interval repair occurs.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path, PurePosixPath
import shutil
import time
import traceback


def sha_file(path, mib_per_second=0):
    h = hashlib.sha256()
    start=time.monotonic();count=0
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block);count+=len(block)
            throttle(start,count,mib_per_second)
    return h.hexdigest()


def throttle(start, byte_count, mib_per_second):
    if mib_per_second:
        delay=byte_count/(mib_per_second*1024**2)-(time.monotonic()-start)
        if delay>0:time.sleep(delay)


def copy_source_bounded(source,target,mib_per_second):
    start=time.monotonic();count=0
    with Path(source).open('rb') as src,Path(target).open('xb') as dst:
        for chunk in iter(lambda:src.read(1024**2),b''):
            dst.write(chunk);count+=len(chunk);throttle(start,count,mib_per_second)


def stat(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def write_new(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(data, bytes): data = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with p.open('xb') as f: f.write(data)
    return hashlib.sha256(data).hexdigest()


def contained(root, rel):
    p = PurePosixPath(rel)
    if p.is_absolute() or str(p) != rel or any(x in ('', '.', '..') for x in p.parts): raise ValueError('noncanonical sidecar path')
    target = root.joinpath(*p.parts)
    if root.resolve() not in target.resolve().parents or target.is_symlink(): raise ValueError('sidecar escaped root or symlink')
    return target


def map_audio(path, mappings):
    matches = [(old, new) for old, new in mappings if path.startswith(old.rstrip('/') + '/')]
    if len(matches) > 1: raise ValueError('ambiguous audio-root mappings')
    return Path(matches[0][1].rstrip('/') + path[len(matches[0][0].rstrip('/')):]) if matches else Path(path)


def decode_verified(path, identity, max_pcm_bytes, mib_per_second=16):
    import numpy as np
    import soundfile as sf
    before = stat(path)
    encoded = sha_file(path,mib_per_second)
    if encoded != identity['file_sha256']: raise ValueError('source encoded SHA differs from sidecar')
    chunks = []; pcm = hashlib.sha256(); frames_read = 0; decode_started=time.monotonic()
    with sf.SoundFile(path) as f:
        if (f.samplerate, f.channels, len(f)) != (identity['sample_rate'], identity['channels'], identity['frames']):
            raise ValueError('source clock/header differs')
        if len(f) * f.channels * 4 > max_pcm_bytes: raise ValueError('source exceeds explicit RAM budget; no hidden crop')
        while True:
            x = f.read(65536, dtype='float32', always_2d=True)
            if not len(x): break
            if not bool(np.isfinite(x).all()): raise ValueError('source PCM nonfinite')
            pcm.update(x.astype('<f4', copy=False).tobytes(order='C')); chunks.append(x); frames_read += len(x)
            throttle(decode_started,frames_read*identity['channels']*4,mib_per_second)
    after = stat(path)
    if before != after or frames_read != identity['frames'] or pcm.hexdigest() != identity['pcm_sha256']:
        raise ValueError('source changed/PCM SHA or frames differ')
    data = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, identity['channels']), dtype='float32')
    return data, {'actual_path': str(path), 'path_mapping_is_readonly_view': True,
                  'encoded_sha256': encoded, 'pcm_sha256': pcm.hexdigest(), 'stat_before': before, 'stat_after': after,
                  'soundfile_version': sf.__version__, 'libsndfile_version': sf.__libsndfile_version__,
                  'PCM_policy': 'decoded_float32le_native_interleaved/v1', 'all_source_channels_decoded': True,
                  'gain_applied': False, 'resampled': False}


def envelope(x, sr, offset=0, max_columns=6000):
    """Min/max per exact native-frame bin: no decimation hiding peaks."""
    import numpy as np
    if len(x) == 0: return np.array([]), np.array([]), np.array([])
    edges = np.linspace(0, len(x), min(max_columns, len(x))+1, dtype='int64')
    return (np.array([(a+b)/(2*sr)+offset for a,b in zip(edges[:-1],edges[1:])]),
            np.array([x[a:b].min() for a,b in zip(edges[:-1],edges[1:])]),
            np.array([x[a:b].max() for a,b in zip(edges[:-1],edges[1:])]))


def spectrogram(x, sr, offset=0, max_columns=2000):
    """25ms Hann / 5ms hop native SR, absolute full-scale amplitude dB."""
    import numpy as np
    if not len(x): return None
    width = max(2, round(.025 * sr)); hop = max(1, round(.005 * sr))
    fft = 1 << (width - 1).bit_length()
    # Analysis-only zero padding for sources shorter than one STFT window.
    if len(x) < width: x = np.pad(x, (0, width-len(x)))
    total = 1 + (len(x)-width)//hop
    edges = np.linspace(0,total,min(total,max_columns)+1,dtype='int64')
    window = np.hanning(width); powers=[]; times=[]
    # Never materialise the full N*25ms frame matrix for long sources.
    for left,right in zip(edges[:-1],edges[1:]):
        summed = np.zeros(fft//2+1,dtype='float64')
        for first in range(int(left),int(right),128):
            indices=np.arange(first,min(first+128,int(right)))[:,None]*hop+np.arange(width)[None,:]
            amplitude=2*abs(np.fft.rfft(x[indices]*window,n=fft,axis=1))/window.sum()
            summed += (amplitude**2).sum(axis=0)
        powers.append(summed/(right-left))
        times.append(((left+right-1)*hop/2+width/2)/sr+offset)
    power=np.array(powers)
    return np.array(times), np.fft.rfftfreq(fft,1/sr), 10*np.log10(np.maximum(power,1e-14)).T



def interval_rows(record, window):
    sr = record['source_identity']['sample_rate']; offset = window['context_start_sample'] / sr
    alignment = window.get('alignment') or {}; items = alignment.get('items') or []
    rows = []
    for label, key in [('raw_argmax', 'native_timestamps_ms'), ('FIX_final', 'fixed_timestamps_ms')]:
        vector = alignment.get(key) or []
        for i in range(0, len(vector)-1, 2):
            a, b = vector[i:i+2]; text = items[i//2].get('text') if i//2 < len(items) else None
            finite = type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
            rows.append({'layer': label, 'native_index': i//2, 'text': text, 'raw_start_ms': a, 'raw_end_ms': b,
                         'source_start': offset + a/1000 if finite else None,
                         'source_end': offset + b/1000 if finite else None,
                         'positive': finite and b > a})
    for u in record.get('units', []):
        if u.get('window_id') == window.get('window_id'):
            rows.append({'layer': 'owned_unit', 'native_index': u.get('native_index'), 'text': u.get('text'),
                         'source_start': u['start_sample']/sr, 'source_end': u['end_sample']/sr,
                         'positive': u['end_sample'] > u['start_sample'], 'occurrence_index': u.get('occurrence_index')})
    return rows


def plot_panels(path, data, sr, record, window=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    channels = data.shape[1]
    if channels > 8: raise ValueError('more than8 channels requires explicit separate review; not silently dropping channels')
    start, end = (0,len(data)) if window is None else (window['context_start_sample'], window['context_end_sample'])
    view = data[start:end]; offset = start/sr; stop=end/sr
    fig, axes = plt.subplots(channels+2, 1, figsize=(16, 2.1*channels+5.4), sharex=True,
                             gridspec_kw={'height_ratios': [1]*channels+[1.7,1.8]}, constrained_layout=True)
    for c in range(channels):
        t,lo,hi=envelope(view[:,c],sr,offset); axes[c].fill_between(t,lo,hi,color='#2471a3',linewidth=.3)
        limit=max(1.0,float(np.abs(view[:,c]).max()) if len(view) else 1);axes[c].set_ylim(-limit,limit)
        axes[c].set_ylabel('native ch%d\nfull scale'%c);axes[c].grid(alpha=.2)
    spec=spectrogram(view[:,0],sr,offset)
    if spec:
        t,f,z=spec
        # STFT timestamps stay at native analysis-window centres. They are not
        # stretched to cover the complete source interval.
        if len(t)==1:
            edges=np.array([t[0]-.0025,t[0]+.0025]);freq_edges=np.r_[0,(f[:-1]+f[1:])/2,f[-1]]
            axes[-2].pcolormesh(edges,freq_edges,z,shading='flat',cmap='magma',vmin=-110,vmax=0)
        else:
            axes[-2].pcolormesh(t,f,z,shading='nearest',cmap='magma',vmin=-110,vmax=0)
    axes[-2].set_ylabel('native ch0 Hz\nabsolute dBFS [-110,0]')
    ax=axes[-1]
    if window is None:
        for i,w in enumerate(record.get('windows',[])):
            a,b=w['context_start_sample']/sr,w['context_end_sample']/sr
            oa,ob=w['owned_start_sample']/sr,w['owned_end_sample']/sr
            ax.plot([a,b],[i%4+1]*2,color='gray',lw=8,alpha=.3)
            ax.plot([oa,ob],[i%4+1]*2,color='#188977',lw=3)
        ax.set_ylabel('context gray\nowned green')
    else:
        colors={'raw_argmax':'#c0392b','FIX_final':'#7d3c98','owned_unit':'#148f77'}
        levels={'raw_argmax':3,'FIX_final':2,'owned_unit':1}
        rows = interval_rows(record,window)
        for row in rows:
            a,b=row['source_start'],row['source_end'];y=levels[row['layer']]+((row['native_index'] or 0)%3)*.12
            if a is None or b is None: continue
            if row['positive']:
                ax.plot([a,b],[y,y],color=colors[row['layer']],lw=2)
            else: ax.plot([a,b],[y,y],color='red',marker='x',lw=.5)
            if max(a,b)<offset or min(a,b)>stop:
                edge=offset if max(a,b)<offset else stop
                ax.text(edge,y,' outside %.3f/%.3f'%(a,b),fontsize=5,color='red',ha='left' if edge==offset else 'right')
            elif len(rows)<180:
                ax.text(max(offset,a),y+.02,str(row['text']),fontsize=5,clip_on=True)
        for sample,color,label in [(window['owned_start_sample'],'#229954','owned start'),(window['owned_end_sample'],'#229954','owned end')]:
            for item in axes: item.axvline(sample/sr,color=color,lw=.8,ls='--')
        ax.set_yticks([1,2,3],['owned units','author FIX','raw argmax']);ax.set_ylim(.7,3.7)
    ax.set_xlim(offset,max(stop,offset+1/sr));ax.set_xlabel('Original source seconds; no concatenation/resample/gain or time repair')
    fig.suptitle(record['source']['source_id']+' | '+('source overview' if window is None else str(window['window_id']))+' | machine diagnostics, human review pending',fontsize=9)
    fig.savefig(path,dpi=130);plt.close(fig)


def run(args):
    started=time.time()
    try: os.nice(5)
    except (AttributeError,PermissionError): pass
    labels=Path(args.labels_root).resolve();out=Path(args.output).absolute()
    if out.exists() or out.is_symlink(): raise ValueError('output must not exist')
    resolved=out.resolve()
    if labels==resolved or labels in resolved.parents or resolved in labels.parents: raise ValueError('output overlaps labels')
    selection_path=Path(args.selection);selection=json.loads(selection_path.read_bytes())
    selected=selection['sources']
    if not 0 < len(selected) <= args.max_sources: raise ValueError('selection exceeds explicit max_sources or empty')
    if len({x['sidecar_relpath'] for x in selected})!=len(selected): raise ValueError('duplicate selected sidecar')
    mappings=[]
    for raw in args.audio_root_map:
        old,new=raw.split('=',1)
        if not Path(old).is_absolute() or not Path(new).is_absolute(): raise ValueError('audio mappings must be absolute')
        mappings.append((old,new))
    out.mkdir(parents=True);results=[]
    for selected_row in selected:
        row=dict(selected_row)
        if not re.fullmatch(r'src_[0-9a-f]{64}',row['source_id']): raise ValueError('noncanonical source ID')
        directory=out/row['source_id'];directory.mkdir()
        try:
            p=contained(labels,row['sidecar_relpath']);raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sidecar_sha256']: raise ValueError('selected sidecar SHA changed')
            r=json.loads(raw)
            if r['source']['source_id']!=row['source_id']: raise ValueError('source ID differs')
            write_new(directory/'sidecar_original.json',raw)
            source=map_audio(r['source']['audio_path'],mappings)
            data,proof=decode_verified(source,r['source_identity'],args.max_pcm_mib*1024**2,args.read_mib_per_second)
            row['source_proof']=proof
            row['source_proof']['registered_audio_path']=r['source']['audio_path']
            if args.copy_audio:
                target=directory/('original'+source.suffix);copy_source_bounded(source,target,args.read_mib_per_second)
                if sha_file(target)!=proof['encoded_sha256'] or stat(source)!=proof['stat_after']:raise ValueError('copied audio or stable source check differs')
                row['audio_copy']=target.name
            plot_panels(directory/'overview.png',data,r['source_identity']['sample_rate'],r)
            windows=r.get('windows',[]);shown=windows[:args.max_windows_per_source];details=[]
            for i,w in enumerate(shown):
                name='window-%04d.png'%i
                plot_panels(directory/name,data,r['source_identity']['sample_rate'],r,w)
                details.append({'window_id':w['window_id'],'figure':name,'interval_rows':interval_rows(r,w),
                                'ASR':w.get('asr'),'alignment':w.get('alignment'),
                                'context_start_sample':w['context_start_sample'],'context_end_sample':w['context_end_sample'],
                                'owned_start_sample':w['owned_start_sample'],'owned_end_sample':w['owned_end_sample']})
            row.update(status='rendered',source_frames=len(data),source_channels=data.shape[1],
                       windows_total=len(windows),windows_detailed=len(shown),omitted_detail_window_ids=[w['window_id'] for w in windows[len(shown):]])
            write_new(directory/'details.json',details)
            body=['<h1>'+html.escape(row['source_id'])+'</h1>', '<p>Machine evidence only. Original gain/source clock. Human complete-word and text review remain pending.</p>',
                  '<img src="overview.png" style="width:100%">']
            if row.get('audio_copy'):body.append('<audio controls src="'+html.escape(row['audio_copy'])+'"></audio>')
            else:body.append('<p>Audio copy not requested; original is read-only at '+html.escape(str(source))+'</p>')
            for d in details:
                body+=['<h2>'+html.escape(str(d['window_id']))+'</h2>','<img style="width:100%" src="'+d['figure']+'">',
                       '<p>raw argmax and author FIX are separate; red nonpositive marks remain. Source units keep midpoint ownership. Outside-view endpoints are listed without repairing them.</p>',
                       '<pre>'+html.escape(json.dumps(d,ensure_ascii=False,indent=2))+'</pre>']
            body.append('<p>Detailed windows omitted by explicit cap: '+html.escape(str(row['omitted_detail_window_ids']))+'</p>')
            write_new(directory/'index.html',('<!doctype html><meta charset="utf-8"><title>Sidecar diagnostics</title>'+''.join(body)).encode())
        except Exception:
            row.update(status='failed_review',traceback=traceback.format_exc())
            if not (directory/'index.html').exists():
                write_new(directory/'index.html',('<!doctype html><meta charset="utf-8"><h1>Review failed; source remains in denominator</h1><pre>'+html.escape(row['traceback'])+'</pre>').encode())
        results.append(row)
        write_new(directory/'review_receipt.json',row)
    files={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()}
    receipt={'schema':'ultra-sidecar-visual-review/v1','selection_sha256':sha_file(selection_path),'tool_sha256':sha_file(__file__),
             'labels_root':str(labels),'audio_root_map':mappings,'results':results,'files':files,
             'rendered':sum(x['status']=='rendered' for x in results),'failed':sum(x['status']!='rendered' for x in results),
             'human_review':None,'model_API_calls':0,'GPU_used':False,'maximum_native_PCM_MiB_per_source':args.max_pcm_mib,'read_MiB_per_second_limit':args.read_mib_per_second,
             'plot_policy':{'all_waveform_channels_up_to8':'native original gain minmax bins','spectrogram_channel':0,'window_ms':25,'hop_ms':5,
                            'spectrogram_time_column_aggregation':'mean linear power if >2000 columns; no model input change',
                            'short_source_STFT_only_padding':'zero pad to25ms only in analysis display','spectrogram_display_dBFS_limits':[-110,0],
                            'intervals':'raw/FIX/unit separate; no clipping or repair of data; original arrays retained in JSON'},
             'wall_seconds':time.time()-started}
    write_new(out/'review_manifest.json',receipt)
    links=['<li><a href="'+html.escape(r['source_id'])+'/index.html">'+html.escape(r['source_id'])+'</a>: '+r['status']+'</li>' for r in results]
    write_new(out/'index.html',('<!doctype html><meta charset="utf-8"><h1>Selected source-clock sidecar diagnostics</h1><p>Not human gold. Selection SHA '+sha_file(selection_path)+'</p><ul>'+''.join(links)+'</ul>').encode())
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels-root',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-sources',type=int,required=True);p.add_argument('--max-windows-per-source',type=int,default=8)
    p.add_argument('--max-pcm-mib',type=int,default=256);p.add_argument('--audio-root-map',action='append',default=[])
    p.add_argument('--copy-audio',action='store_true');p.add_argument('--read-mib-per-second',type=float,default=16)
    a=p.parse_args()
    if not math.isfinite(a.read_mib_per_second) or a.read_mib_per_second<=0:p.error('read limit must be finite positive')
    if min(a.max_sources,a.max_windows_per_source,a.max_pcm_mib)<=0:p.error('positive explicit limits required')
    result=run(a);print(json.dumps({'rendered':result['rendered'],'failed':result['failed'],'output':a.output}))
    raise SystemExit(0 if result['failed']==0 else 2)


if __name__=='__main__':main()
