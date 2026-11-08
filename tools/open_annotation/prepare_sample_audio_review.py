"""把已核原音做成有限API视图：短源整段，长源严格复用原标注上下文窗。

不做VAD删音，不填零，不修字界；解码PCM及来源SHA要与已保存侧车一致。
每个视图保留源采样范围和参考文本，但盲审客户端不会把参考文本发出。
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def need(value, message):
    if not value: raise ValueError(message)

sha=lambda b:hashlib.sha256(b).hexdigest()
def prepare(selection,audio_root,output):
    rows=json.loads(selection.read_text());need(isinstance(rows,list) and 1<=len(rows)<=24,"select 1..24 sources")
    need(len({x['row']['source_id'] for x in rows})==len(rows),'duplicate source')
    need(output.is_absolute() and not output.exists(),'new absolute output required');output.mkdir(mode=0o700)
    cases=[];identities=[]
    stamp=lambda p:(p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
    for source_index,item in enumerate(rows,1):
        need(item['review_id']==f'q{source_index:02d}','ordered qXX review identity required')
        r=item['row'];qid=item['review_id'];p=audio_root/qid/('source'+Path(r['audio_path']).suffix.lower())
        source_stamp=stamp(p);raw=p.read_bytes();need(sha(raw)==r['audio_recorded_file_sha256'],'encoded source SHA differs')
        side=Path(item['local_sidecar']);side_raw=side.read_bytes();need(sha(side_raw)==r['sidecar_sha256'],'sidecar SHA differs');record=json.loads(side_raw);need(record['source']['source_id']==r['source_id'],'sidecar source identity differs');identity=record['source_identity']
        x,sr=sf.read(p,dtype='float32',always_2d=True)
        need(x.shape==(identity['frames'],identity['channels']) and sr==identity['sample_rate'] and np.isfinite(x).all(),'source PCM geometry differs')
        need(sha(x.astype('<f4',copy=False).tobytes())==identity['pcm_sha256'],'native source PCM SHA differs')
        need(stamp(p)==source_stamp and sha(p.read_bytes())==identity['file_sha256'],'local original changed while decoding')
        identities.append(dict(review_id=qid,encoded_sha256=sha(raw),pcm_sha256=identity['pcm_sha256'],frames=len(x),sample_rate=sr,channels=x.shape[1],sidecar_sha256=r['sidecar_sha256']))
        for i,w in enumerate(record['windows']):
            a,b=w['context_start_sample'],w['context_end_sample'];need(0<=a<b<=len(x) and (b-a)/sr<=28.0001,'bounded context window required')
            mono=x[a:b].mean(axis=1,dtype=np.float32)
            if sr!=16000:
                g=math.gcd(sr,16000);mono=resample_poly(mono,16000//g,sr//g).astype(np.float32)
            pcm=mono.astype('<f4',copy=False);need(sha(pcm.tobytes())==w['input_pcm_sha256'],'model input PCM differs')
            path=output/f'{qid}-w{i:02d}.wav';sf.write(path,pcm,16000,subtype='FLOAT')
            cases.append(dict(case_id=path.stem,wavpath=str(path),file_sha256=sha(path.read_bytes()),pcm_float32le_sha256=sha(pcm.tobytes()),frames=len(pcm),sr=16000,source_id=r['source_id'],source_file_sha256=identity['file_sha256'],source_pcm_sha256=identity['pcm_sha256'],source_sample_rate=sr,source_frames=len(x),source_input_sample_bounds=[a,b],source_owned_sample_bounds=[w['owned_start_sample'],w['owned_end_sample']],sidecar_sha256=r['sidecar_sha256'],window_id=w['window_id'],artificial_context_cut_left=a>0,artificial_context_cut_right=b<len(x),reference_text=w['asr'].get('raw_text','')))
    cases.sort(key=lambda c:(0 if c['case_id']=='q15-w00' else 1,c['case_id']))
    manifest=dict(schema='bounded-audio-review-input/v1',unique_original_sources=len(rows),source_ids=sorted({c['source_id'] for c in cases}),source_limit=24,human_gold=False,production_write=False,cases=cases)
    path=output/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(output/'identity-receipt.json').write_text(json.dumps(identities,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),sources=len(rows),manifest=str(path),manifest_sha256=sha(path.read_bytes()),API_calls=0,GPU_calls=0)))

def prepare_unit_crops(selection, audio_root, output, requests):
    """只为明确点选的已发布字词候选做诊断裁片；可疑项不会因此取得使用资格。

    requests是review_id、edge(first/last)、count的有限表；词间包含的其它音素不删除。
    """
    selection_raw=selection.read_bytes();rows=json.loads(selection_raw);selection_pin=sha(selection_raw);need(isinstance(rows,list) and 1<=len(rows)<=24,'select 1..24 sources')
    need(len({x['row']['source_id'] for x in rows})==len(rows),'duplicate source')
    need(all(x['review_id']==f'q{i:02d}' for i,x in enumerate(rows,1)),'ordered qXX review identity required')
    by_id={x['review_id']:x for x in rows}
    requests_raw=requests.read_bytes();requests_pin=sha(requests_raw);jobs=json.loads(requests_raw);need(1<=len(jobs)<=48,'bounded explicit crop requests required')
    need(len(rows)<=24 and len(by_id)==len(rows),'bounded unique selected sources required')
    need(output.is_absolute() and not output.exists(),'new absolute output required');output.mkdir(mode=0o700)
    cases=[]
    for index,job in enumerate(jobs):
        qid=job['review_id'];need(qid in by_id,'crop source outside selected originals')
        item=by_id[qid];r=item['row'];side_raw=Path(item['local_sidecar']).read_bytes();need(sha(side_raw)==r['sidecar_sha256'],'sidecar SHA differs');rec=json.loads(side_raw);need(rec['source']['source_id']==r['source_id'],'source identity differs')
        p=audio_root/qid/('source'+Path(r['audio_path']).suffix.lower());encoded=p.read_bytes();need(sha(encoded)==r['audio_recorded_file_sha256'],'encoded source SHA differs');x,sr=sf.read(p,dtype='float32',always_2d=True)
        ident=rec['source_identity'];need(sr==ident['sample_rate'] and x.shape==(ident['frames'],ident['channels']) and sha(x.astype('<f4',copy=False).tobytes())==ident['pcm_sha256'],'source PCM identity differs')
        n=job['count'];need(type(n)is int and 1<=n<=6 and len(rec['units'])>=n,'crop unit count required');need(job['edge'] in ('first','last'),'known crop edge required')
        units=rec['units'][:n] if job['edge']=='first' else rec['units'][-n:];a=units[0]['start_sample'];b=units[-1]['end_sample'];need(0<=a<b<=len(x) and (b-a)/sr<=28,'bounded candidate crop required')
        pcm=x[a:b].mean(axis=1,dtype=np.float32)
        if sr!=16000:
            g=math.gcd(sr,16000);pcm=resample_poly(pcm,16000//g,sr//g).astype(np.float32)
        pcm=pcm.astype('<f4',copy=False);need(np.isfinite(pcm).all() and sha(p.read_bytes())==ident['file_sha256'],'nonfinite or changing source')
        path=output/f'{qid}-{job["edge"]}{n}-{index:02d}.wav';sf.write(path,pcm,16000,subtype='FLOAT')
        cases.append(dict(case_id=path.stem,wavpath=str(path),file_sha256=sha(path.read_bytes()),pcm_float32le_sha256=sha(pcm.tobytes()),frames=len(pcm),sr=16000,source_id=r['source_id'],source_file_sha256=ident['file_sha256'],source_pcm_sha256=ident['pcm_sha256'],source_sample_rate=sr,source_frames=len(x),source_input_sample_bounds=[a,b],sidecar_sha256=r['sidecar_sha256'],reference_text=' '.join(u['text'] for u in units),selected_units=units,source_decision=r['decision'],source_decision_detail=rec['decision'],selection_sha256=selection_pin,crop_requests_sha256=requests_pin,continuous_source_audio_preserved=True,complete_word_boundary_verified=False,selected_unit_native_indices=[u['native_index'] for u in units],source_windows_alignment=[dict(window_id=w['window_id'],raw_text=w['asr'].get('raw_text',''),alignment=w.get('alignment',{})) for w in rec['windows'] if w['window_id'] in {u['window_id'] for u in units}],diagnostic_candidate_crop=True,usable_for_training=False,usable_for_evaluation=False,artificial_context_cut_left=a>0,artificial_context_cut_right=b<len(x),human_gold=False))
    result=dict(schema='bounded-unit-crop-review/v1',source_ids=sorted({c['source_id'] for c in cases}),cases=cases,human_gold=False,usable_for_training=False,usable_for_evaluation=False)
    path=output/'manifest.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return dict(cases=len(cases),manifest=str(path),manifest_sha256=sha(path.read_bytes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--audio-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--crop-requests',type=Path)
    args=parser.parse_args()
    if args.crop_requests:
        print(json.dumps(prepare_unit_crops(args.selection,args.audio_root,args.output,args.crop_requests),ensure_ascii=False))
    else:
        prepare(args.selection,args.audio_root,args.output)
