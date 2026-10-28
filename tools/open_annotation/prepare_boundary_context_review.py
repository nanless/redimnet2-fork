"""为明确点选的少量原音生成连续上下文盲审WAV及波形/频谱图。

只支持已经核实的16k单声道原件，按原采样点直接切片，保留增益与FLOAT
精度；不利用VAD删音、不参考候选文字、不修正标注。输出不是完整字训练
裁片，更不是人工字界真值。请求和选择清单都必须显式提供SHA。
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import plot_saved_boundary_review as P


def prepare(selection,selection_sha,audio_root,requests,requests_sha,output,*,plots_only=False):
    import numpy as np
    import soundfile as sf
    rows=P.strict_json(P.read_pinned(selection,selection_sha,P.MAX_JSON))
    jobs=P.strict_json(P.read_pinned(requests,requests_sha,P.MAX_JSON))
    P.need(isinstance(rows,list)and 1<=len(rows)<=24,'只允许1—24个明确来源')
    by_id={r['review_id']:r for r in rows}
    P.need(len(by_id)==len(rows)and len({r['row']['source_id']for r in rows})==len(rows),'来源重复')
    P.need(isinstance(jobs,list)and 1<=len(jobs)<=48,'只允许1—48个明确连续上下文')
    seen=set();loaded={}
    for job in jobs:
        P.need(set(job)=={'case_id','review_id','start_sample','end_sample'},'上下文请求字段不符')
        cid,qid=job['case_id'],job['review_id'];a,b=job['start_sample'],job['end_sample']
        P.need(isinstance(cid,str)and P.ID.fullmatch(cid)and cid not in seen,'上下文编号重复或不安全');seen.add(cid)
        P.need(qid in by_id,'上下文来源不在选择清单')
        if qid not in loaded:loaded[qid]=P.load_case(by_id[qid],audio_root)
        rec,pcm,sr,audio,proof=loaded[qid]
        P.need(sr==16000 and proof['channels']==1,'本工具只允许原生16k单声道；禁止隐式变换')
        P.need(type(a)is int and type(b)is int and 0<=a<b<=len(pcm)and b-a<=28*sr,'上下文采样范围无效')
    output=Path(output)
    P.need(output.is_absolute()and not output.exists()and not output.is_symlink()and output.parent.is_dir(),'需要新的绝对输出目录')
    output.mkdir(mode=0o700)
    import matplotlib
    matplotlib.use('Agg')
    cases=[];figures=[]
    for job in jobs:
        cid,qid=job['case_id'],job['review_id'];a,b=job['start_sample'],job['end_sample']
        rec,pcm,sr,audio,proof=loaded[qid];item=by_id[qid]
        figure=output/(cid+'.png');P.make_figure(pcm,sr,rec,P.interval_rows(rec),a/sr,b/sr,cid+' | diagnostic continuous context',figure)
        figures.append(dict(path=str(figure),sha256=P.sha(figure.read_bytes()),source_sample_bounds=[a,b],
                            encoded_sha256=proof['encoded_sha256'],sidecar_sha256=item['row']['sidecar_sha256']))
        if plots_only:continue
        cut=pcm[a:b].astype('<f4',copy=False);path=output/(cid+'.wav');sf.write(path,cut,sr,subtype='FLOAT')
        raw=path.read_bytes();back,actual_sr=sf.read(io.BytesIO(raw),dtype='float32')
        P.need(actual_sr==sr and cut.tobytes()==back.astype('<f4',copy=False).tobytes(),'FLOAT上下文写回不一致')
        cases.append(dict(case_id=cid,wavpath=str(path),file_sha256=P.sha(raw),pcm_float32le_sha256=P.sha(cut.tobytes()),
            frames=len(cut),sr=sr,source_id=rec['source']['source_id'],source_file_sha256=proof['encoded_sha256'],
            source_pcm_sha256=proof['pcm_sha256'],source_sample_rate=sr,source_frames=len(pcm),source_input_sample_bounds=[a,b],
            sidecar_sha256=item['row']['sidecar_sha256'],reference_text='',continuous_source_audio_preserved=True,
            artificial_context_cut_left=a>0,artificial_context_cut_right=b<len(pcm),gain_applied=False,resampled=False,
            diagnostic_context=True,complete_word_boundary_verified=False,usable_for_training=False,usable_for_evaluation=False,human_gold=False))
    manifest=dict(schema='bounded-diagnostic-context-review/v1',source_ids=sorted({r['source']['source_id']for r,_,_,_,_ in loaded.values()}),
        cases=cases,human_gold=False,production_write=False,selection_sha256=selection_sha,requests_sha256=requests_sha)
    (output/'manifest.json').write_bytes(P.json_bytes(manifest))
    receipt=dict(source_sha256=P.sha(Path(__file__).read_bytes()),plot_source_sha256=P.sha(Path(P.__file__).read_bytes()),
        selection_sha256=selection_sha,requests_sha256=requests_sha,figures=figures,plots_only=plots_only,
        manifest=dict(path=str(output/'manifest.json'),sha256=P.sha((output/'manifest.json').read_bytes())),API_calls=0,GPU_calls=0,
        human_gold=False,production_write=False)
    (output/'receipt.json').write_bytes(P.json_bytes(receipt));return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('selection','audio-root','requests','output'):parser.add_argument('--'+field,type=Path,required=True)
    for field in ('selection-sha','requests-sha'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--plots-only',action='store_true');a=parser.parse_args()
    print(P.json_bytes(prepare(a.selection,a.selection_sha,a.audio_root,a.requests,a.requests_sha,a.output,plots_only=a.plots_only)).decode())
