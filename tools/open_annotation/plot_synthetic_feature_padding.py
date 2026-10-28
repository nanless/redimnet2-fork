"""只读取已保存的合成mel数组，画补零与动态范围传播；不重新提特征。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('arrays','result','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arrays-sha',required=True);p.add_argument('--result-sha',required=True);p.add_argument('--font',type=Path)
    a=p.parse_args()
    if sha(a.arrays)!=a.arrays_sha or sha(a.result)!=a.result_sha:raise ValueError('合成输入SHA不符')
    result=json.loads(a.result.read_text())
    if result.get('real_audio_read')is not False or result.get('CUDA_initialized')is not False:raise ValueError('只接受实际纯合成CPU诊断结果')
    if result['saved_synthetic_arrays']['sha256']!=a.arrays_sha:raise ValueError('报告与合成数组不是同一批')
    if a.output.exists()or not a.output.is_absolute()or not a.output.parent.is_dir():raise ValueError('须明确给出全新图文件')
    import numpy as np
    with np.load(a.arrays,allow_pickle=False)as z:
        raw1=z['n16000_b1_raw_log_valid'];raw4=z['n16000_b4_raw_log_valid'];mel1=z['n16000_b1_valid_mel'];mel4=z['n16000_b4_valid_mel']
    if not(raw1.shape==raw4.shape==mel1.shape==mel4.shape==(128,100)):raise ValueError('固定合成诊断数组形状不符')
    dr=np.abs(raw1.astype('float64')-raw4.astype('float64'));dm=np.abs(mel1.astype('float64')-mel4.astype('float64'))
    if np.flatnonzero(dr.max(axis=0)>1e-6).tolist()!=[99]or len(np.flatnonzero(dm.max(axis=0)>1e-6))!=99:raise ValueError('实际诊断数字与图预期不符')
    row=next(x for x in result['results']if x['synthetic_PCM_frames']==16000)
    first=float(dm[:,:33].max());middle=float(dm[:,33:66].max());tail=float(dm[:,66:].max())
    import os,tempfile
    cache=tempfile.TemporaryDirectory(prefix='synthetic-feature-plot-');os.environ.setdefault('MPLCONFIGDIR',cache.name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import LogNorm
    fonts=[a.font]if a.font else[Path('/System/Library/Fonts/STHeiti Medium.ttc'),Path('/System/Library/Fonts/Hiragino Sans GB.ttc')]
    font=next((FontProperties(fname=str(x))for x in fonts if x and x.is_file()),None)
    if font is None:raise ValueError('请用--font指定可显示中文的实际字体')
    fig=plt.figure(figsize=(14,9),facecolor='white');grid=fig.add_gridspec(2,2,height_ratios=[1.25,1],hspace=.43,wspace=.27)
    ax=fig.add_subplot(grid[0,:]);frames=np.arange(100)
    ax.plot(frames,np.maximum(dr.max(axis=0),1e-9),label='原始 log-mel：仅 frame 99 变化',color='#3866a4',lw=2)
    ax.plot(frames,np.maximum(dm.max(axis=0),1e-9),label='最终特征：动态范围地板把差异带到较早帧',color='#cd562f',lw=2)
    ax.set_yscale('log');ax.set_ylim(5e-10,5);ax.set_xlim(0,101);ax.grid(alpha=.18);ax.set_xlabel('有效帧序号（从 0 开始）',fontproperties=font);ax.set_ylabel('每帧最大绝对差（对数轴）',fontproperties=font)
    ax.legend(loc='upper left',prop=font,frameon=True);ax.set_title('同一条合成音频：B1 与补零到更长同伴后的 B4；B4/B8 完全相同',fontproperties=font,fontsize=13,pad=14)
    ax.annotate(f'前 / 中区最大差\n{first:.7g}',xy=(48,middle),xytext=(42,.006),fontproperties=font,fontsize=11,arrowprops={'arrowstyle':'->','color':'#6b7685'},bbox={'boxstyle':'round,pad=.4','fc':'#fff4ec','ec':'none'})
    ax.annotate(f'frame 99\nraw 最大差 {dr.max():.6f}\n最终最大差 {tail:.6f}',xy=(99,tail),xytext=(72,.07),fontproperties=font,fontsize=11,arrowprops={'arrowstyle':'->','color':'#6b7685'},bbox={'boxstyle':'round,pad=.4','fc':'#eef3fa','ec':'none'})
    for col,(data,title,vmax)in enumerate([(dr,'末尾 5 帧：原始 log-mel 差异',float(dr.max())),(dm,'末尾 5 帧：动态范围裁剪 / 缩放后差异',float(dm.max()))]):
        axis=fig.add_subplot(grid[1,col]);image=axis.imshow(np.maximum(data[:,95:],1e-9),origin='lower',aspect='auto',extent=[94.5,99.5,-.5,127.5],norm=LogNorm(vmin=1e-6,vmax=vmax),cmap='magma');axis.set_xticks(range(95,100));axis.set_xlabel('有效帧序号',fontproperties=font);axis.set_ylabel('mel 通道',fontproperties=font);axis.set_title(title,fontproperties=font,fontsize=12,pad=12);bar=fig.colorbar(image,ax=axis,pad=.025);bar.set_label('绝对差（对数颜色）',fontproperties=font)
    peak1=row['batches']['1']['raw_log_global_max'];peak4=row['batches']['4']['raw_log_global_max']
    fig.suptitle('PCM 右补零 → 最后 STFT 帧改变 → 本条最大值改变 → 前面帧的地板也改变',fontproperties=font,fontsize=17,y=.98)
    fig.text(.08,.895,f'16,000 个合成采样点 / 16 kHz；有效帧同为 100。每条最大值：{peak1:.9f} → {peak4:.9f}',fontproperties=font,fontsize=11,color='#445065')
    fig.text(.08,.035,'不是跨批次归约：同伴振幅 ×100，目标特征不变；同长度 B4 与 B1 也不变。\n0 差绘在 1e-9，仅为对数显示下限；不涉及真实原音、权重或文字准确率。',fontproperties=font,fontsize=11,color='#445065')
    fig.subplots_adjust(left=.08,right=.95,top=.84,bottom=.12);fig.savefig(a.output,dpi=180);plt.close(fig)
    print(json.dumps(dict(figure=str(a.output),sha256=sha(a.output),first_region_max=first,middle_region_max=middle,tail_region_max=tail,raw_maximum_mel_frame=np.unravel_index(dr.argmax(),dr.shape),final_maximum_mel_frame=np.unravel_index(dm.argmax(),dm.shape),synthetic_only=True,features_recomputed=False),default=int,ensure_ascii=False))

if __name__=='__main__':main()
