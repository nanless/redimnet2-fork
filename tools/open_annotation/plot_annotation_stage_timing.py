"""从固定的有限worker计时快照画图；不读媒体、加载模型或估算全量ETA。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True,type=Path)
    p.add_argument('--snapshot-sha',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--font',type=Path,help='可选明确中文字体文件；缺省选当前Mac实际存在的黑体或宋体')
    a=p.parse_args()
    raw=a.snapshot.read_bytes()
    if len(raw)>4*1024**2 or hashlib.sha256(raw).hexdigest()!=a.snapshot_sha:
        raise ValueError('固定小快照大小或SHA不符')
    value=json.loads(raw);workers=value['workers']
    elapsed=asr=aligner=sources=0
    for worker in workers.values():
        c=worker['performance']['cumulative'];b=c['backend']
        metrics=[c['worker_elapsed_seconds'],b['asr_seconds'],b['aligner_seconds']]
        if not all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in metrics):
            raise ValueError('无效阶段计时')
        elapsed+=metrics[0];asr+=metrics[1];aligner+=metrics[2]
        sources+=c['counters']['assigned_source_rows']
    if not elapsed>0 or asr+aligner>elapsed or a.output.exists():
        raise ValueError('累计计时矛盾或输出已存在')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    candidates=[a.font]if a.font is not None else [Path('/System/Library/Fonts/STHeiti Medium.ttc'),
                                                   Path('/System/Library/Fonts/Supplemental/Songti.ttc')]
    selected=next((f for f in candidates if f.is_file()),None)
    if selected is None:raise ValueError('请用--font提供实际存在的中文字体')
    font=FontProperties(fname=str(selected))
    fig,ax=plt.subplots(figsize=(10,4.8))
    fig.subplots_adjust(left=.035,right=.985,bottom=.24,top=.83)
    parts=[asr,aligner,elapsed-asr-aligner]
    names=['转写','Ultra字界定位','加载、读写及其他']
    colors=['#1967D2','#0D9488','#BBC6D7']
    left=0
    for name,amount,color in zip(names,parts,colors):
        width=amount/elapsed*100
        ax.barh([0],width,left=left,color=color,label=f'{name}  {width:.2f}%')
        if width>7:
            ax.text(left+width/2,0,f'{width:.1f}%',ha='center',va='center',color='white'if color!=colors[2]else'#203047',fontsize=15)
        left+=width
    ax.set_xlim(0,100);ax.set_ylim(-.8,.8);ax.set_yticks([])
    ax.set_xlabel('各工作进程累计经过时间的占比（包含首次加载）',fontproperties=font)
    ax.set_title(f'同一真实FP32普通注意力试验：前{sources}条来源的阶段耗时',fontproperties=font,fontsize=16,pad=22)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,prop=font,frameon=False)
    ax.text(0,.62,f'{len(workers)}个工作进程累计{elapsed:.2f}秒；转写{asr:.2f}秒，定位{aligner:.2f}秒',fontproperties=font,fontsize=11)
    ax.text(0,-.66,'阶段计时不等于GPU内核时间；此图不推算正式全量加速或剩余天数。',fontproperties=font,fontsize=10,color='#506070')
    for spine in ax.spines.values():spine.set_visible(False)
    ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=170);plt.close(fig)
    print(json.dumps({'output':str(a.output),'snapshot_sha256':a.snapshot_sha,
        'sources':sources,'worker_elapsed_seconds':elapsed,'ASR_seconds':asr,
        'aligner_seconds':aligner,'other_seconds':parts[2],'GPU_kernel_time_measured':False},ensure_ascii=False))


if __name__=='__main__':main()
