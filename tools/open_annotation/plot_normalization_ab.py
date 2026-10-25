"""从已完成的有界对照报告绘图：只读计数，不读音频或模型。

左图比较原生目标项数；右图必须先把A按与B相同的原文范围配组，
不能把换了分母的零长计数直接解释为边界准确率。输出必须为新PNG。
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile


def require(ok,message):
    if not ok:raise ValueError(message)


def load_counts(path,pin):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'报告必须是实际普通文件')
    before=path.stat();require(before.st_size<=16*1024**2,'报告超出有界大小')
    raw=path.read_bytes();after=path.stat()
    snap=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(snap(before)==snap(after) and hashlib.sha256(raw).hexdigest()==pin,'报告字节或摘要改变')
    r=json.loads(raw)
    require(r.get('schema')=='annotation-normalization-AB-audit/v1' and r.get('passed_for_recorded_machine_comparison') is True,
            '只接受已经通过实际记录核对的对照报告')
    n=r.get('source_count');require(type(n)is int and 1<=n<=16,'只绘制最多16条源的诊断')
    require(r.get('human_gold') is False and r.get('label_eligibility_released') is False,'不能把图冒称金标或训练准入')
    g,s=r['geometry_totals'],r['same_raw_text_span_totals']
    counts=[g['A']['items'],g['B']['items'],g['A']['FIX_zero'],s['FIX_A_zero'],s['FIX_B_zero'],s['same_raw_text_spans'],s['merged_A_fragment_groups']]
    require(all(type(x)is int and x>=0 for x in counts),'计数必须为非负整数')
    require(s.get('unmeasured_windows',0)==0 and s['same_raw_text_spans']==g['B']['items']
            and s['FIX_B_zero']==g['B']['FIX_zero'] and s['FIX_A_zero']<=g['A']['FIX_zero']
            and s['time_differences_are_accuracy_errors'] is False,'相同原文范围的分母或计数不完整')
    return r


def render(audit,audit_sha,output):
    output=Path(output);require(not output.exists() and not output.is_symlink(),'输出不能覆盖已有文件或软链接')
    require(output.suffix.lower()=='.png' and output.parent.is_dir(),'请给已有目录下的新PNG路径')
    require(output.resolve()!=Path(audit).resolve(),'输出不能覆盖输入报告')
    r=load_counts(audit,audit_sha);g,s=r['geometry_totals'],r['same_raw_text_span_totals']
    # Only a new temporary Matplotlib cache is used if no cache was supplied.
    with tempfile.TemporaryDirectory(prefix='annotation-plot-cache-')as cache:
        previous=os.environ.get('MPLCONFIGDIR');os.environ.setdefault('MPLCONFIGDIR',cache)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
            fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            vals=[g['A']['items'],g['B']['items']]
            bars=ax[0].bar(['A: split punctuation','B: keep apostrophe'],vals,color=['#456a9c','#3a8577'],width=.58)
            ax[0].bar_label(bars,padding=3);ax[0].set_ylim(0,max(1,math.ceil(max(vals)*1.12/10)*10));ax[0].set_ylabel('Native target items')
            changed=r['target_changed_source_count']
            ax[0].set_title('%d fragments removed from %d sources'%(vals[0]-vals[1],changed))
            vals=[g['A']['FIX_zero'],s['FIX_A_zero'],s['FIX_B_zero']]
            labels=['A native\n%d items'%g['A']['items'],'A regrouped\n%d raw-text spans'%s['same_raw_text_spans'],
                    'B native\n%d raw-text spans'%s['same_raw_text_spans']]
            bars=ax[1].bar(labels,vals,color=['#456a9c','#849dc0','#3a8577'],width=.58)
            ax[1].bar_label(bars,padding=3);ax[1].set_ylim(0,max(vals+[1])+2);ax[1].set_ylabel('Zero-duration FIX intervals')
            ax[1].set_title('Equal text spans: %d vs %d, not %d vs %d'%(s['FIX_A_zero'],s['FIX_B_zero'],g['A']['FIX_zero'],g['B']['FIX_zero']))
            fig.suptitle('%d fixed diagnostic sources: machine observations, no human gold'%r['source_count'],fontsize=13)
            data=io.BytesIO();fig.savefig(data,format='png',dpi=160);plt.close(fig)
        finally:
            if previous is None:os.environ.pop('MPLCONFIGDIR',None)
    with output.open('xb')as stream:stream.write(data.getvalue())
    return {'audit_sha256':audit_sha,'figure_sha256':hashlib.sha256(data.getvalue()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output':str(output),'model_media_GPU_API_execution':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--audit-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(render(a.audit,a.audit_sha,a.output),ensure_ascii=False))


if __name__=='__main__':main()
