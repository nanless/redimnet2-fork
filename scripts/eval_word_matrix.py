#!/usr/bin/env python3
"""Evaluate complete units at their natural length; CPU default, explicit CUDA gate."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('evaluate')
    for name in ('checkpoint','dataset','protocol','output'):e.add_argument('--'+name,type=Path,required=True)
    e.add_argument('--split',choices=['test','dev-cal'],default='test');e.add_argument('--calibration',type=Path)
    e.add_argument('--device',default='cpu');e.add_argument('--allow-gpu',action='store_true');e.add_argument('--gpu-memory-mib',type=int)
    e.add_argument('--allow-exploratory',action='store_true',help='candidate-only research output; cannot calibrate main thresholds')
    e.add_argument('--max-audio-errors',type=int,default=10);e.add_argument('--max-consecutive-audio-errors',type=int,default=3)
    c=sub.add_parser('calibrate')
    c.add_argument('--dev-summary',type=Path,required=True);c.add_argument('--dataset',type=Path,required=True)
    c.add_argument('--output',type=Path,required=True);c.add_argument('--cell',default='4-6x4-6')
    a=p.parse_args(argv)
    # Set before torch import; CPU requests never enumerate or initialize CUDA.
    if a.command=='calibrate' or a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    from word_ft.evaluate import evaluate_words,calibrate_words
    if a.command=='calibrate':result=calibrate_words(a.dev_summary,a.dataset,a.output,cell=a.cell);ok=True
    else:
        result=evaluate_words(a.checkpoint,a.dataset,a.protocol,a.output,split=a.split,device=a.device,
            allow_gpu=a.allow_gpu,gpu_memory_mib=a.gpu_memory_mib,calibration=a.calibration,
            allow_exploratory=a.allow_exploratory,max_audio_errors=a.max_audio_errors,
            max_consecutive_audio_errors=a.max_consecutive_audio_errors);ok=result['complete']
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if ok else 2
if __name__=='__main__':raise SystemExit(main())
