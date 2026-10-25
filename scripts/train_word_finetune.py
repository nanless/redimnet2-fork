#!/usr/bin/env python3
"""CPU-only whole-word/character fine-tuning; private decoding library required."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from word_ft.train import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='完整字词自然变长：每条单独前向、固定条数梯度累积，当前只允许CPU。')
    p.add_argument('--config',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--official',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    p.add_argument('--cpu-threads',type=int,default=2)
    a=vars(p.parse_args());a['config']=json.loads(Path(a['config']).read_text())
    a['dataset_path']=a.pop('dataset');a['official_path']=a.pop('official')
    print(json.dumps(run(**a),ensure_ascii=False))
