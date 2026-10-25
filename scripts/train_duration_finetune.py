#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from duration_ft.train import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='独立同批等长、批间变长微调；默认仅CPU。')
    p.add_argument('--config',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--official',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--resume')
    p.add_argument('--stop-after',type=int);p.add_argument('--gpu-memory-mib',type=float)
    p.add_argument('--cpu-threads',type=int,default=2)
    a=vars(p.parse_args());a['config']=json.loads(Path(a['config']).read_text());a['dataset_path']=a.pop('dataset');a['official_path']=a.pop('official')
    print(json.dumps(run(**a),ensure_ascii=False))
