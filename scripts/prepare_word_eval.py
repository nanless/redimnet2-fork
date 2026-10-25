#!/usr/bin/env python3
"""Prepare immutable whole-reference-utterance unit/count protocols; no model inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from word_ft.prepare import prepare_childmandarin

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dev-speakers',type=int,default=30);p.add_argument('--test-speakers',type=int,default=30)
    p.add_argument('--seed',type=int,default=20261023)
    a=p.parse_args(argv)
    print(json.dumps(prepare_childmandarin(a.corpus_root,a.output,dev_speakers=a.dev_speakers,
        test_speakers=a.test_speakers,seed=a.seed),ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
