#!/usr/bin/env python3
"""独立时长矩阵/校准入口；默认CPU，GPU需要显式授权与显存预算。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duration_ft.evaluate import evaluate_matrix, prepare_calibration, calibrate_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-calibration", help="只读音频头，构造与O身份独立的开发表")
    prepare.add_argument("--audio-root", type=Path, required=True)
    prepare.add_argument("--o-protocol", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--speakers", type=int, default=100)
    prepare.add_argument("--clips-per-speaker", type=int, default=5)
    prepare.add_argument("--min-seconds", type=float, default=3.0)
    for name in ("evaluate", "calibrate"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--allow-gpu", action="store_true")
        command.add_argument("--gpu-memory-mib", type=int)
        if name == "evaluate":
            command.add_argument("--protocol", type=Path, required=True)
            command.add_argument("--audio-root", type=Path, required=True)
            command.add_argument("--calibration", type=Path)
            command.add_argument("--max-audio-errors", type=int, default=10)
            command.add_argument("--max-consecutive-audio-errors", type=int, default=3)
            command.add_argument("--nonstandard-test-protocol", action="store_true")
        else:
            command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-calibration":
        result = prepare_calibration(args.audio_root, args.o_protocol, args.output,
                                     speaker_count=args.speakers, clips_per_speaker=args.clips_per_speaker,
                                     min_seconds=args.min_seconds)
        print(json.dumps({k: v for k, v in result.items() if k not in ("audio", "excluded_unreadable")}, ensure_ascii=False, indent=2))
    else:
        common = dict(device=args.device, allow_gpu=args.allow_gpu, gpu_memory_mib=args.gpu_memory_mib)
        if args.command == "calibrate":
            result = calibrate_model(args.checkpoint, args.selection, args.output, **common)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            result = evaluate_matrix(args.checkpoint, args.protocol, args.audio_root, args.output,
                                     strict_standard=not args.nonstandard_test_protocol,
                                     calibration=args.calibration, max_audio_errors=args.max_audio_errors,
                                     max_consecutive_audio_errors=args.max_consecutive_audio_errors, **common)
            print(json.dumps({"complete": result["complete"], "cells": len(result["cells"]),
                              "unique_crops": result["unique_crops"], "cache_hits": result["cache_hits"],
                              "errors": len(result["extraction_errors"])}, ensure_ascii=False))
            return 0 if result["complete"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
