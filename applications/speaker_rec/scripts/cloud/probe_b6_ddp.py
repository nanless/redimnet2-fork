#!/usr/bin/env python3
"""在单机八卡上检查 NCCL 通信；只有八个进程全部成功才写结果。"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != 8 or local_world != 8 or not 0 <= rank < 8 or not 0 <= local_rank < 8:
        raise RuntimeError("必须由 torchrun --standalone --nnodes=1 --nproc_per_node=8 启动")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError(f"要求恰好八张可见 CUDA 卡，当前为 {torch.cuda.device_count()}")
    if not dist.is_nccl_available():
        raise RuntimeError("当前 PyTorch 不支持 NCCL")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    try:
        value = torch.tensor([rank], dtype=torch.int64, device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        if value.item() != 28:
            raise RuntimeError(f"八进程 rank 求和应为 28，当前为 {value.item()}")
        props = torch.cuda.get_device_properties(local_rank)
        row = {"rank": rank, "local_rank": local_rank, "hostname": socket.gethostname(),
               "gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
               "compute_capability": [props.major, props.minor], "all_reduce_sum": value.item()}
        print("NCCL_RANK " + json.dumps(row, ensure_ascii=False), flush=True)
        rows = [None] * 8
        dist.all_gather_object(rows, row)
        if len({v["hostname"] for v in rows}) != 1 or {v["local_rank"] for v in rows} != set(range(8)):
            raise RuntimeError("本次任务不是单机八卡，或设备编号有重复")
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            result = {"success": True, "backend": "nccl", "world_size": 8, "ranks": rows,
                      "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                      "nccl": list(torch.cuda.nccl.version()),
                      "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            os.replace(temporary, args.output)
            print(f"NCCL_OK world_size=8 sum=28 output={args.output}", flush=True)
        dist.barrier(device_ids=[local_rank])
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
