#!/bin/bash
set -euo pipefail
: "${PTN_NEW_PACKAGE_SHA:?Externally reviewed new package manifest SHA required}"
: "${PTN_NEW_CONTRACT_SHA:?Externally reviewed new model47 contract SHA required}"
PACKAGE_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
printf '%s  %s\n' "$PTN_NEW_PACKAGE_SHA" "$PACKAGE_ROOT/bundle_files.json" | sha256sum -c -
printf '%s  %s\n' "$PTN_NEW_CONTRACT_SHA" "$PACKAGE_ROOT/resume-contract.json" | sha256sum -c -
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_AUDIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/event-review-20261023-v4/runtime_audio/lib:/usr/local/nccl-rdma-sharp-plugins/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64  # storage-policy-allow: 只读引用历史 runtime 解码库(libsndfile)，不是写出路径
exec /root/miniforge3/envs/redimnet2/bin/python -B "$PACKAGE_ROOT/runtime/entry.py" \
  --contract "$PACKAGE_ROOT/resume-contract.json" --contract-sha "$PTN_NEW_CONTRACT_SHA" \
  --package-sha "$PTN_NEW_PACKAGE_SHA" "$@"
