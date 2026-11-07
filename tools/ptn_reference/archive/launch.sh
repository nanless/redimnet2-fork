#!/bin/bash
set -euo pipefail
BUNDLE=/root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-code-model35-20261024-v3
: "${PTN_BUNDLE_MANIFEST_SHA:?Must provide the externally reviewed bundle manifest SHA}"
printf '%s  %s\n' "$PTN_BUNDLE_MANIFEST_SHA" "$BUNDLE/bundle_files.json" | sha256sum -c -
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_AUDIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/event-review-20261023-v4/runtime_audio/lib:/usr/local/nccl-rdma-sharp-plugins/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
exec /root/miniforge3/envs/redimnet2/bin/python -B "$BUNDLE/entry.py" "$@"
