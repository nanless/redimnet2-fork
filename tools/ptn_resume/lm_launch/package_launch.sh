#!/bin/bash
# =============================================================================
# LM 冻结包的执行入口（builder 会把本文件原字节复制为 runtime/lm_launch.sh）
# =============================================================================
# 平台 Entrypoint 期望两个外部 sha：
#   LM_PACKAGE_SHA   包清单 bundle_files.json 的 sha256
#   LM_CONTRACT_SHA  合同 resume-contract.json 的 sha256
# 两个值由维护者在提交时给出，运行时从外部注入——不从包内自取，避免自证。
#
# 这个脚本只做：验包 → 验合同 → 交给 runtime/lm_stage_gate.py。
# 训练逻辑不在包里新写：gate 会调用 source/ 里那份冻结源码树的 STAGE=lm 入口。
set -euo pipefail
: "${LM_PACKAGE_SHA:?Externally reviewed package manifest SHA required}"
: "${LM_CONTRACT_SHA:?Externally reviewed LM contract SHA required}"
PACKAGE_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
printf '%s  %s\n' "$LM_PACKAGE_SHA" "$PACKAGE_ROOT/bundle_files.json" | sha256sum -c -
printf '%s  %s\n' "$LM_CONTRACT_SHA" "$PACKAGE_ROOT/resume-contract.json" | sha256sum -c -
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_AUDIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/event-review-20261023-v4/runtime_audio/lib:/usr/local/nccl-rdma-sharp-plugins/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64  # storage-policy-allow: 只读引用历史 runtime 解码库(libsndfile)，不是写出路径
exec /root/miniforge3/envs/redimnet2/bin/python -B "$PACKAGE_ROOT/runtime/lm_stage_gate.py" \
  --package-root "$PACKAGE_ROOT" \
  --contract "$PACKAGE_ROOT/resume-contract.json" --contract-sha "$LM_CONTRACT_SHA" \
  --package-sha "$LM_PACKAGE_SHA" "$@"
