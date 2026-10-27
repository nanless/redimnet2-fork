#!/bin/bash
# 只执行已固定的恢复控制源；真正GPU资格仍由Python控制器和旧独占父进程检查。
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_AUDIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
if [[ $# != 3 ]]; then
  echo '用法：launch_portable_successor.sh 绝对spec路径 spec的SHA256 控制源清单的SHA256' >&2
  exit 64
fi
spec_path=$1
spec_sha=$2
source_list_sha=$3
[[ "$spec_path" == /* && "$spec_sha" =~ ^[0-9a-f]{64}$ && "$source_list_sha" =~ ^[0-9a-f]{64}$ ]] || exit 64
control_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
printf '%s  %s\n' "$source_list_sha" "$control_root/controller-files.sha256" | sha256sum --check --status
(cd -- "$control_root" && sha256sum --check --strict --status controller-files.sha256)
printf '%s  %s\n' "$spec_sha" "$spec_path" | sha256sum --check --status
# 直接调用已核实际二进制；不让未经核实的python软链接先执行控制源码。
python_binary=/root/miniforge3/envs/qwen3-asr/bin/python3.10
printf '%s  %s\n' cdb9592ef64842d7528ac97c2e03c81f226e5d875d5cc9304b59131e7bbcd505 "$python_binary" | sha256sum --check --status
exec "$python_binary" -B "$control_root/run_portable_successor.py" --spec "$spec_path" --spec-sha "$spec_sha" --execute
