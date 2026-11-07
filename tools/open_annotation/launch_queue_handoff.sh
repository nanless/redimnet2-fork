#!/bin/bash
# 本入口只等待真实平台观察；实例已分配且可能计费，最长等待1800秒。
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_AUDIT BASH_ENV ENV
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
if [[ $# != 3 ]]; then
  echo '用法：launch_queue_handoff.sh 绝对blueprint路径 blueprint的SHA256 外层源清单SHA256' >&2
  exit 64
fi
blueprint_path=$1
blueprint_sha=$2
source_list_sha=$3
[[ "$blueprint_path" == /* && "$blueprint_sha" =~ ^[0-9a-f]{64}$ && "$source_list_sha" =~ ^[0-9a-f]{64}$ ]] || exit 64
outer_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
printf '%s  %s\n' "$source_list_sha" "$outer_root/queue-handoff-files.sha256" | sha256sum --check --status
(cd -- "$outer_root" && sha256sum --check --strict --status queue-handoff-files.sha256)
printf '%s  %s\n' "$blueprint_sha" "$blueprint_path" | sha256sum --check --status
python_binary=/root/miniforge3/envs/qwen3-asr/bin/python3.10
printf '%s  %s\n' cdb9592ef64842d7528ac97c2e03c81f226e5d875d5cc9304b59131e7bbcd505 "$python_binary" | sha256sum --check --status
exec "$python_binary" -B "$outer_root/queue_handoff.py" wait --blueprint "$blueprint_path" --blueprint-sha "$blueprint_sha" --execute
