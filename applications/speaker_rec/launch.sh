#!/bin/bash
set -e

CONFIG=${1:-}

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config.yaml> [num_gpus] [training overrides ...]"
    exit 1
fi

shift
NUM_GPUS=8
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    NUM_GPUS=$1
    shift
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ── 用哪个 Python 环境 ──────────────────────────────────────────────
#
# 训练必须跑在本仓库自己的 conda 环境里（本机是 `redimnet2`），
# 因为它装了这套代码需要的 torch / torchaudio / pandas / pyarrow 等。
# 以前这里直接用 PATH 上的 `torchrun`，于是「在哪个环境里敲命令」决定了
# 实际用哪个解释器：敲错环境就会报缺包，或者更糟——装了两个不同版本的
# torch 时静默跑在错的版本上。
#
# 解析顺序：
#   1. `REDIMNET2_PYTHON` 环境变量（显式指定解释器路径，优先级最高）
#   2. `REDIMNET2_ENV` 环境变量指定的 conda 环境名
#   3. 名字就叫 `redimnet2` 的 conda 环境（本机默认）
#   4. 当前 PATH 上的 `python`（最后兜底，会打印警告）
resolve_python() {
    if [ -n "${REDIMNET2_PYTHON:-}" ]; then
        if [ ! -x "$REDIMNET2_PYTHON" ]; then
            echo "ERROR: REDIMNET2_PYTHON=$REDIMNET2_PYTHON 不可执行" >&2
            exit 1
        fi
        echo "$REDIMNET2_PYTHON"; return
    fi
    local env_name="${REDIMNET2_ENV:-redimnet2}"
    local cand
    for cand in "$HOME/miniforge3/envs/$env_name/bin/python" \
                "$HOME/miniconda3/envs/$env_name/bin/python" \
                "$HOME/anaconda3/envs/$env_name/bin/python" \
                "/opt/conda/envs/$env_name/bin/python"; do
        if [ -x "$cand" ]; then echo "$cand"; return; fi
    done
    if command -v conda >/dev/null 2>&1; then
        local prefix
        prefix="$(conda env list 2>/dev/null | awk -v n="$env_name" '$1==n{print $NF}' | head -1)"
        if [ -n "$prefix" ] && [ -x "$prefix/bin/python" ]; then
            echo "$prefix/bin/python"; return
        fi
    fi
    echo "WARNING: 没找到名为 '$env_name' 的 conda 环境，改用 PATH 上的 python；" \
         "如果报缺包，请用 REDIMNET2_ENV=<环境名> 或 REDIMNET2_PYTHON=<解释器路径> 指定。" >&2
    command -v python
}

PYTHON_BIN="$(resolve_python)"
PY_BIN_DIR="$(dirname "$PYTHON_BIN")"

# 检查这个解释器真的能用；缺包就立刻停下并说清是哪个环境缺什么，
# 而不是让训练跑到一半才在某个 import 上炸掉。
if ! "$PYTHON_BIN" -c 'import torch, torchaudio' >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN 里没有 torch/torchaudio。" >&2
    echo "       请安装依赖，或指定其它环境：" >&2
    echo "         REDIMNET2_ENV=<conda 环境名> $0 ..." >&2
    echo "         REDIMNET2_PYTHON=<python 路径> $0 ..." >&2
    exit 1
fi

# 让 `torchrun` 也来自同一个环境，避免「python 用 A 环境、torchrun 用 B 环境」。
if [ -x "$PY_BIN_DIR/torchrun" ]; then
    TORCHRUN="$PY_BIN_DIR/torchrun"
else
    TORCHRUN="$(command -v torchrun || true)"
    if [ -z "$TORCHRUN" ]; then
        echo "ERROR: 在 $PY_BIN_DIR 与 PATH 上都找不到 torchrun。" >&2
        exit 1
    fi
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/applications/speaker_rec:${PYTHONPATH}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"

export LOGLEVEL="ERROR"
export PYTHONFAULTHANDLER=1

echo "========================================================"
echo "  TRAINING"
echo "  Config: $CONFIG"
echo "  Python: $PYTHON_BIN"
echo "  GPUs:   $NUM_GPUS"
echo "========================================================"

"$TORCHRUN" --nnodes=1:1 --nproc_per_node="$NUM_GPUS" --rdzv-backend=static \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" --master-addr="$MASTER_ADDR" \
    "$REPO_ROOT/applications/speaker_rec/wespeaker_lite/bin/train.py" \
    --config "$CONFIG" "$@"

echo ""
echo "========================================================"
echo "  TRAINING DONE"
echo "========================================================"
