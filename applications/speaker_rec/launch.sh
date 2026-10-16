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
# 除了"能不能 import"，还要挡住**版本太旧**的解释器。
#
# 实测踩过：某个旧环境里是 torch 1.12.1 配 NumPy 2.x，
# `import torch` 能过，但一用 `torch.from_numpy` 就报
# `RuntimeError: Numpy is not available`。那种错误发生在训练跑到一半的时候，
# 很难一眼看出是环境问题，所以在启动时就挡住。
# 这里要求 torch >= 2.0（本仓库的开发环境是 2.4.0）。
if ! "$PYTHON_BIN" -c 'import torch, torchaudio' >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN 里没有 torch/torchaudio。" >&2
    echo "       请安装依赖，或指定其它环境：" >&2
    echo "         REDIMNET2_ENV=<conda 环境名> $0 ..." >&2
    echo "         REDIMNET2_PYTHON=<python 路径> $0 ..." >&2
    exit 1
fi

# torch 版本下限：低于 2.0 的老环境能 import 但会在训练中途炸。
if ! "$PYTHON_BIN" - <<'PYCHK' >/dev/null 2>&1
import sys
import torch
parts = torch.__version__.split('.')
try:
    major, minor = int(parts[0]), int(parts[1])
except (ValueError, IndexError):
    # 版本号形状不认识（例如只有 "2" 没有点）：**不拦**。
    # 这里必须同时接住 IndexError —— 早先只接了 ValueError，
    # 于是 "2".split('.')[1] 抛 IndexError -> heredoc 非零退出 ->
    # 一个 torch 2.x 被误报成"版本低于 2.0"。
    # 两个分支的策略要一致：认不出来就放行，交给后面的 import 检查。
    sys.exit(0)
sys.exit(0 if (major, minor) >= (2, 0) else 1)
PYCHK
then
    echo "ERROR: $PYTHON_BIN 里的 torch 版本低于 2.0（实测版本见下），本仓库需要 2.x。" >&2
    "$PYTHON_BIN" -c 'import torch; print("       torch =", torch.__version__)' >&2 || true
    echo "       请用较新的环境：" >&2
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

# 通信端口。
#
# 原来这里写死 29400，于是**同一台机器上并发跑两个训练必然有一个起不来**，
# 报的是 `EADDRINUSE: address already in use`，而且要在 torchrun 的
# 一大堆堆栈里才找得到这一行（实测踩过：上一个任务超时没退干净，
# 下一次启动就撞上，看起来像配置错误）。
#
# 现在改成：没显式指定就**自动找一个空闲端口**。
# 这样做是安全的，因为这个脚本用的是 `--nnodes=1:1`（单机），
# 主进程和所有 worker 都在本机、共享同一个文件系统与端口空间。
# 如果哪天要做多机训练，必须显式给 `MASTER_PORT`（多机时各节点得用同一个端口），
# 那时不要依赖自动选端口。
if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT="$("$PYTHON_BIN" - <<'PYPORT'
import socket

# 让内核分配一个空闲端口：绑到 0 号端口再读回实际端口，然后立刻释放。
# 这中间有一个很短的窗口期，理论上别的进程可能抢走它；
# 单机训练里这个窗口小到可以忽略，而且真抢走了也只是启动失败、
# 报的还是同一个 EADDRINUSE，重跑一次即可 —— 比写死端口必冲突要好得多。
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PYPORT
)"
    echo "  MASTER_PORT: $MASTER_PORT（自动选的空闲端口；可用 MASTER_PORT=... 覆盖）"
fi

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
