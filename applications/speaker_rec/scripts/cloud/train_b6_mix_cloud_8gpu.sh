#!/usr/bin/env bash
# =============================================================================
# 火山引擎机器学习平台 8 卡训练启动脚本：b6 混合数据线（vox2 + vb2 + cnc2）
# =============================================================================
#
# 这个脚本训练的目标，是复现官方权重 `b6-vb2+vox2+cnc2_v0-lm.pt`
# （VoxCeleb1-O 等错误率 0.255，数字越小越好）。它分两个阶段：
#
#   1. 主训练阶段（STAGE=ptn）：配方 .../redimnet2/b6/ptn.yaml，
#      训练清单 mix_join（11,521,178 句），每卡批大小 96，共 120 轮。
#   2. 大间隔微调阶段（STAGE=lm）：配方 .../redimnet2/b6/lm.yaml，
#      训练清单 mix_join_lm（5,021,010 句），每卡批大小 48、每段 500 帧、
#      do_lm: true，共 1 轮。它通过配置里的 `model_ckpt` 从主训练的结果继续。
#
# -----------------------------------------------------------------------------
# 为什么要生成一份新配置，而不是直接改配方
# -----------------------------------------------------------------------------
# 配方里的路径是写死的（本机是 /root/group-shared/voiceprint/data/wespeaker_data）。
# 其中四个增强目录（MUSAN 的 music/noise/speech 与 RIRS 的 simulated_rirs）
# 藏在 `dataset_args.aug_setup` 这棵列表套字典的树里面。命令行上的
# `--config_overrides` 是按 YAML 覆盖来合并的，而列表是整体替换，所以没法
# 只改这四条路径。因此本脚本的做法是：
#
#   读出配方 YAML -> 把整份配置里所有以 RECIPE_DATA_PREFIX 开头的字符串
#   递归替换成 DATA_ROOT -> 写出一份完整的新配置 -> 交给 launch.sh 用
#   `--config <新配置>` 启动。
#
# 配方文件本身一个字都不改。若 DATA_ROOT 与 RECIPE_DATA_PREFIX 相同
# （例如云端就把共享盘挂在原来那个路径上），替换结果与原文相同，等于不做改写。
#
# -----------------------------------------------------------------------------
# 断点续训（抢占恢复）是怎么做到的
# -----------------------------------------------------------------------------
# 这是本脚本在云端最关键的一点：**同一条命令重跑就会接着上次继续，不会重头再来。**
#
# 依据是训练代码 train.py 的既有行为（不是本脚本实现的）：
#   * 启动时扫描 $EXP_DIR/models/model_<N>.pt，取**修改时间最新**的那个，
#     令 start_epoch = N + 1；
#   * 训练循环 `for epoch in range(start_epoch, num_epochs + 1)`，
#     所以已经跑过的轮次不会再跑；
#   * 已经跑满时（最新轮次 >= num_epochs）什么都不做，正常退出。
#   * 优化器的动量、AMP 的缩放系数也随权重一起存进 model_<N>.pt，
#     续训时会一起读回（见 wespeaker_lite/utils/checkpoint.py）。
#
# 所以火山引擎实例被抢占后，只要 $EXP_DIR 在**掉电不丢的存储**上（例如 vePFS），
# 重新拉起同一条命令即可。平台侧可以把本脚本的输出当作判断依据：
# 看到 `TRAIN_START epoch=37 ...` 就说明前 36 轮已经跑过、这次从第 37 轮继续。
#
# 本脚本为续训做的事只有两条：不删除 $EXP_DIR；不传 `--checkpoint` 钉死某个文件。
#
# -----------------------------------------------------------------------------
# 机器可读的输出行（平台外层可以用 grep 抓）
# -----------------------------------------------------------------------------
#   TRAIN_BEGIN stage=... recipe=... exp_dir=... num_gpus=...
#   TRAIN_PATHS ...            生成的配置里实际会用到的关键路径
#   TRAIN_REWRITE prefix=... -> ... changed=N
#   TRAIN_CHECK name=... status=ok|skipped 各项开训前检查
#   TRAIN_DISK required_gb=... available_gb=... n_ckpt=...
#   TRAIN_START epoch=... num_epochs=...  这次从第几轮开始
#   TRAIN_CKPT path=... epoch=...         最新的权重文件
#   TRAIN_END status=ok|failed reason=... exit=...  最后一次运行的结果
#
# -----------------------------------------------------------------------------
# 用法
# -----------------------------------------------------------------------------
#   NUM_GPUS=8 DATA_ROOT=/mnt/vepfs/wespeaker_data \
#   EXP_DIR=/mnt/vepfs/exp/b6_cloud/ptn \
#   applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh
#
# 大间隔微调阶段：先跑完主训练，再
#   STAGE=lm NUM_GPUS=8 DATA_ROOT=... EXP_DIR=/mnt/vepfs/exp/b6_cloud/lm \
#   PTN_EXP_DIR=/mnt/vepfs/exp/b6_cloud/ptn \
#   applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh
# 也可以用等价的包装脚本 train_b6_mix_cloud_8gpu_lm.sh（它只是把 STAGE 设成 lm）。
#
# 本机 2 卡小样验证（几轮就结束）：
#   NUM_GPUS=2 SAMPLE_NUM_PER_EPOCH=576 NUM_EPOCHS=2 \
#   EXP_DIR=/dev/shm/... DATA_ROOT=/root/group-shared/voiceprint/data/wespeaker_data \
#   applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh
#
# =============================================================================

set -Eeuo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0. 环境变量：每一项都有默认值，都可以在命令行前面覆盖
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

STAGE="${STAGE:-ptn}"
case "$STAGE" in
    ptn)
        DEFAULT_RECIPE="$REPO_ROOT/applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/ptn.yaml"
        DEFAULT_EXP_DIR="$REPO_ROOT/exp/vox2_vb2_cnc2/b6_cloud/ptn"
        DEFAULT_TRAIN_LINES=11521178
        ;;
    lm)
        DEFAULT_RECIPE="$REPO_ROOT/applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/lm.yaml"
        DEFAULT_EXP_DIR="$REPO_ROOT/exp/vox2_vb2_cnc2/b6_cloud/lm"
        DEFAULT_TRAIN_LINES=5021010
        ;;
    *)
        printf 'STAGE 只能是 ptn 或 lm，收到: %s\n' "$STAGE" >&2
        printf 'TRAIN_END status=failed reason=bad_stage stage=%s\n' "$STAGE" >&2
        exit 2
        ;;
esac

# 配方文件（本脚本只读它，不改它）
RECIPE="${RECIPE:-$DEFAULT_RECIPE}"
# 云端数据根目录：配方里所有以 RECIPE_DATA_PREFIX 开头的路径都会改成它下面的同名子路径
DATA_ROOT="${DATA_ROOT:-/root/group-shared/voiceprint/data/wespeaker_data}"
# 配方里写死的那个前缀（本机默认值；换配方时用这个变量告诉脚本要替换什么）
RECIPE_DATA_PREFIX="${RECIPE_DATA_PREFIX:-/root/group-shared/voiceprint/data/wespeaker_data}"
# 训练结果目录：权重、train.log、val_log.csv、spk2id.json 都写在这里；续训靠它
EXP_DIR="${EXP_DIR:-$DEFAULT_EXP_DIR}"
# 用几张卡（launch.sh 的第二个位置参数）
NUM_GPUS="${NUM_GPUS:-8}"
# 用哪个 Python（留空则按 launch.sh 的规则自动找 conda 环境 redimnet2）
PYTHON_BIN="${PYTHON_BIN:-}"
# 通信端口（留空则 launch.sh 自动选一个空闲端口；多机训练时必须显式指定）
MASTER_PORT="${MASTER_PORT:-}"

# 期望的训练清单行数。主训练 11,521,178 句、大间隔微调 5,021,010 句。
# 换成别的数据子集时用这个变量改；设成 0 表示只查文件存在、不比对行数。
EXPECT_TRAIN_LINES="${EXPECT_TRAIN_LINES:-$DEFAULT_TRAIN_LINES}"

# 磁盘预算用的假设：b6 每个权重约 650 MB，另外留出余量。
CKPT_SIZE_MB="${CKPT_SIZE_MB:-650}"
DISK_HEADROOM_GB="${DISK_HEADROOM_GB:-20}"

# 可选的训练覆盖（留空表示用配方里的值）。这些值会直接写进生成的新配置。
NUM_EPOCHS="${NUM_EPOCHS:-}"
SAMPLE_NUM_PER_EPOCH="${SAMPLE_NUM_PER_EPOCH:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SAVE_EPOCH_INTERVAL="${SAVE_EPOCH_INTERVAL:-}"
VOX1_TEST_DIR="${VOX1_TEST_DIR:-}"

# 大间隔微调阶段专用：从哪个主训练结果继续。
PTN_EXP_DIR="${PTN_EXP_DIR:-$REPO_ROOT/exp/vox2_vb2_cnc2/b6_cloud/ptn}"
MODEL_CKPT="${MODEL_CKPT:-}"

# 各项检查的开关（小样测试时可以逐项关掉；1 = 跳过）
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SKIP_DATA_LIST_CHECK="${SKIP_DATA_LIST_CHECK:-0}"
SKIP_UTT2SPK_CHECK="${SKIP_UTT2SPK_CHECK:-0}"
SKIP_AUG_CHECK="${SKIP_AUG_CHECK:-0}"
SKIP_VOX1_CHECK="${SKIP_VOX1_CHECK:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
SKIP_DISK_CHECK="${SKIP_DISK_CHECK:-0}"
SKIP_LOCK_CHECK="${SKIP_LOCK_CHECK:-0}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. 输出与失败处理
# ─────────────────────────────────────────────────────────────────────────────

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(_ts)" "$*"; }

# 统一失败出口：说明原因，打印机器可读的 TRAIN_END，退出非零。
die() {
    local reason="$1"; shift
    printf '[%s] 检查未通过（%s）：%s\n' "$(_ts)" "$reason" "$*" >&2
    printf 'TRAIN_END status=failed reason=%s exit=1\n' "$reason" >&2
    exit 1
}

# 没预料到的失败也留一行 TRAIN_END，免得平台只看到非零退出码却不知道原因。
trap 'rc=$?; printf "[%s] 未预料到的失败：第 %s 行，退出码 %s\n" "$(_ts)" "$LINENO" "$rc" >&2; printf "TRAIN_END status=failed reason=unexpected_error line=%s exit=%s\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR

# ─────────────────────────────────────────────────────────────────────────────
# 2. 找 Python（和 launch.sh 一样的解析顺序）
# ─────────────────────────────────────────────────────────────────────────────

resolve_python() {
    if [ -n "$PYTHON_BIN" ]; then
        [ -x "$PYTHON_BIN" ] && printf '%s' "$PYTHON_BIN"
        return 0
    fi
    if [ -n "${REDIMNET2_PYTHON:-}" ] && [ -x "${REDIMNET2_PYTHON}" ]; then
        printf '%s' "$REDIMNET2_PYTHON"; return 0
    fi
    local env_name="${REDIMNET2_ENV:-redimnet2}" cand
    for cand in "$HOME/miniforge3/envs/$env_name/bin/python" \
                "$HOME/miniconda3/envs/$env_name/bin/python" \
                "$HOME/anaconda3/envs/$env_name/bin/python" \
                "/opt/conda/envs/$env_name/bin/python"; do
        if [ -x "$cand" ]; then printf '%s' "$cand"; return 0; fi
    done
    if command -v conda >/dev/null 2>&1; then
        local prefix
        prefix="$(conda env list 2>/dev/null | awk -v n="$env_name" '$1==n{print $NF}' | head -1 || true)"
        if [ -n "$prefix" ] && [ -x "$prefix/bin/python" ]; then
            printf '%s' "$prefix/bin/python"; return 0
        fi
    fi
    command -v python 2>/dev/null || true
}

PYTHON_BIN="$(resolve_python || true)"
[ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ] || die python_not_found \
    "找不到可用的 Python。请设置 PYTHON_BIN=/path/to/python，或 REDIMNET2_ENV=<conda 环境名>。"
if ! "$PYTHON_BIN" -c 'import torch, yaml' >/dev/null 2>&1; then
    die python_missing_deps "$PYTHON_BIN 里缺 torch 或 PyYAML，请换一个装了依赖的环境。"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. 准备结果目录并加锁（同一目录只允许一个训练在跑）
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$EXP_DIR"
EXP_DIR="$(cd "$EXP_DIR" && pwd)"   # 归一化成绝对路径，后续续训扫描用它
GEN_CONFIG="$EXP_DIR/cloud_config_${STAGE}.yaml"
PATHS_ENV="$EXP_DIR/cloud_paths_${STAGE}.env"
RUN_LOG="$EXP_DIR/cloud_launch_${STAGE}.log"

LOCK_FILE="$EXP_DIR/.cloud_train.lock"
PID_FILE="$EXP_DIR/.cloud_train.pid"

if [ "$SKIP_LOCK_CHECK" = "1" ]; then
    log "跳过占用检查（SKIP_LOCK_CHECK=1）"
else
    if command -v flock >/dev/null 2>&1; then
        # fd 9 在脚本整个生命周期里保持打开，退出时内核自动释放锁。
        exec 9>"$LOCK_FILE"
        if ! flock -n 9; then
            holder="$(cat "$PID_FILE" 2>/dev/null || true)"
            die exp_dir_locked \
                "$EXP_DIR 已经有训练在跑（占用进程 PID=${holder:-未知}）。不要同时启动两个训练，请先确认上一个是否还在。"
        fi
        printf '%s\n' "$$" > "$PID_FILE"
    else
        # 没有 flock 就退回到 PID 文件检查（不如 flock 严格，但能挡住重复启动）。
        if [ -f "$PID_FILE" ]; then
            old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
                die exp_dir_locked "$EXP_DIR 已经有训练在跑（PID=$old_pid）。"
            fi
        fi
        printf '%s\n' "$$" > "$PID_FILE"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. 大间隔微调阶段：先把要续训的主训练权重确定下来
# ─────────────────────────────────────────────────────────────────────────────

ckpt_epoch() {  # 从 model_<N>.pt 里取出 N
    local b="${1##*/}"
    b="${b#model_}"; b="${b%.pt}"
    if [[ "$b" =~ ^[0-9]+$ ]]; then printf '%s' "$b"; else printf '0'; fi
}

# 与 train.py 一致：按修改时间取最新的 model_<N>.pt
newest_ckpt() {
    local dir="$1" line
    [ -d "$dir/models" ] || return 1
    line="$(find "$dir/models" -maxdepth 1 -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 || true)"
    [ -n "$line" ] || return 1
    printf '%s' "${line#* }"
}

# 把相对路径变成绝对路径，免得「检查时按一个目录、训练时又按另一个目录」。
to_abs() {
    local path="$1"
    if [ -d "$path" ]; then
        printf '%s' "$(cd "$path" && pwd)"
    else
        printf '%s/%s' "$(cd "$(dirname "$path")" && pwd)" "$(basename "$path")"
    fi
}

[ -f "$RECIPE" ] || die recipe_missing "配方文件不存在: $RECIPE"
RECIPE="$(to_abs "$RECIPE")"
if [ -d "$DATA_ROOT" ]; then
    DATA_ROOT="$(to_abs "$DATA_ROOT")"
fi

PTN_SPK2ID=""
if [ "$STAGE" = "lm" ]; then
    if [ -z "$MODEL_CKPT" ]; then
        MODEL_CKPT="$(newest_ckpt "$PTN_EXP_DIR" || true)"
        [ -n "$MODEL_CKPT" ] || die ptn_ckpt_missing \
            "大间隔微调需要一个主训练权重，但在 $PTN_EXP_DIR/models 里没找到 model_*.pt。请先跑完 STAGE=ptn，或用 MODEL_CKPT=<权重文件> 指定。"
    fi
    [ -f "$MODEL_CKPT" ] || die ptn_ckpt_missing "MODEL_CKPT 指向的文件不存在: $MODEL_CKPT"
    MODEL_CKPT="$(to_abs "$MODEL_CKPT")"
    PTN_SPK2ID="$PTN_EXP_DIR/spk2id.json"
    [ -f "$PTN_SPK2ID" ] || die ptn_spk2id_missing \
        "找不到主训练保存的说话人编号表 $PTN_SPK2ID；大间隔微调要用它把分类头重新拼起来。"
    log "大间隔微调从主训练权重继续: $MODEL_CKPT (第 $(ckpt_epoch "$MODEL_CKPT") 轮)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. 生成云端配置：递归改写路径
# ─────────────────────────────────────────────────────────────────────────────

export REPO_ROOT RECIPE GEN_CONFIG PATHS_ENV STAGE RECIPE_DATA_PREFIX DATA_ROOT EXP_DIR
export MODEL_CKPT PTN_SPK2ID
export NUM_EPOCHS SAMPLE_NUM_PER_EPOCH BATCH_SIZE SAVE_EPOCH_INTERVAL VOX1_TEST_DIR

if ! "$PYTHON_BIN" - <<'PYGEN'
import os
import shlex
import sys

import yaml


def env(name, default=""):
    value = os.environ.get(name, default)
    return value if value is not None else default


recipe = env("RECIPE")
gen_config = env("GEN_CONFIG")
paths_env = os.environ["PATHS_ENV"]
stage = env("STAGE")
prefix = env("RECIPE_DATA_PREFIX").rstrip("/")
data_root = env("DATA_ROOT").rstrip("/")

try:
    with open(recipe) as handle:
        config = yaml.safe_load(handle)
except FileNotFoundError:
    sys.stderr.write("配方文件不存在: %s\n" % recipe)
    sys.exit(3)

if not isinstance(config, dict):
    sys.stderr.write("配方不是一份 YAML 映射: %s\n" % recipe)
    sys.exit(3)

changed = []


def rewrite(node):
    """递归遍历配置，把以 prefix 开头的字符串改成 data_root 下的同名路径。"""
    if isinstance(node, dict):
        return {key: rewrite(value) for key, value in node.items()}
    if isinstance(node, list):
        return [rewrite(value) for value in node]
    if isinstance(node, str) and prefix:
        if node == prefix:
            new = data_root
        elif node.startswith(prefix + "/"):
            new = data_root + node[len(prefix):]
        else:
            return node
        if new != node:
            changed.append((node, new))
        return new
    return node


config = rewrite(config)

# 明确覆盖的几项（与路径改写无关）
config["exp_dir"] = env("EXP_DIR")
if stage == "lm":
    config["model_ckpt"] = env("MODEL_CKPT")
    config["spk2id_map_path"] = env("PTN_SPK2ID")


def as_int(name):
    raw = env(name).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        sys.stderr.write("%s 必须是整数，收到: %s\n" % (name, raw))
        sys.exit(4)


# 超参覆盖：**只给冒烟测试用**。复现官方结果时必须一个都不设，
# 否则跑出来的就不是官方那组超参了。下面每处覆盖都会打印一行 TRAIN_WARN，
# 让"这次跑偏离了配方"这件事在日志里一眼可见，而不是悄悄发生。
_ov = []
num_epochs_override = as_int("NUM_EPOCHS")
if num_epochs_override is not None:
    _ov.append(("num_epochs", config.get("num_epochs"), num_epochs_override))
    config["num_epochs"] = num_epochs_override
sample_override = as_int("SAMPLE_NUM_PER_EPOCH")
if sample_override is not None:
    _ov.append(("dataset_args.sample_num_per_epoch",
                config.get("dataset_args", {}).get("sample_num_per_epoch"),
                sample_override))
    config.setdefault("dataset_args", {})["sample_num_per_epoch"] = sample_override
batch_override = as_int("BATCH_SIZE")
if batch_override is not None:
    _ov.append(("dataloader_args.batch_size",
                config.get("dataloader_args", {}).get("batch_size"),
                batch_override))
    config.setdefault("dataloader_args", {})["batch_size"] = batch_override
save_override = as_int("SAVE_EPOCH_INTERVAL")
if save_override is not None:
    _ov.append(("save_epoch_interval", config.get("save_epoch_interval"),
                save_override))
    config["save_epoch_interval"] = save_override
for _k, _old, _new in _ov:
    print("TRAIN_WARN 超参被环境变量覆盖，本次已偏离配方: {} = {} -> {}"
          .format(_k, _old, _new))
if _ov:
    print("TRAIN_WARN 共 {} 处超参偏离配方；"
          "复现官方结果请不要设 NUM_EPOCHS / SAMPLE_NUM_PER_EPOCH / "
          "BATCH_SIZE / SAVE_EPOCH_INTERVAL".format(len(_ov)))
vox1_override = env("VOX1_TEST_DIR").strip()
if vox1_override:
    if not isinstance(config.get("validation"), dict):
        config["validation"] = {}
    config["validation"]["vox1"] = vox1_override

# 收集增强目录（aug_setup 树里的 dataset_dir）以及三张关键路径
aug_dirs = []


def collect(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "dataset_dir" and isinstance(value, str):
                aug_dirs.append(value)
            collect(value)
    elif isinstance(node, list):
        for value in node:
            collect(value)


collect(config)
train_data = config.get("train_data", "")
train_label = config.get("train_label", "")
validation = config.get("validation") or {}
vox1 = validation.get("vox1", "") if isinstance(validation, dict) else ""

# 检查有没有漏改的路径（配方里还残留旧前缀就说明改写不完整）。
# DATA_ROOT 与旧前缀相同时不存在"漏改"，整段跳过。
leftover = []


def find_leftover(node):
    if isinstance(node, dict):
        for value in node.values():
            find_leftover(value)
    elif isinstance(node, list):
        for value in node:
            find_leftover(value)
    elif isinstance(node, str) and prefix and node.startswith(prefix + "/"):
        leftover.append(node)


if prefix and data_root != prefix:
    find_leftover(config)

with open(gen_config, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True,
                   default_flow_style=False)

lines = []


def put(key, value):
    lines.append("%s=%s" % (key, shlex.quote(str(value))))


put("CFG_TRAIN_DATA", train_data)
put("CFG_TRAIN_LABEL", train_label)
put("CFG_VOX1", vox1)
put("CFG_NUM_EPOCHS", config.get("num_epochs", 0))
put("CFG_SAVE_EPOCH_INTERVAL", config.get("save_epoch_interval", 1))
put("CFG_NUM_AVG", config.get("num_avg", 0))
put("CFG_BATCH_SIZE", (config.get("dataloader_args") or {}).get("batch_size", 0))
put("CFG_SAMPLE_NUM_PER_EPOCH",
    (config.get("dataset_args") or {}).get("sample_num_per_epoch", 0))
put("CFG_AUG_DIR_COUNT", len(aug_dirs))
lines.append("CFG_AUG_DIRS=(%s)" % " ".join(shlex.quote(d) for d in aug_dirs))
with open(paths_env, "w") as handle:
    handle.write("\n".join(lines) + "\n")

print("TRAIN_REWRITE stage=%s prefix=%s data_root=%s changed=%d leftover=%d"
      % (stage, prefix, data_root, len(changed), len(leftover)))
for old, new in changed[:12]:
    print("TRAIN_REWRITE old=%s new=%s" % (old, new))
if len(changed) > 12:
    print("TRAIN_REWRITE more=%d" % (len(changed) - 12))

print("TRAIN_PATHS config=%s" % gen_config)
print("TRAIN_PATHS exp_dir=%s" % config["exp_dir"])
print("TRAIN_PATHS train_data=%s" % train_data)
print("TRAIN_PATHS train_label=%s" % train_label)
print("TRAIN_PATHS vox1=%s" % vox1)
for index, directory in enumerate(aug_dirs, 1):
    print("TRAIN_PATHS aug_%d=%s" % (index, directory))
if stage == "lm":
    print("TRAIN_PATHS model_ckpt=%s" % config.get("model_ckpt", ""))
    print("TRAIN_PATHS spk2id_map_path=%s" % config.get("spk2id_map_path", ""))

if leftover:
    sys.stderr.write("有 %d 条路径还带着旧前缀 %s，改写不完整：\n" % (len(leftover), prefix))
    for item in leftover[:10]:
        sys.stderr.write("  %s\n" % item)
    sys.exit(5)
PYGEN
then
    die config_generation_failed "生成云端配置失败，请看上面的 Python 报错。"
fi

# PATHS_ENV 由生成脚本写出，供后面各项检查使用。
# shellcheck disable=SC1090
source "$PATHS_ENV"

log "生成的云端配置: $GEN_CONFIG"
log "关键路径已经写在 TRAIN_PATHS 行里，请人工核对一遍。"

# ─────────────────────────────────────────────────────────────────────────────
# 6. 开训前检查（逐项可用 SKIP_* 关掉）
# ─────────────────────────────────────────────────────────────────────────────

check_ok() { printf 'TRAIN_CHECK name=%s status=ok %s\n' "$1" "${2:-}"; }
check_skip() { printf 'TRAIN_CHECK name=%s status=skipped %s\n' "$1" "${2:-}"; }

if [ "$SKIP_PREFLIGHT" = "1" ]; then
    log "跳过全部开训前检查（SKIP_PREFLIGHT=1）"
else

    # 6.1 训练清单存在，行数与期望一致
    if [ "$SKIP_DATA_LIST_CHECK" = "1" ]; then
        check_skip data_list "SKIP_DATA_LIST_CHECK=1"
    else
        [ -f "$CFG_TRAIN_DATA" ] || die data_list_missing "训练样本清单不存在: $CFG_TRAIN_DATA"
        [ -f "$CFG_TRAIN_LABEL" ] || die data_list_missing "说话人对照表不存在: $CFG_TRAIN_LABEL"
        data_lines="$(wc -l < "$CFG_TRAIN_DATA")"
        [ "$data_lines" -gt 0 ] || die data_list_empty "训练样本清单是空文件: $CFG_TRAIN_DATA"
        if [ "$EXPECT_TRAIN_LINES" -gt 0 ] && [ "$data_lines" -ne "$EXPECT_TRAIN_LINES" ]; then
            die data_list_count_mismatch \
                "训练样本清单行数不对：实际 $data_lines 行，期望 $EXPECT_TRAIN_LINES 行（$CFG_TRAIN_DATA）。若这份数据确实该是这个行数，请用 EXPECT_TRAIN_LINES=0 关掉行数比对。"
        fi
        check_ok data_list "data_list_lines=$data_lines expected=$EXPECT_TRAIN_LINES"
    fi

    # 6.2 说话人对照表与样本清单行数一致
    if [ "$SKIP_UTT2SPK_CHECK" = "1" ]; then
        check_skip utt2spk "SKIP_UTT2SPK_CHECK=1"
    else
        [ -f "$CFG_TRAIN_LABEL" ] || die data_list_missing "说话人对照表不存在: $CFG_TRAIN_LABEL"
        label_lines="$(wc -l < "$CFG_TRAIN_LABEL")"
        [ "$label_lines" -gt 0 ] || die data_list_empty "说话人对照表是空文件: $CFG_TRAIN_LABEL"
        data_lines="$(wc -l < "$CFG_TRAIN_DATA")"
        if [ "$label_lines" -ne "$data_lines" ]; then
            die utt2spk_count_mismatch \
                "说话人对照表与样本清单行数不一致：utt2spk $label_lines 行，data.list $data_lines 行。两张表必须一一对应。"
        fi
        check_ok utt2spk "utt2spk_lines=$label_lines data_list_lines=$data_lines"
    fi

    # 6.3 增强音频目录存在且非空（配置里所有 dataset_dir）
    if [ "$SKIP_AUG_CHECK" = "1" ]; then
        check_skip aug_dirs "SKIP_AUG_CHECK=1"
    else
        [ "${CFG_AUG_DIR_COUNT:-0}" -gt 0 ] || die aug_dir_missing \
            "生成的配置里一个 dataset_dir 都没有，增强数据没接上，请检查配方。"
        aug_index=0
        for aug_dir in "${CFG_AUG_DIRS[@]}"; do
            aug_index=$((aug_index + 1))
            [ -d "$aug_dir" ] || die aug_dir_missing "增强音频目录不存在: $aug_dir"
            first_wav="$(find -L "$aug_dir" -type f -name '*.wav' -print -quit 2>/dev/null || true)"
            [ -n "$first_wav" ] || die aug_dir_empty "增强音频目录里找不到 .wav 文件: $aug_dir"
            wav_count="$(find -L "$aug_dir" -type f -name '*.wav' 2>/dev/null | wc -l)"
            printf 'TRAIN_CHECK name=aug_dir status=ok index=%d wavs=%s dir=%s\n' \
                "$aug_index" "$wav_count" "$aug_dir"
        done
    fi

    # 6.4 VoxCeleb1 验证协议存在
    if [ "$SKIP_VOX1_CHECK" = "1" ]; then
        check_skip vox1 "SKIP_VOX1_CHECK=1"
    else
        [ -d "$CFG_VOX1" ] || die vox1_missing "VoxCeleb1 验证目录不存在: $CFG_VOX1"
        [ -s "$CFG_VOX1/veri_test2.txt" ] || die vox1_protocol_missing \
            "找不到 VoxCeleb1-O 试验列表: $CFG_VOX1/veri_test2.txt"
        [ -d "$CFG_VOX1/wav" ] || die vox1_wav_missing "找不到验证音频目录: $CFG_VOX1/wav"
        check_ok vox1 "dir=$CFG_VOX1 trials=$(wc -l < "$CFG_VOX1/veri_test2.txt")"
    fi

    # 6.5 看得见的卡数不少于 NUM_GPUS
    if [ "$SKIP_GPU_CHECK" = "1" ]; then
        check_skip gpu "SKIP_GPU_CHECK=1"
    else
        visible_gpus="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true)"
        [[ "$visible_gpus" =~ ^[0-9]+$ ]] || die gpu_query_failed \
            "查不出可见的 GPU 数量，请确认 $PYTHON_BIN 里能正常 import torch 并看到显卡。"
        if [ "$visible_gpus" -lt "$NUM_GPUS" ]; then
            die gpu_count_too_low \
                "torch 只看到 $visible_gpus 张卡，但 NUM_GPUS=$NUM_GPUS。请把 NUM_GPUS 调小，或检查 CUDA_VISIBLE_DEVICES。"
        fi
        check_ok gpu "visible=$visible_gpus requested=$NUM_GPUS"
    fi

    # 6.6 结果目录所在磁盘的剩余空间够放权重
    if [ "$SKIP_DISK_CHECK" = "1" ]; then
        check_skip disk "SKIP_DISK_CHECK=1"
    else
        num_epochs_cfg="${CFG_NUM_EPOCHS:-0}"
        save_interval_cfg="${CFG_SAVE_EPOCH_INTERVAL:-1}"
        num_avg_cfg="${CFG_NUM_AVG:-0}"
        [ "$save_interval_cfg" -ge 1 ] 2>/dev/null || save_interval_cfg=1
        if [ "$save_interval_cfg" -le 1 ]; then
            n_ckpt="$num_epochs_cfg"
        else
            # 每 save_epoch_interval 轮留一个，另外训练末尾会保留最后 num_avg 轮的权重
            n_ckpt=$(( num_epochs_cfg / save_interval_cfg + num_avg_cfg + 2 ))
        fi
        [ "$n_ckpt" -ge 1 ] || n_ckpt=1
        required_gb=$(( (n_ckpt * CKPT_SIZE_MB + 1023) / 1024 + DISK_HEADROOM_GB ))
        avail_kb="$(df -Pk "$EXP_DIR" | awk 'NR==2 {print $4}')"
        [[ "$avail_kb" =~ ^[0-9]+$ ]] || die disk_query_failed "读不出 $EXP_DIR 所在磁盘的剩余空间。"
        avail_gb=$(( avail_kb / 1024 / 1024 ))
        printf 'TRAIN_DISK required_gb=%d available_gb=%d n_ckpt=%d ckpt_size_mb=%d save_epoch_interval=%d num_epochs=%d headroom_gb=%d\n' \
            "$required_gb" "$avail_gb" "$n_ckpt" "$CKPT_SIZE_MB" "$save_interval_cfg" "$num_epochs_cfg" "$DISK_HEADROOM_GB"
        if [ "$avail_gb" -lt "$required_gb" ]; then
            die disk_too_low \
                "$EXP_DIR 所在磁盘只剩 ${avail_gb} GB，按当前设置需要约 ${required_gb} GB（$n_ckpt 个权重 × ${CKPT_SIZE_MB} MB + ${DISK_HEADROOM_GB} GB 余量）。可以把 save_epoch_interval 调大（例如 5），权重数量会明显减少。"
        fi
        check_ok disk "required_gb=$required_gb available_gb=$avail_gb"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. 启动训练
# ─────────────────────────────────────────────────────────────────────────────

LAUNCH_SH="$REPO_ROOT/applications/speaker_rec/launch.sh"
[ -x "$LAUNCH_SH" ] || die launcher_missing "找不到可执行的 launch.sh: $LAUNCH_SH"

start_epoch=1
prev_ckpt="$(newest_ckpt "$EXP_DIR" || true)"
if [ -n "$prev_ckpt" ]; then
    start_epoch=$(( $(ckpt_epoch "$prev_ckpt") + 1 ))
fi

printf 'TRAIN_BEGIN stage=%s recipe=%s exp_dir=%s num_gpus=%s python=%s\n' \
    "$STAGE" "$RECIPE" "$EXP_DIR" "$NUM_GPUS" "$PYTHON_BIN"
printf 'TRAIN_START epoch=%d num_epochs=%s prev_ckpt=%s\n' \
    "$start_epoch" "${CFG_NUM_EPOCHS:-?}" "${prev_ckpt:-none}"
printf 'TRAIN_LOG path=%s\n' "$RUN_LOG"

# launch.sh 认这两个环境变量：用哪个解释器、用哪个端口。
export REDIMNET2_PYTHON="$PYTHON_BIN"
if [ -n "$MASTER_PORT" ]; then
    export MASTER_PORT
fi

cd "$REPO_ROOT"

# 训练失败时不要让 ERR trap 抢先把脚本带走，这里自己收退出码，
# 好把日志末尾打出来、并给出 TRAIN_END status=failed。
set +e
"$LAUNCH_SH" "$GEN_CONFIG" "$NUM_GPUS" 2>&1 | tee -a "$RUN_LOG"
train_rc=${PIPESTATUS[0]}
set -e

new_ckpt="$(newest_ckpt "$EXP_DIR" || true)"

if [ "$train_rc" -ne 0 ]; then
    printf '\n----- 训练日志末尾（%s）-----\n' "$RUN_LOG" >&2
    tail -n 60 "$RUN_LOG" >&2 || true
    printf 'TRAIN_END status=failed reason=training_failed exit=%d\n' "$train_rc" >&2
    exit "$train_rc"
fi

if [ -n "$new_ckpt" ]; then
    printf 'TRAIN_CKPT path=%s epoch=%s\n' "$new_ckpt" "$(ckpt_epoch "$new_ckpt")"
else
    printf 'TRAIN_CKPT path=none epoch=0\n'
fi
printf 'TRAIN_END status=ok reason=finished exit=0 last_epoch=%s\n' \
    "$(ckpt_epoch "${new_ckpt:-model_0.pt}")"
