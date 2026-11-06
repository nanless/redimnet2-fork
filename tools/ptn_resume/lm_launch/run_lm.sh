#!/usr/bin/env bash
# =============================================================================
# 三语料 B6 大间隔微调（LM）唯一入口 —— 只准备，不提交
# =============================================================================
#
# 这个 wrapper 不做训练，只把"启动前必须成立的条件"变成可以失败的检查，
# 然后把控制权交给仓库既有的云端入口：
#
#     applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh   (STAGE=lm)
#
# 为什么不直接调 run_b6_mix_reproduction.sh：那个脚本的 formal 分支会**先重训
# PTN 120 轮**，并要求同源码的八卡 smoke 成功件；本次 PTN 已经由冻结的 190
# 成员包跑完 48—120，不需要也不应该重训。
#
# 它拒绝的事情（任何一条不满足就退出，不"降级凑合"）：
#   1. EXP_DIR 已存在（唯一新输出；失败后换版本号，不覆盖）
#   2. MODEL_CKPT 不是第 115 轮、或字节不是 pin 值 65497b6a…aad5c
#   3. PTN_EXP_DIR/spk2id.json 缺失（LM 分类头必须按名字重组）
#   4. LM 初始化门 check_b6_lm_init.py 失败（先真跑一遍，再训练）
#   5. 训练超参被外部覆盖（轮数/批大小/每轮样本数/保存间隔）
#   6. SKIP_* 检查开关被打开
#   7. EXP_DIR 落在代码目录或 PTN 段目录内部
#
# 用法（容器内）：
#     bash tools/ptn_resume/lm_launch/run_lm.sh
# 变量见同目录 env.template。退出码：0=训练进程正常退出（训练是否有效另判），
# 2=前置条件不满足，3=输出目录已被占用。
# =============================================================================
set -Eeuo pipefail

PINNED_MODEL_115_SHA256="65497b6a00efac412a82d563aebc1e767056fef23e7f4234f48f89cc8b3aad5c"

log()  { printf 'LM_LAUNCH %s\n' "$*" >&2; }
die()  { local code="$1"; shift; printf 'LM_LAUNCH_FAILED code=%s %s\n' "$code" "$*" >&2; exit "$code"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

DATA_ROOT="${DATA_ROOT:-/root/group-shared/voiceprint/data/wespeaker_data}"
PTN_EXP_DIR="${PTN_EXP_DIR:-/root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn}"
EXP_DIR="${EXP_DIR:-/root/experiments/redimnet2/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/lm}"
MODEL_CKPT="${MODEL_CKPT:-$PTN_EXP_DIR/models/model_115.pt}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/redimnet2/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"

CLOUD_SCRIPT="$REPO_ROOT/applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh"
LM_RECIPE="$REPO_ROOT/applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/official_repro_lm.yaml"
INIT_CHECK="$REPO_ROOT/applications/speaker_rec/scripts/cloud/check_b6_lm_init.py"
LM_LABEL="${LM_LABEL:-$DATA_ROOT/vox2_vb2_cnc2_v0/mix_join_lm/utt2spk}"

abs() { "$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1"; }

# ---- 0. 基本存在性 ---------------------------------------------------------
[ -x "$PYTHON_BIN" ] || die 2 "解释器不可执行: $PYTHON_BIN"
for file in "$CLOUD_SCRIPT" "$LM_RECIPE" "$INIT_CHECK"; do
    [ -f "$file" ] || die 2 "仓库文件缺失: $file"
done
[ -d "$DATA_ROOT" ] || die 2 "DATA_ROOT 不存在: $DATA_ROOT"
[ -d "$PTN_EXP_DIR" ] || die 2 "PTN_EXP_DIR 不存在: $PTN_EXP_DIR"

PTN_EXP_DIR="$(abs "$PTN_EXP_DIR")"
MODEL_CKPT="$(abs "$MODEL_CKPT")"
EXP_DIR="$(abs "$EXP_DIR")"

# ---- 1. 唯一新输出 ---------------------------------------------------------
if [ -e "$EXP_DIR" ]; then
    die 3 "EXP_DIR 已存在，拒绝覆盖/续跑: $EXP_DIR   失败后请改用 …-v2/lm 新目录"
fi
case "$EXP_DIR/" in
    "$REPO_ROOT"/*) die 2 "EXP_DIR 不能放在代码目录内部: $EXP_DIR" ;;
    "$PTN_EXP_DIR"/*) die 2 "EXP_DIR 不能放在 PTN 段目录内部（会污染第三段证据）: $EXP_DIR" ;;
esac

# ---- 2. 权重身份（先查名字、后查字节：便宜的先失败） -----------------------
[ -f "$MODEL_CKPT" ] || die 2 "MODEL_CKPT 不存在: $MODEL_CKPT"
case "$(basename "$MODEL_CKPT")" in
    model_115.pt) : ;;
    *) die 2 "LM 必须从第 115 轮初始化，收到 $(basename "$MODEL_CKPT")（缺省由 mtime 取最新会拿到 model_120.pt）" ;;
esac
[ -f "$PTN_EXP_DIR/spk2id.json" ] || die 2 "PTN 说话人映射缺失: $PTN_EXP_DIR/spk2id.json"

# ---- 3. 训练超参不许被覆盖 -------------------------------------------------
for key in NUM_EPOCHS BATCH_SIZE SAMPLE_NUM_PER_EPOCH SAVE_EPOCH_INTERVAL NUM_AVG RECIPE; do
    if [ -n "${!key:-}" ]; then
        die 2 "拒绝外部覆盖 $key=${!key}：LM 超参只能来自 official_repro_lm.yaml"
    fi
done
for key in SKIP_PREFLIGHT SKIP_DATA_LIST_CHECK SKIP_UTT2SPK_CHECK SKIP_AUG_CHECK \
           SKIP_VOX1_CHECK SKIP_GPU_CHECK SKIP_DISK_CHECK SKIP_LOCK_CHECK; do
    if [ "${!key:-0}" != "0" ]; then
        die 2 "拒绝跳过检查 $key=${!key}"
    fi
done

# ---- 3b. 权重字节（650 MB，放最后算） --------------------------------------
actual_sha="$(sha256sum "$MODEL_CKPT" | awk '{print $1}')"
[ "$actual_sha" = "$PINNED_MODEL_115_SHA256" ] \
    || die 2 "model_115.pt 摘要不符：pin=$PINNED_MODEL_115_SHA256 actual=$actual_sha"

# ---- 4. LM 初始化门（真跑，CPU，只读） -------------------------------------
mkdir -p "$EXP_DIR"
log "step=lm_init_check checkpoint=$MODEL_CKPT epoch=115"
"$PYTHON_BIN" -B "$INIT_CHECK" \
    --checkpoint "$MODEL_CKPT" \
    --ptn-spk2id "$PTN_EXP_DIR/spk2id.json" \
    --lm-label "$LM_LABEL" \
    --ptn-recipe "$REPO_ROOT/applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/official_repro_ptn.yaml" \
    --expected-epoch 115 > "$EXP_DIR/lm_initialization.json" \
    || die 2 "LM 初始化门失败，见 $EXP_DIR/lm_initialization.json"

# ---- 5. 冻结本次启动输入 ---------------------------------------------------
{
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'git_head=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'exp_dir=%s\n' "$EXP_DIR"
    printf 'ptn_exp_dir=%s\n' "$PTN_EXP_DIR"
    printf 'model_ckpt=%s\n' "$MODEL_CKPT"
    printf 'model_ckpt_sha256=%s\n' "$actual_sha"
    printf 'data_root=%s\n' "$DATA_ROOT"
    printf 'lm_recipe=%s\n' "$LM_RECIPE"
    printf 'lm_recipe_sha256=%s\n' "$(sha256sum "$LM_RECIPE" | awk '{print $1}')"
    printf 'num_gpus=%s\n' "$NUM_GPUS"
    printf 'automatic_retry=disabled\n'
} > "$EXP_DIR/lm_launch_identity.env"

log "step=train entry=$CLOUD_SCRIPT stage=lm exp_dir=$EXP_DIR"

# ---- 6. 交给既有云端入口 ---------------------------------------------------
export REPO_ROOT DATA_ROOT PTN_EXP_DIR EXP_DIR MODEL_CKPT PYTHON_BIN
export STAGE=lm REPRO=1 NUM_GPUS
exec bash "$CLOUD_SCRIPT"
