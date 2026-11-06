#!/usr/bin/env bash
# Guard-path tests for tools/ptn_resume/lm_launch/run_lm.sh.
#
# Everything here is a refusal path: no torch, no GPU, no network, no real
# checkpoint. It proves the wrapper stops before training on each of the seven
# conditions it claims to enforce. It does NOT prove the training path works —
# that needs the real 650 MB model_115.pt on the training host.
#
# Run:  bash tools/ptn_resume/tests/test_lm_launch_guards.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WRAP="$REPO/tools/ptn_resume/lm_launch/run_lm.sh"

[ -f "$WRAP" ] || { echo "missing wrapper: $WRAP" >&2; exit 1; }

TMP="$(mktemp -d /tmp/lmwrap.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

PTN="$TMP/ptn"
mkdir -p "$PTN/models" "$TMP/data" "$TMP/empty" "$TMP/taken"
printf '{}\n' > "$PTN/spk2id.json"
printf 'not-a-checkpoint' > "$PTN/models/model_115.pt"
printf 'other' > "$PTN/models/model_120.pt"

export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export NUM_GPUS=1 DATA_ROOT="$TMP/data"

rc=0
run() {
    local name="$1" expected="$2" needle="$3"
    shift 3
    local out code
    out="$(env "$@" bash "$WRAP" 2>&1)"
    code=$?
    if [ "$code" != "$expected" ]; then
        printf 'FAIL %s exit=%s want=%s\n' "$name" "$code" "$expected"
        printf '%s\n' "$out" | tail -2
        rc=1
        return
    fi
    case "$out" in
        *"$needle"*) printf 'PASS %s exit=%s\n' "$name" "$code" ;;
        *) printf 'FAIL %s missing:%s\n' "$name" "$needle"; printf '%s\n' "$out" | tail -2; rc=1 ;;
    esac
}

run exp_dir_exists      3 "拒绝覆盖"        PTN_EXP_DIR="$PTN" EXP_DIR="$TMP/taken"
run wrong_ckpt_name     2 "必须从第 115 轮"  PTN_EXP_DIR="$PTN" EXP_DIR="$TMP/b/lm" MODEL_CKPT="$PTN/models/model_120.pt"
run missing_spk2id      2 "说话人映射缺失"   PTN_EXP_DIR="$TMP/empty" EXP_DIR="$TMP/c/lm" MODEL_CKPT="$PTN/models/model_115.pt"
run hyperparam_override 2 "拒绝外部覆盖"     PTN_EXP_DIR="$PTN" EXP_DIR="$TMP/d/lm" NUM_EPOCHS=2
run skip_switch         2 "拒绝跳过检查"     PTN_EXP_DIR="$PTN" EXP_DIR="$TMP/e/lm" SKIP_GPU_CHECK=1
run sha_mismatch        2 "摘要不符"         PTN_EXP_DIR="$PTN" EXP_DIR="$TMP/f/lm"
run exp_inside_ptn      2 "PTN 段目录内部"   PTN_EXP_DIR="$PTN" EXP_DIR="$PTN/lm"

echo "----"
if [ "$rc" = 0 ]; then echo "ALL GUARD TESTS PASSED"; fi
exit "$rc"
