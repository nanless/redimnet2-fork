#!/usr/bin/env bash
# =============================================================================
# run_firered_rttm_local_2gpu.sh — 在本机两块 A800 上给说话人验证训练音频写 RTTM
#
# 做三件事：
#   1. 开跑前把该检查的都检查一遍（代码版本、模型校验和、样本清单校验和、
#      GPU 是否空闲、分片是否会把同一个文件写两遍）；
#   2. 起 2 块卡，每块卡 PROCESSES_PER_GPU 个进程，每个进程只处理样本清单里
#      属于自己那份的行，合起来正好覆盖整个清单一次；
#   3. 跑完再逐条复查写出来的 RTTM，并把各 rank 的计数加起来核对总数。
#
# 中途被打断也不要紧：可以直接重跑同一个命令，已经写好并且检查通过的
# RTTM 会被跳过，只补没做完的部分。最多自动重试 MAX_PASSES 轮。
#
# 试跑（每条音频大约 0.05 秒，200 条几秒钟就完）：
#   LIMIT_PER_RANK=200 OUTPUT_DIR=/dev/shm/rttm_smoke ./run_firered_rttm_local_2gpu.sh
#
# 正式跑（放在 nohup 或 tmux 里，日志在 ${OUTPUT_DIR}/.runtime/）：
#   OUTPUT_DIR=/root/group-shared/voiceprint/data/wespeaker_data/.../rttm \
#       nohup ./run_firered_rttm_local_2gpu.sh > rttm.log 2>&1 &
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# ---------------------------------------------------------------------------
# 路径与运行参数。全部可以用环境变量覆盖。
# ---------------------------------------------------------------------------
readonly DEFAULT_MANIFEST="/root/group-shared/voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/data.list"
readonly DEFAULT_AUDIO_ROOT="/root/group-shared/voiceprint/data/speech/speaker_verification"
readonly DEFAULT_OUTPUT_DIR="/root/group-shared/voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/rttm_firered"
readonly DEFAULT_FIRERED_ROOT="/root/code/github_repos/FireRedVAD"
readonly DEFAULT_MODEL_DIR="${DEFAULT_FIRERED_ROOT}/pretrained_models/FireRedVAD/VAD"

MANIFEST="${MANIFEST:-${DEFAULT_MANIFEST}}"
AUDIO_ROOT="${AUDIO_ROOT:-${DEFAULT_AUDIO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
FIRERED_ROOT="${FIRERED_ROOT:-${DEFAULT_FIRERED_ROOT}}"
MODEL_DIR="${MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/bin/python}"

NUM_GPUS="${NUM_GPUS:-2}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-2}"
LIMIT_PER_RANK="${LIMIT_PER_RANK:-}"
MAX_PASSES="${MAX_PASSES:-3}"
RTTM_SUFFIX="${RTTM_SUFFIX:-.rttm}"
ROOT_DEPTH="${ROOT_DEPTH:-2}"

# 输出布局：
#   OUTPUT_MODE=beside  写在音频文件旁边（音频目录必须可写）
#   OUTPUT_MODE=mirror  写到 OUTPUT_DIR 下、按音频路径搭出来的镜像树里（默认）
OUTPUT_MODE="${OUTPUT_MODE:-mirror}"

REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
RUN_ORACLE="${RUN_ORACLE:-1}"
ORACLE_ITEMS="${ORACLE_ITEMS:-32}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"

# 期望的总行数。样本清单换了就要跟着改，或者在命令行里重新给一个值。
EXPECTED_TOTAL="${EXPECTED_TOTAL:-11521178}"

ANNOTATOR="${SCRIPT_DIR}/annotate_firered_offline.py"
SELFTEST="${SCRIPT_DIR}/test_firered_rttm.py"

case "${PROCESSES_PER_GPU}" in
    1)
        DEFAULT_BATCH_SIZE=2048; DEFAULT_BATCH_POOL_SIZE=4096
        DEFAULT_MAX_PADDED_FRAMES=2400000; DEFAULT_NUM_WORKERS=24
        DEFAULT_WRITER_THREADS=16 ;;
    2)
        DEFAULT_BATCH_SIZE=1024; DEFAULT_BATCH_POOL_SIZE=2048
        DEFAULT_MAX_PADDED_FRAMES=1200000; DEFAULT_NUM_WORKERS=16
        DEFAULT_WRITER_THREADS=8 ;;
    *)
        echo "[rttm][错误] PROCESSES_PER_GPU 只能是 1 或 2，现在是 ${PROCESSES_PER_GPU}" >&2
        exit 1 ;;
esac
NUM_RANKS="$((NUM_GPUS * PROCESSES_PER_GPU))"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
BATCH_POOL_SIZE="${BATCH_POOL_SIZE:-${DEFAULT_BATCH_POOL_SIZE}}"
MAX_PADDED_FRAMES="${MAX_PADDED_FRAMES:-${DEFAULT_MAX_PADDED_FRAMES}}"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
WRITER_THREADS="${WRITER_THREADS:-${DEFAULT_WRITER_THREADS}}"
WORKER_CHUNKSIZE="${WORKER_CHUNKSIZE:-8}"
PROCESS_WAVE_STAGGER_SECONDS="${PROCESS_WAVE_STAGGER_SECONDS:-12}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RUNTIME_DIR="${OUTPUT_DIR}/.runtime"

fail() {
    echo "[rttm][错误] $*" >&2
    exit 1
}
note() { echo "[rttm] $*"; }

# ---------------------------------------------------------------------------
# 第一步：开跑前的检查
# ---------------------------------------------------------------------------
[[ -x "${PYTHON_BIN}" ]] || fail "找不到可执行的 Python：${PYTHON_BIN}"
[[ -s "${MANIFEST}" ]] || fail "样本清单不存在或为空：${MANIFEST}"
[[ -d "${FIRERED_ROOT}" ]] || fail "FireRedVAD 代码目录不存在：${FIRERED_ROOT}"
[[ -s "${MODEL_DIR}/model.pth.tar" ]] || fail "模型文件不存在：${MODEL_DIR}/model.pth.tar"
[[ -s "${MODEL_DIR}/cmvn.ark" ]] || fail "倒谱均值方差文件不存在：${MODEL_DIR}/cmvn.ark"
[[ -s "${ANNOTATOR}" ]] || fail "标注脚本不存在：${ANNOTATOR}"
[[ "${NUM_RANKS}" =~ ^[1-9][0-9]*$ ]] || fail "rank 数不合法：${NUM_RANKS}"
[[ "${MAX_PASSES}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_PASSES 不合法：${MAX_PASSES}"
[[ "${ROOT_DEPTH}" =~ ^[0-9]+$ ]] || fail "ROOT_DEPTH 不合法：${ROOT_DEPTH}"
[[ "${OUTPUT_MODE}" == "mirror" || "${OUTPUT_MODE}" == "beside" ]] || \
    fail "OUTPUT_MODE 只能是 mirror 或 beside，现在是 ${OUTPUT_MODE}"
if [[ -n "${LIMIT_PER_RANK}" ]]; then
    [[ "${LIMIT_PER_RANK}" =~ ^[1-9][0-9]*$ ]] || fail "LIMIT_PER_RANK 不合法：${LIMIT_PER_RANK}"
fi

# 依赖检查。VAD 这套代码需要 kaldiio，它不在本仓库的 redimnet2 环境里，
# 所以默认用的是 base 环境 /root/miniforge3/bin/python。
"${PYTHON_BIN}" - <<'PY' || fail "Python 依赖不全（需要 torch / soundfile / numpy / scipy / kaldiio）"
import importlib, sys
missing = []
for name in ("torch", "soundfile", "numpy", "scipy", "kaldiio", "kaldi_native_fbank"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name} ({exc})")
if missing:
    print("缺少：" + "; ".join(missing), file=sys.stderr)
    raise SystemExit(1)
import torch
if not torch.cuda.is_available():
    print("torch 看不到 CUDA", file=sys.stderr)
    raise SystemExit(1)
PY

VISIBLE_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
[[ "${VISIBLE_GPUS}" -ge "${NUM_GPUS}" ]] || \
    fail "可见 GPU 数 ${VISIBLE_GPUS} 少于需要的 ${NUM_GPUS}"

if [[ "${REQUIRE_IDLE_GPUS}" == "1" ]]; then
    # 同一台机器可能还有别的训练任务在用卡。这里只检查、不抢卡：
    # 如果某块卡已经被占，就退出并提示先确认那是不是自己的任务。
    mapfile -t GPU_MEMORY_USED < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "${#GPU_MEMORY_USED[@]}" -ge "${NUM_GPUS}" ]] || fail "读不到全部 GPU 的显存占用"
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        used="${GPU_MEMORY_USED[gpu]//[[:space:]]/}"
        [[ "${used}" =~ ^[0-9]+$ ]] || fail "GPU ${gpu} 的显存读数异常：${used}"
        [[ "${used}" -lt 1024 ]] || fail \
            "GPU ${gpu} 已被占用（${used} MiB）——先确认那是不是自己的任务，\
确认无误后可用 REQUIRE_IDLE_GPUS=0 跳过这项检查"
    done
    note "两块 GPU 都空闲"
fi

FIRERED_GIT_SHA="$(git -C "${FIRERED_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
TOOL_GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
MODEL_SHA256="$(sha256sum "${MODEL_DIR}/model.pth.tar" | awk '{print $1}')"
MANIFEST_SHA256="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
MANIFEST_LINES="$(wc -l < "${MANIFEST}" | tr -d ' ')"

if [[ -z "${LIMIT_PER_RANK}" && "${EXPECTED_TOTAL}" != "0" && \
      "${MANIFEST_LINES}" != "${EXPECTED_TOTAL}" ]]; then
    fail "样本清单行数 ${MANIFEST_LINES} 与期望的 ${EXPECTED_TOTAL} 不一致。\
确认清单没被改动后，用 EXPECTED_TOTAL=${MANIFEST_LINES} 重新运行"
fi

mkdir -p "${OUTPUT_DIR}" "${RUNTIME_DIR}"
[[ -w "${OUTPUT_DIR}" ]] || fail "输出目录不可写：${OUTPUT_DIR}"
df -h "${OUTPUT_DIR}" | tail -1

echo "============================================================"
note "样本清单   = ${MANIFEST}"
note "           行数=${MANIFEST_LINES} sha256=${MANIFEST_SHA256}"
note "音频根目录 = ${AUDIO_ROOT}"
note "输出目录   = ${OUTPUT_DIR}  （布局 ${OUTPUT_MODE}）"
note "输出命名   = <音频名>${RTTM_SUFFIX}，另存一个 .<音频名>${RTTM_SUFFIX}.json 记录来源"
note "模型       = ${MODEL_DIR}"
note "            sha256=${MODEL_SHA256}"
note "FireRed 版本 = ${FIRERED_GIT_SHA}"
note "本仓库版本   = ${TOOL_GIT_SHA}"
note "GPU=${NUM_GPUS} 每卡进程=${PROCESSES_PER_GPU} rank 总数=${NUM_RANKS}"
note "每进程 batch=${BATCH_SIZE} pool=${BATCH_POOL_SIZE} 补零上限=${MAX_PADDED_FRAMES}"
note "读音频子进程=${NUM_WORKERS} 写文件线程=${WRITER_THREADS}"
note "每个 rank 上限 = ${LIMIT_PER_RANK:-全部}"
echo "============================================================"

if [[ "${RUN_SELFTEST}" == "1" ]]; then
    note "先跑单元自检（不需要 GPU）"
    "${PYTHON_BIN}" "${SELFTEST}" || fail "单元自检未通过，先修好再跑正式任务"
fi

# ---------------------------------------------------------------------------
# 第二步：公共参数
# ---------------------------------------------------------------------------
common_args=(
    --manifest "${MANIFEST}"
    --audio-root "${AUDIO_ROOT}"
    --firered-root "${FIRERED_ROOT}"
    --model-dir "${MODEL_DIR}"
    --firered-git-sha "${FIRERED_GIT_SHA}"
    --tool-git-sha "${TOOL_GIT_SHA}"
    --manifest-sha256 "${MANIFEST_SHA256}"
    --world-size "${NUM_RANKS}"
    --physical-gpu-count "${NUM_GPUS}"
    --processes-per-gpu "${PROCESSES_PER_GPU}"
    --rttm-suffix "${RTTM_SUFFIX}"
    --root-depth "${ROOT_DEPTH}"
    --num-workers "${NUM_WORKERS}"
    --writer-threads "${WRITER_THREADS}"
    --worker-chunksize "${WORKER_CHUNKSIZE}"
    --batch-size "${BATCH_SIZE}"
    --batch-pool-size "${BATCH_POOL_SIZE}"
    --max-padded-frames "${MAX_PADDED_FRAMES}"
)
if [[ "${OUTPUT_MODE}" == "mirror" ]]; then
    common_args+=(--output-root "${OUTPUT_DIR}")
fi
if [[ -n "${LIMIT_PER_RANK}" ]]; then
    common_args+=(--limit "${LIMIT_PER_RANK}")
fi

export PYTHONPATH="${FIRERED_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS MKL_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# 分片自检：证明各 rank 合起来正好覆盖清单一次，既不重复也不遗漏
# ---------------------------------------------------------------------------
note "检查分片覆盖"
shard_report="$("${PYTHON_BIN}" "${ANNOTATOR}" --mode shard-plan "${common_args[@]}" \
    --rank 0 --manifest-sha256 "${MANIFEST_SHA256}")" || fail "分片检查未通过：${shard_report}"
echo "${shard_report}" > "${RUNTIME_DIR}/shard_plan.json"

# ---------------------------------------------------------------------------
# 重名自检：证明没有两条音频会写到同一个输出文件。
#
# **这一步必须在这里做一次，不能让每个 rank 各做一遍。**
# 检查需要看整份清单（一千多万行），实测单次约 23 分钟、纯 CPU、GPU 全程空闲。
# 之前每个 rank 都在启动时各做一次，4 个 rank 就是 4 倍的空转，
# 日志看起来像"卡死"（GPU 0%、只有一行输出），非常容易被误杀。
# 现在由启动脚本做一次，rank 拿到 `--no-collision-check` 直接开跑。
# ---------------------------------------------------------------------------
note "检查输出路径是否重名（只需做一次，可能要十几分钟）"
if [ "${SKIP_COLLISION_CHECK:-0}" = "1" ]; then
    note "已按要求跳过重名自检（SKIP_COLLISION_CHECK=1）"
else
    collision_report="$("${PYTHON_BIN}" "${ANNOTATOR}" --mode shard-plan "${common_args[@]}" \
        --rank 0 --manifest-sha256 "${MANIFEST_SHA256}" --collision-only)" \
        || fail "重名自检未通过：${collision_report}"
    echo "${collision_report}" > "${RUNTIME_DIR}/collision_global.json"
    "${PYTHON_BIN}" -c '
import json, sys
r = json.loads(sys.argv[1])
print("[rttm] 重名自检：清单 {} 条，输出路径 {} 个，冲突 {} 个".format(
    format(r["records"], ","), format(r["distinct_outputs"], ","),
    r["collision_count"]))
if not r["unique"]:
    for c in r["collisions_head"][:5]:
        print("[rttm]   冲突：{} <- {}".format(c["output"], c["sources"][:3]))
    raise SystemExit(1)
' "${collision_report}" || fail "有两条音频会写到同一个输出文件，已停下"
fi
"${PYTHON_BIN}" -c '
import json, sys
r = json.loads(sys.argv[1])
print("[rttm] 分片：清单 {} 行，{} 个 rank 合计覆盖 {} 行，重复 {} 行，遗漏 {} 行".format(
    r["manifest_records"], r["world_size"], r["covered"],
    r["duplicate_count"], r["missing_count"]))
if not r["consistent"]:
    print("[rttm][错误] 分片有重复或遗漏", file=sys.stderr)
    raise SystemExit(1)
' "${shard_report}" || fail "分片有重复或遗漏，已停止"

# ---------------------------------------------------------------------------
# 第三步：数值对照（batch 推理 vs 逐条推理）
# ---------------------------------------------------------------------------
if [[ "${RUN_ORACLE}" == "1" ]]; then
    note "数值对照：批量推理与逐条推理是否完全一致（用 GPU 0）"
    oracle_args=("${common_args[@]}")
    oracle_limit="${ORACLE_ITEMS}"
    if [[ -n "${LIMIT_PER_RANK}" && "${LIMIT_PER_RANK}" -lt "${oracle_limit}" ]]; then
        oracle_limit="${LIMIT_PER_RANK}"
    fi
    oracle_args+=(--limit "${oracle_limit}")
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ANNOTATOR}" \
        --mode oracle --rank 0 --world-size "${NUM_RANKS}" \
        --physical-gpu-count "${NUM_GPUS}" --processes-per-gpu "${PROCESSES_PER_GPU}" \
        "${oracle_args[@]}" | tee "${RUNTIME_DIR}/oracle.log" || fail "数值对照未通过"
fi

# ---------------------------------------------------------------------------
# 第四步：起进程。两块卡各按「先第 1 轮全部 rank、再第 2 轮全部 rank」的顺序
# 分批启动，后一批错开几秒，避免所有进程同时冲去读同一批音频。
# ---------------------------------------------------------------------------
run_all_ranks() {
    local mode="$1" pass="$2"
    local -a pids=()
    local slot gpu rank
    for ((slot = 0; slot < PROCESSES_PER_GPU; slot++)); do
        for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
            rank="$((slot * NUM_GPUS + gpu))"
            note "启动 mode=${mode} rank=${rank}/${NUM_RANKS} 物理卡=${gpu} 第${slot}批"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ANNOTATOR}" \
                --mode "${mode}" --rank "${rank}" "${common_args[@]}" \
                --no-collision-check \
                >"${RUNTIME_DIR}/${mode}.pass${pass}.rank${rank}.log" 2>&1 &
            pids[rank]="$!"
        done
        if [[ "${mode}" == "infer" && "${slot}" -lt "$((PROCESSES_PER_GPU - 1))" && \
              "${PROCESS_WAVE_STAGGER_SECONDS}" -gt 0 ]]; then
            note "等 ${PROCESS_WAVE_STAGGER_SECONDS} 秒再起下一批"
            sleep "${PROCESS_WAVE_STAGGER_SECONDS}"
        fi
    done
    local failed=0 status
    set +e
    for ((rank = 0; rank < NUM_RANKS; rank++)); do
        wait "${pids[rank]}"
        status=$?
        if [[ "${status}" -ne 0 ]]; then
            failed=1
            echo "[rttm][错误] ${mode} rank=${rank} 退出码 ${status}" >&2
            tail -80 "${RUNTIME_DIR}/${mode}.pass${pass}.rank${rank}.log" >&2 || true
        fi
    done
    set -e
    return "${failed}"
}

# ---------------------------------------------------------------------------
# 第五步：核对各 rank 的计数之和是不是等于应该处理的条数
# ---------------------------------------------------------------------------
validation_totals_ok() {
    local expected="${MANIFEST_LINES}"
    if [[ -n "${LIMIT_PER_RANK}" ]]; then
        expected="$((LIMIT_PER_RANK * NUM_RANKS))"
    fi
    "${PYTHON_BIN}" -c '
import json, pathlib, sys
runtime = pathlib.Path(sys.argv[1]); expected = int(sys.argv[2]); num_ranks = int(sys.argv[3])
summaries = []
for r in range(num_ranks):
    tag = "rank{:05d}-of{:05d}".format(r, num_ranks)
    summaries.append(json.loads((runtime / "validation.{}.json".format(tag)).read_text()))
processed = sum(int(s["processed"]) for s in summaries)
valid = sum(int(s["valid"]) for s in summaries)
invalid = sum(int(s["invalid"]) for s in summaries)
empty = sum(int(s["empty_rttm"]) for s in summaries)
segments = sum(int(s["segments_total"]) for s in summaries)
speech = sum(float(s["speech_seconds"]) for s in summaries)
audio = sum(float(s["audio_seconds"]) for s in summaries)
print("[rttm] 复查合计：处理={} 通过={} 未通过={} 其中空标注={}"
      "（空标注 = 有音频但没检出语音）".format(processed, valid, invalid, empty))
print("[rttm] 语音段共 {} 段，语音合计 {:.1f} 秒 / 音频合计 {:.1f} 秒（占比 {:.2f}%）".format(
    segments, speech, audio, 100 * speech / audio if audio else 0.0))
if not (processed == valid == expected and invalid == 0):
    print("[rttm][错误] 期望 processed=valid={} invalid=0，实际对不上".format(expected),
          file=sys.stderr)
    raise SystemExit(1)
' "${RUNTIME_DIR}" "${expected}" "${NUM_RANKS}"
}

# ---------------------------------------------------------------------------
# 主循环：最多 MAX_PASSES 轮。每轮先补没做完的，再全量复查。
# 中途失败不会删掉已经写好的结果，所以下一轮只补缺口。
# ---------------------------------------------------------------------------
for ((pass = 1; pass <= MAX_PASSES; pass++)); do
    note "第 ${pass}/${MAX_PASSES} 轮：写标注"
    if run_all_ranks infer "${pass}"; then
        note "全部 ${NUM_RANKS} 个 rank 都跑完了，开始逐条复查"
        if run_all_ranks validate "${pass}"; then
            if validation_totals_ok; then
                echo "============================================================"
                note "全部完成：${OUTPUT_DIR}"
                echo "============================================================"
                exit 0
            fi
            echo "[rttm][错误] 各 rank 复查都返回 0，但合计条数对不上" >&2
        fi
    fi
    echo "[rttm][提示] 第 ${pass} 轮没跑完；已经写好的结果都保留着，下一轮只补缺口" >&2
done

fail "已经跑满 MAX_PASSES=${MAX_PASSES} 轮还没完成。请查看 ${RUNTIME_DIR} 下的日志，\
不要删除已有的输出，修好问题后重跑同一条命令即可（会自动跳过已完成的部分）"
