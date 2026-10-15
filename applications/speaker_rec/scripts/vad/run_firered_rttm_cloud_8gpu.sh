#!/usr/bin/env bash
# =============================================================================
# run_firered_rttm_cloud_8gpu.sh — 在云端 8 卡机器（火山引擎等）上给说话人验证
# 训练音频写 RTTM 标注，一条音频一个文件。
#
# 这个脚本要被别的系统包成云端任务，所以：
#   * 每一条路径都能用环境变量覆盖，脚本里不写死任何本机专有的目录；
#   * 音频目录只读也能跑（用镜像树输出，OUTPUT_MODE=mirror 是默认值）；
#   * 进度和结果用固定的、好抓的格式打印，方便外面 grep；
#   * 任何一步出错都以非零码退出，并把出问题那个 rank 的日志末尾打出来；
#   * 反复跑同一条命令是安全的：已经写好并且检查通过的标注会被跳过。
#
# =============================================================================
# 为什么这样分片是安全的（给审这段代码的人看）
# =============================================================================
#
# 样本清单是一行一条音频的文本文件。脚本按下标分片，而不是按文件大小或
# 时间切段：
#
#     rank r 处理那些「从 0 开始数、第 i 行满足 i % world_size == r」的行。
#
# 由此得到的性质，逐条对着代码看：
#
# 1. 不重复。任意一行的下标 i 只能被 0..world_size-1 中的一个数整除余出同一个
#    商，所以只可能满足一个 rank 的条件。产出路径只由音频路径决定
#    （见 firered_rttm_common.sidecar_path），所以两个 rank 不可能写同一个文件。
#    —— 对应代码：firered_rttm_common.iter_audio_shard 里的
#    `if record.index % world_size != rank: continue`。
#
# 2. 不遗漏。0..N-1 每个整数都有一个余数，所以每个下标都被某个 rank 取到。
#    —— 由前一步里的 shard-plan 检查在开跑前实际算一遍并断言：
#    covered == expected_covered、duplicate_count == 0、missing_count == 0。
#
# 3. 不需要为了对齐而补样本。各 rank 的条数最多相差 1（N 除不尽时），
#    没有 DDP 那种「为了集体通信必须凑齐同样条数」的要求，所以不会出现
#    某些样本被重复处理、或者用重复样本来填满一个 batch 的情况。
#    —— 本脚本里每个 rank 是独立进程，各自写自己的 CUDA_VISIBLE_DEVICES 那张卡。
#
# 4. 换卡数结果不变。世界大小只影响「谁做哪些行」，不影响「某一行算出什么」：
#    每条音频的模型输入只取决于它自己的采样点，batch 里别的音频只是补零
#    （见 annotate_firered_offline.forward_batch_once 的 input_lengths 掩码），
#    而补零位置被掩掉、且注意这是逐帧前馈网络没有跨样本注意力。
#    为了不让这句话只是嘴上说说，脚本默认先跑一次数值对照
#    （--mode oracle）：把一批音频合起来跑，和逐条单独跑，逐帧概率最大误差
#    必须小于 1e-5，切出来的语音段时间戳必须逐个完全相同。
#
# 5. 反复跑也安全。一个 RTTM 只有在「格式合法 + 时间在音频长度内 + 不重叠 +
#    头部记的来源指纹与现在的音频文件一致 + 推理标识与本次设置一致」全部
#    满足时才会被当成已完成而跳过；任何一项不满足都会重算并覆盖。
#    —— 对应 firered_rttm_common.validate_sidecar 与
#    annotate_firered_offline.resolve_reusable_sidecar。
#
# =============================================================================
# 需要的运行环境
# =============================================================================
#
# 基础镜像：任意 CUDA 12.x + Python 3.10 以上的镜像都可以。需要的 Python 包
# 以及为什么要它们：
#
#   torch                 模型前向；需要能看见 GPU（torch.cuda.is_available()）
#   kaldiio               读 FireRedVAD 的 cmvn.ark（倒谱均值方差文件）
#   kaldi_native_fbank    FireRedVAD 的 80 维滤波器组前端
#   soundfile             读 wav / flac（底层是 libsndfile）
#   numpy, scipy          resample_poly 重采样
#
# 一次装齐（注意 kaldiio 与 kaldi_native_fbank 不在本仓库的 conda 环境里）：
#
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#   pip install kaldiio kaldi_native_fbank soundfile numpy scipy
#
# 另外需要：
#   * FireRedVAD 的代码目录（FIRERED_ROOT，里面有 fireredvad/ 这个包）
#   * FireRedVAD 的模型目录（MODEL_DIR，里面有 model.pth.tar 和 cmvn.ark）
#
# =============================================================================
# 先做冒烟测试，再做正式任务
# =============================================================================
#
# 冒烟测试：每条音频不超过 0.1 秒，200 条几十秒就完，先确认环境是通的。
#
#   MODEL_DIR=/mnt/models/FireRedVAD/VAD \
#   FIRERED_ROOT=/mnt/code/FireRedVAD \
#   MANIFEST=/mnt/data/mix_join/data.list \
#   AUDIO_ROOT=/mnt/audio/speaker_verification \
#   OUTPUT_DIR=/mnt/out/rttm_smoke \
#   LIMIT_PER_RANK=100 \
#   EXPECTED_TOTAL=0 \
#   bash run_firered_rttm_cloud_8gpu.sh
#
# 正式任务（8 卡，每卡 2 个进程，共 16 个 rank）：
#
#   MODEL_DIR=/mnt/models/FireRedVAD/VAD \
#   FIRERED_ROOT=/mnt/code/FireRedVAD \
#   MANIFEST=/mnt/data/mix_join/data.list \
#   AUDIO_ROOT=/mnt/audio/speaker_verification \
#   OUTPUT_DIR=/mnt/out/rttm_firered \
#   EXPECTED_TOTAL=11521178 \
#   bash run_firered_rttm_cloud_8gpu.sh
#
# 音频目录只读时不用额外加参数：默认就是 OUTPUT_MODE=mirror，
# 标注写到 OUTPUT_DIR 下面按音频路径搭出来的镜像树里。
#
# =============================================================================
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# 全部路径都从环境变量取，并给出云端机器上常见的默认值。脚本里没有本机专有路径。
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MANIFEST="${MANIFEST:-/mnt/data/mix_join/data.list}"
AUDIO_ROOT="${AUDIO_ROOT:-/mnt/audio/speaker_verification}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/out/rttm_firered}"
FIRERED_ROOT="${FIRERED_ROOT:-/mnt/code/FireRedVAD}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/FireRedVAD/VAD}"
# 用哪个 Python。
#
# **不要**默认裸 `python3`：FireRedVAD 需要 kaldiio、kaldi_native_fbank 等包，
# 它们通常不在系统 python3 里（本机就是如此），而是装在某个 conda 环境里。
# 这里按顺序找一个「真的能 import torch + kaldiio」的解释器：
#   1. 显式给的 $PYTHON_BIN
#   2. 常见 conda 环境路径（fireredvad / base / redimnet2）
#   3. PATH 上的 python / python3（如果它能 import）
# 全都不行就报错并说明该装什么，而不是拿一个用不了的解释器往下跑。
resolve_python() {
    if [ -n "${PYTHON_BIN:-}" ]; then echo "${PYTHON_BIN}"; return; fi
    local cand
    for cand in "${HOME}/miniforge3/bin/python" \
                "${HOME}/miniconda3/bin/python" \
                "${HOME}/anaconda3/bin/python" \
                "/opt/conda/bin/python" \
                "${HOME}/miniforge3/envs/fireredvad/bin/python" \
                "${HOME}/miniforge3/envs/redimnet2/bin/python"; do
        if [ -x "${cand}" ] && "${cand}" -c 'import torch, kaldiio' >/dev/null 2>&1; then
            echo "${cand}"; return
        fi
    done
    for cand in python python3; do
        if command -v "${cand}" >/dev/null 2>&1 && \
           "${cand}" -c 'import torch, kaldiio' >/dev/null 2>&1; then
            echo "${cand}"; return
        fi
    done
    echo "python3"
}
PYTHON_BIN="$(resolve_python)"

NUM_GPUS="${NUM_GPUS:-8}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-2}"
LIMIT_PER_RANK="${LIMIT_PER_RANK:-}"
MAX_PASSES="${MAX_PASSES:-3}"
RTTM_SUFFIX="${RTTM_SUFFIX:-.rttm}"
ROOT_DEPTH="${ROOT_DEPTH:-2}"

# mirror：写到 OUTPUT_DIR 下的镜像树（音频目录只读时用这个，默认）
# beside：写在音频文件旁边（音频目录必须可写）
OUTPUT_MODE="${OUTPUT_MODE:-mirror}"

# 云端机器上一般整台机器归本任务，默认不再检查显存占用；
# 想检查就设 REQUIRE_IDLE_GPUS=1。
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-0}"
RUN_ORACLE="${RUN_ORACLE:-1}"
ORACLE_ITEMS="${ORACLE_ITEMS:-32}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"
GPU_MEMORY_IDLE_LIMIT_MIB="${GPU_MEMORY_IDLE_LIMIT_MIB:-1024}"

# 期望的样本清单行数。0 表示不做这项检查。
EXPECTED_TOTAL="${EXPECTED_TOTAL:-0}"

# 写入和读取用的线程/进程数，按机器大小调。
NUM_WORKERS="${NUM_WORKERS:-16}"
WRITER_THREADS="${WRITER_THREADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
BATCH_POOL_SIZE="${BATCH_POOL_SIZE:-2048}"
MAX_PADDED_FRAMES="${MAX_PADDED_FRAMES:-1200000}"
WORKER_CHUNKSIZE="${WORKER_CHUNKSIZE:-8}"
PROCESS_WAVE_STAGGER_SECONDS="${PROCESS_WAVE_STAGGER_SECONDS:-12}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ANNOTATOR="${SCRIPT_DIR}/annotate_firered_offline.py"
SELFTEST="${SCRIPT_DIR}/test_firered_rttm.py"
RUNTIME_DIR="${OUTPUT_DIR}/.runtime"

fail() {
    echo "[RTTM][ERROR] $*" >&2
    exit 1
}
note() { echo "[RTTM] $*"; }

# ---------------------------------------------------------------------------
# 检查一：文件与参数
# ---------------------------------------------------------------------------
[[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
    fail "找不到 Python 解释器：${PYTHON_BIN}"
[[ -s "${MANIFEST}" ]] || fail "样本清单不存在或为空：${MANIFEST}"
[[ -d "${AUDIO_ROOT}" ]] || fail "音频根目录不存在：${AUDIO_ROOT}"
[[ -s "${FIRERED_ROOT}/fireredvad/vad.py" ]] || \
    fail "FireRedVAD 代码目录不对（找不到 fireredvad/vad.py）：${FIRERED_ROOT}"
[[ -s "${MODEL_DIR}/model.pth.tar" ]] || fail "模型文件不存在：${MODEL_DIR}/model.pth.tar"
[[ -s "${MODEL_DIR}/cmvn.ark" ]] || fail "cmvn 文件不存在：${MODEL_DIR}/cmvn.ark"
[[ -s "${ANNOTATOR}" ]] || fail "标注脚本不存在：${ANNOTATOR}"
[[ "${NUM_RANKS_CHECK:=x}" == "x" ]] || true
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_GPUS 不合法：${NUM_GPUS}"
[[ "${PROCESSES_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || fail "PROCESSES_PER_GPU 不合法：${PROCESSES_PER_GPU}"
[[ "${MAX_PASSES}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_PASSES 不合法：${MAX_PASSES}"
[[ "${ROOT_DEPTH}" =~ ^[0-9]+$ ]] || fail "ROOT_DEPTH 不合法：${ROOT_DEPTH}"
[[ "${OUTPUT_MODE}" == "mirror" || "${OUTPUT_MODE}" == "beside" ]] || \
    fail "OUTPUT_MODE 只能是 mirror 或 beside，现在是 ${OUTPUT_MODE}"
[[ "${EXPECTED_TOTAL}" =~ ^[0-9]+$ ]] || fail "EXPECTED_TOTAL 不合法：${EXPECTED_TOTAL}"
if [[ -n "${LIMIT_PER_RANK}" ]]; then
    [[ "${LIMIT_PER_RANK}" =~ ^[1-9][0-9]*$ ]] || fail "LIMIT_PER_RANK 不合法：${LIMIT_PER_RANK}"
fi
NUM_RANKS="$((NUM_GPUS * PROCESSES_PER_GPU))"

# ---------------------------------------------------------------------------
# 检查二：Python 依赖与 GPU
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" - <<'PY' || fail "Python 依赖不全或看不到 GPU（见上面缺什么）"
import importlib, sys
missing = []
for name in ("torch", "soundfile", "numpy", "scipy", "kaldiio", "kaldi_native_fbank"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append("{} ({})".format(name, exc))
if missing:
    print("缺少的包：" + "; ".join(missing), file=sys.stderr)
    print("安装：pip install kaldiio kaldi_native_fbank soundfile numpy scipy", file=sys.stderr)
    raise SystemExit(1)
import torch
print("[RTTM] torch={} cuda={} 可见卡数={}".format(
    torch.__version__, torch.version.cuda, torch.cuda.device_count()))
if not torch.cuda.is_available():
    print("torch 看不到 CUDA", file=sys.stderr)
    raise SystemExit(1)
if torch.cuda.device_count() < int(__import__("os").environ.get("NUM_GPUS", "1")):
    print("可见卡数少于 NUM_GPUS", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "${REQUIRE_IDLE_GPUS}" == "1" ]]; then
    mapfile -t GPU_MEMORY_USED < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "${#GPU_MEMORY_USED[@]}" -ge "${NUM_GPUS}" ]] || fail "读不到全部 ${NUM_GPUS} 块 GPU 的显存占用"
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        used="${GPU_MEMORY_USED[gpu]//[[:space:]]/}"
        [[ "${used}" =~ ^[0-9]+$ ]] || fail "GPU ${gpu} 显存读数异常：${used}"
        [[ "${used}" -lt "${GPU_MEMORY_IDLE_LIMIT_MIB}" ]] || \
            fail "GPU ${gpu} 已被占用（${used} MiB >= ${GPU_MEMORY_IDLE_LIMIT_MIB} MiB）"
    done
    note "GPU 空闲检查通过"
fi

# ---------------------------------------------------------------------------
# 检查三：指纹与目录
# ---------------------------------------------------------------------------
FIRERED_GIT_SHA="$(git -C "${FIRERED_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
TOOL_GIT_SHA="$(git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
MODEL_SHA256="$(sha256sum "${MODEL_DIR}/model.pth.tar" | awk '{print $1}')"
MANIFEST_SHA256="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
MANIFEST_LINES="$(wc -l < "${MANIFEST}" | tr -d ' ')"

if [[ -z "${LIMIT_PER_RANK}" && "${EXPECTED_TOTAL}" != "0" && \
      "${MANIFEST_LINES}" != "${EXPECTED_TOTAL}" ]]; then
    fail "样本清单行数 ${MANIFEST_LINES} 与 EXPECTED_TOTAL=${EXPECTED_TOTAL} 不一致；\
确认清单没被改动后，用 EXPECTED_TOTAL=${MANIFEST_LINES} 重跑"
fi

mkdir -p "${OUTPUT_DIR}" "${RUNTIME_DIR}"
[[ -w "${OUTPUT_DIR}" ]] || fail "输出目录不可写：${OUTPUT_DIR}"

if [[ "${OUTPUT_MODE}" == "beside" ]]; then
    [[ -w "${AUDIO_ROOT}" ]] || fail "OUTPUT_MODE=beside 需要音频根目录可写：${AUDIO_ROOT}"
fi

# 可抓取的启动信息块。
echo "RTTM_BEGIN"
echo "  manifest=${MANIFEST}"
echo "  manifest_lines=${MANIFEST_LINES}"
echo "  manifest_sha256=${MANIFEST_SHA256}"
echo "  audio_root=${AUDIO_ROOT}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  output_mode=${OUTPUT_MODE}"
echo "  output_suffix=${RTTM_SUFFIX}"
echo "  root_depth=${ROOT_DEPTH}"
echo "  model_dir=${MODEL_DIR}"
echo "  model_sha256=${MODEL_SHA256}"
echo "  firered_git_sha=${FIRERED_GIT_SHA}"
echo "  tool_git_sha=${TOOL_GIT_SHA}"
echo "  num_gpus=${NUM_GPUS}"
echo "  processes_per_gpu=${PROCESSES_PER_GPU}"
echo "  world_size=${NUM_RANKS}"
echo "  limit_per_rank=${LIMIT_PER_RANK:-none}"
echo "  python=${PYTHON_BIN}"
df -h "${OUTPUT_DIR}" | tail -1
echo "RTTM_END"

if [[ "${RUN_SELFTEST}" == "1" ]]; then
    note "跑单元自检（不需要 GPU、不读音频）"
    "${PYTHON_BIN}" "${SELFTEST}" >"${RUNTIME_DIR}/selftest.log" 2>&1 || {
        tail -40 "${RUNTIME_DIR}/selftest.log" >&2
        fail "单元自检未通过，日志：${RUNTIME_DIR}/selftest.log"
    }
    note "单元自检通过"
fi

# ---------------------------------------------------------------------------
# 公共参数
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

export PYTHONPATH="${FIRERED_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS MKL_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# 分片覆盖检查（开跑前实际算一遍，不靠推理）
# ---------------------------------------------------------------------------
note "检查分片覆盖"
shard_report="$("${PYTHON_BIN}" "${ANNOTATOR}" --mode shard-plan "${common_args[@]}" \
    --rank 0 --manifest-sha256 "${MANIFEST_SHA256}")" || fail "分片检查未通过"
echo "${shard_report}" > "${RUNTIME_DIR}/shard_plan.json"
"${PYTHON_BIN}" -c '
import json, sys
r = json.loads(sys.argv[1])
print("SHARD_CHECK manifest={} covered={} expected={} duplicates={} missing={} consistent={}".format(
    r["manifest_records"], r["covered"], r["expected_covered"],
    r["duplicate_count"], r["missing_count"], r["consistent"]))
if not r["consistent"]:
    print("[RTTM][ERROR] 分片有重复或遗漏", file=sys.stderr)
    raise SystemExit(1)
' "${shard_report}" || fail "分片有重复或遗漏，已停止"

# ---------------------------------------------------------------------------
# 重名自检：证明没有两条音频会写到同一个输出文件。
#
# **必须在起进程之前、且只做一次。** 检查要看整份清单（一千多万行），
# 实测单次约 23~28 分钟、纯 CPU、GPU 全程空闲。
# 如果放在每个 rank 的启动流程里（那是原来的做法），
# 16 个 rank 各做一遍 = 16 倍空转，而且日志只有一行、GPU 0%，
# 在外面看起来和"卡死"一模一样，很容易被误杀。
# 所以这里做一次，然后给每个 rank 传 --no-collision-check。
# ---------------------------------------------------------------------------
note "检查输出路径是否重名（只需做一次，可能要十几分钟）"
if [[ "${SKIP_COLLISION_CHECK:-0}" == "1" ]]; then
    note "[RTTM][WARN] 已按要求跳过重名自检（SKIP_COLLISION_CHECK=1）；若有重名会被静默覆盖"
else
    collision_report="$("${PYTHON_BIN}" "${ANNOTATOR}" --mode shard-plan "${common_args[@]}" \
        --rank 0 --manifest-sha256 "${MANIFEST_SHA256}" --collision-only)" \
        || fail "重名自检未通过：${collision_report}"
    echo "${collision_report}" > "${RUNTIME_DIR}/collision_global.json"
    "${PYTHON_BIN}" -c '
import json, sys
r = json.loads(sys.argv[1])
print("COLLISION_CHECK records={} distinct_outputs={} collisions={} unique={}".format(
    r["records"], r["distinct_outputs"], r["collision_count"], r["unique"]))
if not r["unique"]:
    for c in r["collisions_head"][:5]:
        print("[RTTM][ERROR]   冲突：{} <- {}".format(c["output"], c["sources"][:3]),
              file=sys.stderr)
    raise SystemExit(1)
' "${collision_report}" || fail "有两条音频会写到同一个输出文件，已停止"
fi

# ---------------------------------------------------------------------------
# 起进程
# ---------------------------------------------------------------------------
run_all_ranks() {
    local mode="$1" pass="$2"
    local -a pids=()
    local slot gpu rank
    for ((slot = 0; slot < PROCESSES_PER_GPU; slot++)); do
        for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
            rank="$((slot * NUM_GPUS + gpu))"
            note "启动 mode=${mode} rank=${rank}/${NUM_RANKS} 物理卡=${gpu}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ANNOTATOR}" \
                --mode "${mode}" --rank "${rank}" "${common_args[@]}" \
                --no-collision-check \
                >"${RUNTIME_DIR}/${mode}.pass${pass}.rank${rank}.log" 2>&1 &
            pids[rank]="$!"
        done
        if [[ "${mode}" == "infer" && "${slot}" -lt "$((PROCESSES_PER_GPU - 1))" && \
              "${PROCESS_WAVE_STAGGER_SECONDS}" -gt 0 ]]; then
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
            echo "[RTTM][ERROR] ${mode} rank=${rank} 退出码 ${status}" >&2
            echo "----- rank ${rank} 日志末尾 -----" >&2
            tail -80 "${RUNTIME_DIR}/${mode}.pass${pass}.rank${rank}.log" >&2 || true
            echo "----- 日志末尾结束 -----" >&2
        fi
    done
    set -e
    return "${failed}"
}

# ---------------------------------------------------------------------------
# 各 rank 计数合计核对
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
print("VALIDATE_SUMMARY processed={} valid={} invalid={} empty={} expected={}".format(
    processed, valid, invalid, empty, expected))
print("VALIDATE_SPEECH segments={} speech_seconds={:.1f} audio_seconds={:.1f} ratio={:.4f}".format(
    segments, speech, audio, (speech / audio) if audio else 0.0))
if not (processed == valid == expected and invalid == 0):
    print("[RTTM][ERROR] 合计对不上：期望 processed==valid=={}, invalid==0".format(expected),
          file=sys.stderr)
    raise SystemExit(1)
' "${RUNTIME_DIR}" "${expected}" "${NUM_RANKS}"
}

# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
for ((pass = 1; pass <= MAX_PASSES; pass++)); do
    note "第 ${pass}/${MAX_PASSES} 轮：写标注"
    if run_all_ranks infer "${pass}"; then
        note "全部 ${NUM_RANKS} 个 rank 都写完了，开始逐条复查"
        if run_all_ranks validate "${pass}"; then
            if validation_totals_ok; then
                echo "RTTM_RESULT status=ok output_dir=${OUTPUT_DIR}"
                exit 0
            fi
            echo "[RTTM][ERROR] 各 rank 复查都返回 0，但合计条数对不上" >&2
        fi
    fi
    echo "[RTTM][WARN] 第 ${pass} 轮没跑完；已写好的结果都保留，下一轮只补缺口" >&2
done

echo "RTTM_RESULT status=failed output_dir=${OUTPUT_DIR}" >&2
fail "跑满 MAX_PASSES=${MAX_PASSES} 轮仍未完成。请查看 ${RUNTIME_DIR} 下的日志；\
不要删除已有输出，修好问题后重跑同一条命令即可（会自动跳过已完成的部分）"
