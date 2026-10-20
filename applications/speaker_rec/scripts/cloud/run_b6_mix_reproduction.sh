#!/usr/bin/env bash
# 三语料 B6：八卡短测，或在短测成功后运行正式主训练、大间隔微调与评测。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/redimnet2/bin/python}"
MODE="${MODE:?必须明确 MODE=smoke 或 MODE=formal}"
RUN_ROOT="${RUN_ROOT:?必须指定本次任务的持久输出 RUN_ROOT}"
SOURCE_CHECKSUM_FILE="${SOURCE_CHECKSUM_FILE:?必须指定冻结源码的 SHA-256 校验文件}"
DATA_AUDIT_FILE="${DATA_AUDIT_FILE:?必须指定已完成完整数据检查的 data_audit.json}"
[[ "$MODE" == smoke || "$MODE" == formal ]] || { echo 'MODE 只接受 smoke 或 formal' >&2; exit 2; }
[[ "$RUN_ROOT" == /* && "$RUN_ROOT" != / ]] || { echo 'RUN_ROOT 必须是专属绝对目录' >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -f "$SOURCE_CHECKSUM_FILE" && -f "$DATA_AUDIT_FILE" ]] || { echo 'Python、源码校验文件或数据检查报告不存在' >&2; exit 2; }
# 专用流程不接受外部训练参数，防止平台环境悄悄改变正式配方。
for key in NUM_EPOCHS SAMPLE_NUM_PER_EPOCH BATCH_SIZE SAVE_EPOCH_INTERVAL RECIPE MODEL_CKPT STAGE EXP_DIR PTN_EXP_DIR REPRO NUM_GPUS RECIPE_DATA_PREFIX VOX1_TEST_DIR EXPECT_TRAIN_LINES; do
    if [[ -v "$key" ]]; then echo "专用流程拒绝外部训练覆盖: $key" >&2; exit 2; fi
done
export REPO_ROOT PYTHON_BIN MODE
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/applications/speaker_rec"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export DATA_ROOT=/root/group-shared/voiceprint/data/wespeaker_data
export SHARED_DATA_MOUNT="${SHARED_DATA_MOUNT:-/shared/group-shared}"
# 禁止继承跳过检查的开关。
export SKIP_PREFLIGHT=0 SKIP_DATA_LIST_CHECK=0 SKIP_UTT2SPK_CHECK=0 SKIP_AUG_CHECK=0
export SKIP_VOX1_CHECK=0 SKIP_GPU_CHECK=0 SKIP_DISK_CHECK=0 SKIP_LOCK_CHECK=0
cd "$REPO_ROOT"
RUN_ROOT="$($PYTHON_BIN - "$RUN_ROOT" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys
p, repo = map(lambda s: Path(s).resolve(), sys.argv[1:])
if p == repo or p in repo.parents:
    raise SystemExit('RUN_ROOT 不能是代码目录或其祖先目录')
print(p)
PY
)"
export RUN_ROOT
mkdir -p "$RUN_ROOT/checks"
exec 9>"$RUN_ROOT/.reproduction.lock"
flock -n 9 || { echo "已有任务占用 $RUN_ROOT" >&2; exit 3; }
ATTEMPT="$(date -u +%Y%m%dT%H%M%S)-$$"
ATTEMPT_DIR="$RUN_ROOT/checks/$ATTEMPT"
mkdir -p "$ATTEMPT_DIR"
export ATTEMPT_DIR
exec > >(tee -a "$RUN_ROOT/reproduction.log") 2>&1
trap 'rc=$?; echo "REPRODUCTION_FAILED mode=$MODE exit=$rc line=$LINENO" >&2; exit "$rc"' ERR
SOURCE_CHECKSUM_FILE="$($PYTHON_BIN -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$SOURCE_CHECKSUM_FILE")"
SOURCE_CHECKSUM_SHA256="$(sha256sum "$SOURCE_CHECKSUM_FILE" | awk '{print $1}')"
export SOURCE_CHECKSUM_FILE SOURCE_CHECKSUM_SHA256
DATA_AUDIT_SHA256="$(sha256sum "$DATA_AUDIT_FILE" | awk '{print $1}')"
OFFICIAL_REFERENCE="${OFFICIAL_REFERENCE:-/root/code/github_repos/redimnet2-fork/checkpoints/pretrained/b6-vb2+vox2+cnc2_v0-lm.pt}"
[[ -f "$OFFICIAL_REFERENCE" && -r "$OFFICIAL_REFERENCE" ]] || { echo "官方参考权重不可读: $OFFICIAL_REFERENCE" >&2; exit 2; }
OFFICIAL_REFERENCE="$($PYTHON_BIN -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$OFFICIAL_REFERENCE")"
OFFICIAL_REFERENCE_SHA256="$(sha256sum "$OFFICIAL_REFERENCE" | awk '{print $1}')"
export DATA_AUDIT_SHA256 OFFICIAL_REFERENCE OFFICIAL_REFERENCE_SHA256
PTN_RECIPE="$REPO_ROOT/applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/official_repro_ptn.yaml"
LM_RECIPE="$REPO_ROOT/applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/official_repro_lm.yaml"
export PTN_RECIPE LM_RECIPE
LAUNCHER="$SCRIPT_DIR/train_b6_mix_cloud_8gpu.sh"
ACTIVE_PTN_RECIPE="$PTN_RECIPE"
ACTIVE_LM_RECIPE="$LM_RECIPE"
if [[ "$MODE" == smoke ]]; then
    ACTIVE_PTN_RECIPE="$RUN_ROOT/checks/smoke_recipes/ptn.yaml"
    ACTIVE_LM_RECIPE="$RUN_ROOT/checks/smoke_recipes/lm.yaml"
fi
export ACTIVE_PTN_RECIPE ACTIVE_LM_RECIPE
verify_source() {
    [[ "$(sha256sum "$SOURCE_CHECKSUM_FILE" | awk '{print $1}')" == "$SOURCE_CHECKSUM_SHA256" ]] || {
        echo '源码校验文件在运行期间发生变化' >&2; return 1;
    }
    sha256sum -c "$SOURCE_CHECKSUM_FILE" >> "$ATTEMPT_DIR/source_check.log"
}
verify_source
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
import redimnet2
repo = Path(os.environ['REPO_ROOT']).resolve()
loaded = Path(redimnet2.__file__).resolve()
if repo not in loaded.parents:
    raise SystemExit(f'导入了错误版本的模型代码: {loaded}')
root = Path(os.environ['RUN_ROOT'])
identity = root / 'run_identity.json'
value = {'mode': os.environ['MODE'], 'run_root': str(root),
         'source_checksum_sha256': os.environ['SOURCE_CHECKSUM_SHA256']}
if identity.exists() and json.loads(identity.read_text()) != value:
    raise SystemExit('现有输出目录属于不同模式或源码，拒绝混用')
if os.environ['MODE'] == 'smoke' and (root / 'ptn/models/model_1.pt').exists():
    raise SystemExit('短测目录已有训练权重；重新短测请使用新的 RUN_ROOT')
identity.write_text(json.dumps(value, indent=2) + '\n')
if os.environ['MODE'] == 'formal':
    smoke_path = Path(os.environ.get('SMOKE_SUCCESS_FILE', '')).resolve(strict=True)
    if not smoke_path.is_file():
        raise SystemExit('正式任务必须指定 SMOKE_SUCCESS_FILE')
    smoke = json.loads(smoke_path.read_text())
    if not smoke.get('success') or smoke.get('mode') != 'smoke':
        raise SystemExit('八卡短测尚未通过')
    scaler_audit = smoke.get('grad_scaler_resume_audit', {})
    if not scaler_audit.get('success') or scaler_audit.get('grad_scaler_restore_messages') != 8:
        raise SystemExit('短测缺少八个进程的 GradScaler 恢复验收')
    proofs = smoke.get('smoke_recipe_proofs', {})
    if not all(proofs.get(stage, {}).get('ok') for stage in ['ptn', 'lm']):
        raise SystemExit('短测缺少与正式学习率一致的时间尺度检查')
    if smoke.get('source_checksum_sha256') != value['source_checksum_sha256']:
        raise SystemExit('短测源码与正式任务源码不同，必须重新短测')
    if smoke.get('data_audit_sha256') != os.environ['DATA_AUDIT_SHA256']:
        raise SystemExit('短测与正式任务采用了不同的完整数据检查报告')
    if smoke.get('official_reference_sha256') != os.environ['OFFICIAL_REFERENCE_SHA256']:
        raise SystemExit('短测与正式任务采用了不同的官方参考权重')
    if Path(smoke['run_root']).resolve() == root:
        raise SystemExit('短测与正式任务必须使用不同输出目录')
    (root / 'checks/accepted_smoke_success.json').write_text(json.dumps(smoke, ensure_ascii=False, indent=2) + '\n')
print('导入源码检查通过:', loaded)
PY
bash "$SCRIPT_DIR/prepare_b6_cloud_paths.sh"
# 验证完整数据检查报告和四张表的哈希。无需重新扫描全部训练音频。
"$PYTHON_BIN" - "$DATA_AUDIT_FILE" <<'PY'
import hashlib, json, os
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text())
if not report.get('ok') or not report.get('lm_speakers_subset_of_ptn'):
    raise SystemExit('完整数据检查报告没有通过')
root = Path(os.environ['DATA_ROOT'])
checks = {}
for stage, directory in [('ptn', 'mix_join'), ('lm', 'mix_join_lm')]:
    for filename in ['data.list', 'utt2spk']:
        path = root / 'vox2_vb2_cnc2_v0' / directory / filename
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != report[stage]['sha256'][filename]:
            raise SystemExit(f'训练表已变化: {path}')
        checks[str(path)] = h.hexdigest()
(Path(os.environ['ATTEMPT_DIR']) / 'training_table_hashes.json').write_text(json.dumps(checks, indent=2) + '\n')
print('四张完整训练表的 SHA-256 与已完成检查一致')
PY
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f'要求恰好八张可见 GPU，当前为 {torch.cuda.device_count()}')
print('八张 GPU 可见:', [torch.cuda.get_device_name(i) for i in range(8)])
PY
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    "$SCRIPT_DIR/probe_b6_ddp.py" --output "$ATTEMPT_DIR/nccl_probe.json"
if [[ "$MODE" == smoke ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/check_b6_mix_data.py" --only-paths --data-root "$DATA_ROOT" \
        --output "$RUN_ROOT/checks/cloud_path_check.json"
else
    "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
smoke = json.loads(Path(os.environ['SMOKE_SUCCESS_FILE']).read_text())
p = Path(smoke['run_root']) / 'checks/cloud_path_check.json'
r = json.loads(p.read_text())
if not r.get('ok'):
    raise SystemExit('短测的云端路径检查报告没有通过')
# 正式任务也实际访问代表性音频，防止沿用报告但没有挂载同一套盘。
paths = [p for v in r['augmentation'].values() for p in v['sampled_audio']]
paths += [p for v in r['evaluation'].values() for p in v.get('sampled_audio', [])]
import soundfile as sf
for item in paths:
    info = sf.info(item['path'])
    if info.samplerate != 16000 or info.channels != 1 or info.frames != item['frames']:
        raise SystemExit(f'云端音频与短测数据不同: {item["path"]}')
print('正式任务复核短测数据路径通过，文件数:', len(paths))
PY
fi
if [[ "$MODE" == smoke ]]; then
    mkdir -p "$RUN_ROOT/checks/smoke_recipes"
    "$PYTHON_BIN" "$SCRIPT_DIR/make_b6_smoke_recipe.py" --recipe "$PTN_RECIPE" --stage ptn \
        --sample-num-per-epoch 24576 --num-gpus 8 --output "$ACTIVE_PTN_RECIPE" \
        | tee "$ATTEMPT_DIR/ptn_smoke_recipe_proof.json"
    "$PYTHON_BIN" "$SCRIPT_DIR/make_b6_smoke_recipe.py" --recipe "$LM_RECIPE" --stage lm \
        --sample-num-per-epoch 12288 --num-gpus 8 --output "$ACTIVE_LM_RECIPE" \
        | tee "$ATTEMPT_DIR/lm_smoke_recipe_proof.json"
    "$PYTHON_BIN" - <<'PYPROOF'
import json, os
from pathlib import Path
for stage, required in [('ptn', 64), ('lm', 32)]:
    proof = json.loads((Path(os.environ['ATTEMPT_DIR']) / f'{stage}_smoke_recipe_proof.json').read_text())
    if not proof.get('ok') or proof.get('proof', {}).get('steps_checked', 0) < required:
        raise SystemExit(f'{stage} 短测时间尺度证明不完整')
    if proof['proof'].get('max_abs_lr_difference', float('inf')) > 1e-12:
        raise SystemExit(f'{stage} 短测学习率与正式训练不同')
print('短测学习率逐点检查通过，PTN连续64步、LM32步均与正式训练一致')
PYPROOF
fi

check_stage() {
    local stage="$1" epoch="$2" stdout_log="$3" label="$4"
    "$PYTHON_BIN" - "$RUN_ROOT/$stage" "$epoch" "$stdout_log" "$ATTEMPT_DIR/$label.json" <<'PY'
import datetime, json, math, re, statistics, sys
from pathlib import Path
import torch
torch.set_num_threads(4)
exp, epoch, stdout_path, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
text = stdout_path.read_text(errors='replace')
if 'Traceback (most recent call last)' in text or 'could NOT restore optimizer' in text:
    raise SystemExit('训练日志包含异常或优化器恢复失败')
train_log = (exp / 'train.log').read_text(errors='replace')
rows = []
for line in train_log.splitlines():
    fields = line.split('|')
    if len(fields) < 8:
        continue
    try:
        numbers = [float(x.strip()) for x in fields[1:7]]
    except ValueError:
        continue
    if int(numbers[0]) != epoch:
        continue
    if not all(math.isfinite(v) for v in numbers):
        raise SystemExit(f'训练统计出现非有限数值: {line}')
    stamp = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line)
    timestamp = datetime.datetime.strptime(stamp.group(), '%Y-%m-%d %H:%M:%S,%f').timestamp() if stamp else None
    rows.append({'timestamp': timestamp, 'epoch': int(numbers[0]), 'step': int(numbers[1]), 'lr': numbers[2],
                 'margin': numbers[3], 'loss': numbers[4], 'accuracy': numbers[5]})
if not rows:
    raise SystemExit(f'没有找到第 {epoch} 轮的有限训练损失记录')
checkpoint = exp / 'models' / f'model_{epoch}.pt'
try:
    obj = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
except RuntimeError as error:
    if 'mmap' not in str(error):
        raise
    obj = torch.load(checkpoint, map_location='cpu', weights_only=False)
if obj.get('epoch') != epoch or not isinstance(obj.get('model'), dict) or not obj['model']:
    raise SystemExit('训练检查点的轮数或模型内容错误')
count = 0
for key, value in obj['model'].items():
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not torch.isfinite(value).all().item():
            raise SystemExit(f'模型参数包含非有限数值: {key}')
        count += 1
optimizer = obj.get('optimizer')
if not isinstance(optimizer, dict) or not optimizer.get('state') or not optimizer.get('param_groups'):
    raise SystemExit('训练检查点没有有效的优化器状态')
windows = []
for before, after in zip(rows, rows[1:]):
    if before['timestamp'] is not None and after['timestamp'] is not None and after['step'] > before['step'] and after['timestamp'] > before['timestamp']:
        windows.append({'start_step': before['step'], 'end_step': after['step'], 'elapsed_seconds': after['timestamp'] - before['timestamp'],
                        'seconds_per_step': (after['timestamp'] - before['timestamp']) / (after['step'] - before['step'])})
timing = {'windows': windows, 'steps_in_full_ptn': 1800120, 'includes_validation_or_checkpoint_time': False}
if windows:
    timing['median_seconds_per_step'] = statistics.median(v['seconds_per_step'] for v in windows)
    timing['estimated_full_ptn_seconds_without_validation_or_checkpoint'] = 1800120 * timing['median_seconds_per_step']
scaler = obj.get('scaler')
required_scaler = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
if not isinstance(scaler, dict) or not required_scaler.issubset(scaler):
    raise SystemExit('训练检查点缺少有效的 GradScaler 状态')
if not all(isinstance(scaler[k], (int, float)) and math.isfinite(scaler[k]) for k in required_scaler):
    raise SystemExit('GradScaler 状态包含非有限或非数值内容')
if not (scaler['scale'] > 0 and scaler['growth_factor'] > 1 and 0 < scaler['backoff_factor'] < 1
        and type(scaler['growth_interval']) is int and scaler['growth_interval'] > 0
        and type(scaler['_growth_tracker']) is int and 0 <= scaler['_growth_tracker'] < scaler['growth_interval']):
    raise SystemExit('GradScaler 状态参数范围不正确')
optimizer_tensors = 0
for state in optimizer['state'].values():
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all().item():
                raise SystemExit(f'优化器状态包含非有限数值: {key}')
            optimizer_tensors += 1
result = {'success': True, 'timing': timing, 'checkpoint': str(checkpoint), 'epoch': epoch,
          'scaler': scaler, 'finite_optimizer_tensors': optimizer_tensors,
          'finite_model_tensors': count, 'optimizer_state_items': len(optimizer['state']),
          'training_rows': rows, 'stdout_log': str(stdout_path)}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('检查点与有限训练损失检查通过:', checkpoint)
PY
}
run_stage() {
    local stage="$1" recipe="$2" logfile="$3"
    shift 3
    verify_source
    env REPRO=1 NUM_GPUS=8 STAGE="$stage" RECIPE="$recipe" EXP_DIR="$RUN_ROOT/$stage" \
        PYTHON_BIN="$PYTHON_BIN" REPO_ROOT="$REPO_ROOT" DATA_ROOT="$DATA_ROOT" \
        "$@" bash "$LAUNCHER" 2>&1 | tee "$logfile"
}
if [[ "$MODE" == smoke ]]; then
    echo '短测每轮32次更新，学习率时间尺度经逐点验证；轮数覆盖仅用于启动、保存、恢复及阶段切换检查。'
    run_stage ptn "$ACTIVE_PTN_RECIPE" "$ATTEMPT_DIR/ptn_first.stdout.log" \
        NUM_EPOCHS=1 SAMPLE_NUM_PER_EPOCH=24576 BATCH_SIZE=96 SAVE_EPOCH_INTERVAL=1
    check_stage ptn 1 "$ATTEMPT_DIR/ptn_first.stdout.log" ptn_first
    run_stage ptn "$ACTIVE_PTN_RECIPE" "$ATTEMPT_DIR/ptn_resume.stdout.log" \
        NUM_EPOCHS=2 SAMPLE_NUM_PER_EPOCH=24576 BATCH_SIZE=96 SAVE_EPOCH_INTERVAL=1
    check_stage ptn 2 "$ATTEMPT_DIR/ptn_resume.stdout.log" ptn_resume
    "$PYTHON_BIN" - "$ATTEMPT_DIR/ptn_resume.stdout.log" "$RUN_ROOT/ptn/train.log" <<'PY'
from pathlib import Path
import re, sys
text = Path(sys.argv[1]).read_text(errors='replace')
train = Path(sys.argv[2]).read_text(errors='replace')
if 'TRAIN_START epoch=2 ' not in text or not re.search(r'start_epoch:\s*2\b', train):
    raise SystemExit('没有证据证明实际从第 2 轮恢复')
if 'resume: restored optimizer state from' not in text:
    raise SystemExit('没有证据证明实际恢复了优化器状态')
print('短测恢复验证通过: start_epoch=2，优化器状态已恢复')
PY
    "$PYTHON_BIN" - "$ATTEMPT_DIR" "$RUN_ROOT/checks/grad_scaler_resume_audit.json" <<'PYSCALER'
import datetime, json, os, sys
from pathlib import Path
attempt, output = map(Path, sys.argv[1:])
text = (attempt / 'ptn_resume.stdout.log').read_text(errors='replace')
restored = text.count('resume: restored GradScaler state from')
optimizer_restored = text.count('resume: restored optimizer state from')
if restored != 8 or optimizer_restored != 8:
    raise SystemExit(f'恢复日志不完整: GradScaler={restored}/8, optimizer={optimizer_restored}/8')
if 'could NOT restore GradScaler' in text or 'could NOT restore optimizer' in text:
    raise SystemExit('存在恢复 GradScaler 或优化器失败的日志')
before = json.loads((attempt / 'ptn_first.json').read_text())
after = json.loads((attempt / 'ptn_resume.json').read_text())
if not before.get('success') or not after.get('success') or before['epoch'] != 1 or after['epoch'] != 2:
    raise SystemExit('缺少第一轮及恢复轮的有效检查点证据')
report = {'success': True, 'grad_scaler_restore_messages': restored,
          'optimizer_restore_messages': optimizer_restored,
          'model_1_scaler': before['scaler'], 'model_2_scaler': after['scaler'],
          'saved_scaler_states_valid_and_finite': True, 'start_epoch': 2,
          'source_checksum_sha256': os.environ['SOURCE_CHECKSUM_SHA256'],
          'stdout_log': str(attempt / 'ptn_resume.stdout.log'),
          'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
temporary = output.with_name(output.name + '.tmp')
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, output)
print('八进程 GradScaler 与优化器恢复检查通过:', output)
PYSCALER
    ptn_epoch=2
else
    run_stage ptn "$ACTIVE_PTN_RECIPE" "$ATTEMPT_DIR/ptn.stdout.log"
    check_stage ptn 120 "$ATTEMPT_DIR/ptn.stdout.log" ptn_final
    [[ -f "$RUN_ROOT/ptn/models/model_115.pt" && -f "$RUN_ROOT/ptn/models/model_120.pt" ]]
    ptn_epoch=115
fi
"$PYTHON_BIN" "$SCRIPT_DIR/check_b6_lm_init.py" \
    --checkpoint "$RUN_ROOT/ptn/models/model_$ptn_epoch.pt" \
    --ptn-spk2id "$RUN_ROOT/ptn/spk2id.json" \
    --lm-label "$DATA_ROOT/vox2_vb2_cnc2_v0/mix_join_lm/utt2spk" \
    --ptn-recipe "$PTN_RECIPE" --expected-epoch "$ptn_epoch" \
    | tee "$ATTEMPT_DIR/lm_initialization.json"
if [[ "$MODE" == smoke ]]; then
    run_stage lm "$ACTIVE_LM_RECIPE" "$ATTEMPT_DIR/lm.stdout.log" \
        PTN_EXP_DIR="$RUN_ROOT/ptn" MODEL_CKPT="$RUN_ROOT/ptn/models/model_2.pt" \
        NUM_EPOCHS=1 SAMPLE_NUM_PER_EPOCH=12288 BATCH_SIZE=48 SAVE_EPOCH_INTERVAL=1
else
    run_stage lm "$ACTIVE_LM_RECIPE" "$ATTEMPT_DIR/lm.stdout.log" \
        PTN_EXP_DIR="$RUN_ROOT/ptn" MODEL_CKPT="$RUN_ROOT/ptn/models/model_115.pt"
fi
check_stage lm 1 "$ATTEMPT_DIR/lm.stdout.log" lm_final
"$PYTHON_BIN" - "$ATTEMPT_DIR/lm.stdout.log" <<'PYLM'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(errors='replace')
# 同阶段 LM 恢复允许恢复自身优化器；首次 LM 必须只继承 PTN 模型。
if 'TRAIN_START epoch=1 ' in text:
    marker = 'cold start: model_ckpt loads model parameters only; optimizer/scaler start fresh'
    if marker not in text or 'resume: restored optimizer state from' in text or 'resume: restored GradScaler state from' in text:
        raise SystemExit('首次微调没有明确使用全新优化器和混合精度状态')
print('微调初始化日志检查通过')
PYLM
verify_source
if [[ "$MODE" == formal ]]; then
    RUN_ROOT="$RUN_ROOT" PYTHON_BIN="$PYTHON_BIN" REPO_ROOT="$REPO_ROOT" \
        bash "$SCRIPT_DIR/evaluate_b6_mix_reproduction.sh"
fi
"$PYTHON_BIN" - <<'PY'
import datetime, json, os
from pathlib import Path
import torch
root, attempt = Path(os.environ['RUN_ROOT']), Path(os.environ['ATTEMPT_DIR'])
mode = os.environ['MODE']
probe = json.loads((attempt / 'nccl_probe.json').read_text())
if not probe.get('success') or probe.get('world_size') != 8:
    raise SystemExit('八卡通信检查未通过')
checks = {p.stem: json.loads(p.read_text()) for p in attempt.glob('*.json')
          if p.stem in {'ptn_first', 'ptn_resume', 'ptn_final', 'lm_final', 'lm_initialization'}}
result = {'success': True, 'mode': mode, 'run_root': str(root),
          'source_checksum_sha256': os.environ['SOURCE_CHECKSUM_SHA256'],
          'source_checksum_file': os.environ['SOURCE_CHECKSUM_FILE'],
          'data_audit_sha256': os.environ['DATA_AUDIT_SHA256'],
          'official_reference': os.environ['OFFICIAL_REFERENCE'],
          'official_reference_sha256': os.environ['OFFICIAL_REFERENCE_SHA256'],
          'gpu_names': [r['gpu_name'] for r in probe['ranks']], 'world_size': 8,
          'python': probe['python'], 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'formal_ptn_recipe': os.environ['PTN_RECIPE'], 'formal_lm_recipe': os.environ['LM_RECIPE'],
          'checkpoint_checks': checks, 'attempt_directory': str(attempt),
          'shortened_training_for_operational_checks_only': mode == 'smoke',
          'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
if mode == 'smoke':
    result['grad_scaler_resume_audit'] = json.loads((root / 'checks/grad_scaler_resume_audit.json').read_text())
    result['smoke_recipe_proofs'] = {stage: json.loads((attempt / f'{stage}_smoke_recipe_proof.json').read_text()) for stage in ['ptn', 'lm']}
    result['active_ptn_recipe'] = os.environ['ACTIVE_PTN_RECIPE']
    result['active_lm_recipe'] = os.environ['ACTIVE_LM_RECIPE']
    result['smoke_overrides'] = {'ptn': {'sample_num_per_epoch': 24576, 'batch_size_per_gpu': 96,
                                        'first_num_epochs': 1, 'resume_num_epochs': 2},
                                 'lm': {'sample_num_per_epoch': 12288, 'batch_size_per_gpu': 48, 'num_epochs': 1}}
output = root / f'{mode}_success.json'
temporary = output.with_name(output.name + '.tmp')
temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, output)
print('REPRODUCTION_SUCCESS', output)
PY
