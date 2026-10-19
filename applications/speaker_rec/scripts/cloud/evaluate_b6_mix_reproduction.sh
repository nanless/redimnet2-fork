#!/usr/bin/env bash
# Evaluate the completed three-corpus B6 LM and a frozen official reference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
: "${RUN_ROOT:?Set RUN_ROOT to this reproduction experiment directory}"
PYTHON_BIN="${PYTHON_BIN:-${REDIMNET2_PYTHON:-python3}}"
OFFICIAL_REFERENCE="${OFFICIAL_REFERENCE:-/root/code/github_repos/redimnet2-fork/checkpoints/pretrained/b6-vb2+vox2+cnc2_v0-lm.pt}"
VOX1_ROOT="${VOX1_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_verification/voxceleb1}"
VOICES_ROOT="${VOICES_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_verification/voices}"
CNCELEB_ROOT="${CNCELEB_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_verification/cnceleb}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

EVAL_DIR="$RUN_ROOT/evaluation"
TRAIN_CKPT="$RUN_ROOT/lm/models/model_1.pt"
TRAIN_CONFIG="$RUN_ROOT/lm/config.yaml"
PUBLISHED="$RUN_ROOT/published/b6-vb2+vox2+cnc2_v0-lm.pt"
REFERENCE="$RUN_ROOT/reference/official-b6-vb2+vox2+cnc2_v0-lm.pt"
for path in "$TRAIN_CKPT" "$TRAIN_CONFIG" "$OFFICIAL_REFERENCE"; do
    [[ -f "$path" ]] || { echo "Required file missing: $path" >&2; exit 1; }
done
mkdir -p "$EVAL_DIR" "$(dirname "$PUBLISHED")" "$(dirname "$REFERENCE")"
exec 9>"$EVAL_DIR/.evaluation.lock"
flock -n 9 || { echo "Another evaluation is using $EVAL_DIR" >&2; exit 1; }
rm -f "$EVAL_DIR/summary.json" "$EVAL_DIR/summary.csv"

# Check or create the fixed reference without modifying the source checkpoint.
REFERENCE_SHA="$(sha256sum "$OFFICIAL_REFERENCE" | awk '{print $1}')"
if [[ -e "$REFERENCE" ]]; then
    [[ "$(sha256sum "$REFERENCE" | awk '{print $1}')" == "$REFERENCE_SHA" ]] || {
        echo "Existing official reference has a different SHA256: $REFERENCE" >&2; exit 1;
    }
else
    ref_temp="$REFERENCE.tmp.$$"
    cp "$OFFICIAL_REFERENCE" "$ref_temp"
    [[ "$(sha256sum "$ref_temp" | awk '{print $1}')" == "$REFERENCE_SHA" ]] || {
        rm -f "$ref_temp"; echo "Official reference copy differs" >&2; exit 1;
    }
    mv "$ref_temp" "$REFERENCE"
fi
printf '%s  %s\n' "$REFERENCE_SHA" "$REFERENCE" > "$RUN_ROOT/reference/SHA256SUMS"

# Export once. A resumed invocation may reuse only the same checkpoint/config bytes.
if [[ ! -e "$PUBLISHED" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/export_checkpoint.py" \
        --checkpoint "$TRAIN_CKPT" --training-config "$TRAIN_CONFIG" --output "$PUBLISHED"
fi
"$PYTHON_BIN" - "$TRAIN_CKPT" "$TRAIN_CONFIG" "$PUBLISHED" <<'PY'
import hashlib, sys
from pathlib import Path
import torch

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

source, config, output = map(Path, sys.argv[1:])
info = torch.load(output, map_location='cpu', weights_only=False).get('export_info', {})
for key, path in [('source_checkpoint_sha256', source), ('source_training_config_sha256', config)]:
    if info.get(key) != digest(path):
        raise SystemExit(f'Refusing to reuse exported checkpoint: {key} does not match {path}')
print('Exported checkpoint source/config SHA256 verified:', output)
PY

IFS=',' read -r -a eval_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ ${#eval_gpus[@]} -ge 6 ]] || { echo "Six visible GPU identifiers are needed for evaluation" >&2; exit 1; }
names=(Vox1-O Vox1-E Vox1-H Vox1-B VOICES CN-Celeb)
pids=()
cleanup_children() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap 'cleanup_children; exit 130' INT TERM
for index in "${!names[@]}"; do
    name="${names[$index]}"
    args=(--checkpoints "$PUBLISHED" "$REFERENCE" --device cuda --out-csv "$EVAL_DIR/$name.csv")
    if [[ $index -le 2 ]]; then
        script="$REPO_ROOT/scripts/eval_vox1_checkpoints.py"
        args+=(--vox1-root "$VOX1_ROOT" --protocols "${name#Vox1-}")
    else
        script="$REPO_ROOT/scripts/eval_ood_checkpoints.py"
        args+=(--benchmarks "$name" --vox1-root "$VOX1_ROOT" --voices-root "$VOICES_ROOT" \
               --cnceleb-root "$CNCELEB_ROOT" --cnceleb-mode avg)
    fi
    printf 'Starting %s on GPU %s; log=%s\n' "$name" "${eval_gpus[$index]}" "$EVAL_DIR/$name.log"
    CUDA_VISIBLE_DEVICES="${eval_gpus[$index]}" "$PYTHON_BIN" "$script" "${args[@]}" \
        > "$EVAL_DIR/$name.log" 2>&1 &
    pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "Completed ${names[$index]}"
    else
        status=$?
        echo "Evaluation ${names[$index]} exited $status; see $EVAL_DIR/${names[$index]}.log" >&2
        failed=1
    fi
done
trap - INT TERM
[[ "$failed" -eq 0 ]] || exit 1

"$PYTHON_BIN" - "$EVAL_DIR" "$PUBLISHED" "$REFERENCE" "$TRAIN_CKPT" "$TRAIN_CONFIG" \
    "$VOX1_ROOT" "$VOICES_ROOT" "$CNCELEB_ROOT" <<'PY'
import csv, hashlib, json, math, os, sys
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

out, published, reference, source, config, vox1, voices, cnceleb = map(Path, sys.argv[1:])
rows = []
for benchmark in ['Vox1-O', 'Vox1-E', 'Vox1-H', 'Vox1-B', 'VOICES', 'CN-Celeb']:
    with (out / f'{benchmark}.csv').open() as stream:
        entries = list(csv.DictReader(stream))
    expected = {published.name, reference.name}
    if len(entries) != 2 or {r.get('checkpoint') for r in entries} != expected:
        raise SystemExit(f'{benchmark}: expected exactly one result for each of the two checkpoints')
    measured = {r['checkpoint']: float(r[benchmark]) for r in entries}
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in measured.values()):
        raise SystemExit(f'{benchmark}: EER must be finite and between 0 and 100 percent')
    ours, official = measured[published.name], measured[reference.name]
    rows.append({'benchmark': benchmark, 'reproduction_eer_percent': ours,
                 'reference_eer_percent': official, 'difference_percentage_points': round(ours-official, 6)})
metadata = {'evaluation_complete': True, 'published': str(published),
            'published_sha256': digest(published), 'reference': str(reference),
            'reference_sha256': digest(reference), 'training_checkpoint': str(source),
            'training_checkpoint_sha256': digest(source), 'training_config': str(config),
            'training_config_sha256': digest(config),
            'data_roots': {'VoxCeleb1': str(vox1), 'VOICES': str(voices), 'CN-Celeb': str(cnceleb)},
            'protocol': 'full utterances, eager float32, L2-normalized embeddings, cosine; CN-Celeb avg enrollment',
            'results': rows}
temporary_csv = out / f'.summary.csv.tmp.{os.getpid()}'
with temporary_csv.open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary_csv, out / 'summary.csv')
temporary_json = out / f'.summary.json.tmp.{os.getpid()}'
with temporary_json.open('w') as stream:
    json.dump(metadata, stream, ensure_ascii=False, indent=2)
os.replace(temporary_json, out / 'summary.json')
print(json.dumps(metadata, ensure_ascii=False, indent=2))
PY
