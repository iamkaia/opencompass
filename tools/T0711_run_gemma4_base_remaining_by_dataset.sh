#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-google/gemma-4-e4b-it}"
OUT_ROOT="${OUT_ROOT:-outputs/T0711_gemma4_base_remaining_by_dataset}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MAX_OUT_LEN="${MAX_OUT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HF_NUM_GPUS="${HF_NUM_GPUS:-1}"
MODEL_KWARGS="${MODEL_KWARGS:-torch_dtype=torch.bfloat16}"
STOP_WORD="${STOP_WORD:-<end_of_turn>}"

DATASETS_CSV="${DATASETS:-piqa_gen,race_gen_sft_prompt,SuperGLUE_RTE_gen,siqa_gen,sst2_gen}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0711_run_gemma4_base_remaining_by_dataset.sh \
    > T0711_gemma4_base_remaining_by_dataset.nohup.log 2>&1 &

Defaults:
  MODEL_PATH=google/gemma-4-e4b-it
  DATASETS=piqa_gen,race_gen_sft_prompt,SuperGLUE_RTE_gen,siqa_gen,sst2_gen
  OUT_ROOT=outputs/T0711_gemma4_base_remaining_by_dataset
  MAX_SEQ_LEN=2048
  MAX_OUT_LEN=128
  BATCH_SIZE=1
  HF_NUM_GPUS=1

Override DATASETS with a comma-separated list to run a smaller slice.
EOF
}

case "${1:-run}" in
    -h|--help|help) usage; exit 0 ;;
    run) ;;
    *) usage; exit 2 ;;
esac

mkdir -p "$LOG_DIR"
split_csv "$DATASETS_CSV"

log "[INFO] model=$MODEL_PATH datasets=${SPLIT_RESULT[*]} out_root=$OUT_ROOT"

for dataset in "${SPLIT_RESULT[@]}"; do
    work_dir="$OUT_ROOT/$dataset"
    log_file="$LOG_DIR/${dataset}.log"
    if [[ -e "$work_dir" || -e "$log_file" ]]; then
        log "[SKIP] existing dataset=$dataset work_dir=$work_dir log=$log_file"
        continue
    fi
    log "[START] dataset=$dataset"
    python -u run.py \
        --hf-type chat \
        --hf-path "$MODEL_PATH" \
        --datasets "$dataset" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --max-out-len "$MAX_OUT_LEN" \
        --batch-size "$BATCH_SIZE" \
        --hf-num-gpus "$HF_NUM_GPUS" \
        --model-kwargs "$MODEL_KWARGS" \
        --stop-words "$STOP_WORD" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] dataset=$dataset work_dir=$work_dir log=$log_file"
done

log "[DONE] all requested datasets"
