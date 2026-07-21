#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
ROUTER_CKPT="${MRS_ONLY_ROUTER_CKPT:?MRS_ONLY_ROUTER_CKPT is required}"
SHARPNESS="${SHARPNESS:-1.0}"
SHARPNESS_TAG="$(printf '%s' "$SHARPNESS" | sed 's/-/m/g; s/\./p/g')"
ROUTING_TOPK="${ROUTING_TOPK:-}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"

LOG_DIR="$OUT_ROOT/logs"
OC_DIR="$OUT_ROOT/opencompass"
RECORD_DIR="$OUT_ROOT/router_records"
LABEL="$(basename "$ROUTER_CKPT")"
LOG_FILE="$LOG_DIR/${LABEL}_weighted_sum_sharp${SHARPNESS_TAG}_${RUN_STAMP}.log"
WORK_DIR="$OC_DIR/${LABEL}_weighted_sum_sharp${SHARPNESS_TAG}_${RUN_STAMP}"
RECORD_PATH="$RECORD_DIR/${LABEL}_weighted_sum_sharp${SHARPNESS_TAG}_${RUN_STAMP}.jsonl"

[[ -f "$ROUTER_CKPT/router_heads.pt" && -f "$ROUTER_CKPT/router_config.json" ]] || {
    echo "missing router checkpoint: $ROUTER_CKPT" >&2
    exit 1
}
mkdir -p "$LOG_DIR" "$OC_DIR" "$RECORD_DIR"

T0531_MRS_ROUTER_CKPT="$ROUTER_CKPT" \
T0531_ROUTER_BERT_INIT="$BERT" \
T0531_ROUTING_MODE=weighted_sum \
T0531_ROUTING_SHARPNESS="$SHARPNESS" \
T0531_ROUTING_TOPK="$ROUTING_TOPK" \
T0531_ROUTER_RECORD_TAG="T0615_${LABEL}_weighted_sum" \
T0601_ROUTER_RECORD_PATH="$RECORD_PATH" \
    "$PY" -u run.py \
    --models "$MODEL_CONFIG" \
    --datasets SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen \
    --work-dir "$WORK_DIR" \
    --debug \
    > "$LOG_FILE" 2>&1
