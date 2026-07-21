#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0616_target_weight_oracle_train_split_${RUN_STAMP}}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
ROUTER_CKPT="${ROUTER_CKPT:-$SOURCE_EXP_ROOT/routers/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmse1p0_3expert_sst2words}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
DATASET_CONFIG="${DATASET_CONFIG:-router_train_split_gen}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
SHARPNESS="${SHARPNESS:-1.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
EMPTY_TARGET_FALLBACK="${EMPTY_TARGET_FALLBACK:-uniform}"

WEIGHT_DIR="$OUT_ROOT/oracle_weights"
LOG_DIR="$OUT_ROOT/logs"
OC_DIR="$OUT_ROOT/opencompass"
RECORD_DIR="$OUT_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$OUT_ROOT/pipeline.log}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0616_build_target_weights_and_eval_train_split_qwen.sh

Builds train-split oracle weight JSONL files in a fresh OUT_ROOT, then runs
OpenCompass router_train_split_gen eval for:
  - correct_conf_or_loss at t1, t0.5, t0.25
  - original correct_conf_ce at t1, t0.5, t0.25, using uniform empty fallback by default
  - self_if_available_else_correct_conf_ce at t1, t0.5, t0.25, using uniform empty fallback by default
  - one uniform baseline

Outputs:
  OUT_ROOT/oracle_weights/*.jsonl
  OUT_ROOT/opencompass/*
  OUT_ROOT/router_records/*.jsonl
  OUT_ROOT/logs/*.log
  OUT_ROOT/pipeline.log

Env overrides:
  OUT_ROOT, CACHE_ROOT, ROUTER_CKPT, BASE_MODEL_PATH, LOCAL_FILES_ONLY=1
  EMPTY_TARGET_FALLBACK default uniform; choices zero, uniform, loss_softmax
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

setup_root() {
    if [[ -e "$OUT_ROOT" ]]; then
        echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$WEIGHT_DIR" "$LOG_DIR" "$OC_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

require_inputs() {
    [[ -f "$CACHE_ROOT/train/manifest.json" ]] || {
        echo "missing cache train manifest: $CACHE_ROOT/train/manifest.json" >&2
        exit 1
    }
    [[ -f "$ROUTER_CKPT/router_heads.pt" && -f "$ROUTER_CKPT/router_config.json" ]] || {
        echo "missing router checkpoint: $ROUTER_CKPT" >&2
        exit 1
    }
}

temp_tag() {
    printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

export_one() {
    local mode="$1" temp="$2" label="$3" out_file log_file
    out_file="$WEIGHT_DIR/${label}.jsonl"
    log_file="$LOG_DIR/export_${label}.log"
    [[ ! -e "$out_file" && ! -e "$log_file" ]] || {
        echo "export output exists: $out_file $log_file" >&2
        exit 1
    }

    local -a local_args=()
    if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
        local_args+=(--local_files_only)
    fi

    log "[START] export label=$label mode=$mode temp=$temp out=$out_file"
    "$PY" -u tools/export_cache_oracle_weights.py \
        --cache_root "$CACHE_ROOT" \
        --split train \
        --out "$out_file" \
        --mode "$mode" \
        --temperature "$temp" \
        --empty_target_fallback "$EMPTY_TARGET_FALLBACK" \
        --base_model_path "$BASE_MODEL_PATH" \
        "${local_args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] export label=$label log=$log_file"
}

run_train_split_eval() {
    local label="$1" routing_mode="$2" oracle_path="${3:-}" log_file work_dir record_path
    log_file="$LOG_DIR/eval_train_split_${label}_${routing_mode}.log"
    work_dir="$OC_DIR/qwen_train_split_${label}_${routing_mode}_${RUN_STAMP}"
    record_path="$RECORD_DIR/qwen_train_split_${label}_${routing_mode}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] eval label=$label routing_mode=$routing_mode oracle=${oracle_path:-none}"
    T0531_MRS_ROUTER_CKPT="$ROUTER_CKPT" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$routing_mode" \
    T0531_ROUTING_SHARPNESS="$SHARPNESS" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0610_ORACLE_WEIGHT_PATH="$oracle_path" \
    T0531_ROUTER_RECORD_TAG="T0616_${label}_${routing_mode}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "$DATASET_CONFIG" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] eval label=$label log=$log_file"
}

build_all_weights() {
    local temp tag
    for temp in 1.0 0.5 0.25; do
        tag="$(temp_tag "$temp")"
        export_one correct_conf_or_loss "$temp" "correct_conf_or_loss_t${tag}"
        export_one correct_conf_ce "$temp" "correct_conf_ce_empty_${EMPTY_TARGET_FALLBACK}_t${tag}"
        export_one self_if_available_else_correct_conf_ce "$temp" "self_if_available_else_correct_conf_ce_empty_${EMPTY_TARGET_FALLBACK}_t${tag}"
    done
}

eval_all_weights() {
    local file label
    for file in "$WEIGHT_DIR"/*.jsonl; do
        label="$(basename "$file" .jsonl)"
        run_train_split_eval "$label" cache_oracle_weighted_sum "$file"
    done
    run_train_split_eval uniform_baseline uniform
}

main() {
    case "${1:-}" in
        -h|--help|help) usage; exit 0 ;;
        "") ;;
        *) usage; exit 2 ;;
    esac

    setup_root
    require_inputs
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT"
    log "[INFO] router_ckpt=$ROUTER_CKPT"
    log "[INFO] dataset=$DATASET_CONFIG model_config=$MODEL_CONFIG"
    log "[INFO] base_model_path=$BASE_MODEL_PATH local_files_only=$LOCAL_FILES_ONLY"
    log "[INFO] empty_target_fallback=$EMPTY_TARGET_FALLBACK for correct_conf_ce-style no-correct samples"
    log "[INFO] eval routing: cache_oracle_weighted_sum plus uniform baseline, sharpness=$SHARPNESS topk=${ROUTING_TOPK:-none}"

    build_all_weights
    eval_all_weights

    log "[DONE] all target-weight train_split evals output=$OUT_ROOT"
}

main "$@"
