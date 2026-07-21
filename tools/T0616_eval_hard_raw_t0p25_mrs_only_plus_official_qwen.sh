#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-./T0616_hard_correct_conf_raw_t0p25_freezebert_full_20260625_110157}"
OUT_ROOT="${OUT_ROOT:-./T0616_hard_correct_conf_raw_t0p25_mrs_only_plus_official_eval_${RUN_STAMP}}"
RUN_LABEL="${RUN_LABEL:-T0616HARD_OFFICIAL}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
ROUTING_MODE="${ROUTING_MODE:-hard}"
ROUTING_SHARPNESS="${ROUTING_SHARPNESS:-1.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"

LOG_DIR="$OUT_ROOT/logs"
OC_OFFICIAL_DIR="$OUT_ROOT/opencompass_official"
RECORD_DIR="$OUT_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$OUT_ROOT/pipeline.log}"

MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
OFFICIAL_ALL_DATASETS=(
    SuperGLUE_BoolQ_gen
    medmcqa_gen_sft_prompt
    obqa_main_gen
    ARC_c_gen
    piqa_gen
    race_gen_sft_prompt
    SuperGLUE_RTE_gen
    siqa_gen
    sst2_gen
)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash ./tools/T0616_eval_hard_raw_t0p25_mrs_only_plus_official_qwen.sh > T0616_hard_raw_t0p25_mrs_only_plus_official_eval.nohup.log 2>&1 &

This reuses trained routers from SOURCE_RUN_ROOT and runs official OpenCompass eval only.
It intentionally skips train, train_split, and new_only eval.

Defaults:
  SOURCE_RUN_ROOT=./T0616_hard_correct_conf_raw_t0p25_freezebert_full_20260625_110157
  ROUTING_MODE=hard
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

setup_root() {
    if [[ -e "$OUT_ROOT" ]]; then
        echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$OC_OFFICIAL_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

require_router() {
    local router="$1"
    [[ -f "$router/router_heads.pt" && -f "$router/router_config.json" ]] || {
        echo "missing router checkpoint: $router" >&2
        exit 1
    }
}

task_dataset_for() {
    case "$1" in
        boolq) echo "SuperGLUE_BoolQ_gen" ;;
        rte) echo "SuperGLUE_RTE_gen" ;;
        siqa) echo "siqa_gen" ;;
        piqa) echo "piqa_gen" ;;
        openbookqa) echo "obqa_main_gen" ;;
        arc_c) echo "ARC_c_gen" ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

mrs_only_router() {
    echo "$SOURCE_RUN_ROOT/routers/router_T0616HARD_qwen3_fp16_mrs_only_freezebert_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

mrs_plus_router() {
    local task="$1"
    echo "$SOURCE_RUN_ROOT/routers/router_T0616HARD_qwen3_fp16_mrs_plus_${task}_freezebert_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

eval_official() {
    local label="$1" router="$2"
    shift 2
    local log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router"

    log_file="$LOG_DIR/${RUN_LABEL}_official_${label}_${ROUTING_MODE}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_${ROUTING_MODE}_${RUN_STAMP}"
    record_path="$RECORD_DIR/${RUN_LABEL}_official_${label}_${ROUTING_MODE}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] official label=$label mode=$ROUTING_MODE router=$router datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$ROUTING_MODE" \
    T0531_ROUTING_SHARPNESS="$ROUTING_SHARPNESS" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_official_${label}_${ROUTING_MODE}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official label=$label mode=$ROUTING_MODE log=$log_file"
}

eval_all_official() {
    local task dataset
    eval_official "mrs_only" "$(mrs_only_router)" "${OFFICIAL_ALL_DATASETS[@]}"
    for task in "${TASK_LIST[@]}"; do
        dataset="$(task_dataset_for "$task")"
        eval_official "mrs_plus_${task}" "$(mrs_plus_router "$task")" "$dataset" "${MRS_DATASETS[@]}"
    done
}

main() {
    case "${1:-}" in
        -h|--help|help) usage; exit 0 ;;
        "") ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_CSV_RESULT[@]}")
    local task
    for task in "${TASK_LIST[@]}"; do
        validate_task "$task"
    done

    setup_root
    log "[INFO] source_run_root=$SOURCE_RUN_ROOT"
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] tasks=${TASK_LIST[*]} cuda=$CUDA_VISIBLE_DEVICES"
    log "[INFO] eval=official_only routing_mode=$ROUTING_MODE routing_sharpness=$ROUTING_SHARPNESS"
    log "[INFO] skipped=train,train_split,new_only"

    eval_all_official

    log "[DONE] official-only mrs_only + mrs_plus eval output=$OUT_ROOT"
}

main "$@"
