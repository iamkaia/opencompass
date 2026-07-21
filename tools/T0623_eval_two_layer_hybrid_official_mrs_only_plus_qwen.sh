#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-20260623_024009}"
RUN_LABEL="${RUN_LABEL:-T0623_hybrid}"
OUT_ROOT="${OUT_ROOT:-./T0623_wsum_hybrid_joint_marginal_raw_t0p25_freezebert_full_${RUN_STAMP}}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
SHARPNESS_LIST="${SHARPNESS_LIST:-1.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
EVAL_TAG="${EVAL_TAG:-official_only}"

LOG_DIR="$OUT_ROOT/logs"
ROUTER_DIR="$OUT_ROOT/routers"
OC_OFFICIAL_DIR="$OUT_ROOT/opencompass_official"
RECORD_DIR="$OUT_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$OUT_ROOT/pipeline_${EVAL_TAG}.log}"

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
    cat <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash ./tools/T0623_eval_two_layer_hybrid_official_mrs_only_plus_qwen.sh

Official OpenCompass eval only for the already-trained T0623 two-layer hybrid routers.

This script does NOT train, does NOT run train_split, and does NOT run new_only.
It evaluates:
  - mrs_only
  - mrs_plus_boolq
  - mrs_plus_rte
  - mrs_plus_siqa
  - mrs_plus_piqa
  - mrs_plus_openbookqa
  - mrs_plus_arc_c

Defaults:
  RUN_STAMP=20260623_024009
  OUT_ROOT=./T0623_wsum_hybrid_joint_marginal_raw_t0p25_freezebert_full_20260623_024009
  SHARPNESS_LIST=1.0
  EVAL_TAG=official_only

Outputs are written under:
  $OUT_ROOT/opencompass_official/*_${EVAL_TAG}/
  $OUT_ROOT/logs/${RUN_LABEL}_${EVAL_TAG}_official_*.log
  $OUT_ROOT/router_records/${RUN_LABEL}_${EVAL_TAG}_official_*.jsonl
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

temp_tag() {
    printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
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
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_mrs_only_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

mrs_plus_router() {
    local task="$1"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_mrs_plus_${task}_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

setup_root() {
    [[ -d "$OUT_ROOT" ]] || {
        echo "missing OUT_ROOT: $OUT_ROOT" >&2
        exit 1
    }
    mkdir -p "$LOG_DIR" "$OC_OFFICIAL_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

eval_official() {
    local label="$1" router="$2" sharpness="$3"
    shift 3
    local sharp_tag log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router"
    sharp_tag="$(temp_tag "$sharpness")"
    log_file="$LOG_DIR/${RUN_LABEL}_${EVAL_TAG}_official_${label}_sharp${sharp_tag}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}_${EVAL_TAG}"
    record_path="$RECORD_DIR/${RUN_LABEL}_${EVAL_TAG}_official_${label}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] official_only label=$label sharpness=$sharpness datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_${EVAL_TAG}_official_${label}_sharp${sharp_tag}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official_only label=$label sharpness=$sharpness log=$log_file"
}

eval_all_official() {
    local sharpness task dataset
    split_csv "$SHARPNESS_LIST"
    local -a sharpness_values=("${SPLIT_CSV_RESULT[@]}")
    for sharpness in "${sharpness_values[@]}"; do
        eval_official "mrs_only" "$(mrs_only_router)" "$sharpness" "${OFFICIAL_ALL_DATASETS[@]}"
        for task in "${TASK_LIST[@]}"; do
            dataset="$(task_dataset_for "$task")"
            eval_official "mrs_plus_${task}" "$(mrs_plus_router "$task")" "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
        done
    done
}

main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
        usage
        exit 0
    fi
    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_CSV_RESULT[@]}")
    local task
    for task in "${TASK_LIST[@]}"; do
        validate_task "$task"
    done
    setup_root
    log "[INFO] official_only out_root=$OUT_ROOT run_label=$RUN_LABEL sharpness=$SHARPNESS_LIST tasks=${TASK_LIST[*]} eval_tag=$EVAL_TAG"
    eval_all_official
    log "[DONE] official_only output=$OUT_ROOT"
}

main "$@"
