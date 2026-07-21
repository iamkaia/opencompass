#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_ROOT="${SOURCE_ROOT:-./T0616_wsum_correct_conf_raw_t0p25_freezebert_full_20260616_061945}"
OUT_ROOT="${OUT_ROOT:-./T0622_t0616_first_weights_all_layers_${RUN_STAMP}}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
SHARPNESS_LIST="${SHARPNESS_LIST:-1.0,2.0,3.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
TRAIN_SPLIT_DATASET="${TRAIN_SPLIT_DATASET:-router_train_split_gen}"

LOG_DIR="$OUT_ROOT/logs"
TRAIN_SPLIT_DIR="$OUT_ROOT/opencompass_train_split"
OFFICIAL_DIR="$OUT_ROOT/opencompass_official"
RECORD_DIR="$OUT_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$OUT_ROOT/pipeline.log}"

MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
OFFICIAL_ALL_DATASETS=(
    SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen
    piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash ./tools/T0622_eval_t0616_first_weights_all_layers_qwen.sh [all|train_split|official]

Reuses the existing T0616 router checkpoints. No router is retrained.
The learned weighted_sum first_weights are applied to every decoder layer.

Optional environment variables:
  TASKS=boolq,rte,siqa,piqa,openbookqa,arc_c
  SHARPNESS_LIST=1.0,2.0,3.0
  ROUTING_TOPK=
  SOURCE_ROOT=./T0616_wsum_correct_conf_raw_t0p25_freezebert_full_20260616_061945
  OUT_ROOT=./T0622_t0616_first_weights_all_layers_<timestamp>
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

split_csv() {
    local IFS=,
    read -r -a SPLIT_RESULT <<< "$1"
}

temp_tag() { printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'; }

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

task_dataset_for() {
    case "$1" in
        boolq) echo SuperGLUE_BoolQ_gen ;;
        rte) echo SuperGLUE_RTE_gen ;;
        siqa) echo siqa_gen ;;
        piqa) echo piqa_gen ;;
        openbookqa) echo obqa_main_gen ;;
        arc_c) echo ARC_c_gen ;;
    esac
}

mrs_only_router() {
    echo "$SOURCE_ROOT/routers/router_T0616_qwen3_fp16_mrs_only_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

mrs_plus_router() {
    echo "$SOURCE_ROOT/routers/router_T0616_qwen3_fp16_mrs_plus_${1}_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

new_only_router() {
    echo "$SOURCE_ROOT/routers/router_T0616_qwen3_fp16_new_only_${1}_from_mrs_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

require_router() {
    [[ -f "$1/router_heads.pt" && -f "$1/router_config.json" ]] || {
        echo "missing router checkpoint: $1" >&2
        exit 1
    }
}

setup_root() {
    if [[ -e "$OUT_ROOT" ]]; then
        echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$TRAIN_SPLIT_DIR" "$OFFICIAL_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

run_opencompass() {
    local label="$1" router="$2" sharpness="$3" work_dir="$4" record_path="$5" log_file="$6"
    shift 6
    local -a datasets=("$@")
    require_router "$router"
    log "[START] label=$label sharpness=$sharpness router=$router datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_SHARE_FIRST_WEIGHTS_ALL_LAYERS=1 \
    T0531_ROUTER_RECORD_TAG="T0622_${label}_first_weights_all_layers" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] label=$label log=$log_file work_dir=$work_dir record=$record_path"
}

eval_train_split() {
    local label="$1" router="$2"
    run_opencompass \
        "train_split_${label}" "$router" 1.0 \
        "$TRAIN_SPLIT_DIR/${label}_first_weights_all_layers_${RUN_STAMP}" \
        "$RECORD_DIR/T0622_train_split_${label}_first_weights_all_layers_${RUN_STAMP}.jsonl" \
        "$LOG_DIR/T0622_train_split_${label}_first_weights_all_layers_${RUN_STAMP}.log" \
        "$TRAIN_SPLIT_DATASET"
}

eval_official() {
    local label="$1" router="$2" sharpness="$3"
    shift 3
    local tag
    tag="$(temp_tag "$sharpness")"
    run_opencompass \
        "official_${label}_sharp${tag}" "$router" "$sharpness" \
        "$OFFICIAL_DIR/${label}_first_weights_all_layers_sharp${tag}_${RUN_STAMP}" \
        "$RECORD_DIR/T0622_official_${label}_first_weights_all_layers_sharp${tag}_${RUN_STAMP}.jsonl" \
        "$LOG_DIR/T0622_official_${label}_first_weights_all_layers_sharp${tag}_${RUN_STAMP}.log" \
        "$@"
}

run_train_split_all() {
    local task
    eval_train_split mrs_only "$(mrs_only_router)"
    for task in "${TASK_LIST[@]}"; do
        eval_train_split "mrs_plus_${task}" "$(mrs_plus_router "$task")"
    done
    for task in "${TASK_LIST[@]}"; do
        eval_train_split "new_only_${task}" "$(new_only_router "$task")"
    done
}

run_official_all() {
    local sharpness task dataset
    split_csv "$SHARPNESS_LIST"
    local -a sharpness_values=("${SPLIT_RESULT[@]}")
    for sharpness in "${sharpness_values[@]}"; do
        eval_official mrs_only "$(mrs_only_router)" "$sharpness" "${OFFICIAL_ALL_DATASETS[@]}"
        for task in "${TASK_LIST[@]}"; do
            dataset="$(task_dataset_for "$task")"
            eval_official "mrs_plus_${task}" "$(mrs_plus_router "$task")" "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
        done
        for task in "${TASK_LIST[@]}"; do
            dataset="$(task_dataset_for "$task")"
            eval_official "new_only_${task}" "$(new_only_router "$task")" "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
        done
    done
}

main() {
    local mode="${1:-all}" task
    case "$mode" in
        all|train_split|official) ;;
        -h|--help|help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_RESULT[@]}")
    for task in "${TASK_LIST[@]}"; do validate_task "$task"; done

    setup_root
    log "[INFO] source_root=$SOURCE_ROOT"
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] mode=$mode tasks=${TASK_LIST[*]} sharpness=$SHARPNESS_LIST cuda=$CUDA_VISIBLE_DEVICES"
    log "[INFO] routing=weighted_sum first_weights shared across all decoder layers; no training"

    [[ "$mode" == all || "$mode" == train_split ]] && run_train_split_all
    [[ "$mode" == all || "$mode" == official ]] && run_official_all
    log "[DONE] output=$OUT_ROOT"
}

main "$@"
