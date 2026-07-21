#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./T0616_freezebert_weighted_sum_selftarget_vs_original_${RUN_STAMP}}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
TASK_CSV="${TASK_CSV:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-1.0}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
SHARPNESS="${SHARPNESS:-1.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
TRAIN_SPLIT_DATASET_CONFIG="${TRAIN_SPLIT_DATASET_CONFIG:-router_train_split_gen}"

LOG_DIR="$EXP_ROOT/logs"
ROUTER_DIR="$EXP_ROOT/routers"
OFFICIAL_DIR="$EXP_ROOT/opencompass_official"
TRAIN_SPLIT_DIR="$EXP_ROOT/opencompass_train_split"
RECORD_DIR="$EXP_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$EXP_ROOT/run_${RUN_STAMP}.log}"

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
  CUDA_VISIBLE_DEVICES=0 bash tools/T0616_train_freezebert_weighted_sum_selftarget_vs_original_qwen.sh [all|train|eval] [self|original|both] [task_csv]

Defaults:
  stage=all
  variant=both
  task_csv=boolq,rte,siqa,piqa,openbookqa,arc_c

What it does:
  self      target_distribution_policy=self_task_if_available
  original  target_distribution_policy=cache_oracle

Both variants use the same existing cached dataset, freeze BERT, train with
weighted_sum_marginal_mse only, then run weighted_sum OpenCompass official
eval and weighted_sum router_train_split eval.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

setup_root_log() {
    mkdir -p "$(dirname "$ROOT_LOG")"
    if [[ -e "$ROOT_LOG" ]]; then
        echo "root log already exists: $ROOT_LOG" >&2
        exit 1
    fi
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
}

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

validate_variant() {
    case "$1" in
        self|original) ;;
        *) echo "unsupported variant: $1" >&2; exit 2 ;;
    esac
}

target_policy_for() {
    case "$1" in
        self) echo "self_task_if_available" ;;
        original) echo "cache_oracle" ;;
    esac
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

require_cache() {
    [[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || {
        echo "missing completed cache: $CACHE_ROOT" >&2
        exit 1
    }
}

require_router() {
    local router_root="$1"
    [[ -f "$router_root/router_heads.pt" && -f "$router_root/router_config.json" ]] || {
        echo "missing router checkpoint: $router_root" >&2
        exit 1
    }
}

router_dir_for() {
    local variant="$1"
    local task="${2:-}"
    if [[ -z "$task" ]]; then
        echo "$ROUTER_DIR/router_T0616_qwen3_fp16_${variant}_mrs_only_freezebert_wsmse_auxonly_t1_3expert_sst2words"
    else
        echo "$ROUTER_DIR/router_T0616_qwen3_fp16_${variant}_mrs_plus_${task}_freezebert_wsmse_auxonly_t1_3expert_sst2words"
    fi
}

train_one() {
    local variant="$1" task="${2:-}" sample_tasks output_dir log_file policy label
    validate_variant "$variant"
    policy="$(target_policy_for "$variant")"
    if [[ -z "$task" ]]; then
        label="${variant}_mrs_only"
        sample_tasks="$EXPERTS"
    else
        validate_task "$task"
        label="${variant}_mrs_plus_${task}"
        sample_tasks="$EXPERTS,$task"
    fi
    output_dir="$(router_dir_for "$variant" "$task")"
    log_file="$LOG_DIR/T0616_train_qwen3_fp16_${label}_freezebert_wsmse_auxonly_${RUN_STAMP}.log"
    [[ ! -e "$output_dir" && ! -e "$log_file" ]] || {
        echo "router output/log exists: $output_dir $log_file" >&2
        exit 1
    }

    log "[START] train label=$label policy=$policy cache=$CACHE_ROOT out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --joint_loss cache_oracle_matrix_kl \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature "$TARGET_TEMPERATURE" \
        --pair_loss_normalization sample_minmax \
        --weighted_sum_marginal_mse_weight "$MSE_WEIGHT" \
        --weighted_sum_aux_only \
        --target_distribution_policy "$policy" \
        --best_metric weighted_sum_marginal_mse \
        --early_stop_patience 2 \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1
    log "[DONE] train label=$label out=$output_dir"
}

eval_official() {
    local variant="$1" task="${2:-}" router label log_file work_dir record_path
    local -a datasets
    validate_variant "$variant"
    router="$(router_dir_for "$variant" "$task")"
    require_router "$router"
    if [[ -z "$task" ]]; then
        label="${variant}_mrs_only"
        datasets=("${OFFICIAL_ALL_DATASETS[@]}")
    else
        validate_task "$task"
        label="${variant}_mrs_plus_${task}"
        datasets=("$(task_dataset_for "$task")" "${MRS_DATASETS[@]}")
    fi
    log_file="$LOG_DIR/T0616_official_qwen3_fp16_${label}_weighted_sum_${RUN_STAMP}.log"
    work_dir="$OFFICIAL_DIR/qwen3_fp16_${label}_weighted_sum_${RUN_STAMP}"
    record_path="$RECORD_DIR/T0616_official_qwen3_fp16_${label}_weighted_sum_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] official label=$label sharpness=$SHARPNESS datasets=${datasets[*]} log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS="$SHARPNESS" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="T0616_official_${label}_weighted_sum" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official label=$label log=$log_file"
}

eval_train_split() {
    local variant="$1" task="${2:-}" router label log_file work_dir record_path
    validate_variant "$variant"
    router="$(router_dir_for "$variant" "$task")"
    require_router "$router"
    if [[ -z "$task" ]]; then
        label="${variant}_mrs_only"
    else
        validate_task "$task"
        label="${variant}_mrs_plus_${task}"
    fi
    log_file="$LOG_DIR/T0616_train_split_qwen3_fp16_${label}_weighted_sum_${RUN_STAMP}.log"
    work_dir="$TRAIN_SPLIT_DIR/qwen3_fp16_${label}_weighted_sum_${RUN_STAMP}"
    record_path="$RECORD_DIR/T0616_train_split_qwen3_fp16_${label}_weighted_sum_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "train_split eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] train_split label=$label sharpness=$SHARPNESS dataset=$TRAIN_SPLIT_DATASET_CONFIG log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS="$SHARPNESS" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="T0616_train_split_${label}_weighted_sum" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "$TRAIN_SPLIT_DATASET_CONFIG" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] train_split label=$label log=$log_file"
}

run_train_variant() {
    local variant="$1"
    shift
    local task
    train_one "$variant"
    for task in "$@"; do
        train_one "$variant" "$task"
    done
}

run_eval_variant() {
    local variant="$1"
    shift
    local task
    eval_train_split "$variant"
    eval_official "$variant"
    for task in "$@"; do
        eval_train_split "$variant" "$task"
        eval_official "$variant" "$task"
    done
}

main() {
    local stage="${1:-all}" variant_arg="${2:-both}" task_csv="${3:-$TASK_CSV}"
    local -a variants tasks
    local variant task

    case "$stage" in
        all|train|eval) ;;
        -h|--help|help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
    case "$variant_arg" in
        both) variants=(self original) ;;
        self|original) variants=("$variant_arg") ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$task_csv"
    tasks=("${SPLIT_CSV_RESULT[@]}")
    [[ "${#tasks[@]}" -gt 0 ]] || { echo "empty task list" >&2; exit 2; }
    for task in "${tasks[@]}"; do
        validate_task "$task"
    done

    require_cache
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OFFICIAL_DIR" "$TRAIN_SPLIT_DIR" "$RECORD_DIR"
    setup_root_log
    log "[INFO] stage=$stage variants=${variants[*]} tasks=${tasks[*]} cuda=$CUDA_VISIBLE_DEVICES"
    log "[INFO] exp_root=$EXP_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT"
    log "[INFO] freeze_bert=1 joint_loss=cache_oracle_matrix_kl aux_only=1 mse_weight=$MSE_WEIGHT target_temperature=$TARGET_TEMPERATURE"
    log "[INFO] eval routing_mode=weighted_sum sharpness=$SHARPNESS routing_topk=${ROUTING_TOPK:-none}"

    for variant in "${variants[@]}"; do
        case "$stage" in
            all)
                run_train_variant "$variant" "${tasks[@]}"
                run_eval_variant "$variant" "${tasks[@]}"
                ;;
            train)
                run_train_variant "$variant" "${tasks[@]}"
                ;;
            eval)
                run_eval_variant "$variant" "${tasks[@]}"
                ;;
        esac
    done

    log "[DONE] stage=$stage exp_root=$EXP_ROOT"
}

main "$@"
