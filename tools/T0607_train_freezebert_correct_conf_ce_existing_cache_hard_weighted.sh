#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./T0607_freezebert_correct_conf_ce_${RUN_STAMP}}"
LOG_DIR="$EXP_ROOT/logs"
ROUTER_DIR="$EXP_ROOT/routers"
OC_DIR="$EXP_ROOT/opencompass"
RECORD_DIR="$EXP_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-./T0607_freezebert_correct_conf_ce_${RUN_STAMP}.log}"

BERT="${BERT:-./task_classifier_ckpt}"
EXPERTS="medmcqa,race,sst2"
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
ROUTING_TOPK="${ROUTING_TOPK:-3}"
ROUTING_SHARPNESS="${ROUTING_SHARPNESS:-1.0}"

LLAMA_CACHE_ROOT="${LLAMA_CACHE_ROOT:-./t0602_taskcls_trainbert_chattemplate_20260603_02/caches/llama_0602_9task_3expert_official_eval_aligned_chattemplate_800_200}"
QWEN_CACHE_ROOT="${QWEN_CACHE_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0607_train_freezebert_correct_conf_ce_existing_cache_hard_weighted.sh [all|train|eval] [llama|qwen|both] [task_csv]

Defaults:
  stage=all
  family=both
  task_csv=boolq,rte,siqa,piqa,openbookqa,arc_c

This uses existing cached router datasets only. It does not rebuild cache.
Training:
  --joint_loss correct_conf_ce
  --supervision_mode oracle_loss
  --pair_loss_normalization sample_minmax
  --best_metric route_correct_acc
  --freeze_bert
  no weighted_sum training objective

OpenCompass eval:
  hard routing
  weighted_sum with routing_topk=3
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

setup_root_log() {
    if [[ -e "$ROOT_LOG" ]]; then
        echo "root log already exists: $ROOT_LOG" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$ROOT_LOG")"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
    log "[INFO] exp_root=$EXP_ROOT"
}

prepare_new_root() {
    if [[ -e "$EXP_ROOT" ]]; then
        echo "EXP_ROOT already exists; refusing to overwrite: $EXP_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
}

prepare_existing_root() {
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
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

require_family() {
    case "$1" in
        llama|qwen) ;;
        *) echo "unsupported family: $1" >&2; exit 2 ;;
    esac
}

family_label() {
    case "$1" in
        llama) echo "llama" ;;
        qwen) echo "qwen3_fp16" ;;
    esac
}

family_suffix() {
    case "$1" in
        llama) echo "3expert" ;;
        qwen) echo "3expert_sst2words" ;;
    esac
}

cache_for() {
    case "$1" in
        llama) echo "$LLAMA_CACHE_ROOT" ;;
        qwen) echo "$QWEN_CACHE_ROOT" ;;
    esac
}

model_config_for() {
    case "$1" in
        llama) echo "T0531_mrs_ablation_hard_routing.py" ;;
        qwen) echo "T0531_mrs_ablation_sst2words_hard_routing.py" ;;
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
    local cache_root="$1"
    if [[ ! -f "$cache_root/train/manifest.json" || ! -f "$cache_root/validation/manifest.json" ]]; then
        echo "missing completed cache: $cache_root" >&2
        exit 1
    fi
}

require_router() {
    local router_root="$1"
    if [[ ! -f "$router_root/router_heads.pt" || ! -f "$router_root/router_config.json" ]]; then
        echo "missing router checkpoint: $router_root" >&2
        exit 1
    fi
}

common_training_args() {
    printf '%s\n' \
        --router_dim 512 \
        --batch_size 32 \
        --epochs 10 \
        --lr 2e-4 \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 1.0 \
        --pair_loss_normalization sample_minmax \
        --best_metric route_correct_acc \
        --early_stop_patience 2 \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch
}

mrs_router_dir_for() {
    local family="$1"
    echo "$ROUTER_DIR/router_T0607_$(family_label "$family")_mrs_only_taskcls_freezebert_correct_conf_ce_t1_$(family_suffix "$family")"
}

router_dir_for() {
    local family="$1"
    local task="$2"
    echo "$ROUTER_DIR/router_T0607_$(family_label "$family")_mrs_plus_${task}_taskcls_freezebert_correct_conf_ce_t1_$(family_suffix "$family")"
}

train_mrs_only() {
    local family="$1" cache_root output_dir log_file
    local -a args
    require_family "$family"
    cache_root="$(cache_for "$family")"
    require_cache "$cache_root"
    output_dir="$(mrs_router_dir_for "$family")"
    log_file="$LOG_DIR/T0607_train_$(family_label "$family")_mrs_only_taskcls_freezebert_correct_conf_ce_${RUN_STAMP}.log"
    [[ ! -e "$output_dir" && ! -e "$log_file" ]] || {
        echo "router output/log exists: $output_dir $log_file" >&2
        exit 1
    }
    mapfile -t args < <(common_training_args)
    log "[START] train family=$family label=mrs_only freeze_bert=1 cache=$cache_root out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache_root" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train family=$family label=mrs_only out=$output_dir"
}

train_one() {
    local family="$1" task="$2" cache_root output_dir log_file
    local -a args
    require_family "$family"
    validate_task "$task"
    cache_root="$(cache_for "$family")"
    require_cache "$cache_root"
    output_dir="$(router_dir_for "$family" "$task")"
    log_file="$LOG_DIR/T0607_train_$(family_label "$family")_mrs_plus_${task}_taskcls_freezebert_correct_conf_ce_${RUN_STAMP}.log"
    [[ ! -e "$output_dir" && ! -e "$log_file" ]] || {
        echo "router output/log exists: $output_dir $log_file" >&2
        exit 1
    }
    mapfile -t args < <(common_training_args)
    log "[START] train family=$family task=$task freeze_bert=1 cache=$cache_root out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache_root" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS,$task" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train family=$family task=$task out=$output_dir"
}

eval_router() {
    local family="$1" label="$2" router_ckpt="$3" routing_mode="$4"
    shift 4
    local model_config log_file work_dir record_path topk_suffix
    local -a datasets=("$@")
    require_family "$family"
    require_router "$router_ckpt"
    model_config="$(model_config_for "$family")"
    topk_suffix=""
    if [[ "$routing_mode" == "weighted_sum" ]]; then
        topk_suffix="_topk${ROUTING_TOPK}"
    fi
    log_file="$LOG_DIR/T0607_opencompass_$(family_label "$family")_${label}_${routing_mode}${topk_suffix}_${RUN_STAMP}.log"
    work_dir="$OC_DIR/$(family_label "$family")_${label}_${routing_mode}${topk_suffix}_T0607_${RUN_STAMP}"
    record_path="$RECORD_DIR/T0607_$(family_label "$family")_${label}_${routing_mode}${topk_suffix}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }
    log "[START] opencompass family=$family label=$label mode=$routing_mode topk=${ROUTING_TOPK} datasets=${datasets[*]} log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$routing_mode" \
    T0531_ROUTING_TOPK="$([[ "$routing_mode" == "weighted_sum" ]] && echo "$ROUTING_TOPK" || true)" \
    T0531_ROUTING_SHARPNESS="$ROUTING_SHARPNESS" \
    T0531_ROUTER_RECORD_TAG="T0607_${label}_${routing_mode}${topk_suffix}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass family=$family label=$label mode=$routing_mode log=$log_file"
}

eval_mrs_only() {
    local family="$1" routing_mode="$2"
    eval_router "$family" "mrs_only_taskcls_freezebert" "$(mrs_router_dir_for "$family")" "$routing_mode" \
        SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
}

eval_one() {
    local family="$1" task="$2" routing_mode="$3" task_dataset
    validate_task "$task"
    task_dataset="$(task_dataset_for "$task")"
    eval_router "$family" "mrs_plus_${task}_taskcls_freezebert" "$(router_dir_for "$family" "$task")" "$routing_mode" \
        "$task_dataset" "${MRS_DATASETS[@]}"
}

run_train_family() {
    local family="$1"
    shift
    local task
    train_mrs_only "$family"
    for task in "$@"; do
        train_one "$family" "$task"
    done
}

run_eval_family() {
    local family="$1"
    shift
    local mode task
    for mode in hard weighted_sum; do
        eval_mrs_only "$family" "$mode"
        for task in "$@"; do
            eval_one "$family" "$task" "$mode"
        done
    done
}

main() {
    local stage="${1:-all}" family_arg="${2:-both}" task_csv="${3:-$DEFAULT_TASKS}"
    local -a families tasks
    local family task

    case "$stage" in
        all|train|eval) ;;
        -h|--help|help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac

    case "$family_arg" in
        both) families=(llama qwen) ;;
        llama|qwen) families=("$family_arg") ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$task_csv"
    tasks=("${SPLIT_CSV_RESULT[@]}")
    [[ "${#tasks[@]}" -gt 0 ]] || { echo "empty task list" >&2; exit 2; }
    for task in "${tasks[@]}"; do
        validate_task "$task"
    done

    if [[ "$stage" == "all" || "$stage" == "train" ]]; then
        prepare_new_root
    else
        prepare_existing_root
    fi
    setup_root_log

    log "[INFO] stage=$stage families=${families[*]} tasks=${tasks[*]} cuda=$CUDA_VISIBLE_DEVICES"
    log "[INFO] llama_cache=$LLAMA_CACHE_ROOT"
    log "[INFO] qwen_cache=$QWEN_CACHE_ROOT"
    log "[INFO] train_loss=correct_conf_ce best_metric=route_correct_acc freeze_bert=1 weighted_sum_train_loss=0"
    log "[INFO] eval_modes=hard,weighted_sum routing_topk=$ROUTING_TOPK routing_sharpness=$ROUTING_SHARPNESS"

    for family in "${families[@]}"; do
        case "$stage" in
            all)
                run_train_family "$family" "${tasks[@]}"
                run_eval_family "$family" "${tasks[@]}"
                ;;
            train)
                run_train_family "$family" "${tasks[@]}"
                ;;
            eval)
                run_eval_family "$family" "${tasks[@]}"
                ;;
        esac
    done

    log "[DONE] stage=$stage exp_root=$EXP_ROOT"
}

main "$@"
