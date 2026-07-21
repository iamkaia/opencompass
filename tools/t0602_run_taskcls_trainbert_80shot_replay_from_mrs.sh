#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SRC_EXP_ROOT="${SRC_EXP_ROOT:-/home/u9472191/opencompass/t0602_bert_ablation_hard_20260602_082523}"
EXP_ROOT="${EXP_ROOT:-./runs/t0602_taskcls_trainbert_80shot_replay_from_mrs_${RUN_STAMP}}"
LOG_DIR="$EXP_ROOT/logs"
SUBSET_CACHE_DIR="$EXP_ROOT/subset_caches"
ROUTER_DIR="$EXP_ROOT/routers"
OC_DIR="$EXP_ROOT/opencompass"
RECORD_DIR="$EXP_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$EXP_ROOT/root_${RUN_STAMP}.log}"

EXPERTS="medmcqa,race,sst2"
BERT="./task_classifier_ckpt"
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"
PER_TASK="${PER_TASK:-80}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/t0602_run_taskcls_trainbert_80shot_replay_from_mrs.sh [train|eval|all] [llama|qwen|both] [task_csv]

Uses existing caches and existing MRS taskcls trainbert router checkpoints under:
  SRC_EXP_ROOT=/home/u9472191/opencompass/t0602_bert_ablation_hard_20260602_082523

For each replay task, creates a subset cache with:
  medmcqa,race,sst2,<task> each PER_TASK=80 samples by default

Then trains with:
  --load_from existing MRS taskcls trainbert checkpoint
  --train_bert

OpenCompass eval runs MRS + that new task only.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

setup_root_log() {
    mkdir -p "$(dirname "$ROOT_LOG")"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
    log "[INFO] exp_root=$EXP_ROOT"
    log "[INFO] src_exp_root=$SRC_EXP_ROOT"
}

prepare_dirs() {
    mkdir -p "$LOG_DIR" "$SUBSET_CACHE_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
}

family_label() {
    case "$1" in llama) echo "llama" ;; qwen) echo "qwen3_fp16" ;; *) usage; exit 2 ;; esac
}

family_suffix() {
    case "$1" in llama) echo "3expert" ;; qwen) echo "3expert_sst2words" ;; *) usage; exit 2 ;; esac
}

src_cache_for() {
    case "$1" in
        llama) echo "$SRC_EXP_ROOT/caches/llama_0602_9task_3expert_official_eval_aligned_800_200" ;;
        qwen) echo "$SRC_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_800_200" ;;
    esac
}

src_mrs_router_for() {
    case "$1" in
        llama) echo "$SRC_EXP_ROOT/routers/router_T0602_llama_mrs_only_taskcls_trainbert_correct_conf_ce_t1_3expert" ;;
        qwen) echo "$SRC_EXP_ROOT/routers/router_T0602_qwen3_fp16_mrs_only_taskcls_trainbert_correct_conf_ce_t1_3expert_sst2words" ;;
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

validate_task() {
    case "$1" in boolq|rte|siqa|piqa|openbookqa|arc_c) ;; *) echo "unsupported task: $1" >&2; exit 2 ;; esac
}

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
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

subset_cache_for() {
    local family="$1" task="$2"
    echo "$SUBSET_CACHE_DIR/$(family_label "$family")_mrs_plus_${task}_${PER_TASK}shot_from_t0602_cache"
}

router_dir_for() {
    local family="$1" task="$2"
    echo "$ROUTER_DIR/router_T0602_$(family_label "$family")_mrs_plus_${task}_taskcls_trainbert_${PER_TASK}shot_loadmrs_correct_conf_ce_t1_$(family_suffix "$family")"
}

common_training_args() {
    printf '%s\n' \
        --router_dim 512 \
        --batch_size 16 \
        --epochs 10 \
        --lr 2e-4 \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 1.0 \
        --pair_loss_normalization sample_minmax \
        --best_metric route_correct_acc \
        --early_stop_patience 2 \
        --train_bert \
        --save_route_records \
        --eval_train_each_epoch
}

build_subset_cache() {
    local family="$1" task="$2" src_cache subset_cache log_file tasks
    src_cache="$(src_cache_for "$family")"
    subset_cache="$(subset_cache_for "$family" "$task")"
    log_file="$LOG_DIR/subset_cache_$(family_label "$family")_${task}_${PER_TASK}shot_${RUN_STAMP}.log"
    tasks="$EXPERTS,$task"
    require_cache "$src_cache"
    if [[ -e "$subset_cache" ]]; then
        if [[ -f "$subset_cache/train/manifest.json" && -f "$subset_cache/validation/manifest.json" ]]; then
            log "[SKIP] subset cache exists family=$family task=$task cache=$subset_cache"
            return
        fi
        echo "subset cache exists but incomplete: $subset_cache" >&2
        exit 1
    fi
    log "[START] subset_cache family=$family task=$task per_task=$PER_TASK src=$src_cache out=$subset_cache log=$log_file"
    "$PY" -u tools/t0602_subset_cached_router_dataset.py \
        --src_root "$src_cache" \
        --dst_root "$subset_cache" \
        --tasks "$tasks" \
        --per_task "$PER_TASK" \
        > "$log_file" 2>&1
    log "[DONE] subset_cache family=$family task=$task out=$subset_cache"
}

train_one() {
    local family="$1" task="$2" subset_cache output_dir load_from log_file
    local -a args
    subset_cache="$(subset_cache_for "$family" "$task")"
    output_dir="$(router_dir_for "$family" "$task")"
    load_from="$(src_mrs_router_for "$family")"
    log_file="$LOG_DIR/train_$(family_label "$family")_mrs_plus_${task}_${PER_TASK}shot_loadmrs_${RUN_STAMP}.log"
    require_cache "$subset_cache"
    require_router "$load_from"
    [[ ! -e "$output_dir" ]] || { echo "router output exists: $output_dir" >&2; exit 1; }
    mapfile -t args < <(common_training_args)
    log "[START] train family=$family task=$task subset=$subset_cache load_from=$load_from out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$subset_cache" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --load_from "$load_from" \
        --sample_task_names "$EXPERTS,$task" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train family=$family task=$task out=$output_dir"
}

eval_one() {
    local family="$1" task="$2" router_ckpt model_config task_dataset log_file work_dir record_path
    local -a datasets
    router_ckpt="$(router_dir_for "$family" "$task")"
    require_router "$router_ckpt"
    model_config="$(model_config_for "$family")"
    task_dataset="$(task_dataset_for "$task")"
    datasets=("$task_dataset" medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
    log_file="$LOG_DIR/opencompass_$(family_label "$family")_mrs_plus_${task}_${PER_TASK}shot_loadmrs_hard_${RUN_STAMP}.log"
    work_dir="$OC_DIR/$(family_label "$family")_mrs_plus_${task}_${PER_TASK}shot_loadmrs_hard"
    record_path="$RECORD_DIR/$(family_label "$family")_mrs_plus_${task}_${PER_TASK}shot_loadmrs_hard.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists for $family $task" >&2; exit 1;
    }
    log "[START] opencompass family=$family task=$task datasets=${datasets[*]} log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="hard" \
    T0531_ROUTER_RECORD_TAG="t0602_${PER_TASK}shot_${task}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass family=$family task=$task log=$log_file"
}

run_family() {
    local stage="$1" family="$2"
    local -a tasks=("${@:3}")
    local task
    for task in "${tasks[@]}"; do
        case "$stage" in all|train)
            build_subset_cache "$family" "$task"
            train_one "$family" "$task"
            ;;
        esac
        case "$stage" in all|eval)
            eval_one "$family" "$task"
            ;;
        esac
    done
}

main() {
    local stage="${1:-all}" family_arg="${2:-both}" task_csv="${3:-$DEFAULT_TASKS}"
    local -a families tasks
    case "$stage" in all|train|eval) ;; *) usage; exit 2 ;; esac
    case "$family_arg" in
        both) families=(llama qwen) ;;
        llama|qwen) families=("$family_arg") ;;
        *) usage; exit 2 ;;
    esac
    split_csv "$task_csv"; tasks=("${SPLIT_CSV_RESULT[@]}")
    local task family
    for task in "${tasks[@]}"; do validate_task "$task"; done
    setup_root_log
    prepare_dirs
    log "[INFO] stage=$stage family_arg=$family_arg tasks=${tasks[*]} per_task=$PER_TASK"
    for family in "${families[@]}"; do run_family "$stage" "$family" "${tasks[@]}"; done
    log "[DONE] stage=$stage exp_root=$EXP_ROOT"
}

main "$@"
