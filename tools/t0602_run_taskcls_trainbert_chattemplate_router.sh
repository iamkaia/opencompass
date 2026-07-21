#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./runs/t0602_taskcls_trainbert_chattemplate_${RUN_STAMP}}"
LOG_DIR="$EXP_ROOT/logs"
CACHE_DIR="$EXP_ROOT/caches"
ROUTER_DIR="$EXP_ROOT/routers"
OC_DIR="$EXP_ROOT/opencompass"
RECORD_DIR="$EXP_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$EXP_ROOT/root_${RUN_STAMP}.log}"

DATA_ROOT="${DATA_ROOT:-./0602_router_train_dataset}"
EXPERTS="medmcqa,race,sst2"
BERT="./task_classifier_ckpt"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"
ALL_CACHE_TASKS="boolq,medmcqa,openbookqa,arc_c,piqa,race,rte,siqa,sst2"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/t0602_run_taskcls_trainbert_chattemplate_router.sh [all|cache|train|eval] [llama|qwen|both] [task_csv]

Runs only:
  bert_init=./task_classifier_ckpt
  --train_bert
  routing_mode=hard

MRS-only OpenCompass eval runs all 9 datasets.
MRS+one-task replay eval runs MRS + that new task only.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

setup_root_log() {
    mkdir -p "$(dirname "$ROOT_LOG")"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
    log "[INFO] exp_root=$EXP_ROOT"
}

prepare_dirs() {
    if [[ -e "$EXP_ROOT" ]]; then
        local root_log_abs exp_root_abs root_log_dir_abs entry allowed
        root_log_abs="$(readlink -f "$ROOT_LOG")"
        exp_root_abs="$(readlink -f "$EXP_ROOT")"
        root_log_dir_abs="$(dirname "$root_log_abs")"
        allowed=1
        shopt -s nullglob dotglob
        for entry in "$EXP_ROOT"/*; do
            if [[ "$root_log_dir_abs" == "$exp_root_abs" && "$(readlink -f "$entry")" == "$root_log_abs" ]]; then
                continue
            fi
            allowed=0
            break
        done
        shopt -u nullglob dotglob
        if [[ "$allowed" != 1 ]]; then
            echo "EXP_ROOT already exists; refusing to overwrite: $EXP_ROOT" >&2
            exit 1
        fi
    fi
    mkdir -p "$LOG_DIR" "$CACHE_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
}

require_family() {
    case "$1" in llama|qwen) ;; *) usage; exit 2 ;; esac
}

family_label() {
    case "$1" in llama) echo "llama" ;; qwen) echo "qwen3_fp16" ;; esac
}

family_suffix() {
    case "$1" in llama) echo "3expert" ;; qwen) echo "3expert_sst2words" ;; esac
}

cache_for() {
    case "$1" in
        llama) echo "$CACHE_DIR/llama_0602_9task_3expert_official_eval_aligned_chattemplate_800_200" ;;
        qwen) echo "$CACHE_DIR/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_800_200" ;;
    esac
}

base_model_for() {
    case "$1" in
        llama) echo "meta-llama/Llama-2-7b-chat-hf" ;;
        qwen) echo "Qwen/Qwen3-4B-Instruct-2507" ;;
    esac
}

middle_layer_for() {
    case "$1" in llama) echo 15 ;; qwen) echo 18 ;; esac
}

lora_root_for() {
    case "$1" in
        llama) echo "./saves/llama2-7b-chat-hf/lora" ;;
        qwen) echo "./saves/Qwen/Qwen3-4B-Instruct-2507/lora" ;;
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

require_data_root() {
    local task
    for task in boolq medmcqa openbookqa arc_c piqa race rte siqa sst2; do
        if [[ ! -f "$DATA_ROOT/$task/train.jsonl" || ! -f "$DATA_ROOT/$task/validation.jsonl" ]]; then
            echo "missing DATA_ROOT files under: $DATA_ROOT/$task" >&2
            exit 1
        fi
    done
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
        --train_bert \
        --save_route_records \
        --eval_train_each_epoch
}

mrs_router_dir_for() {
    local family="$1"
    echo "$ROUTER_DIR/router_T0602_$(family_label "$family")_mrs_only_taskcls_trainbert_chattemplate_correct_conf_ce_t1_$(family_suffix "$family")"
}

router_dir_for() {
    local family="$1"
    local task="$2"
    echo "$ROUTER_DIR/router_T0602_$(family_label "$family")_mrs_plus_${task}_taskcls_trainbert_chattemplate_correct_conf_ce_t1_$(family_suffix "$family")"
}

build_cache_one() {
    local family="$1" cache_root log_file lora_root
    require_family "$family"
    require_data_root
    cache_root="$(cache_for "$family")"
    log_file="$LOG_DIR/cache_$(family_label "$family")_chattemplate_800_200_${RUN_STAMP}.log"
    lora_root="$(lora_root_for "$family")"

    if [[ -e "$cache_root" ]]; then
        echo "cache output already exists; refusing to overwrite: $cache_root" >&2
        exit 1
    fi

    log "[START] cache family=$family cache=$cache_root log=$log_file"
    "$PY" -u build_cached_router_pair_dataset.py \
        --data_root "$DATA_ROOT" \
        --feature_root "$cache_root" \
        --task_names "$ALL_CACHE_TASKS" \
        --expert_names "$EXPERTS" \
        --base_model_path "$(base_model_for "$family")" \
        --router_bert_init "$BERT" \
        --batch_size 8 \
        --max_train_samples 800 \
        --max_val_samples 200 \
        --first_layer_idx 0 \
        --middle_layer_idx "$(middle_layer_for "$family")" \
        --router_dim 512 \
        --dtype float16 \
        --score_mode official_eval_aligned_generation \
        --cache_prompt_template chat_template \
        --chunk_size 2048 \
        --seed 42 \
        --lora_medmcqa "$lora_root/sft_medmcqa" \
        --lora_race "$lora_root/sft_race" \
        --lora_sst2 "$lora_root/sft_sst2" \
        > "$log_file" 2>&1
    log "[DONE] cache family=$family cache=$cache_root"
}

train_mrs_only() {
    local family="$1" cache_root output_dir log_file
    local -a args
    cache_root="$(cache_for "$family")"
    require_cache "$cache_root"
    output_dir="$(mrs_router_dir_for "$family")"
    log_file="$LOG_DIR/train_$(family_label "$family")_mrs_only_taskcls_trainbert_chattemplate_${RUN_STAMP}.log"
    [[ ! -e "$output_dir" ]] || { echo "router output exists: $output_dir" >&2; exit 1; }
    mapfile -t args < <(common_training_args)
    log "[START] train_mrs_only family=$family out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache_root" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train_mrs_only family=$family out=$output_dir"
}

train_one() {
    local family="$1" task="$2" cache_root output_dir log_file
    local -a args
    validate_task "$task"
    cache_root="$(cache_for "$family")"
    require_cache "$cache_root"
    output_dir="$(router_dir_for "$family" "$task")"
    log_file="$LOG_DIR/train_$(family_label "$family")_mrs_plus_${task}_taskcls_trainbert_chattemplate_${RUN_STAMP}.log"
    [[ ! -e "$output_dir" ]] || { echo "router output exists: $output_dir" >&2; exit 1; }
    mapfile -t args < <(common_training_args)
    log "[START] train family=$family task=$task out=$output_dir log=$log_file"
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
    local family="$1" label="$2" router_ckpt="$3"
    shift 3
    local model_config log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router_ckpt"
    model_config="$(model_config_for "$family")"
    log_file="$LOG_DIR/opencompass_$(family_label "$family")_${label}_hard_${RUN_STAMP}.log"
    work_dir="$OC_DIR/$(family_label "$family")_${label}_hard"
    record_path="$RECORD_DIR/$(family_label "$family")_${label}_hard.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists for $family $label" >&2; exit 1;
    }
    log "[START] opencompass family=$family label=$label datasets=${datasets[*]} log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="hard" \
    T0531_ROUTER_RECORD_TAG="t0602_chattemplate_${label}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass family=$family label=$label log=$log_file"
}

eval_mrs_only() {
    local family="$1"
    local -a datasets
    datasets=(
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
    eval_router "$family" "mrs_only_taskcls_trainbert_chattemplate_alltasks" "$(mrs_router_dir_for "$family")" "${datasets[@]}"
}

eval_one() {
    local family="$1" task="$2" task_dataset
    task_dataset="$(task_dataset_for "$task")"
    eval_router "$family" "mrs_plus_${task}_taskcls_trainbert_chattemplate" "$(router_dir_for "$family" "$task")" "$task_dataset" "${MRS_DATASETS[@]}"
}

run_family() {
    local stage="$1" family="$2"
    local -a tasks=("${@:3}")
    case "$stage" in
        all|cache) build_cache_one "$family" ;;
    esac
    case "$stage" in
        all|train)
            train_mrs_only "$family"
            local task
            for task in "${tasks[@]}"; do train_one "$family" "$task"; done
            ;;
    esac
    case "$stage" in
        all|eval)
            eval_mrs_only "$family"
            local task
            for task in "${tasks[@]}"; do eval_one "$family" "$task"; done
            ;;
    esac
}

main() {
    local stage="${1:-all}" family_arg="${2:-both}" task_csv="${3:-$DEFAULT_TASKS}"
    local -a families tasks
    case "$stage" in all|cache|train|eval) ;; *) usage; exit 2 ;; esac
    case "$family_arg" in
        both) families=(llama qwen) ;;
        llama|qwen) families=("$family_arg") ;;
        *) usage; exit 2 ;;
    esac
    split_csv "$task_csv"; tasks=("${SPLIT_CSV_RESULT[@]}")
    local task family
    for task in "${tasks[@]}"; do validate_task "$task"; done
    if [[ "$stage" == "all" || "$stage" == "cache" ]]; then prepare_dirs; else mkdir -p "$LOG_DIR" "$CACHE_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"; fi
    setup_root_log
    log "[INFO] stage=$stage family_arg=$family_arg tasks=${tasks[*]}"
    for family in "${families[@]}"; do run_family "$stage" "$family" "${tasks[@]}"; done
    log "[DONE] stage=$stage exp_root=$EXP_ROOT"
}

main "$@"
