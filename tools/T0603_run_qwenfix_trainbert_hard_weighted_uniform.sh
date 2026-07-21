#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_${RUN_STAMP}}"
OLD_ROOT="${OLD_ROOT:-./t0602_taskcls_trainbert_chattemplate_20260603_02}"
LOG_DIR="$EXP_ROOT/logs"
CACHE_DIR="$EXP_ROOT/caches"
ROUTER_DIR="$EXP_ROOT/routers"
OC_DIR="$EXP_ROOT/opencompass"
RECORD_DIR="$EXP_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$EXP_ROOT/T0603_root_${RUN_STAMP}.log}"

OLD_LOG_DIR="$OLD_ROOT/logs"
OLD_OC_DIR="$OLD_ROOT/opencompass"
OLD_RECORD_DIR="$OLD_ROOT/router_records"

DATA_ROOT="${DATA_ROOT:-./0602_router_train_dataset}"
BERT="./task_classifier_ckpt"
EXPERTS="medmcqa,race,sst2"
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"
ALL_CACHE_TASKS="boolq,medmcqa,openbookqa,arc_c,piqa,race,rte,siqa,sst2"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0603_run_qwenfix_trainbert_hard_weighted_uniform.sh [all|qwen_all|qwen_cache|qwen_train|qwen_eval|uniform|llama_uniform|qwen_uniform] [task_csv]

Runs qwen3 task_classifier_ckpt + --train_bert with the Qwen chat-template newline cache fix.
Qwen outputs go to EXP_ROOT, default ./T0603_qwenfix_trainbert_chattemplate_<timestamp>.
Llama uniform outputs go to OLD_ROOT, default ./t0602_taskcls_trainbert_chattemplate_20260603_02.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

setup_root_log() {
    mkdir -p "$(dirname "$ROOT_LOG")"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
    log "[INFO] exp_root=$EXP_ROOT"
    log "[INFO] old_root=$OLD_ROOT"
}

prepare_new_dirs() {
    if [[ -e "$EXP_ROOT" ]]; then
        echo "EXP_ROOT already exists; refusing to overwrite: $EXP_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$CACHE_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
}

prepare_existing_dirs() {
    mkdir -p "$LOG_DIR" "$CACHE_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
}

prepare_old_dirs() {
    [[ -d "$OLD_ROOT" ]] || { echo "missing OLD_ROOT: $OLD_ROOT" >&2; exit 1; }
    mkdir -p "$OLD_LOG_DIR" "$OLD_OC_DIR" "$OLD_RECORD_DIR"
}

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

validate_task() {
    case "$1" in boolq|rte|siqa|piqa|openbookqa|arc_c) ;; *) echo "unsupported task: $1" >&2; exit 2 ;; esac
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
    [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]] || {
        echo "missing completed cache: $cache_root" >&2; exit 1;
    }
}

require_router() {
    local router_root="$1"
    [[ -f "$router_root/router_heads.pt" && -f "$router_root/router_config.json" ]] || {
        echo "missing router checkpoint: $router_root" >&2; exit 1;
    }
}

qwen_cache_root() {
    echo "$CACHE_DIR/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200"
}

qwen_lora_root() {
    echo "./saves/Qwen/Qwen3-4B-Instruct-2507/lora"
}

qwen_router_dir_for() {
    local task="${1:-}"
    if [[ -z "$task" ]]; then
        echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_3expert_sst2words"
    else
        echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_correct_conf_ce_t1_3expert_sst2words"
    fi
}

old_router_dir_for() {
    local family="$1" task="${2:-}"
    case "$family" in
        llama)
            if [[ -z "$task" ]]; then
                echo "$OLD_ROOT/routers/router_T0602_llama_mrs_only_taskcls_trainbert_chattemplate_correct_conf_ce_t1_3expert"
            else
                echo "$OLD_ROOT/routers/router_T0602_llama_mrs_plus_${task}_taskcls_trainbert_chattemplate_correct_conf_ce_t1_3expert"
            fi
            ;;
        qwen)
            if [[ -z "$task" ]]; then
                echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_3expert_sst2words"
            else
                echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_correct_conf_ce_t1_3expert_sst2words"
            fi
            ;;
        *) echo "bad family=$family" >&2; exit 2 ;;
    esac
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

build_qwen_cache() {
    local cache_root log_file lora_root
    require_data_root
    cache_root="$(qwen_cache_root)"
    log_file="$LOG_DIR/T0603_cache_qwen3_fp16_qwenfix_800_200_${RUN_STAMP}.log"
    lora_root="$(qwen_lora_root)"
    [[ ! -e "$cache_root" ]] || { echo "cache output exists: $cache_root" >&2; exit 1; }
    log "[START] qwen cache cache=$cache_root log=$log_file"
    "$PY" -u build_cached_router_pair_dataset.py \
        --data_root "$DATA_ROOT" \
        --feature_root "$cache_root" \
        --task_names "$ALL_CACHE_TASKS" \
        --expert_names "$EXPERTS" \
        --base_model_path "Qwen/Qwen3-4B-Instruct-2507" \
        --router_bert_init "$BERT" \
        --batch_size 8 \
        --max_train_samples 800 \
        --max_val_samples 200 \
        --first_layer_idx 0 \
        --middle_layer_idx 18 \
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
    log "[DONE] qwen cache cache=$cache_root"
}

train_qwen_mrs_only() {
    local cache_root output_dir log_file
    local -a args
    cache_root="$(qwen_cache_root)"
    output_dir="$(qwen_router_dir_for)"
    log_file="$LOG_DIR/T0603_train_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_${RUN_STAMP}.log"
    require_cache "$cache_root"
    [[ ! -e "$output_dir" ]] || { echo "router output exists: $output_dir" >&2; exit 1; }
    mapfile -t args < <(common_training_args)
    log "[START] train qwen mrs_only out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache_root" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train qwen mrs_only out=$output_dir"
}

train_qwen_one() {
    local task="$1" cache_root output_dir log_file
    local -a args
    validate_task "$task"
    cache_root="$(qwen_cache_root)"
    output_dir="$(qwen_router_dir_for "$task")"
    log_file="$LOG_DIR/T0603_train_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_${RUN_STAMP}.log"
    require_cache "$cache_root"
    [[ ! -e "$output_dir" ]] || { echo "router output exists: $output_dir" >&2; exit 1; }
    mapfile -t args < <(common_training_args)
    log "[START] train qwen task=$task out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache_root" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS,$task" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train qwen task=$task out=$output_dir"
}

qwen_model_config() { echo "T0531_mrs_ablation_sst2words_hard_routing.py"; }
llama_model_config() { echo "T0531_mrs_ablation_hard_routing.py"; }

eval_router() {
    local family="$1" label="$2" router_ckpt="$3" routing_mode="$4" out_scope="$5"
    shift 5
    local model_config log_dir oc_dir record_dir family_label log_file work_dir record_path tag
    local -a datasets=("$@")
    require_router "$router_ckpt"
    case "$family" in
        qwen) model_config="$(qwen_model_config)"; family_label="qwen3_fp16" ;;
        llama) model_config="$(llama_model_config)"; family_label="llama" ;;
        *) echo "bad family=$family" >&2; exit 2 ;;
    esac
    if [[ "$out_scope" == "old" ]]; then
        log_dir="$OLD_LOG_DIR"; oc_dir="$OLD_OC_DIR"; record_dir="$OLD_RECORD_DIR"; tag="T0603_uniform_old_${label}"
    else
        log_dir="$LOG_DIR"; oc_dir="$OC_DIR"; record_dir="$RECORD_DIR"; tag="T0603_qwenfix_${label}"
    fi
    mkdir -p "$log_dir" "$oc_dir" "$record_dir"
    log_file="$log_dir/T0603_opencompass_${family_label}_${label}_${routing_mode}_${RUN_STAMP}.log"
    work_dir="$oc_dir/${family_label}_${label}_${routing_mode}_T0603_${RUN_STAMP}"
    record_path="$record_dir/T0603_${family_label}_${label}_${routing_mode}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists: family=$family label=$label mode=$routing_mode" >&2; exit 1;
    }
    log "[START] opencompass family=$family label=$label mode=$routing_mode datasets=${datasets[*]} log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$routing_mode" \
    T0531_ROUTER_RECORD_TAG="$tag" \
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
    local family="$1" routing_mode="$2" out_scope="$3" router_ckpt label
    label="mrs_only_taskcls_trainbert_qwenfix"
    if [[ "$family" == "llama" ]]; then label="mrs_only_taskcls_trainbert_chattemplate"; fi
    router_ckpt="$(old_router_dir_for "$family")"
    eval_router "$family" "$label" "$router_ckpt" "$routing_mode" "$out_scope" \
        SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
}

eval_one() {
    local family="$1" task="$2" routing_mode="$3" out_scope="$4" router_ckpt label task_dataset
    task_dataset="$(task_dataset_for "$task")"
    label="mrs_plus_${task}_taskcls_trainbert_qwenfix"
    if [[ "$family" == "llama" ]]; then label="mrs_plus_${task}_taskcls_trainbert_chattemplate"; fi
    router_ckpt="$(old_router_dir_for "$family" "$task")"
    eval_router "$family" "$label" "$router_ckpt" "$routing_mode" "$out_scope" "$task_dataset" "${MRS_DATASETS[@]}"
}

run_qwen_train() {
    local -a tasks=("$@")
    train_qwen_mrs_only
    local task
    for task in "${tasks[@]}"; do train_qwen_one "$task"; done
}

run_qwen_eval_modes() {
    local -a tasks=("$@")
    local mode task
    for mode in hard weighted_sum; do
        eval_mrs_only qwen "$mode" new
        for task in "${tasks[@]}"; do eval_one qwen "$task" "$mode" new; done
    done
}

run_qwen_uniform() {
    eval_mrs_only qwen uniform new
}

run_llama_uniform() {
    prepare_old_dirs
    eval_mrs_only llama uniform old
}

main() {
    local stage="${1:-all}" task_csv="${2:-$DEFAULT_TASKS}"
    local -a tasks
    case "$stage" in all|qwen_all|qwen_cache|qwen_train|qwen_eval|uniform|llama_uniform|qwen_uniform) ;; *) usage; exit 2 ;; esac
    split_csv "$task_csv"; tasks=("${SPLIT_CSV_RESULT[@]}")
    local task
    for task in "${tasks[@]}"; do validate_task "$task"; done

    case "$stage" in
        all|qwen_all|qwen_cache) prepare_new_dirs ;;
        *) prepare_existing_dirs ;;
    esac
    setup_root_log
    log "[INFO] stage=$stage tasks=${tasks[*]}"

    case "$stage" in
        all|qwen_all)
            build_qwen_cache
            run_qwen_train "${tasks[@]}"
            run_qwen_eval_modes "${tasks[@]}"
            run_qwen_uniform
            run_llama_uniform
            ;;
        qwen_cache)
            build_qwen_cache
            ;;
        qwen_train)
            run_qwen_train "${tasks[@]}"
            ;;
        qwen_eval)
            run_qwen_eval_modes "${tasks[@]}"
            ;;
        qwen_uniform)
            run_qwen_uniform
            ;;
        llama_uniform)
            run_llama_uniform
            ;;
        uniform)
            run_qwen_uniform
            run_llama_uniform
            ;;
    esac
    log "[DONE] stage=$stage exp_root=$EXP_ROOT old_root=$OLD_ROOT"
}

main "$@"
