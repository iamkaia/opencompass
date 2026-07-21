#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./t0602_bert_ablation_hard_${RUN_STAMP}}"
LOG_DIR="$EXP_ROOT/logs"
CACHE_DIR="$EXP_ROOT/caches"
ROUTER_DIR="$EXP_ROOT/routers"
OC_DIR="$EXP_ROOT/opencompass"
RECORD_DIR="$EXP_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-./t0602_bert_ablation_hard_${RUN_STAMP}.log}"

DATA_ROOT="${DATA_ROOT:-./0602_router_train_dataset}"
EXPERTS="medmcqa,race,sst2"
TASKCLS_BERT="./task_classifier_ckpt"
TINY_BERT="prajjwal1/bert-tiny"
CACHE_BERT="$TASKCLS_BERT"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"
ALL_CACHE_TASKS="boolq,medmcqa,openbookqa,arc_c,piqa,race,rte,siqa,sst2"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/t0602_run_bert_ablation_hard_router.sh [all|cache|train|eval] [llama|qwen|both] [task_csv]

Defaults:
  stage=all
  family=both
  task_csv=boolq,rte,siqa,piqa,openbookqa,arc_c

This builds 0602_router_train_dataset caches with all 9 tasks and exact
sample caps:
  --max_train_samples 800
  --max_val_samples 200

Then, for MRS-only and every MRS+one-task router, it trains four BERT variants:
  task_classifier_ckpt + train_bert
  task_classifier_ckpt + freeze_bert
  prajjwal1/bert-tiny + train_bert
  prajjwal1/bert-tiny + freeze_bert

OpenCompass eval is hard routing only.
EOF
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S %Z"
}

log() {
    echo "[$(timestamp)] $*"
}

setup_root_log() {
    if [[ -e "$ROOT_LOG" ]]; then
        echo "root log already exists: $ROOT_LOG" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$ROOT_LOG")"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
}

prepare_dirs() {
    if [[ -e "$EXP_ROOT" ]]; then
        echo "EXP_ROOT already exists; refusing to overwrite: $EXP_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$CACHE_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
}

require_family() {
    case "$1" in
        llama|qwen) ;;
        *) usage; exit 2 ;;
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
    case "$1" in
        llama) echo 15 ;;
        qwen) echo 18 ;;
    esac
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

require_data_root() {
    local task
    for task in boolq medmcqa openbookqa arc_c piqa race rte siqa sst2; do
        if [[ ! -f "$DATA_ROOT/$task/train.jsonl" || ! -f "$DATA_ROOT/$task/validation.jsonl" ]]; then
            echo "missing 0602 router train dataset files under: $DATA_ROOT/$task" >&2
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

bert_init_for() {
    case "$1" in
        taskcls) echo "$TASKCLS_BERT" ;;
        tiny) echo "$TINY_BERT" ;;
        *) echo "unsupported bert variant: $1" >&2; exit 2 ;;
    esac
}

bert_label_for() {
    case "$1" in
        taskcls) echo "taskcls" ;;
        tiny) echo "tiny" ;;
        *) echo "unsupported bert variant: $1" >&2; exit 2 ;;
    esac
}

train_mode_arg() {
    case "$1" in
        trainbert) printf '%s\n' --train_bert ;;
        freezebert) printf '%s\n' --freeze_bert ;;
        *) echo "unsupported bert train mode: $1" >&2; exit 2 ;;
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
        --save_route_records \
        --eval_train_each_epoch
}

variant_label() {
    local bert_variant="$1"
    local bert_mode="$2"
    echo "$(bert_label_for "$bert_variant")_${bert_mode}"
}

router_dir_for() {
    local family="$1"
    local task="$2"
    local bert_variant="$3"
    local bert_mode="$4"
    echo "$ROUTER_DIR/router_T0602_$(family_label "$family")_mrs_plus_${task}_$(variant_label "$bert_variant" "$bert_mode")_correct_conf_ce_t1_$(family_suffix "$family")"
}

mrs_router_dir_for() {
    local family="$1"
    local bert_variant="$2"
    local bert_mode="$3"
    echo "$ROUTER_DIR/router_T0602_$(family_label "$family")_mrs_only_$(variant_label "$bert_variant" "$bert_mode")_correct_conf_ce_t1_$(family_suffix "$family")"
}

build_cache_one() {
    local family="$1"
    local cache_root log_file lora_root

    require_family "$family"
    require_data_root
    cache_root="$(cache_for "$family")"
    log_file="$LOG_DIR/cache_$(family_label "$family")_0602_9task_800_200_${RUN_STAMP}.log"
    lora_root="$(lora_root_for "$family")"

    if [[ -e "$cache_root" ]]; then
        echo "cache output already exists; refusing to overwrite: $cache_root" >&2
        exit 1
    fi

    log "[START] cache family=$family data_root=$DATA_ROOT cache=$cache_root log=$log_file"
    "$PY" -u build_cached_router_pair_dataset.py \
        --data_root "$DATA_ROOT" \
        --feature_root "$cache_root" \
        --task_names "$ALL_CACHE_TASKS" \
        --expert_names "$EXPERTS" \
        --base_model_path "$(base_model_for "$family")" \
        --router_bert_init "$CACHE_BERT" \
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
    log "[DONE] cache family=$family cache=$cache_root log=$log_file"
}

train_mrs_only() {
    local family="$1"
    local bert_variant="$2"
    local bert_mode="$3"
    local cache_root output_dir log_file bert_init
    local -a args mode_args

    require_family "$family"
    cache_root="$(cache_for "$family")"
    require_cache "$cache_root"
    bert_init="$(bert_init_for "$bert_variant")"
    output_dir="$(mrs_router_dir_for "$family" "$bert_variant" "$bert_mode")"
    log_file="$LOG_DIR/train_$(family_label "$family")_mrs_only_$(variant_label "$bert_variant" "$bert_mode")_${RUN_STAMP}.log"

    if [[ -e "$output_dir" || -e "$log_file" ]]; then
        echo "router output/log already exists; refusing to overwrite: $output_dir $log_file" >&2
        exit 1
    fi

    mapfile -t args < <(common_training_args)
    mapfile -t mode_args < <(train_mode_arg "$bert_mode")
    log "[START] train_mrs_only family=$family bert=$bert_init mode=$bert_mode sample_task_names=$EXPERTS out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache_root" \
        --bert_init "$bert_init" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        "${mode_args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train_mrs_only family=$family variant=$(variant_label "$bert_variant" "$bert_mode") out=$output_dir log=$log_file"
}

train_one() {
    local family="$1"
    local task="$2"
    local bert_variant="$3"
    local bert_mode="$4"
    local cache_root output_dir log_file bert_init
    local -a args mode_args

    require_family "$family"
    validate_task "$task"
    cache_root="$(cache_for "$family")"
    require_cache "$cache_root"
    bert_init="$(bert_init_for "$bert_variant")"
    output_dir="$(router_dir_for "$family" "$task" "$bert_variant" "$bert_mode")"
    log_file="$LOG_DIR/train_$(family_label "$family")_mrs_plus_${task}_$(variant_label "$bert_variant" "$bert_mode")_${RUN_STAMP}.log"

    if [[ -e "$output_dir" || -e "$log_file" ]]; then
        echo "router output/log already exists; refusing to overwrite: $output_dir $log_file" >&2
        exit 1
    fi

    mapfile -t args < <(common_training_args)
    mapfile -t mode_args < <(train_mode_arg "$bert_mode")
    log "[START] train family=$family task=$task bert=$bert_init mode=$bert_mode sample_task_names=$EXPERTS,$task out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache_root" \
        --bert_init "$bert_init" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS,$task" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        "${mode_args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train family=$family task=$task variant=$(variant_label "$bert_variant" "$bert_mode") out=$output_dir log=$log_file"
}

eval_mrs_only() {
    local family="$1"
    local bert_variant="$2"
    local bert_mode="$3"
    local router_ckpt model_config log_file work_dir record_path bert_init variant
    local -a datasets

    require_family "$family"
    variant="$(variant_label "$bert_variant" "$bert_mode")"
    router_ckpt="$(mrs_router_dir_for "$family" "$bert_variant" "$bert_mode")"
    require_router "$router_ckpt"
    bert_init="$(bert_init_for "$bert_variant")"
    model_config="$(model_config_for "$family")"
    datasets=("${MRS_DATASETS[@]}")
    log_file="$LOG_DIR/opencompass_$(family_label "$family")_mrs_only_${variant}_hard_${RUN_STAMP}.log"
    work_dir="$OC_DIR/$(family_label "$family")_mrs_only_${variant}_hard"
    record_path="$RECORD_DIR/$(family_label "$family")_mrs_only_${variant}_hard.jsonl"

    if [[ -e "$work_dir" || -e "$record_path" || -e "$log_file" ]]; then
        echo "eval output already exists; refusing to overwrite: $work_dir $record_path $log_file" >&2
        exit 1
    fi

    log "[START] opencompass_mrs_only family=$family variant=$variant routing=hard datasets=${datasets[*]} work_dir=$work_dir log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$bert_init" \
    T0531_ROUTING_MODE="hard" \
    T0531_ROUTER_RECORD_TAG="t0602_bert_ablation_mrs_only_${variant}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass_mrs_only family=$family variant=$variant routing=hard log=$log_file"
}

eval_one() {
    local family="$1"
    local task="$2"
    local bert_variant="$3"
    local bert_mode="$4"
    local router_ckpt model_config task_dataset log_file work_dir record_path bert_init variant
    local -a datasets

    require_family "$family"
    validate_task "$task"
    variant="$(variant_label "$bert_variant" "$bert_mode")"
    router_ckpt="$(router_dir_for "$family" "$task" "$bert_variant" "$bert_mode")"
    require_router "$router_ckpt"
    bert_init="$(bert_init_for "$bert_variant")"
    model_config="$(model_config_for "$family")"
    task_dataset="$(task_dataset_for "$task")"
    datasets=("$task_dataset" "${MRS_DATASETS[@]}")
    log_file="$LOG_DIR/opencompass_$(family_label "$family")_mrs_plus_${task}_${variant}_hard_${RUN_STAMP}.log"
    work_dir="$OC_DIR/$(family_label "$family")_mrs_plus_${task}_${variant}_hard"
    record_path="$RECORD_DIR/$(family_label "$family")_mrs_plus_${task}_${variant}_hard.jsonl"

    if [[ -e "$work_dir" || -e "$record_path" || -e "$log_file" ]]; then
        echo "eval output already exists; refusing to overwrite: $work_dir $record_path $log_file" >&2
        exit 1
    fi

    log "[START] opencompass family=$family task=$task variant=$variant routing=hard datasets=${datasets[*]} work_dir=$work_dir log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$bert_init" \
    T0531_ROUTING_MODE="hard" \
    T0531_ROUTER_RECORD_TAG="t0602_bert_ablation_mrs_plus_${task}_${variant}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass family=$family task=$task variant=$variant routing=hard log=$log_file"
}

run_variant_for_family() {
    local stage="$1"
    local family="$2"
    local bert_variant="$3"
    local bert_mode="$4"
    shift 4
    local task

    case "$stage" in
        all)
            train_mrs_only "$family" "$bert_variant" "$bert_mode"
            eval_mrs_only "$family" "$bert_variant" "$bert_mode"
            for task in "$@"; do
                train_one "$family" "$task" "$bert_variant" "$bert_mode"
                eval_one "$family" "$task" "$bert_variant" "$bert_mode"
            done
            ;;
        train)
            train_mrs_only "$family" "$bert_variant" "$bert_mode"
            for task in "$@"; do
                train_one "$family" "$task" "$bert_variant" "$bert_mode"
            done
            ;;
        eval)
            eval_mrs_only "$family" "$bert_variant" "$bert_mode"
            for task in "$@"; do
                eval_one "$family" "$task" "$bert_variant" "$bert_mode"
            done
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main() {
    local stage="${1:-all}"
    local family_arg="${2:-both}"
    local task_csv="${3:-$DEFAULT_TASKS}"
    local -a families tasks bert_variants bert_modes
    local family task bert_variant bert_mode

    case "$stage" in
        all|cache|train|eval) ;;
        *) usage; exit 2 ;;
    esac

    case "$family_arg" in
        llama) families=(llama) ;;
        qwen) families=(qwen) ;;
        both) families=(llama qwen) ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$task_csv"
    tasks=("${SPLIT_CSV_RESULT[@]}")
    if [[ "${#tasks[@]}" -eq 0 ]]; then
        echo "empty task list" >&2
        exit 2
    fi
    for task in "${tasks[@]}"; do
        validate_task "$task"
    done

    bert_variants=(taskcls tiny)
    bert_modes=(trainbert freezebert)

    setup_root_log
    prepare_dirs
    log "[INFO] exp_root=$EXP_ROOT"
    log "[INFO] data_root=$DATA_ROOT"
    log "[INFO] stage=$stage families=${families[*]} tasks=${tasks[*]} bert_variants=${bert_variants[*]} bert_modes=${bert_modes[*]} cuda=$CUDA_VISIBLE_DEVICES"

    for family in "${families[@]}"; do
        if [[ "$stage" == "all" || "$stage" == "cache" ]]; then
            build_cache_one "$family"
        fi
        if [[ "$stage" != "cache" ]]; then
            for bert_variant in "${bert_variants[@]}"; do
                for bert_mode in "${bert_modes[@]}"; do
                    run_variant_for_family "$stage" "$family" "$bert_variant" "$bert_mode" "${tasks[@]}"
                done
            done
        fi
    done

    log "[DONE] all outputs are under $EXP_ROOT"
    log "[END] finished_at=$(timestamp)"
}

main "$@"
