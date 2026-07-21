#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BERT="${BERT:-./task_classifier_ckpt}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
MRS_TASKS="${MRS_TASKS:-medmcqa,race,sst2}"
DEFAULT_NEW_TASKS="boolq,rte,siqa,piqa"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0531_run_mrs_plus_tasks_from0_trainbert.sh {llama|qwen|both} [task_csv]

Examples:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0531_run_mrs_plus_tasks_from0_trainbert.sh qwen
  CUDA_VISIBLE_DEVICES=1 bash tools/T0531_run_mrs_plus_tasks_from0_trainbert.sh llama boolq,rte
  CUDA_VISIBLE_DEVICES=0 bash tools/T0531_run_mrs_plus_tasks_from0_trainbert.sh qwen openbookqa,arc_c

This trains one router from scratch on MRS plus the requested sample tasks.
It does not pass --load_from, so router heads and pair classifier start fresh.
BERT is fine-tuned with --train_bert.
EOF
}

require_family() {
    case "$1" in
        llama|qwen) ;;
        *) echo "unknown family: $1" >&2; usage; exit 2 ;;
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

mrs_cache_for() {
    case "$1" in
        llama) echo "./0527_llama_cache_mrs_3expert_official_eval_aligned" ;;
        qwen) echo "./0527_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words" ;;
    esac
}

new4_cache_for() {
    case "$1" in
        llama) echo "./0527_llama_cache_4other_3expert_official_eval_aligned" ;;
        qwen) echo "./0527_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words" ;;
    esac
}

arc_obqa_cache_for() {
    case "$1" in
        llama) echo "./0528_llama_cache_arc_c_openbookqa_3expert_official_eval_aligned" ;;
        qwen) echo "./0528_qwen3_fp16_cache_arc_c_openbookqa_3expert_official_eval_aligned_sst2words" ;;
    esac
}

require_cache() {
    local cache_root="$1"
    if [[ ! -f "$cache_root/train/manifest.json" || ! -f "$cache_root/validation/manifest.json" ]]; then
        echo "missing completed cache: $cache_root" >&2
        exit 1
    fi
}

join_csv() {
    local IFS=,
    echo "$*"
}

task_slug() {
    printf '%s' "$1" | tr ',' '_' | tr -c 'A-Za-z0-9_-' '_'
}

contains_task_group() {
    local task_csv="$1"
    local group="$2"
    local IFS=,
    local task
    read -r -a tasks <<< "$task_csv"
    for task in "${tasks[@]}"; do
        case "$group:$task" in
            new4:boolq|new4:rte|new4:siqa|new4:piqa) return 0 ;;
            arc_obqa:arc_c|arc_obqa:openbookqa) return 0 ;;
        esac
    done
    return 1
}

validate_tasks() {
    local task_csv="$1"
    local IFS=,
    local task
    read -r -a tasks <<< "$task_csv"
    for task in "${tasks[@]}"; do
        case "$task" in
            boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
            *)
                echo "unsupported requested task: $task" >&2
                echo "supported: boolq,rte,siqa,piqa,openbookqa,arc_c" >&2
                exit 2
                ;;
        esac
    done
}

feature_roots_for() {
    local family="$1"
    local task_csv="$2"
    local -a roots
    roots=("$(mrs_cache_for "$family")")
    if contains_task_group "$task_csv" new4; then
        roots+=("$(new4_cache_for "$family")")
    fi
    if contains_task_group "$task_csv" arc_obqa; then
        roots+=("$(arc_obqa_cache_for "$family")")
    fi
    join_csv "${roots[@]}"
}

training_args() {
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

run_family() {
    local family="$1"
    local task_csv="$2"
    local feature_roots sample_tasks output_dir log_file
    local -a common_args roots

    require_family "$family"
    validate_tasks "$task_csv"

    feature_roots="$(feature_roots_for "$family" "$task_csv")"
    IFS=, read -r -a roots <<< "$feature_roots"
    for root in "${roots[@]}"; do
        require_cache "$root"
    done

    sample_tasks="$(join_csv "$MRS_TASKS" "$task_csv")"
    output_dir="./router_T0531_$(family_label "$family")_from0_mrs_plus_$(task_slug "$task_csv")_trainbert_correct_conf_ce_t1_$(family_suffix "$family")"
    log_file="T0531_$(family_label "$family")_from0_mrs_plus_$(task_slug "$task_csv")_trainbert.log"

    if [[ -e "$output_dir" ]]; then
        echo "output already exists: $output_dir" >&2
        echo "choose a different task list or remove the obsolete output deliberately" >&2
        exit 1
    fi

    mapfile -t common_args < <(training_args)
    echo "[START] family=$family feature_roots=$feature_roots sample_task_names=$sample_tasks log=$log_file out_dir=$output_dir"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$feature_roots" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        "${common_args[@]}" \
        > "$log_file" 2>&1
    echo "[DONE] family=$family log=$log_file out_dir=$output_dir"
}

main() {
    local family="${1:-}"
    local task_csv="${2:-$DEFAULT_NEW_TASKS}"

    [[ -n "$family" && $# -le 2 ]] || { usage; exit 2; }
    case "$family" in
        llama) run_family llama "$task_csv" ;;
        qwen) run_family qwen "$task_csv" ;;
        both)
            run_family llama "$task_csv"
            run_family qwen "$task_csv"
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "$@"
