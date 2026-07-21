#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BERT="${BERT:-./task_classifier_ckpt}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
MRS_TASKS="${MRS_TASKS:-medmcqa,race,sst2}"
DEFAULT_NEW_TASKS=(boolq rte siqa piqa)

LLAMA_MRS_CACHE="./0527_llama_cache_mrs_3expert_official_eval_aligned"
LLAMA_NEW_CACHE="./0527_llama_cache_4other_3expert_official_eval_aligned"
QWEN_MRS_CACHE="./0527_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words"
QWEN_NEW_CACHE="./0527_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0527_run_sequential_router_training.sh train {new_only|replay} {llama|qwen} [boolq,rte,siqa,piqa]
  CUDA_VISIBLE_DEVICES=0 EVAL_AFTER_STEP=0 bash tools/T0527_run_sequential_router_training.sh train {new_only|replay} {llama|qwen}

Protocol:
  new_only: MRS checkpoint -> current new task only -> next new task only -> ...
  replay:   MRS checkpoint -> MRS+seen new tasks -> MRS+seen new tasks -> ...

Default task order is boolq,rte,siqa,piqa.
Each step loads the previous sequential checkpoint and writes an eval_seen run
unless EVAL_AFTER_STEP=0 is set.
EOF
}

require_mode() {
    case "$1" in
        new_only|replay) ;;
        *) echo "unknown mode: $1" >&2; usage; exit 2 ;;
    esac
}

require_family() {
    case "$1" in
        llama|qwen) ;;
        *) echo "unknown family: $1" >&2; usage; exit 2 ;;
    esac
}

cache_mrs_for() {
    case "$1" in
        llama) echo "$LLAMA_MRS_CACHE" ;;
        qwen) echo "$QWEN_MRS_CACHE" ;;
    esac
}

cache_new_for() {
    case "$1" in
        llama) echo "$LLAMA_NEW_CACHE" ;;
        qwen) echo "$QWEN_NEW_CACHE" ;;
    esac
}

mrs_router_for() {
    case "$1" in
        llama) echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
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
        echo "missing trained router checkpoint: $router_root" >&2
        exit 1
    fi
}

require_new_output() {
    local output_dir="$1"
    if [[ -e "$output_dir" ]]; then
        echo "output already exists: $output_dir" >&2
        echo "choose a fresh run name or remove the obsolete output deliberately" >&2
        exit 1
    fi
}

join_csv() {
    local IFS=,
    echo "$*"
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
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch
}

wandb_args() {
    [[ "${WANDB:-0}" == "1" ]] || return 0
    printf '%s\n' \
        --wandb \
        --wandb_project "${WANDB_PROJECT:-0527_router_sequential}" \
        --wandb_group "${WANDB_GROUP:-T0527_sequential}" \
        --wandb_tags "${WANDB_TAGS:-sequential,router,T0527}"
}

step_output_dir() {
    local mode="$1"
    local family="$2"
    local step="$3"
    local task="$4"
    printf './router_T0527_%s_sequential_%s_step%02d_%s_correct_conf_ce_t1_%s' \
        "$(family_label "$family")" "$mode" "$step" "$task" "$(family_suffix "$family")"
}

run_eval_seen() {
    local mode="$1"
    local family="$2"
    local step="$3"
    local task="$4"
    local ckpt="$5"
    local seen_csv="$6"
    local mrs_cache="$7"
    local new_cache="$8"
    local feature_roots output_dir log_file

    feature_roots="$mrs_cache,$new_cache"
    output_dir="${ckpt}_eval_seen"
    log_file="T0527_$(family_label "$family")_sequential_${mode}_step$(printf '%02d' "$step")_${task}_eval_seen.log"
    require_new_output "$output_dir"

    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$feature_roots" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --load_from "$ckpt" \
        --sample_task_names "$seen_csv" \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 1.0 \
        --pair_loss_normalization sample_minmax \
        --freeze_bert \
        --save_route_records \
        --eval_only \
        > "$log_file" 2>&1
    echo "[EVAL_DONE] step=$step task=$task log=$log_file out_dir=$output_dir sample_task_names=$seen_csv"
}

run_train() {
    local mode="$1"
    local family="$2"
    local task_csv="${3:-}"
    local mrs_cache new_cache prev_ckpt
    local -a tasks common_args wb_args seen_new_tasks
    local step task feature_roots sample_tasks output_dir log_file seen_csv

    require_mode "$mode"
    require_family "$family"

    if [[ -n "$task_csv" ]]; then
        IFS=, read -r -a tasks <<< "$task_csv"
    else
        tasks=("${DEFAULT_NEW_TASKS[@]}")
    fi

    mrs_cache="$(cache_mrs_for "$family")"
    new_cache="$(cache_new_for "$family")"
    prev_ckpt="$(mrs_router_for "$family")"
    require_cache "$mrs_cache"
    require_cache "$new_cache"
    require_router "$prev_ckpt"
    mapfile -t common_args < <(training_args)
    mapfile -t wb_args < <(wandb_args)

    echo "[SEQ_START] mode=$mode family=$family base=$prev_ckpt tasks=$(join_csv "${tasks[@]}")"
    seen_new_tasks=()
    step=0
    for task in "${tasks[@]}"; do
        step=$((step + 1))
        seen_new_tasks+=("$task")

        if [[ "$mode" == "new_only" ]]; then
            feature_roots="$new_cache"
            sample_tasks="$task"
        else
            feature_roots="$mrs_cache,$new_cache"
            sample_tasks="$(join_csv "$MRS_TASKS" "${seen_new_tasks[@]}")"
        fi

        output_dir="$(step_output_dir "$mode" "$family" "$step" "$task")"
        log_file="T0527_$(family_label "$family")_sequential_${mode}_step$(printf '%02d' "$step")_${task}.log"
        require_new_output "$output_dir"

        echo "[TRAIN_START] step=$step task=$task load_from=$prev_ckpt sample_task_names=$sample_tasks log=$log_file"
        "$PY" -u train_internal_two_router_compact_cached_joint.py \
            --feature_roots "$feature_roots" \
            --bert_init "$BERT" \
            --out_dir "$output_dir" \
            --load_from "$prev_ckpt" \
            --sample_task_names "$sample_tasks" \
            --expert_names "$EXPERTS" \
            "${common_args[@]}" \
            "${wb_args[@]}" \
            > "$log_file" 2>&1
        echo "[TRAIN_DONE] step=$step task=$task out_dir=$output_dir"

        prev_ckpt="$output_dir"
        if [[ "${EVAL_AFTER_STEP:-1}" == "1" ]]; then
            seen_csv="$(join_csv "$MRS_TASKS" "${seen_new_tasks[@]}")"
            run_eval_seen "$mode" "$family" "$step" "$task" "$prev_ckpt" "$seen_csv" "$mrs_cache" "$new_cache"
        fi
    done
    echo "[SEQ_DONE] final_checkpoint=$prev_ckpt"
}

[[ $# -ge 3 && $# -le 4 ]] || { usage; exit 2; }
cmd="$1"
case "$cmd" in
    train)
        run_train "$2" "$3" "${4:-}"
        ;;
    *)
        echo "unknown command: $cmd" >&2
        usage
        exit 2
        ;;
esac
