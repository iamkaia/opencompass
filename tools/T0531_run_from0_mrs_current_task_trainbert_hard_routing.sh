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
DEFAULT_NEW_TASKS=(boolq rte siqa piqa)
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0531_run_from0_mrs_current_task_trainbert_hard_routing.sh {llama|qwen|both} [boolq,rte,siqa,piqa]

Protocol:
  step01 boolq: train from 0 on medmcqa,race,sst2,boolq, then run OpenCompass.
  step02 rte:   train from 0 on medmcqa,race,sst2,rte, then run OpenCompass.
  step03 siqa:  train from 0 on medmcqa,race,sst2,siqa, then run OpenCompass.
  step04 piqa:  train from 0 on medmcqa,race,sst2,piqa, then run OpenCompass.

Important:
  No --load_from is passed. Every step starts from fresh router heads.
  --train_bert is passed, so BERT is fine-tuned with the router heads.
  Full 0527 MRS and 4other caches are used; each selected task contributes all
  cached rows from that split.
EOF
}

require_family() {
    case "$1" in
        llama|qwen) ;;
        *) echo "unknown family: $1" >&2; usage; exit 2 ;;
    esac
}

require_task() {
    case "$1" in
        boolq|rte|siqa|piqa) ;;
        *) echo "unknown task: $1" >&2; usage; exit 2 ;;
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

new_cache_for() {
    case "$1" in
        llama) echo "./0527_llama_cache_4other_3expert_official_eval_aligned" ;;
        qwen) echo "./0527_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words" ;;
    esac
}

model_config_for() {
    case "$1" in
        llama) echo "T0527_replay_hard_routing.py" ;;
        qwen) echo "T0527_replay_sst2words_hard_routing.py" ;;
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

step_output_for() {
    local family="$1"
    local step="$2"
    local task="$3"
    printf './router_T0531_%s_from0_mrs_current_trainbert_step%02d_%s_correct_conf_ce_t1_%s' \
        "$(family_label "$family")" "$step" "$task" "$(family_suffix "$family")"
}

datasets_for_task() {
    case "$1" in
        boolq) echo "SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen" ;;
        rte) echo "SuperGLUE_RTE_gen medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen" ;;
        siqa) echo "siqa_gen medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen" ;;
        piqa) echo "piqa_gen medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen" ;;
    esac
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

wandb_args() {
    [[ "${WANDB:-0}" == "1" ]] || return 0
    printf '%s\n' \
        --wandb \
        --wandb_project "${WANDB_PROJECT:-0531_router_from0_mrs_current_trainbert}" \
        --wandb_group "${WANDB_GROUP:-T0531_from0_mrs_current_trainbert}" \
        --wandb_tags "${WANDB_TAGS:-from0,mrs,current_task,trainbert}"
}

train_one() {
    local family="$1"
    local step="$2"
    local task="$3"
    local mrs_cache="$4"
    local new_cache="$5"
    local sample_tasks output_dir log_file feature_roots
    local -a common_args wb_args

    sample_tasks="$(join_csv "$MRS_TASKS" "$task")"
    feature_roots="$mrs_cache,$new_cache"
    output_dir="$(step_output_for "$family" "$step" "$task")"
    log_file="T0531_$(family_label "$family")_from0_mrs_current_trainbert_step$(printf '%02d' "$step")_${task}.log"

    if [[ -e "$output_dir" && ! -f "$output_dir/router_heads.pt" ]]; then
        echo "router output exists but is incomplete: $output_dir" >&2
        exit 1
    fi
    if [[ -f "$output_dir/router_heads.pt" ]]; then
        echo "[SKIP] train family=$family step=$step task=$task output=$output_dir" >&2
        echo "$output_dir"
        return
    fi

    mapfile -t common_args < <(training_args)
    mapfile -t wb_args < <(wandb_args)

    echo "[START] train family=$family step=$step task=$task from0=1 sample_task_names=$sample_tasks log=$log_file output=$output_dir" >&2
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$feature_roots" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        "${common_args[@]}" \
        "${wb_args[@]}" \
        > "$log_file" 2>&1
    echo "[DONE] train family=$family step=$step task=$task output=$output_dir" >&2
    echo "$output_dir"
}

eval_one() {
    local family="$1"
    local step="$2"
    local task="$3"
    local router_ckpt="$4"
    local model_config dataset_string log_file record_tag
    local -a datasets

    model_config="$(model_config_for "$family")"
    dataset_string="$(datasets_for_task "$task")"
    read -r -a datasets <<< "$dataset_string"
    record_tag="from0_mrs_current_trainbert_step$(printf '%02d' "$step")"
    log_file="T0531_opencompass_$(family_label "$family")_from0_mrs_current_trainbert_step$(printf '%02d' "$step")_${task}_${RUN_ID}.log"

    echo "[START] eval family=$family step=$step task=$task ckpt=$router_ckpt log=$log_file"
    T0531_REPLAY_TASK="$task" \
    T0531_SEQ_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTER_RECORD_TAG="$record_tag" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] eval family=$family step=$step task=$task ckpt=$router_ckpt log=$log_file"
}

run_family() {
    local family="$1"
    local task_csv="$2"
    local mrs_cache new_cache step task ckpt
    local -a tasks

    require_family "$family"
    IFS=, read -r -a tasks <<< "$task_csv"
    for task in "${tasks[@]}"; do
        require_task "$task"
    done

    mrs_cache="$(mrs_cache_for "$family")"
    new_cache="$(new_cache_for "$family")"
    require_cache "$mrs_cache"
    require_cache "$new_cache"

    echo "[SEQ_START] family=$family tasks=$(join_csv "${tasks[@]}") mrs_cache=$mrs_cache new_cache=$new_cache"
    step=0
    for task in "${tasks[@]}"; do
        step=$((step + 1))
        ckpt="$(train_one "$family" "$step" "$task" "$mrs_cache" "$new_cache")"
        eval_one "$family" "$step" "$task" "$ckpt"
    done
    echo "[SEQ_DONE] family=$family"
}

main() {
    local family="${1:-}"
    local task_csv="${2:-$(join_csv "${DEFAULT_NEW_TASKS[@]}")}"

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
