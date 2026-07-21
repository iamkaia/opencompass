#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DATA_ROOT="./0527_router_train_dataset"
BERT="./task_classifier_ckpt"
EXPERTS="medmcqa,race,sst2"
MRS_TASKS="medmcqa,race,sst2"
NEW_TASKS=(boolq rte siqa piqa)
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0531_run_mrs_replay50_sequential_hard_routing.sh [llama|qwen|both]

Pipeline:
  1. Build a 50-sample replay cache for MRS tasks.
  2. Reuse the existing 4-new-task cache, or build it if missing.
  3. Start from the existing MRS router checkpoint.
  4. Train boolq -> rte -> siqa -> piqa sequentially with replay.
  5. After each step, run OpenCompass hard_routing for that step's checkpoint.
EOF
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

join_csv() {
    local IFS=,
    echo "$*"
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

mrs_router_for() {
    case "$1" in
        llama) echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

new_cache_for() {
    case "$1" in
        llama) echo "./0527_llama_cache_4other_3expert_official_eval_aligned" ;;
        qwen) echo "./0527_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words" ;;
    esac
}

mrs_replay_cache_for() {
    case "$1" in
        llama) echo "./0527_llama_cache_mrs_replay50_3expert_official_eval_aligned" ;;
        qwen) echo "./0527_qwen3_fp16_cache_mrs_replay50_3expert_official_eval_aligned_sst2words" ;;
    esac
}

train_output_for() {
    local family="$1"
    local step="$2"
    local task="$3"
    printf './router_T0527_%s_replay50_step%02d_%s_correct_conf_ce_t1_%s' \
        "$(family_label "$family")" "$step" "$task" "$(family_suffix "$family")"
}

datasets_for_task() {
    case "$1" in
        boolq)
            echo "SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt"
            ;;
        rte)
            echo "SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen"
            ;;
        siqa)
            echo "siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen"
            ;;
        piqa)
            echo "piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen"
            ;;
    esac
}

ensure_cache() {
    local cache_root="$1"
    local build_cmd="$2"
    local log_file="$3"
    if [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]]; then
        echo "[SKIP] cache=$cache_root"
        return
    fi
    if [[ -e "$cache_root" ]]; then
        echo "cache output exists but is incomplete: $cache_root" >&2
        exit 1
    fi
    echo "[START] cache=$cache_root log=$log_file"
    eval "$build_cmd" > "$log_file" 2>&1
    echo "[DONE] cache=$cache_root log=$log_file"
}

ensure_router() {
    local router_root="$1"
    if [[ ! -f "$router_root/router_heads.pt" || ! -f "$router_root/router_config.json" ]]; then
        echo "missing trained router checkpoint: $router_root" >&2
        exit 1
    fi
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

train_one() {
    local family="$1"
    local step="$2"
    local task="$3"
    local prev_ckpt="$4"
    local mrs_cache="$5"
    local new_cache="$6"
    local sample_tasks="$7"
    local output_dir feature_roots log_file
    local -a common_args

    output_dir="$(train_output_for "$family" "$step" "$task")"
    log_file="T0531_$(family_label "$family")_replay50_step$(printf '%02d' "$step")_${task}.log"
    feature_roots="$mrs_cache,$new_cache"

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
    echo "[START] train family=$family step=$step task=$task load_from=$prev_ckpt log=$log_file output=$output_dir" >&2
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$feature_roots" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --load_from "$prev_ckpt" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        "${common_args[@]}" \
        > "$log_file" 2>&1
    echo "[DONE] train family=$family step=$step task=$task output=$output_dir" >&2
    echo "$output_dir"
}

eval_one() {
    local family="$1"
    local task="$2"
    local router_ckpt="$3"
    local model_config dataset_string log_file
    local -a datasets

    model_config="T0527_replay_hard_routing.py"
    if [[ "$family" == "qwen" ]]; then
        model_config="T0527_replay_sst2words_hard_routing.py"
    fi
    dataset_string="$(datasets_for_task "$task")"
    read -r -a datasets <<< "$dataset_string"
    log_file="T0531_opencompass_$(family_label "$family")_replay50_step_${task}_${RUN_ID}.log"

    echo "[START] eval family=$family task=$task ckpt=$router_ckpt log=$log_file"
    T0531_REPLAY_TASK="$task" T0531_SEQ_ROUTER_CKPT="$router_ckpt" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] eval family=$family task=$task ckpt=$router_ckpt log=$log_file"
}

run_family() {
    local family="$1"
    local mrs_router mrs_cache new_cache base_model log_prefix
    local step task prev_ckpt current_ckpt seen_new_tasks sample_tasks

    require_family "$family"
    mrs_router="$(mrs_router_for "$family")"
    mrs_cache="$(mrs_replay_cache_for "$family")"
    new_cache="$(new_cache_for "$family")"
    base_model="$(base_model_for "$family")"
    log_prefix="$(family_label "$family")"

    ensure_router "$mrs_router"
    ensure_cache "$new_cache" \
        "$PY -u build_cached_router_pair_dataset.py --data_root \"$DATA_ROOT\" --feature_root \"$new_cache\" --task_names boolq,rte,siqa,piqa --expert_names \"$EXPERTS\" --base_model_path \"$base_model\" --router_bert_init \"$BERT\" --batch_size 8 --max_train_samples 200 --max_val_samples 50 --first_layer_idx 0 --middle_layer_idx \"$(middle_layer_for "$family")\" --router_dim 512 --dtype float16 --score_mode official_eval_aligned_generation --chunk_size 2048 --seed 42 --lora_medmcqa \"$(lora_root_for "$family")/sft_medmcqa\" --lora_race \"$(lora_root_for "$family")/sft_race\" --lora_sst2 \"$(lora_root_for "$family")/sft_sst2\"" \
        "T0531_${log_prefix}_build_cache_4other.log"
    ensure_cache "$mrs_cache" \
        "$PY -u build_cached_router_pair_dataset.py --data_root \"$DATA_ROOT\" --feature_root \"$mrs_cache\" --task_names medmcqa,race,sst2 --expert_names \"$EXPERTS\" --base_model_path \"$base_model\" --router_bert_init \"$BERT\" --batch_size 8 --max_train_samples 50 --max_val_samples 50 --first_layer_idx 0 --middle_layer_idx \"$(middle_layer_for "$family")\" --router_dim 512 --dtype float16 --score_mode official_eval_aligned_generation --chunk_size 2048 --seed 42 --lora_medmcqa \"$(lora_root_for "$family")/sft_medmcqa\" --lora_race \"$(lora_root_for "$family")/sft_race\" --lora_sst2 \"$(lora_root_for "$family")/sft_sst2\"" \
        "T0531_${log_prefix}_build_cache_mrs_replay50.log"

    prev_ckpt="$mrs_router"
    seen_new_tasks=()
    step=0
    for task in "${NEW_TASKS[@]}"; do
        step=$((step + 1))
        seen_new_tasks+=("$task")
        sample_tasks="$(join_csv "$MRS_TASKS" "${seen_new_tasks[@]}")"
        current_ckpt="$(train_one "$family" "$step" "$task" "$prev_ckpt" "$mrs_cache" "$new_cache" "$sample_tasks")"
        eval_one "$family" "$task" "$current_ckpt"
        prev_ckpt="$current_ckpt"
    done
}

main() {
    local family="${1:-both}"
    case "$family" in
        llama) run_family llama ;;
        qwen) run_family qwen ;;
        both)
            run_family llama
            run_family qwen
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "${1:-both}"
