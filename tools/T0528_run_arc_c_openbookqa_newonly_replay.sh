#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DATA_ROOT="./0527_router_train_dataset"
BERT="./task_classifier_ckpt"
EXPERTS="medmcqa,race,sst2"
TASKS=(openbookqa arc_c)
MODES=(new_only replay)
FAMILIES=(llama qwen)
ROUTINGS=(weighted_sum hard_routing)
RUN_ID="$(date +"%Y%m%d_%H%M%S")"

LLAMA_MRS_CACHE="./0527_llama_cache_mrs_3expert_official_eval_aligned"
QWEN_MRS_CACHE="./0527_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0528_run_arc_c_openbookqa_newonly_replay.sh [all|cache|train|eval] [llama|qwen|both]

Default: all both

Pipeline:
  1. Build cached router-pair datasets for openbookqa and ARC-c.
  2. Train new_only routers loaded from the existing MRS router.
  3. Train replay routers on MRS + openbookqa/ARC-c.
  4. Run OpenCompass for weighted_sum and hard_routing.
     Each router also runs the original MRS datasets:
     medmcqa_gen_sft_prompt, race_gen_sft_prompt, and sst2_gen.

Every step runs synchronously and writes its own log.
EOF
}

stage="${1:-all}"
family_arg="${2:-both}"

case "$stage" in
    all|cache|train|eval) ;;
    *) usage; exit 2 ;;
esac

case "$family_arg" in
    llama) FAMILIES=(llama) ;;
    qwen) FAMILIES=(qwen) ;;
    both) FAMILIES=(llama qwen) ;;
    *) usage; exit 2 ;;
esac

log_dir="T0528_arc_c_openbookqa_logs_${RUN_ID}"
mkdir -p "$log_dir"

cache_for() {
    case "$1" in
        llama) echo "./0528_llama_cache_arc_c_openbookqa_3expert_official_eval_aligned" ;;
        qwen) echo "./0528_qwen3_fp16_cache_arc_c_openbookqa_3expert_official_eval_aligned_sst2words" ;;
    esac
}

mrs_cache_for() {
    case "$1" in
        llama) echo "$LLAMA_MRS_CACHE" ;;
        qwen) echo "$QWEN_MRS_CACHE" ;;
    esac
}

mrs_router_for() {
    case "$1" in
        llama) echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

router_dir_for() {
    local family="$1"
    local mode="$2"
    local task="$3"
    case "$family:$mode" in
        llama:new_only)
            echo "./router_T0528_llama_new_only_${task}_from_mrs_correct_conf_ce_t1_3expert"
            ;;
        llama:replay)
            echo "./router_T0528_llama_replay_mrs_${task}_correct_conf_ce_t1_3expert"
            ;;
        qwen:new_only)
            echo "./router_T0528_qwen3_fp16_new_only_${task}_from_mrs_correct_conf_ce_t1_3expert_sst2words"
            ;;
        qwen:replay)
            echo "./router_T0528_qwen3_fp16_replay_mrs_${task}_correct_conf_ce_t1_3expert_sst2words"
            ;;
    esac
}

model_config_for() {
    local family="$1"
    local routing="$2"
    case "$family:$routing" in
        llama:weighted_sum) echo "T0528_arc_obqa_weighted_sum_topk.py" ;;
        llama:hard_routing) echo "T0528_arc_obqa_hard_routing.py" ;;
        qwen:weighted_sum) echo "T0528_arc_obqa_sst2words_weighted_sum_topk.py" ;;
        qwen:hard_routing) echo "T0528_arc_obqa_sst2words_hard_routing.py" ;;
    esac
}

dataset_configs_for() {
    case "$1" in
        openbookqa) echo "obqa_main_gen medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen" ;;
        arc_c) echo "ARC_c_gen medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen" ;;
    esac
}

middle_layer_for() {
    case "$1" in
        llama) echo 15 ;;
        qwen) echo 18 ;;
    esac
}

base_model_for() {
    case "$1" in
        llama) echo "meta-llama/Llama-2-7b-chat-hf" ;;
        qwen) echo "Qwen/Qwen3-4B-Instruct-2507" ;;
    esac
}

lora_root_for() {
    case "$1" in
        llama) echo "./saves/llama2-7b-chat-hf/lora" ;;
        qwen) echo "./saves/Qwen/Qwen3-4B-Instruct-2507/lora" ;;
    esac
}

require_manifest() {
    local cache_root="$1"
    [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]]
}

require_completed_cache() {
    local cache_root="$1"
    if ! require_manifest "$cache_root"; then
        echo "missing completed cache: $cache_root" >&2
        exit 1
    fi
}

require_router() {
    local router_root="$1"
    if [[ ! -f "$router_root/router_heads.pt" ]]; then
        echo "missing trained router checkpoint: $router_root/router_heads.pt" >&2
        exit 1
    fi
}

ensure_new_or_completed_cache() {
    local cache_root="$1"
    if [[ -e "$cache_root" ]] && ! require_manifest "$cache_root"; then
        echo "cache output exists but is incomplete: $cache_root" >&2
        exit 1
    fi
}

ensure_new_or_completed_router() {
    local router_root="$1"
    if [[ -e "$router_root" && ! -f "$router_root/router_heads.pt" ]]; then
        echo "router output exists but is incomplete: $router_root" >&2
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

verify_data_root() {
    "$PY" -u tools/verify_router_train_dataset_split_provenance.py \
        --data_root "$DATA_ROOT"
}

run_cache() {
    local family="$1"
    local cache_root log_file lora_root
    cache_root="$(cache_for "$family")"
    log_file="$log_dir/cache_${family}.log"
    lora_root="$(lora_root_for "$family")"
    ensure_new_or_completed_cache "$cache_root"
    if require_manifest "$cache_root"; then
        echo "[SKIP] cache family=$family output=$cache_root"
        return
    fi

    echo "[START] cache family=$family log=$log_file output=$cache_root"
    "$PY" -u build_cached_router_pair_dataset.py \
        --data_root "$DATA_ROOT" \
        --feature_root "$cache_root" \
        --task_names "openbookqa,arc_c" \
        --expert_names "$EXPERTS" \
        --base_model_path "$(base_model_for "$family")" \
        --router_bert_init "$BERT" \
        --batch_size 8 \
        --max_train_samples 200 \
        --max_val_samples 50 \
        --first_layer_idx 0 \
        --middle_layer_idx "$(middle_layer_for "$family")" \
        --router_dim 512 \
        --dtype float16 \
        --score_mode official_eval_aligned_generation \
        --chunk_size 2048 \
        --seed 42 \
        --lora_medmcqa "$lora_root/sft_medmcqa" \
        --lora_race "$lora_root/sft_race" \
        --lora_sst2 "$lora_root/sft_sst2" \
        > "$log_file" 2>&1
    echo "[DONE] cache family=$family log=$log_file output=$cache_root"
}

run_train_one() {
    local family="$1"
    local mode="$2"
    local task="$3"
    local cache_root mrs_cache mrs_router output_dir log_file feature_roots sample_tasks
    local -a common_args load_args
    cache_root="$(cache_for "$family")"
    mrs_cache="$(mrs_cache_for "$family")"
    mrs_router="$(mrs_router_for "$family")"
    output_dir="$(router_dir_for "$family" "$mode" "$task")"
    log_file="$log_dir/train_${family}_${mode}_${task}.log"
    require_completed_cache "$cache_root"
    ensure_new_or_completed_router "$output_dir"
    if [[ -f "$output_dir/router_heads.pt" ]]; then
        echo "[SKIP] train family=$family mode=$mode task=$task output=$output_dir"
        return
    fi

    if [[ "$mode" == "new_only" ]]; then
        require_router "$mrs_router"
        feature_roots="$cache_root"
        sample_tasks="$task"
        load_args=(--load_from "$mrs_router")
    else
        require_completed_cache "$mrs_cache"
        feature_roots="$mrs_cache,$cache_root"
        sample_tasks="$EXPERTS,$task"
        load_args=()
    fi
    mapfile -t common_args < <(training_args)

    echo "[START] train family=$family mode=$mode task=$task log=$log_file output=$output_dir"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$feature_roots" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        "${load_args[@]}" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        "${common_args[@]}" \
        > "$log_file" 2>&1
    echo "[DONE] train family=$family mode=$mode task=$task log=$log_file output=$output_dir"
}

run_eval_one() {
    local family="$1"
    local mode="$2"
    local task="$3"
    local routing="$4"
    local router_root model_config dataset_config_string log_file
    local -a dataset_configs
    router_root="$(router_dir_for "$family" "$mode" "$task")"
    model_config="$(model_config_for "$family" "$routing")"
    dataset_config_string="$(dataset_configs_for "$task")"
    read -r -a dataset_configs <<< "$dataset_config_string"
    log_file="$log_dir/opencompass_${family}_${mode}_${task}_${routing}.log"
    require_router "$router_root"

    echo "[START] opencompass family=$family mode=$mode task=$task routing=$routing datasets=$dataset_config_string log=$log_file"
    T0528_ROUTER_MODE="$mode" T0528_ROUTER_TASK="$task" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${dataset_configs[@]}" \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] opencompass family=$family mode=$mode task=$task routing=$routing log=$log_file"
}

if [[ "$stage" == "all" || "$stage" == "cache" ]]; then
    verify_data_root
    for family in "${FAMILIES[@]}"; do
        run_cache "$family"
    done
fi

if [[ "$stage" == "all" || "$stage" == "train" ]]; then
    for family in "${FAMILIES[@]}"; do
        for mode in "${MODES[@]}"; do
            for task in "${TASKS[@]}"; do
                run_train_one "$family" "$mode" "$task"
            done
        done
    done
fi

if [[ "$stage" == "all" || "$stage" == "eval" ]]; then
    for family in "${FAMILIES[@]}"; do
        for mode in "${MODES[@]}"; do
            for task in "${TASKS[@]}"; do
                for routing in "${ROUTINGS[@]}"; do
                    run_eval_one "$family" "$mode" "$task" "$routing"
                done
            done
        done
    done
fi
