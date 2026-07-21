#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
BERT="./task_classifier_ckpt"
EXPERTS="medmcqa,race,sst2"
NEW_TASKS=(boolq rte siqa piqa)

LLAMA_NEW_CACHE="./0525_llama_cache_4other_3expert_official_eval_aligned"
LLAMA_MRS_CACHE="./0525_llama_cache_mrs_3expert_official_eval_aligned"
LLAMA_BASE_ROUTER="./router_0525_llama_correct_conf_ce_t1_mrs_3expert"

QWEN_NEW_CACHE="./0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words"
QWEN_MRS_CACHE="./0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words"
QWEN_BASE_ROUTER="./router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert_sst2words"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0527_run_new_only_router_training.sh train {llama|qwen} {boolq|rte|siqa|piqa}
  CUDA_VISIBLE_DEVICES=0 bash tools/T0527_run_new_only_router_training.sh eval_original {llama|qwen} {boolq|rte|siqa|piqa}
  bash tools/T0527_run_new_only_router_training.sh inspect {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val} [extra inspector args]
  bash tools/T0527_run_new_only_router_training.sh inspect_original {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val} [extra inspector args]
  bash tools/T0527_run_new_only_router_training.sh export {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val}
  bash tools/T0527_run_new_only_router_training.sh export_original {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val}
EOF
}

require_task() {
    local requested="$1"
    local task
    for task in "${NEW_TASKS[@]}"; do
        [[ "$requested" == "$task" ]] && return 0
    done
    echo "unknown new task: $requested" >&2
    usage
    exit 2
}

new_cache_for() {
    case "$1" in
        llama) echo "$LLAMA_NEW_CACHE" ;;
        qwen) echo "$QWEN_NEW_CACHE" ;;
        *) echo "unknown model: $1" >&2; exit 2 ;;
    esac
}

mrs_cache_for() {
    case "$1" in
        llama) echo "$LLAMA_MRS_CACHE" ;;
        qwen) echo "$QWEN_MRS_CACHE" ;;
        *) echo "unknown model: $1" >&2; exit 2 ;;
    esac
}

router_dir_for() {
    local family="$1"
    local task="$2"
    case "$family" in
        llama) echo "./router_T0527_llama_newonly_${task}_from_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_newonly_${task}_from_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
        *) echo "unknown model: $family" >&2; exit 2 ;;
    esac
}

require_cache() {
    local cache_root="$1"
    if [[ ! -f "$cache_root/train/manifest.json" || ! -f "$cache_root/validation/manifest.json" ]]; then
        echo "missing completed cache: $cache_root" >&2
        exit 1
    fi
}

run_original_eval() {
    local family="$1"
    local trained_task="$2"
    local cache trained_router output_dir log_file
    require_task "$trained_task"
    cache="$(mrs_cache_for "$family")"
    require_cache "$cache"
    trained_router="$(router_dir_for "$family" "$trained_task")"
    output_dir="${trained_router}_original_mrs_records"
    log_file="T0527_${family}_router_newonly_${trained_task}_eval_original_mrs.log"
    nohup "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --load_from "$trained_router" \
        --sample_task_names medmcqa,race,sst2 \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 1.0 \
        --pair_loss_normalization sample_minmax \
        --freeze_bert \
        --save_route_records \
        --eval_only \
        > "$log_file" 2>&1 &
    echo "started original-task eval: $log_file out_dir=$output_dir"
}

start_train() {
    local family="$1"
    local task="$2"
    local cache base_router output_dir log_file
    require_task "$task"
    cache="$(new_cache_for "$family")"
    require_cache "$cache"
    output_dir="$(router_dir_for "$family" "$task")"
    case "$family" in
        llama)
            base_router="$LLAMA_BASE_ROUTER"
            log_file="T0527_llama_router_newonly_${task}_from_mrs_3expert.log"
            ;;
        qwen)
            base_router="$QWEN_BASE_ROUTER"
            log_file="T0527_qwen3_fp16_router_newonly_${task}_from_mrs_3expert_sst2words.log"
            ;;
    esac

    nohup "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --load_from "$base_router" \
        --sample_task_names "$task" \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --batch_size 32 \
        --epochs 10 \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 1.0 \
        --pair_loss_normalization sample_minmax \
        --best_metric route_correct_acc \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1 &
    echo "started train: $log_file out_dir=$output_dir sample_task=$task"
}

best_records_path() {
    local output_dir="$1"
    local split="$2"
    local epoch
    epoch="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_epoch"])' "$output_dir/best_metrics.json")"
    echo "$output_dir/route_records_${split}_epoch${epoch}.json"
}

inspect_records() {
    local family="$1"
    local task="$2"
    local split="$3"
    shift 3
    require_task "$task"
    exec "$PY" tools/T0527_inspect_new_only_router_records.py \
        --run_dir "$(router_dir_for "$family" "$task")" \
        --split "$split" \
        "$@"
}

inspect_original_records() {
    local family="$1"
    local task="$2"
    local split="$3"
    shift 3
    require_task "$task"
    exec "$PY" tools/T0527_inspect_new_only_router_records.py \
        --run_dir "$(router_dir_for "$family" "$task")_original_mrs_records" \
        --split "$split" \
        --eval_only \
        "$@"
}

export_details() {
    local family="$1"
    local task="$2"
    local split="$3"
    local cache output_dir records_path cache_split output_json
    require_task "$task"
    cache="$(new_cache_for "$family")"
    output_dir="$(router_dir_for "$family" "$task")"
    records_path="$(best_records_path "$output_dir" "$split")"
    if [[ "$split" == "train_eval" ]]; then
        cache_split="train"
    else
        cache_split="validation"
    fi
    output_json="T0527_${family}_newonly_${task}_${split}_sample_details.json"
    "$PY" tools/export_router_sample_details.py \
        --cache_roots "$cache" \
        --records_path "$records_path" \
        --split "$cache_split" \
        --sample_tasks "$task" \
        --out "$output_json"
}

export_original_details() {
    local family="$1"
    local task="$2"
    local split="$3"
    local cache output_dir records_path cache_split output_json
    require_task "$task"
    cache="$(mrs_cache_for "$family")"
    output_dir="$(router_dir_for "$family" "$task")_original_mrs_records"
    if [[ "$split" == "train_eval" ]]; then
        cache_split="train"
        records_path="$output_dir/route_records_train_eval_only.json"
    else
        cache_split="validation"
        records_path="$output_dir/route_records_val_eval_only.json"
    fi
    output_json="T0527_${family}_newonly_${task}_original_mrs_${split}_sample_details.json"
    "$PY" tools/export_router_sample_details.py \
        --cache_roots "$cache" \
        --records_path "$records_path" \
        --split "$cache_split" \
        --sample_tasks medmcqa,race,sst2 \
        --out "$output_json"
}

case "${1:-}" in
    train)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        start_train "$2" "$3"
        ;;
    eval_original)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        run_original_eval "$2" "$3"
        ;;
    inspect)
        [[ $# -ge 4 ]] || { usage; exit 2; }
        inspect_records "$2" "$3" "$4" "${@:5}"
        ;;
    inspect_original)
        [[ $# -ge 4 ]] || { usage; exit 2; }
        inspect_original_records "$2" "$3" "$4" "${@:5}"
        ;;
    export)
        [[ $# -eq 4 ]] || { usage; exit 2; }
        export_details "$2" "$3" "$4"
        ;;
    export_original)
        [[ $# -eq 4 ]] || { usage; exit 2; }
        export_original_details "$2" "$3" "$4"
        ;;
    *)
        usage
        exit 2
        ;;
esac
