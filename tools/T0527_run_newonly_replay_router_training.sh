#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BERT="./task_classifier_ckpt"
EXPERTS="medmcqa,race,sst2"
NEW_TASKS=(boolq rte siqa piqa)

LLAMA_MRS_CACHE="./0527_llama_cache_mrs_3expert_official_eval_aligned"
LLAMA_NEW_CACHE="./0527_llama_cache_4other_3expert_official_eval_aligned"
QWEN_MRS_CACHE="./0527_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words"
QWEN_NEW_CACHE="./0527_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0527_run_newonly_replay_router_training.sh train_mrs {llama|qwen}
  CUDA_VISIBLE_DEVICES=0 bash tools/T0527_run_newonly_replay_router_training.sh train {new_only|replay} {llama|qwen} {boolq|rte|siqa|piqa}
  CUDA_VISIBLE_DEVICES=0 bash tools/T0527_run_newonly_replay_router_training.sh eval_mrs {new_only|replay} {llama|qwen} {boolq|rte|siqa|piqa}
  bash tools/T0527_run_newonly_replay_router_training.sh inspect_mrs_baseline {llama|qwen} {train_eval|val} [extra inspector args]
  bash tools/T0527_run_newonly_replay_router_training.sh inspect {new_only|replay} {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val} [extra inspector args]
  bash tools/T0527_run_newonly_replay_router_training.sh inspect_mrs {new_only|replay} {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val} [extra inspector args]
  bash tools/T0527_run_newonly_replay_router_training.sh export_mrs_baseline {llama|qwen} {train_eval|val}
  bash tools/T0527_run_newonly_replay_router_training.sh export {new_only|replay} {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val}
  bash tools/T0527_run_newonly_replay_router_training.sh export_mrs {new_only|replay} {llama|qwen} {boolq|rte|siqa|piqa} {train_eval|val}
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

mrs_cache_for() {
    case "$1" in
        llama) echo "$LLAMA_MRS_CACHE" ;;
        qwen) echo "$QWEN_MRS_CACHE" ;;
    esac
}

new_cache_for() {
    case "$1" in
        llama) echo "$LLAMA_NEW_CACHE" ;;
        qwen) echo "$QWEN_NEW_CACHE" ;;
    esac
}

mrs_router_dir_for() {
    case "$1" in
        llama) echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

router_dir_for() {
    local mode="$1"
    local family="$2"
    local task="$3"
    case "$family:$mode" in
        llama:new_only) echo "./router_T0527_llama_new_only_${task}_from_mrs_correct_conf_ce_t1_3expert" ;;
        llama:replay) echo "./router_T0527_llama_replay_mrs_${task}_correct_conf_ce_t1_3expert" ;;
        qwen:new_only) echo "./router_T0527_qwen3_fp16_new_only_${task}_from_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
        qwen:replay) echo "./router_T0527_qwen3_fp16_replay_mrs_${task}_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

require_cache() {
    local cache_root="$1"
    if [[ ! -f "$cache_root/train/manifest.json" || ! -f "$cache_root/validation/manifest.json" ]]; then
        echo "missing completed cache: $cache_root" >&2
        echo "build it first with tools/T0527_build_cached_mrs_4other_3expert.sh" >&2
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

require_new_output() {
    local output_dir="$1"
    if [[ -e "$output_dir" ]]; then
        echo "output already exists: $output_dir" >&2
        echo "use a new output name or remove the obsolete/incomplete output deliberately" >&2
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

start_mrs_baseline() {
    local family="$1"
    local cache output_dir log_file
    local -a common_args
    require_family "$family"
    cache="$(mrs_cache_for "$family")"
    require_cache "$cache"
    output_dir="$(mrs_router_dir_for "$family")"
    require_new_output "$output_dir"
    log_file="T0527_${family}_router_mrs_baseline_3expert.log"
    mapfile -t common_args < <(training_args)

    nohup "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS" \
        --expert_names "$EXPERTS" \
        "${common_args[@]}" \
        > "$log_file" 2>&1 &
    echo "started MRS baseline: pid=$! family=$family log=$log_file out_dir=$output_dir"
}

start_train() {
    local mode="$1"
    local family="$2"
    local task="$3"
    local mrs_cache new_cache feature_roots sample_tasks base_router output_dir log_file
    local -a common_args load_args
    require_mode "$mode"
    require_family "$family"
    require_task "$task"
    mrs_cache="$(mrs_cache_for "$family")"
    new_cache="$(new_cache_for "$family")"
    require_cache "$new_cache"

    if [[ "$mode" == "new_only" ]]; then
        base_router="$(mrs_router_dir_for "$family")"
        require_router "$base_router"
        feature_roots="$new_cache"
        sample_tasks="$task"
        load_args=(--load_from "$base_router")
    else
        require_cache "$mrs_cache"
        feature_roots="$mrs_cache,$new_cache"
        sample_tasks="$EXPERTS,$task"
        load_args=()
    fi

    output_dir="$(router_dir_for "$mode" "$family" "$task")"
    require_new_output "$output_dir"
    log_file="T0527_${family}_router_${mode}_${task}_3expert.log"
    mapfile -t common_args < <(training_args)

    nohup "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$feature_roots" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        "${load_args[@]}" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        "${common_args[@]}" \
        > "$log_file" 2>&1 &
    echo "started train: pid=$! mode=$mode family=$family task=$task log=$log_file out_dir=$output_dir sample_tasks=$sample_tasks"
}

start_mrs_eval() {
    local mode="$1"
    local family="$2"
    local task="$3"
    local cache trained_router output_dir log_file
    require_mode "$mode"
    require_family "$family"
    require_task "$task"
    cache="$(mrs_cache_for "$family")"
    require_cache "$cache"
    trained_router="$(router_dir_for "$mode" "$family" "$task")"
    require_router "$trained_router"
    output_dir="${trained_router}_eval_mrs"
    require_new_output "$output_dir"
    log_file="T0527_${family}_router_${mode}_${task}_eval_mrs.log"

    nohup "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --load_from "$trained_router" \
        --sample_task_names "$EXPERTS" \
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
    echo "started MRS eval: pid=$! mode=$mode family=$family task=$task log=$log_file out_dir=$output_dir"
}

best_records_path() {
    local output_dir="$1"
    local split="$2"
    local epoch
    [[ -f "$output_dir/best_metrics.json" ]] || { echo "missing $output_dir/best_metrics.json" >&2; exit 1; }
    epoch="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_epoch"])' "$output_dir/best_metrics.json")"
    echo "$output_dir/route_records_${split}_epoch${epoch}.json"
}

cache_split_for() {
    case "$1" in
        train_eval) echo train ;;
        val) echo validation ;;
        *) echo "unknown split: $1" >&2; usage; exit 2 ;;
    esac
}

inspect_mrs_baseline() {
    local family="$1"
    local split="$2"
    shift 2
    require_family "$family"
    exec "$PY" tools/T0527_inspect_new_only_router_records.py \
        --run_dir "$(mrs_router_dir_for "$family")" \
        --split "$split" \
        "$@"
}

inspect_trained() {
    local mode="$1"
    local family="$2"
    local task="$3"
    local split="$4"
    shift 4
    require_mode "$mode"
    require_family "$family"
    require_task "$task"
    exec "$PY" tools/T0527_inspect_new_only_router_records.py \
        --run_dir "$(router_dir_for "$mode" "$family" "$task")" \
        --split "$split" \
        "$@"
}

inspect_mrs_eval() {
    local mode="$1"
    local family="$2"
    local task="$3"
    local split="$4"
    shift 4
    require_mode "$mode"
    require_family "$family"
    require_task "$task"
    exec "$PY" tools/T0527_inspect_new_only_router_records.py \
        --run_dir "$(router_dir_for "$mode" "$family" "$task")_eval_mrs" \
        --split "$split" \
        --eval_only \
        "$@"
}

export_mrs_baseline() {
    local family="$1"
    local split="$2"
    local output_dir records_path cache_split output_json
    require_family "$family"
    output_dir="$(mrs_router_dir_for "$family")"
    records_path="$(best_records_path "$output_dir" "$split")"
    cache_split="$(cache_split_for "$split")"
    output_json="T0527_${family}_mrs_baseline_${split}_sample_details.json"
    "$PY" tools/export_router_sample_details.py \
        --cache_roots "$(mrs_cache_for "$family")" \
        --records_path "$records_path" \
        --split "$cache_split" \
        --sample_tasks "$EXPERTS" \
        --out "$output_json"
}

export_trained() {
    local mode="$1"
    local family="$2"
    local task="$3"
    local split="$4"
    local cache_roots sample_tasks output_dir records_path cache_split output_json
    require_mode "$mode"
    require_family "$family"
    require_task "$task"
    if [[ "$mode" == "new_only" ]]; then
        cache_roots="$(new_cache_for "$family")"
        sample_tasks="$task"
    else
        cache_roots="$(mrs_cache_for "$family"),$(new_cache_for "$family")"
        sample_tasks="$EXPERTS,$task"
    fi
    output_dir="$(router_dir_for "$mode" "$family" "$task")"
    records_path="$(best_records_path "$output_dir" "$split")"
    cache_split="$(cache_split_for "$split")"
    output_json="T0527_${family}_${mode}_${task}_${split}_sample_details.json"
    "$PY" tools/export_router_sample_details.py \
        --cache_roots "$cache_roots" \
        --records_path "$records_path" \
        --split "$cache_split" \
        --sample_tasks "$sample_tasks" \
        --out "$output_json"
}

export_mrs_eval() {
    local mode="$1"
    local family="$2"
    local task="$3"
    local split="$4"
    local output_dir records_path cache_split output_json
    require_mode "$mode"
    require_family "$family"
    require_task "$task"
    output_dir="$(router_dir_for "$mode" "$family" "$task")_eval_mrs"
    cache_split="$(cache_split_for "$split")"
    if [[ "$split" == "train_eval" ]]; then
        records_path="$output_dir/route_records_train_eval_only.json"
    else
        records_path="$output_dir/route_records_val_eval_only.json"
    fi
    output_json="T0527_${family}_${mode}_${task}_eval_mrs_${split}_sample_details.json"
    "$PY" tools/export_router_sample_details.py \
        --cache_roots "$(mrs_cache_for "$family")" \
        --records_path "$records_path" \
        --split "$cache_split" \
        --sample_tasks "$EXPERTS" \
        --out "$output_json"
}

case "${1:-}" in
    train_mrs)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        start_mrs_baseline "$2"
        ;;
    train)
        [[ $# -eq 4 ]] || { usage; exit 2; }
        start_train "$2" "$3" "$4"
        ;;
    eval_mrs)
        [[ $# -eq 4 ]] || { usage; exit 2; }
        start_mrs_eval "$2" "$3" "$4"
        ;;
    inspect_mrs_baseline)
        [[ $# -ge 3 ]] || { usage; exit 2; }
        inspect_mrs_baseline "$2" "$3" "${@:4}"
        ;;
    inspect)
        [[ $# -ge 5 ]] || { usage; exit 2; }
        inspect_trained "$2" "$3" "$4" "$5" "${@:6}"
        ;;
    inspect_mrs)
        [[ $# -ge 5 ]] || { usage; exit 2; }
        inspect_mrs_eval "$2" "$3" "$4" "$5" "${@:6}"
        ;;
    export_mrs_baseline)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        export_mrs_baseline "$2" "$3"
        ;;
    export)
        [[ $# -eq 5 ]] || { usage; exit 2; }
        export_trained "$2" "$3" "$4" "$5"
        ;;
    export_mrs)
        [[ $# -eq 5 ]] || { usage; exit 2; }
        export_mrs_eval "$2" "$3" "$4" "$5"
        ;;
    *)
        usage
        exit 2
        ;;
esac
