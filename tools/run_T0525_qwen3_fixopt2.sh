#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
DATA_ROOT="./0525_router_train_datasets"
BERT="./task_classifier_ckpt"
LORA_ROOT="./saves/Qwen/Qwen3-4B-Instruct-2507/lora"

CACHE_MRS="./0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words"
CACHE_4OTHER="./0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words"

ROUTER_MRS="./router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert_sst2words"
ROUTER_BOOLQ="./router_0525_qwen3_fp16_4sum_boolq_correct_conf_ce_t1_mrs_3expert_sst2words"
ROUTER_RTE="./router_0525_qwen3_fp16_4sum_rte_correct_conf_ce_t1_mrs_3expert_sst2words"
ROUTER_SIQA="./router_0525_qwen3_fp16_4sum_siqa_correct_conf_ce_t1_mrs_3expert_sst2words"
ROUTER_PIQA="./router_0525_qwen3_fp16_4sum_piqa_correct_conf_ce_t1_mrs_3expert_sst2words"

start_cache() {
    local output_root="$1"
    local sample_tasks="$2"
    local max_train_samples="$3"
    local max_val_samples="$4"
    local log_file="$5"
    nohup "$PY" -u build_cached_router_pair_dataset.py \
        --data_root "$DATA_ROOT" \
        --feature_root "$output_root" \
        --task_names "$sample_tasks" \
        --expert_names medmcqa,race,sst2 \
        --base_model_path "$MODEL" \
        --router_bert_init "$BERT" \
        --batch_size 8 \
        --max_train_samples "$max_train_samples" \
        --max_val_samples "$max_val_samples" \
        --first_layer_idx 0 \
        --middle_layer_idx 18 \
        --router_dim 512 \
        --dtype float16 \
        --score_mode official_eval_aligned_generation \
        --lora_medmcqa "$LORA_ROOT/sft_medmcqa" \
        --lora_race "$LORA_ROOT/sft_race" \
        --lora_sst2 "$LORA_ROOT/sft_sst2" \
        > "$log_file" 2>&1 &
    echo "started: $log_file"
}

require_cache() {
    local cache_root="$1"
    if [[ ! -f "$cache_root/train/manifest.json" || ! -f "$cache_root/validation/manifest.json" ]]; then
        echo "missing completed cache: $cache_root" >&2
        echo "finish cache jobs before starting router training" >&2
        exit 1
    fi
}

start_router() {
    local feature_roots="$1"
    local sample_tasks="$2"
    local output_dir="$3"
    local log_file="$4"
    nohup "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$feature_roots" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names medmcqa,race,sst2 \
        --router_dim 512 \
        --joint_loss correct_conf_ce \
        --correct_soft_ce_temperature 1.0 \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1 &
    echo "started: $log_file"
}

export_details() {
    local cache_roots="$1"
    local sample_tasks="$2"
    local route_dir="$3"
    local split="$4"
    local output_json="$5"
    local epoch
    epoch="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1] + '/best_metrics.json'))['best_epoch'])" "$route_dir")"
    "$PY" tools/export_router_sample_details.py \
        --cache_roots "$cache_roots" \
        --records_path "$route_dir/route_records_${split}_epoch${epoch}.json" \
        --split "$([[ "$split" == "train_eval" ]] && echo train || echo validation)" \
        --sample_tasks "$sample_tasks" \
        --out "$output_json"
}

case "${1:-}" in
    cache_mrs)
        start_cache "$CACHE_MRS" medmcqa,race,sst2 600 150 T0525_qwen3_fp16_cache_mrs_3expert_sst2words.log
        ;;
    cache_4other)
        start_cache "$CACHE_4OTHER" boolq,rte,siqa,piqa 800 200 T0525_qwen3_fp16_cache_4other_3expert_sst2words.log
        ;;
    train_mrs)
        require_cache "$CACHE_MRS"
        start_router "$CACHE_MRS" medmcqa,race,sst2 "$ROUTER_MRS" T0525_qwen3_fp16_router_mrs_3expert_sst2words.log
        ;;
    train_boolq)
        require_cache "$CACHE_MRS"
        require_cache "$CACHE_4OTHER"
        start_router "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,boolq "$ROUTER_BOOLQ" T0525_qwen3_fp16_router_4sum_boolq_mrs_3expert_sst2words.log
        ;;
    train_rte)
        require_cache "$CACHE_MRS"
        require_cache "$CACHE_4OTHER"
        start_router "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,rte "$ROUTER_RTE" T0525_qwen3_fp16_router_4sum_rte_mrs_3expert_sst2words.log
        ;;
    train_siqa)
        require_cache "$CACHE_MRS"
        require_cache "$CACHE_4OTHER"
        start_router "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,siqa "$ROUTER_SIQA" T0525_qwen3_fp16_router_4sum_siqa_mrs_3expert_sst2words.log
        ;;
    train_piqa)
        require_cache "$CACHE_MRS"
        require_cache "$CACHE_4OTHER"
        start_router "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,piqa "$ROUTER_PIQA" T0525_qwen3_fp16_router_4sum_piqa_mrs_3expert_sst2words.log
        ;;
    export_mrs_train)
        export_details "$CACHE_MRS" medmcqa,race,sst2 "$ROUTER_MRS" train_eval T0525_qwen3_mrs_train_sample_details_sst2words.json
        ;;
    export_mrs_val)
        export_details "$CACHE_MRS" medmcqa,race,sst2 "$ROUTER_MRS" val T0525_qwen3_mrs_val_sample_details_sst2words.json
        ;;
    export_boolq_train)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,boolq "$ROUTER_BOOLQ" train_eval T0525_qwen3_4sum_boolq_train_sample_details_sst2words.json
        ;;
    export_boolq_val)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,boolq "$ROUTER_BOOLQ" val T0525_qwen3_4sum_boolq_val_sample_details_sst2words.json
        ;;
    export_rte_train)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,rte "$ROUTER_RTE" train_eval T0525_qwen3_4sum_rte_train_sample_details_sst2words.json
        ;;
    export_rte_val)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,rte "$ROUTER_RTE" val T0525_qwen3_4sum_rte_val_sample_details_sst2words.json
        ;;
    export_siqa_train)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,siqa "$ROUTER_SIQA" train_eval T0525_qwen3_4sum_siqa_train_sample_details_sst2words.json
        ;;
    export_siqa_val)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,siqa "$ROUTER_SIQA" val T0525_qwen3_4sum_siqa_val_sample_details_sst2words.json
        ;;
    export_piqa_train)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,piqa "$ROUTER_PIQA" train_eval T0525_qwen3_4sum_piqa_train_sample_details_sst2words.json
        ;;
    export_piqa_val)
        export_details "$CACHE_MRS,$CACHE_4OTHER" medmcqa,race,sst2,piqa "$ROUTER_PIQA" val T0525_qwen3_4sum_piqa_val_sample_details_sst2words.json
        ;;
    inspect_mrs|inspect_boolq|inspect_rte|inspect_siqa|inspect_piqa)
        RUN="${1#inspect_}"
        shift
        exec "$PY" tools/inspect_T0525_qwen3_router_records.py "$RUN" "$@"
        ;;
    *)
        echo "usage: bash tools/run_T0525_qwen3_fixopt2.sh {cache_mrs|cache_4other|train_mrs|train_boolq|train_rte|train_siqa|train_piqa|export_mrs_train|export_mrs_val|export_boolq_train|export_boolq_val|export_rte_train|export_rte_val|export_siqa_train|export_siqa_val|export_piqa_train|export_piqa_val|inspect_mrs|inspect_boolq|inspect_rte|inspect_siqa|inspect_piqa}" >&2
        exit 2
        ;;
esac
