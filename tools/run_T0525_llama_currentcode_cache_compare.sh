#!/usr/bin/env bash
set -euo pipefail

JOB="${1:?use eval_mrs, eval_boolq, eval_rte, eval_siqa, eval_piqa, inspect_<run>, cache_mrs, cache_4other, compare_mrs, or compare_4other}"
PY="/home/u9472191/.conda/envs/opencompass/bin/python"
DATA_ROOT="./0525_router_train_datasets"
MODEL="meta-llama/Llama-2-7b-chat-hf"
MRS_OLD="./0525_llama_cache_mrs_3expert_official_eval_aligned"
OTHER_OLD="./0525_llama_cache_4other_3expert_official_eval_aligned"
MRS_NEW="./0525_llama_cache_mrs_3expert_official_eval_aligned_currentcode_compare"
OTHER_NEW="./0525_llama_cache_4other_3expert_official_eval_aligned_currentcode_compare"
COMMON=(--data_root "$DATA_ROOT" --base_model_path "$MODEL" --router_bert_init ./task_classifier_ckpt --expert_names medmcqa,race,sst2 --batch_size 8 --first_layer_idx 0 --middle_layer_idx 15 --router_dim 512 --dtype float16 --score_mode official_eval_aligned_generation --lora_medmcqa ./saves/llama2-7b-chat-hf/lora/sft_medmcqa --lora_race ./saves/llama2-7b-chat-hf/lora/sft_race --lora_sst2 ./saves/llama2-7b-chat-hf/lora/sft_sst2)
EVAL_COMMON=(--bert_init ./task_classifier_ckpt --expert_names medmcqa,race,sst2 --batch_size 16 --max_bert_len 512 --router_dim 512 --sample_feature_mode none --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --self_preserve_weight 1.0 --eval_only --save_route_records)

export_eval_details() {
    local cache_roots="$1"
    local sample_tasks="$2"
    local output_dir="$3"
    local split="$4"
    local output_json="$5"
    local records_file
    local cache_split
    if [[ "$split" == "train" ]]; then
        records_file="$output_dir/route_records_train_eval_only.json"
        cache_split="train"
    else
        records_file="$output_dir/route_records_val_eval_only.json"
        cache_split="validation"
    fi
    "$PY" tools/export_router_sample_details.py \
        --cache_roots "$cache_roots" \
        --records_path "$records_file" \
        --split "$cache_split" \
        --sample_tasks "$sample_tasks" \
        --out "$output_json"
}

case "$JOB" in
    eval_mrs)
        exec "$PY" -u train_internal_two_router_compact_cached_joint.py --feature_roots "$MRS_OLD" --out_dir ./router_0525_llama_correct_conf_ce_t1_mrs_3expert_records_currentviewer --sample_task_names medmcqa,race,sst2 --load_from ./router_0525_llama_correct_conf_ce_t1_mrs_3expert "${EVAL_COMMON[@]}"
        ;;
    eval_boolq)
        exec "$PY" -u train_internal_two_router_compact_cached_joint.py --feature_roots "$MRS_OLD,$OTHER_OLD" --out_dir ./router_0525_llama_4sum_boolq_correct_conf_ce_t1_mrs_3expert_records_currentviewer --sample_task_names medmcqa,race,sst2,boolq --load_from ./router_0525_llama_4sum_boolq_correct_conf_ce_t1_mrs_3expert "${EVAL_COMMON[@]}"
        ;;
    eval_rte)
        exec "$PY" -u train_internal_two_router_compact_cached_joint.py --feature_roots "$MRS_OLD,$OTHER_OLD" --out_dir ./router_0525_llama_4sum_rte_correct_conf_ce_t1_mrs_3expert_records_currentviewer --sample_task_names medmcqa,race,sst2,rte --load_from ./router_0525_llama_4sum_rte_correct_conf_ce_t1_mrs_3expert "${EVAL_COMMON[@]}"
        ;;
    eval_siqa)
        exec "$PY" -u train_internal_two_router_compact_cached_joint.py --feature_roots "$MRS_OLD,$OTHER_OLD" --out_dir ./router_0525_llama_4sum_siqa_correct_conf_ce_t1_mrs_3expert_records_currentviewer --sample_task_names medmcqa,race,sst2,siqa --load_from ./router_0525_llama_4sum_siqa_correct_conf_ce_t1_mrs_3expert "${EVAL_COMMON[@]}"
        ;;
    eval_piqa)
        exec "$PY" -u train_internal_two_router_compact_cached_joint.py --feature_roots "$MRS_OLD,$OTHER_OLD" --out_dir ./router_0525_llama_4sum_piqa_correct_conf_ce_t1_mrs_3expert_records_currentviewer --sample_task_names medmcqa,race,sst2,piqa --load_from ./router_0525_llama_4sum_piqa_correct_conf_ce_t1_mrs_3expert "${EVAL_COMMON[@]}"
        ;;
    inspect_mrs|inspect_boolq|inspect_rte|inspect_siqa|inspect_piqa)
        RUN="${JOB#inspect_}"
        shift
        exec "$PY" tools/inspect_T0525_llama_router_records.py "$RUN" "$@"
        ;;
    export_mrs_train)
        export_eval_details "$MRS_OLD" medmcqa,race,sst2 ./router_0525_llama_correct_conf_ce_t1_mrs_3expert_records_currentviewer train T0525_llama_mrs_train_sample_details.json
        ;;
    export_mrs_val)
        export_eval_details "$MRS_OLD" medmcqa,race,sst2 ./router_0525_llama_correct_conf_ce_t1_mrs_3expert_records_currentviewer val T0525_llama_mrs_val_sample_details.json
        ;;
    export_boolq_train)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,boolq ./router_0525_llama_4sum_boolq_correct_conf_ce_t1_mrs_3expert_records_currentviewer train T0525_llama_4sum_boolq_train_sample_details.json
        ;;
    export_boolq_val)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,boolq ./router_0525_llama_4sum_boolq_correct_conf_ce_t1_mrs_3expert_records_currentviewer val T0525_llama_4sum_boolq_val_sample_details.json
        ;;
    export_rte_train)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,rte ./router_0525_llama_4sum_rte_correct_conf_ce_t1_mrs_3expert_records_currentviewer train T0525_llama_4sum_rte_train_sample_details.json
        ;;
    export_rte_val)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,rte ./router_0525_llama_4sum_rte_correct_conf_ce_t1_mrs_3expert_records_currentviewer val T0525_llama_4sum_rte_val_sample_details.json
        ;;
    export_siqa_train)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,siqa ./router_0525_llama_4sum_siqa_correct_conf_ce_t1_mrs_3expert_records_currentviewer train T0525_llama_4sum_siqa_train_sample_details.json
        ;;
    export_siqa_val)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,siqa ./router_0525_llama_4sum_siqa_correct_conf_ce_t1_mrs_3expert_records_currentviewer val T0525_llama_4sum_siqa_val_sample_details.json
        ;;
    export_piqa_train)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,piqa ./router_0525_llama_4sum_piqa_correct_conf_ce_t1_mrs_3expert_records_currentviewer train T0525_llama_4sum_piqa_train_sample_details.json
        ;;
    export_piqa_val)
        export_eval_details "$MRS_OLD,$OTHER_OLD" medmcqa,race,sst2,piqa ./router_0525_llama_4sum_piqa_correct_conf_ce_t1_mrs_3expert_records_currentviewer val T0525_llama_4sum_piqa_val_sample_details.json
        ;;
    cache_mrs)
        exec "$PY" -u build_cached_router_pair_dataset.py "${COMMON[@]}" --feature_root "$MRS_NEW" --task_names medmcqa,race,sst2 --max_train_samples 600 --max_val_samples 150
        ;;
    cache_4other)
        exec "$PY" -u build_cached_router_pair_dataset.py "${COMMON[@]}" --feature_root "$OTHER_NEW" --task_names boolq,rte,siqa,piqa --max_train_samples 800 --max_val_samples 200
        ;;
    compare_mrs)
        exec "$PY" tools/compare_T0525_llama_cached_matrices.py --old_root "$MRS_OLD" --new_root "$MRS_NEW"
        ;;
    compare_4other)
        exec "$PY" tools/compare_T0525_llama_cached_matrices.py --old_root "$OTHER_OLD" --new_root "$OTHER_NEW"
        ;;
    *)
        printf 'Unknown job: %s\n' "$JOB" >&2
        exit 2
        ;;
esac
