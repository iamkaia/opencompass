#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"

declare -A MODEL_CONFIGS=(
    [mrs]="moe_lora_internal_compact_cachedjoint_0525_mrs_3expert_weighted_sum.py"
    [boolq]="moe_lora_internal_compact_cachedjoint_0525_4sum_boolq_mrs_3expert_weighted_sum.py"
    [rte]="moe_lora_internal_compact_cachedjoint_0525_4sum_rte_mrs_3expert_weighted_sum.py"
    [siqa]="moe_lora_internal_compact_cachedjoint_0525_4sum_siqa_mrs_3expert_weighted_sum.py"
    [piqa]="moe_lora_internal_compact_cachedjoint_0525_4sum_piqa_mrs_3expert_weighted_sum.py"
)

usage() {
    echo "usage: CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_llama_opencompass_weighted_sum.sh {mrs|boolq|rte|siqa|piqa}"
}

run_eval() {
    local router_key="$1"
    local model_config="${MODEL_CONFIGS[$router_key]:-}"
    local log_file="T0525_run_opencompass_llama_${router_key}_mrs_3expert_weighted_sum_$(date +"%Y%m%d_%H%M%S").log"
    local -a datasets

    if [[ -z "$model_config" ]]; then
        usage >&2
        exit 2
    fi

    case "$router_key" in
        mrs)
            datasets=(medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt)
            ;;
        boolq)
            datasets=(SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt)
            ;;
        rte)
            datasets=(SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen)
            ;;
        siqa)
            datasets=(siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen)
            ;;
        piqa)
            datasets=(piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen)
            ;;
    esac

    nohup "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --debug \
        > "$log_file" 2>&1 &
    echo "started: $log_file datasets=${datasets[*]}"
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

run_eval "$1"
