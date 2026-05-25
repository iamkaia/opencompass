#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
DATASETS=(SuperGLUE_RTE_gen piqa_gen race_gen_sft_prompt sst2_gen medmcqa_gen_sft_prompt siqa_gen SuperGLUE_BoolQ_gen)

run_eval() {
    local model_config="$1"
    local log_file="$2"
    nohup "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${DATASETS[@]}" \
        --max-num-workers 1 \
        > "$log_file" 2>&1 &
    echo "started: $log_file"
    echo "wait for this job to finish before starting another Qwen evaluation on the same GPU"
}

case "${1:-}" in
    mrs)
        run_eval moe_lora_internal_compact_cachedjoint_0525_mrs_3expert_sst2words T0525_opencompass_qwen3_mrs_3expert_sst2words.log
        ;;
    boolq)
        run_eval moe_lora_internal_compact_cachedjoint_0525_4sum_boolq_mrs_3expert_sst2words T0525_opencompass_qwen3_4sum_boolq_mrs_3expert_sst2words.log
        ;;
    rte)
        run_eval moe_lora_internal_compact_cachedjoint_0525_4sum_rte_mrs_3expert_sst2words T0525_opencompass_qwen3_4sum_rte_mrs_3expert_sst2words.log
        ;;
    siqa)
        run_eval moe_lora_internal_compact_cachedjoint_0525_4sum_siqa_mrs_3expert_sst2words T0525_opencompass_qwen3_4sum_siqa_mrs_3expert_sst2words.log
        ;;
    piqa)
        run_eval moe_lora_internal_compact_cachedjoint_0525_4sum_piqa_mrs_3expert_sst2words T0525_opencompass_qwen3_4sum_piqa_mrs_3expert_sst2words.log
        ;;
    *)
        echo "usage: bash tools/run_T0525_qwen3_opencompass_sst2words.sh {mrs|boolq|rte|siqa|piqa}" >&2
        exit 2
        ;;
esac
