#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODE="${1:-train_eval}"

export PAIR_CONSTRAINT=diagonal
export WEIGHTED_SUM_AUX_ONLY=1
export MSE_WEIGHT=1.0
export BEST_METRIC=weighted_sum_marginal_mse
export TARGET_TEMPERATURE=0.25
export PAIR_LOSS_NORMALIZATION=none
export TARGET_EMPTY_FALLBACK=uniform
export TARGET_DISTRIBUTION_POLICY=cache_oracle
export MIDDLE_LAYER_IDX=16
export SHARPNESS_LIST=3.0
export INCLUDE_MRS_PLUS=1
export INCLUDE_NEW_ONLY=0
export EVAL_BASELINES=0
export TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
export RESUME=1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

run_qwen35() {
    export RUN_LABEL="T0720_diagconstr_qwen35_two_layer_mse"
    export OUT_ROOT="./T0720_diagconstr_qwen35_two_layer_mse_${RUN_STAMP}"
    export CACHE_ROOT="./T0708_half_split_32layer_qwen35_two_layer_20260708_035620/cache/qwen35_9task_3expert_two_layer_half16_800_200"
    export BASE_MODEL="Qwen/Qwen3.5-4B"
    export LORA_ROOT="./saves/Qwen/Qwen3.5-4B/lora"
    export MODEL_CONFIG="T0704_qwen35_mrs_ablation_two_layer.py"
    export BASE_MODEL_CONFIG="T0704_qwen35_base_model.py"
    log "[START] qwen35 train"
    bash tools/T0704_train_qwen35_two_layer_wsum.sh train
    log "[START] qwen35 eval"
    bash tools/T0704_train_qwen35_two_layer_wsum.sh eval
    log "[DONE] qwen35 out=$OUT_ROOT"
}

run_llama2() {
    export RUN_LABEL="T0720_diagconstr_llama2_two_layer_mse"
    export OUT_ROOT="./T0720_diagconstr_llama2_two_layer_mse_${RUN_STAMP}"
    export CACHE_ROOT="./T0708_half_split_32layer_llama2_7b_chat_two_layer_20260708_035620/cache/llama2_7b_chat_9task_3expert_two_layer_half16_800_200"
    export BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
    export LORA_ROOT="./saves/llama2-7b-chat-hf/lora"
    export MODEL_CONFIG="T0708_llama2_7b_chat_mrs_ablation_two_layer.py"
    export BASE_MODEL_CONFIG="T0708_llama2_7b_chat_base_model.py"
    export CACHE_PROMPT_TEMPLATE=chat_template
    log "[START] llama2 train"
    bash tools/T0707_train_llama3_8b_two_layer_wsum.sh train
    log "[START] llama2 eval"
    bash tools/T0707_train_llama3_8b_two_layer_wsum.sh eval
    log "[DONE] llama2 out=$OUT_ROOT"
}

case "$MODE" in
    train_eval)
        run_qwen35
        run_llama2
        ;;
    qwen35)
        run_qwen35
        ;;
    llama2)
        run_llama2
        ;;
    *)
        echo "usage: bash tools/T0720_run_diag_constrained_two_layer_mse_qwen35_llama2.sh [train_eval|qwen35|llama2]" >&2
        exit 2
        ;;
esac
