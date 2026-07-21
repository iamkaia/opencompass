#!/usr/bin/env bash
set -euo pipefail

export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export RUN_LABEL="${RUN_LABEL:-T0704_qwen35}"
export OUT_ROOT="${OUT_ROOT:-./T0704_qwen35_single_all_layers_wsum_t0p25_${RUN_STAMP}}"
export CACHE_ROOT="${CACHE_ROOT:-$OUT_ROOT/cache/qwen35_9task_3expert_single_all_layers_800_200}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
export LORA_ROOT="${LORA_ROOT:-./saves/Qwen/Qwen3.5-4B/lora}"
export MODEL_CONFIG="${MODEL_CONFIG:-T0704_qwen35_mrs_ablation_single_all_layers.py}"
export MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-18}"
export SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
export TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.25}"
export EXPERT_CE_WEIGHT="${EXPERT_CE_WEIGHT:-0.0}"
export MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"

exec bash tools/T0702_train_llama2_single_all_layers_wsum.sh "$@"
