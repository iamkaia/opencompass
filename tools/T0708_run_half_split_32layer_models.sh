#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-16}"
export SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
export EVAL_BASELINES="${EVAL_BASELINES:-1}"

MODE="${1:-all}"
MODELS_CSV="${MODELS:-qwen35,llama3_8b,llama2_7b_chat}"
FAMILIES_CSV="${FAMILIES:-two_layer,single_layer}"
OUT_PREFIX="${OUT_PREFIX:-T0708_half_split_32layer}"
SPLIT_TAG="${SPLIT_TAG:-half16}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0708_run_half_split_32layer_models.sh all \
    > T0708_half_split_32layer.nohup.log 2>&1 &

Modes are passed through to the family launchers:
  all       cache + train + router eval + base_model/uniform baselines
  cache     cache only
  train     train routers from existing cache
  eval      eval trained routers only
  baseline  eval base_model and uniform only

Default scope:
  MODELS=qwen35,llama3_8b,llama2_7b_chat
  FAMILIES=two_layer,single_layer
  MIDDLE_LAYER_IDX=16
  SPLIT_TAG=half16
  base_model max_out_len=128
  router/uniform max_out_len follows each model config default unless MAX_OUT_LEN is set
EOF
}

run_two_layer() {
    local model="$1"
    case "$model" in
        qwen35)
            RUN_LABEL="T0708_qwen35_${SPLIT_TAG}_two" \
            OUT_ROOT="./${OUT_PREFIX}_qwen35_two_layer_${RUN_STAMP}" \
            CACHE_ROOT="./${OUT_PREFIX}_qwen35_two_layer_${RUN_STAMP}/cache/qwen35_9task_3expert_two_layer_${SPLIT_TAG}_800_200" \
            BASE_MODEL=Qwen/Qwen3.5-4B \
            LORA_ROOT=./saves/Qwen/Qwen3.5-4B/lora \
            MODEL_CONFIG=T0704_qwen35_mrs_ablation_two_layer.py \
            BASE_MODEL_CONFIG=T0704_qwen35_base_model.py \
            CACHE_PROMPT_TEMPLATE=chat_template \
            MAX_OUT_LEN= \
            bash tools/T0704_train_qwen35_two_layer_wsum.sh "$MODE"
            ;;
        llama3_8b)
            RUN_LABEL="T0708_llama3_8b_${SPLIT_TAG}_two" \
            OUT_ROOT="./${OUT_PREFIX}_llama3_8b_two_layer_${RUN_STAMP}" \
            CACHE_ROOT="./${OUT_PREFIX}_llama3_8b_two_layer_${RUN_STAMP}/cache/llama3_8b_9task_3expert_two_layer_${SPLIT_TAG}_800_200" \
            BASE_MODEL=meta-llama/Meta-Llama-3-8B \
            LORA_ROOT=./saves/Meta-Llama-3-8B/lora \
            MODEL_CONFIG=T0707_llama3_8b_mrs_ablation_two_layer.py \
            BASE_MODEL_CONFIG=T0707_llama3_8b_base_model.py \
            CACHE_PROMPT_TEMPLATE=raw \
            MAX_OUT_LEN= \
            bash tools/T0707_train_llama3_8b_two_layer_wsum.sh "$MODE"
            ;;
        llama2_7b_chat)
            RUN_LABEL="T0708_llama2_${SPLIT_TAG}_two" \
            OUT_ROOT="./${OUT_PREFIX}_llama2_7b_chat_two_layer_${RUN_STAMP}" \
            CACHE_ROOT="./${OUT_PREFIX}_llama2_7b_chat_two_layer_${RUN_STAMP}/cache/llama2_7b_chat_9task_3expert_two_layer_${SPLIT_TAG}_800_200" \
            BASE_MODEL=meta-llama/Llama-2-7b-chat-hf \
            LORA_ROOT=./saves/llama2-7b-chat-hf/lora \
            MODEL_CONFIG=T0708_llama2_7b_chat_mrs_ablation_two_layer.py \
            BASE_MODEL_CONFIG=T0708_llama2_7b_chat_base_model.py \
            CACHE_PROMPT_TEMPLATE=chat_template \
            MAX_OUT_LEN= \
            bash tools/T0707_train_llama3_8b_two_layer_wsum.sh "$MODE"
            ;;
        *) echo "unsupported model for two_layer: $model" >&2; exit 2 ;;
    esac
}

run_single_layer() {
    local model="$1"
    case "$model" in
        qwen35)
            RUN_LABEL="T0708_qwen35_${SPLIT_TAG}_single" \
            OUT_ROOT="./${OUT_PREFIX}_qwen35_single_layer_${RUN_STAMP}" \
            CACHE_ROOT="./${OUT_PREFIX}_qwen35_single_layer_${RUN_STAMP}/cache/qwen35_9task_3expert_single_all_layers_${SPLIT_TAG}_800_200" \
            BASE_MODEL=Qwen/Qwen3.5-4B \
            LORA_ROOT=./saves/Qwen/Qwen3.5-4B/lora \
            MODEL_CONFIG=T0704_qwen35_mrs_ablation_single_all_layers.py \
            BASE_MODEL_CONFIG=T0704_qwen35_base_model.py \
            MAX_OUT_LEN= \
            bash tools/T0704_train_qwen35_single_all_layers_wsum.sh "$MODE"
            ;;
        llama3_8b)
            RUN_LABEL="T0708_llama3_8b_${SPLIT_TAG}_single" \
            OUT_ROOT="./${OUT_PREFIX}_llama3_8b_single_layer_${RUN_STAMP}" \
            CACHE_ROOT="./${OUT_PREFIX}_llama3_8b_single_layer_${RUN_STAMP}/cache/llama3_8b_9task_3expert_single_all_layers_${SPLIT_TAG}_800_200" \
            BASE_MODEL=meta-llama/Meta-Llama-3-8B \
            LORA_ROOT=./saves/Meta-Llama-3-8B/lora \
            MODEL_CONFIG=T0708_llama3_8b_mrs_ablation_single_all_layers.py \
            BASE_MODEL_CONFIG=T0707_llama3_8b_base_model.py \
            CACHE_PROMPT_TEMPLATE=raw \
            MAX_OUT_LEN= \
            bash tools/T0702_train_llama2_single_all_layers_wsum.sh "$MODE"
            ;;
        llama2_7b_chat)
            RUN_LABEL="T0708_llama2_${SPLIT_TAG}_single" \
            OUT_ROOT="./${OUT_PREFIX}_llama2_7b_chat_single_layer_${RUN_STAMP}" \
            CACHE_ROOT="./${OUT_PREFIX}_llama2_7b_chat_single_layer_${RUN_STAMP}/cache/llama2_7b_chat_9task_3expert_single_all_layers_${SPLIT_TAG}_800_200" \
            BASE_MODEL=meta-llama/Llama-2-7b-chat-hf \
            LORA_ROOT=./saves/llama2-7b-chat-hf/lora \
            MODEL_CONFIG=T0702_mrs_ablation_single_all_layers.py \
            BASE_MODEL_CONFIG=T0708_llama2_7b_chat_base_model.py \
            MAX_OUT_LEN= \
            bash tools/T0702_train_llama2_single_all_layers_wsum.sh "$MODE"
            ;;
        *) echo "unsupported model for single_layer: $model" >&2; exit 2 ;;
    esac
}

main() {
    case "$MODE" in
        -h|--help|help) usage; exit 0 ;;
        all|cache|train|eval|baseline) ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$MODELS_CSV"
    local -a models=("${SPLIT_RESULT[@]}")
    split_csv "$FAMILIES_CSV"
    local -a families=("${SPLIT_RESULT[@]}")

    log "[INFO] run_stamp=$RUN_STAMP models=${models[*]} families=${families[*]} middle_layer_idx=$MIDDLE_LAYER_IDX split_tag=$SPLIT_TAG mode=$MODE"
    log "[INFO] base_model max_out_len=128 via base configs; router/uniform max_out_len uses config default unless MAX_OUT_LEN is explicitly exported"

    local model family
    for model in "${models[@]}"; do
        for family in "${families[@]}"; do
            log "[START] model=$model family=$family"
            case "$family" in
                two_layer) run_two_layer "$model" ;;
                single_layer) run_single_layer "$model" ;;
                *) echo "unsupported family: $family" >&2; exit 2 ;;
            esac
            log "[DONE] model=$model family=$family"
        done
    done
}

main "$@"
