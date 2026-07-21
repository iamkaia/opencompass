#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

MODE="${1:-train_eval}"
MODELS_CSV="${MODELS:-qwen35,llama2_7b_chat}"
OUT_PREFIX="${OUT_PREFIX:-T0718_half16_pairmain_two_layer}"

export MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-16}"
export SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
export INCLUDE_MRS_PLUS="${INCLUDE_MRS_PLUS:-1}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
export EVAL_BASELINES="${EVAL_BASELINES:-0}"
export TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"

# Main change from the previous weighted-sum branch:
# keep the pair-level objective in backprop, then add marginal MSE for runtime
# weighted_sum alignment instead of replacing the whole loss with marginal MSE.
export WEIGHTED_SUM_AUX_ONLY="${WEIGHTED_SUM_AUX_ONLY:-0}"
export MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
export BEST_METRIC="${BEST_METRIC:-router_argmax_score}"
export TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.25}"
export PAIR_LOSS_NORMALIZATION="${PAIR_LOSS_NORMALIZATION:-none}"
export TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
export TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0718_rerun_half16_pairmain_two_layer_qwen35_llama2.sh train_eval \
    > T0718_half16_pairmain_two_layer.nohup.log 2>&1 &

Modes:
  train_eval  train routers from existing half16 cache, then eval weighted_sum
  train       train routers only
  eval        eval trained routers only

Scope:
  MODELS=qwen35,llama2_7b_chat
  MIDDLE_LAYER_IDX=16
  cache is reused from T0708_half_split_32layer_*_two_layer_20260708_035620

Objective override:
  WEIGHTED_SUM_AUX_ONLY=0
  MSE_WEIGHT=1.0
  BEST_METRIC=router_argmax_score
EOF
}

check_cache() {
    local cache_root="$1"
    [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]] || {
        echo "missing reusable cache: $cache_root" >&2
        exit 1
    }
}

run_qwen35() {
    local out_root="./${OUT_PREFIX}_qwen35_${RUN_STAMP}"
    local cache_root="./T0708_half_split_32layer_qwen35_two_layer_20260708_035620/cache/qwen35_9task_3expert_two_layer_half16_800_200"
    check_cache "$cache_root"

    log "[START] qwen35 mode=$1 out_root=$out_root cache_root=$cache_root"
    RUN_LABEL="T0718_qwen35_half16_pairmain" \
    OUT_ROOT="$out_root" \
    CACHE_ROOT="$cache_root" \
    BASE_MODEL=Qwen/Qwen3.5-4B \
    LORA_ROOT=./saves/Qwen/Qwen3.5-4B/lora \
    MODEL_CONFIG=T0704_qwen35_mrs_ablation_two_layer.py \
    BASE_MODEL_CONFIG=T0704_qwen35_base_model.py \
    RESUME=1 \
        bash tools/T0704_train_qwen35_two_layer_wsum.sh "$1"
    log "[DONE] qwen35 mode=$1 out_root=$out_root"
}

run_llama2_7b_chat() {
    local out_root="./${OUT_PREFIX}_llama2_7b_chat_${RUN_STAMP}"
    local cache_root="./T0708_half_split_32layer_llama2_7b_chat_two_layer_20260708_035620/cache/llama2_7b_chat_9task_3expert_two_layer_half16_800_200"
    check_cache "$cache_root"

    log "[START] llama2_7b_chat mode=$1 out_root=$out_root cache_root=$cache_root"
    RUN_LABEL="T0718_llama2_half16_pairmain" \
    OUT_ROOT="$out_root" \
    CACHE_ROOT="$cache_root" \
    BASE_MODEL=meta-llama/Llama-2-7b-chat-hf \
    LORA_ROOT=./saves/llama2-7b-chat-hf/lora \
    MODEL_CONFIG=T0708_llama2_7b_chat_mrs_ablation_two_layer.py \
    BASE_MODEL_CONFIG=T0708_llama2_7b_chat_base_model.py \
    CACHE_PROMPT_TEMPLATE=chat_template \
    RESUME=1 \
        bash tools/T0707_train_llama3_8b_two_layer_wsum.sh "$1"
    log "[DONE] llama2_7b_chat mode=$1 out_root=$out_root"
}

run_model_mode() {
    local model="$1" mode="$2"
    case "$model" in
        qwen35) run_qwen35 "$mode" ;;
        llama2_7b_chat) run_llama2_7b_chat "$mode" ;;
        *) echo "unsupported model: $model" >&2; exit 2 ;;
    esac
}

main() {
    case "$MODE" in
        -h|--help|help) usage; exit 0 ;;
        train_eval|train|eval) ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$MODELS_CSV"
    local -a models=("${SPLIT_RESULT[@]}")

    log "[INFO] run_stamp=$RUN_STAMP models=${models[*]} mode=$MODE middle_layer_idx=$MIDDLE_LAYER_IDX"
    log "[INFO] objective=correct_conf_ce + ${MSE_WEIGHT}*weighted_sum_marginal_mse weighted_sum_aux_only=$WEIGHTED_SUM_AUX_ONLY best_metric=$BEST_METRIC"
    log "[INFO] eval_baselines=$EVAL_BASELINES include_mrs_plus=$INCLUDE_MRS_PLUS include_new_only=$INCLUDE_NEW_ONLY sharpness=$SHARPNESS_LIST"

    local model
    for model in "${models[@]}"; do
        case "$MODE" in
            train_eval)
                run_model_mode "$model" train
                run_model_mode "$model" eval
                ;;
            train|eval)
                run_model_mode "$model" "$MODE"
                ;;
        esac
    done
}

main "$@"
