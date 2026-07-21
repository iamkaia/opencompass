#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

MODE="${1:-train_eval}"
MODELS_CSV="${MODELS:-qwen35,llama2_7b_chat}"
OUT_PREFIX="${OUT_PREFIX:-T0718_half16_single_vs_two_search}"
TWO_GRID_CSV="${TWO_GRID:-w25,w50,w75}"

export MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-16}"
export SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
export INCLUDE_MRS_PLUS="${INCLUDE_MRS_PLUS:-1}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
export EVAL_BASELINES="${EVAL_BASELINES:-0}"
export TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0718_search_half16_single_vs_two_qwen35_llama2.sh train_eval \
    > T0718_half16_single_vs_two_search.nohup.log 2>&1 &

Modes:
  train_eval  train from existing half16 caches, then eval weighted_sum
  train       train only
  eval        eval only
  single      train+eval single_layer baselines only
  two         train+eval two_layer grid only

Scope:
  MODELS=qwen35,llama2_7b_chat
  MIDDLE_LAYER_IDX=16
  Reuses existing T0708 half16 single/two caches.

Two-layer grid names:
  w25     pair CE + 25*wsmse: temp=0.25, none, best=router_argmax_score
  w50     pair CE + 50*wsmse: temp=0.25, none, best=router_argmax_score
  w75     pair CE + 75*wsmse: temp=0.25, none, best=router_argmax_score
  hardce  optional pure T0603-style pair CE: MSE=0, temp=1.0, sample_minmax, best=route_correct_acc

Override:
  TWO_GRID=w50
EOF
}

check_cache() {
    local cache_root="$1"
    [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]] || {
        echo "missing reusable cache: $cache_root" >&2
        exit 1
    }
}

single_cache_root() {
    case "$1" in
        qwen35)
            echo "./T0708_half_split_32layer_qwen35_single_layer_20260708_035620/cache/qwen35_9task_3expert_single_all_layers_half16_800_200"
            ;;
        llama2_7b_chat)
            echo "./T0708_half_split_32layer_llama2_7b_chat_single_layer_20260708_035620/cache/llama2_7b_chat_9task_3expert_single_all_layers_half16_800_200"
            ;;
        *) echo "unsupported model: $1" >&2; exit 2 ;;
    esac
}

two_cache_root() {
    case "$1" in
        qwen35)
            echo "./T0708_half_split_32layer_qwen35_two_layer_20260708_035620/cache/qwen35_9task_3expert_two_layer_half16_800_200"
            ;;
        llama2_7b_chat)
            echo "./T0708_half_split_32layer_llama2_7b_chat_two_layer_20260708_035620/cache/llama2_7b_chat_9task_3expert_two_layer_half16_800_200"
            ;;
        *) echo "unsupported model: $1" >&2; exit 2 ;;
    esac
}

run_single_model_mode() {
    local model="$1" mode="$2" cache_root out_root
    cache_root="$(single_cache_root "$model")"
    check_cache "$cache_root"
    out_root="./${OUT_PREFIX}_${model}_single_${RUN_STAMP}"

    log "[START] single model=$model mode=$mode out_root=$out_root cache_root=$cache_root"
    case "$model" in
        qwen35)
            RUN_LABEL="T0718_qwen35_half16_single" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL=Qwen/Qwen3.5-4B \
            LORA_ROOT=./saves/Qwen/Qwen3.5-4B/lora \
            MODEL_CONFIG=T0704_qwen35_mrs_ablation_single_all_layers.py \
            BASE_MODEL_CONFIG=T0704_qwen35_base_model.py \
            RESUME=1 \
                bash tools/T0704_train_qwen35_single_all_layers_wsum.sh "$mode"
            ;;
        llama2_7b_chat)
            RUN_LABEL="T0718_llama2_half16_single" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL=meta-llama/Llama-2-7b-chat-hf \
            LORA_ROOT=./saves/llama2-7b-chat-hf/lora \
            MODEL_CONFIG=T0702_mrs_ablation_single_all_layers.py \
            BASE_MODEL_CONFIG=T0708_llama2_7b_chat_base_model.py \
            CACHE_PROMPT_TEMPLATE=chat_template \
            RESUME=1 \
                bash tools/T0702_train_llama2_single_all_layers_wsum.sh "$mode"
            ;;
    esac
    log "[DONE] single model=$model mode=$mode out_root=$out_root"
}

apply_two_grid() {
    local grid="$1"
    export WEIGHTED_SUM_AUX_ONLY=0
    export TARGET_DISTRIBUTION_POLICY=cache_oracle
    export TARGET_EMPTY_FALLBACK=uniform
    export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
    case "$grid" in
        hardce)
            export MSE_WEIGHT=0
            export TARGET_TEMPERATURE=1.0
            export PAIR_LOSS_NORMALIZATION=sample_minmax
            export BEST_METRIC=route_correct_acc
            ;;
        w25)
            export MSE_WEIGHT=25
            export TARGET_TEMPERATURE=0.25
            export PAIR_LOSS_NORMALIZATION=none
            export BEST_METRIC=router_argmax_score
            ;;
        w50)
            export MSE_WEIGHT=50
            export TARGET_TEMPERATURE=0.25
            export PAIR_LOSS_NORMALIZATION=none
            export BEST_METRIC=router_argmax_score
            ;;
        w75)
            export MSE_WEIGHT=75
            export TARGET_TEMPERATURE=0.25
            export PAIR_LOSS_NORMALIZATION=none
            export BEST_METRIC=router_argmax_score
            ;;
        *)
            echo "unsupported TWO_GRID entry: $grid" >&2
            exit 2
            ;;
    esac
}

run_two_model_grid_mode() {
    local model="$1" grid="$2" mode="$3" cache_root out_root
    cache_root="$(two_cache_root "$model")"
    check_cache "$cache_root"
    apply_two_grid "$grid"
    out_root="./${OUT_PREFIX}_${model}_two_${grid}_${RUN_STAMP}"

    log "[START] two model=$model grid=$grid mode=$mode out_root=$out_root cache_root=$cache_root"
    log "[INFO] two grid=$grid mse=$MSE_WEIGHT aux_only=$WEIGHTED_SUM_AUX_ONLY temp=$TARGET_TEMPERATURE norm=$PAIR_LOSS_NORMALIZATION best=$BEST_METRIC"
    case "$model" in
        qwen35)
            RUN_LABEL="T0718_qwen35_half16_two_${grid}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL=Qwen/Qwen3.5-4B \
            LORA_ROOT=./saves/Qwen/Qwen3.5-4B/lora \
            MODEL_CONFIG=T0704_qwen35_mrs_ablation_two_layer.py \
            BASE_MODEL_CONFIG=T0704_qwen35_base_model.py \
            RESUME=1 \
                bash tools/T0704_train_qwen35_two_layer_wsum.sh "$mode"
            ;;
        llama2_7b_chat)
            RUN_LABEL="T0718_llama2_half16_two_${grid}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL=meta-llama/Llama-2-7b-chat-hf \
            LORA_ROOT=./saves/llama2-7b-chat-hf/lora \
            MODEL_CONFIG=T0708_llama2_7b_chat_mrs_ablation_two_layer.py \
            BASE_MODEL_CONFIG=T0708_llama2_7b_chat_base_model.py \
            CACHE_PROMPT_TEMPLATE=chat_template \
            RESUME=1 \
                bash tools/T0707_train_llama3_8b_two_layer_wsum.sh "$mode"
            ;;
    esac
    log "[DONE] two model=$model grid=$grid mode=$mode out_root=$out_root"
}

run_single_all() {
    local mode="$1" model
    split_csv "$MODELS_CSV"
    for model in "${SPLIT_RESULT[@]}"; do
        run_single_model_mode "$model" "$mode"
    done
}

run_two_all() {
    local mode="$1" model grid
    split_csv "$MODELS_CSV"
    local -a models=("${SPLIT_RESULT[@]}")
    split_csv "$TWO_GRID_CSV"
    local -a grids=("${SPLIT_RESULT[@]}")
    for model in "${models[@]}"; do
        for grid in "${grids[@]}"; do
            run_two_model_grid_mode "$model" "$grid" "$mode"
        done
    done
}

main() {
    case "$MODE" in
        -h|--help|help) usage; exit 0 ;;
        train_eval|train|eval|single|two) ;;
        *) usage; exit 2 ;;
    esac

    log "[INFO] run_stamp=$RUN_STAMP mode=$MODE models=$MODELS_CSV two_grid=$TWO_GRID_CSV middle_layer_idx=$MIDDLE_LAYER_IDX"
    log "[INFO] include_mrs_plus=$INCLUDE_MRS_PLUS include_new_only=$INCLUDE_NEW_ONLY eval_baselines=$EVAL_BASELINES sharpness=$SHARPNESS_LIST"

    case "$MODE" in
        train_eval)
            run_single_all train
            run_single_all eval
            run_two_all train
            run_two_all eval
            ;;
        train)
            run_single_all train
            run_two_all train
            ;;
        eval)
            run_single_all eval
            run_two_all eval
            ;;
        single)
            run_single_all train
            run_single_all eval
            ;;
        two)
            run_two_all train
            run_two_all eval
            ;;
    esac
}

main "$@"
