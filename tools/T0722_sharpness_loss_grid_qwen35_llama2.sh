#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

MODE="${1:-train_eval}"
MODELS_CSV="${MODELS:-qwen3.5_4b,llama2_7b_chat}"
SPLITS_CSV="${SPLITS:-half16,late21}"
OBJECTIVES_CSV="${OBJECTIVES:-pure_mse,mix_w25,mix_w50,mix_w75}"
OUT_PREFIX="${OUT_PREFIX:-T0722_sharpness_loss_grid}"

export SHARPNESS_LIST="${SHARPNESS_LIST:-1.0,5.0,7.0}"
export INCLUDE_MRS_PLUS="${INCLUDE_MRS_PLUS:-1}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
export EVAL_BASELINES="${EVAL_BASELINES:-0}"
export TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
export TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
export TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
export TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.25}"
export PAIR_LOSS_NORMALIZATION="${PAIR_LOSS_NORMALIZATION:-none}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0722_sharpness_loss_grid_qwen35_llama2.sh train_eval \
    > T0722_sharpness_loss_grid_qwen35_llama2.nohup.log 2>&1 &

Modes:
  train_eval   train every objective, then eval sharpness 1/5/7
  train        train every objective only
  eval         eval existing routers only, using the same RUN_STAMP/OUT_PREFIX

Default scope:
  MODELS=qwen3.5_4b,llama2_7b_chat
  SPLITS=half16,late21
  OBJECTIVES=pure_mse,mix_w25,mix_w50,mix_w75
  SHARPNESS_LIST=1.0,5.0,7.0

Splits:
  half16: 32-layer models split at middle_layer_idx=16
  late21: 32-layer models split at middle_layer_idx=21, i.e. later 1/3 is layers 21..31

Objectives:
  pure_mse:
    weighted_sum_marginal_mse only
  mix_wK:
    correct_conf_ce + K * weighted_sum_marginal_mse, K in 25/50/75

This launcher reuses the existing T0708 half16 and T0710 late21 two-layer caches.
It does not run base_model/uniform baselines by default.
EOF
}

canonical_model() {
    case "$1" in
        qwen35|qwen3.5_4b) echo qwen3.5_4b ;;
        llama2_7b_chat) echo llama2_7b_chat ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

model_base() {
    case "$1" in
        qwen3.5_4b) echo "Qwen/Qwen3.5-4B" ;;
        llama2_7b_chat) echo "meta-llama/Llama-2-7b-chat-hf" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

model_lora_root() {
    case "$1" in
        qwen3.5_4b) echo "./saves/Qwen/Qwen3.5-4B/lora" ;;
        llama2_7b_chat) echo "./saves/llama2-7b-chat-hf/lora" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

model_config() {
    case "$1" in
        qwen3.5_4b) echo "T0704_qwen35_mrs_ablation_two_layer.py" ;;
        llama2_7b_chat) echo "T0708_llama2_7b_chat_mrs_ablation_two_layer.py" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

base_model_config() {
    case "$1" in
        qwen3.5_4b) echo "T0704_qwen35_base_model.py" ;;
        llama2_7b_chat) echo "T0708_llama2_7b_chat_base_model.py" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

middle_layer_idx() {
    case "$1" in
        half16) echo 16 ;;
        late21) echo 21 ;;
        *) echo "unsupported split=$1" >&2; exit 2 ;;
    esac
}

cache_root() {
    local model="$1" split="$2"
    case "$model:$split" in
        qwen3.5_4b:half16)
            echo "./T0708_half_split_32layer_qwen35_two_layer_20260708_035620/cache/qwen35_9task_3expert_two_layer_half16_800_200"
            ;;
        llama2_7b_chat:half16)
            echo "./T0708_half_split_32layer_llama2_7b_chat_two_layer_20260708_035620/cache/llama2_7b_chat_9task_3expert_two_layer_half16_800_200"
            ;;
        qwen3.5_4b:late21)
            echo "./T0710_late_third_32layer_qwen35_two_layer_20260713_182410/cache/qwen35_9task_3expert_two_layer_late21_800_200"
            ;;
        llama2_7b_chat:late21)
            echo "./T0710_late_third_32layer_llama2_7b_chat_two_layer_20260713_182410/cache/llama2_7b_chat_9task_3expert_two_layer_late21_800_200"
            ;;
        *) echo "unsupported model/split: $model $split" >&2; exit 2 ;;
    esac
}

check_cache() {
    local root="$1"
    [[ -f "$root/train/manifest.json" && -f "$root/validation/manifest.json" ]] || {
        echo "missing cache: $root" >&2
        exit 1
    }
}

objective_env() {
    case "$1" in
        pure_mse)
            export WEIGHTED_SUM_AUX_ONLY=1
            export MSE_WEIGHT=1.0
            export BEST_METRIC=weighted_sum_marginal_mse
            ;;
        mix_w25)
            export WEIGHTED_SUM_AUX_ONLY=0
            export MSE_WEIGHT=25
            export BEST_METRIC=router_argmax_score
            ;;
        mix_w50)
            export WEIGHTED_SUM_AUX_ONLY=0
            export MSE_WEIGHT=50
            export BEST_METRIC=router_argmax_score
            ;;
        mix_w75)
            export WEIGHTED_SUM_AUX_ONLY=0
            export MSE_WEIGHT=75
            export BEST_METRIC=router_argmax_score
            ;;
        *) echo "unsupported objective=$1" >&2; exit 2 ;;
    esac
}

run_one() {
    local model split objective mode out_root root middle
    model="$(canonical_model "$1")"
    split="$2"
    objective="$3"
    mode="$4"
    root="$(cache_root "$model" "$split")"
    middle="$(middle_layer_idx "$split")"
    out_root="./${OUT_PREFIX}_${model}_${split}_two_${objective}_${RUN_STAMP}"

    check_cache "$root"
    objective_env "$objective"
    export MIDDLE_LAYER_IDX="$middle"

    log "[START] model=$model split=$split objective=$objective mode=$mode sharpness=$SHARPNESS_LIST out=$out_root"
    case "$model" in
        qwen3.5_4b)
            RUN_LABEL="T0722_qwen35_${split}_${objective}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$root" \
            BASE_MODEL="$(model_base "$model")" \
            LORA_ROOT="$(model_lora_root "$model")" \
            MODEL_CONFIG="$(model_config "$model")" \
            BASE_MODEL_CONFIG="$(base_model_config "$model")" \
            CACHE_PROMPT_TEMPLATE=chat_template \
            RESUME=1 \
                bash tools/T0704_train_qwen35_two_layer_wsum.sh "$mode"
            ;;
        llama2_7b_chat)
            RUN_LABEL="T0722_llama2_${split}_${objective}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$root" \
            BASE_MODEL="$(model_base "$model")" \
            LORA_ROOT="$(model_lora_root "$model")" \
            MODEL_CONFIG="$(model_config "$model")" \
            BASE_MODEL_CONFIG="$(base_model_config "$model")" \
            CACHE_PROMPT_TEMPLATE=chat_template \
            RESUME=1 \
                bash tools/T0707_train_llama3_8b_two_layer_wsum.sh "$mode"
            ;;
    esac
    log "[DONE] model=$model split=$split objective=$objective mode=$mode out=$out_root"
}

run_grid() {
    local mode="$1" raw_model model split objective
    split_csv "$MODELS_CSV"
    local -a models=("${SPLIT_RESULT[@]}")
    split_csv "$SPLITS_CSV"
    local -a splits=("${SPLIT_RESULT[@]}")
    split_csv "$OBJECTIVES_CSV"
    local -a objectives=("${SPLIT_RESULT[@]}")

    for raw_model in "${models[@]}"; do
        model="$(canonical_model "$raw_model")"
        for split in "${splits[@]}"; do
            for objective in "${objectives[@]}"; do
                run_one "$model" "$split" "$objective" "$mode"
            done
        done
    done
}

main() {
    case "$MODE" in
        -h|--help|help) usage; exit 0 ;;
        train_eval|train|eval) ;;
        *) usage; exit 2 ;;
    esac

    log "[INFO] run_stamp=$RUN_STAMP mode=$MODE models=$MODELS_CSV splits=$SPLITS_CSV objectives=$OBJECTIVES_CSV"
    log "[INFO] sharpness=$SHARPNESS_LIST include_mrs_plus=$INCLUDE_MRS_PLUS include_new_only=$INCLUDE_NEW_ONLY eval_baselines=$EVAL_BASELINES"

    case "$MODE" in
        train_eval)
            run_grid train
            run_grid eval
            ;;
        train|eval)
            run_grid "$MODE"
            ;;
    esac
}

main "$@"
