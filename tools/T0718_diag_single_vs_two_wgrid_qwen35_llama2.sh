#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

MODE="${1:-train_eval}"
MODELS_CSV="${MODELS:-qwen35,llama2_7b_chat}"
GRID_CSV="${GRID:-w25,w50,w75}"
OUT_PREFIX="${OUT_PREFIX:-T0718_half16_diagcmp}"

export MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-16}"
export SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
export INCLUDE_MRS_PLUS="${INCLUDE_MRS_PLUS:-1}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
export EVAL_BASELINES="${EVAL_BASELINES:-0}"
export TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
export TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
export TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0718_diag_single_vs_two_wgrid_qwen35_llama2.sh train_eval \
    > T0718_diag_single_vs_two_wgrid.nohup.log 2>&1 &

Modes:
  train_eval   build diagonal cache, train, then eval
  train        build diagonal cache and train only
  eval         eval only
  diag_cache   build diagonal single cache only
  two          train+eval two_layer grid only
  diag_single  build cache, train+eval diagonal single pure-MSE and mix grid

Scope:
  MODELS=qwen35,llama2_7b_chat
  GRID=w25,w50,w75
  Reuses T0708 half16 two_layer caches, then derives diagonal single caches.

Comparison:
  two_layer wK:
    correct_conf_ce + K * weighted_sum_marginal_mse
  diag_single_mix_wK:
    expert_ce + K * weighted_sum_mse over diag(loss_matrix) from the same two_layer cache
  diag_single_mse:
    1.0 * weighted_sum_mse over diag(loss_matrix) from the same two_layer cache
EOF
}

check_cache() {
    local cache_root="$1"
    [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]] || {
        echo "missing cache: $cache_root" >&2
        exit 1
    }
}

grid_weight() {
    case "$1" in
        w25) echo 25 ;;
        w50) echo 50 ;;
        w75) echo 75 ;;
        *) echo "unsupported grid=$1" >&2; exit 2 ;;
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
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

diag_cache_root() {
    local model="$1"
    echo "./${OUT_PREFIX}_${model}_diag_cache_${RUN_STAMP}/cache/diag_single_from_two_half16"
}

build_diag_cache() {
    local model="$1" src dst
    src="$(two_cache_root "$model")"
    dst="$(diag_cache_root "$model")"
    check_cache "$src"
    if [[ -f "$dst/train/manifest.json" && -f "$dst/validation/manifest.json" ]]; then
        log "[SKIP] diag cache model=$model cache=$dst"
        return 0
    fi
    if [[ -e "$dst" ]]; then
        echo "incomplete diag cache exists: $dst" >&2
        exit 1
    fi
    log "[START] diag cache model=$model source=$src output=$dst"
    "$PY" -u tools/convert_two_layer_cache_to_diag_single.py \
        --source_cache "$src" \
        --output_cache "$dst"
    log "[DONE] diag cache model=$model output=$dst"
}

run_two() {
    local model="$1" grid="$2" mode="$3" weight out_root cache_root
    weight="$(grid_weight "$grid")"
    out_root="./${OUT_PREFIX}_${model}_two_${grid}_${RUN_STAMP}"
    cache_root="$(two_cache_root "$model")"
    check_cache "$cache_root"

    export WEIGHTED_SUM_AUX_ONLY=0
    export MSE_WEIGHT="$weight"
    export TARGET_TEMPERATURE=0.25
    export PAIR_LOSS_NORMALIZATION=none
    export BEST_METRIC=router_argmax_score

    log "[START] two model=$model grid=$grid mode=$mode mse=$MSE_WEIGHT out=$out_root"
    case "$model" in
        qwen35)
            RUN_LABEL="T0718_diagcmp_qwen35_two_${grid}" \
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
            RUN_LABEL="T0718_diagcmp_llama2_two_${grid}" \
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
    log "[DONE] two model=$model grid=$grid mode=$mode out=$out_root"
}

run_single_launcher() {
    local model="$1" cache_root="$2" out_root="$3" mode="$4" expert_ce_weight="$5" mse_weight="$6" label="$7"
    check_cache "$cache_root"
    log "[START] single model=$model label=$label mode=$mode expert_ce=$expert_ce_weight mse=$mse_weight out=$out_root"
    case "$model" in
        qwen35)
            RUN_LABEL="T0718_diagcmp_qwen35_${label}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL=Qwen/Qwen3.5-4B \
            LORA_ROOT=./saves/Qwen/Qwen3.5-4B/lora \
            MODEL_CONFIG=T0704_qwen35_mrs_ablation_single_all_layers.py \
            BASE_MODEL_CONFIG=T0704_qwen35_base_model.py \
            MSE_WEIGHT="$mse_weight" \
            EXPERT_CE_WEIGHT="$expert_ce_weight" \
            RESUME=1 \
                bash tools/T0704_train_qwen35_single_all_layers_wsum.sh "$mode"
            ;;
        llama2_7b_chat)
            RUN_LABEL="T0718_diagcmp_llama2_${label}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL=meta-llama/Llama-2-7b-chat-hf \
            LORA_ROOT=./saves/llama2-7b-chat-hf/lora \
            MODEL_CONFIG=T0702_mrs_ablation_single_all_layers.py \
            BASE_MODEL_CONFIG=T0708_llama2_7b_chat_base_model.py \
            CACHE_PROMPT_TEMPLATE=chat_template \
            MSE_WEIGHT="$mse_weight" \
            EXPERT_CE_WEIGHT="$expert_ce_weight" \
            RESUME=1 \
                bash tools/T0702_train_llama2_single_all_layers_wsum.sh "$mode"
            ;;
    esac
    log "[DONE] single model=$model label=$label mode=$mode out=$out_root"
}

run_diag_single() {
    local model="$1" mode="$2" cache_root out_root
    cache_root="$(diag_cache_root "$model")"
    out_root="./${OUT_PREFIX}_${model}_diag_single_mse_${RUN_STAMP}"
    [[ "$mode" == "eval" ]] || build_diag_cache "$model"
    run_single_launcher "$model" "$cache_root" "$out_root" "$mode" 0 1.0 "diag_single_mse"
}

run_diag_single_mix() {
    local model="$1" grid="$2" mode="$3" weight cache_root out_root
    weight="$(grid_weight "$grid")"
    cache_root="$(diag_cache_root "$model")"
    out_root="./${OUT_PREFIX}_${model}_diag_single_mix_${grid}_${RUN_STAMP}"
    [[ "$mode" == "eval" ]] || build_diag_cache "$model"
    run_single_launcher "$model" "$cache_root" "$out_root" "$mode" 1.0 "$weight" "diag_single_mix_${grid}"
}

run_all_train_or_eval() {
    local mode="$1" model grid
    split_csv "$MODELS_CSV"
    local -a models=("${SPLIT_RESULT[@]}")
    split_csv "$GRID_CSV"
    local -a grids=("${SPLIT_RESULT[@]}")
    for model in "${models[@]}"; do
        [[ "$mode" == "eval" ]] || build_diag_cache "$model"
        run_diag_single "$model" "$mode"
        for grid in "${grids[@]}"; do
            run_diag_single_mix "$model" "$grid" "$mode"
            run_two "$model" "$grid" "$mode"
        done
    done
}

main() {
    case "$MODE" in
        -h|--help|help) usage; exit 0 ;;
        train_eval|train|eval|diag_cache|two|diag_single) ;;
        *) usage; exit 2 ;;
    esac
    log "[INFO] run_stamp=$RUN_STAMP mode=$MODE models=$MODELS_CSV grid=$GRID_CSV out_prefix=$OUT_PREFIX"
    log "[INFO] include_mrs_plus=$INCLUDE_MRS_PLUS include_new_only=$INCLUDE_NEW_ONLY eval_baselines=$EVAL_BASELINES sharpness=$SHARPNESS_LIST"

    local model grid
    split_csv "$MODELS_CSV"
    local -a models=("${SPLIT_RESULT[@]}")
    split_csv "$GRID_CSV"
    local -a grids=("${SPLIT_RESULT[@]}")
    case "$MODE" in
        train_eval)
            run_all_train_or_eval train
            run_all_train_or_eval eval
            ;;
        train|eval)
            run_all_train_or_eval "$MODE"
            ;;
        diag_cache)
            for model in "${models[@]}"; do build_diag_cache "$model"; done
            ;;
        two)
            for model in "${models[@]}"; do
                for grid in "${grids[@]}"; do
                    run_two "$model" "$grid" train
                    run_two "$model" "$grid" eval
                done
            done
            ;;
        diag_single)
            for model in "${models[@]}"; do
                run_diag_single "$model" train
                run_diag_single "$model" eval
                for grid in "${grids[@]}"; do
                    run_diag_single_mix "$model" "$grid" train
                    run_diag_single_mix "$model" "$grid" eval
                done
            done
            ;;
    esac
}

main "$@"
