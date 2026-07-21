#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

MODE="${1:-all}"
MODELS_CSV="${MODELS:-qwen3_4b,qwen3.5_4b,llama2_7b_chat}"
GRID_CSV="${GRID:-w25,w50,w75}"
OUT_PREFIX="${OUT_PREFIX:-T0720_late_third}"

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
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0720_late_third_two_layer_wgrid_qwen3_qwen35_llama2.sh all \
    > T0720_late_third_qwen3_qwen35_llama2.nohup.log 2>&1 &

Modes:
  all             run two_layer w25/w50/w75 for all MODELS, plus qwen3_4b pure-MSE two_layer and diag-single pure-MSE
  two             run only two_layer w25/w50/w75 for MODELS
  qwen3_puremse   run only qwen3_4b pure-MSE two_layer
  qwen3_diag      derive qwen3_4b diagonal cache from two_layer cache, then run pure-MSE single_all_layers
  diag_cache      derive qwen3_4b diagonal cache only

Defaults:
  MODELS=qwen3_4b,qwen3.5_4b,llama2_7b_chat
  GRID=w25,w50,w75
  qwen3.5_4b/llama2 late-third cache uses middle_layer_idx=21
  qwen3_4b default middle_layer_idx=24 because local HF config has 36 layers

Objectives:
  two_layer wK:
    correct_conf_ce + K * weighted_sum_marginal_mse
  qwen3_4b pure-MSE two_layer:
    1.0 * weighted_sum_marginal_mse only
  qwen3_4b diag-single pure-MSE:
    1.0 * weighted_sum_mse over diag(loss_matrix) from the qwen3_4b two_layer cache
EOF
}

grid_weight() {
    case "$1" in
        w25) echo 25 ;;
        w50) echo 50 ;;
        w75) echo 75 ;;
        *) echo "unsupported grid=$1" >&2; exit 2 ;;
    esac
}

model_middle_layer() {
    case "$1" in
        qwen3_4b) echo "${QWEN3_4B_MIDDLE_LAYER_IDX:-24}" ;;
        qwen35|qwen3.5_4b) echo "${QWEN35_MIDDLE_LAYER_IDX:-21}" ;;
        llama2_7b_chat) echo "${LLAMA2_MIDDLE_LAYER_IDX:-21}" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

model_base() {
    case "$1" in
        qwen3_4b) echo "Qwen/Qwen3-4B-Instruct-2507" ;;
        qwen35|qwen3.5_4b) echo "Qwen/Qwen3.5-4B" ;;
        llama2_7b_chat) echo "meta-llama/Llama-2-7b-chat-hf" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

model_lora_root() {
    case "$1" in
        qwen3_4b) echo "./saves/Qwen/Qwen3-4B-Instruct-2507/lora" ;;
        qwen35|qwen3.5_4b) echo "./saves/Qwen/Qwen3.5-4B/lora" ;;
        llama2_7b_chat) echo "./saves/llama2-7b-chat-hf/lora" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

model_base_config() {
    case "$1" in
        qwen3_4b) echo "hf_qwen3_4b_instruct_2507_64.py" ;;
        qwen35|qwen3.5_4b) echo "T0704_qwen35_base_model.py" ;;
        llama2_7b_chat) echo "T0708_llama2_7b_chat_base_model.py" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

two_cache_root() {
    case "$1" in
        qwen3_4b)
            echo "${QWEN3_4B_TWO_CACHE_ROOT:-./${OUT_PREFIX}_qwen3_4b_two_cache_late$(model_middle_layer qwen3_4b)_${RUN_STAMP}/cache/qwen3_4b_9task_3expert_two_layer_late$(model_middle_layer qwen3_4b)_800_200}"
            ;;
        qwen35|qwen3.5_4b)
            echo "${QWEN35_TWO_CACHE_ROOT:-./T0710_late_third_32layer_qwen35_two_layer_20260713_182410/cache/qwen35_9task_3expert_two_layer_late21_800_200}"
            ;;
        llama2_7b_chat)
            echo "${LLAMA2_TWO_CACHE_ROOT:-./T0710_late_third_32layer_llama2_7b_chat_two_layer_20260713_182410/cache/llama2_7b_chat_9task_3expert_two_layer_late21_800_200}"
            ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

diag_cache_root() {
    echo "./${OUT_PREFIX}_qwen3_4b_diag_cache_from_puremse_two_late$(model_middle_layer qwen3_4b)_${RUN_STAMP}/cache/diag_single_from_two_layer"
}

check_cache() {
    local cache_root="$1"
    [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]] || {
        echo "missing cache: $cache_root" >&2
        exit 1
    }
}

build_qwen3_two_cache() {
    local cache_root out_root middle
    cache_root="$(two_cache_root qwen3_4b)"
    middle="$(model_middle_layer qwen3_4b)"
    out_root="./${OUT_PREFIX}_qwen3_4b_two_cache_late${middle}_${RUN_STAMP}"

    if [[ -f "$cache_root/train/manifest.json" && -f "$cache_root/validation/manifest.json" ]]; then
        log "[SKIP] qwen3_4b two cache cache=$cache_root"
        return 0
    fi

    log "[START] qwen3_4b two cache middle_layer_idx=$middle cache=$cache_root"
    RUN_LABEL="T0720_qwen3_4b_cache_late${middle}" \
    OUT_ROOT="$out_root" \
    CACHE_ROOT="$cache_root" \
    BASE_MODEL="$(model_base qwen3_4b)" \
    LORA_ROOT="$(model_lora_root qwen3_4b)" \
    MODEL_CONFIG=T0720_qwen_router_mrs_ablation_two_layer_env.py \
    BASE_MODEL_CONFIG=hf_qwen3_4b_instruct_2507_64.py \
    MIDDLE_LAYER_IDX="$middle" \
    RESUME=1 \
        bash tools/T0704_train_qwen35_two_layer_wsum.sh cache
    log "[DONE] qwen3_4b two cache cache=$cache_root"
}

ensure_two_cache() {
    case "$1" in
        qwen3_4b) build_qwen3_two_cache ;;
        qwen35|qwen3.5_4b|llama2_7b_chat) check_cache "$(two_cache_root "$1")" ;;
        *) echo "unsupported model=$1" >&2; exit 2 ;;
    esac
}

run_two() {
    local model="$1" grid="$2" mode="$3" weight out_root cache_root middle
    weight="$(grid_weight "$grid")"
    cache_root="$(two_cache_root "$model")"
    middle="$(model_middle_layer "$model")"
    out_root="./${OUT_PREFIX}_${model}_two_${grid}_late${middle}_${RUN_STAMP}"

    ensure_two_cache "$model"
    export WEIGHTED_SUM_AUX_ONLY=0
    export MSE_WEIGHT="$weight"
    export TARGET_TEMPERATURE=0.25
    export PAIR_LOSS_NORMALIZATION=none
    export BEST_METRIC=router_argmax_score
    export MIDDLE_LAYER_IDX="$middle"

    log "[START] two model=$model grid=$grid mode=$mode middle_layer_idx=$middle mse=$MSE_WEIGHT out=$out_root"
    case "$model" in
        qwen3_4b|qwen35|qwen3.5_4b)
            RUN_LABEL="T0720_${model}_two_${grid}_late${middle}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL="$(model_base "$model")" \
            LORA_ROOT="$(model_lora_root "$model")" \
            MODEL_CONFIG=T0720_qwen_router_mrs_ablation_two_layer_env.py \
            BASE_MODEL_CONFIG="$(model_base_config "$model")" \
            T0720_ROUTER_RECORD_PREFIX="$model" \
            RESUME=1 \
                bash tools/T0704_train_qwen35_two_layer_wsum.sh "$mode"
            ;;
        llama2_7b_chat)
            RUN_LABEL="T0720_llama2_two_${grid}_late${middle}" \
            OUT_ROOT="$out_root" \
            CACHE_ROOT="$cache_root" \
            BASE_MODEL="$(model_base "$model")" \
            LORA_ROOT="$(model_lora_root "$model")" \
            MODEL_CONFIG=T0708_llama2_7b_chat_mrs_ablation_two_layer.py \
            BASE_MODEL_CONFIG=T0708_llama2_7b_chat_base_model.py \
            CACHE_PROMPT_TEMPLATE=chat_template \
            RESUME=1 \
                bash tools/T0707_train_llama3_8b_two_layer_wsum.sh "$mode"
            ;;
    esac
    log "[DONE] two model=$model grid=$grid mode=$mode out=$out_root"
}

run_qwen3_puremse_two() {
    local mode="$1" out_root cache_root middle
    middle="$(model_middle_layer qwen3_4b)"
    cache_root="$(two_cache_root qwen3_4b)"
    out_root="./${OUT_PREFIX}_qwen3_4b_two_puremse_late${middle}_${RUN_STAMP}"

    ensure_two_cache qwen3_4b
    export WEIGHTED_SUM_AUX_ONLY=1
    export MSE_WEIGHT=1.0
    export TARGET_TEMPERATURE=0.25
    export PAIR_LOSS_NORMALIZATION=none
    export BEST_METRIC=weighted_sum_marginal_mse
    export MIDDLE_LAYER_IDX="$middle"

    log "[START] qwen3_4b pure-MSE two mode=$mode middle_layer_idx=$middle out=$out_root"
    RUN_LABEL="T0720_qwen3_4b_two_puremse_late${middle}" \
    OUT_ROOT="$out_root" \
    CACHE_ROOT="$cache_root" \
    BASE_MODEL="$(model_base qwen3_4b)" \
    LORA_ROOT="$(model_lora_root qwen3_4b)" \
    MODEL_CONFIG=T0720_qwen_router_mrs_ablation_two_layer_env.py \
    BASE_MODEL_CONFIG="$(model_base_config qwen3_4b)" \
    T0720_ROUTER_RECORD_PREFIX=qwen3_4b_puremse \
    RESUME=1 \
        bash tools/T0704_train_qwen35_two_layer_wsum.sh "$mode"
    log "[DONE] qwen3_4b pure-MSE two mode=$mode out=$out_root"
}

build_qwen3_diag_cache() {
    local src dst
    ensure_two_cache qwen3_4b
    src="$(two_cache_root qwen3_4b)"
    dst="$(diag_cache_root)"
    if [[ -f "$dst/train/manifest.json" && -f "$dst/validation/manifest.json" ]]; then
        log "[SKIP] qwen3_4b diag cache cache=$dst"
        return 0
    fi
    if [[ -e "$dst" ]]; then
        echo "incomplete diag cache exists: $dst" >&2
        exit 1
    fi
    log "[START] qwen3_4b diag cache source=$src output=$dst"
    "$PY" -u tools/convert_two_layer_cache_to_diag_single.py \
        --source_cache "$src" \
        --output_cache "$dst"
    log "[DONE] qwen3_4b diag cache output=$dst"
}

run_qwen3_diag_single_puremse() {
    local mode="$1" cache_root out_root middle
    middle="$(model_middle_layer qwen3_4b)"
    cache_root="$(diag_cache_root)"
    out_root="./${OUT_PREFIX}_qwen3_4b_diag_single_puremse_from_two_late${middle}_${RUN_STAMP}"
    [[ "$mode" == "eval" ]] || build_qwen3_diag_cache
    check_cache "$cache_root"

    log "[START] qwen3_4b diag-single pure-MSE mode=$mode middle_layer_idx=$middle out=$out_root"
    RUN_LABEL="T0720_qwen3_4b_diag_single_puremse_late${middle}" \
    OUT_ROOT="$out_root" \
    CACHE_ROOT="$cache_root" \
    BASE_MODEL="$(model_base qwen3_4b)" \
    LORA_ROOT="$(model_lora_root qwen3_4b)" \
    MODEL_CONFIG=T0720_qwen_router_mrs_ablation_single_all_layers_env.py \
    BASE_MODEL_CONFIG="$(model_base_config qwen3_4b)" \
    T0720_ROUTER_RECORD_PREFIX=qwen3_4b_diag_single_puremse \
    MIDDLE_LAYER_IDX="$middle" \
    EXPERT_CE_WEIGHT=0 \
    MSE_WEIGHT=1.0 \
    RESUME=1 \
        bash tools/T0704_train_qwen35_single_all_layers_wsum.sh "$mode"
    log "[DONE] qwen3_4b diag-single pure-MSE mode=$mode out=$out_root"
}

run_two_grid_train_eval() {
    local model grid
    split_csv "$MODELS_CSV"
    local -a models=("${SPLIT_RESULT[@]}")
    split_csv "$GRID_CSV"
    local -a grids=("${SPLIT_RESULT[@]}")
    for model in "${models[@]}"; do
        for grid in "${grids[@]}"; do
            run_two "$model" "$grid" train
            run_two "$model" "$grid" eval
        done
    done
}

main() {
    case "$MODE" in
        -h|--help|help) usage; exit 0 ;;
        all|two|qwen3_puremse|qwen3_diag|diag_cache) ;;
        *) usage; exit 2 ;;
    esac

    log "[INFO] run_stamp=$RUN_STAMP mode=$MODE models=$MODELS_CSV grid=$GRID_CSV out_prefix=$OUT_PREFIX"
    log "[INFO] qwen3_4b_mid=$(model_middle_layer qwen3_4b) qwen3.5_4b_mid=$(model_middle_layer qwen3.5_4b) llama2_mid=$(model_middle_layer llama2_7b_chat)"
    log "[INFO] include_mrs_plus=$INCLUDE_MRS_PLUS include_new_only=$INCLUDE_NEW_ONLY eval_baselines=$EVAL_BASELINES sharpness=$SHARPNESS_LIST"

    case "$MODE" in
        all)
            run_two_grid_train_eval
            run_qwen3_puremse_two train
            run_qwen3_puremse_two eval
            run_qwen3_diag_single_puremse train
            run_qwen3_diag_single_puremse eval
            ;;
        two)
            run_two_grid_train_eval
            ;;
        qwen3_puremse)
            run_qwen3_puremse_two train
            run_qwen3_puremse_two eval
            ;;
        qwen3_diag)
            run_qwen3_diag_single_puremse train
            run_qwen3_diag_single_puremse eval
            ;;
        diag_cache)
            build_qwen3_diag_cache
            ;;
    esac
}

main "$@"
