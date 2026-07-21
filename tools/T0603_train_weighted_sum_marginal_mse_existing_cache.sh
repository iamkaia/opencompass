#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXP_ROOT="${EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
WEIGHT_TAG="${WEIGHT_TAG:-$(printf '%s' "$MSE_WEIGHT" | sed 's/-/m/g; s/\./p/g')}"
AUX_ONLY="${AUX_ONLY:-0}"
JOINT_LOSS="${JOINT_LOSS:-correct_conf_ce}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-1.0}"
LOG_PREFIX="${LOG_PREFIX:-T0603}"
BEST_METRIC="${BEST_METRIC:-route_correct_acc}"

LOG_DIR="$EXP_ROOT/logs"
CACHE_ROOT="${CACHE_ROOT:-$EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
ROUTER_DIR="$EXP_ROOT/routers"
BERT="./task_classifier_ckpt"
EXPERTS="medmcqa,race,sst2"
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0603_train_weighted_sum_marginal_mse_existing_cache.sh [all|mrs_only|tasks] [task_csv]

Uses the existing T0603 cached dataset under EXP_ROOT and writes new router
checkpoints/logs into the same EXP_ROOT with a wsmse suffix.

Environment:
  EXP_ROOT     default ./T0603_qwenfix_trainbert_chattemplate_20260603_063927
  CACHE_ROOT   default the existing cache directory under EXP_ROOT
  MSE_WEIGHT  default 1.0
  WEIGHT_TAG  default derived from MSE_WEIGHT, e.g. 1p0
  AUX_ONLY    set to 1 to train only weighted_sum_marginal_mse
  JOINT_LOSS  target construction mode; default correct_conf_ce
  TARGET_TEMPERATURE target distribution temperature; default 1.0
  LOG_PREFIX  default T0603
  BEST_METRIC default route_correct_acc
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

validate_task() {
    case "$1" in boolq|rte|siqa|piqa|openbookqa|arc_c) ;; *) echo "unsupported task: $1" >&2; exit 2 ;; esac
}

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

require_cache() {
    [[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || {
        echo "missing completed cache: $CACHE_ROOT" >&2
        exit 1
    }
}

router_dir_for() {
    local task="${1:-}"
    if [[ -z "$task" ]]; then
        echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmse${WEIGHT_TAG}_3expert_sst2words"
    else
        echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmse${WEIGHT_TAG}_3expert_sst2words"
    fi
}

common_training_args() {
    printf '%s\n' \
        --router_dim 512 \
        --batch_size 32 \
        --epochs 10 \
        --lr 2e-4 \
        --joint_loss "$JOINT_LOSS" \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature "$TARGET_TEMPERATURE" \
        --pair_loss_normalization sample_minmax \
        --weighted_sum_marginal_mse_weight "$MSE_WEIGHT" \
        --best_metric "$BEST_METRIC" \
        --early_stop_patience 2 \
        --train_bert \
        --save_route_records \
        --eval_train_each_epoch
    if [[ "$AUX_ONLY" == "1" ]]; then
        printf '%s\n' --weighted_sum_aux_only
    fi
}

train_one() {
    local label="$1" sample_tasks="$2" output_dir log_file
    local -a args
    require_cache
    mkdir -p "$LOG_DIR" "$ROUTER_DIR"
    output_dir="$(router_dir_for "$label")"
    if [[ -z "$label" ]]; then
        log_file="$LOG_DIR/${LOG_PREFIX}_train_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_wsmse${WEIGHT_TAG}_${RUN_STAMP}.log"
    else
        log_file="$LOG_DIR/${LOG_PREFIX}_train_qwen3_fp16_mrs_plus_${label}_taskcls_trainbert_qwenfix_wsmse${WEIGHT_TAG}_${RUN_STAMP}.log"
    fi
    [[ ! -e "$output_dir" ]] || { echo "router output exists: $output_dir" >&2; exit 1; }
    [[ ! -e "$log_file" ]] || { echo "log output exists: $log_file" >&2; exit 1; }
    mapfile -t args < <(common_training_args)
    log "[START] train label=${label:-mrs_only} mse_weight=$MSE_WEIGHT out=$output_dir log=$log_file"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        --out_dir "$output_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        "${args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] train label=${label:-mrs_only} out=$output_dir"
}

main() {
    local mode="${1:-all}"
    local task_csv="${2:-$DEFAULT_TASKS}"
    log "[INFO] exp_root=$EXP_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT"
    log "[INFO] mse_weight=$MSE_WEIGHT weight_tag=$WEIGHT_TAG aux_only=$AUX_ONLY joint_loss=$JOINT_LOSS target_temperature=$TARGET_TEMPERATURE log_prefix=$LOG_PREFIX best_metric=$BEST_METRIC"
    case "$mode" in
        all)
            train_one "" "$EXPERTS"
            split_csv "$task_csv"
            for task in "${SPLIT_CSV_RESULT[@]}"; do
                validate_task "$task"
                train_one "$task" "$EXPERTS,$task"
            done
            ;;
        mrs_only)
            train_one "" "$EXPERTS"
            ;;
        tasks)
            split_csv "$task_csv"
            for task in "${SPLIT_CSV_RESULT[@]}"; do
                validate_task "$task"
                train_one "$task" "$EXPERTS,$task"
            done
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "$@"
