#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0616_hard_correct_conf_raw_t0p25_freezebert_mrs_only_plus_${RUN_STAMP}}"
RUN_LABEL="${RUN_LABEL:-T0616HARD_MRSPLUS}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
BEST_METRIC="${BEST_METRIC:-route_correct_acc}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.25}"
PAIR_LOSS_NORMALIZATION="${PAIR_LOSS_NORMALIZATION:-none}"
TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
ROUTING_MODE="${ROUTING_MODE:-hard}"
ROUTING_SHARPNESS="${ROUTING_SHARPNESS:-1.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"

LOG_DIR="$OUT_ROOT/logs"
ROUTER_DIR="$OUT_ROOT/routers"
OC_OFFICIAL_DIR="$OUT_ROOT/opencompass_official"
RECORD_DIR="$OUT_ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-$OUT_ROOT/pipeline.log}"

MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
OFFICIAL_ALL_DATASETS=(
    SuperGLUE_BoolQ_gen
    medmcqa_gen_sft_prompt
    obqa_main_gen
    ARC_c_gen
    piqa_gen
    race_gen_sft_prompt
    SuperGLUE_RTE_gen
    siqa_gen
    sst2_gen
)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash ./tools/T0616_train_hard_raw_t0p25_freezebert_mrs_only_plus_qwen.sh > T0616_hard_raw_t0p25_mrs_only_plus.nohup.log 2>&1 &

Pipeline:
  1. Train mrs_only router with freeze_bert.
  2. Train mrs_plus routers with freeze_bert.
  3. Run official OpenCompass eval with hard routing.

This script intentionally skips:
  - train_split eval
  - new_only training/eval

Training target/loss:
  joint_loss=correct_conf_ce
  correct_soft_ce_temperature=0.25
  pair_loss_normalization=none
  target_empty_fallback=uniform
  target_distribution_policy=cache_oracle
  weighted_sum_marginal_mse_weight=0.0
  no weighted_sum_aux_only
  freeze_bert

Outputs are written under a fresh OUT_ROOT.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

setup_root() {
    if [[ -e "$OUT_ROOT" ]]; then
        echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OC_OFFICIAL_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

require_inputs() {
    [[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || {
        echo "missing completed cache: $CACHE_ROOT" >&2
        exit 1
    }
}

require_router() {
    local router="$1"
    [[ -f "$router/router_heads.pt" && -f "$router/router_config.json" ]] || {
        echo "missing router checkpoint: $router" >&2
        exit 1
    }
}

task_dataset_for() {
    case "$1" in
        boolq) echo "SuperGLUE_BoolQ_gen" ;;
        rte) echo "SuperGLUE_RTE_gen" ;;
        siqa) echo "siqa_gen" ;;
        piqa) echo "piqa_gen" ;;
        openbookqa) echo "obqa_main_gen" ;;
        arc_c) echo "ARC_c_gen" ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

mrs_only_router() {
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_mrs_only_freezebert_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

mrs_plus_router() {
    local task="$1"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_mrs_plus_${task}_freezebert_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

train_router() {
    local label="$1" sample_tasks="$2" out_dir="$3" log_file
    log_file="$LOG_DIR/${RUN_LABEL}_train_${label}_${RUN_STAMP}.log"
    [[ ! -e "$out_dir" && ! -e "$log_file" ]] || {
        echo "train output exists: $out_dir $log_file" >&2
        exit 1
    }

    log "[START] train label=$label sample_tasks=$sample_tasks out=$out_dir"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        --out_dir "$out_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature "$TARGET_TEMPERATURE" \
        --pair_loss_normalization "$PAIR_LOSS_NORMALIZATION" \
        --target_empty_fallback "$TARGET_EMPTY_FALLBACK" \
        --target_distribution_policy "$TARGET_DISTRIBUTION_POLICY" \
        --weighted_sum_marginal_mse_weight 0.0 \
        --best_metric "$BEST_METRIC" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1
    log "[DONE] train label=$label out=$out_dir log=$log_file"
}

eval_official() {
    local label="$1" router="$2"
    shift 2
    local log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router"

    log_file="$LOG_DIR/${RUN_LABEL}_official_${label}_${ROUTING_MODE}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_${ROUTING_MODE}_${RUN_STAMP}"
    record_path="$RECORD_DIR/${RUN_LABEL}_official_${label}_${ROUTING_MODE}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] official label=$label mode=$ROUTING_MODE datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$ROUTING_MODE" \
    T0531_ROUTING_SHARPNESS="$ROUTING_SHARPNESS" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_official_${label}_${ROUTING_MODE}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official label=$label mode=$ROUTING_MODE log=$log_file"
}

train_all() {
    local task
    train_router "mrs_only" "$EXPERTS" "$(mrs_only_router)"
    for task in "${TASK_LIST[@]}"; do
        train_router "mrs_plus_${task}" "$EXPERTS,$task" "$(mrs_plus_router "$task")"
    done
}

eval_all_official() {
    local task dataset
    eval_official "mrs_only" "$(mrs_only_router)" "${OFFICIAL_ALL_DATASETS[@]}"
    for task in "${TASK_LIST[@]}"; do
        dataset="$(task_dataset_for "$task")"
        eval_official "mrs_plus_${task}" "$(mrs_plus_router "$task")" "$dataset" "${MRS_DATASETS[@]}"
    done
}

main() {
    case "${1:-}" in
        -h|--help|help) usage; exit 0 ;;
        "") ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_CSV_RESULT[@]}")
    local task
    for task in "${TASK_LIST[@]}"; do
        validate_task "$task"
    done

    setup_root
    require_inputs
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT"
    log "[INFO] tasks=${TASK_LIST[*]} experts=$EXPERTS cuda=$CUDA_VISIBLE_DEVICES"
    log "[INFO] train_loss=correct_conf_ce weighted_sum_marginal_mse_weight=0.0 weighted_sum_aux_only=0 best_metric=$BEST_METRIC freeze_bert=1"
    log "[INFO] target_temperature=$TARGET_TEMPERATURE pair_loss_normalization=$PAIR_LOSS_NORMALIZATION target_empty_fallback=$TARGET_EMPTY_FALLBACK target_distribution_policy=$TARGET_DISTRIBUTION_POLICY"
    log "[INFO] eval=official_only routing_mode=$ROUTING_MODE routing_sharpness=$ROUTING_SHARPNESS"
    log "[INFO] skipped=train_split,new_only"

    train_all
    eval_all_official

    log "[DONE] mrs_only + mrs_plus hard-routing raw t0.25 freezebert pipeline output=$OUT_ROOT"
}

main "$@"
