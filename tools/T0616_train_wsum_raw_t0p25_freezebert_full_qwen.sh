#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0616_wsum_correct_conf_raw_t0p25_freezebert_full_${RUN_STAMP}}"
RUN_LABEL="${RUN_LABEL:-T0616}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
TRAIN_SPLIT_DATASET="${TRAIN_SPLIT_DATASET:-router_train_split_gen}"
SHARPNESS_LIST="${SHARPNESS_LIST:-1.0,2.0,3.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
WEIGHTED_SUM_AUX_ONLY="${WEIGHTED_SUM_AUX_ONLY:-1}"
BEST_METRIC="${BEST_METRIC:-weighted_sum_marginal_mse}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.25}"
PAIR_LOSS_NORMALIZATION="${PAIR_LOSS_NORMALIZATION:-none}"
TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"

LOG_DIR="$OUT_ROOT/logs"
ROUTER_DIR="$OUT_ROOT/routers"
OC_TRAIN_SPLIT_DIR="$OUT_ROOT/opencompass_train_split"
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
  CUDA_VISIBLE_DEVICES=0 bash tools/T0616_train_wsum_raw_t0p25_freezebert_full_qwen.sh

Pipeline:
  1. Train mrs_only weighted_sum router with freeze_bert.
  2. Train mrs_plus routers with freeze_bert.
  3. Train new_only routers with freeze_bert, each loading the mrs_only checkpoint.
  4. Run OpenCompass train_split for mrs_only, every mrs_plus, and every new_only router.
  5. Run official OpenCompass weighted_sum eval at sharpness 1/2/3 for all routers.

Training target/loss:
  joint_loss=correct_conf_ce
  correct_soft_ce_temperature=0.25
  pair_loss_normalization=none
  target_empty_fallback=uniform
  target_distribution_policy=cache_oracle
  weighted_sum_marginal_mse_weight=1.0
  weighted_sum_aux_only
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
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OC_TRAIN_SPLIT_DIR" "$OC_OFFICIAL_DIR" "$RECORD_DIR"
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

temp_tag() {
    printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
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
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_mrs_only_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

mrs_plus_router() {
    local task="$1"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_mrs_plus_${task}_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

new_only_router() {
    local task="$1"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_new_only_${task}_from_mrs_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

train_router() {
    local label="$1" sample_tasks="$2" out_dir="$3" load_from="${4:-}" log_file
    log_file="$LOG_DIR/${RUN_LABEL}_train_${label}_${RUN_STAMP}.log"
    [[ ! -e "$out_dir" && ! -e "$log_file" ]] || {
        echo "train output exists: $out_dir $log_file" >&2
        exit 1
    }

    local -a load_args=()
    local -a objective_args=(
        --weighted_sum_marginal_mse_weight "$MSE_WEIGHT"
        --best_metric "$BEST_METRIC"
    )
    if [[ "$WEIGHTED_SUM_AUX_ONLY" == "1" ]]; then
        objective_args+=(--weighted_sum_aux_only)
    elif [[ "$WEIGHTED_SUM_AUX_ONLY" != "0" ]]; then
        echo "WEIGHTED_SUM_AUX_ONLY must be 0 or 1, got: $WEIGHTED_SUM_AUX_ONLY" >&2
        exit 2
    fi
    if [[ -n "$load_from" ]]; then
        require_router "$load_from"
        load_args+=(--load_from "$load_from")
    fi

    log "[START] train label=$label sample_tasks=$sample_tasks load_from=${load_from:-none} out=$out_dir"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        "${load_args[@]}" \
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
        "${objective_args[@]}" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1
    log "[DONE] train label=$label out=$out_dir log=$log_file"
}

eval_train_split() {
    local label="$1" router="$2" log_file work_dir record_path
    require_router "$router"
    log_file="$LOG_DIR/${RUN_LABEL}_train_split_${label}_${RUN_STAMP}.log"
    work_dir="$OC_TRAIN_SPLIT_DIR/${label}_weighted_sum_${RUN_STAMP}"
    record_path="$RECORD_DIR/${RUN_LABEL}_train_split_${label}_weighted_sum_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "train_split eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] train_split label=$label router=$router"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS=1.0 \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_train_split_${label}_weighted_sum" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "$TRAIN_SPLIT_DATASET" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] train_split label=$label log=$log_file"
}

eval_official() {
    local label="$1" router="$2" sharpness="$3"
    shift 3
    local sharp_tag log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router"
    sharp_tag="$(temp_tag "$sharpness")"
    log_file="$LOG_DIR/${RUN_LABEL}_official_${label}_sharp${sharp_tag}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}"
    record_path="$RECORD_DIR/${RUN_LABEL}_official_${label}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] official label=$label sharpness=$sharpness datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_official_${label}_sharp${sharp_tag}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official label=$label sharpness=$sharpness log=$log_file"
}

train_all() {
    local task
    train_router "mrs_only" "$EXPERTS" "$(mrs_only_router)"

    for task in "${TASK_LIST[@]}"; do
        train_router "mrs_plus_${task}" "$EXPERTS,$task" "$(mrs_plus_router "$task")"
    done

    for task in "${TASK_LIST[@]}"; do
        train_router "new_only_${task}" "$task" "$(new_only_router "$task")" "$(mrs_only_router)"
    done
}

eval_all_train_split() {
    local task
    eval_train_split "mrs_only" "$(mrs_only_router)"
    for task in "${TASK_LIST[@]}"; do
        eval_train_split "mrs_plus_${task}" "$(mrs_plus_router "$task")"
    done
    for task in "${TASK_LIST[@]}"; do
        eval_train_split "new_only_${task}" "$(new_only_router "$task")"
    done
}

eval_all_official() {
    local sharpness task dataset
    split_csv "$SHARPNESS_LIST"
    local -a sharpness_values=("${SPLIT_CSV_RESULT[@]}")
    for sharpness in "${sharpness_values[@]}"; do
        eval_official "mrs_only" "$(mrs_only_router)" "$sharpness" "${OFFICIAL_ALL_DATASETS[@]}"
        for task in "${TASK_LIST[@]}"; do
            dataset="$(task_dataset_for "$task")"
            eval_official "mrs_plus_${task}" "$(mrs_plus_router "$task")" "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
        done
        for task in "${TASK_LIST[@]}"; do
            dataset="$(task_dataset_for "$task")"
            eval_official "new_only_${task}" "$(new_only_router "$task")" "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
        done
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
    log "[INFO] run_label=$RUN_LABEL joint_loss=correct_conf_ce weighted_sum_marginal_mse_weight=$MSE_WEIGHT weighted_sum_aux_only=$WEIGHTED_SUM_AUX_ONLY best_metric=$BEST_METRIC freeze_bert=1 target_temperature=$TARGET_TEMPERATURE pair_loss_normalization=$PAIR_LOSS_NORMALIZATION target_empty_fallback=$TARGET_EMPTY_FALLBACK target_distribution_policy=$TARGET_DISTRIBUTION_POLICY"
    log "[INFO] eval train_split=$TRAIN_SPLIT_DATASET official_sharpness=$SHARPNESS_LIST"

    train_all
    eval_all_train_split
    eval_all_official

    log "[DONE] full weighted_sum raw t0.25 freezebert pipeline output=$OUT_ROOT"
}

main "$@"
