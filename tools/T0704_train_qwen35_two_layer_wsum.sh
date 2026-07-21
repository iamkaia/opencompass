#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0704_qwen35_two_layer_wsum_t0p25_${RUN_STAMP}}"
RUN_LABEL="${RUN_LABEL:-T0704_qwen35_two}"
RESUME="${RESUME:-0}"

DATA_ROOT="${DATA_ROOT:-./0602_router_train_dataset}"
CACHE_ROOT="${CACHE_ROOT:-$OUT_ROOT/cache/qwen35_9task_3expert_two_layer_800_200}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
LORA_ROOT="${LORA_ROOT:-./saves/Qwen/Qwen3.5-4B/lora}"
MODEL_CONFIG="${MODEL_CONFIG:-T0704_qwen35_mrs_ablation_two_layer.py}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-T0704_qwen35_base_model.py}"
MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-18}"

SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-800}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-200}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
WEIGHTED_SUM_AUX_ONLY="${WEIGHTED_SUM_AUX_ONLY:-1}"
BEST_METRIC="${BEST_METRIC:-weighted_sum_marginal_mse}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.25}"
PAIR_LOSS_NORMALIZATION="${PAIR_LOSS_NORMALIZATION:-none}"
PAIR_CONSTRAINT="${PAIR_CONSTRAINT:-none}"
TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
INCLUDE_MRS_PLUS="${INCLUDE_MRS_PLUS:-1}"
INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
EVAL_BASELINES="${EVAL_BASELINES:-1}"
MAX_OUT_LEN="${MAX_OUT_LEN:-}"
if [[ -n "$MAX_OUT_LEN" ]]; then
    export T0704_MAX_OUT_LEN="$MAX_OUT_LEN"
fi

LOG_DIR="$OUT_ROOT/logs"
ROUTER_DIR="${ROUTER_DIR:-$OUT_ROOT/routers}"
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
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0704_train_qwen35_two_layer_wsum.sh all > T0704_qwen35_two_layer_wsum.nohup.log 2>&1 &

Modes:
  all       build Qwen3.5 two-layer cache, train routers, eval official, eval base/uniform
  cache     build cache only
  train     train routers from existing cache
  eval      eval trained routers
  baseline  eval base_model and uniform

Defaults match the T0616 marginal-MSE weighted_sum branch:
  joint_loss=correct_conf_ce
  weighted_sum_marginal_mse_weight=1.0
  weighted_sum_aux_only=1
  best_metric=weighted_sum_marginal_mse
  target_temperature=0.25
  pair_loss_normalization=none
  target_empty_fallback=uniform
  freeze_bert
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }
temp_tag() { printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'; }

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task=$1" >&2; exit 2 ;;
    esac
}

task_dataset() {
    case "$1" in
        boolq) echo SuperGLUE_BoolQ_gen ;;
        rte) echo SuperGLUE_RTE_gen ;;
        siqa) echo siqa_gen ;;
        piqa) echo piqa_gen ;;
        openbookqa) echo obqa_main_gen ;;
        arc_c) echo ARC_c_gen ;;
        *) echo "unsupported task=$1" >&2; exit 2 ;;
    esac
}

mrs_router() { echo "$ROUTER_DIR/mrs_only_two_layer"; }
plus_router() { echo "$ROUTER_DIR/mrs_plus_${1}_two_layer"; }
new_router() { echo "$ROUTER_DIR/new_only_${1}_from_mrs_two_layer"; }

setup() {
    if [[ -e "$OUT_ROOT" && "$RESUME" != "1" ]]; then
        echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
        echo "Set RESUME=1 to continue, or use a fresh OUT_ROOT." >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OC_OFFICIAL_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

summary_exists() {
    local work_dir="$1"
    [[ -d "$work_dir" ]] || return 1
    find "$work_dir" -path '*/summary/*.md' -type f -print -quit | grep -q .
}

record_and_summary_exist() {
    local record_path="$1" work_dir="$2"
    [[ -s "$record_path" ]] && summary_exists "$work_dir"
}

move_partial_eval_outputs() {
    local work_dir="$1" record_path="$2" log_file="$3"
    local backup_dir="$OUT_ROOT/resume_incomplete_${RUN_STAMP}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    log "[WARN] incomplete eval found; moving partial outputs to $backup_dir"
    [[ -e "$work_dir" ]] && mv "$work_dir" "$backup_dir/"
    [[ -n "$record_path" && -e "$record_path" ]] && mv "$record_path" "$backup_dir/"
    [[ -e "$log_file" ]] && mv "$log_file" "$backup_dir/"
}

require_cache() {
    [[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || {
        echo "missing cache: $CACHE_ROOT" >&2
        exit 1
    }
}

require_router() {
    [[ -f "$1/router_heads.pt" && -f "$1/router_config.json" ]] || {
        echo "missing router: $1" >&2
        exit 1
    }
}

router_is_loadable() {
    local router="$1"
    [[ -f "$router/router_heads.pt" && -f "$router/router_config.json" && -f "$router/best_metrics.json" ]] || return 1
    "$PY" - "$router/router_heads.pt" >/dev/null 2>&1 <<'PY'
import sys
import torch

torch.load(sys.argv[1], map_location="cpu")
PY
}

build_cache() {
    local log_file="$LOG_DIR/cache_${RUN_STAMP}.log"
    if [[ "$RESUME" == "1" && -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]]; then
        log "[SKIP] cache existing_cache=$CACHE_ROOT"
        return 0
    fi
    [[ ! -e "$CACHE_ROOT" && ! -e "$log_file" ]] || {
        echo "cache output exists: $CACHE_ROOT $log_file" >&2
        exit 1
    }

    log "[START] build Qwen3.5 two-layer cache cache=$CACHE_ROOT"
    "$PY" -u build_cached_router_pair_dataset.py \
        --data_root "$DATA_ROOT" \
        --feature_root "$CACHE_ROOT" \
        --task_names boolq,medmcqa,openbookqa,arc_c,piqa,race,rte,siqa,sst2 \
        --expert_names "$EXPERTS" \
        --base_model_path "$BASE_MODEL" \
        --router_bert_init "$BERT" \
        --batch_size "$CACHE_BATCH_SIZE" \
        --max_train_samples "$MAX_TRAIN_SAMPLES" \
        --max_val_samples "$MAX_VAL_SAMPLES" \
        --first_layer_idx 0 \
        --middle_layer_idx "$MIDDLE_LAYER_IDX" \
        --router_dim 512 \
        --dtype float16 \
        --score_mode official_eval_aligned_generation \
        --cache_prompt_template chat_template \
        --chunk_size 2048 \
        --seed 42 \
        --lora_medmcqa "$LORA_ROOT/sft_medmcqa" \
        --lora_race "$LORA_ROOT/sft_race" \
        --lora_sst2 "$LORA_ROOT/sft_sst2" \
        > "$log_file" 2>&1
    log "[DONE] cache=$CACHE_ROOT log=$log_file"
}

train_one() {
    local label="$1" sample_tasks="$2" out_dir="$3" load_from="${4:-}"
    local log_file="$LOG_DIR/train_${label}_${RUN_STAMP}.log"
    local -a load_args=()
    local -a objective_args=(
        --weighted_sum_marginal_mse_weight "$MSE_WEIGHT"
        --best_metric "$BEST_METRIC"
    )
    require_cache
    if [[ "$RESUME" == "1" && -e "$out_dir" ]]; then
        if router_is_loadable "$out_dir"; then
            log "[SKIP] train label=$label router=$out_dir"
            return 0
        fi
        local corrupt_dir="${out_dir}.corrupt_${RUN_STAMP}"
        log "[WARN] corrupt/incomplete router found; moving $out_dir -> $corrupt_dir"
        mv "$out_dir" "$corrupt_dir"
    fi
    if [[ "$RESUME" == "1" && -e "$log_file" ]]; then
        mv "$log_file" "${log_file}.partial_${RUN_STAMP}_$(date +%Y%m%d_%H%M%S)"
    fi
    [[ ! -e "$out_dir" && ! -e "$log_file" ]] || {
        echo "train output exists: $out_dir $log_file" >&2
        exit 1
    }
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

    log "[START] train label=$label sample_tasks=$sample_tasks load_from=${load_from:-none}"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        --out_dir "$out_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        "${load_args[@]}" \
        --router_dim 512 \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature "$TARGET_TEMPERATURE" \
        --pair_loss_normalization "$PAIR_LOSS_NORMALIZATION" \
        --pair_constraint "$PAIR_CONSTRAINT" \
        --target_empty_fallback "$TARGET_EMPTY_FALLBACK" \
        --target_distribution_policy "$TARGET_DISTRIBUTION_POLICY" \
        "${objective_args[@]}" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1
    log "[DONE] train label=$label router=$out_dir log=$log_file"
}

train_all() {
    local task
    train_one mrs_only "$EXPERTS" "$(mrs_router)"
    if [[ "$INCLUDE_MRS_PLUS" == "1" ]]; then
        for task in "${TASK_LIST[@]}"; do
            train_one "mrs_plus_$task" "$EXPERTS,$task" "$(plus_router "$task")"
        done
    fi
    if [[ "$INCLUDE_NEW_ONLY" == "1" ]]; then
        for task in "${TASK_LIST[@]}"; do
            train_one "new_only_$task" "$task" "$(new_router "$task")" "$(mrs_router)"
        done
    fi
}

eval_one() {
    local label="$1" router="$2" mode="$3" sharpness="$4"
    shift 4
    local tag work_dir record log_file
    require_router "$router"
    tag="$(temp_tag "$sharpness")"
    work_dir="$OC_OFFICIAL_DIR/${label}_${mode}_sharp${tag}_${RUN_STAMP}"
    record="$RECORD_DIR/${label}_${mode}_sharp${tag}_${RUN_STAMP}.jsonl"
    log_file="$LOG_DIR/eval_${label}_${mode}_sharp${tag}_${RUN_STAMP}.log"
    if [[ "$RESUME" == "1" ]] && record_and_summary_exist "$record" "$work_dir"; then
        log "[SKIP] eval label=$label mode=$mode sharpness=$sharpness work_dir=$work_dir"
        return 0
    fi
    if [[ "$RESUME" == "1" && ( -e "$work_dir" || -e "$record" || -e "$log_file" ) ]]; then
        move_partial_eval_outputs "$work_dir" "$record" "$log_file"
    fi
    [[ ! -e "$work_dir" && ! -e "$record" && ! -e "$log_file" ]] || {
        echo "eval output exists: $work_dir $record $log_file" >&2
        exit 1
    }

    log "[START] eval label=$label mode=$mode sharpness=$sharpness datasets=$*"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$mode" \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_PAIR_CONSTRAINT="$PAIR_CONSTRAINT" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_${label}_${mode}_sharp${tag}" \
    T0601_ROUTER_RECORD_PATH="$record" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "$@" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] eval label=$label mode=$mode work_dir=$work_dir log=$log_file"
}

eval_base_model() {
    local work_dir="$OC_OFFICIAL_DIR/base_model_${RUN_STAMP}"
    local log_file="$LOG_DIR/eval_base_model_${RUN_STAMP}.log"
    if [[ "$RESUME" == "1" ]] && summary_exists "$work_dir"; then
        log "[SKIP] eval base_model work_dir=$work_dir"
        return 0
    fi
    if [[ "$RESUME" == "1" && ( -e "$work_dir" || -e "$log_file" ) ]]; then
        move_partial_eval_outputs "$work_dir" "" "$log_file"
    fi
    [[ ! -e "$work_dir" && ! -e "$log_file" ]] || {
        echo "base eval output exists: $work_dir $log_file" >&2
        exit 1
    }
    log "[START] eval base_model datasets=${OFFICIAL_ALL_DATASETS[*]}"
    "$PY" -u run.py \
        --models "$BASE_MODEL_CONFIG" \
        --datasets "${OFFICIAL_ALL_DATASETS[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] eval base_model work_dir=$work_dir log=$log_file"
}

eval_all() {
    local sharpness task dataset
    split_csv "$SHARPNESS_LIST"
    local -a sharps=("${SPLIT_RESULT[@]}")
    for sharpness in "${sharps[@]}"; do
        eval_one mrs_only "$(mrs_router)" weighted_sum "$sharpness" "${OFFICIAL_ALL_DATASETS[@]}"
        if [[ "$INCLUDE_MRS_PLUS" == "1" ]]; then
            for task in "${TASK_LIST[@]}"; do
                dataset="$(task_dataset "$task")"
                eval_one "mrs_plus_$task" "$(plus_router "$task")" weighted_sum "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
            done
        fi
        if [[ "$INCLUDE_NEW_ONLY" == "1" ]]; then
            for task in "${TASK_LIST[@]}"; do
                dataset="$(task_dataset "$task")"
                eval_one "new_only_$task" "$(new_router "$task")" weighted_sum "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
            done
        fi
    done
}

eval_baselines() {
    local sharpness
    [[ "$EVAL_BASELINES" == "1" ]] || return 0
    eval_base_model
    split_csv "$SHARPNESS_LIST"
    local -a sharps=("${SPLIT_RESULT[@]}")
    for sharpness in "${sharps[@]}"; do
        eval_one uniform "$(mrs_router)" uniform "$sharpness" "${OFFICIAL_ALL_DATASETS[@]}"
    done
}

main() {
    local mode="${1:-all}" task
    case "$mode" in
        -h|--help|help) usage; exit 0 ;;
        all|cache|train|eval|baseline) ;;
        *) usage; exit 2 ;;
    esac
    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_RESULT[@]}")
    for task in "${TASK_LIST[@]}"; do validate_task "$task"; done

    setup
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT base_model=$BASE_MODEL lora_root=$LORA_ROOT middle_layer_idx=$MIDDLE_LAYER_IDX"
    log "[INFO] objective=two_layer weighted_sum_marginal_mse_weight=$MSE_WEIGHT weighted_sum_aux_only=$WEIGHTED_SUM_AUX_ONLY best_metric=$BEST_METRIC target_temperature=$TARGET_TEMPERATURE pair_loss_normalization=$PAIR_LOSS_NORMALIZATION pair_constraint=$PAIR_CONSTRAINT"
    log "[INFO] tasks=${TASK_LIST[*]} experts=$EXPERTS sharpness=$SHARPNESS_LIST include_mrs_plus=$INCLUDE_MRS_PLUS include_new_only=$INCLUDE_NEW_ONLY eval_baselines=$EVAL_BASELINES"
    [[ -n "$MAX_OUT_LEN" ]] && log "[INFO] max_out_len_override=$MAX_OUT_LEN"

    [[ "$mode" == all || "$mode" == cache ]] && build_cache
    [[ "$mode" == all || "$mode" == train ]] && train_all
    [[ "$mode" == all || "$mode" == eval ]] && eval_all
    [[ "$mode" == all || "$mode" == baseline ]] && eval_baselines
    log "[DONE] output=$OUT_ROOT"
}

main "$@"
