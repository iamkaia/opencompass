#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0701_llama2_wsum_correct_conf_raw_t0p25_freezebert_from_t0602_cache_${RUN_STAMP}}"
RUN_LABEL="${RUN_LABEL:-T0701}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./t0602_taskcls_trainbert_chattemplate_20260603_02}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/llama_0602_9task_3expert_official_eval_aligned_chattemplate_800_200}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
SHARPNESS="${SHARPNESS:-3.0}"
RESUME="${RESUME:-0}"
SPLIT_OFFICIAL_DATASETS="${SPLIT_OFFICIAL_DATASETS:-0}"

BERT="${BERT:-./task_classifier_ckpt}"
ROUTER_MODEL_CONFIG="${ROUTER_MODEL_CONFIG:-T0531_mrs_ablation_hard_routing.py}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-hf_llama2_7b_chat.py}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
BEST_METRIC="${BEST_METRIC:-weighted_sum_marginal_mse}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.25}"
PAIR_LOSS_NORMALIZATION="${PAIR_LOSS_NORMALIZATION:-none}"
TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"

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
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0701_train_llama2_wsum_raw_t0p25_freezebert_from_t0602_cache.sh \
    > T0701_llama2_wsum_t0p25_from_t0602_cache.nohup.log 2>&1 &

Runs a Llama-2-7b-chat version of the T0616 weighted_sum/MSE setup using the
existing T0602 Llama cache:
  ./t0602_taskcls_trainbert_chattemplate_20260603_02/caches/llama_0602_9task_3expert_official_eval_aligned_chattemplate_800_200

Training target/loss:
  joint_loss=correct_conf_ce
  correct_soft_ce_temperature=0.25
  pair_loss_normalization=none
  target_empty_fallback=uniform
  target_distribution_policy=cache_oracle
  weighted_sum_marginal_mse_weight=1.0
  weighted_sum_aux_only
  best_metric=weighted_sum_marginal_mse
  freeze_bert

Pipeline:
  train mrs_only, mrs_plus_<task>, and new_only_<task>
  eval weighted_sum sharp3 on all trained routers
  eval uniform baseline on all official datasets
  eval base_model baseline on all official datasets

Modes:
  all   train + eval
  train train only
  eval  eval only, using existing routers in OUT_ROOT

Resume:
  RESUME=1 OUT_ROOT=./existing_root RUN_STAMP=<same_stamp> bash tools/T0701_train_llama2_wsum_raw_t0p25_freezebert_from_t0602_cache.sh

If one multi-dataset OpenCompass eval keeps dying midway, resume with:
  SPLIT_OFFICIAL_DATASETS=1 RESUME=1 OUT_ROOT=./existing_root RUN_STAMP=<same_stamp> bash tools/T0701_train_llama2_wsum_raw_t0p25_freezebert_from_t0602_cache.sh
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

temp_tag() {
    printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

dataset_tag() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9_]/_/g'
}

setup_root() {
    if [[ -e "$OUT_ROOT" ]]; then
        if [[ "$RESUME" != "1" ]]; then
            echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
            echo "Set RESUME=1 to continue an interrupted run." >&2
            exit 1
        fi
    else
        mkdir -p "$OUT_ROOT"
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

router_is_loadable() {
    local router="$1"
    [[ -f "$router/router_heads.pt" && -f "$router/router_config.json" && -f "$router/best_metrics.json" ]] || return 1
    "$PY" - "$router/router_heads.pt" >/dev/null 2>&1 <<'PY'
import sys
import torch

torch.load(sys.argv[1], map_location="cpu")
PY
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
    echo "$ROUTER_DIR/router_${RUN_LABEL}_llama2_mrs_only_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert"
}

mrs_plus_router() {
    local task="$1"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_llama2_mrs_plus_${task}_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert"
}

new_only_router() {
    local task="$1"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_llama2_new_only_${task}_from_mrs_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert"
}

move_partial_eval_outputs() {
    local work_dir="$1" record_path="$2" log_file="$3"
    local backup_dir="$OUT_ROOT/resume_incomplete_${RUN_STAMP}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    log "[WARN] incomplete eval found; moving partial outputs to $backup_dir"
    [[ -e "$work_dir" ]] && mv "$work_dir" "$backup_dir/"
    [[ -e "$record_path" ]] && mv "$record_path" "$backup_dir/"
    [[ -e "$log_file" ]] && mv "$log_file" "$backup_dir/"
}

train_router() {
    local label="$1" sample_tasks="$2" out_dir="$3" load_from="${4:-}" log_file
    log_file="$LOG_DIR/${RUN_LABEL}_train_${label}_${RUN_STAMP}.log"

    if [[ "$RESUME" == "1" && -e "$out_dir" ]]; then
        if router_is_loadable "$out_dir"; then
            log "[SKIP] train label=$label existing_router=$out_dir"
            return 0
        fi
        local corrupt_dir="${out_dir}.corrupt_${RUN_STAMP}"
        log "[WARN] corrupt/incomplete router found; moving $out_dir -> $corrupt_dir"
        mv "$out_dir" "$corrupt_dir"
    fi

    if [[ "$RESUME" == "1" && -e "$log_file" ]]; then
        local old_log="${log_file}.partial_${RUN_STAMP}_$(date +%Y%m%d_%H%M%S)"
        log "[WARN] moving partial train log $log_file -> $old_log"
        mv "$log_file" "$old_log"
    fi

    [[ ! -e "$out_dir" && ! -e "$log_file" ]] || {
        echo "train output exists: $out_dir $log_file" >&2
        echo "If this is an interrupted run, set RESUME=1 or use a fresh OUT_ROOT." >&2
        exit 1
    }

    local -a load_args=()
    if [[ -n "$load_from" ]]; then
        require_router "$load_from"
        load_args+=(--load_from "$load_from")
    fi

    log "[START] train label=$label sample_tasks=$sample_tasks load_from=${load_from:-none}"
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
        --weighted_sum_marginal_mse_weight "$MSE_WEIGHT" \
        --weighted_sum_aux_only \
        --best_metric "$BEST_METRIC" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1
    log "[DONE] train label=$label out=$out_dir log=$log_file"
}

eval_router_official() {
    local label="$1" router="$2" routing_mode="$3" sharpness="$4"
    shift 4
    local sharp_tag log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router"
    sharp_tag="$(temp_tag "$sharpness")"
    log_file="$LOG_DIR/${RUN_LABEL}_official_${label}_${routing_mode}_sharp${sharp_tag}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_${routing_mode}_sharp${sharp_tag}_${RUN_STAMP}"
    record_path="$RECORD_DIR/${RUN_LABEL}_official_${label}_${routing_mode}_sharp${sharp_tag}_${RUN_STAMP}.jsonl"

    if [[ "$SPLIT_OFFICIAL_DATASETS" == "1" ]]; then
        if record_and_summary_exist "$record_path" "$work_dir"; then
            log "[SKIP] official label=$label routing_mode=$routing_mode sharpness=$sharpness existing_combined_work_dir=$work_dir"
            return 0
        fi
        local dataset
        for dataset in "${datasets[@]}"; do
            eval_router_official_dataset "$label" "$router" "$routing_mode" "$sharpness" "$dataset"
        done
        return 0
    elif [[ "$SPLIT_OFFICIAL_DATASETS" != "0" ]]; then
        echo "SPLIT_OFFICIAL_DATASETS must be 0 or 1, got: $SPLIT_OFFICIAL_DATASETS" >&2
        exit 2
    fi

    if [[ "$RESUME" == "1" ]] && record_and_summary_exist "$record_path" "$work_dir"; then
        log "[SKIP] official label=$label routing_mode=$routing_mode sharpness=$sharpness existing_work_dir=$work_dir"
        return 0
    fi
    if [[ "$RESUME" == "1" && ( -e "$work_dir" || -e "$record_path" || -e "$log_file" ) ]]; then
        move_partial_eval_outputs "$work_dir" "$record_path" "$log_file"
    fi
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        echo "Use RESUME=1 for partial eval collisions, or use a fresh OUT_ROOT." >&2
        exit 1
    }

    log "[START] official label=$label routing_mode=$routing_mode sharpness=$sharpness datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$routing_mode" \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_official_${label}_${routing_mode}_sharp${sharp_tag}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$ROUTER_MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official label=$label routing_mode=$routing_mode sharpness=$sharpness log=$log_file"
}

eval_router_official_dataset() {
    local label="$1" router="$2" routing_mode="$3" sharpness="$4" dataset="$5"
    local sharp_tag dataset_tag_value log_file work_dir record_path
    require_router "$router"
    sharp_tag="$(temp_tag "$sharpness")"
    dataset_tag_value="$(dataset_tag "$dataset")"
    log_file="$LOG_DIR/${RUN_LABEL}_official_${label}_${routing_mode}_sharp${sharp_tag}_${dataset_tag_value}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_${routing_mode}_sharp${sharp_tag}_${dataset_tag_value}_${RUN_STAMP}"
    record_path="$RECORD_DIR/${RUN_LABEL}_official_${label}_${routing_mode}_sharp${sharp_tag}_${dataset_tag_value}_${RUN_STAMP}.jsonl"

    if [[ "$RESUME" == "1" ]] && record_and_summary_exist "$record_path" "$work_dir"; then
        log "[SKIP] official dataset label=$label routing_mode=$routing_mode sharpness=$sharpness dataset=$dataset existing_work_dir=$work_dir"
        return 0
    fi
    if [[ "$RESUME" == "1" && ( -e "$work_dir" || -e "$record_path" || -e "$log_file" ) ]]; then
        move_partial_eval_outputs "$work_dir" "$record_path" "$log_file"
    fi
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        echo "Use RESUME=1 for partial eval collisions, or use a fresh OUT_ROOT." >&2
        exit 1
    }

    log "[START] official dataset label=$label routing_mode=$routing_mode sharpness=$sharpness dataset=$dataset"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$routing_mode" \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_official_${label}_${routing_mode}_sharp${sharp_tag}_${dataset_tag_value}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$ROUTER_MODEL_CONFIG" \
        --datasets "$dataset" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official dataset label=$label routing_mode=$routing_mode sharpness=$sharpness dataset=$dataset log=$log_file"
}

eval_base_model_official() {
    local label="base_model" log_file work_dir
    log_file="$LOG_DIR/${RUN_LABEL}_official_${label}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_${RUN_STAMP}"

    if [[ "$RESUME" == "1" ]] && summary_exists "$work_dir"; then
        log "[SKIP] official label=$label existing_work_dir=$work_dir"
        return 0
    fi
    if [[ "$RESUME" == "1" && ( -e "$work_dir" || -e "$log_file" ) ]]; then
        move_partial_eval_outputs "$work_dir" "/tmp/nonexistent_${RUN_LABEL}_${RUN_STAMP}_base_model_record" "$log_file"
    fi
    [[ ! -e "$work_dir" && ! -e "$log_file" ]] || {
        echo "base_model eval output exists: $work_dir $log_file" >&2
        echo "Use RESUME=1 for partial eval collisions, or use a fresh OUT_ROOT." >&2
        exit 1
    }

    log "[START] official label=$label datasets=${OFFICIAL_ALL_DATASETS[*]}"
    "$PY" -u run.py \
        --models "$BASE_MODEL_CONFIG" \
        --datasets "${OFFICIAL_ALL_DATASETS[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official label=$label log=$log_file"
}

train_all() {
    local task base_router
    base_router="$(mrs_only_router)"
    train_router "mrs_only" "$EXPERTS" "$base_router"

    for task in "${TASK_LIST[@]}"; do
        train_router "mrs_plus_${task}" "$EXPERTS,$task" "$(mrs_plus_router "$task")"
    done

    for task in "${TASK_LIST[@]}"; do
        train_router "new_only_${task}" "$task" "$(new_only_router "$task")" "$base_router"
    done
}

eval_all_official() {
    local task dataset
    eval_router_official "mrs_only" "$(mrs_only_router)" weighted_sum "$SHARPNESS" "${OFFICIAL_ALL_DATASETS[@]}"
    eval_router_official "uniform" "$(mrs_only_router)" uniform 1.0 "${OFFICIAL_ALL_DATASETS[@]}"
    eval_base_model_official

    for task in "${TASK_LIST[@]}"; do
        dataset="$(task_dataset_for "$task")"
        eval_router_official "mrs_plus_${task}" "$(mrs_plus_router "$task")" weighted_sum "$SHARPNESS" "$dataset" "${MRS_DATASETS[@]}"
    done

    for task in "${TASK_LIST[@]}"; do
        dataset="$(task_dataset_for "$task")"
        eval_router_official "new_only_${task}" "$(new_only_router "$task")" weighted_sum "$SHARPNESS" "$dataset" "${MRS_DATASETS[@]}"
    done
}

main() {
    local mode="${1:-all}" task
    case "$mode" in
        -h|--help|help) usage; exit 0 ;;
        all|train|eval) ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_CSV_RESULT[@]}")
    for task in "${TASK_LIST[@]}"; do
        validate_task "$task"
    done

    setup_root
    require_inputs
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT"
    log "[INFO] router_model_config=$ROUTER_MODEL_CONFIG base_model_config=$BASE_MODEL_CONFIG"
    log "[INFO] tasks=${TASK_LIST[*]} experts=$EXPERTS cuda=$CUDA_VISIBLE_DEVICES"
    log "[INFO] target_temperature=$TARGET_TEMPERATURE official_sharpness=$SHARPNESS routing_topk=disabled"
    log "[INFO] train_loss=correct_conf_ce weighted_sum_aux_only=1 weighted_sum_marginal_mse_weight=$MSE_WEIGHT best_metric=$BEST_METRIC freeze_bert=1"
    log "[INFO] baselines=uniform,base_model resume=$RESUME split_official_datasets=$SPLIT_OFFICIAL_DATASETS"

    [[ "$mode" == "all" || "$mode" == "train" ]] && train_all
    [[ "$mode" == "all" || "$mode" == "eval" ]] && eval_all_official

    log "[DONE] llama2 weighted_sum raw t0.25 freezebert pipeline output=$OUT_ROOT"
}

main "$@"
