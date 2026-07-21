#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0629_t0616_target_temp_sharp57_newonly_4sum_${RUN_STAMP}}"
RUN_LABEL="${RUN_LABEL:-T0629}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
TARGET_TEMPERATURES="${TARGET_TEMPERATURES:-0.25,0.1}"
SHARPNESS_LIST="${SHARPNESS_LIST:-5.0,7.0}"
RESUME="${RESUME:-0}"

BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
BEST_METRIC="${BEST_METRIC:-weighted_sum_marginal_mse}"
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
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0629_target_temp_sharp57_newonly_4sum_qwen.sh > T0629_target_temp_sharp57_newonly_4sum.nohup.log 2>&1 &

Runs the T0616 two-layer weighted_sum ablation:
  target_temperature=0.25 with routing_sharpness=5,7
  target_temperature=0.1  with routing_sharpness=5,7

No topk routing is used. The "4sum" setting is mrs_plus_<task>:
  sample_task_names=medmcqa,race,sst2,<task>

Outputs are written under OUT_ROOT.

Resume an interrupted run without deleting partial outputs:
  RESUME=1 OUT_ROOT=./T0629_t0616_target_temp_sharp57_newonly_4sum_20260629_045338 \
    CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0629_target_temp_sharp57_newonly_4sum_qwen.sh \
    > T0629_target_temp_sharp57_newonly_4sum.resume.nohup.log 2>&1 &
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
    local temp="$1" temp_tag_value
    temp_tag_value="$(temp_tag "$temp")"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_mrs_only_t${temp_tag_value}_freezebert_wsum_correctconf_emptyuniform_3expert_sst2words"
}

four_sum_router() {
    local temp="$1" task="$2" temp_tag_value
    temp_tag_value="$(temp_tag "$temp")"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_4sum_${task}_t${temp_tag_value}_freezebert_wsum_correctconf_emptyuniform_3expert_sst2words"
}

new_only_router() {
    local temp="$1" task="$2" temp_tag_value
    temp_tag_value="$(temp_tag "$temp")"
    echo "$ROUTER_DIR/router_${RUN_LABEL}_qwen3_fp16_new_only_${task}_from_mrs_t${temp_tag_value}_freezebert_wsum_correctconf_emptyuniform_3expert_sst2words"
}

train_router() {
    local temp="$1" label="$2" sample_tasks="$3" out_dir="$4" load_from="${5:-}"
    local temp_tag_value log_file
    temp_tag_value="$(temp_tag "$temp")"
    log_file="$LOG_DIR/${RUN_LABEL}_train_${label}_t${temp_tag_value}_${RUN_STAMP}.log"
    if [[ "$RESUME" == "1" && -e "$out_dir" ]]; then
        if router_is_loadable "$out_dir"; then
            log "[SKIP] train temp=$temp label=$label existing_router=$out_dir"
            return 0
        fi
        local corrupt_dir="${out_dir}.corrupt_${RUN_STAMP}"
        log "[WARN] corrupt/incomplete router found; moving $out_dir -> $corrupt_dir"
        mv "$out_dir" "$corrupt_dir"
    fi
    [[ ! -e "$out_dir" && ! -e "$log_file" ]] || {
        echo "train output exists: $out_dir $log_file" >&2
        echo "If this is an interrupted run, remove only the incomplete target directory/log or use a fresh OUT_ROOT." >&2
        exit 1
    }

    local -a load_args=()
    if [[ -n "$load_from" ]]; then
        require_router "$load_from"
        load_args+=(--load_from "$load_from")
    fi

    log "[START] train temp=$temp label=$label sample_tasks=$sample_tasks load_from=${load_from:-none}"
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
        --correct_soft_ce_temperature "$temp" \
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
    log "[DONE] train temp=$temp label=$label out=$out_dir log=$log_file"
}

eval_official() {
    local temp="$1" label="$2" router="$3" sharpness="$4"
    shift 4
    local temp_tag_value sharp_tag log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router"
    temp_tag_value="$(temp_tag "$temp")"
    sharp_tag="$(temp_tag "$sharpness")"
    log_file="$LOG_DIR/${RUN_LABEL}_official_${label}_t${temp_tag_value}_sharp${sharp_tag}_${RUN_STAMP}.log"
    work_dir="$OC_OFFICIAL_DIR/${label}_t${temp_tag_value}_sharp${sharp_tag}_${RUN_STAMP}"
    record_path="$RECORD_DIR/${RUN_LABEL}_official_${label}_t${temp_tag_value}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}.jsonl"
    if [[ "$RESUME" == "1" && -s "$record_path" ]] && find "$work_dir" -path '*/summary/*.md' -type f -print -quit | grep -q .; then
        log "[SKIP] official temp=$temp label=$label sharpness=$sharpness existing_work_dir=$work_dir"
        return 0
    fi
    if [[ "$RESUME" == "1" && ( -e "$work_dir" || -e "$record_path" || -e "$log_file" ) ]]; then
        local backup_dir="$OUT_ROOT/resume_incomplete_${RUN_STAMP}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        log "[WARN] incomplete official eval found; moving partial outputs to $backup_dir"
        [[ -e "$work_dir" ]] && mv "$work_dir" "$backup_dir/"
        [[ -e "$record_path" ]] && mv "$record_path" "$backup_dir/"
        [[ -e "$log_file" ]] && mv "$log_file" "$backup_dir/"
    fi
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "official eval output exists: $work_dir $record_path $log_file" >&2
        echo "Use a fresh RUN_STAMP/OUT_ROOT for partial eval collisions." >&2
        exit 1
    }

    log "[START] official temp=$temp label=$label sharpness=$sharpness datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTER_RECORD_TAG="${RUN_LABEL}_official_${label}_t${temp_tag_value}_sharp${sharp_tag}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] official temp=$temp label=$label sharpness=$sharpness log=$log_file"
}

train_all() {
    local temp task base_router
    for temp in "${TEMP_LIST[@]}"; do
        base_router="$(mrs_only_router "$temp")"
        train_router "$temp" "mrs_only" "$EXPERTS" "$base_router"

        for task in "${TASK_LIST[@]}"; do
            train_router "$temp" "4sum_${task}" "$EXPERTS,$task" "$(four_sum_router "$temp" "$task")"
        done

        for task in "${TASK_LIST[@]}"; do
            train_router "$temp" "new_only_${task}" "$task" "$(new_only_router "$temp" "$task")" "$base_router"
        done
    done
}

eval_all_official() {
    local temp sharpness task dataset
    for temp in "${TEMP_LIST[@]}"; do
        for sharpness in "${SHARP_LIST[@]}"; do
            eval_official "$temp" "mrs_only" "$(mrs_only_router "$temp")" "$sharpness" "${OFFICIAL_ALL_DATASETS[@]}"
            for task in "${TASK_LIST[@]}"; do
                dataset="$(task_dataset_for "$task")"
                eval_official "$temp" "4sum_${task}" "$(four_sum_router "$temp" "$task")" "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
            done
            for task in "${TASK_LIST[@]}"; do
                dataset="$(task_dataset_for "$task")"
                eval_official "$temp" "new_only_${task}" "$(new_only_router "$temp" "$task")" "$sharpness" "$dataset" "${MRS_DATASETS[@]}"
            done
        done
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
    split_csv "$TARGET_TEMPERATURES"
    TEMP_LIST=("${SPLIT_CSV_RESULT[@]}")
    split_csv "$SHARPNESS_LIST"
    SHARP_LIST=("${SPLIT_CSV_RESULT[@]}")
    for task in "${TASK_LIST[@]}"; do
        validate_task "$task"
    done

    setup_root
    require_inputs
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT"
    log "[INFO] tasks=${TASK_LIST[*]} experts=$EXPERTS cuda=$CUDA_VISIBLE_DEVICES"
    log "[INFO] target_temperatures=${TEMP_LIST[*]} sharpness=${SHARP_LIST[*]} routing_topk=disabled"
    log "[INFO] train_loss=correct_conf_ce weighted_sum_aux_only=1 weighted_sum_marginal_mse_weight=$MSE_WEIGHT best_metric=$BEST_METRIC freeze_bert=1"
    log "[INFO] eval_scope=official mrs_only_4sum_new_only"
    log "[INFO] resume=$RESUME"

    [[ "$mode" == "all" || "$mode" == "train" ]] && train_all
    [[ "$mode" == "all" || "$mode" == "eval" ]] && eval_all_official

    log "[DONE] target temperature x sharpness ablation output=$OUT_ROOT"
}

main "$@"
