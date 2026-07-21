#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TWO_LAYER_ROOT="${TWO_LAYER_ROOT:-./T0616_wsum_correct_conf_raw_t0p25_freezebert_full_20260616_061945}"
SINGLE_ROOT="${SINGLE_ROOT:-./T0622_single_all_layers_wsum_run1}"
TWO_LAYER_RUN_LABEL="${TWO_LAYER_RUN_LABEL:-T0616}"
SINGLE_RUN_LABEL="${SINGLE_RUN_LABEL:-T0622_single}"
EVAL_TAG="${EVAL_TAG:-top2_official_only_${RUN_STAMP}}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
SHARPNESS_LIST="${SHARPNESS_LIST:-2.0,3.0}"
ROUTING_TOPK="${ROUTING_TOPK:-2}"
FAMILIES="${FAMILIES:-two_layer,single_all_layers}"

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
    cat <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash ./tools/T0624_eval_two_layer_vs_single_top2_official_qwen.sh > ./T0624_eval_two_layer_vs_single_top2_official.nohup.log 2>&1 &

Eval only. No training, no train_split, no new_only.

Runs official OpenCompass weighted_sum eval for:
  - two-layer MSE-only routers under TWO_LAYER_ROOT
  - single_all_layers MSE-only routers under SINGLE_ROOT

Defaults:
  TWO_LAYER_ROOT=./T0616_wsum_correct_conf_raw_t0p25_freezebert_full_20260616_061945
  SINGLE_ROOT=./T0622_single_all_layers_wsum_run1
  SHARPNESS_LIST=2.0,3.0
  ROUTING_TOPK=2
  FAMILIES=two_layer,single_all_layers

Outputs:
  $TWO_LAYER_ROOT/logs/*_${EVAL_TAG}_*.log
  $TWO_LAYER_ROOT/opencompass_official/*_${EVAL_TAG}/
  $TWO_LAYER_ROOT/router_records/*_${EVAL_TAG}_*.jsonl
  $SINGLE_ROOT/logs/*_${EVAL_TAG}_*.log
  $SINGLE_ROOT/opencompass_official/*_${EVAL_TAG}/
  $SINGLE_ROOT/router_records/*_${EVAL_TAG}_*.jsonl
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

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

validate_family() {
    case "$1" in
        two_layer|single_all_layers) ;;
        *) echo "unsupported family: $1" >&2; exit 2 ;;
    esac
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

require_router() {
    local router="$1"
    [[ -f "$router/router_heads.pt" && -f "$router/router_config.json" ]] || {
        echo "missing router checkpoint: $router" >&2
        exit 1
    }
}

two_mrs_only_router() {
    echo "$TWO_LAYER_ROOT/routers/router_${TWO_LAYER_RUN_LABEL}_qwen3_fp16_mrs_only_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

two_mrs_plus_router() {
    local task="$1"
    echo "$TWO_LAYER_ROOT/routers/router_${TWO_LAYER_RUN_LABEL}_qwen3_fp16_mrs_plus_${task}_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words"
}

single_mrs_only_router() {
    echo "$SINGLE_ROOT/routers/mrs_only_single_all_layers"
}

single_mrs_plus_router() {
    local task="$1"
    echo "$SINGLE_ROOT/routers/mrs_plus_${task}_single_all_layers"
}

setup_roots() {
    [[ -d "$TWO_LAYER_ROOT" ]] || { echo "missing TWO_LAYER_ROOT: $TWO_LAYER_ROOT" >&2; exit 1; }
    [[ -d "$SINGLE_ROOT" ]] || { echo "missing SINGLE_ROOT: $SINGLE_ROOT" >&2; exit 1; }
    mkdir -p "$TWO_LAYER_ROOT/logs" "$TWO_LAYER_ROOT/opencompass_official" "$TWO_LAYER_ROOT/router_records"
    mkdir -p "$SINGLE_ROOT/logs" "$SINGLE_ROOT/opencompass_official" "$SINGLE_ROOT/router_records"
}

eval_official() {
    local family="$1" label="$2" router="$3" sharpness="$4" share_first="$5"
    shift 5
    local root run_label sharp_tag log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router"
    case "$family" in
        two_layer) root="$TWO_LAYER_ROOT"; run_label="$TWO_LAYER_RUN_LABEL" ;;
        single_all_layers) root="$SINGLE_ROOT"; run_label="$SINGLE_RUN_LABEL" ;;
        *) echo "unsupported family: $family" >&2; exit 2 ;;
    esac
    sharp_tag="$(temp_tag "$sharpness")"
    log_file="$root/logs/${run_label}_${EVAL_TAG}_${family}_${label}_topk${ROUTING_TOPK}_sharp${sharp_tag}.log"
    work_dir="$root/opencompass_official/${family}_${label}_weighted_sum_topk${ROUTING_TOPK}_sharp${sharp_tag}_${EVAL_TAG}"
    record_path="$root/router_records/${run_label}_${EVAL_TAG}_${family}_${label}_weighted_sum_topk${ROUTING_TOPK}_sharp${sharp_tag}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }

    log "[START] family=$family label=$label topk=$ROUTING_TOPK sharpness=$sharpness datasets=${datasets[*]}"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_SHARE_FIRST_WEIGHTS_ALL_LAYERS="$share_first" \
    T0531_ROUTER_RECORD_TAG="${run_label}_${EVAL_TAG}_${family}_${label}_topk${ROUTING_TOPK}_sharp${sharp_tag}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] family=$family label=$label sharpness=$sharpness log=$log_file"
}

eval_family() {
    local family="$1" sharpness="$2" task dataset
    case "$family" in
        two_layer)
            eval_official "$family" mrs_only "$(two_mrs_only_router)" "$sharpness" 0 "${OFFICIAL_ALL_DATASETS[@]}"
            for task in "${TASK_LIST[@]}"; do
                dataset="$(task_dataset_for "$task")"
                eval_official "$family" "mrs_plus_${task}" "$(two_mrs_plus_router "$task")" "$sharpness" 0 "$dataset" "${MRS_DATASETS[@]}"
            done
            ;;
        single_all_layers)
            eval_official "$family" mrs_only "$(single_mrs_only_router)" "$sharpness" 1 "${OFFICIAL_ALL_DATASETS[@]}"
            for task in "${TASK_LIST[@]}"; do
                dataset="$(task_dataset_for "$task")"
                eval_official "$family" "mrs_plus_${task}" "$(single_mrs_plus_router "$task")" "$sharpness" 1 "$dataset" "${MRS_DATASETS[@]}"
            done
            ;;
        *) echo "unsupported family: $family" >&2; exit 2 ;;
    esac
}

main() {
    case "${1:-}" in
        -h|--help|help) usage; exit 0 ;;
        "") ;;
        *) usage; exit 2 ;;
    esac

    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_CSV_RESULT[@]}")
    local task sharpness
    for task in "${TASK_LIST[@]}"; do
        validate_task "$task"
    done
    setup_roots
    log "[INFO] eval_tag=$EVAL_TAG two_layer_root=$TWO_LAYER_ROOT single_root=$SINGLE_ROOT"
    split_csv "$FAMILIES"
    FAMILY_LIST=("${SPLIT_CSV_RESULT[@]}")
    local family
    for family in "${FAMILY_LIST[@]}"; do
        validate_family "$family"
    done
    log "[INFO] eval_only=1 train_split=0 new_only=0 routing_mode=weighted_sum routing_topk=$ROUTING_TOPK sharpness_list=$SHARPNESS_LIST families=${FAMILY_LIST[*]}"
    log "[INFO] two_layer source is weighted_sum_aux_only MSE-only; single_all_layers source uses expert_ce_weight=0.0 and weighted_sum_mse_weight=1.0."

    split_csv "$SHARPNESS_LIST"
    local -a sharpness_values=("${SPLIT_CSV_RESULT[@]}")
    for sharpness in "${sharpness_values[@]}"; do
        for family in "${FAMILY_LIST[@]}"; do
            eval_family "$family" "$sharpness"
        done
    done
    log "[DONE] eval_tag=$EVAL_TAG"
}

main "$@"
