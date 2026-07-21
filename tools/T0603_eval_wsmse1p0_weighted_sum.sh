#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXP_ROOT="${EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
EVAL_ROOT="${EVAL_ROOT:-$EXP_ROOT}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WEIGHT_TAG="${WEIGHT_TAG:-1p0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
SHARPNESS="${SHARPNESS:-1.0}"
SHARPNESS_TAG="$(printf '%s' "$SHARPNESS" | sed 's/-/m/g; s/\./p/g')"

LOG_DIR="$EVAL_ROOT/logs"
OC_DIR="$EVAL_ROOT/opencompass"
RECORD_DIR="$EVAL_ROOT/router_records"
ROUTER_DIR="$EXP_ROOT/routers"
BERT="./task_classifier_ckpt"
MODEL_CONFIG="T0531_mrs_ablation_sst2words_hard_routing.py"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
DEFAULT_TASKS="openbookqa,piqa,siqa"

usage() {
    cat >&2 <<'EOF'
usage:
  bash tools/T0603_eval_wsmse1p0_weighted_sum.sh [all|mrs_only|tasks] [task_csv]

Runs OpenCompass weighted_sum eval for T0603 wsmse checkpoints.
MRS-only runs all datasets; mrs_plus tasks run task + MRS datasets.

Environment:
  EXP_ROOT      default ./T0603_qwenfix_trainbert_chattemplate_20260603_063927
  EVAL_ROOT     output root; default EXP_ROOT
  WEIGHT_TAG    default 1p0
  ROUTING_TOPK  optional, e.g. 3
  SHARPNESS     runtime routing sharpness; default 1.0
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

validate_task() {
    case "$1" in boolq|rte|siqa|piqa|openbookqa|arc_c) ;; *) echo "unsupported task: $1" >&2; exit 2 ;; esac
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

router_dir_for() {
    local task="${1:-}"
    if [[ -z "$task" ]]; then
        echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmse${WEIGHT_TAG}_3expert_sst2words"
    else
        echo "$ROUTER_DIR/router_T0603_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmse${WEIGHT_TAG}_3expert_sst2words"
    fi
}

require_router() {
    local router_root="$1"
    [[ -f "$router_root/router_heads.pt" && -f "$router_root/router_config.json" ]] || {
        echo "missing router checkpoint: $router_root" >&2
        exit 1
    }
}

eval_router() {
    local label="$1" router_ckpt="$2"
    shift 2
    local log_file work_dir record_path tag mode_suffix
    local -a datasets=("$@")
    require_router "$router_ckpt"
    mkdir -p "$LOG_DIR" "$OC_DIR" "$RECORD_DIR"
    mode_suffix="weighted_sum_wsmse${WEIGHT_TAG}_sharp${SHARPNESS_TAG}"
    if [[ -n "$ROUTING_TOPK" ]]; then
        mode_suffix="${mode_suffix}_topk${ROUTING_TOPK}"
    fi
    log_file="$LOG_DIR/T0603_opencompass_qwen3_fp16_${label}_${mode_suffix}_${RUN_STAMP}.log"
    work_dir="$OC_DIR/qwen3_fp16_${label}_${mode_suffix}_T0603_${RUN_STAMP}"
    record_path="$RECORD_DIR/T0603_qwen3_fp16_${label}_${mode_suffix}_${RUN_STAMP}.jsonl"
    tag="T0603_qwenfix_${label}_${mode_suffix}"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists: label=$label mode=$mode_suffix" >&2
        exit 1
    }
    log "[START] opencompass label=$label datasets=${datasets[*]} router=$router_ckpt log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="weighted_sum" \
    T0531_ROUTING_SHARPNESS="$SHARPNESS" \
    T0531_ROUTER_RECORD_TAG="$tag" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass label=$label log=$log_file"
}

eval_mrs_only() {
    eval_router "mrs_only_taskcls_trainbert_qwenfix" "$(router_dir_for)" \
        SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
}

eval_one() {
    local task="$1"
    validate_task "$task"
    eval_router "mrs_plus_${task}_taskcls_trainbert_qwenfix" "$(router_dir_for "$task")" \
        "$(task_dataset_for "$task")" "${MRS_DATASETS[@]}"
}

main() {
    local mode="${1:-all}"
    local task_csv="${2:-$DEFAULT_TASKS}"
    log "[INFO] exp_root=$EXP_ROOT eval_root=$EVAL_ROOT weight_tag=$WEIGHT_TAG sharpness=$SHARPNESS routing_topk=${ROUTING_TOPK:-none}"
    case "$mode" in
        all)
            eval_mrs_only
            split_csv "$task_csv"
            for task in "${SPLIT_CSV_RESULT[@]}"; do eval_one "$task"; done
            ;;
        mrs_only)
            eval_mrs_only
            ;;
        tasks)
            split_csv "$task_csv"
            for task in "${SPLIT_CSV_RESULT[@]}"; do eval_one "$task"; done
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
