#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
OUT_ROOT="${OUT_ROOT:-./T0610_matrixkl_official_eval_sharp${SHARPNESS_TAG:-5}_${RUN_STAMP}}"
SHARPNESS="${SHARPNESS:-5.0}"
SHARPNESS_TAG="${SHARPNESS_TAG:-$(printf '%s' "$SHARPNESS" | sed 's/-/m/g; s/\./p/g')}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
ROUTING_MODE="${ROUTING_MODE:-weighted_sum}"

LOG_DIR="$OUT_ROOT/logs"
OC_DIR="$OUT_ROOT/opencompass"
RECORD_DIR="$OUT_ROOT/router_records"
ROUTER_DIR="$EXP_ROOT/routers"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0610_eval_matrixkl_weighted_sharp_qwen.sh [all|mrs_only|tasks] [task_csv]

Runs official OpenCompass weighted_sum eval for T0610 matrixKL routers with
runtime routing sharpness.

Modes:
  mrs_only  Evaluate the MRS-only router on BoolQ, MedMCQA, OBQA, ARC-C, PIQA,
            RACE, RTE, SIQA, and SST2.
  tasks     Evaluate mrs_plus_<task> routers on <task> + MRS datasets.
  all       Run mrs_only, then tasks for task_csv.

Environment:
  EXP_ROOT       default ./T0603_qwenfix_trainbert_chattemplate_20260603_063927
  OUT_ROOT       default ./T0610_matrixkl_official_eval_sharp<SHARPNESS>_<timestamp>
  ROUTING_MODE   default weighted_sum; set uniform to rerun uniform baselines
  SHARPNESS      default 5.0
  ROUTING_TOPK   optional, e.g. 3
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
        echo "$ROUTER_DIR/router_T0610_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_cacheoraclematrixkl_t1_3expert_sst2words"
    else
        echo "$ROUTER_DIR/router_T0610_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_cacheoraclematrixkl_t1_3expert_sst2words"
    fi
}

require_router() {
    local router_root="$1"
    [[ -f "$router_root/router_heads.pt" && -f "$router_root/router_config.json" ]] || {
        echo "missing router checkpoint: $router_root" >&2
        echo "Train this matrixKL router first, or run mrs_only if only the MRS-only router exists." >&2
        exit 1
    }
}

eval_router() {
    local label="$1" router_ckpt="$2"
    shift 2
    local mode_suffix log_file work_dir record_path tag
    local -a datasets=("$@")
    require_router "$router_ckpt"
    mkdir -p "$LOG_DIR" "$OC_DIR" "$RECORD_DIR"
    if [[ "$ROUTING_MODE" == "weighted_sum" ]]; then
        mode_suffix="weighted_sum_matrixkl_sharp${SHARPNESS_TAG}"
    else
        mode_suffix="${ROUTING_MODE}_matrixklrouter"
    fi
    if [[ "$ROUTING_MODE" == "weighted_sum" && -n "$ROUTING_TOPK" ]]; then
        mode_suffix="${mode_suffix}_topk${ROUTING_TOPK}"
    fi
    log_file="$LOG_DIR/T0610_opencompass_qwen3_fp16_${label}_${mode_suffix}_${RUN_STAMP}.log"
    work_dir="$OC_DIR/qwen3_fp16_${label}_${mode_suffix}_${RUN_STAMP}"
    record_path="$RECORD_DIR/T0610_qwen3_fp16_${label}_${mode_suffix}_${RUN_STAMP}.jsonl"
    tag="T0610_matrixkl_${label}_${mode_suffix}"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists: label=$label mode=$mode_suffix" >&2
        exit 1
    }
    log "[START] label=$label mode=$ROUTING_MODE sharpness=$SHARPNESS datasets=${datasets[*]} router=$router_ckpt log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="$ROUTING_MODE" \
    T0531_ROUTING_SHARPNESS="$SHARPNESS" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTER_RECORD_TAG="$tag" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] label=$label log=$log_file"
}

eval_mrs_only() {
    eval_router "mrs_only_taskcls_trainbert_qwenfix" "$(router_dir_for)" \
        SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
}

eval_one_task() {
    local task="$1"
    validate_task "$task"
    eval_router "mrs_plus_${task}_taskcls_trainbert_qwenfix" "$(router_dir_for "$task")" \
        "$(task_dataset_for "$task")" "${MRS_DATASETS[@]}"
}

main() {
    local mode="${1:-all}"
    local task_csv="${2:-$DEFAULT_TASKS}"
    log "[INFO] exp_root=$EXP_ROOT out_root=$OUT_ROOT routing_mode=$ROUTING_MODE sharpness=$SHARPNESS routing_topk=${ROUTING_TOPK:-none}"
    case "$mode" in
        all)
            eval_mrs_only
            split_csv "$task_csv"
            for task in "${SPLIT_CSV_RESULT[@]}"; do eval_one_task "$task"; done
            ;;
        mrs_only)
            eval_mrs_only
            ;;
        tasks)
            split_csv "$task_csv"
            for task in "${SPLIT_CSV_RESULT[@]}"; do eval_one_task "$task"; done
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
