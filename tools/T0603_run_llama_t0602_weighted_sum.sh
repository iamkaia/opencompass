#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
QWEN_ROOT="${QWEN_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
LLAMA_ROOT="${LLAMA_ROOT:-./t0602_taskcls_trainbert_chattemplate_20260603_02}"
ROOT_LOG="${ROOT_LOG:-./T0603_qwen_llama_weighted_sum_topk3_${RUN_STAMP}.root.log}"
ROUTING_TOPK="${ROUTING_TOPK:-3}"
ROUTING_SHARPNESS="${ROUTING_SHARPNESS:-1.0}"

BERT="./task_classifier_ckpt"
DEFAULT_TASKS="boolq,rte,siqa,piqa,openbookqa,arc_c"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0603_run_llama_t0602_weighted_sum.sh [all|qwen|llama|qwen_mrs|qwen_replay|llama_mrs|llama_replay] [task_csv]

Runs OpenCompass weighted_sum with routing_topk=3 for existing qwen and llama routers.
When stage=all, qwen runs first, then llama. Outputs are written back to each root.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

setup_root_log() {
    mkdir -p "$(dirname "$ROOT_LOG")"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
    log "[INFO] qwen_root=$QWEN_ROOT"
    log "[INFO] llama_root=$LLAMA_ROOT"
    log "[INFO] routing_mode=weighted_sum routing_topk=$ROUTING_TOPK routing_sharpness=$ROUTING_SHARPNESS"
}

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

require_router() {
    local router_root="$1"
    [[ -f "$router_root/router_heads.pt" && -f "$router_root/router_config.json" ]] || {
        echo "missing router checkpoint: $router_root" >&2; exit 1;
    }
}

root_for() {
    case "$1" in
        qwen) echo "$QWEN_ROOT" ;;
        llama) echo "$LLAMA_ROOT" ;;
        *) echo "unsupported family: $1" >&2; exit 2 ;;
    esac
}

model_config_for() {
    case "$1" in
        qwen) echo "T0531_mrs_ablation_sst2words_hard_routing.py" ;;
        llama) echo "T0531_mrs_ablation_hard_routing.py" ;;
        *) echo "unsupported family: $1" >&2; exit 2 ;;
    esac
}

family_label_for() {
    case "$1" in
        qwen) echo "qwen3_fp16" ;;
        llama) echo "llama" ;;
        *) echo "unsupported family: $1" >&2; exit 2 ;;
    esac
}

router_dir_for() {
    local family="$1" task="${2:-}" root
    root="$(root_for "$family")"
    case "$family" in
        qwen)
            if [[ -z "$task" ]]; then
                echo "$root/routers/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_3expert_sst2words"
            else
                echo "$root/routers/router_T0603_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_correct_conf_ce_t1_3expert_sst2words"
            fi
            ;;
        llama)
            if [[ -z "$task" ]]; then
                echo "$root/routers/router_T0602_llama_mrs_only_taskcls_trainbert_chattemplate_correct_conf_ce_t1_3expert"
            else
                echo "$root/routers/router_T0602_llama_mrs_plus_${task}_taskcls_trainbert_chattemplate_correct_conf_ce_t1_3expert"
            fi
            ;;
        *) echo "unsupported family: $family" >&2; exit 2 ;;
    esac
}

eval_router() {
    local family="$1" label="$2" router_ckpt="$3"
    shift 3
    local root log_dir oc_dir record_dir model_config family_label log_file work_dir record_path
    local -a datasets=("$@")
    require_router "$router_ckpt"
    root="$(root_for "$family")"
    [[ -d "$root" ]] || { echo "missing root for $family: $root" >&2; exit 1; }
    log_dir="$root/logs"
    oc_dir="$root/opencompass"
    record_dir="$root/router_records"
    model_config="$(model_config_for "$family")"
    family_label="$(family_label_for "$family")"
    mkdir -p "$log_dir" "$oc_dir" "$record_dir"
    log_file="$log_dir/T0603_opencompass_${family_label}_${label}_weighted_sum_topk${ROUTING_TOPK}_${RUN_STAMP}.log"
    work_dir="$oc_dir/${family_label}_${label}_weighted_sum_topk${ROUTING_TOPK}_T0603_${RUN_STAMP}"
    record_path="$record_dir/T0603_${family_label}_${label}_weighted_sum_topk${ROUTING_TOPK}_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists for family=$family label=$label" >&2; exit 1;
    }
    log "[START] opencompass family=$family label=$label mode=weighted_sum topk=$ROUTING_TOPK datasets=${datasets[*]} log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE="weighted_sum" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" \
    T0531_ROUTING_SHARPNESS="$ROUTING_SHARPNESS" \
    T0531_ROUTER_RECORD_TAG="T0603_${family_label}_${label}_weighted_sum_topk${ROUTING_TOPK}" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass family=$family label=$label mode=weighted_sum topk=$ROUTING_TOPK log=$log_file"
}

eval_mrs() {
    local family="$1" label
    case "$family" in
        qwen) label="mrs_only_taskcls_trainbert_qwenfix" ;;
        llama) label="mrs_only_taskcls_trainbert_chattemplate" ;;
        *) echo "unsupported family: $family" >&2; exit 2 ;;
    esac
    eval_router "$family" "$label" "$(router_dir_for "$family")" \
        SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
}

eval_one() {
    local family="$1" task="$2" task_dataset label
    task_dataset="$(task_dataset_for "$task")"
    case "$family" in
        qwen) label="mrs_plus_${task}_taskcls_trainbert_qwenfix" ;;
        llama) label="mrs_plus_${task}_taskcls_trainbert_chattemplate" ;;
        *) echo "unsupported family: $family" >&2; exit 2 ;;
    esac
    eval_router "$family" "$label" "$(router_dir_for "$family" "$task")" "$task_dataset" "${MRS_DATASETS[@]}"
}

run_family() {
    local family="$1" part="$2"
    shift 2
    local -a tasks=("$@")
    local task
    case "$part" in
        all|mrs) eval_mrs "$family" ;;
    esac
    case "$part" in
        all|replay)
            for task in "${tasks[@]}"; do eval_one "$family" "$task"; done
            ;;
    esac
}

main() {
    local stage="${1:-all}" task_csv="${2:-$DEFAULT_TASKS}"
    local -a tasks
    case "$stage" in all|qwen|llama|qwen_mrs|qwen_replay|llama_mrs|llama_replay) ;; *) usage; exit 2 ;; esac
    split_csv "$task_csv"; tasks=("${SPLIT_CSV_RESULT[@]}")
    local task
    for task in "${tasks[@]}"; do validate_task "$task"; done
    setup_root_log
    log "[INFO] stage=$stage tasks=${tasks[*]}"
    case "$stage" in
        all)
            run_family qwen all "${tasks[@]}"
            run_family llama all "${tasks[@]}"
            ;;
        qwen) run_family qwen all "${tasks[@]}" ;;
        llama) run_family llama all "${tasks[@]}" ;;
        qwen_mrs) run_family qwen mrs "${tasks[@]}" ;;
        qwen_replay) run_family qwen replay "${tasks[@]}" ;;
        llama_mrs) run_family llama mrs "${tasks[@]}" ;;
        llama_replay) run_family llama replay "${tasks[@]}" ;;
    esac
    log "[DONE] stage=$stage qwen_root=$QWEN_ROOT llama_root=$LLAMA_ROOT"
}

main "$@"
