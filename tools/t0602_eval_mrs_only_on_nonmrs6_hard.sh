#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-}"
ROOT_LOG="${ROOT_LOG:-}"

NON_MRS_DATASETS=(
    SuperGLUE_BoolQ_gen
    SuperGLUE_RTE_gen
    siqa_gen
    piqa_gen
    obqa_main_gen
    ARC_c_gen
)

usage() {
    cat >&2 <<'EOF'
usage:
  EXP_ROOT=./t0602_bert_ablation_hard_<timestamp> CUDA_VISIBLE_DEVICES=0 bash tools/t0602_eval_mrs_only_on_nonmrs6_hard.sh

This only runs hard-routing OpenCompass eval for existing MRS-only routers on
the six non-MRS datasets:
  boolq, rte, siqa, piqa, openbookqa, arc_c

It does not train routers and does not overwrite existing eval outputs.
EOF
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S %Z"
}

log() {
    echo "[$(timestamp)] $*"
}

setup_root_log() {
    if [[ -z "$ROOT_LOG" ]]; then
        ROOT_LOG="$EXP_ROOT/t0602_eval_mrs_only_nonmrs6_hard_${RUN_STAMP}.log"
    fi
    if [[ -e "$ROOT_LOG" ]]; then
        echo "root log already exists: $ROOT_LOG" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$ROOT_LOG")"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "[INFO] root_log=$ROOT_LOG"
}

require_exp_root() {
    if [[ -z "$EXP_ROOT" ]]; then
        usage
        exit 2
    fi
    if [[ ! -d "$EXP_ROOT/routers" ]]; then
        echo "missing routers dir under EXP_ROOT: $EXP_ROOT" >&2
        exit 1
    fi
}

model_config_for() {
    case "$1" in
        llama) echo "T0531_mrs_ablation_hard_routing.py" ;;
        qwen) echo "T0531_mrs_ablation_sst2words_hard_routing.py" ;;
        *) echo "unsupported family: $1" >&2; exit 2 ;;
    esac
}

bert_init_for_router() {
    local router_name="$1"
    case "$router_name" in
        *"_tiny_"*) echo "prajjwal1/bert-tiny" ;;
        *) echo "./task_classifier_ckpt" ;;
    esac
}

family_for_router() {
    local router_name="$1"
    case "$router_name" in
        router_T0602_llama_*) echo "llama" ;;
        router_T0602_qwen3_fp16_*) echo "qwen" ;;
        *) echo "unsupported router name: $router_name" >&2; exit 2 ;;
    esac
}

require_router() {
    local router_root="$1"
    if [[ ! -f "$router_root/router_heads.pt" || ! -f "$router_root/router_config.json" ]]; then
        echo "missing router checkpoint: $router_root" >&2
        exit 1
    fi
}

eval_router() {
    local router_ckpt="$1"
    local router_name family model_config bert_init safe_name log_file work_dir record_path

    require_router "$router_ckpt"
    router_name="$(basename "$router_ckpt")"
    family="$(family_for_router "$router_name")"
    model_config="$(model_config_for "$family")"
    bert_init="$(bert_init_for_router "$router_name")"
    safe_name="${router_name}_nonmrs6_hard"
    log_file="$EXP_ROOT/logs/opencompass_${safe_name}_${RUN_STAMP}.log"
    work_dir="$EXP_ROOT/opencompass/${safe_name}"
    record_path="$EXP_ROOT/router_records/${safe_name}.jsonl"

    mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/opencompass" "$EXP_ROOT/router_records"

    if [[ -e "$work_dir" || -e "$record_path" || -e "$log_file" ]]; then
        echo "eval output already exists; refusing to overwrite: $work_dir $record_path $log_file" >&2
        exit 1
    fi

    log "[START] opencompass router=$router_name family=$family bert=$bert_init routing=hard datasets=${NON_MRS_DATASETS[*]} work_dir=$work_dir log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$bert_init" \
    T0531_ROUTING_MODE="hard" \
    T0531_ROUTER_RECORD_TAG="t0602_mrs_only_nonmrs6" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${NON_MRS_DATASETS[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] opencompass router=$router_name log=$log_file record=$record_path"
}

main() {
    local router
    require_exp_root
    setup_root_log
    log "[INFO] exp_root=$EXP_ROOT"
    log "[INFO] cuda=$CUDA_VISIBLE_DEVICES"

    shopt -s nullglob
    local routers=("$EXP_ROOT"/routers/router_T0602_*_mrs_only_*)
    if [[ "${#routers[@]}" -eq 0 ]]; then
        echo "no MRS-only routers found under: $EXP_ROOT/routers" >&2
        exit 1
    fi

    for router in "${routers[@]}"; do
        eval_router "$router"
    done

    log "[DONE] finished MRS-only non-MRS-6 hard eval under $EXP_ROOT"
    log "[END] finished_at=$(timestamp)"
}

main "$@"
