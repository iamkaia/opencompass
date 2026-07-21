#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXP_ROOT="${EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${CACHE_ROOT:-$EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
ROUTER_DIR="${ROUTER_DIR:-$EXP_ROOT/routers}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
BERT="${BERT:-./task_classifier_ckpt}"
TASK_CSV="${TASK_CSV:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
BEST_METRIC="${BEST_METRIC:-cache_oracle_matrix_kl}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
TRAIN_BERT="${TRAIN_BERT:-1}"
SAVE_ROUTE_RECORDS="${SAVE_ROUTE_RECORDS:-1}"
EVAL_TRAIN_EACH_EPOCH="${EVAL_TRAIN_EACH_EPOCH:-1}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0610_train_matrixkl_mrs_plus_qwen.sh [task_csv]

Trains T0610 Qwen matrixKL routers for mrs_plus_<task>, using samples from
medmcqa,race,sst2,<task> and experts medmcqa,race,sst2.

Default task_csv:
  boolq,rte,siqa,piqa,openbookqa,arc_c

Environment:
  EXP_ROOT              default ./T0603_qwenfix_trainbert_chattemplate_20260603_063927
  CACHE_ROOT            default $EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200
  TRAIN_BERT            default 1; set 0 to freeze BERT
  BATCH_SIZE            default 32
  EPOCHS                default 10
  LR                    default 2e-4
  BEST_METRIC           default cache_oracle_matrix_kl
  EARLY_STOP_PATIENCE   default 2
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

require_cache() {
    [[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || {
        echo "missing completed cache: $CACHE_ROOT" >&2
        exit 1
    }
}

router_dir_for() {
    local task="$1"
    echo "$ROUTER_DIR/router_T0610_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_cacheoraclematrixkl_t1_3expert_sst2words"
}

train_one() {
    local task="$1"
    local out_dir log_file
    validate_task "$task"
    out_dir="$(router_dir_for "$task")"
    log_file="$LOG_DIR/T0610_train_qwen3_fp16_mrs_plus_${task}_taskcls_trainbert_qwenfix_cacheoraclematrixkl_t1_${RUN_STAMP}.log"
    [[ ! -e "$out_dir" ]] || { echo "router output exists: $out_dir" >&2; exit 1; }
    [[ ! -e "$log_file" ]] || { echo "log output exists: $log_file" >&2; exit 1; }

    local -a extra_args=()
    if [[ "$TRAIN_BERT" == "1" ]]; then
        extra_args+=(--train_bert)
    fi
    if [[ "$SAVE_ROUTE_RECORDS" == "1" ]]; then
        extra_args+=(--save_route_records)
    fi
    if [[ "$EVAL_TRAIN_EACH_EPOCH" == "1" ]]; then
        extra_args+=(--eval_train_each_epoch)
    fi

    log "[START] task=$task out=$out_dir log=$log_file train_bert=$TRAIN_BERT"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        --out_dir "$out_dir" \
        --sample_task_names "${EXPERTS},${task}" \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --joint_loss cache_oracle_matrix_kl \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 1.0 \
        --pair_loss_normalization sample_minmax \
        --weighted_sum_marginal_mse_weight 0.0 \
        --best_metric "$BEST_METRIC" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        "${extra_args[@]}" \
        > "$log_file" 2>&1
    log "[DONE] task=$task out=$out_dir"
}

main() {
    local task_csv="${1:-$TASK_CSV}"
    case "$task_csv" in -h|--help|help) usage; exit 0 ;; esac
    require_cache
    mkdir -p "$ROUTER_DIR" "$LOG_DIR"
    log "[INFO] exp_root=$EXP_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT"
    log "[INFO] task_csv=$task_csv experts=$EXPERTS"
    split_csv "$task_csv"
    local task
    for task in "${SPLIT_CSV_RESULT[@]}"; do
        train_one "$task"
    done
    log "[DONE] all requested matrixKL mrs_plus routers"
}

main "$@"
