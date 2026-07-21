#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-python}"
BERT="${BERT:-./task_classifier_ckpt}"
DATASET_CONFIG="${DATASET_CONFIG:-router_train_split_gen}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
ROOT="${ROOT:-./T0610_router_train_split_eval_${RUN_STAMP}}"
LOG_DIR="$ROOT/logs"
OC_DIR="$ROOT/opencompass"
RECORD_DIR="$ROOT/router_records"
ROOT_LOG="${ROOT_LOG:-./T0610_router_train_split_eval_${RUN_STAMP}.root.log}"
ROUTER_CKPT="${ROUTER_CKPT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927/routers/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmse1p0_3expert_sst2words}"
SHARPNESS="${SHARPNESS:-1.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"

usage() {
  cat <<'EOF' >&2
Usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0610_eval_router_train_split_qwen.sh [all|uniform|hard|weighted_sum|cache_oracle_weighted_sum|cache_oracle_hard|oracle]

Env overrides:
  ROUTER_CKPT      router checkpoint dir; default is T0603 qwen mrs_only ce_plus_wsmse1
  ROOT             output root; default ./T0610_router_train_split_eval_<timestamp>
  DATASET_CONFIG   default router_train_split_gen
  ORACLE_WEIGHT_PATH required for cache_oracle_* modes
  SHARPNESS       runtime weighted_sum sharpness; default 1.0
  ROUTING_TOPK    optional weighted_sum top-k
EOF
}

log() {
  mkdir -p "$(dirname "$ROOT_LOG")"
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT_LOG"
}

run_one() {
  local routing_mode="$1"
  local label log_file work_dir record_path
  label="$(basename "$ROUTER_CKPT")"
  mkdir -p "$LOG_DIR" "$OC_DIR" "$RECORD_DIR"
  log_file="$LOG_DIR/qwen_train_split_${label}_${routing_mode}_${RUN_STAMP}.log"
  work_dir="$OC_DIR/qwen_train_split_${label}_${routing_mode}_${RUN_STAMP}"
  record_path="$RECORD_DIR/qwen_train_split_${label}_${routing_mode}_${RUN_STAMP}.jsonl"
  [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
    echo "eval output exists for routing_mode=$routing_mode" >&2
    exit 1
  }
  log "[START] routing_mode=$routing_mode router=$ROUTER_CKPT dataset=$DATASET_CONFIG log=$log_file"
  T0531_MRS_ROUTER_CKPT="$ROUTER_CKPT" \
  T0531_ROUTER_BERT_INIT="$BERT" \
  T0531_ROUTING_MODE="$routing_mode" \
  T0531_ROUTING_SHARPNESS="$SHARPNESS" \
  T0531_ROUTING_TOPK="$ROUTING_TOPK" \
  T0610_ORACLE_WEIGHT_PATH="${ORACLE_WEIGHT_PATH:-}" \
  T0531_ROUTER_RECORD_TAG="T0610_train_split_${label}_${routing_mode}" \
  T0601_ROUTER_RECORD_PATH="$record_path" \
    "$PY" -u run.py \
      --models "$MODEL_CONFIG" \
      --datasets "$DATASET_CONFIG" \
      --work-dir "$work_dir" \
      --debug \
      > "$log_file" 2>&1
  log "[DONE] routing_mode=$routing_mode log=$log_file"
}

case "$MODE" in
  all)
    run_one uniform
    run_one hard
    run_one weighted_sum
    ;;
  oracle)
    run_one cache_oracle_hard
    run_one cache_oracle_weighted_sum
    ;;
  uniform|hard|weighted_sum|cache_oracle_weighted_sum|cache_oracle_hard)
    run_one "$MODE"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
