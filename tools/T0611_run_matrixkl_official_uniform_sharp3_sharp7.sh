#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${1:-all}"
TASK_CSV="${2:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT_PREFIX="${BASE_OUT_PREFIX:-T0611_matrixkl_official_eval}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0611_run_matrixkl_official_uniform_sharp3_sharp7.sh [all|mrs_only|tasks] [task_csv]

Runs three official OpenCompass eval batches for the T0610 matrixKL routers:
  1. uniform
  2. weighted_sum with SHARPNESS=3.0
  3. weighted_sum with SHARPNESS=7.0

Outputs are written to new T0611_* folders. The per-batch nohup/stdout log
should also be redirected by the caller to a T0611_* log file.

Environment:
  BASE_OUT_PREFIX default T0611_matrixkl_official_eval
  RUN_STAMP       default current timestamp
EOF
}

case "$MODE" in
    all|mrs_only|tasks) ;;
    -h|--help|help) usage; exit 0 ;;
    *) usage; exit 2 ;;
esac

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(timestamp)] $*"; }

run_eval() {
    local label="$1" routing_mode="$2" sharpness="$3" out_root="$4"
    log "[START] $label mode=$routing_mode sharpness=$sharpness out_root=$out_root"
    ROUTING_MODE="$routing_mode" \
    SHARPNESS="$sharpness" \
    OUT_ROOT="$out_root" \
        bash tools/T0610_eval_matrixkl_weighted_sharp_qwen.sh "$MODE" "$TASK_CSV"
    log "[DONE] $label out_root=$out_root"
}

log "[INFO] mode=$MODE task_csv=$TASK_CSV run_stamp=$RUN_STAMP"

run_eval "uniform" "uniform" "1.0" "./${BASE_OUT_PREFIX}_uniform_${RUN_STAMP}"
run_eval "sharp3" "weighted_sum" "3.0" "./${BASE_OUT_PREFIX}_sharp3_${RUN_STAMP}"
run_eval "sharp7" "weighted_sum" "7.0" "./${BASE_OUT_PREFIX}_sharp7_${RUN_STAMP}"

log "[DONE] all T0611 uniform/sharp3/sharp7 evals"
