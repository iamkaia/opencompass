#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${1:-all}"
MODELS_CSV="${MODELS:-qwen35,llama3_8b,llama2_7b_chat}"

HALF_RUN_STAMP="${HALF_RUN_STAMP:-20260708_035620}"
HALF_FAMILIES="${HALF_FAMILIES:-two_layer,single_layer}"
HALF_OUT_PREFIX="${HALF_OUT_PREFIX:-T0708_half_split_32layer}"

LATE_RUN_STAMP="${LATE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LATE_MIDDLE_LAYER_IDX="${LATE_MIDDLE_LAYER_IDX:-21}"
LATE_SPLIT_TAG="${LATE_SPLIT_TAG:-late21}"
LATE_OUT_PREFIX="${LATE_OUT_PREFIX:-T0710_late_third_32layer}"
LATE_FAMILIES="${LATE_FAMILIES:-two_layer}"

export SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
export EVAL_BASELINES="${EVAL_BASELINES:-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0710_run_half_then_late_third_32layer.sh all \
    > T0710_half_then_late_third_32layer.nohup.log 2>&1 &

What it does:
  1. Resume the existing half split run with MIDDLE_LAYER_IDX=16.
  2. Start a fresh late-third two-layer run with MIDDLE_LAYER_IDX=21.

Defaults:
  MODELS=qwen35,llama3_8b,llama2_7b_chat
  HALF_RUN_STAMP=20260708_035620
  HALF_FAMILIES=two_layer,single_layer
  LATE_RUN_STAMP=<current timestamp>
  LATE_MIDDLE_LAYER_IDX=21
  LATE_FAMILIES=two_layer
  EVAL_BASELINES=1

The late-third split is for 32-layer models:
  first segment: layers 0..20
  second segment: layers 21..31
EOF
}

main() {
    case "$MODE" in
        -h|--help|help) usage; exit 0 ;;
        all|cache|train|eval|baseline) ;;
        *) usage; exit 2 ;;
    esac

    log "[INFO] step=half_resume models=$MODELS_CSV families=$HALF_FAMILIES run_stamp=$HALF_RUN_STAMP middle_layer_idx=16"
    RESUME=1 \
    RUN_STAMP="$HALF_RUN_STAMP" \
    MODELS="$MODELS_CSV" \
    FAMILIES="$HALF_FAMILIES" \
    MIDDLE_LAYER_IDX=16 \
    SPLIT_TAG=half16 \
    OUT_PREFIX="$HALF_OUT_PREFIX" \
        bash tools/T0708_run_half_split_32layer_models.sh "$MODE"
    log "[DONE] step=half_resume run_stamp=$HALF_RUN_STAMP"

    log "[INFO] step=late_third models=$MODELS_CSV families=$LATE_FAMILIES run_stamp=$LATE_RUN_STAMP middle_layer_idx=$LATE_MIDDLE_LAYER_IDX split_tag=$LATE_SPLIT_TAG"
    RUN_STAMP="$LATE_RUN_STAMP" \
    MODELS="$MODELS_CSV" \
    FAMILIES="$LATE_FAMILIES" \
    MIDDLE_LAYER_IDX="$LATE_MIDDLE_LAYER_IDX" \
    SPLIT_TAG="$LATE_SPLIT_TAG" \
    OUT_PREFIX="$LATE_OUT_PREFIX" \
        bash tools/T0708_run_half_split_32layer_models.sh "$MODE"
    log "[DONE] step=late_third run_stamp=$LATE_RUN_STAMP"
}

main "$@"

